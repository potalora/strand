from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.local_ai import pipeline as pipeline_module
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
)
from app.services.local_ai.manifest import parse_manifest
from app.services.local_ai.pipeline import MemoryCheckpointStore, StrictLocalPipeline
from app.services.local_ai.types import ModelRole
from tests.test_strict_local_pipeline import (
    _Manager,
    _job_and_upload,
    _rasterizer,
    _trusted_source_digest,
)


def _empty_extraction() -> dict[str, object]:
    return {
        "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
        "patient": None,
        **{category: [] for category in FACT_CATEGORY_NAMES},
        "unresolved_fields": [],
        "rejected_fields": [],
    }


def _lab(name: str, value: str, *, page_number: int = 1) -> dict[str, object]:
    evidence = f"{name} {value} mg/dL"
    return {
        "name": name,
        "value": value,
        "unit": "mg/dL",
        "verbatim": evidence,
        "page_number": page_number,
        "evidence_excerpt": evidence,
    }


def test_fhir_mapper_exception_quarantines_only_rejected_fact(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Canary MRN 8675309 takes warfarin"
    extraction = _empty_extraction()
    extraction["labs"] = [
        _lab("Glucose", "95"),
        _lab("Creatinine", "1.0"),
    ]
    mapper_calls: list[str] = []

    def fail_one_fact(entity, *args, **kwargs):
        mapper_calls.append(entity.text)
        if entity.text == "Creatinine":
            raise RuntimeError(canary)
        return entity_to_health_record_dict(entity, *args, **kwargs)

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        fail_one_fact,
    )
    caplog.set_level(logging.DEBUG)

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL. Creatinine 1.0 mg/dL."},
        upload_id="fhir-mapper-quarantine",
    )

    assert mapper_calls == ["Glucose", "Creatinine"]
    assert [fact.name for fact in result.labs] == ["Glucose"]
    assert result.rejected_fields == ["chunks[0].labs[1]:fhir_mapping_rejected"]
    assert canary not in json.dumps(result.model_dump(mode="json"))
    assert canary not in caplog.text


def test_empty_fhir_mapper_output_quarantines_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [_lab("Glucose", "95")]
    monkeypatch.setattr(
        pipeline_module,
        "validated_extraction_to_health_record_dicts",
        lambda *_args, **_kwargs: [],
    )

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL."},
        upload_id="empty-fhir-mapper-output",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fhir_mapping_rejected"]


def test_final_server_owned_extraction_invariant_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [_lab("Glucose", "95")]
    real_validate = pipeline_module.validate_clinical_extraction

    def fail_final_invariant(raw, *args, **kwargs):
        if isinstance(raw, dict) and all(
            category in raw for category in FACT_CATEGORY_NAMES
        ):
            raise LocalValidationError("Local extraction merge invariant failed.")
        return real_validate(raw, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "validate_clinical_extraction",
        fail_final_invariant,
    )

    with pytest.raises(
        LocalValidationError,
        match="Local extraction merge invariant failed",
    ):
        StrictLocalPipeline._validate_extraction_result(
            extraction,
            {1: "Glucose 95 mg/dL."},
            upload_id="fatal-final-invariant",
        )


def test_raw_fact_validation_budget_is_shared_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "_MAX_RAW_ITEMS_PER_DOCUMENT",
        2,
        raising=False,
    )
    chunks = []
    pages = {}
    for page_number, (name, value) in enumerate(
        (
            ("Glucose", "95"),
            ("Creatinine", "1.0"),
            ("Sodium", "140"),
        ),
        start=1,
    ):
        extraction = _empty_extraction()
        extraction["labs"] = [_lab(name, value, page_number=page_number)]
        chunks.append(
            {
                "page_numbers": [page_number],
                "extraction": extraction,
            }
        )
        pages[page_number] = f"{name} {value} mg/dL."

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": chunks,
        },
        pages,
        upload_id="document-global-work-budget",
    )

    assert [fact.name for fact in result.labs] == ["Glucose", "Creatinine"]
    assert result.rejected_fields == ["extraction.validation_work:limit_exceeded"]


def test_exhausted_work_budget_does_not_iterate_large_fact_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneSerializationOnly(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("validation iterated the exhausted fact tail")
            return super().__iter__()

    monkeypatch.setattr(
        pipeline_module,
        "_MAX_RAW_ITEMS_PER_DOCUMENT",
        1,
        raising=False,
    )
    extraction = _empty_extraction()
    facts = OneSerializationOnly([_lab("Glucose", "95")] * 5_000)
    extraction["labs"] = facts

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL."},
        upload_id="large-exhausted-tail",
    )

    assert facts.iterations == 1
    assert [fact.name for fact in result.labs] == ["Glucose"]
    assert result.rejected_fields == ["extraction.validation_work:limit_exceeded"]


@pytest.mark.asyncio
async def test_cached_extraction_reapplies_server_owned_fact_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "Patient Cache Canary MRN 121212"

    class TwoLabManager(_Manager):
        async def run_attested(self, manifest, role, payload, on_progress=None):
            self._require_attested_manifest(manifest)
            del on_progress
            self.calls.append((role, deepcopy(payload)))
            if role is ModelRole.OCR:
                return {
                    "markdown": "Glucose 95 mg/dL. Creatinine 1.0 mg/dL.",
                    "page_number": payload["page_number"],
                }
            if role is ModelRole.EXTRACTION:
                extraction = _empty_extraction()
                extraction["labs"] = [
                    _lab("Glucose", "95"),
                    _lab("Creatinine", "1.0"),
                ]
                return extraction
            raise AssertionError("summary is not part of ingestion")

    job, upload = _job_and_upload()
    manager = TwoLabManager()
    checkpoints = MemoryCheckpointStore()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=checkpoints,
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
    )

    first = await pipeline.run_ingestion(
        job,
        upload,
        tmp_path / "encrypted.pdf",
    )
    assert [fact.name for fact in first.validated_extraction.labs] == [
        "Glucose",
        "Creatinine",
    ]

    cache_key = (str(job.id), str(upload.id))
    stored = checkpoints._extractions[cache_key]
    cached_result = deepcopy(stored.extraction_result)
    cached_result["rejected_fields"] = [canary]
    checkpoints._extractions[cache_key] = replace(
        stored,
        extraction_result=cached_result,
    )
    mapper_calls: list[str] = []

    def changed_mapper(entity, *args, **kwargs):
        mapper_calls.append(entity.text)
        if entity.text == "Creatinine":
            raise RuntimeError(canary)
        return entity_to_health_record_dict(entity, *args, **kwargs)

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        changed_mapper,
    )
    manager.calls.clear()

    second = await pipeline.run_ingestion(
        job,
        upload,
        tmp_path / "encrypted.pdf",
    )

    assert manager.calls == []
    assert mapper_calls == ["Glucose", "Creatinine"]
    assert [fact.name for fact in second.validated_extraction.labs] == ["Glucose"]
    assert second.rejected_fields == ["chunks[0].labs[1]:fhir_mapping_rejected"]
    assert canary not in json.dumps(second.validated_extraction.model_dump(mode="json"))
