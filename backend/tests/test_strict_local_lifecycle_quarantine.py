from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from app.services.local_ai.adapters import (
    to_extracted_entities,
    validated_extraction_to_health_record_dicts,
)
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

_ZERO_UUID = UUID(int=0)


def _empty_extraction() -> dict[str, object]:
    return {
        "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
        "patient": None,
        **{category: [] for category in FACT_CATEGORY_NAMES},
        "unresolved_fields": [],
        "rejected_fields": [],
    }


def _fact(name_field: str, name: str, evidence: str, **fields) -> dict[str, object]:
    return {
        name_field: name,
        "verbatim": evidence,
        "page_number": 1,
        "evidence_excerpt": evidence,
        **fields,
    }


_NON_PROMOTABLE_LIFECYCLE_CASES = (
    (
        "medications",
        _fact("name", "Metformin", "Metformin stopped", status="stopped"),
        "Metformin stopped",
    ),
    (
        "medications",
        _fact(
            "name",
            "Metformin",
            "Metformin historical",
            status="historical",
        ),
        "Metformin historical",
    ),
    (
        "medications",
        _fact("name", "Metformin", "Metformin listed", status="unknown"),
        "Metformin listed",
    ),
    (
        "allergies",
        _fact(
            "substance",
            "Penicillin",
            "Penicillin allergy inactive",
            assertion="present",
            status="inactive",
        ),
        "Penicillin allergy inactive",
    ),
    (
        "allergies",
        _fact(
            "substance",
            "Penicillin",
            "Penicillin allergy resolved",
            assertion="present",
            status="resolved",
        ),
        "Penicillin allergy resolved",
    ),
    (
        "care_plans",
        _fact(
            "title",
            "Care plan",
            "Care plan inactive",
            status="inactive",
        ),
        "Care plan inactive",
    ),
    (
        "care_plans",
        _fact(
            "title",
            "Care plan",
            "Care plan resolved",
            status="resolved",
        ),
        "Care plan resolved",
    ),
    (
        "encounters",
        _fact(
            "name",
            "Office appointment",
            "Office appointment in progress",
            visit_type="office",
            status="in_progress",
        ),
        "Office appointment in progress",
    ),
    (
        "encounters",
        _fact(
            "name",
            "Office appointment",
            "Office appointment planned",
            visit_type="office",
            status="planned",
        ),
        "Office appointment planned",
    ),
    (
        "encounters",
        _fact(
            "name",
            "Office appointment",
            "Office appointment listed",
            visit_type="office",
            status="unknown",
        ),
        "Office appointment listed",
    ),
    (
        "immunizations",
        _fact(
            "name",
            "Influenza vaccine",
            "Influenza vaccine entered in error",
            status="entered_in_error",
        ),
        "Influenza vaccine entered in error",
    ),
    (
        "immunizations",
        _fact(
            "name",
            "Influenza vaccine",
            "Influenza vaccine listed",
            status="unknown",
        ),
        "Influenza vaccine listed",
    ),
    (
        "diagnostic_reports",
        _fact(
            "name",
            "CT chest",
            "CT chest preliminary report showed stable nodule",
            findings="stable nodule",
            assertion="present",
            status="preliminary",
        ),
        "CT chest preliminary report showed stable nodule",
    ),
    (
        "diagnostic_reports",
        _fact(
            "name",
            "CT chest",
            "CT chest amended report showed stable nodule",
            findings="stable nodule",
            assertion="present",
            status="amended",
        ),
        "CT chest amended report showed stable nodule",
    ),
    (
        "diagnostic_reports",
        _fact(
            "name",
            "CT chest",
            "CT chest report listed stable nodule",
            findings="stable nodule",
            assertion="present",
            status="unknown",
        ),
        "CT chest report listed stable nodule",
    ),
)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    _NON_PROMOTABLE_LIFECYCLE_CASES,
)
def test_non_promotable_lifecycle_fact_is_quarantined_before_evidence_persistence(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    extraction = _empty_extraction()
    extraction[category] = [deepcopy(fact)]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: page},
        upload_id=f"lifecycle-{category}",
    )

    assert getattr(result, category) == []
    assert result.evidence == ()
    assert result.rejected_fields == [f"chunks[0].{category}[0]:adapter_rejected"]


def test_accepted_fact_evidence_entity_and_record_counts_align() -> None:
    extraction = _empty_extraction()
    extraction["medications"] = [
        _fact(
            "name",
            "Metformin",
            "Metformin active",
            status="active",
        ),
        _fact(
            "name",
            "Warfarin",
            "Warfarin stopped",
            status="stopped",
        ),
    ]
    extraction["labs"] = [
        _fact(
            "name",
            "Glucose",
            "Glucose 95 mg/dL",
            value="95",
            unit="mg/dL",
        )
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Metformin active. Warfarin stopped. Glucose 95 mg/dL."},
        upload_id="aligned-accepted-facts",
    )
    facts = [
        fact for category in FACT_CATEGORY_NAMES for fact in getattr(result, category)
    ]
    entities = to_extracted_entities(result)
    records = validated_extraction_to_health_record_dicts(
        result,
        _ZERO_UUID,
        _ZERO_UUID,
        _ZERO_UUID,
    )

    assert [fact.name for fact in result.medications] == ["Metformin"]
    assert len(facts) == len(result.evidence) == len(entities) == len(records) == 2
    assert result.rejected_fields == ["chunks[0].medications[1]:adapter_rejected"]


@pytest.mark.asyncio
async def test_cached_extraction_reapplies_lifecycle_quarantine(
    tmp_path: Path,
) -> None:
    page = "Metformin active. Warfarin stopped."
    active = _fact(
        "name",
        "Metformin",
        "Metformin active",
        status="active",
    )
    stopped = _fact(
        "name",
        "Warfarin",
        "Warfarin stopped",
        status="stopped",
    )

    class LifecycleManager(_Manager):
        async def run_attested(self, manifest, role, payload, on_progress=None):
            self._require_attested_manifest(manifest)
            del on_progress
            self.calls.append((role, deepcopy(payload)))
            if role is ModelRole.OCR:
                return {
                    "markdown": page,
                    "page_number": payload["page_number"],
                }
            if role is ModelRole.EXTRACTION:
                extraction = _empty_extraction()
                extraction["medications"] = [deepcopy(active)]
                return extraction
            raise AssertionError("summary is not part of ingestion")

    job, upload = _job_and_upload()
    manager = LifecycleManager()
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
    await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    cache_key = (str(job.id), str(upload.id))
    stored = checkpoints._extractions[cache_key]
    cached_result = _empty_extraction()
    cached_result["medications"] = [deepcopy(active), deepcopy(stopped)]
    checkpoints._extractions[cache_key] = replace(
        stored,
        extraction_result=cached_result,
    )
    manager.calls.clear()

    result = await pipeline.run_ingestion(
        job,
        upload,
        tmp_path / "encrypted.pdf",
    )

    assert manager.calls == []
    assert [fact.name for fact in result.validated_extraction.medications] == [
        "Metformin"
    ]
    assert len(result.evidence) == len(result.entities) == 1
    assert result.rejected_fields == ["chunks[0].medications[1]:adapter_rejected"]
