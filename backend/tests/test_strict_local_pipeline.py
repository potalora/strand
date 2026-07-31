from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.models.local_ai import LocalAIJob, LocalAIPage
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.services.local_ai import pipeline as pipeline_module
from app.services.extraction.intra_doc_dedup import dedup_within_document
from app.services.local_ai.adapters import (
    to_extracted_entities,
    validated_extraction_to_health_record_dicts,
)
from app.services.local_ai.checkpoint_store import CheckpointStore, OCRCheckpoint
from app.services.local_ai.errors import (
    LocalPolicyError,
    LocalValidationError,
    LocalWorkerError,
)
from app.services.local_ai.extraction_validator import validate_clinical_extraction
from app.services.local_ai.manifest import (
    canonicalize_manifest_snapshot,
    parse_manifest,
)
from app.services.local_ai.pipeline import (
    MemoryCheckpointStore,
    StrictLocalPipeline,
)
from app.services.local_ai.rasterizer import RasterizedPage
from app.services.local_ai.types import ModelRole
from app.utils.file_utils import encrypt_stream


def _artifact(role: str) -> dict[str, object]:
    return {
        "role": role,
        "repository": f"owner/{role}",
        "revision": {"ocr": "a", "extraction": "b", "summary": "c"}[role] * 40,
        "quantization": "4bit",
        "license": "apache-2.0",
        "attribution": f"https://huggingface.co/owner/{role}",
        "decode_limits": {"max_input_tokens": 4096, "max_output_tokens": 1024},
        "files": [
            {
                "path": f"{role}/model.safetensors",
                "sha256": {"ocr": "a", "extraction": "b", "summary": "c"}[role] * 64,
                "size": 10,
            }
        ],
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "fixtures-v1",
        "artifacts": [_artifact(role) for role in ("ocr", "extraction", "summary")],
    }


def _empty_extraction() -> dict[str, object]:
    return {
        "schema_version": "clinical-document-extraction.v1",
        "patient": None,
        "medications": [],
        "conditions": [],
        "procedures": [],
        "labs": [],
        "allergies": [],
        "encounters": [],
        "immunizations": [],
        "vital_signs": [],
        "diagnostic_reports": [],
        "care_plans": [],
        "unresolved_fields": [],
        "rejected_fields": [],
    }


@dataclass
class _Job:
    id: uuid.UUID
    processing_mode: str
    manifest_snapshot: dict[str, object]
    manifest_sha256: str

    def revalidate_manifest_snapshot(self) -> dict[str, object]:
        snapshot, digest = canonicalize_manifest_snapshot(self.manifest_snapshot)
        if snapshot != self.manifest_snapshot or digest != self.manifest_sha256:
            raise LocalPolicyError("Stored manifest does not match.")
        return snapshot


@dataclass
class _Upload:
    id: uuid.UUID
    user_id: uuid.UUID
    file_hash: str
    processing_mode: str
    processing_manifest: dict[str, object]
    processing_schema_version: str


class _Manager:
    def __init__(self, *, failure_role: ModelRole | None = None) -> None:
        self.calls: list[tuple[ModelRole, dict[str, Any]]] = []
        self.failure_role = failure_role

    async def run(
        self,
        role: ModelRole,
        payload: dict[str, Any],
        on_progress=None,
    ) -> dict[str, object]:
        del on_progress
        self.calls.append((role, deepcopy(payload)))
        if role is self.failure_role:
            raise LocalWorkerError("Local worker failed.")
        if role is ModelRole.OCR:
            page_number = payload["page_number"]
            return {
                "markdown": f"Page {page_number}: no clinical facts.",
                "page_number": page_number,
            }
        if role is ModelRole.EXTRACTION:
            return _empty_extraction()
        raise AssertionError("summary is not part of ingestion")


def _job_and_upload(*, file_hash: str = "d" * 64) -> tuple[_Job, _Upload]:
    snapshot, digest = canonicalize_manifest_snapshot(_manifest_payload())
    job = _Job(
        id=uuid.uuid4(),
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
    )
    upload = _Upload(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        file_hash=file_hash,
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    return job, upload


def _trusted_source_digest(_encrypted_path: Path, _scratch: object) -> str:
    return "d" * 64


def _rasterizer(page_count: int):
    def rasterize(_encrypted_path, scratch, **_limits):
        for page_number in range(1, page_count + 1):
            filename = f"fake-{page_number}.png"
            path = scratch.create_file(filename, f"png-{page_number}".encode())
            yield RasterizedPage(
                page_number=page_number,
                path=path,
                width=10,
                height=10,
                sha256=f"{page_number:064x}",
            )
            scratch.remove_file(filename)

    return rasterize


@pytest.mark.asyncio
async def test_pipeline_canonicalizes_relative_local_roots_before_worker_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    job, upload = _job_and_upload()
    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=Path("scratch"),
        model_dir=Path("pack"),
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
    )

    await pipeline.run_ingestion(
        job=job,
        upload=upload,
        encrypted_path=Path("encrypted.pdf"),
    )

    for _role, payload in manager.calls:
        assert Path(payload["scratch_dir"]).is_absolute()
        assert Path(payload["manifest_path"]).is_absolute()
        assert Path(payload["model_dir"]).is_absolute()


@pytest.mark.asyncio
async def test_pipeline_runs_ocr_then_extraction_and_returns_validated_entities(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    manager = _Manager()
    checkpoints = MemoryCheckpointStore()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=checkpoints,
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(2),
        source_digest=_trusted_source_digest,
    )

    result = await pipeline.run_ingestion(
        job=job,
        upload=upload,
        encrypted_path=tmp_path / "encrypted.pdf",
    )

    assert result.page_markdown == {
        1: "Page 1: no clinical facts.",
        2: "Page 2: no clinical facts.",
    }
    assert result.entities == []
    assert result.evidence == []
    assert [role for role, _payload in manager.calls] == [
        ModelRole.OCR,
        ModelRole.OCR,
        ModelRole.EXTRACTION,
    ]
    for role, payload in manager.calls:
        identity = payload["manifest_identity"]
        assert identity["role"] == role.value
        assert identity["manifest_sha256"] == job.manifest_sha256
        assert identity["roles"] == ["extraction", "ocr", "summary"]
        assert Path(payload["scratch_dir"]).parent == tmp_path / "scratch"
    assert not (tmp_path / "scratch" / str(job.id)).exists()


@pytest.mark.asyncio
async def test_pipeline_forwards_content_free_extraction_chunk_progress(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    observed: list[dict[str, object]] = []

    class ProgressManager(_Manager):
        async def run(self, role, payload, on_progress=None):
            if role is ModelRole.EXTRACTION and on_progress is not None:
                await on_progress(
                    {
                        "role": "extraction",
                        "stage": "processing",
                        "current": 1,
                        "total": 2,
                    }
                )
            return await super().run(role, payload, on_progress)

    pipeline = StrictLocalPipeline(
        manager=ProgressManager(),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(2),
        source_digest=_trusted_source_digest,
        on_progress=observed.append,
    )

    await pipeline.run_ingestion(
        job=job,
        upload=upload,
        encrypted_path=tmp_path / "encrypted.pdf",
    )

    assert {
        "stage": "extraction",
        "model_role": "extraction",
        "worker_current": 1,
        "worker_total": 2,
    } in observed


@pytest.mark.asyncio
async def test_pipeline_wires_internal_worker_liveness_without_public_progress(
    tmp_path: Path,
) -> None:
    """Worker activity can renew a durable lease without inventing UI progress."""

    job, upload = _job_and_upload()
    public_progress: list[dict[str, object]] = []
    durable_heartbeats: list[int] = []

    class LivenessManager(_Manager):
        async def run(
            self,
            role,
            payload,
            on_progress=None,
            on_liveness=None,
        ):
            if on_liveness is not None:
                value = on_liveness()
                if hasattr(value, "__await__"):
                    await value
            return await super().run(role, payload, on_progress)

    pipeline = StrictLocalPipeline(
        manager=LivenessManager(),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
        on_progress=public_progress.append,
        on_liveness=lambda: durable_heartbeats.append(1),
    )

    await pipeline.run_ingestion(
        job=job,
        upload=upload,
        encrypted_path=tmp_path / "encrypted.pdf",
    )

    assert durable_heartbeats == [1, 1]
    assert all("activity" not in item for item in public_progress)


@pytest.mark.asyncio
async def test_pipeline_merges_chunked_extraction_from_one_worker_run(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()

    class LargeDocumentManager(_Manager):
        async def run(self, role, payload, on_progress=None):
            del on_progress
            self.calls.append((role, deepcopy(payload)))
            if role is ModelRole.OCR:
                page_number = payload["page_number"]
                evidence = f"BatchLab{page_number} {page_number}.0 mg"
                return {
                    "markdown": f"{'ordinary text ' * 700}\n{evidence}",
                    "page_number": page_number,
                }
            if role is ModelRole.EXTRACTION:
                return {
                    "result_type": "chunked_clinical_extraction.v1",
                    "chunks": [
                        {
                            "page_numbers": [item["page_number"]],
                            "extraction": {
                                **_empty_extraction(),
                                "labs": [
                                    {
                                        "name": (f"BatchLab{item['page_number']}"),
                                        "value": (f"{item['page_number']}.0"),
                                        "unit": "mg",
                                        "verbatim": (
                                            f"BatchLab{item['page_number']} "
                                            f"{item['page_number']}.0 mg"
                                        ),
                                        "page_number": item["page_number"],
                                        "evidence_excerpt": (
                                            f"BatchLab{item['page_number']} "
                                            f"{item['page_number']}.0 mg"
                                        ),
                                    }
                                ],
                            },
                        }
                        for item in payload["page_markdown"]
                    ],
                }
            raise AssertionError("summary is not part of ingestion")

    manager = LargeDocumentManager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(5),
        source_digest=_trusted_source_digest,
    )

    result = await pipeline.run_ingestion(
        job,
        upload,
        tmp_path / "encrypted.pdf",
    )

    extraction_payloads = [
        payload for role, payload in manager.calls if role is ModelRole.EXTRACTION
    ]
    assert len(extraction_payloads) == 1
    assert len(result.validated_extraction.labs) == 5
    assert len(result.evidence) == 5


@pytest.mark.parametrize(
    "page_numbers",
    ([2, 1], [1, 1], [1], [1, 3]),
)
def test_chunked_extraction_rejects_non_partitioned_pages(
    page_numbers: list[int],
) -> None:
    with pytest.raises(LocalValidationError, match="chunk"):
        StrictLocalPipeline._validate_extraction_result(
            {
                "result_type": "chunked_clinical_extraction.v1",
                "chunks": [
                    {
                        "page_numbers": page_numbers,
                        "extraction": _empty_extraction(),
                    }
                ],
            },
            {1: "Page one.", 2: "Page two."},
            upload_id="chunk-partition",
        )


def test_chunked_extraction_quarantines_fact_referencing_another_chunk() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "HbA1c",
            "value": "6.1",
            "unit": "%",
            "verbatim": "HbA1c 6.1 %",
            "page_number": 2,
            "evidence_excerpt": "HbA1c 6.1 %",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": extraction,
                },
                {
                    "page_numbers": [2],
                    "extraction": _empty_extraction(),
                },
            ],
        },
        {1: "HbA1c 6.1 %", 2: "HbA1c 6.1 %"},
        upload_id="chunk-cross-reference",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fact_validation_failed"]


def test_chunked_extraction_quarantines_only_invalid_fact_in_chunk() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose 95 mg/dL",
        },
        {
            "name": "Invented result",
            "value": "999",
            "unit": "mg/dL",
            "verbatim": "Invented result 999 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Invented result 999 mg/dL",
        },
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": extraction,
                }
            ],
        },
        {1: "Glucose 95 mg/dL"},
        upload_id="chunk-invalid-fact",
    )

    assert [fact.name for fact in result.labs] == ["Glucose"]
    assert len(result.evidence) == 1
    assert result.rejected_fields == ["chunks[0].labs[1]:fact_validation_failed"]


def test_quarantine_rebinds_unique_exact_source_segment_and_maps_fact() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose value 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose value 95 mg/dL",
        }
    ]
    source_row = "| Glucose | 95 | mg/dL |"

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {
            1: "\n".join(
                (
                    "| Test | Value | Unit |",
                    "| --- | --- | --- |",
                    source_row,
                    "| Glucose | 120 | mg/dL |",
                )
            )
        },
        upload_id="locator-rebind-unique",
    )
    records = validated_extraction_to_health_record_dicts(
        result,
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )

    assert len(result.labs) == 1
    assert result.labs[0].verbatim == source_row
    assert result.labs[0].evidence_excerpt == source_row
    assert len(result.evidence) == 1
    assert len(records) == 1
    assert records[0]["record_type"] == "observation"
    assert result.rejected_fields == []


def test_quarantine_does_not_rewrite_already_valid_normalized_locator() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "GLUCOSE 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "panel summary: glucose 95 mg/dl (confirmed)",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Panel summary: GLUCOSE   95 mg/dL (confirmed)"},
        upload_id="locator-rebind-valid",
    )

    assert len(result.labs) == 1
    assert result.labs[0].verbatim == "GLUCOSE 95 mg/dL"
    assert (
        result.labs[0].evidence_excerpt == "panel summary: glucose 95 mg/dl (confirmed)"
    )
    assert result.rejected_fields == []


@pytest.mark.parametrize(
    "page_text",
    (
        "Creatinine 1.0 mg/dL",
        "Glucose 95 mg/dL\nGlucose 95 mg/dL",
    ),
    ids=("absent", "ambiguous"),
)
def test_quarantine_does_not_rebind_absent_or_ambiguous_source(
    page_text: str,
) -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: page_text},
        upload_id="locator-rebind-unresolved",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fact_validation_failed"]


def test_quarantine_never_rebinds_locator_across_pages() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 2,
            "evidence_excerpt": "invalid locator",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {
            1: "Glucose 95 mg/dL",
            2: "Creatinine 1.0 mg/dL",
        },
        upload_id="locator-rebind-page-boundary",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fact_validation_failed"]


def test_quarantine_rebind_does_not_promote_unsupported_optional_fields() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
            "normalized_value": "95",
            "normalization_method": "identity",
            "normalization_version": "model-authored.v1",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL"},
        upload_id="locator-rebind-unsupported",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fact_validation_failed"]


def test_evidence_source_scan_discards_partial_index_when_segment_cap_exceeded() -> (
    None
):
    source = "Glucose 95 mg/dL\n" + ("x\n" * (32 * 1024))

    scan = pipeline_module._source_segments(source, limit=32)

    assert scan.overflowed is True
    assert scan.segments == ()
    assert scan.candidates_inspected == 33


def test_quarantine_does_not_rebind_from_partial_index_after_page_cap() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
        }
    ]
    page_text = "Glucose 95 mg/dL\n" + ("x\n" * (32 * 1024))

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: page_text},
        upload_id="locator-rebind-page-cap",
    )

    assert result.labs == []
    assert result.evidence == ()
    assert result.rejected_fields == ["chunks[0].labs[0]:fact_validation_failed"]


def test_quarantine_reuses_bounded_page_index_for_repeated_invalid_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
        },
        {
            "name": "Creatinine",
            "value": "1.0",
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
        },
    ]
    original_source_segments = pipeline_module._source_segments
    calls = 0

    def counting_source_segments(
        source: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        return original_source_segments(source, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "_source_segments",
        counting_source_segments,
    )

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL\nCreatinine 1.0 mg/dL"},
        upload_id="locator-rebind-page-cache",
    )

    assert [fact.name for fact in result.labs] == ["Glucose", "Creatinine"]
    assert calls == 1


def test_quarantine_fails_closed_when_document_rebind_work_cap_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "_MAX_EVIDENCE_SOURCE_SEGMENTS_PER_PAGE",
        10,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_MAX_EVIDENCE_SOURCE_SEGMENT_WORK_PER_DOCUMENT",
        5,
    )
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": name,
            "value": value,
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": page_number,
            "evidence_excerpt": "invalid locator",
        }
        for page_number, name, value in (
            (1, "Glucose", "95"),
            (2, "Creatinine", "1.0"),
            (3, "Sodium", "140"),
        )
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {
            1: "Glucose 95 mg/dL",
            2: "Creatinine 1.0 mg/dL",
            3: "Sodium 140 mg/dL",
        },
        upload_id="locator-rebind-document-cap",
    )

    assert [fact.name for fact in result.labs] == ["Glucose", "Creatinine"]
    assert result.rejected_fields == ["chunks[0].labs[2]:fact_validation_failed"]


@pytest.mark.parametrize(
    ("name", "value", "page_text"),
    (
        ("G" * 513, "95", f"{'G' * 513} 95 mg/dL"),
        ("Glucose", "9" * 513, f"Glucose {'9' * 513} mg/dL"),
    ),
    ids=("subject", "numeric"),
)
def test_quarantine_rejects_oversized_match_terms_before_regex_compilation(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    page_text: str,
) -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": name,
            "value": value,
            "unit": "mg/dL",
            "verbatim": "invalid locator",
            "page_number": 1,
            "evidence_excerpt": "invalid locator",
        }
    ]
    original_exact_term_pattern = pipeline_module._exact_term_pattern
    calls = 0

    def counting_exact_term_pattern(
        term: str,
        *,
        numeric: bool = False,
    ) -> object:
        nonlocal calls
        calls += 1
        return original_exact_term_pattern(term, numeric=numeric)

    monkeypatch.setattr(
        pipeline_module,
        "_exact_term_pattern",
        counting_exact_term_pattern,
    )

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: page_text},
        upload_id="locator-rebind-oversized-term",
    )

    assert result.labs == []
    assert calls == 0


def test_chunked_extraction_quarantines_adapter_rejected_fact() -> None:
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose 95 mg/dL",
        },
        {
            "name": "Business analyst",
            "verbatim": "Business analyst",
            "page_number": 1,
            "evidence_excerpt": "Business analyst",
        },
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": extraction,
                }
            ],
        },
        {1: "Glucose 95 mg/dL. Business analyst."},
        upload_id="chunk-adapter-rejection",
    )

    assert [fact.name for fact in result.labs] == ["Glucose"]
    assert result.rejected_fields == ["chunks[0].labs[1]:adapter_rejected"]


def test_chunked_extraction_preserves_semantic_duplicate_evidence_from_every_page() -> (
    None
):
    chunks = []
    for page_number in (1, 2):
        extraction = _empty_extraction()
        extraction["labs"] = [
            {
                "name": "HbA1c",
                "value": "6.1",
                "unit": "%",
                "verbatim": "HbA1c 6.1 %",
                "page_number": page_number,
                "evidence_excerpt": "HbA1c 6.1 %",
            }
        ]
        chunks.append(
            {
                "page_numbers": [page_number],
                "extraction": extraction,
            }
        )

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": chunks,
        },
        {1: "HbA1c 6.1 %", 2: "HbA1c 6.1 %"},
        upload_id="chunk-dedup",
    )

    assert [fact.page_number for fact in result.labs] == [1, 2]
    assert [evidence.page_number for evidence in result.evidence] == [1, 2]


def test_corroborating_chunk_facts_map_to_one_record_with_all_evidence() -> None:
    chunks = []
    for page_number in (1, 2):
        extraction = _empty_extraction()
        extraction["labs"] = [
            {
                "name": "HbA1c",
                "value": "6.1",
                "unit": "%",
                "verbatim": "HbA1c 6.1 %",
                "page_number": page_number,
                "evidence_excerpt": "HbA1c 6.1 %",
            }
        ]
        chunks.append(
            {
                "page_numbers": [page_number],
                "extraction": extraction,
            }
        )
    validated = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": chunks,
        },
        {1: "HbA1c 6.1 %", 2: "HbA1c 6.1 %"},
        upload_id="chunk-record-dedup",
    )
    identifiers = [uuid.uuid4() for _ in range(3)]
    entities = to_extracted_entities(validated)
    records = validated_extraction_to_health_record_dicts(
        validated,
        *identifiers,
    )

    deduplicated = dedup_within_document(list(zip(entities, records, strict=True)))

    assert len(deduplicated) == 1
    resource = deduplicated[0][1]["fhir_resource"]
    assert resource["_extraction_metadata"]["_evidence_ids"] == [
        evidence.id for evidence in validated.evidence
    ]


def test_chunked_extraction_omits_conflicting_patient_field_and_keeps_facts() -> None:
    first = _empty_extraction()
    first["patient"] = {
        "name": "Alice Example",
        "date_of_birth": "1970-01-02",
    }
    first["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose 95 mg/dL",
        }
    ]
    second = _empty_extraction()
    second["patient"] = {
        "name": "Bob Example",
        "date_of_birth": "1970-01-02",
    }
    second["labs"] = [
        {
            "name": "Creatinine",
            "value": "1.0",
            "unit": "mg/dL",
            "verbatim": "Creatinine 1.0 mg/dL",
            "page_number": 2,
            "evidence_excerpt": "Creatinine 1.0 mg/dL",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": first,
                },
                {
                    "page_numbers": [2],
                    "extraction": second,
                },
            ],
        },
        {
            1: "Patient Alice Example DOB 1970-01-02. Glucose 95 mg/dL",
            2: "Patient Bob Example DOB 1970-01-02. Creatinine 1.0 mg/dL",
        },
        upload_id="chunk-patient-conflict",
    )

    assert result.patient is not None
    assert result.patient.name is None
    assert result.patient.date_of_birth == "1970-01-02"
    assert [fact.name for fact in result.labs] == ["Glucose", "Creatinine"]
    assert result.rejected_fields == ["patient.name:conflict"]


def test_chunked_extraction_clears_duplicate_model_fact_ids() -> None:
    chunks = []
    pages = {}
    for page_number, name in ((1, "Glucose"), (2, "Creatinine")):
        evidence = f"{name} {page_number}.0 mg/dL"
        pages[page_number] = evidence
        extraction = _empty_extraction()
        extraction["labs"] = [
            {
                "fact_id": "lab-1",
                "name": name,
                "value": f"{page_number}.0",
                "unit": "mg/dL",
                "verbatim": evidence,
                "page_number": page_number,
                "evidence_excerpt": evidence,
            }
        ]
        chunks.append(
            {
                "page_numbers": [page_number],
                "extraction": extraction,
            }
        )

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": chunks,
        },
        pages,
        upload_id="chunk-duplicate-fact-id",
    )

    assert [fact.name for fact in result.labs] == ["Glucose", "Creatinine"]
    assert [fact.fact_id for fact in result.labs] == [None, None]


def test_chunked_extraction_caps_accepted_facts_and_quarantines_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.local_ai.pipeline.MAX_FACTS_PER_CATEGORY",
        2,
    )
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": name,
            "verbatim": name,
            "page_number": 1,
            "evidence_excerpt": name,
        }
        for name in ("LabOne", "LabTwo", "LabThree")
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": extraction,
                }
            ],
        },
        {1: "LabOne. LabTwo. LabThree."},
        upload_id="chunk-fact-cap",
    )

    assert [fact.name for fact in result.labs] == ["LabOne", "LabTwo"]
    assert result.rejected_fields == ["chunks[0].labs[2]:fact_limit_exceeded"]


def test_quarantine_metadata_never_contains_invalid_clinical_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "private invalid clinical canary"
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": canary,
            "verbatim": canary,
            "page_number": 1,
            "evidence_excerpt": canary,
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {
                    "page_numbers": [1],
                    "extraction": extraction,
                }
            ],
        },
        {1: "No matching grounded value appears."},
        upload_id="chunk-private-rejection",
    )

    assert result.labs == []
    assert canary not in repr(result.rejected_fields)
    assert canary not in caplog.text


def test_quarantine_discards_model_diagnostics_and_reserved_sentinel() -> None:
    canary = "private model-authored clinical canary"
    first = _empty_extraction()
    first["unresolved_fields"] = [canary]
    first["rejected_fields"] = [
        canary,
        "chunks[1].labs[0]:fact_validation_failed",
        "extraction.rejections:limit_exceeded",
    ]
    second = _empty_extraction()
    second["labs"] = [
        {
            "name": "Invented result",
            "verbatim": "Invented result",
            "page_number": 2,
            "evidence_excerpt": "Invented result",
        }
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        {
            "result_type": "chunked_clinical_extraction.v1",
            "chunks": [
                {"page_numbers": [1], "extraction": first},
                {"page_numbers": [2], "extraction": second},
            ],
        },
        {1: "Page one.", 2: "No grounded clinical result."},
        upload_id="chunk-model-diagnostics",
    )

    assert result.unresolved_fields == []
    assert result.rejected_fields == ["chunks[1].labs[0]:fact_validation_failed"]
    assert canary not in repr(result)


def test_quarantine_bounds_raw_item_validation_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.local_ai.pipeline._MAX_RAW_ITEMS_PER_DOCUMENT",
        2,
    )
    calls = 0

    def counted_validate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return validate_clinical_extraction(*args, **kwargs)

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.validate_clinical_extraction",
        counted_validate,
    )
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": f"Invented result {index}",
            "verbatim": f"Invented result {index}",
            "page_number": 1,
            "evidence_excerpt": f"Invented result {index}",
        }
        for index in range(4)
    ]
    extraction["unresolved_fields"] = ["private unresolved value"]
    extraction["rejected_fields"] = ["private rejected value"]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "No grounded clinical results."},
        upload_id="chunk-work-budget",
    )

    assert calls == 3  # two raw-item validations plus the final invariant pass
    assert result.rejected_fields == [
        "extraction.validation_work:limit_exceeded",
        "chunks[0].labs[0]:fact_validation_failed",
        "chunks[0].labs[1]:fact_validation_failed",
    ]
    assert result.unresolved_fields == []


def test_quarantine_deduplicates_before_classifying_fact_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.local_ai.pipeline.MAX_FACTS_PER_CATEGORY",
        1,
    )
    extraction = _empty_extraction()
    extraction["labs"] = [
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose 95 mg/dL",
        },
        {
            "name": "Glucose",
            "value": "95",
            "unit": "mg/dL",
            "verbatim": "Glucose 95 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Glucose 95 mg/dL",
        },
        {
            "name": "Creatinine",
            "value": "1.0",
            "unit": "mg/dL",
            "verbatim": "Creatinine 1.0 mg/dL",
            "page_number": 1,
            "evidence_excerpt": "Creatinine 1.0 mg/dL",
        },
    ]

    result = StrictLocalPipeline._validate_extraction_result(
        extraction,
        {1: "Glucose 95 mg/dL. Creatinine 1.0 mg/dL."},
        upload_id="chunk-dedup-before-cap",
    )

    assert [fact.name for fact in result.labs] == ["Glucose"]
    assert result.rejected_fields == ["chunks[0].labs[2]:fact_limit_exceeded"]


@pytest.mark.asyncio
async def test_pipeline_reuses_only_exact_ocr_checkpoint(tmp_path: Path) -> None:
    job, upload = _job_and_upload()
    checkpoints = MemoryCheckpointStore()
    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=checkpoints,
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
    )

    first = await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")
    stored = checkpoints._extractions[(str(job.id), str(upload.id))]
    assert stored.prompt_version == "nuextract3-clinical.v3"
    manager.calls.clear()
    second = await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert first.page_markdown == second.page_markdown
    assert [role for role, _payload in manager.calls] == []


@pytest.mark.asyncio
async def test_pipeline_ignores_mismatched_extraction_checkpoint(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    checkpoints = MemoryCheckpointStore()
    manager = _Manager()
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
    checkpoints._extractions[cache_key] = replace(
        stored,
        page_bindings_sha256="e" * 64,
    )
    raw_stored = checkpoints._raw_extractions[cache_key]
    checkpoints._raw_extractions[cache_key] = replace(
        raw_stored,
        page_bindings_sha256="e" * 64,
    )
    manager.calls.clear()

    await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert [role for role, _payload in manager.calls] == [ModelRole.EXTRACTION]


@pytest.mark.asyncio
async def test_pipeline_quarantines_invalid_fact_from_exact_extraction_checkpoint(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    checkpoints = MemoryCheckpointStore()
    manager = _Manager()
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
    invalid = _empty_extraction()
    invalid["conditions"] = [
        {
            "name": "Ungrounded condition",
            "assertion": "present",
            "verbatim": "Ungrounded condition",
            "page_number": 1,
            "evidence_excerpt": "Ungrounded condition",
        }
    ]
    checkpoints._extractions[cache_key] = replace(
        stored,
        extraction_result=invalid,
    )
    manager.calls.clear()

    result = await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert result.validated_extraction.conditions == []
    assert result.rejected_fields == ["chunks[0].conditions[0]:fact_validation_failed"]
    assert manager.calls == []


@pytest.mark.asyncio
async def test_rtf_bypasses_ocr_and_reuses_text_checkpoint(tmp_path: Path) -> None:
    plaintext = rb"{\rtf1\ansi Follow-up note: no clinical facts.\par}"
    job, upload = _job_and_upload(file_hash=hashlib.sha256(plaintext).hexdigest())
    checkpoints = MemoryCheckpointStore()
    manager = _Manager()
    source = tmp_path / "encrypted.rtf"
    source.write_bytes(b"".join(encrypt_stream([plaintext])))

    def must_not_rasterize(*_args, **_kwargs):
        raise AssertionError("RTF must not enter image rasterization")

    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=checkpoints,
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=must_not_rasterize,
    )

    first = await pipeline.run_ingestion(job, upload, source)
    manager.calls.clear()
    second = await pipeline.run_ingestion(job, upload, source)

    assert first.page_markdown == second.page_markdown
    assert "Follow-up note: no clinical facts." in first.page_markdown[1]
    assert [role for role, _payload in manager.calls] == []
    assert await checkpoints.count_pages(str(job.id)) == 1
    assert not (tmp_path / "scratch" / str(job.id)).exists()


@pytest.mark.asyncio
async def test_rtf_checkpoint_reuse_rejects_changed_plaintext_source(
    tmp_path: Path,
) -> None:
    first_plaintext = rb"{\rtf1\ansi First local record.\par}"
    second_plaintext = rb"{\rtf1\ansi Changed local record.\par}"
    job, upload = _job_and_upload(file_hash=hashlib.sha256(first_plaintext).hexdigest())
    source = tmp_path / "encrypted.rtf"
    source.write_bytes(b"".join(encrypt_stream([first_plaintext])))
    pipeline = StrictLocalPipeline(
        manager=_Manager(),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
    )

    first = await pipeline.run_ingestion(job, upload, source)
    source.write_bytes(b"".join(encrypt_stream([second_plaintext])))

    with pytest.raises(LocalValidationError, match="digest"):
        await pipeline.run_ingestion(job, upload, source)

    assert "First local record." in first.page_markdown[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".pdf", ".tif", ".tiff"])
async def test_image_checkpoint_reuse_rejects_changed_plaintext_source(
    tmp_path: Path,
    suffix: str,
) -> None:
    first_plaintext = b"first local image document"
    second_plaintext = b"changed local image document"
    job, upload = _job_and_upload(file_hash=hashlib.sha256(first_plaintext).hexdigest())
    source = tmp_path / f"encrypted{suffix}"
    source.write_bytes(b"".join(encrypt_stream([first_plaintext])))
    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    await pipeline.run_ingestion(job, upload, source)
    manager.calls.clear()
    source.write_bytes(b"".join(encrypt_stream([second_plaintext])))

    with pytest.raises(LocalValidationError, match="digest"):
        await pipeline.run_ingestion(job, upload, source)

    assert manager.calls == []


@pytest.mark.asyncio
async def test_pipeline_passes_only_bounded_selected_page_images_to_extraction(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()

    class TableManager(_Manager):
        def __init__(self) -> None:
            super().__init__()
            self.selected_images: dict[str, str] | None = None

        async def run(self, role, payload, on_progress=None):
            if role is ModelRole.OCR:
                page_number = payload["page_number"]
                return {
                    "markdown": (
                        "| Test | Value |\n| --- | --- |\n| HbA1c | 6.1% |"
                        if page_number <= 10
                        else "Ordinary prose"
                    ),
                    "page_number": page_number,
                }
            if role is ModelRole.EXTRACTION:
                self.selected_images = dict(payload["image_paths"])
                assert len(self.selected_images) == 8
                assert all(
                    Path(path).is_file() for path in self.selected_images.values()
                )
                assert all(
                    Path(path).parent == Path(payload["scratch_dir"])
                    for path in self.selected_images.values()
                )
                return _empty_extraction()
            return await super().run(role, payload, on_progress)

    manager = TableManager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(12),
        source_digest=_trusted_source_digest,
    )

    await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert manager.selected_images is not None
    assert sorted(manager.selected_images) == [str(index) for index in range(1, 9)]
    assert not (tmp_path / "scratch" / str(job.id)).exists()


@pytest.mark.asyncio
async def test_pipeline_constrains_selected_images_to_aggregate_pixel_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.pipeline as pipeline_module

    job, upload = _job_and_upload()
    monkeypatch.setattr(pipeline_module, "_MAX_SELECTED_IMAGE_PIXELS", 250)

    class TableManager(_Manager):
        def __init__(self) -> None:
            super().__init__()
            self.selected_images: dict[str, str] | None = None

        async def run(self, role, payload, on_progress=None):
            if role is ModelRole.OCR:
                return {
                    "markdown": "| Test | Value |\n| --- | --- |\n| HbA1c | 6.1% |",
                    "page_number": payload["page_number"],
                }
            if role is ModelRole.EXTRACTION:
                self.selected_images = dict(payload["image_paths"])
                return _empty_extraction()
            return await super().run(role, payload, on_progress)

    manager = TableManager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(5),
        source_digest=_trusted_source_digest,
    )

    await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert manager.selected_images is not None
    assert sorted(manager.selected_images) == ["1", "2"]


def test_extraction_payload_rejects_canonical_source_over_worker_limit(
    tmp_path: Path,
) -> None:
    job, _upload = _job_and_upload()
    pipeline = StrictLocalPipeline(
        manager=_Manager(),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
    )
    raw_markdown = "\n" * (2 * 1024 * 1024 + 256)

    with pytest.raises(LocalValidationError, match="content limit"):
        pipeline._extraction_payload(
            str(job.id),
            {1: raw_markdown},
            {},
            tmp_path / "locked-manifest.json",
        )


def test_persistable_evidence_retains_offset_representation_and_hash() -> None:
    page = "Header   here\nHbA1c 6.1 %"
    excerpt = "HbA1c 6.1 %"
    validated = validate_clinical_extraction(
        {
            "labs": [
                {
                    "name": "HbA1c",
                    "verbatim": excerpt,
                    "page_number": 1,
                    "evidence_excerpt": excerpt,
                    "value": "6.1",
                    "unit": "%",
                    "assertion": "present",
                }
            ]
        },
        {1: page},
        upload_id="offset-upload",
    )

    evidence = StrictLocalPipeline._persistable_evidence(validated)[0]

    assert evidence.offset_representation == "whitespace-collapsed-casefold-v1"
    assert evidence.excerpt_sha256 == hashlib.sha256("hba1c 6.1 %".encode()).hexdigest()
    assert evidence.start_offset != page.index(excerpt)


def test_evidence_checkpoint_reuse_requires_hash_and_offset_representation() -> None:
    from app.api.upload import _strict_evidence_checkpoint_matches

    evidence = SimpleNamespace(
        page_number=2,
        excerpt="HbA1c 6.1 %",
        start_offset=12,
        end_offset=24,
        field_paths=("labs[0].value",),
        excerpt_sha256="a" * 64,
        offset_representation="whitespace-collapsed-casefold-v1",
    )
    row = SimpleNamespace(
        page_number=evidence.page_number,
        excerpt=evidence.excerpt,
        start_offset=evidence.start_offset,
        end_offset=evidence.end_offset,
        field_paths=list(evidence.field_paths),
        source_metadata={
            "manifest_sha256": "b" * 64,
            "excerpt_sha256": evidence.excerpt_sha256,
            "offset_representation": evidence.offset_representation,
        },
    )

    assert _strict_evidence_checkpoint_matches(row, evidence, "b" * 64)
    for key in ("excerpt_sha256", "offset_representation"):
        original = row.source_metadata.pop(key)
        assert not _strict_evidence_checkpoint_matches(row, evidence, "b" * 64)
        row.source_metadata[key] = original


@pytest.mark.asyncio
async def test_first_ocr_failure_publishes_only_content_free_progress(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    progress: list[dict[str, object]] = []
    pipeline = StrictLocalPipeline(
        manager=_Manager(failure_role=ModelRole.OCR),
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
        on_progress=progress.append,
    )

    with pytest.raises(LocalWorkerError):
        await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert progress[0] == {
        "stage": "ocr",
        "model_role": "ocr",
        "page_index": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cloud_assisted", "custom_local", "prompt_only"])
async def test_pipeline_rejects_non_strict_mode_before_worker_use(
    tmp_path: Path,
    mode: str,
) -> None:
    job, upload = _job_and_upload()
    job.processing_mode = mode
    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    with pytest.raises(LocalPolicyError):
        await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert manager.calls == []


@pytest.mark.asyncio
async def test_pipeline_rejects_changed_schema_snapshot_before_worker_use(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    upload.processing_schema_version = "clinical-document-extraction.v2"
    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=MemoryCheckpointStore(),
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
    )

    with pytest.raises(LocalPolicyError):
        await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert manager.calls == []


@pytest.mark.asyncio
async def test_worker_failure_is_fail_closed_and_preserves_completed_pages(
    tmp_path: Path,
) -> None:
    job, upload = _job_and_upload()
    checkpoints = MemoryCheckpointStore()
    manager = _Manager(failure_role=ModelRole.EXTRACTION)
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=checkpoints,
        manifest=parse_manifest(job.manifest_snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(2),
        source_digest=_trusted_source_digest,
    )

    with pytest.raises(LocalWorkerError, match="failed"):
        await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert await checkpoints.count_pages(str(job.id)) == 2
    assert [role for role, _payload in manager.calls] == [
        ModelRole.OCR,
        ModelRole.OCR,
        ModelRole.EXTRACTION,
    ]
    assert not (tmp_path / "scratch" / str(job.id)).exists()


def test_pipeline_module_has_no_cloud_capable_imports() -> None:
    source = Path(__file__).parents[1] / "app" / "services" / "local_ai" / "pipeline.py"
    text = source.read_text(encoding="utf-8")

    forbidden = (
        "services.ai.llm",
        "text_extractor",
        "entity_extractor",
        "section_parser",
        "langextract",
        "google",
        "openai",
        "anthropic",
    )
    assert all(name not in text for name in forbidden)


def test_empty_extraction_fixture_is_json_serializable() -> None:
    assert json.loads(json.dumps(_empty_extraction())) == _empty_extraction()


@pytest.mark.asyncio
async def test_database_checkpoint_store_replaces_stale_page_atomically(
    db_session,
) -> None:
    user = User(email="pipeline-checkpoint@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="checkpoint.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/checkpoint.pdf",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="ocr",
    )
    db_session.add(job)
    await db_session.commit()
    store = CheckpointStore(db_session)
    first = OCRCheckpoint(
        page_number=1,
        checkpoint_key="1" * 64,
        image_sha256="2" * 64,
        markdown="Private OCR result",
        width=10,
        height=20,
        warnings=("table_detected",),
    )
    second = OCRCheckpoint(
        page_number=1,
        checkpoint_key="3" * 64,
        image_sha256="4" * 64,
        markdown="Replacement OCR result",
        width=30,
        height=40,
        warnings=(),
    )

    await store.put_ocr_page(job.id, first)
    assert (
        await store.get_ocr_page(job.id, 1, first.checkpoint_key, first.image_sha256)
    ) == first
    await store.put_ocr_page(job.id, second)

    assert (
        await store.get_ocr_page(job.id, 1, first.checkpoint_key, first.image_sha256)
        is None
    )
    assert (
        await store.get_ocr_page(job.id, 1, second.checkpoint_key, second.image_sha256)
    ) == second
    rows = (
        (
            await db_session.execute(
                select(LocalAIPage).where(LocalAIPage.job_id == job.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_strict_autoconfirm_uses_closed_validated_fhir_mapper(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _autoconfirm_and_finish

    user = User(email="strict-mapper@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-mapper.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-mapper.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.commit()
    page = "Hypertension active"
    validated = validate_clinical_extraction(
        {
            "conditions": [
                {
                    "name": "Hypertension",
                    "verbatim": page,
                    "page_number": 1,
                    "evidence_excerpt": page,
                    "assertion": "present",
                }
            ]
        },
        pages={1: page},
        upload_id=str(upload.id),
    )
    entities = to_extracted_entities(validated)

    def reject_generic_mapping(*_args, **_kwargs):
        raise AssertionError(
            "strict local must not use the generic mutable-entity mapper"
        )

    monkeypatch.setattr("app.api.upload._build_record_dicts", reject_generic_mapping)

    await _autoconfirm_and_finish(
        db_session,
        upload,
        upload.id,
        user.id,
        entities,
        SimpleNamespace(primary_visit_date=None),
        original_text=page,
        run_dedup=False,
        strict_validated_extraction=validated,
    )

    record = (
        await db_session.execute(
            select(HealthRecord).where(HealthRecord.source_file_id == upload.id)
        )
    ).scalar_one()
    metadata = record.fhir_resource["_extraction_metadata"]
    assert metadata["_evidence_ids"] == [validated.evidence[0].id]
    assert metadata["_source_page"] == 1
    assert upload.ingestion_status == "completed"


@pytest.mark.asyncio
async def test_strict_autoconfirm_preserves_grounded_fact_date(
    db_session,
) -> None:
    from app.api.upload import _autoconfirm_and_finish

    user = User(email="strict-date@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-date.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-date.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.commit()
    page = "Printed 2025-01-01. HbA1c 6.1 % on 2024-02-03."
    excerpt = "HbA1c 6.1 % on 2024-02-03"
    validated = validate_clinical_extraction(
        {
            "labs": [
                {
                    "name": "HbA1c",
                    "verbatim": excerpt,
                    "page_number": 1,
                    "evidence_excerpt": excerpt,
                    "value": "6.1",
                    "unit": "%",
                    "date": "2024-02-03",
                    "assertion": "present",
                }
            ]
        },
        {1: page},
        upload_id=str(upload.id),
    )
    entities = to_extracted_entities(validated)

    await _autoconfirm_and_finish(
        db_session,
        upload,
        upload.id,
        user.id,
        entities,
        SimpleNamespace(primary_visit_date=None),
        original_text=page,
        run_dedup=False,
        strict_validated_extraction=validated,
    )

    record = (
        await db_session.execute(
            select(HealthRecord).where(HealthRecord.source_file_id == upload.id)
        )
    ).scalar_one()
    assert record.effective_date.date().isoformat() == "2024-02-03"


@pytest.mark.asyncio
async def test_deferred_strict_autoconfirm_rolls_back_records_and_status_together(
    db_session,
) -> None:
    from app.api.upload import _autoconfirm_and_finish

    user = User(email="strict-atomic@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-atomic.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-atomic.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.commit()
    page = "Hypertension active"
    validated = validate_clinical_extraction(
        {
            "conditions": [
                {
                    "name": "Hypertension",
                    "verbatim": page,
                    "page_number": 1,
                    "evidence_excerpt": page,
                    "assertion": "present",
                }
            ]
        },
        {1: page},
        upload_id=str(upload.id),
    )
    entities = to_extracted_entities(validated)

    records = await _autoconfirm_and_finish(
        db_session,
        upload,
        upload.id,
        user.id,
        entities,
        SimpleNamespace(primary_visit_date=None),
        original_text=page,
        run_dedup=False,
        strict_validated_extraction=validated,
        defer_finalization=True,
    )

    assert len(records) == 1
    assert upload.ingestion_status == "processing"
    upload_id = upload.id
    await db_session.rollback()
    persisted = (
        (
            await db_session.execute(
                select(HealthRecord).where(HealthRecord.source_file_id == upload_id)
            )
        )
        .scalars()
        .all()
    )
    assert persisted == []


@pytest.mark.asyncio
async def test_strict_runner_does_not_revive_cancelled_job(db_session) -> None:
    from app.api.upload import _run_strict_local_ingestion_for_upload

    user = User(email="strict-no-revive@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-no-revive.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-no-revive.pdf",
        ingestion_status="cancelled",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        cancel_requested=True,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="cancelled",
        stage="cancelled",
        cancel_requested=True,
    )
    db_session.add(job)
    await db_session.commit()

    with pytest.raises(LocalPolicyError, match="cancelled"):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path("/private/strict-no-revive.pdf"),
            user.id,
        )

    await db_session.rollback()
    await db_session.refresh(job)
    assert job.status == "cancelled"
    assert job.stage == "cancelled"


@pytest.mark.asyncio
async def test_strict_failure_terminalizes_upload_and_job_together(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _run_strict_local_ingestion_for_upload

    user = User(email="strict-paired-failure@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-paired-failure.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-paired-failure.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="queued",
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()

    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: None,
    )

    with pytest.raises(LocalPolicyError, match="unavailable"):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            user.id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert upload.progress_stage is None
    assert upload.processing_completed_at is not None
    assert upload.ingestion_errors == [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": "local_policy_error",
        }
    ]
    assert job.status == "failed"
    assert job.stage == "failed"
    assert job.completed_at == upload.processing_completed_at


@pytest.mark.asyncio
async def test_unstructured_worker_branches_before_cloud_configuration(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _process_unstructured

    user = User(email="strict-branch@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-branch.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-branch.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.commit()
    strict_runner = AsyncMock()

    @asynccontextmanager
    async def session_factory():
        yield db_session

    def fail_cloud_config(*_args, **_kwargs):
        raise AssertionError("cloud configuration was loaded")

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.api.upload._run_strict_local_ingestion_for_upload",
        strict_runner,
        raising=False,
    )
    monkeypatch.setattr("app.services.ai.llm.load_llm_config", fail_cloud_config)

    await _process_unstructured(upload.id, Path(upload.storage_path), user.id)

    strict_runner.assert_awaited_once()


@pytest.mark.asyncio
async def test_strict_runner_rejects_job_with_a_different_upload_snapshot(
    db_session,
) -> None:
    """The worker must not choose a newer mismatched job for this upload."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    user = User(email="strict-job-snapshot-mismatch@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload_manifest, _ = canonicalize_manifest_snapshot(_manifest_payload())
    job_manifest = _manifest_payload()
    job_manifest["pack_revision"] = "apple-m4-16gb-v2"
    job_manifest, _ = canonicalize_manifest_snapshot(job_manifest)
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-mismatch.pdf",
        mime_type="application/pdf",
        file_hash="m" * 64,
        storage_path="/private/strict-mismatch.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=upload_manifest,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    db_session.add(
        LocalAIJob(
            user_id=user.id,
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=job_manifest,
            status="queued",
            stage="queued",
        )
    )
    await db_session.commit()

    with pytest.raises(LocalPolicyError, match="job snapshot is unavailable"):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            user.id,
        )


@pytest.mark.asyncio
async def test_strict_worker_cancellation_becomes_cancelled_not_failed(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _process_unstructured

    user = User(email="strict-cancel@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-cancel.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/strict-cancel.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.commit()

    @asynccontextmanager
    async def session_factory():
        yield db_session

    async def cancel_during_run(_db, current_upload, *_args):
        current_upload.cancel_requested = True
        await _db.commit()
        raise LocalWorkerError("Local worker cancelled.")

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.api.upload._run_strict_local_ingestion_for_upload",
        cancel_during_run,
    )

    await _process_unstructured(upload.id, Path(upload.storage_path), user.id)

    await db_session.refresh(upload)
    assert upload.ingestion_status == "cancelled"
    assert upload.ingestion_errors == []
