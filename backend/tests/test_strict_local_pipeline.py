from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
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
from app.services.local_ai.adapters import to_extracted_entities
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
    manager.calls.clear()
    second = await pipeline.run_ingestion(job, upload, tmp_path / "encrypted.pdf")

    assert first.page_markdown == second.page_markdown
    assert [role for role, _payload in manager.calls] == [ModelRole.EXTRACTION]


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
    assert [role for role, _payload in manager.calls] == [ModelRole.EXTRACTION]
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
            "error_type": "LocalPolicyError",
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
