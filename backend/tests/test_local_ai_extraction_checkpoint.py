from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select, text

from app.models.local_ai import LocalAIExtractionCheckpoint, LocalAIJob
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.services.local_ai.checkpoint_store import (
    CheckpointStore,
    ExtractionCheckpoint,
    raw_extraction_result_sha256,
)
from app.services.local_ai.checkpoints import extraction_checkpoint_key
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    canonicalize_manifest_snapshot,
    parse_manifest,
)
from app.services.local_ai.pipeline import StrictLocalPipeline
from app.services.local_ai.protocol import MAX_MESSAGE_BYTES
from app.services.local_ai.types import ModelRole
from tests.test_strict_local_pipeline import (
    _Manager,
    _rasterizer,
    _trusted_source_digest,
)

_PHI_MARKER = "checkpoint-patient-private-marker"
_RAW_PHI_MARKER = "raw-private-checkpoint-canary"


def _manifest_payload() -> dict[str, object]:
    artifacts = []
    for role, marker in (("ocr", "a"), ("extraction", "b"), ("summary", "c")):
        artifacts.append(
            {
                "role": role,
                "repository": f"owner/{role}",
                "revision": marker * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/owner/{role}",
                "decode_limits": {
                    "max_input_tokens": 4096,
                    "max_output_tokens": 1024,
                },
                "files": [
                    {
                        "path": f"{role}/model.safetensors",
                        "sha256": marker * 64,
                        "size": 10,
                    }
                ],
            }
        )
    return {
        "schema_version": 2,
        "pack_revision": "apple-m4-16gb-v2",
        "platform": "apple_silicon",
        "runtime": {
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "d" * 64,
        },
        "validation_suite_version": "fixtures-v1",
        "artifacts": artifacts,
    }


def _checkpoint(
    upload_id: object,
    *,
    source_sha256: str,
    manifest_sha256: str,
) -> ExtractionCheckpoint:
    ocr_text_sha256 = "3" * 64
    schema_version = "clinical-document-extraction.v1"
    prompt_version = "nuextract3-clinical.v1"
    return ExtractionCheckpoint(
        upload_id=str(upload_id),
        checkpoint_key=extraction_checkpoint_key(
            ocr_text_sha256,
            schema_version,
            prompt_version,
            manifest_sha256,
        ),
        source_sha256=source_sha256,
        ocr_text_sha256=ocr_text_sha256,
        page_bindings_sha256="4" * 64,
        manifest_sha256=manifest_sha256,
        schema_version=schema_version,
        prompt_version=prompt_version,
        page_count=2,
        extraction_result={
            "schema_version": "clinical-document-extraction.v1",
            "patient": {"name": _PHI_MARKER},
        },
    )


@pytest.mark.asyncio
async def test_database_store_commits_encrypted_exact_extraction_checkpoint(
    db_session,
) -> None:
    from app.models.local_ai import LocalAIExtractionCheckpoint

    user = User(email="extraction-checkpoint@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, manifest_digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="checkpoint.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/synthetic/checkpoint.pdf",
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
        stage="extraction",
    )
    db_session.add(job)
    await db_session.commit()

    store = CheckpointStore(db_session)
    checkpoint = _checkpoint(
        upload.id,
        source_sha256=upload.file_hash,
        manifest_sha256=manifest_digest,
    )
    job_id = job.id
    upload_id = upload.id
    assert await store.put_extraction(job_id, checkpoint) == checkpoint

    db_session.expire_all()
    assert await store.get_extraction(job_id, upload_id, checkpoint) == checkpoint
    assert (
        await store.get_extraction(
            job_id,
            upload_id,
            replace(checkpoint, page_bindings_sha256="6" * 64),
        )
        is None
    )

    row_count = (
        await db_session.execute(
            text(
                "SELECT count(*) FROM local_ai_extraction_checkpoints "
                "WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )
    ).scalar_one()
    raw_result = (
        await db_session.execute(
            text(
                "SELECT extraction_result FROM local_ai_extraction_checkpoints "
                "WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )
    ).scalar_one()
    assert row_count == 1
    assert _PHI_MARKER.encode() not in bytes(raw_result)
    assert (
        await db_session.get(
            LocalAIExtractionCheckpoint,
            (
                await db_session.execute(
                    text(
                        "SELECT id FROM local_ai_extraction_checkpoints "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                )
            ).scalar_one(),
        )
    ).extraction_result["patient"]["name"] == _PHI_MARKER


@pytest.mark.asyncio
async def test_downstream_rollback_preserves_checkpoint_and_retry_skips_extraction(
    db_session,
    tmp_path: Path,
) -> None:
    user = User(email="checkpoint-retry@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="retry.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/synthetic/retry.pdf",
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
        stage="extraction",
    )
    db_session.add(job)
    await db_session.commit()
    job_id = job.id
    upload_id = upload.id

    manager = _Manager()
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=CheckpointStore(db_session),
        manifest=parse_manifest(snapshot),
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

    upload.extracted_text = "\n\n".join(first.page_markdown.values())
    await db_session.commit()
    upload.ingestion_status = "mapping_fhir"
    await db_session.flush()
    await db_session.rollback()
    job = await db_session.get(LocalAIJob, job_id)
    upload = await db_session.get(UploadedFile, upload_id)
    assert job is not None
    assert upload is not None

    checkpoint_count = (
        (
            await db_session.execute(
                select(LocalAIExtractionCheckpoint.id).where(
                    LocalAIExtractionCheckpoint.job_id == job_id
                )
            )
        )
        .scalars()
        .all()
    )
    manager.calls.clear()
    second = await pipeline.run_ingestion(
        job,
        upload,
        tmp_path / "encrypted.pdf",
    )

    assert len(checkpoint_count) == 1
    assert second.validated_extraction == first.validated_extraction
    assert manager.calls == []


@pytest.mark.asyncio
async def test_validation_failure_preserves_raw_checkpoint_and_retry_skips_manager(
    db_session,
    tmp_path: Path,
) -> None:
    user = User(email="raw-checkpoint-retry@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="raw-retry.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/synthetic/raw-retry.pdf",
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
        stage="extraction",
    )
    db_session.add(job)
    await db_session.commit()
    job_id = job.id
    upload_id = upload.id

    class InvalidExtractionManager(_Manager):
        async def run_attested(self, manifest, role, payload, on_progress=None):
            self._require_attested_manifest(manifest)
            if role is ModelRole.EXTRACTION:
                del on_progress
                self.calls.append((role, payload))
                return {
                    "schema_version": "clinical-document-extraction.v1",
                    "unexpected_envelope_field": _RAW_PHI_MARKER,
                }
            return await super().run_attested(
                manifest,
                role,
                payload,
                on_progress=on_progress,
            )

    manager = InvalidExtractionManager()
    manager.expected_manifest = parse_manifest(snapshot)
    stages: list[str] = []
    pipeline = StrictLocalPipeline(
        manager=manager,
        checkpoints=CheckpointStore(db_session),
        manifest=parse_manifest(snapshot),
        scratch_root=tmp_path / "scratch",
        model_dir=tmp_path / "pack",
        rasterize=_rasterizer(1),
        source_digest=_trusted_source_digest,
        on_progress=lambda value: stages.append(str(value["stage"])),
    )

    with pytest.raises(LocalValidationError):
        await pipeline.run_ingestion(
            job,
            upload,
            tmp_path / "encrypted.pdf",
        )
    await db_session.rollback()
    raw_row = (
        await db_session.execute(
            select(LocalAIExtractionCheckpoint).where(
                LocalAIExtractionCheckpoint.job_id == job_id
            )
        )
    ).scalar_one()
    raw_ciphertext = (
        await db_session.execute(
            text(
                "SELECT raw_extraction_result "
                "FROM local_ai_extraction_checkpoints WHERE job_id = :job_id"
            ),
            {"job_id": job_id},
        )
    ).scalar_one()
    assert (
        raw_row.raw_extraction_result["data"]["unexpected_envelope_field"]
        == _RAW_PHI_MARKER
    )
    assert raw_row.extraction_result is None
    assert _RAW_PHI_MARKER.encode() not in bytes(raw_ciphertext)
    assert stages[-2:] == ["persisting_raw_extraction", "validating_extraction"]

    job = await db_session.get(LocalAIJob, job_id)
    upload = await db_session.get(UploadedFile, upload_id)
    assert job is not None
    assert upload is not None
    manager.calls.clear()

    with pytest.raises(LocalValidationError):
        await pipeline.run_ingestion(
            job,
            upload,
            tmp_path / "encrypted.pdf",
        )

    assert manager.calls == []


def test_raw_extraction_checkpoint_enforces_protocol_byte_limit() -> None:
    with pytest.raises(LocalValidationError):
        raw_extraction_result_sha256("x" * (MAX_MESSAGE_BYTES + 1))
