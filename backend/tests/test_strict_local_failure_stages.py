"""Failure-stage and safe-taxonomy regressions for strict-local ingestion."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from inspect import isawaitable
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.local_ai import ExtractionEvidence, LocalAIJob
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.services.local_ai.errors import (
    LocalAIError,
    LocalPolicyError,
    LocalValidationError,
    LocalWorkerError,
)
from app.services.local_ai.manifest import (
    canonicalize_manifest_snapshot,
    parse_manifest,
)
from app.services.local_ai.types import ProcessingMode
from tests.test_strict_local_pipeline import _manifest_payload


async def _strict_job(
    db_session,
    *,
    label: str,
    upload_snapshot: dict[str, object] | None = None,
    job_snapshot: dict[str, object] | None = None,
) -> tuple[UploadedFile, LocalAIJob]:
    user = User(login_identifier=f"strict-stage-{label}@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload_snapshot = snapshot if upload_snapshot is None else upload_snapshot
    job_snapshot = snapshot if job_snapshot is None else job_snapshot
    upload = UploadedFile(
        user_id=user.id,
        filename=f"{label}.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path=f"/private/{label}.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=upload_snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=job_snapshot,
        status="queued",
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()
    return upload, job


def _pipeline_result(*, evidence: tuple[object, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        page_markdown={1: "No clinical facts."},
        entities=[],
        unresolved_fields=[],
        rejected_fields=[],
        evidence=evidence,
        validated_extraction=object(),
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    upload: UploadedFile,
    *,
    result: SimpleNamespace,
) -> None:
    manifest = parse_manifest(upload.processing_manifest)
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    async def run_ingestion(_self, _job, _upload, _file_path):
        return result

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        run_ingestion,
    )


def _persisted_failure_state(upload: UploadedFile, job: LocalAIJob) -> str:
    return json.dumps(
        {
            "job_failure": job.failure,
            "job_progress": job.progress,
            "job_audit_metadata": job.audit_metadata,
            "upload_errors": upload.ingestion_errors,
            "upload_progress": upload.ingestion_progress,
            "upload_detail": upload.progress_detail,
        },
        default=str,
        sort_keys=True,
    )


@pytest.mark.asyncio
async def test_worker_category_failure_retains_safe_progress_for_api(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal ingestion keeps counters/category and drops worker content."""

    from app.api.local_ai import _job_response
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="worker-category-progress")
    manifest = parse_manifest(upload.processing_manifest)
    canary = "patient-secret-worker-error-canary"
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    async def fail_with_progress(self, _job, _upload, _file_path):
        progress = self.on_progress(
            {
                "stage": "extraction",
                "model_role": "extraction",
                "worker_current": 1,
                "worker_total": 2,
                "attempt": 2,
                "attempt_limit": 12,
                "output_tokens": 300,
                "output_token_limit": 16_384,
                "splits_used": 1,
                "split_limit": 7,
            }
        )
        if isawaitable(progress):
            await progress
        raise LocalWorkerError(
            canary,
            category="invalid_structured_output",
        )

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        fail_with_progress,
    )

    with pytest.raises(LocalWorkerError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["code"] == "invalid_structured_output"
    assert job.progress == {
        "stage": "failed",
        "model_role": "extraction",
        "worker_current": 1,
        "worker_total": 2,
        "attempt": 2,
        "attempt_limit": 12,
        "output_tokens": 300,
        "output_token_limit": 16_384,
        "splits_used": 1,
        "split_limit": 7,
    }
    assert upload.ingestion_errors[0]["error_type"] == "invalid_structured_output"
    api_payload = _job_response(job).model_dump(mode="json", exclude_none=True)
    serialized = json.dumps(api_payload, sort_keys=True)
    assert api_payload["progress"]["output_tokens"] == 300
    assert api_payload["failure"]["code"] == "invalid_structured_output"
    assert canary not in serialized


@pytest.mark.asyncio
async def test_durable_liveness_failure_rolls_back_main_content_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An isolated lease write failure must terminalize without partial content."""

    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="liveness-failure")
    manifest = parse_manifest(upload.processing_manifest)
    canary = "patient-secret-uncommitted-liveness-canary"
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    async def fail_liveness(*_args, **_kwargs) -> None:
        raise RuntimeError(canary)

    async def run_ingestion(self, _job, current_upload, _file_path):
        progress = self.on_progress(
            {
                "stage": "extraction",
                "model_role": "extraction",
                "worker_current": 0,
                "worker_total": 1,
            }
        )
        if isawaitable(progress):
            await progress
        current_upload.extracted_text = canary
        heartbeat = self.on_liveness()
        if isawaitable(heartbeat):
            await heartbeat
        raise AssertionError("liveness failure should stop the pipeline")

    monkeypatch.setattr(
        "app.api.upload._refresh_strict_local_job_lease",
        fail_liveness,
    )
    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        run_ingestion,
    )

    with pytest.raises(LocalAIError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.extracted_text is None
    assert upload.ingestion_status == "failed"
    assert job.status == "failed"
    assert job.failure["stage"] == "extraction"
    assert job.failure["code"] == "local_ai_extraction_failed"
    assert canary not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_stale_strict_worker_failure_after_shutdown_recovery_preserves_queued_work(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation-resistant old worker cannot overwrite shutdown recovery."""
    from app.api import upload as upload_module
    from app.main import _recover_unstructured_jobs_on_startup

    upload, job = await _strict_job(db_session, label="stale-worker-recovery")
    manifest = parse_manifest(upload.processing_manifest)
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    pipeline_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_worker = asyncio.Event()
    new_pipeline_started = asyncio.Event()
    release_new_worker = asyncio.Event()
    pipeline_calls = 0

    async def resist_shutdown(_self, _job, _upload, _file_path):
        nonlocal pipeline_calls
        pipeline_calls += 1
        if pipeline_calls == 2:
            new_pipeline_started.set()
            await release_new_worker.wait()
            return _pipeline_result()
        pipeline_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_worker.wait()
        raise LocalAIError("late strict worker failure")

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        resist_shutdown,
    )
    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def run_old_worker() -> None:
        await upload_module._process_unstructured(
            upload.id,
            Path(upload.storage_path),
            upload.user_id,
        )

    monkeypatch.setattr(upload_module, "async_session_factory", session_factory)
    worker = asyncio.create_task(run_old_worker())
    new_worker: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(pipeline_started.wait(), timeout=1)
        async with session_factory() as verify:
            old_job = await verify.get(LocalAIJob, job.id)
            assert old_job is not None and old_job.started_at is not None
            old_claim_started_at = old_job.started_at
        worker.cancel()
        await asyncio.wait_for(cancellation_seen.wait(), timeout=1)

        async with session_factory() as recovery_db:
            await _recover_unstructured_jobs_on_startup(recovery_db)
            await recovery_db.commit()

        async with session_factory() as verify:
            recovered_upload = await verify.get(UploadedFile, upload.id)
            recovered_job = await verify.get(LocalAIJob, job.id)
            assert recovered_upload is not None and recovered_job is not None
            assert recovered_upload.ingestion_status == "pending_extraction"
            assert (recovered_job.status, recovered_job.stage) == ("queued", "recovery")

        new_worker = asyncio.create_task(run_old_worker())
        await asyncio.wait_for(new_pipeline_started.wait(), timeout=1)
        async with session_factory() as verify:
            new_upload = await verify.get(UploadedFile, upload.id)
            new_job = await verify.get(LocalAIJob, job.id)
            assert new_upload is not None and new_job is not None
            assert new_upload.ingestion_status == "processing"
            assert new_job.status == "processing"
            new_claim_started_at = new_job.started_at
            assert new_claim_started_at is not None
            assert new_claim_started_at != old_claim_started_at

        release_worker.set()
        await asyncio.wait_for(worker, timeout=1)

        async with session_factory() as verify:
            new_upload = await verify.get(UploadedFile, upload.id)
            new_job = await verify.get(LocalAIJob, job.id)
            assert new_upload is not None and new_job is not None
            assert new_upload.ingestion_status == "processing"
            assert new_job.status == "processing"
            assert new_job.started_at == new_claim_started_at
    finally:
        release_worker.set()
        release_new_worker.set()
        if not worker.done():
            worker.cancel()
        if new_worker is not None and not new_worker.done():
            new_worker.cancel()
        pending = [worker]
        if new_worker is not None:
            pending.append(new_worker)
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_checkpoint_flush_has_server_owned_phase_before_persistence(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encrypted extraction fields must flush under a stable internal phase."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="checkpoint-phase")
    _patch_pipeline(monkeypatch, upload, result=_pipeline_result())
    canary = "patient-secret-checkpoint-canary"

    def fail_checkpoint_flush(session, _flush_context, _instances) -> None:
        if (
            job.stage == "persisting_extraction_checkpoint"
            and upload.extracted_text is not None
        ):
            raise RuntimeError(canary)

    event.listen(db_session.sync_session, "before_flush", fail_checkpoint_flush)

    async def complete_mapping(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", complete_mapping)
    try:
        with pytest.raises(LocalAIError):
            await _run_strict_local_ingestion_for_upload(
                db_session,
                upload,
                Path(upload.storage_path),
                upload.user_id,
            )
    finally:
        event.remove(db_session.sync_session, "before_flush", fail_checkpoint_flush)

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["stage"] == "persisting_extraction_checkpoint"
    assert job.failure["code"] == "local_ai_persisting_extraction_checkpoint_failed"
    assert canary not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_evidence_flush_has_server_owned_phase_before_persistence(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encrypted evidence rows must flush under a stable internal phase."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="evidence-phase")
    evidence = SimpleNamespace(
        id="evidence-1",
        page_number=1,
        excerpt="No clinical facts.",
        start_offset=0,
        end_offset=18,
        field_paths=("conditions[0].name",),
        excerpt_sha256="a" * 64,
        offset_representation="whitespace-collapsed-casefold-v1",
    )
    _patch_pipeline(
        monkeypatch,
        upload,
        result=_pipeline_result(evidence=(evidence,)),
    )
    canary = "patient-secret-evidence-canary"

    def fail_evidence_flush(session, _flush_context, _instances) -> None:
        if job.stage != "persisting_evidence":
            return
        if any(isinstance(item, ExtractionEvidence) for item in session.new):
            raise RuntimeError(canary)

    event.listen(db_session.sync_session, "before_flush", fail_evidence_flush)

    async def complete_mapping(*_args, **_kwargs):
        return []

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", complete_mapping)
    try:
        with pytest.raises(LocalAIError):
            await _run_strict_local_ingestion_for_upload(
                db_session,
                upload,
                Path(upload.storage_path),
                upload.user_id,
            )
    finally:
        event.remove(db_session.sync_session, "before_flush", fail_evidence_flush)

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["stage"] == "persisting_evidence"
    assert job.failure["code"] == "local_ai_persisting_evidence_failed"
    assert canary not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_fhir_mapping_failure_uses_phase_code_and_never_persists_phi(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected mapper exception must persist only server-owned taxonomy."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="mapping-phase")
    _patch_pipeline(monkeypatch, upload, result=_pipeline_result())
    canary = "Patient Canary has warfarin 7.5 mg nightly"

    async def fail_mapping(*_args, **_kwargs):
        raise RuntimeError(canary)

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", fail_mapping)

    with pytest.raises(LocalAIError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["stage"] == "mapping_fhir"
    assert job.failure["code"] == "local_ai_mapping_fhir_failed"
    assert upload.ingestion_errors == [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": "local_ai_mapping_fhir_failed",
        }
    ]
    persisted = _persisted_failure_state(upload, job)
    assert canary not in persisted
    assert "RuntimeError" not in persisted


@pytest.mark.asyncio
async def test_finalization_failure_uses_server_owned_phase(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal-state failures must not be misreported as model extraction."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="finalization-phase")
    _patch_pipeline(monkeypatch, upload, result=_pipeline_result())

    async def complete_mapping(*_args, **_kwargs):
        return []

    lock_calls = 0

    async def fail_first_terminal_lock(*_args, **_kwargs):
        nonlocal lock_calls
        lock_calls += 1
        if lock_calls == 1:
            raise RuntimeError("patient-secret-finalization-canary")
        return False

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", complete_mapping)
    monkeypatch.setattr(
        "app.api.upload._lock_strict_terminal_state",
        fail_first_terminal_lock,
    )

    with pytest.raises(LocalAIError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    await db_session.refresh(job)
    assert job.failure["stage"] == "finalizing"
    assert job.failure["code"] == "local_ai_finalizing_failed"


@pytest.mark.asyncio
async def test_strict_finalization_runs_non_llm_dedup_in_terminal_transaction(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict finalization includes exact cross-upload dedup before terminal commit."""
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from app.api.upload import _run_strict_local_ingestion_for_upload
    from app.services.dedup.orchestrator import DedupSummary

    upload, job = await _strict_job(db_session, label="strict-final-dedup")
    pipeline_result = _pipeline_result()
    _patch_pipeline(monkeypatch, upload, result=pipeline_result)

    async def run_with_progress(self, _job, _upload, _file_path):
        progress = self.on_progress(
            {
                "stage": "extraction",
                "model_role": "extraction",
                "worker_current": 1,
                "worker_total": 1,
                "attempt": 3,
                "attempt_limit": 12,
                "output_tokens": 400,
                "output_token_limit": 16_384,
                "active_memory_bytes": 1_000,
                "peak_memory_bytes": 2_000,
            }
        )
        if isawaitable(progress):
            await progress
        return pipeline_result

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        run_with_progress,
    )
    patient_id = uuid4()
    mapped_record = SimpleNamespace(
        id=uuid4(),
        patient_id=patient_id,
        fhir_resource={},
    )

    async def complete_mapping(*_args, **_kwargs):
        return [mapped_record]

    run_dedup = AsyncMock(return_value=DedupSummary(total_candidates=1, auto_merged=1))
    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", complete_mapping)
    monkeypatch.setattr(
        "app.services.dedup.orchestrator.run_upload_dedup",
        run_dedup,
    )

    await _run_strict_local_ingestion_for_upload(
        db_session,
        upload,
        Path(upload.storage_path),
        upload.user_id,
    )

    run_dedup.assert_awaited_once_with(
        upload.id,
        patient_id,
        upload.user_id,
        db_session,
        processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
        commit=False,
    )
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "completed_with_merges"
    assert upload.dedup_summary == {
        "total_candidates": 1,
        "auto_merged": 1,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }
    assert job.status == "completed"
    assert job.progress == {
        "stage": "completed",
        "model_role": "extraction",
        "worker_current": 1,
        "worker_total": 1,
        "attempt": 3,
        "attempt_limit": 12,
        "output_tokens": 400,
        "output_token_limit": 16_384,
        "active_memory_bytes": 1_000,
        "peak_memory_bytes": 2_000,
        "pages_completed": 1,
    }


@pytest.mark.asyncio
async def test_strict_dedup_failure_rolls_back_mapped_records_before_terminal_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dedup exception cannot commit mapped records ahead of failed terminal state."""
    from sqlalchemy import select

    from app.api.upload import _run_strict_local_ingestion_for_upload
    from app.models.patient import Patient
    from app.models.record import HealthRecord

    upload, job = await _strict_job(db_session, label="strict-dedup-rollback")
    _patch_pipeline(monkeypatch, upload, result=_pipeline_result())

    async def map_one_record(db, *_args, **_kwargs):
        patient = Patient(user_id=upload.user_id)
        db.add(patient)
        await db.flush()
        record = HealthRecord(
            patient_id=patient.id,
            user_id=upload.user_id,
            record_type="condition",
            fhir_resource_type="Condition",
            fhir_resource={"resourceType": "Condition"},
            source_format="ai_extracted",
            source_file_id=upload.id,
            status="active",
            code_value="38341003",
            display_text="Synthetic condition",
        )
        db.add(record)
        await db.flush()
        return [record]

    async def fail_dedup(*_args, **_kwargs):
        raise RuntimeError("private-dedup-failure-canary")

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", map_one_record)
    monkeypatch.setattr(
        "app.services.dedup.orchestrator.run_upload_dedup",
        fail_dedup,
    )

    with pytest.raises(LocalAIError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    mapped = (
        (
            await db_session.execute(
                select(HealthRecord).where(HealthRecord.source_file_id == upload.id)
            )
        )
        .scalars()
        .all()
    )
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert mapped == []
    assert upload.ingestion_status == "failed"
    assert job.status == "failed"
    assert job.failure["stage"] == "finalizing"
    assert "private-dedup-failure-canary" not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_local_ai_error_code_survives_server_phase_failure(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known local errors retain their public code while the phase stays precise."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="known-error-phase")
    _patch_pipeline(monkeypatch, upload, result=_pipeline_result())
    canary = "patient-secret-validation-canary"

    async def fail_mapping(*_args, **_kwargs):
        raise LocalValidationError(canary)

    monkeypatch.setattr("app.api.upload._autoconfirm_and_finish", fail_mapping)

    with pytest.raises(LocalValidationError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path(upload.storage_path),
            upload.user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["stage"] == "mapping_fhir"
    assert job.failure["code"] == "local_validation_error"
    assert upload.ingestion_errors == [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": "local_validation_error",
        }
    ]
    assert canary not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_outer_worker_preserves_committed_strict_failure_taxonomy(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic worker boundary must not replace the runner's safe failure."""
    from app.api.upload import _process_unstructured

    upload, job = await _strict_job(db_session, label="outer-preserves-taxonomy")
    upload_id = upload.id
    user_id = upload.user_id
    job_id = job.id
    canary = "Patient Canary takes apixaban 5 mg twice daily"
    expected_failure = {
        "stage": "mapping_fhir",
        "code": "local_validation_error",
        "message": "Strict-local processing did not complete.",
        "model_role": "extraction",
        "retryable": False,
        "checkpoint_preserved": True,
        "cloud_fallback_attempted": False,
    }
    expected_upload_errors = [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": "local_validation_error",
        }
    ]

    @asynccontextmanager
    async def session_factory():
        yield db_session

    async def committed_failure(db, current_upload, *_args) -> None:
        current_job = await db.get(LocalAIJob, job_id)
        assert current_job is not None
        current_job.status = "failed"
        current_job.stage = "failed"
        current_job.failure = expected_failure
        current_upload.ingestion_status = "failed"
        current_upload.progress_stage = None
        current_upload.progress_detail = None
        current_upload.ingestion_errors = expected_upload_errors
        await db.commit()
        raise LocalValidationError(canary)

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.api.upload._run_strict_local_ingestion_for_upload",
        committed_failure,
    )

    await _process_unstructured(
        upload_id,
        Path("/private/outer-preserves-taxonomy.pdf"),
        user_id,
    )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors == expected_upload_errors
    assert job.status == "failed"
    assert job.failure == expected_failure
    assert canary not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_upload_lock_failure_terminalizes_latest_strict_job_and_upload(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recoverable lock exception must leave paired, content-free failure state."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="preflight-lock")
    user_id = upload.user_id
    original_execute = db_session.execute
    execute_calls = 0

    async def fail_first_execute(*args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 1:
            raise RuntimeError("patient-secret-lock-canary")
        return await original_execute(*args, **kwargs)

    monkeypatch.setattr(db_session, "execute", fail_first_execute)

    with pytest.raises(LocalAIError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path("/private/preflight-lock.pdf"),
            user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors[0]["error_type"] == "local_ai_preflight_failed"
    assert job.status == "failed"
    assert job.failure["stage"] == "preflight"
    assert job.failure["code"] == "local_ai_preflight_failed"
    assert "patient-secret-lock-canary" not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_invalid_upload_snapshot_terminalizes_latest_strict_job_and_upload(
    db_session,
) -> None:
    """Snapshot validation failures belong to the paired preflight lifecycle."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(
        db_session,
        label="preflight-snapshot",
        upload_snapshot={"untrusted": "patient-secret-snapshot-canary"},
    )
    upload_id = upload.id
    user_id = upload.user_id

    with pytest.raises(LocalPolicyError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path("/private/preflight-snapshot.pdf"),
            user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.id == upload_id
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors[0]["error_type"] == "local_policy_error"
    assert job.status == "failed"
    assert job.failure["stage"] == "preflight"
    assert job.failure["code"] == "local_policy_error"
    assert "patient-secret-snapshot-canary" not in _persisted_failure_state(upload, job)


@pytest.mark.asyncio
async def test_mismatched_job_lookup_terminalizes_latest_strict_job_and_upload(
    db_session,
) -> None:
    """A stale strict job must fail with its upload instead of remaining queued."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    other_manifest = _manifest_payload()
    other_manifest["pack_revision"] = "apple-m4-16gb-other"
    other_snapshot, _digest = canonicalize_manifest_snapshot(other_manifest)
    upload, job = await _strict_job(
        db_session,
        label="preflight-job-lookup",
        job_snapshot=other_snapshot,
    )
    user_id = upload.user_id

    with pytest.raises(LocalPolicyError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path("/private/preflight-job-lookup.pdf"),
            user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors[0]["error_type"] == "local_policy_error"
    assert job.status == "failed"
    assert job.failure["stage"] == "preflight"
    assert job.failure["code"] == "local_policy_error"


@pytest.mark.asyncio
async def test_cancelled_preflight_keeps_existing_terminal_state(db_session) -> None:
    """Preflight error handling must not turn an acknowledged cancel into failure."""
    from app.api.upload import _run_strict_local_ingestion_for_upload

    upload, job = await _strict_job(db_session, label="preflight-cancelled")
    user_id = upload.user_id
    upload.ingestion_status = "cancelled"
    upload.cancel_requested = True
    upload.ingestion_errors = []
    job.status = "cancelled"
    job.stage = "cancelled"
    job.cancel_requested = True
    job.failure = None
    await db_session.commit()

    with pytest.raises(LocalPolicyError):
        await _run_strict_local_ingestion_for_upload(
            db_session,
            upload,
            Path("/private/preflight-cancelled.pdf"),
            user_id,
        )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "cancelled"
    assert upload.ingestion_errors == []
    assert job.status == "cancelled"
    assert job.stage == "cancelled"
    assert job.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_stage",
    [
        "extraction",
        "persisting_raw_extraction",
        "validating_extraction",
        "persisting_extraction_checkpoint",
    ],
)
async def test_outer_worker_preserves_each_extraction_failure_phase(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    """Every extraction boundary must retain a distinct content-free code."""
    from app.api.upload import _process_unstructured

    upload, job = await _strict_job(
        db_session,
        label=f"phase-{failure_stage.replace('_', '-')}",
    )
    upload_id = upload.id
    user_id = upload.user_id
    manifest = parse_manifest(upload.processing_manifest)
    canary = f"Patient Canary private failure at {failure_stage}"

    @asynccontextmanager
    async def session_factory():
        yield db_session

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    async def fail_at_phase(self, _job, _upload, _file_path):
        assert self.on_progress is not None
        progress = self.on_progress(
            {
                "stage": failure_stage,
                "model_role": "extraction",
            }
        )
        if isawaitable(progress):
            await progress
        raise RuntimeError(canary)

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        fail_at_phase,
    )
    caplog.set_level(logging.ERROR, logger="app.api.upload")

    await _process_unstructured(
        upload_id,
        Path(f"/private/{failure_stage}.pdf"),
        user_id,
    )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    expected_code = f"local_ai_{failure_stage}_failed"
    assert job.failure["stage"] == failure_stage
    assert job.failure["code"] == expected_code
    assert upload.ingestion_errors == [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": expected_code,
        }
    ]
    persisted = _persisted_failure_state(upload, job)
    assert canary not in persisted
    assert canary not in caplog.text
    assert "RuntimeError" not in persisted
    assert "RuntimeError" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "untrusted_field",
    ["stage", "model_role", "worker_current"],
)
async def test_untrusted_pipeline_progress_never_enters_failure_telemetry(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    untrusted_field: str,
) -> None:
    """Only fixed server-owned phase names may become persisted telemetry."""
    from app.api.upload import _process_unstructured

    upload, job = await _strict_job(db_session, label="phase-stage-allowlist")
    upload_id = upload.id
    user_id = upload.user_id
    manifest = parse_manifest(upload.processing_manifest)
    canary = "patient-secret-in-untrusted-stage"

    @asynccontextmanager
    async def session_factory():
        yield db_session

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )

    async def publish_untrusted_stage(self, _job, _upload, _file_path):
        assert self.on_progress is not None
        first = self.on_progress(
            {
                "stage": "extraction",
                "model_role": "extraction",
            }
        )
        if isawaitable(first):
            await first
        untrusted_progress: dict[str, object] = {
            "stage": "extraction",
            "model_role": "extraction",
        }
        untrusted_progress[untrusted_field] = canary
        second = self.on_progress(untrusted_progress)
        if isawaitable(second):
            await second

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        publish_untrusted_stage,
    )

    await _process_unstructured(
        upload_id,
        Path("/private/phase-stage-allowlist.pdf"),
        user_id,
    )

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert job.failure["stage"] == "extraction"
    assert job.failure["code"] == "local_policy_error"
    assert canary not in _persisted_failure_state(upload, job)
