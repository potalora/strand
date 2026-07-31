"""Privacy regressions for strict-local processing logs and failure state."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import LocalAIJob
from app.models.patient import Patient
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.api.local_ai import _job_response
from app.services.local_ai.errors import LocalWorkerError
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from tests.test_strict_local_pipeline import _manifest_payload


def test_job_projection_drops_untrusted_failure_and_progress_content() -> None:
    """The public job projection must never serialize stored diagnostic text."""
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    job = LocalAIJob(
        user_id=uuid4(),
        upload_id=uuid4(),
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="failed",
        stage="failed",
        progress={"page_index": 1, "private": "sensitive patient content"},
        failure={
            "stage": "extracting",
            "code": "local_worker_error",
            "retryable": True,
            "message": "sensitive patient content",
        },
    )
    job.id = uuid4()
    job.cancel_requested = False
    job.created_at = datetime.now(timezone.utc)
    job.updated_at = datetime.now(timezone.utc)

    payload = _job_response(job).model_dump(mode="json")

    encoded = json.dumps(payload)
    assert payload["progress"] == {"page_index": 1}
    assert payload["failure"] == {
        "stage": "extracting",
        "code": "local_worker_error",
        "retryable": True,
        "checkpoint_preserved": False,
        "cloud_fallback_attempted": False,
    }
    assert "message" not in encoded
    assert "sensitive patient content" not in encoded


@pytest.mark.asyncio
async def test_strict_local_failure_never_logs_or_persists_exception_phi(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model/worker exception must not turn clinical content into telemetry."""
    from app.api.upload import _process_unstructured

    canaries = (
        "Patient Canary-Quasar",
        "DOB 03/14/1978",
        "ocr-fragment-ALPHA-991",
        "warfarin 7.5 mg nightly",
        "summary-canary-BRAVO-771",
    )
    user = User(email="strict-log-privacy@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user.id,
        filename="strict-private.pdf",
        mime_type="application/pdf",
        file_hash="f" * 64,
        storage_path="/private/strict-private.pdf",
        ingestion_status="pending_extraction",
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

    async def fail_with_phi(*_args, **_kwargs) -> None:
        raise RuntimeError(" | ".join(canaries))

    monkeypatch.setattr("app.api.upload.async_session_factory", session_factory)
    monkeypatch.setattr(
        "app.api.upload._run_strict_local_ingestion_for_upload",
        fail_with_phi,
    )

    caplog.set_level(logging.ERROR, logger="app.api.upload")
    await _process_unstructured(upload.id, Path(upload.storage_path), user.id)

    await db_session.refresh(upload)
    persisted_failure_state = json.dumps(
        {
            "errors": upload.ingestion_errors,
            "progress": upload.ingestion_progress,
            "detail": upload.progress_detail,
            "notices": upload.notices,
        },
        default=str,
    )
    captured_logs = caplog.text

    for canary in canaries:
        assert canary not in captured_logs
        assert canary not in persisted_failure_state
    assert upload.ingestion_status == "failed"
    assert upload.ingestion_errors == [
        {
            "error": "Processing failed. Please retry or contact support.",
            "error_type": "local_ai_error",
        }
    ]
    assert str(upload.id) in captured_logs
    assert "local_ai_error" in captured_logs


@pytest.mark.asyncio
async def test_strict_local_summary_recovery_never_logs_exception_message(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recovery logging may identify a job but never format worker exceptions."""
    from app.services.ai.summarizer import resume_grounded_local_summary_jobs

    canary = "summary-worker-secret-canary-FOXTROT-884"
    user = User(email="strict-summary-log@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    patient = Patient(user_id=user.id)
    db_session.add(patient)
    await db_session.flush()
    snapshot, digest = canonicalize_manifest_snapshot(_manifest_payload())
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=user.id,
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="locked",
        user_prompt="grounded",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=user.id,
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        status="queued",
        stage="queued",
        progress={"stage": "queued"},
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    @asynccontextmanager
    async def session_factory():
        yield db_session

    async def fail_summary(*_args, **_kwargs) -> None:
        raise LocalWorkerError(canary)

    monkeypatch.setattr(
        "app.services.ai.summarizer.generate_grounded_local_summary",
        fail_summary,
    )
    caplog.set_level(logging.ERROR, logger="app.services.ai.summarizer")

    await resume_grounded_local_summary_jobs(
        [job.id],
        session_factory=session_factory,
    )

    await db_session.refresh(job)
    persisted = json.dumps(
        {"progress": job.progress, "failure": job.failure},
        default=str,
    )
    assert canary not in caplog.text
    assert canary not in persisted
    assert str(job.id) in caplog.text
    assert job.failure == {
        "stage": "summary",
        "code": "local_worker_error",
        "message": "Strict-local summary did not complete.",
        "retryable": False,
        "cloud_fallback_attempted": False,
    }
