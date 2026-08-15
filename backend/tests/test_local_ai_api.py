"""Authenticated, document-free model-pack lifecycle API tests."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select, update

import app.services.local_ai.pack_operations as pack_operations_module
from app.config import settings
from app.models.ai_summary import AISummaryPrompt
from app.models.audit import AuditLog
from app.models.local_ai import LocalAIJob
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
    canonicalize_manifest_snapshot,
)
from app.services.local_ai.pack_operations import PackOperationStore
from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import (
    _issue_runtime_validation_receipt,
)
from tests.conftest import auth_headers, create_test_patient
from tests.test_strict_local_pipeline import _manifest_payload


@pytest_asyncio.fixture
async def local_ai_operator_headers(client, monkeypatch: pytest.MonkeyPatch):
    headers, user_id = await auth_headers(
        client,
        email="local-ai-operator@example.com",
    )
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({UUID(user_id)}))
    return headers, UUID(user_id)


def _manifest(
    revision: str = "apple-m4-16gb-v1",
) -> tuple[LocalAIManifest, dict[ModelRole, bytes]]:
    contents = {
        ModelRole.OCR: b"ocr-weights",
        ModelRole.EXTRACTION: b"extraction-weights",
        ModelRole.SUMMARY: b"summary-weights",
    }
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository={
                ModelRole.OCR: "sahilchachra/ovisocr2-int4-mlx",
                ModelRole.EXTRACTION: "numind/NuExtract3-mlx-4bits",
                ModelRole.SUMMARY: "mlx-community/Qwen3.5-9B-MLX-4bit",
            }[role],
            revision=str(index) * 40,
            quantization="int4" if role is ModelRole.OCR else "4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 32768, "max_output_tokens": 4096},
            files=(
                ManifestFile(
                    path="model.safetensors",
                    sha256=hashlib.sha256(contents[role]).hexdigest(),
                    size=len(contents[role]),
                ),
            ),
        )
        for index, role in enumerate(ModelRole, start=1)
    )
    return (
        LocalAIManifest(
            schema_version=2,
            pack_revision=revision,
            platform="apple_silicon",
            runtime={
                "name": "mlx-vlm",
                "version": "0.5.0",
                "worker_identity_scheme": "local-ai-worker-bundle.v1",
                "worker_bundle_sha256": "a" * 64,
            },
            validation_suite_version="local-ai-fixtures-v1",
            artifacts=artifacts,
        ),
        contents,
    )


def _write_manifest(path: Path, manifest: LocalAIManifest) -> None:
    path.write_text(
        json.dumps(asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )


def _manifest_dict(manifest: LocalAIManifest) -> dict:
    return json.loads(json.dumps(asdict(manifest)))


def _receipt(manifest: LocalAIManifest):
    return _issue_runtime_validation_receipt(
        manifest,
        WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256=manifest.runtime["worker_bundle_sha256"],
        ),
    )


def _install_pack(
    root: Path,
    manifest: LocalAIManifest,
    contents: dict[ModelRole, bytes],
) -> ArtifactStore:
    store = ArtifactStore(root)
    staging = store.stage(manifest.pack_revision)
    for artifact in manifest.artifacts:
        for model_file in artifact.files:
            path = staging / artifact.role.value / model_file.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[artifact.role])
    store.activate_validated(
        staging,
        manifest,
        _receipt(manifest),
    )
    return store


@pytest.mark.asyncio
async def test_retrying_failed_strict_upload_requeues_its_existing_job(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="strict-retry@example.com",
    )
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="fixture-001.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="a" * 64,
        storage_path="/tmp/fixture-001.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc),
        processing_completed_at=datetime.now(timezone.utc),
        retry_count=3,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        progress={"stage": "extraction"},
        failure={"code": "local_worker_error", "retryable": True},
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(job)
    await db_session.commit()

    with patch(
        "app.api.upload.start_extraction_worker",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["triggered"] == 1
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.processing_started_at is None
    assert upload.processing_completed_at is None
    assert upload.retry_count == 0
    assert job.status == "queued"
    assert job.stage == "queued"
    assert job.progress == {}
    assert job.failure is None
    assert job.started_at is None
    assert job.completed_at is None


@pytest.mark.asyncio
async def test_ingestion_job_status_is_bounded_and_never_returns_failure_message(
    client,
    db_session,
) -> None:
    """Job status projects only bounded counters and stable failure taxonomy."""
    headers, user_id = await auth_headers(client, email="bounded-job@example.com")
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="bounded.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="b" * 64,
        storage_path="/tmp/bounded.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        progress={
            "model_role": "extraction",
            "page_index": 2,
            "page_total": 8,
            "attempt": 1,
            "sensitive": "sensitive patient content",
        },
        failure={
            "stage": "extracting",
            "code": "local_worker_error",
            "retryable": True,
            "checkpoint_preserved": True,
            "cloud_fallback_attempted": False,
            "message": "sensitive patient content",
        },
    )
    db_session.add(job)
    await db_session.commit()

    response = await client.get(
        "/api/v1/local-ai/jobs?kind=ingestion&active_only=false",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()[0]
    assert payload["upload_id"] == str(upload.id)
    assert payload["summary_prompt_id"] is None
    assert payload["processing_mode"] == "validated_strict_local"
    assert payload["progress"] == {
        "model_role": "extraction",
        "page_index": 2,
        "page_total": 8,
        "attempt": 1,
    }
    assert payload["failure"] == {
        "stage": "extracting",
        "code": "local_worker_error",
        "retryable": True,
        "checkpoint_preserved": True,
        "cloud_fallback_attempted": False,
    }
    assert payload["updated_at"]
    assert "message" not in json.dumps(payload)
    assert "sensitive patient content" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_active_jobs_hide_manual_zip_children_until_triggered(
    client,
    db_session,
) -> None:
    """A staged manual child is not active or cancellable in the monitor."""
    headers, user_id = await auth_headers(
        client,
        email="manual-job-gate@example.com",
    )
    manifest, _contents = _manifest()
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="zip-child.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="c" * 64,
        storage_path="/tmp/zip-child.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        manual_extraction_required=True,
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="queued",
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()

    before = await client.get(
        "/api/v1/local-ai/jobs?kind=ingestion&active_only=true",
        headers=headers,
    )
    all_jobs = await client.get(
        "/api/v1/local-ai/jobs?kind=ingestion&active_only=false",
        headers=headers,
    )
    trigger = await client.post(
        "/api/v1/upload/trigger-extraction",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )
    after = await client.get(
        "/api/v1/local-ai/jobs?kind=ingestion&active_only=true",
        headers=headers,
    )

    assert before.status_code == 200
    assert before.json() == []
    assert [item["id"] for item in all_jobs.json()] == [str(job.id)]
    assert trigger.status_code == 200
    assert trigger.json()["triggered"] == 1
    assert [item["id"] for item in after.json()] == [str(job.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_status", "expected_status"),
    [
        (None, "strict_job_unavailable"),
        ("failed", "strict_job_not_queued"),
    ],
)
async def test_manual_strict_child_requires_queued_uncancelled_job(
    client,
    db_session,
    job_status: str | None,
    expected_status: str,
) -> None:
    """Manual release fails closed without a resumable strict ingestion job."""
    headers, user_id = await auth_headers(
        client,
        email=f"manual-strict-{job_status or 'missing'}@example.com",
    )
    manifest, _contents = _manifest()
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="strict-child.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash=f"strict-manual-{job_status or 'missing'}",
        storage_path="/tmp/medtimeline-zip-set-example/strict-child.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        manual_extraction_required=True,
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    if job_status is not None:
        db_session.add(
            LocalAIJob(
                user_id=UUID(user_id),
                upload_id=upload.id,
                kind="ingestion",
                processing_mode="validated_strict_local",
                manifest_snapshot=_manifest_dict(manifest),
                status=job_status,
                stage=job_status,
                cancel_requested=False,
            )
        )
    await db_session.commit()

    response = await client.post(
        "/api/v1/upload/trigger-extraction",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {"upload_id": str(upload.id), "status": expected_status}
    ]
    await db_session.refresh(upload)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.manual_extraction_required is True


@pytest.mark.asyncio
async def test_retry_ingestion_job_requeues_owned_pair_and_preserves_pages(
    client,
    db_session,
) -> None:
    """Only a retryable failed ingestion job can atomically re-enter the worker queue."""
    from app.models.local_ai import LocalAIPage

    headers, user_id = await auth_headers(client, email="retry-job-owner@example.com")
    other_headers, _other_id = await auth_headers(
        client,
        email="retry-job-other@example.com",
    )
    manifest, _contents = _manifest()

    async def make_job(
        *,
        status: str,
        retryable: bool = True,
        cancel_requested: bool = False,
    ) -> tuple[UploadedFile, LocalAIJob]:
        upload = UploadedFile(
            id=uuid4(),
            user_id=UUID(user_id),
            filename=f"retry-{status}-{uuid4().hex}.pdf",
            mime_type="application/pdf",
            file_size_bytes=100,
            file_hash=uuid4().hex * 2,
            storage_path=f"/tmp/retry-{uuid4().hex}.pdf",
            ingestion_status="failed",
            file_category="unstructured",
            processing_mode="validated_strict_local",
            processing_manifest=_manifest_dict(manifest),
            processing_schema_version="clinical-document-extraction.v1",
            processing_started_at=datetime.now(timezone.utc),
            processing_completed_at=datetime.now(timezone.utc),
            progress_stage="failed",
            progress_detail={"sensitive": "patient content"},
            ingestion_errors=[{"error": "terminal failure"}],
            retry_count=3,
        )
        db_session.add(upload)
        await db_session.flush()
        job = LocalAIJob(
            user_id=UUID(user_id),
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=_manifest_dict(manifest),
            status=status,
            stage=status,
            failure={
                "stage": status,
                "code": "local_worker_error",
                "retryable": retryable,
            },
            cancel_requested=cancel_requested,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        db_session.add(job)
        await db_session.flush()
        return upload, job

    upload, retryable_job = await make_job(status="failed")
    upload.manual_extraction_required = True
    page = LocalAIPage(
        job_id=retryable_job.id,
        page_number=1,
        checkpoint_key="1" * 64,
        image_sha256="2" * 64,
        ocr_result={"markdown": "retained checkpoint"},
        warnings=[],
    )
    db_session.add(page)
    _active_upload, active_job = await make_job(status="processing")
    _completed_upload, completed_job = await make_job(status="completed")
    _cancelled_upload, cancelled_job = await make_job(status="cancelled")
    _non_retryable_upload, non_retryable_job = await make_job(
        status="failed", retryable=False
    )
    await db_session.commit()

    with patch("app.api.upload.start_extraction_worker", new=Mock()) as wake_worker:
        cross_owner = await client.post(
            f"/api/v1/local-ai/jobs/{retryable_job.id}/retry",
            headers=other_headers,
        )
        response = await client.post(
            f"/api/v1/local-ai/jobs/{retryable_job.id}/retry",
            headers=headers,
        )
        active = await client.post(
            f"/api/v1/local-ai/jobs/{active_job.id}/retry", headers=headers
        )
        completed = await client.post(
            f"/api/v1/local-ai/jobs/{completed_job.id}/retry", headers=headers
        )
        cancelled = await client.post(
            f"/api/v1/local-ai/jobs/{cancelled_job.id}/retry", headers=headers
        )
        non_retryable = await client.post(
            f"/api/v1/local-ai/jobs/{non_retryable_job.id}/retry", headers=headers
        )

    assert cross_owner.status_code == 404
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    for rejected in (active, completed, cancelled, non_retryable):
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == "This job cannot be retried."
    wake_worker.assert_called_once()
    await db_session.refresh(upload)
    await db_session.refresh(retryable_job)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.processing_started_at is None
    assert upload.processing_completed_at is None
    assert upload.progress_stage is None
    assert upload.progress_detail is None
    assert upload.ingestion_errors == []
    assert upload.retry_count == 0
    assert upload.manual_extraction_required is False
    assert retryable_job.status == "queued"
    assert retryable_job.stage == "queued"
    assert retryable_job.progress == {}
    assert retryable_job.failure is None
    assert retryable_job.started_at is None
    assert retryable_job.completed_at is None
    retained_page = await db_session.get(LocalAIPage, page.id)
    assert retained_page is not None
    assert retained_page.checkpoint_key == "1" * 64
    assert retained_page.image_sha256 == "2" * 64
    assert retained_page.ocr_result == {"markdown": "retained checkpoint"}
    assert retained_page.warnings == []


@pytest.mark.asyncio
async def test_retry_ingestion_job_locks_upload_before_job(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The job-ID retry path uses the same upload-then-job lock order as upload retry."""
    headers, user_id = await auth_headers(client, email="retry-lock-order@example.com")
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="retry-lock-order.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="c" * 64,
        storage_path="/tmp/retry-lock-order.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        failure={"stage": "failed", "code": "local_worker_error", "retryable": True},
    )
    db_session.add(job)
    await db_session.commit()

    lock_order: list[str] = []
    original_execute = db_session.execute

    async def track_execute(statement, *args, **kwargs):
        sql = str(statement)
        if "FOR UPDATE" in sql:
            if "FROM uploaded_files" in sql:
                lock_order.append("upload")
            elif "FROM local_ai_jobs" in sql:
                lock_order.append("job")
        return await original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", track_execute)
    with patch("app.api.upload.start_extraction_worker", new=Mock()):
        response = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/retry",
            headers=headers,
        )

    assert response.status_code == 200
    assert lock_order[:2] == ["upload", "job"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_mode", "upload_mode"),
    [
        ("cloud_assisted", "validated_strict_local"),
        ("validated_strict_local", "cloud_assisted"),
    ],
)
async def test_retry_ingestion_job_rejects_mismatched_strict_local_modes(
    client,
    db_session,
    job_mode: str,
    upload_mode: str,
) -> None:
    """Both halves of a strict-local retry must retain the validated mode."""
    headers, user_id = await auth_headers(
        client,
        email=f"retry-mode-{job_mode}-{upload_mode}@example.com",
    )
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="retry-mode.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash=uuid4().hex * 2,
        storage_path="/tmp/retry-mode.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode=upload_mode,
        processing_manifest=(
            _manifest_dict(manifest)
            if upload_mode == "validated_strict_local"
            else None
        ),
        processing_schema_version=(
            "clinical-document-extraction.v1"
            if upload_mode == "validated_strict_local"
            else None
        ),
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode=job_mode,
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        failure={"stage": "failed", "code": "local_worker_error", "retryable": True},
    )
    db_session.add(job)
    await db_session.commit()

    with patch("app.api.upload.start_extraction_worker", new=Mock()) as wake_worker:
        response = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/retry",
            headers=headers,
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "This job cannot be retried."
    wake_worker.assert_not_called()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert job.status == "failed"


@pytest.mark.asyncio
async def test_retry_commit_precedes_audit_and_worker_wake(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An audit failure cannot roll back the paired retry or wake an uncommitted job."""
    headers, user_id = await auth_headers(client, email="retry-commit@example.com")
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="retry-commit.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="d" * 64,
        storage_path="/tmp/retry-commit.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        failure={"stage": "failed", "code": "local_worker_error", "retryable": True},
    )
    db_session.add(job)
    await db_session.commit()

    events: list[str] = []
    original_commit = db_session.commit

    async def track_commit() -> None:
        await original_commit()
        events.append("commit")

    async def fail_audit(*_args, **_kwargs) -> None:
        events.append("audit")
        raise RuntimeError("synthetic audit failure")

    def wake_worker() -> None:
        events.append("wake")

    monkeypatch.setattr(db_session, "commit", track_commit)
    monkeypatch.setattr("app.api.local_ai.log_audit_event", fail_audit)
    monkeypatch.setattr("app.api.upload.start_extraction_worker", wake_worker)

    with pytest.raises(RuntimeError, match="synthetic audit failure"):
        await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/retry",
            headers=headers,
        )

    assert events == ["commit", "audit"]
    await db_session.rollback()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "pending_extraction"
    assert job.status == "queued"


@pytest.mark.asyncio
async def test_job_cancel_terminalizes_queued_strict_ingestion_pair(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="queued-cancel.pdf",
        mime_type="application/pdf",
        file_size_bytes=128,
        file_hash=uuid4().hex,
        storage_path="/tmp/queued-cancel.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="queued",
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()

    response = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/cancel",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.cancel_requested) == ("cancelled", True)
    assert upload.processing_completed_at is not None
    assert (job.status, job.stage, job.cancel_requested) == (
        "cancelled",
        "cancelled",
        True,
    )
    assert job.completed_at == upload.processing_completed_at


@pytest.mark.asyncio
async def test_job_cancel_commits_processing_pair_before_worker_ipc(
    client,
    db_session,
) -> None:
    import app.database as database_module

    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    claimed_at = datetime.now(timezone.utc)
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="processing-job-cancel.pdf",
        mime_type="application/pdf",
        file_size_bytes=128,
        file_hash=uuid4().hex,
        storage_path="/tmp/processing-job-cancel.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=claimed_at,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="ocr",
        started_at=claimed_at,
    )
    db_session.add(job)
    await db_session.commit()

    events: list[str] = []
    original_commit = db_session.commit

    async def track_commit() -> None:
        await original_commit()
        events.append("commit")

    async def observe_registered(job_id: str) -> bool:
        events.append("cancel_registered")
        async with database_module.async_session_factory() as observer:
            durable_upload = await observer.get(UploadedFile, upload.id)
            durable_job = await observer.get(LocalAIJob, job.id)
            assert durable_upload is not None and durable_upload.cancel_requested
            assert durable_job is not None and durable_job.cancel_requested
            assert durable_upload.ingestion_status == "processing"
            assert durable_job.status == "processing"
        return False

    async def observe_fallback(job_id: str) -> None:
        events.append("cancel")

    db_session.commit = track_commit  # type: ignore[method-assign]
    with (
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
            new=AsyncMock(side_effect=observe_registered),
        ),
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel",
            new=AsyncMock(side_effect=observe_fallback),
        ),
    ):
        response = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=headers,
        )

    assert response.status_code == 200
    assert events[:3] == ["commit", "cancel_registered", "cancel"]
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.cancel_requested) == (
        "processing",
        True,
    )
    assert (job.status, job.cancel_requested) == ("processing", True)


@pytest.mark.asyncio
async def test_job_cancel_reloads_pair_after_stale_route_discovery(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.database as database_module

    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    claimed_at = datetime.now(timezone.utc)
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="stale-route-cancel.pdf",
        mime_type="application/pdf",
        file_size_bytes=128,
        file_hash=uuid4().hex,
        storage_path="/tmp/stale-route-cancel.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=claimed_at,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="ocr",
        started_at=claimed_at,
    )
    db_session.add(job)
    await db_session.commit()

    original_execute = db_session.execute
    race_injected = False

    async def inject_incoherent_upload(statement, *args, **kwargs):
        nonlocal race_injected
        if not race_injected and "FROM uploaded_files" in str(statement):
            race_injected = True
            async with database_module.async_session_factory() as racer:
                await racer.execute(
                    update(UploadedFile)
                    .where(UploadedFile.id == upload.id)
                    .values(ingestion_status="completed")
                )
                await racer.commit()
        return await original_execute(statement, *args, **kwargs)

    db_session.execute = inject_incoherent_upload  # type: ignore[method-assign]
    with patch(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        new_callable=AsyncMock,
    ) as cancel_registered:
        response = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=headers,
        )

    assert race_injected
    assert response.status_code == 409
    assert response.json()["detail"] == "This job cannot be cancelled."
    cancel_registered.assert_not_awaited()
    await db_session.rollback()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.cancel_requested) == (
        "completed",
        False,
    )
    assert (job.status, job.cancel_requested) == ("processing", False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_defect",
    (
        "different_upload_manifest",
        "noncanonical_upload_manifest",
        "different_job_digest",
        "wrong_upload_schema",
    ),
)
async def test_strict_cancellation_rejects_identity_mismatch_without_mutation(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    identity_defect: str,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload_snapshot = copy.deepcopy(snapshot)
    schema_version = "clinical-document-extraction.v1"
    if identity_defect == "different_upload_manifest":
        changed = copy.deepcopy(snapshot)
        changed["pack_revision"] = "different-revision"
        upload_snapshot, _changed_digest = canonicalize_manifest_snapshot(changed)
    elif identity_defect == "noncanonical_upload_manifest":
        upload_snapshot["artifacts"][0]["license"] = "APACHE-2.0"
    elif identity_defect == "wrong_upload_schema":
        schema_version = "clinical-document-extraction.v2"

    upload = UploadedFile(
        user_id=UUID(user_id),
        filename=f"{identity_defect}.pdf",
        mime_type="application/pdf",
        file_size_bytes=128,
        file_hash=uuid4().hex,
        storage_path=f"/tmp/{identity_defect}.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=upload_snapshot,
        processing_schema_version=schema_version,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="queued",
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()

    if identity_defect == "different_job_digest":
        original_canonicalize = canonicalize_manifest_snapshot

        def mismatched_digest(value):
            canonical, _actual_digest = original_canonicalize(value)
            return canonical, "f" * 64 if digest != "f" * 64 else "e" * 64

        monkeypatch.setattr(
            "app.services.local_ai.ingestion_lifecycle.canonicalize_manifest_snapshot",
            mismatched_digest,
        )

    job_response = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/cancel",
        headers=headers,
    )
    bulk_response = await client.post(
        "/api/v1/upload/cancel",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )

    assert job_response.status_code == 409
    assert job_response.json() == {"detail": "This job cannot be cancelled."}
    assert bulk_response.status_code == 200
    assert bulk_response.json() == {"cancelled": [], "skipped": [str(upload.id)]}
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (job.status, job.cancel_requested) == ("queued", False)
    assert (upload.ingestion_status, upload.cancel_requested) == (
        "pending_extraction",
        False,
    )


@pytest.mark.asyncio
async def test_strict_cancellation_rejects_incoherent_pair_without_repair(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="incoherent.pdf",
        mime_type="application/pdf",
        file_size_bytes=128,
        file_hash=uuid4().hex,
        storage_path="/tmp/incoherent.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="preflight",
    )
    db_session.add(job)
    await db_session.commit()

    job_response = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/cancel",
        headers=headers,
    )
    bulk_response = await client.post(
        "/api/v1/upload/cancel",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )

    assert job_response.status_code == 409
    assert bulk_response.json() == {"cancelled": [], "skipped": [str(upload.id)]}
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (job.status, job.cancel_requested) == ("processing", False)
    assert (upload.ingestion_status, upload.cancel_requested) == (
        "pending_extraction",
        False,
    )


@pytest.mark.asyncio
async def test_retryable_failed_hydration_is_owner_scoped_and_excludes_manual_children(
    client,
    db_session,
) -> None:
    owner_headers, owner_id = await auth_headers(
        client,
        email="retryable-hydration-owner@example.com",
    )
    _other_headers, other_id = await auth_headers(
        client,
        email="retryable-hydration-other@example.com",
    )
    owner_patient = await create_test_patient(db_session, owner_id)
    other_patient = await create_test_patient(db_session, other_id)
    manifest, _contents = _manifest()
    snapshot = _manifest_dict(manifest)

    async def add_summary_job(
        *,
        user_id: str,
        patient_id: UUID,
        status: str,
        failure: dict | None = None,
    ) -> LocalAIJob:
        prompt = AISummaryPrompt(
            id=uuid4(),
            user_id=UUID(user_id),
            patient_id=patient_id,
            summary_type="full",
            processing_mode="validated_strict_local",
            scope_filter={},
            system_prompt="synthetic system prompt",
            user_prompt="synthetic user prompt",
            target_model="locked-local-summary",
            suggested_config={},
            record_count=1,
            model_provenance={},
            generated_at=datetime.now(timezone.utc),
        )
        job = LocalAIJob(
            user_id=UUID(user_id),
            summary_prompt_id=prompt.id,
            kind="summary",
            processing_mode="validated_strict_local",
            manifest_snapshot=snapshot,
            status=status,
            stage=status,
            failure=failure,
        )
        db_session.add_all([prompt, job])
        await db_session.flush()
        return job

    async def add_ingestion_job(
        *,
        status: str,
        failure: dict | None = None,
        manual: bool = False,
        filename: str,
    ) -> LocalAIJob:
        upload = UploadedFile(
            user_id=UUID(owner_id),
            filename=filename,
            mime_type="application/pdf",
            file_size_bytes=128,
            file_hash=uuid4().hex,
            storage_path=f"/tmp/{filename}",
            ingestion_status="failed" if status == "failed" else "completed",
            file_category="unstructured",
            manual_extraction_required=manual,
            processing_mode="validated_strict_local",
            processing_manifest=snapshot,
            processing_schema_version="clinical-document-extraction.v1",
        )
        db_session.add(upload)
        await db_session.flush()
        job = LocalAIJob(
            user_id=UUID(owner_id),
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=snapshot,
            status=status,
            stage=status,
            failure=failure,
        )
        db_session.add(job)
        await db_session.flush()
        return job

    active_summary = await add_summary_job(
        user_id=owner_id,
        patient_id=owner_patient.id,
        status="processing",
    )
    retryable_failure = await add_summary_job(
        user_id=owner_id,
        patient_id=owner_patient.id,
        status="failed",
        failure={"code": "worker_timeout", "retryable": True},
    )
    manual_child = await add_ingestion_job(
        status="failed",
        failure={"code": "worker_timeout", "retryable": True},
        manual=True,
        filename="manual-zip-child.pdf",
    )
    non_retryable = await add_ingestion_job(
        status="failed",
        failure={"code": "policy_failure", "retryable": False},
        filename="non-retryable.pdf",
    )
    cancelled = await add_ingestion_job(
        status="cancelled",
        filename="cancelled.pdf",
    )
    completed = await add_ingestion_job(
        status="completed",
        filename="completed.pdf",
    )
    foreign_retryable = await add_summary_job(
        user_id=other_id,
        patient_id=other_patient.id,
        status="failed",
        failure={"code": "worker_timeout", "retryable": True},
    )
    await db_session.commit()

    expanded = await client.get(
        "/api/v1/local-ai/jobs?active_only=true&include_retryable_failed=true",
        headers=owner_headers,
    )
    active_only = await client.get(
        "/api/v1/local-ai/jobs?active_only=true",
        headers=owner_headers,
    )
    all_owner_jobs = await client.get(
        "/api/v1/local-ai/jobs?active_only=false&include_retryable_failed=true",
        headers=owner_headers,
    )
    summary_jobs = await client.get(
        "/api/v1/local-ai/jobs?kind=summary&active_only=true&include_retryable_failed=true",
        headers=owner_headers,
    )

    assert expanded.status_code == 200
    assert {item["id"] for item in expanded.json()} == {
        str(active_summary.id),
        str(retryable_failure.id),
    }
    assert {item["id"] for item in active_only.json()} == {str(active_summary.id)}
    assert {item["id"] for item in all_owner_jobs.json()} == {
        str(active_summary.id),
        str(retryable_failure.id),
        str(manual_child.id),
        str(non_retryable.id),
        str(cancelled.id),
        str(completed.id),
    }
    assert {item["id"] for item in summary_jobs.json()} == {
        str(active_summary.id),
        str(retryable_failure.id),
    }
    assert all("message" not in json.dumps(item) for item in expanded.json())
    excluded = {
        str(manual_child.id),
        str(non_retryable.id),
        str(cancelled.id),
        str(completed.id),
        str(foreign_retryable.id),
    }
    assert excluded.isdisjoint({item["id"] for item in expanded.json()})


@pytest.mark.asyncio
async def test_retryable_failed_hydration_skips_malformed_json_before_limit(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="retryable-hydration-malformed@example.com",
    )
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    base_time = datetime.now(timezone.utc)

    async def add_job(
        *,
        index: int,
        status: str,
        failure: dict | None,
        created_at: datetime,
    ) -> LocalAIJob:
        upload = UploadedFile(
            user_id=UUID(user_id),
            filename=f"hydration-limit-{index}.pdf",
            mime_type="application/pdf",
            file_size_bytes=128,
            file_hash=uuid4().hex,
            storage_path=f"/tmp/hydration-limit-{index}.pdf",
            ingestion_status="processing" if status == "processing" else "failed",
            file_category="unstructured",
            processing_mode="validated_strict_local",
            processing_manifest=snapshot,
            processing_schema_version="clinical-document-extraction.v1",
            created_at=created_at,
        )
        db_session.add(upload)
        await db_session.flush()
        job = LocalAIJob(
            user_id=UUID(user_id),
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=snapshot,
            status=status,
            stage=status,
            failure=failure,
            created_at=created_at,
        )
        db_session.add(job)
        await db_session.flush()
        return job

    for index in range(49):
        await add_job(
            index=index,
            status="processing",
            failure=None,
            created_at=base_time - timedelta(seconds=index),
        )
    malformed_jobs = []
    malformed_failures = (
        None,
        {},
        {"retryable": None},
        {"retryable": "true"},
        {"retryable": "yes"},
        {"retryable": 1},
        {"retryable": {}},
        {"retryable": []},
        {"retryable": False},
    )
    for offset, failure in enumerate(malformed_failures, start=49):
        malformed_jobs.append(
            await add_job(
                index=offset,
                status="failed",
                failure=failure,
                created_at=base_time - timedelta(seconds=offset),
            )
        )
    valid_retryable = await add_job(
        index=58,
        status="failed",
        failure={"retryable": True},
        created_at=base_time - timedelta(seconds=100),
    )
    await db_session.commit()

    response = await client.get(
        "/api/v1/local-ai/jobs?active_only=true&include_retryable_failed=true",
        headers=headers,
    )

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()]
    assert len(ids) == 50
    assert str(valid_retryable.id) in ids
    assert {str(job.id) for job in malformed_jobs}.isdisjoint(ids)


@pytest.mark.asyncio
async def test_summary_job_status_and_cancel_are_owner_scoped_and_content_free(
    client,
    db_session,
) -> None:
    owner_headers, owner_id = await auth_headers(
        client,
        email="summary-job-owner@example.com",
    )
    other_headers, _other_id = await auth_headers(
        client,
        email="summary-job-other@example.com",
    )
    patient = await create_test_patient(db_session, owner_id)
    manifest, _contents = _manifest()
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(owner_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="sensitive system prompt",
        user_prompt="sensitive patient content",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=1,
        model_provenance={"sensitive": "must not be returned"},
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(owner_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="processing",
        stage="summary",
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    with patch(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        new_callable=AsyncMock,
        return_value=True,
    ) as cancel_registered:
        other_list = await client.get(
            "/api/v1/local-ai/jobs?kind=summary&active_only=true",
            headers=other_headers,
        )
        other_get = await client.get(
            f"/api/v1/local-ai/jobs/{job.id}",
            headers=other_headers,
        )
        other_cancel = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=other_headers,
        )
        owner_list = await client.get(
            "/api/v1/local-ai/jobs?kind=summary&active_only=true",
            headers=owner_headers,
        )
        owner_cancel = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=owner_headers,
        )

    assert other_list.status_code == 200
    assert other_list.json() == []
    assert other_get.status_code == 404
    assert other_cancel.status_code == 404
    assert owner_list.status_code == 200
    assert len(owner_list.json()) == 1
    owner_payload = owner_list.json()[0]
    assert owner_payload["id"] == str(job.id)
    assert owner_payload["upload_id"] is None
    assert owner_payload["summary_prompt_id"] == str(prompt.id)
    assert owner_payload["kind"] == "summary"
    assert owner_payload["processing_mode"] == "validated_strict_local"
    assert owner_payload["status"] == "processing"
    assert owner_payload["stage"] == "summary"
    assert owner_payload["progress"] is None
    assert owner_payload["failure"] is None
    assert owner_payload["cancel_requested"] is False
    assert owner_payload["created_at"] == job.created_at.isoformat().replace(
        "+00:00", "Z"
    )
    assert owner_payload["updated_at"]
    assert owner_payload["started_at"] is None
    assert owner_payload["completed_at"] is None
    assert owner_cancel.status_code == 200
    assert owner_cancel.json()["cancel_requested"] is True
    assert set(owner_cancel.json()) == {
        "id",
        "upload_id",
        "summary_prompt_id",
        "kind",
        "processing_mode",
        "status",
        "stage",
        "progress",
        "failure",
        "cancel_requested",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    }
    cancel_registered.assert_awaited_once_with(str(job.id))
    await db_session.refresh(job)
    assert job.cancel_requested is True


@pytest.mark.asyncio
async def test_retry_failed_summary_commits_before_waking_summary_runner(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary retries use their own durable claim and do not wake before commit."""
    headers, user_id = await auth_headers(client, email="summary-retry@example.com")
    patient = await create_test_patient(db_session, user_id)
    manifest, _contents = _manifest()
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(user_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="Locked policy",
        user_prompt="Grounded facts",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=0,
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        progress={"stage": "failed"},
        failure={"code": "local_worker_error", "retryable": True},
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    enqueue = Mock()
    monkeypatch.setattr(
        "app.services.local_ai.summary_runner.local_summary_runner.enqueue",
        enqueue,
    )
    response = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/retry",
        headers=headers,
    )
    repeat = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/retry",
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert repeat.status_code == 409
    assert response.json()["status"] == "queued"
    await db_session.refresh(job)
    assert (job.status, job.stage, job.failure) == ("queued", "queued", None)
    enqueue.assert_called_once_with(job.id)

    job.status = "failed"
    job.failure = {"code": "local_worker_error", "retryable": False}
    await db_session.commit()
    nonretryable = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/retry",
        headers=headers,
    )

    job.failure = {"code": "local_worker_error", "retryable": True}
    job.cancel_requested = True
    await db_session.commit()
    cancelled = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/retry",
        headers=headers,
    )

    job.cancel_requested = False
    job.status = "processing"
    await db_session.commit()
    active = await client.post(
        f"/api/v1/local-ai/jobs/{job.id}/retry",
        headers=headers,
    )

    assert nonretryable.status_code == 409
    assert cancelled.status_code == 409
    assert active.status_code == 409
    enqueue.assert_called_once_with(job.id)

    async def fail_audit(*_args, **_kwargs) -> None:
        raise RuntimeError("audit unavailable")

    job.status = "failed"
    job.failure = {"code": "local_worker_error", "retryable": True}
    await db_session.commit()
    monkeypatch.setattr("app.api.local_ai.log_audit_event", fail_audit)

    with pytest.raises(RuntimeError, match="audit unavailable"):
        await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/retry",
            headers=headers,
        )

    await db_session.refresh(job)
    assert (job.status, job.stage, job.failure) == ("queued", "queued", None)
    enqueue.assert_called_once_with(job.id)


def test_lifecycle_operations_take_owner_only_cross_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[int] = []

    class FakeFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        @staticmethod
        def flock(_descriptor: int, operation: int) -> None:
            events.append(operation)

    monkeypatch.setattr(
        pack_operations_module,
        "fcntl",
        FakeFcntl,
        raising=False,
    )
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))

    operation_store.create(action="install", manifest=manifest)

    assert events == [FakeFcntl.LOCK_EX, FakeFcntl.LOCK_UN]
    assert (
        operation_store.directory / ".lifecycle.lock"
    ).stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_pack_mutation_guard_locks_before_rechecking_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.local_ai as local_ai_module

    events: list[str] = []

    async def acquire(_db: object) -> None:
        events.append("lock")

    async def recheck(_db: object) -> None:
        events.append("recheck")

    monkeypatch.setattr(
        local_ai_module,
        "acquire_local_ai_lifecycle_lock",
        acquire,
    )
    monkeypatch.setattr(local_ai_module, "_require_no_active_jobs", recheck)

    await local_ai_module._acquire_pack_mutation_db_guard(object())

    assert events == ["lock", "recheck"]


def test_lifecycle_transition_is_compare_and_swap(tmp_path: Path) -> None:
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))
    operation = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Downloading verified model files.",
    )

    with pytest.raises(LocalValidationError, match="state changed"):
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state="paused",
            message="Model pack operation paused.",
            retryable=True,
        )

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "running"


def test_startup_reconciliation_completes_only_exact_validated_activation(
    tmp_path: Path,
) -> None:
    manifest, contents = _manifest()
    store = _install_pack(tmp_path, manifest, contents)
    operation_store = PackOperationStore(store)
    operation = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Running local validation fixtures.",
    )

    assert operation_store.reconcile_interrupted() == 1

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "completed"
    assert persisted["bytes_done"] == persisted["bytes_total"]
    assert persisted["retryable"] is False


@pytest.mark.parametrize(
    ("action", "initial_state", "retryable"),
    [
        ("install", "queued", True),
        ("verify", "paused", True),
        ("rollback", "running", False),
    ],
)
def test_startup_reconciliation_never_strands_nonterminal_operations(
    tmp_path: Path,
    action: str,
    initial_state: str,
    retryable: bool,
) -> None:
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))
    operation = operation_store.create(action=action, manifest=manifest)
    if initial_state != "queued":
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state=initial_state,
            message=(
                "Model pack operation paused."
                if initial_state == "paused"
                else "Restoring previous verified model pack."
            ),
            retryable=initial_state == "paused",
        )

    assert operation_store.reconcile_interrupted() == 1

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "failed"
    assert persisted["bytes_done"] == 0
    assert persisted["retryable"] is retryable


@pytest.fixture
def local_ai_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, contents = _manifest()
    manifest_path = tmp_path / "pack.lock.json"
    model_root = tmp_path / "models"
    _write_manifest(manifest_path, manifest)
    monkeypatch.setattr(settings, "local_ai_manifest_path", str(manifest_path))
    monkeypatch.setattr(settings, "local_ai_model_dir", str(model_root))
    monkeypatch.setattr(
        "app.api.local_ai._available_released_manifest", lambda: manifest
    )
    monkeypatch.setattr(
        "app.api.local_ai.platform_profile",
        lambda: ("apple_silicon", True),
    )
    return manifest, contents, model_root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("post", "/api/v1/local-ai/install"),
        (
            "get",
            "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001",
        ),
        (
            "post",
            "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001/resume",
        ),
        (
            "post",
            "/api/v1/local-ai/operations/00000000-0000-0000-0000-000000000001/retry",
        ),
        ("post", "/api/v1/local-ai/verify"),
        ("post", "/api/v1/local-ai/update"),
        ("post", "/api/v1/local-ai/rollback"),
        ("delete", "/api/v1/local-ai/models/summary"),
        ("delete", "/api/v1/local-ai"),
    ),
)
async def test_non_operator_cannot_access_any_global_pack_endpoint(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    method: str,
    path: str,
) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="ordinary-pack-user@example.com",
    )
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)

    response = await getattr(client, method)(path, headers=headers)

    assert response.status_code == 403
    assert response.json() == {
        "detail": "Local model pack management requires a machine operator."
    }


@pytest.mark.asyncio
async def test_global_pack_endpoint_preserves_401_without_credentials(client) -> None:
    response = await client.post("/api/v1/local-ai/install")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_status_is_readable_but_hides_operation_detail_from_non_operator(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, _contents, model_root = local_ai_paths
    headers, _user_id = await auth_headers(
        client,
        email="status-only-user@example.com",
    )
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))
    PackOperationStore(ArtifactStore(model_root)).create(
        action="install",
        manifest=manifest,
    )

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["can_manage_pack"] is False
    assert response.json()["operation"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_state", ("queued", "paused"))
async def test_non_operator_verify_operation_reports_truthful_verifying_state(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    operation_state: str,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _user_id = await auth_headers(
        client,
        email=f"status-verify-{operation_state}@example.com",
    )
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))
    store = _install_pack(model_root, manifest, contents)
    operation_store = PackOperationStore(store)
    operation = operation_store.create(action="verify", manifest=manifest)
    if operation_state == "paused":
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state="paused",
            message="Model pack operation paused.",
            retryable=True,
        )

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["can_manage_pack"] is False
    assert response.json()["operation"] is None
    assert response.json()["state"] == "verifying"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("install", "update"))
@pytest.mark.parametrize(
    ("operation_state", "message", "expected_pack_state"),
    (
        ("queued", None, "downloading"),
        ("running", "Running local validation fixtures.", "verifying"),
    ),
)
async def test_non_operator_install_update_status_distinguishes_lifecycle_phase(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    action: str,
    operation_state: str,
    message: str | None,
    expected_pack_state: str,
) -> None:
    manifest, _contents, model_root = local_ai_paths
    headers, _user_id = await auth_headers(
        client,
        email=f"status-{action}-{operation_state}@example.com",
    )
    monkeypatch.setattr(settings, "_local_ai_operator_ids", frozenset({uuid4()}))
    store = ArtifactStore(model_root)
    if action == "update":
        prior, prior_contents = _manifest("apple-m4-16gb-prior")
        store = _install_pack(model_root, prior, prior_contents)
    operation_store = PackOperationStore(store)
    operation = operation_store.create(action=action, manifest=manifest)
    if operation_state == "running":
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state="running",
            message=message,
        )

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["can_manage_pack"] is False
    assert response.json()["operation"] is None
    assert response.json()["state"] == expected_pack_state


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("install", "update", "verify"))
async def test_public_lifecycle_rejects_missing_release_evidence(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
    action: str,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, contents, model_root = local_ai_paths
    if action != "install":
        _install_pack(model_root, manifest, contents)
    monkeypatch.setattr(
        local_ai_module,
        "_available_released_manifest",
        lambda: (_ for _ in ()).throw(
            HTTPException(
                status_code=409, detail="A validated local model pack is not available."
            )
        ),
    )
    headers, _ = local_ai_operator_headers

    response = await client.post(f"/api/v1/local-ai/{action}", headers=headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "A validated local model pack is not available."


@pytest.mark.asyncio
async def test_install_returns_document_free_operation(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"operation_id", "state"}
    assert body["state"] == "queued"
    assert all(
        forbidden not in json.dumps(body).lower()
        for forbidden in ("upload", "document", "prompt", "excerpt", "patient")
    )


@pytest.mark.asyncio
async def test_operation_status_is_persisted_and_document_free(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]
    response = await client.get(
        f"/api/v1/local-ai/operations/{operation_id}",
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "id",
        "action",
        "state",
        "current_role",
        "bytes_done",
        "bytes_total",
        "message",
        "retryable",
    }
    assert body["action"] == "install"
    assert body["state"] == "queued"
    assert "pack" not in body
    assert "manifest" not in body


@pytest.mark.asyncio
async def test_second_lifecycle_mutation_is_refused_while_one_is_nonterminal(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    first = await client.post("/api/v1/local-ai/install", headers=headers)
    second = await client.post("/api/v1/local-ai/install", headers=headers)

    assert first.status_code == 202
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_resume_and_retry_enforce_operation_state(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]

    resume = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/resume",
        headers=headers,
    )
    retry = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )

    assert resume.status_code == 409
    assert retry.status_code == 409


@pytest.mark.asyncio
async def test_resume_and_retry_restart_from_zero_after_staging_cleanup(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    _manifest_value, _contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]
    operation_store = PackOperationStore(ArtifactStore(model_root))

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="paused",
        message="Model pack operation paused.",
        retryable=True,
        bytes_done=5,
    )
    resumed = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/resume",
        headers=headers,
    )
    assert resumed.status_code == 200
    assert resumed.json()["state"] == "queued"
    assert resumed.json()["bytes_done"] == 0

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )
    retried = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )
    assert retried.status_code == 200
    assert retried.json()["state"] == "queued"
    assert retried.json()["bytes_done"] == 0

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=False,
    )
    refused = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )
    assert refused.status_code == 409


@pytest.mark.asyncio
async def test_status_reflects_exact_validation_receipt_without_fake_memory(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    monkeypatch.setattr(settings, "local_ai_enabled", False)
    monkeypatch.setattr(
        settings,
        "local_ai_release_evidence_path",
        str(model_root.parent / "missing.release.json"),
    )
    headers, _ = local_ai_operator_headers
    _install_pack(model_root, manifest, contents)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["platform"] == "apple_silicon"
    assert body["compatible"] is True
    assert body["enabled"] is False
    assert body["can_manage_pack"] is True
    assert body["active_revision"] == manifest.pack_revision
    assert body["state"] == "preview"
    assert body["status_reason"] == "feature_disabled"
    assert body["available_revision"] == manifest.pack_revision
    assert len(body["models"]) == 3
    assert all(model["installed"] is True for model in body["models"])
    assert all(model["validated"] is False for model in body["models"])
    assert all(model["expected_memory_bytes"] is None for model in body["models"])
    assert [model["download_bytes"] for model in body["models"]] == [
        sum(file.size for file in artifact.files) for artifact in manifest.artifacts
    ]


@pytest.mark.asyncio
async def test_status_distinguishes_missing_release_evidence_when_enabled(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    monkeypatch.setattr(settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        settings,
        "local_ai_release_evidence_path",
        str(model_root.parent / "missing.release.json"),
    )
    headers, _ = await auth_headers(client)
    _install_pack(model_root, manifest, contents)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["state"] == "preview"
    assert body["status_reason"] == "release_evidence_missing"


@pytest.mark.asyncio
async def test_status_marks_a_different_installed_pack_as_update_available(
    client,
    local_ai_paths,
) -> None:
    _available, _contents, model_root = local_ai_paths
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    _install_pack(model_root, prior, prior_contents)
    headers, _ = await auth_headers(client)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["state"] == "update_available"


@pytest.mark.asyncio
async def test_verify_marks_only_a_successfully_verified_active_pack_ready(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers
    _install_pack(model_root, manifest, contents)

    async def verified(*_args, **_kwargs):
        return _receipt(manifest)

    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", verified)
    response = await client.post("/api/v1/local-ai/verify", headers=headers)
    assert response.status_code == 202

    status = await client.get("/api/v1/local-ai/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["state"] == "preview"
    assert all(not model["validated"] for model in status.json()["models"])


@pytest.mark.asyncio
async def test_install_validates_staging_before_atomic_activation(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers
    store = ArtifactStore(model_root)
    validation_saw_inactive_pack = False

    async def staged_download(
        selected_manifest,
        selected_store,
        progress_callback,
    ):
        assert selected_manifest == manifest
        staging = selected_store.stage(manifest.pack_revision)
        for artifact in manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
            progress_callback(
                {
                    "role": artifact.role.value,
                    "bytes_done": sum(file.size for file in artifact.files),
                    "bytes_total": sum(file.size for file in artifact.files),
                }
            )
        selected_store.verify(staging, manifest)
        return staging

    async def verify_candidate(selected_manifest, candidate_path):
        nonlocal validation_saw_inactive_pack
        assert selected_manifest == manifest
        assert candidate_path.parent == store.staging_dir
        validation_saw_inactive_pack = store.active_revision() is None
        return _receipt(manifest)

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", verify_candidate)

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert validation_saw_inactive_pack is True
    assert store.active_revision() == manifest.pack_revision
    assert PackOperationStore(store).is_validated(manifest)
    terminal_audit = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.action == "local_ai.operation.completed")
        )
    ).scalar_one()
    assert terminal_audit.details == {
        "action": "install",
        "operation_id": response.json()["operation_id"],
        "outcome": "completed",
        "pack_revision": manifest.pack_revision,
    }


@pytest.mark.asyncio
async def test_job_admitted_during_validation_prevents_pack_activation(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, user_id = local_ai_operator_headers
    store = ArtifactStore(model_root)

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        staging = selected_store.stage(selected_manifest.pack_revision)
        for artifact in selected_manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
        return staging

    async def admit_job_before_activation(_manifest, _candidate_path):
        upload = UploadedFile(
            user_id=user_id,
            filename="already-admitted.pdf",
            mime_type="application/pdf",
            file_hash="b" * 64,
            storage_path="/private/already-admitted.pdf",
            ingestion_status="pending_extraction",
            file_category="unstructured",
            processing_mode="validated_strict_local",
            processing_manifest=_manifest_dict(manifest),
            processing_schema_version="clinical-document-extraction.v1",
        )
        db_session.add(upload)
        await db_session.flush()
        db_session.add(
            LocalAIJob(
                user_id=user_id,
                upload_id=upload.id,
                kind="ingestion",
                processing_mode="validated_strict_local",
                manifest_snapshot=_manifest_dict(manifest),
                status="queued",
                stage="queued",
            )
        )
        await db_session.commit()
        return _receipt(manifest)

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr(
        "app.api.local_ai.verify_installed_pack",
        admit_job_before_activation,
    )

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() is None
    operation = PackOperationStore(store).get(response.json()["operation_id"])
    assert operation is not None
    assert operation["state"] == "failed"
    assert list(store.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_runtime_validation_never_activates_staged_pack(
    client,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers
    store = ArtifactStore(model_root)

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        assert selected_manifest == manifest
        staging = selected_store.stage(manifest.pack_revision)
        for artifact in manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
        selected_store.verify(staging, manifest)
        return staging

    async def reject_candidate(_selected_manifest, _candidate_path) -> None:
        raise RuntimeError("synthetic fixture failure")

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", reject_candidate)

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() is None
    assert list(store.staging_dir.iterdir()) == []
    status_response = await client.get("/api/v1/local-ai/status", headers=headers)
    assert status_response.json()["state"] == "failed"
    assert all(not model["validated"] for model in status_response.json()["models"])
    assert "synthetic fixture failure" not in caplog.text


@pytest.mark.asyncio
async def test_failed_update_validation_preserves_prior_verified_active_pointer(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    available, available_contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    store = _install_pack(model_root, prior, prior_contents)
    PackOperationStore(store).mark_validated(
        prior,
        _receipt(prior),
    )

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        assert selected_manifest == available
        staging = selected_store.stage(available.pack_revision)
        for artifact in available.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(available_contents[artifact.role])
        selected_store.verify(staging, available)
        return staging

    async def reject_candidate(_selected_manifest, _candidate_path) -> None:
        raise RuntimeError("synthetic fixture failure")

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", reject_candidate)

    response = await client.post("/api/v1/local-ai/update", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() == prior.pack_revision
    assert PackOperationStore(store).is_validated(prior)
    assert not (store.packs_dir / available.pack_revision).exists()
    assert list(store.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_download_persists_paused_retryable_operation(
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, _contents, model_root = local_ai_paths
    operation_store = PackOperationStore(ArtifactStore(model_root))
    operation = operation_store.create(action="install", manifest=manifest)

    async def cancel_download(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        cancel_download,
    )
    from app.api.local_ai import run_operation

    with pytest.raises(asyncio.CancelledError):
        await run_operation(operation["id"])

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "paused"
    assert persisted["message"] == "Model pack operation paused."
    assert persisted["retryable"] is True


@pytest.mark.asyncio
async def test_rollback_never_points_at_an_unvalidated_previous_pack(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    current, current_contents, model_root = local_ai_paths
    headers, _ = local_ai_operator_headers
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    store = _install_pack(model_root, prior, prior_contents)
    _install_pack(model_root, current, current_contents)
    store.remove_validation_receipt(prior)
    calls = 0
    original_rollback = ArtifactStore.rollback

    def observed_rollback(selected_store) -> None:
        nonlocal calls
        calls += 1
        original_rollback(selected_store)

    monkeypatch.setattr(ArtifactStore, "rollback", observed_rollback)

    response = await client.post("/api/v1/local-ai/rollback", headers=headers)

    assert response.status_code == 202
    assert calls == 0
    assert store.active_revision() == current.pack_revision
    operation = PackOperationStore(store).get(response.json()["operation_id"])
    assert operation is not None
    assert operation["state"] == "failed"


@pytest.mark.asyncio
async def test_remove_refuses_any_active_local_job(
    client,
    db_session,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, _contents, _model_root = local_ai_paths
    headers, user_id = local_ai_operator_headers
    upload = UploadedFile(
        user_id=user_id,
        filename="note.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/note.pdf",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="1",
    )
    db_session.add(upload)
    await db_session.flush()
    db_session.add(
        LocalAIJob(
            user_id=user_id,
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=_manifest_dict(manifest),
            status="queued",
            stage="preflight",
        )
    )
    await db_session.commit()

    response = await client.delete("/api/v1/local-ai", headers=headers)

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_operator_can_remove_one_model_role(
    client,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _user_id = local_ai_operator_headers
    store = _install_pack(model_root, manifest, contents)

    response = await client.delete(
        "/api/v1/local-ai/models/summary",
        headers=headers,
    )

    assert response.status_code == 204
    assert store.active_revision() is None
    assert not (store.packs_dir / manifest.pack_revision / "summary").exists()


@pytest.mark.asyncio
async def test_pack_audit_details_are_content_free(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    local_ai_operator_headers,
) -> None:
    headers, _ = local_ai_operator_headers

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    response = await client.post("/api/v1/local-ai/install", headers=headers)
    assert response.status_code == 202

    rows = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.action == "local_ai.install")
        )
    ).scalars()
    details = [row.details for row in rows]
    assert details
    serialized = json.dumps(details).lower()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "excerpt",
            "patient",
            "prompt",
            "document",
            "upload",
            "clinical",
        )
    )
