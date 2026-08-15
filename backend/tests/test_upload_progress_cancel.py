"""Tests for extraction-progress batch scoping, cooperative cancel, and
section-level progress (remediation summary items 2a-iii and 2a-iv, backend).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.uploaded_file import UploadedFile
from app.models.local_ai import LocalAIJob
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from tests.conftest import TEST_DB_URL, auth_headers
from tests.test_strict_local_pipeline import _manifest_payload


def _mk_upload(user_id, status: str, **kw) -> UploadedFile:
    return UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename=kw.pop("filename", f"f_{uuid4().hex[:8]}.rtf"),
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash=f"hash_{uuid4().hex}",
        storage_path=kw.pop("storage_path", f"/tmp/{uuid4().hex}.rtf"),
        ingestion_status=status,
        file_category=kw.pop("file_category", "unstructured"),
        **kw,
    )


# ---------------------------------------------------------------------------
# 2a-iii — extraction-progress scoped to a batch via ?ids=
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_progress_scoped_to_ids(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)

    a = _mk_upload(user_id, "completed", record_count=3)
    b = _mk_upload(user_id, "processing")
    c = _mk_upload(user_id, "pending_extraction")
    for u in (a, b, c):
        db_session.add(u)
    await db_session.commit()

    # Scope to just a + b — c (pending) must be excluded.
    resp = await client.get(
        f"/api/v1/upload/extraction-progress?ids={a.id},{b.id}", headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["completed"] == 1
    assert data["processing"] == 1
    assert data["pending"] == 0
    assert data["records_created"] == 3


@pytest.mark.asyncio
async def test_extraction_progress_without_ids_counts_all(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)
    for status in ("completed", "processing", "pending_extraction"):
        db_session.add(
            _mk_upload(user_id, status, record_count=1 if status == "completed" else 0)
        )
    await db_session.commit()

    data = (
        await client.get("/api/v1/upload/extraction-progress", headers=headers)
    ).json()
    assert data["total"] == 3


@pytest.mark.asyncio
async def test_extraction_progress_ids_still_user_scoped(
    client: AsyncClient, db_session: AsyncSession
):
    """An id belonging to another user passed in ?ids= must not leak."""
    from app.models.user import User

    headers, user_id = await auth_headers(client)
    other = User(
        id=uuid4(),
        email="prog_other_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(other)
    await db_session.flush()

    mine = _mk_upload(user_id, "completed", record_count=2)
    theirs = _mk_upload(other.id, "completed", record_count=9)
    db_session.add(mine)
    db_session.add(theirs)
    await db_session.commit()

    data = (
        await client.get(
            f"/api/v1/upload/extraction-progress?ids={mine.id},{theirs.id}",
            headers=headers,
        )
    ).json()
    assert data["total"] == 1
    assert data["records_created"] == 2


@pytest.mark.asyncio
async def test_extraction_progress_counts_cancelled_as_terminal(
    client: AsyncClient, db_session: AsyncSession
):
    """A cancelled file is terminal/done — it must be in the total and counted
    toward the terminal (completed) bucket so the progress bar can reach 100%,
    never as pending/processing."""
    headers, user_id = await auth_headers(client)
    db_session.add(_mk_upload(user_id, "cancelled"))
    await db_session.commit()

    data = (
        await client.get("/api/v1/upload/extraction-progress", headers=headers)
    ).json()
    assert data["total"] == 1
    assert data["processing"] == 0
    assert data["pending"] == 0
    assert data["completed"] == 1


# ---------------------------------------------------------------------------
# 2a-iv — cooperative cancel endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_sets_flag_on_active_files(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)
    pending = _mk_upload(user_id, "pending_extraction")
    processing = _mk_upload(user_id, "processing")
    completed = _mk_upload(user_id, "completed")
    for u in (pending, processing, completed):
        db_session.add(u)
    await db_session.commit()

    resp = await client.post(
        "/api/v1/upload/cancel",
        json={"upload_ids": [str(pending.id), str(processing.id), str(completed.id)]},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["cancelled"]) == {str(pending.id), str(processing.id)}
    assert data["skipped"] == [str(completed.id)]

    for u in (pending, processing):
        await db_session.refresh(u)
        assert u.cancel_requested is True
    await db_session.refresh(completed)
    assert completed.cancel_requested is False


@pytest.mark.asyncio
async def test_cancel_terminates_active_strict_local_worker(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user_id,
        "processing",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user_id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="ocr",
    )
    db_session.add(job)
    await db_session.commit()

    events: list[str] = []
    original_commit = db_session.commit

    async def track_commit() -> None:
        await original_commit()
        events.append("commit")

    async def track_registered(job_id: str) -> bool:
        events.append("cancel_registered")
        return True

    db_session.commit = track_commit  # type: ignore[method-assign]

    with (
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
            new=AsyncMock(side_effect=track_registered),
            create=True,
        ) as cancel_registered,
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel",
            new_callable=AsyncMock,
        ) as cancel,
    ):
        response = await client.post(
            "/api/v1/upload/cancel",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert response.status_code == 200
    assert events[:2] == ["commit", "cancel_registered"]
    assert events.count("cancel_registered") == 1
    cancel_registered.assert_awaited_once_with(str(job.id))
    cancel.assert_not_awaited()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "processing"
    assert upload.cancel_requested is True
    assert job.status == "processing"
    assert job.cancel_requested is True


@pytest.mark.asyncio
async def test_cancel_finishes_queued_strict_job_without_reserving_worker_cancel(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user_id,
        "pending_extraction",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user_id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="queued",
        stage="preflight",
        progress={
            "stage": "extraction",
            "model_role": "extraction",
            "attempt": 2,
            "attempt_limit": 12,
            "output_tokens": 300,
            "output_token_limit": 16_384,
            "private": "must-not-survive-cancel",
        },
    )
    db_session.add(job)
    await db_session.commit()

    with (
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
            new_callable=AsyncMock,
            create=True,
        ) as cancel_registered,
        patch(
            "app.services.local_ai.model_manager.local_model_manager.cancel",
            new_callable=AsyncMock,
        ) as cancel,
    ):
        response = await client.post(
            "/api/v1/upload/cancel",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert response.status_code == 200
    cancel_registered.assert_not_awaited()
    cancel.assert_not_awaited()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.cancel_requested is True
    assert upload.ingestion_status == "cancelled"
    assert upload.progress_stage is None
    assert upload.progress_detail is None
    assert upload.processing_completed_at is not None
    assert job.status == "cancelled"
    assert job.stage == "cancelled"
    assert job.cancel_requested is True
    assert job.failure is None
    assert job.completed_at == upload.processing_completed_at
    assert job.progress == {
        "stage": "cancelled",
        "model_role": "extraction",
        "attempt": 2,
        "attempt_limit": 12,
        "output_tokens": 300,
        "output_token_limit": 16_384,
    }


@pytest.mark.asyncio
async def test_strict_cancel_check_reloads_flags_from_database(
    db_session: AsyncSession,
) -> None:
    from app.api.upload import _lock_strict_terminal_state
    from app.models.user import User

    user = User(
        email=f"strict-cancel-{uuid4().hex}@example.com",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.flush()
    user_id = user.id
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user_id,
        "processing",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user_id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="ocr",
    )
    db_session.add(job)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory() as other:
            other_upload = await other.get(UploadedFile, upload.id)
            other_job = await other.get(LocalAIJob, job.id)
            assert other_upload is not None and other_job is not None
            other_upload.cancel_requested = True
            other_job.cancel_requested = True
            await other.commit()
    finally:
        await engine.dispose()

    assert upload.cancel_requested is False
    assert job.cancel_requested is False
    assert await _lock_strict_terminal_state(db_session, upload.id, job.id) is True
    await db_session.rollback()


@pytest.mark.asyncio
async def test_strict_progress_uses_isolated_transaction_when_runner_session_is_poisoned(
    db_session: AsyncSession,
) -> None:
    from app.api import upload as upload_module
    from app.models.user import User

    user = User(
        email=f"strict-progress-isolated-{uuid4().hex}@example.com",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user.id,
        "processing",
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

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        with pytest.raises(DBAPIError):
            await db_session.execute(text("SELECT 1 / 0"))

        await upload_module._persist_strict_local_progress(
            runner_db=db_session,
            upload_id=upload.id,
            job_id=job.id,
            user_id=user.id,
            stage="extraction",
            progress={
                "stage": "extraction",
                "model_role": "extraction",
                "worker_current": 6,
                "worker_total": 54,
            },
        )

        async with factory() as verify:
            persisted_upload = await verify.get(UploadedFile, upload.id)
            persisted_job = await verify.get(LocalAIJob, job.id)
            assert persisted_upload is not None and persisted_job is not None
            assert persisted_upload.progress_stage == "local_extraction"
            assert persisted_upload.progress_detail == {
                "stage": "extraction",
                "model_role": "extraction",
                "worker_current": 6,
                "worker_total": 54,
            }
            assert persisted_job.stage == "extraction"
            assert persisted_job.progress == persisted_upload.progress_detail
    finally:
        await db_session.rollback()
        await engine.dispose()


@pytest.mark.asyncio
async def test_strict_progress_fails_closed_after_cancel_without_partial_write(
    db_session: AsyncSession,
) -> None:
    from app.api import upload as upload_module
    from app.models.user import User
    from app.services.local_ai.errors import LocalPolicyError

    user = User(
        email=f"strict-progress-cancel-{uuid4().hex}@example.com",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user.id,
        "processing",
        cancel_requested=True,
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
        progress={"stage": "ocr"},
    )
    db_session.add(job)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        with pytest.raises(LocalPolicyError, match="cancelled"):
            await upload_module._persist_strict_local_progress(
                runner_db=db_session,
                upload_id=upload.id,
                job_id=job.id,
                user_id=user.id,
                stage="extraction",
                progress={"stage": "extraction", "worker_current": 6},
            )

        async with factory() as verify:
            persisted_upload = await verify.get(UploadedFile, upload.id)
            persisted_job = await verify.get(LocalAIJob, job.id)
            assert persisted_upload is not None and persisted_job is not None
            assert persisted_upload.progress_stage is None
            assert persisted_upload.progress_detail is None
            assert persisted_job.stage == "ocr"
            assert persisted_job.progress == {"stage": "ocr"}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_completion_lock_linearizes_before_late_cancel(
    db_session: AsyncSession,
) -> None:
    from app.api.upload import (
        _lock_strict_terminal_state,
        cancel_extraction,
    )
    from app.models.user import User
    from app.schemas.upload import CancelExtractionRequest

    user = User(
        email=f"terminal-lock-{uuid4().hex}@example.com",
        password_hash="x",
    )
    db_session.add(user)
    await db_session.flush()
    user_id = user.id
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user_id,
        "processing",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user_id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="processing",
        stage="extraction",
    )
    db_session.add(job)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    assert not await _lock_strict_terminal_state(db_session, upload.id, job.id)
    try:

        async def late_cancel():
            async with factory() as cancel_db:
                return await cancel_extraction(
                    CancelExtractionRequest(upload_ids=[str(upload.id)]),
                    user_id,
                    cancel_db,
                )

        cancel_task = asyncio.create_task(late_cancel())
        await asyncio.sleep(0.05)
        assert not cancel_task.done()

        upload.ingestion_status = "completed"
        job.status = "completed"
        job.stage = "completed"
        await db_session.commit()

        response = await asyncio.wait_for(cancel_task, timeout=2)
        assert response.cancelled == []
        assert response.skipped == [str(upload.id)]
        await db_session.refresh(upload)
        await db_session.refresh(job)
        assert upload.cancel_requested is False
        assert job.cancel_requested is False
        assert upload.ingestion_status == "completed"
        assert job.status == "completed"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_cancel_skips_unknown_and_other_users(
    client: AsyncClient, db_session: AsyncSession
):
    from app.models.user import User

    headers, user_id = await auth_headers(client)
    other = User(
        id=uuid4(),
        email="cancel_other_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(other)
    await db_session.flush()
    theirs = _mk_upload(other.id, "processing")
    db_session.add(theirs)
    await db_session.commit()

    missing = str(uuid4())
    resp = await client.post(
        "/api/v1/upload/cancel",
        json={"upload_ids": [missing, str(theirs.id)]},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] == []
    assert set(data["skipped"]) == {missing, str(theirs.id)}
    # The other user's file must be untouched.
    await db_session.refresh(theirs)
    assert theirs.cancel_requested is False


@pytest.mark.asyncio
async def test_trigger_extraction_rejects_cancelled(
    client: AsyncClient, db_session: AsyncSession
):
    """A deliberately cancelled file must NOT be re-triggerable."""
    headers, user_id = await auth_headers(client)
    cancelled = _mk_upload(user_id, "cancelled")
    db_session.add(cancelled)
    await db_session.commit()

    with (
        patch("app.api.upload._process_unstructured", new_callable=AsyncMock),
        patch("app.api.upload.start_extraction_worker"),
    ):
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(cancelled.id)]},
            headers=headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 0
    assert data["failed"] == 1
    await db_session.refresh(cancelled)
    assert cancelled.ingestion_status == "cancelled"


# ---------------------------------------------------------------------------
# Section-level progress fields on the per-file status payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_payload_exposes_progress_fields(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)
    upload = _mk_upload(
        user_id,
        "processing",
        progress_stage="extracting_entities",
        progress_detail={"section_index": 3, "section_total": 8},
    )
    db_session.add(upload)
    await db_session.commit()

    resp = await client.get(f"/api/v1/upload/{upload.id}/status", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["progress_stage"] == "extracting_entities"
    assert data["progress_detail"] == {"section_index": 3, "section_total": 8}


@pytest.mark.asyncio
async def test_status_payload_exposes_typed_strict_local_failure(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    headers, user_id = await auth_headers(client)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user_id,
        "failed",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user_id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        status="failed",
        stage="extraction",
        failure={
            "stage": "extraction",
            "code": "local_worker_error",
            "message": "Strict-local processing did not complete.",
            "model_role": "extraction",
            "repository": "owner/extraction",
            "revision": "b" * 40,
            "retryable": False,
            "checkpoint_preserved": True,
            "cloud_fallback_attempted": False,
        },
    )
    db_session.add(job)
    await db_session.commit()

    response = await client.get(
        f"/api/v1/upload/{upload.id}/status",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["local_run"]["privacy_mode"] == "validated_strict_local"
    assert {model["role"] for model in payload["local_run"]["models"]} == {
        "ocr",
        "extraction",
    }
    assert all(
        model["repository"] != "mlx-community/Qwen3.5-9B-MLX-4bit"
        for model in payload["local_run"]["models"]
    )
    assert payload["local_failure"] == job.failure
    assert payload["local_failure"]["cloud_fallback_attempted"] is False
    assert payload["local_job_id"] == str(job.id)

    history = await client.get("/api/v1/upload/history", headers=headers)
    assert history.status_code == 200
    history_item = next(
        item for item in history.json()["items"] if item["id"] == str(upload.id)
    )
    assert history_item["local_job_id"] == str(job.id)
    assert history_item["local_failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_pending_extraction_list_exposes_progress_fields(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)
    upload = _mk_upload(
        user_id,
        "processing",
        progress_stage="scrubbing_phi",
        progress_detail={"section_index": 0, "section_total": 4},
    )
    db_session.add(upload)
    await db_session.commit()

    data = (
        await client.get(
            "/api/v1/upload/pending-extraction?statuses=processing", headers=headers
        )
    ).json()
    assert data["total"] == 1
    item = data["files"][0]
    assert item["progress_stage"] == "scrubbing_phi"
    assert item["progress_detail"] == {"section_index": 0, "section_total": 4}


# ---------------------------------------------------------------------------
# Worker cooperative-cancel abort (runs _process_unstructured against test DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_aborts_when_cancel_requested(db_session: AsyncSession):
    """When cancel_requested is set before the worker claims a file, the worker
    must mark it ``cancelled`` WITHOUT doing any extraction work."""
    from app.api import upload as upload_module
    from app.models.user import User

    user = User(
        id=uuid4(),
        email="worker_cancel_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    upload = _mk_upload(user.id, "pending_extraction", cancel_requested=True)
    db_session.add(upload)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    extract_text_mock = AsyncMock()
    try:
        with (
            patch.object(upload_module, "async_session_factory", factory),
            patch(
                "app.services.extraction.text_extractor.extract_text", extract_text_mock
            ),
        ):
            await upload_module._process_unstructured(
                upload.id, Path("/tmp/does-not-matter.rtf"), user.id
            )

        async with factory() as verify:
            row = (
                await verify.execute(
                    select(UploadedFile).where(UploadedFile.id == upload.id)
                )
            ).scalar_one()
            assert row.ingestion_status == "cancelled"
            assert row.processing_completed_at is not None
        extract_text_mock.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_strict_worker_early_cancel_terminalizes_upload_and_job(
    db_session: AsyncSession,
) -> None:
    from app.api import upload as upload_module
    from app.models.user import User

    user = User(
        id=uuid4(),
        email=f"strict-worker-cancel-{uuid4().hex}@example.com",
        password_hash="x",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = _mk_upload(
        user.id,
        "pending_extraction",
        cancel_requested=True,
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
        stage="preflight",
        progress={
            "stage": "extraction",
            "model_role": "extraction",
            "splits_used": 3,
            "split_limit": 7,
            "active_memory_bytes": 1_000,
            "peak_memory_bytes": 2_000,
            "private": "must-not-survive-worker-cancel",
        },
    )
    db_session.add(job)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        with patch.object(upload_module, "async_session_factory", factory):
            await upload_module._process_unstructured(
                upload.id,
                Path("/tmp/does-not-matter.rtf"),
                user.id,
            )

        async with factory() as verify:
            persisted_upload = await verify.get(UploadedFile, upload.id)
            persisted_job = await verify.get(LocalAIJob, job.id)
            assert persisted_upload is not None and persisted_job is not None
            assert persisted_upload.ingestion_status == "cancelled"
            assert persisted_job.status == "cancelled"
            assert persisted_job.stage == "cancelled"
            assert persisted_job.cancel_requested is True
            assert persisted_job.completed_at is not None
            assert persisted_job.progress == {
                "stage": "cancelled",
                "model_role": "extraction",
                "splits_used": 3,
                "split_limit": 7,
                "active_memory_bytes": 1_000,
                "peak_memory_bytes": 2_000,
            }
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_strict_cancel_fallback_terminalizes_every_active_job() -> None:
    """A poisoned first transaction cannot leave another active worker behind."""
    from types import SimpleNamespace

    from app.api.upload import _mark_cancelled

    user_id = uuid4()
    upload_id = uuid4()
    jobs = [
        SimpleNamespace(
            cancel_requested=False,
            status=status,
            stage="extraction",
            legacy_manifest_diagnostic=lambda: None,
            progress={
                "stage": "extraction",
                "model_role": "extraction",
                "attempt": index,
                "attempt_limit": 12,
                "private": f"private-{index}",
            },
            failure={"code": "old"},
            completed_at=None,
        )
        for index, status in ((1, "queued"), (2, "processing"))
    ]

    class Rows:
        def scalars(self):
            return self

        def all(self):
            return jobs

    execute_calls = 0

    async def execute(*_args, **_kwargs):
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 1:
            raise RuntimeError("force fallback")
        if execute_calls == 2:
            return None
        return Rows()

    fake_db = SimpleNamespace(
        execute=execute,
        rollback=AsyncMock(),
        commit=AsyncMock(),
    )
    upload = SimpleNamespace(
        id=upload_id,
        user_id=user_id,
        processing_mode="validated_strict_local",
    )

    await _mark_cancelled(fake_db, upload)

    fake_db.rollback.assert_awaited_once()
    fake_db.commit.assert_awaited_once()
    assert execute_calls == 3
    for index, job in enumerate(jobs, start=1):
        assert job.cancel_requested is True
        assert (job.status, job.stage) == ("cancelled", "cancelled")
        assert job.progress == {
            "stage": "cancelled",
            "model_role": "extraction",
            "attempt": index,
            "attempt_limit": 12,
        }
        assert job.failure is None
        assert job.completed_at is not None


@pytest.mark.asyncio
async def test_worker_writes_section_progress(db_session: AsyncSession):
    """A normal extraction run records progress_stage / progress_detail as it
    advances through the pipeline (verified mid-flight at the entity stage)."""
    from app.api import upload as upload_module
    from app.models.user import User
    from app.services.extraction.entity_extractor import (
        ExtractedEntity,
        ExtractionResult,
    )

    user = User(
        id=uuid4(),
        email="worker_progress_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    rtf_path = Path("/tmp") / f"progress_{uuid4().hex}.rtf"
    rtf_path.write_bytes(
        rb"{\rtf1\ansi Patient has hypertension. Plan: continue Lisinopril.}"
    )

    upload = _mk_upload(
        user.id,
        "pending_extraction",
        storage_path=str(rtf_path),
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    seen_stage: dict[str, str | None] = {}

    async def fake_extract(
        text, source_file, api_key, progress_callback=None, config=None
    ):
        # Capture the DB stage at the moment entity extraction runs.
        async with factory() as s:
            row = (
                await s.execute(
                    select(UploadedFile).where(UploadedFile.id == upload.id)
                )
            ).scalar_one()
            seen_stage["stage"] = row.progress_stage
        if progress_callback is not None:
            progress_callback("extracting_entities", 1, 1)
        return ExtractionResult(
            source_file=source_file,
            source_text=text,
            entities=[
                ExtractedEntity(
                    entity_class="condition",
                    text="Hypertension",
                    attributes={"status": "active"},
                )
            ],
        )

    try:
        with (
            patch.object(upload_module, "async_session_factory", factory),
            patch(
                "app.services.extraction.entity_extractor.extract_entities_async",
                side_effect=fake_extract,
            ),
            patch(
                "app.services.ingestion.coordinator._run_dedup_background",
                new_callable=AsyncMock,
            ),
        ):
            await upload_module._process_unstructured(upload.id, rtf_path, user.id)

        assert seen_stage.get("stage") == "extracting_entities"

        async with factory() as verify:
            row = (
                await verify.execute(
                    select(UploadedFile).where(UploadedFile.id == upload.id)
                )
            ).scalar_one()
            # Terminal-ish: handed off to dedup or completed, never failed.
            assert row.ingestion_status in (
                "dedup_scanning",
                "completed",
                "awaiting_review",
            )
            assert row.progress_detail is not None
            assert row.progress_detail.get("section_total", 0) >= 1
    finally:
        await engine.dispose()
        rtf_path.unlink(missing_ok=True)
