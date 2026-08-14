"""Fast integration test: a DB INSERT failure during unstructured extraction
must leave the upload cleanly 'failed' with the REAL error — not stuck in
'processing' (which makes _recover_stuck_files retry 3x and finally mislabel it
as a generic TimeoutError).

Gemini is fully mocked, so this is a fast test (no GEMINI_API_KEY needed).
"""

from __future__ import annotations

import uuid
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.uploaded_file import UploadedFile
from app.models.local_ai import LocalAIJob, LocalAIPage
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult
from app.services.extraction.text_extractor import FileType
from tests.conftest import TEST_DB_URL, auth_headers
from tests.test_strict_local_pipeline import _manifest_payload


async def _queued_strict_pair(
    db_session: AsyncSession,
    *,
    user_id: uuid.UUID,
) -> tuple[UploadedFile, LocalAIJob]:
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename=f"claim-{uuid.uuid4().hex}.pdf",
        mime_type="application/pdf",
        file_size_bytes=512,
        file_hash=uuid.uuid4().hex,
        storage_path=f"/private/{uuid.uuid4().hex}.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
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
        stage="queued",
        progress={"stage": "queued", "attempt": 2, "attempt_limit": 12},
    )
    db_session.add(job)
    await db_session.flush()
    return upload, job


@pytest_asyncio.fixture
async def test_session_factory():
    """Session factory targeting medtimeline_test (what conftest truncates)."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _poison_record_dict(entity, user_id, patient_id, source_file_id=None):
    """Stand-in for entity_to_health_record_dict that yields a record Postgres
    rejects on INSERT: a non-datetime string in the timestamptz column. This
    reproduces the real-world failure (a bad extracted date reaching the DB)
    that poisons the session and made the original except-handler commit throw."""
    return {
        "id": uuid.uuid4(),
        "patient_id": patient_id,
        "user_id": user_id,
        "record_type": "condition",
        "fhir_resource_type": "Condition",
        "fhir_resource": {"resourceType": "Condition", "code": {"text": "x"}},
        "content_hash": "deadbeef",
        "source_format": "ai_extracted",
        "source_file_id": source_file_id,
        "effective_date": "this-is-not-a-timestamptz",  # asyncpg DataError on INSERT
        "status": "active",
        "category": ["condition"],
        "code_display": "x",
        "display_text": "x",
        "is_duplicate": False,
        "confidence_score": 0.8,
        "ai_extracted": True,
    }


@pytest.mark.asyncio
async def test_insert_failure_marks_failed_with_real_error_not_timeout(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
):
    from app.api.upload import _process_unstructured

    # ── Arrange: user + a pending unstructured upload row in the test DB ──
    headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)

    upload_id = uuid.uuid4()
    upload = UploadedFile(
        id=upload_id,
        user_id=user_id,
        filename="note.pdf",
        mime_type="application/pdf",
        file_size_bytes=1234,
        file_hash="hash-" + upload_id.hex,
        storage_path="/tmp/does-not-need-to-exist.pdf",
        ingestion_status="processing",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    one_entity = ExtractionResult(
        source_file="note.pdf",
        source_text="t",
        entities=[ExtractedEntity(entity_class="condition", text="hypertension")],
    )

    # ── Act: drive extraction with Gemini mocked + a poisoned record dict ──
    raised: Exception | None = None
    with (
        patch("app.api.upload.async_session_factory", test_session_factory),
        patch(
            "app.services.extraction.text_extractor.extract_text",
            new=AsyncMock(return_value=("Patient has hypertension.", "pdf")),
        ),
        patch(
            "app.services.extraction.text_extractor.detect_file_type",
            return_value=FileType.PDF,
        ),
        patch(
            "app.services.extraction.entity_extractor.extract_entities_async",
            new=AsyncMock(return_value=one_entity),
        ),
        patch(
            "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
            new=_poison_record_dict,
        ),
        patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ),
    ):
        try:
            await _process_unstructured(upload_id, Path(upload.storage_path), user_id)
        except Exception as exc:  # noqa: BLE001 — we assert on clean handling below
            raised = exc

    # ── Assert: handler recovered the session and recorded the real failure ──
    assert raised is None, (
        f"_process_unstructured should swallow the INSERT failure, but it "
        f"raised {type(raised).__name__}: {raised}"
    )

    async with test_session_factory() as check:
        row = (
            await check.execute(
                select(UploadedFile).where(UploadedFile.id == upload_id)
            )
        ).scalar_one()
        status = row.ingestion_status
        completed_at = row.processing_completed_at
        errors = row.ingestion_errors or []

    assert status == "failed", f"expected status 'failed', got '{status}'"
    assert completed_at is not None, "processing_completed_at must be set on failure"

    # The stored error must reflect the REAL DB failure, never the misleading
    # generic timeout that _recover_stuck_files would later stamp.
    error_types = {e.get("error_type") for e in errors}
    assert error_types, f"no error recorded: {errors}"
    assert "TimeoutError" not in error_types, (
        f"failure was mislabeled as a timeout: {errors}"
    )
    assert any(et and ("Error" in et or "Exception" in et) for et in error_types), (
        f"expected a real DB error type, got {error_types}"
    )


@pytest.mark.asyncio
async def test_stuck_strict_job_requeues_without_losing_page_checkpoints(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _recover_stuck_files

    _headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename="recover.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/recover.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        retry_count=0,
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
        updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    db_session.add(job)
    await db_session.flush()
    page = LocalAIPage(
        job_id=job.id,
        page_number=1,
        checkpoint_key="1" * 64,
        image_sha256="2" * 64,
        ocr_result={"markdown": "Private checkpoint", "width": 10, "height": 10},
        warnings=[],
    )
    db_session.add(page)
    await db_session.commit()

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    monkeypatch.setattr("app.api.upload.settings.extraction_timeout_minutes", 1)
    monkeypatch.setattr("app.api.upload.settings.extraction_max_retries", 3)

    await _recover_stuck_files()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "pending_extraction"
    assert job.status == "queued"
    assert job.stage == "recovery"
    assert await db_session.get(LocalAIPage, page.id) is not None


@pytest.mark.asyncio
async def test_live_strict_job_uses_fresh_job_heartbeat_not_generic_upload_age(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import (
        _recover_stuck_files,
        _refresh_strict_local_job_lease,
    )

    _headers, uid_str = await auth_headers(
        client,
        email="live-strict-heartbeat@example.com",
    )
    user_id = uuid.UUID(uid_str)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename="live.pdf",
        mime_type="application/pdf",
        file_hash="c" * 64,
        storage_path="/private/live.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        retry_count=0,
        progress_stage="local_extraction",
        progress_detail={
            "stage": "extraction",
            "model_role": "extraction",
            "worker_current": 0,
            "worker_total": 4,
        },
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
        progress={
            "stage": "extraction",
            "model_role": "extraction",
            "worker_current": 0,
            "worker_total": 4,
        },
        updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    db_session.add(job)
    await db_session.commit()

    public_job_progress = dict(job.progress)
    public_upload_progress = dict(upload.progress_detail)
    await _refresh_strict_local_job_lease(
        db_session,
        upload_id=upload.id,
        job_id=job.id,
        user_id=user_id,
    )
    await db_session.refresh(job)
    assert job.updated_at > datetime.now(timezone.utc) - timedelta(minutes=1)
    assert job.progress == public_job_progress
    assert upload.progress_detail == public_upload_progress

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    monkeypatch.setattr("app.api.upload.settings.extraction_timeout_minutes", 1)
    monkeypatch.setattr("app.api.upload.settings.local_ai_worker_timeout_seconds", 5)

    await _recover_stuck_files()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "processing"
    assert upload.retry_count == 0
    assert job.status == "processing"
    assert job.stage == "extraction"


@pytest.mark.asyncio
async def test_claim_pending_files_can_exclude_strict_jobs_waiting_for_model_slot(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _claim_pending_files

    _headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    strict_upload = UploadedFile(
        user_id=user_id,
        filename="queued-strict.pdf",
        mime_type="application/pdf",
        file_hash="8" * 64,
        storage_path="/private/queued-strict.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="validated_strict_local",
    )
    non_strict_upload = UploadedFile(
        user_id=user_id,
        filename="queued-custom.rtf",
        mime_type="application/rtf",
        file_hash="9" * 64,
        storage_path="/private/queued-custom.rtf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    manual_zip_child = UploadedFile(
        user_id=user_id,
        filename="manual-child.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/manual-child.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="cloud_assisted",
        manual_extraction_required=True,
    )
    db_session.add_all([strict_upload, non_strict_upload, manual_zip_child])
    await db_session.commit()

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    claimed = await _claim_pending_files(1)

    assert claimed == [
        (
            str(non_strict_upload.id),
            non_strict_upload.storage_path,
            str(user_id),
            "cloud_assisted",
        )
    ]
    await db_session.refresh(strict_upload)
    assert strict_upload.ingestion_status == "pending_extraction"
    await db_session.refresh(manual_zip_child)
    assert manual_zip_child.ingestion_status == "pending_extraction"


@pytest.mark.asyncio
async def test_claim_next_strict_ingestion_pair_advances_both_rows_atomically(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
) -> None:
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
    )

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    await db_session.commit()

    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert claim.upload_id == upload.id
    assert claim.job_id == job.id
    assert claim.user_id == upload.user_id
    assert claim.storage_path == upload.storage_path
    assert upload.ingestion_status == "processing"
    assert upload.processing_started_at == claim.claimed_at
    assert upload.progress_stage == "local_preflight"
    assert job.status == "processing"
    assert job.stage == "preflight"
    assert job.started_at == claim.claimed_at


@pytest.mark.asyncio
async def test_committed_strict_claim_enters_pipeline_with_original_timestamp(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.api.upload import _run_strict_local_ingestion_for_upload
    from app.services.local_ai.errors import LocalPolicyError
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
    )
    from app.services.local_ai.manifest import parse_manifest

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    await db_session.commit()
    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    manifest = parse_manifest(upload.processing_manifest)
    monkeypatch.setattr(
        "app.api.upload.settings.local_ai_model_dir",
        str(tmp_path / "models"),
    )
    monkeypatch.setattr(
        "app.services.local_ai.artifact_store.ArtifactStore.active_manifest",
        lambda _self: manifest,
    )
    entered: list[tuple[uuid.UUID, datetime | None, datetime | None]] = []

    async def stop_after_entry(_self, current_job, current_upload, _path):
        entered.append(
            (
                current_job.id,
                current_job.started_at,
                current_upload.processing_started_at,
            )
        )
        raise LocalPolicyError("synthetic pipeline stop")

    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        stop_after_entry,
    )

    async with test_session_factory() as runner_db:
        claimed_upload = await runner_db.get(UploadedFile, upload.id)
        assert claimed_upload is not None
        with pytest.raises(LocalPolicyError, match="synthetic pipeline stop"):
            await _run_strict_local_ingestion_for_upload(
                runner_db,
                claimed_upload,
                Path(claim.storage_path),
                claim.user_id,
                strict_claim=claim,
            )

    assert entered == [(job.id, claim.claimed_at, claim.claimed_at)]
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.processing_started_at == claim.claimed_at
    assert job.started_at == claim.claimed_at


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_field", ("job_id", "claimed_at", "file_path"))
async def test_stale_strict_claim_returns_without_pipeline_or_mutation(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stale_field: str,
) -> None:
    from app.api.upload import _run_strict_local_ingestion_for_upload
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
    )

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    await db_session.commit()
    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    stale_claim = claim
    if stale_field == "job_id":
        stale_claim = replace(claim, job_id=uuid.uuid4())
    elif stale_field == "claimed_at":
        stale_claim = replace(claim, claimed_at=claim.claimed_at + timedelta(seconds=1))
    runner_path = (
        tmp_path / "wrong-claim-path.pdf"
        if stale_field == "file_path"
        else Path(claim.storage_path)
    )
    monkeypatch.setattr(
        "app.api.upload.settings.local_ai_model_dir",
        str(tmp_path / "models"),
    )
    pipeline_run = AsyncMock()
    model_run = AsyncMock()
    monkeypatch.setattr(
        "app.services.local_ai.pipeline.StrictLocalPipeline.run_ingestion",
        pipeline_run,
    )
    monkeypatch.setattr(
        "app.services.local_ai.model_manager.local_model_manager.run",
        model_run,
    )

    async with test_session_factory() as runner_db:
        claimed_upload = await runner_db.get(UploadedFile, upload.id)
        assert claimed_upload is not None
        await _run_strict_local_ingestion_for_upload(
            runner_db,
            claimed_upload,
            runner_path,
            claim.user_id,
            strict_claim=stale_claim,
        )

    pipeline_run.assert_not_awaited()
    model_run.assert_not_awaited()
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.processing_started_at) == (
        "processing",
        claim.claimed_at,
    )
    assert (job.status, job.started_at) == ("processing", claim.claimed_at)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("create_task", "add_done_callback"))
async def test_failed_strict_task_scheduling_requeues_exact_claim_and_releases_slots_once(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from app.api.upload import _schedule_claimed_extraction
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
    )

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    checkpoint = LocalAIPage(
        job_id=job.id,
        page_number=1,
        checkpoint_key="1" * 64,
        image_sha256="2" * 64,
        ocr_result={"markdown": "Synthetic checkpoint", "width": 10, "height": 10},
        warnings=[],
    )
    db_session.add(checkpoint)
    await db_session.commit()

    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)

    child_started = asyncio.Event()
    child_terminated = asyncio.Event()
    child_task: asyncio.Task[None] | None = None

    async def controlled_child(*_args, release_slots, **_kwargs) -> None:
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            release_slots()
            child_terminated.set()

    monkeypatch.setattr("app.api.upload._process_and_release", controlled_child)

    class CallbackFailureTask(asyncio.Task[None]):
        def add_done_callback(self, _callback, *, context=None) -> None:
            raise RuntimeError("synthetic persistent callback failure")

    def task_factory(coroutine):
        nonlocal child_task
        if failure_point == "create_task":
            coroutine.close()
            raise RuntimeError("synthetic create_task failure")
        child_task = CallbackFailureTask(coroutine)
        return child_task

    if failure_point == "add_done_callback":
        from app.services.local_ai import ingestion_lifecycle

        original_compensate = (
            ingestion_lifecycle.compensate_unstarted_strict_ingestion_claim
        )

        async def compensate_after_child_termination(db, current_claim):
            assert child_started.is_set()
            assert child_terminated.is_set()
            assert child_task is not None and child_task.done()
            return await original_compensate(db, current_claim)

        monkeypatch.setattr(
            ingestion_lifecycle,
            "compensate_unstarted_strict_ingestion_claim",
            compensate_after_child_termination,
        )

    general_slot = asyncio.Semaphore(1)
    strict_slot = asyncio.Semaphore(1)
    await general_slot.acquire()
    await strict_slot.acquire()

    scheduled = await _schedule_claimed_extraction(
        general_slot,
        claim,
        strict_sem=strict_slot,
        create_task_fn=task_factory,
    )

    assert scheduled is False
    if failure_point == "add_done_callback":
        assert child_started.is_set()
        assert child_terminated.is_set()
        assert child_task is not None and child_task.cancelled()
    await asyncio.wait_for(general_slot.acquire(), timeout=0.2)
    await asyncio.wait_for(strict_slot.acquire(), timeout=0.2)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(general_slot.acquire(), timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(strict_slot.acquire(), timeout=0.05)

    await db_session.refresh(upload)
    await db_session.refresh(job)
    await db_session.refresh(checkpoint)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.processing_started_at is None
    assert upload.progress_stage is None
    assert job.status == "queued"
    assert job.stage == "queued"
    assert job.started_at is None
    assert job.progress == {"stage": "queued", "attempt": 2, "attempt_limit": 12}
    assert checkpoint.job_id == job.id


@pytest.mark.asyncio
async def test_persistent_strict_scheduling_failures_back_off_before_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.upload as upload_module
    from app.services.local_ai.ingestion_lifecycle import StrictIngestionClaim

    claim = StrictIngestionClaim(
        upload_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        storage_path="/private/synthetic-claim.pdf",
        claimed_at=datetime.now(timezone.utc),
    )
    general_slot = asyncio.Semaphore(1)
    strict_slot = asyncio.Semaphore(1)
    scheduling_attempts = 0
    backoffs: list[float] = []

    class ImmediateReclaim(BaseException):
        pass

    class ClaimSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def commit(self) -> None:
            return None

        async def rollback(self) -> None:
            return None

    async def claim_pair(_db):
        return claim

    async def fail_scheduling(sem, _claim, *, strict_sem):
        nonlocal scheduling_attempts
        scheduling_attempts += 1
        if scheduling_attempts > len(backoffs) + 1:
            raise ImmediateReclaim
        strict_sem.release()
        sem.release()
        return False

    async def record_backoff(delay: float) -> None:
        backoffs.append(delay)
        if len(backoffs) == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(upload_module, "_get_extraction_semaphore", lambda: general_slot)
    monkeypatch.setattr(
        upload_module,
        "_get_strict_extraction_semaphore",
        lambda: strict_slot,
    )
    monkeypatch.setattr(upload_module, "async_session_factory", ClaimSession)
    monkeypatch.setattr(
        "app.services.local_ai.ingestion_lifecycle.claim_next_strict_ingestion_pair",
        claim_pair,
    )
    monkeypatch.setattr(upload_module, "_schedule_claimed_extraction", fail_scheduling)
    monkeypatch.setattr(upload_module.asyncio, "sleep", record_backoff)

    with pytest.raises(asyncio.CancelledError):
        await upload_module._extraction_worker()

    assert scheduling_attempts == 3
    assert backoffs == [2, 2, 2]


@pytest.mark.asyncio
async def test_callback_failure_keeps_live_child_tracked_for_shutdown(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.upload as upload_module
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
    )

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    await db_session.commit()
    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    allow_exit = asyncio.Event()
    cancellation_seen = asyncio.Event()
    child_terminated = asyncio.Event()
    child_task: asyncio.Task[None] | None = None

    async def stubborn_child(*_args, release_slots, **_kwargs) -> None:
        try:
            while not allow_exit.is_set():
                try:
                    await allow_exit.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
        finally:
            release_slots()
            child_terminated.set()

    monkeypatch.setattr("app.api.upload._process_and_release", stubborn_child)

    class PersistentCallbackFailureTask(asyncio.Task[None]):
        def add_done_callback(self, callback, *, context=None) -> None:
            raise RuntimeError("synthetic persistent callback failure")

    def task_factory(coroutine):
        nonlocal child_task
        child_task = PersistentCallbackFailureTask(coroutine)
        return child_task

    compensate = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.services.local_ai.ingestion_lifecycle.compensate_unstarted_strict_ingestion_claim",
        compensate,
    )
    general_slot = asyncio.Semaphore(1)
    strict_slot = asyncio.Semaphore(1)
    await general_slot.acquire()
    await strict_slot.acquire()

    scheduled = await upload_module._schedule_claimed_extraction(
        general_slot,
        claim,
        strict_sem=strict_slot,
        create_task_fn=task_factory,
        callback_drain_timeout_seconds=0.01,
    )

    assert scheduled is False
    assert cancellation_seen.is_set()
    assert child_task is not None and not child_task.done()
    assert child_task in upload_module._extraction_tasks
    compensate.assert_not_awaited()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(general_slot.acquire(), timeout=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(strict_slot.acquire(), timeout=0.05)
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.processing_started_at) == (
        "processing",
        claim.claimed_at,
    )
    assert (job.status, job.started_at) == ("processing", claim.claimed_at)

    monkeypatch.setattr(
        upload_module.settings,
        "local_ai_shutdown_drain_seconds",
        0.01,
    )
    await upload_module.stop_extraction_worker()
    assert not child_terminated.is_set()
    assert not child_task.done()
    assert child_task in upload_module._extraction_tasks
    compensate.assert_not_awaited()

    allow_exit.set()
    child_task.cancel()
    while not child_task.done():
        await asyncio.sleep(0)
    try:
        child_task.result()
    except asyncio.CancelledError:
        pass
    upload_module._extraction_tasks.discard(child_task)
    assert child_terminated.is_set()
    assert child_task.done()
    await asyncio.wait_for(general_slot.acquire(), timeout=0.2)
    await asyncio.wait_for(strict_slot.acquire(), timeout=0.2)


@pytest.mark.asyncio
async def test_compensation_rejects_stale_claim_path_without_mutation(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
) -> None:
    from app.services.local_ai.ingestion_lifecycle import (
        claim_next_strict_ingestion_pair,
        compensate_unstarted_strict_ingestion_claim,
    )

    _headers, uid_str = await auth_headers(client)
    upload, job = await _queued_strict_pair(
        db_session,
        user_id=uuid.UUID(uid_str),
    )
    await db_session.commit()
    async with test_session_factory() as claim_db:
        claim = await claim_next_strict_ingestion_pair(claim_db)
        assert claim is not None
        await claim_db.commit()

    stale_claim = replace(claim, storage_path="/private/stale-claim.pdf")
    async with test_session_factory() as compensation_db:
        assert not await compensate_unstarted_strict_ingestion_claim(
            compensation_db,
            stale_claim,
        )
        await compensation_db.rollback()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.processing_started_at) == (
        "processing",
        claim.claimed_at,
    )
    assert (job.status, job.started_at) == ("processing", claim.claimed_at)


@pytest.mark.asyncio
async def test_claim_cancel_race_never_exposes_a_split_pair(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
) -> None:
    from app.services.local_ai.ingestion_lifecycle import (
        cancel_strict_ingestion_pair,
        claim_next_strict_ingestion_pair,
        compensate_unstarted_strict_ingestion_claim,
    )

    _headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    upload, job = await _queued_strict_pair(db_session, user_id=user_id)
    await db_session.commit()

    upload_locked = asyncio.Event()
    release_job_lock = asyncio.Event()
    async with test_session_factory() as claim_db, test_session_factory() as cancel_db:
        original_execute = claim_db.execute

        async def pause_before_job_lock(statement, *args, **kwargs):
            if "FROM local_ai_jobs" in str(statement):
                upload_locked.set()
                await release_job_lock.wait()
            return await original_execute(statement, *args, **kwargs)

        claim_db.execute = pause_before_job_lock  # type: ignore[method-assign]

        async def claim_and_commit():
            claimed = await claim_next_strict_ingestion_pair(claim_db)
            await claim_db.commit()
            return claimed

        async def cancel_and_commit():
            cancelled = await cancel_strict_ingestion_pair(
                cancel_db,
                user_id=user_id,
                upload_id=upload.id,
                expected_job_id=job.id,
            )
            await cancel_db.commit()
            return cancelled

        claim_task = asyncio.create_task(claim_and_commit())
        await asyncio.wait_for(upload_locked.wait(), timeout=1)
        cancel_task = asyncio.create_task(cancel_and_commit())
        await asyncio.sleep(0.05)
        assert cancel_task.done() is False
        release_job_lock.set()
        claim = await asyncio.wait_for(claim_task, timeout=2)
        cancellation = await asyncio.wait_for(cancel_task, timeout=2)

    assert claim is not None
    assert cancellation.worker_cancel_required is True
    async with test_session_factory() as compensation_db:
        assert await compensate_unstarted_strict_ingestion_claim(
            compensation_db,
            claim,
        )
        await compensation_db.commit()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert (upload.ingestion_status, upload.cancel_requested) == ("cancelled", True)
    assert (job.status, job.cancel_requested) == ("cancelled", True)
    assert job.completed_at == upload.processing_completed_at


@pytest.mark.asyncio
async def test_stuck_strict_upload_at_max_retries_fails_queued_job_too(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _recover_stuck_files

    _headers, uid_str = await auth_headers(
        client,
        email="queued-strict-timeout@example.com",
    )
    user_id = uuid.UUID(uid_str)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename="queued-timeout.pdf",
        mime_type="application/pdf",
        file_hash="e" * 64,
        storage_path="/private/queued-timeout.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        retry_count=3,
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
        stage="queued",
    )
    db_session.add(job)
    await db_session.commit()

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    monkeypatch.setattr("app.api.upload.settings.extraction_timeout_minutes", 1)
    monkeypatch.setattr("app.api.upload.settings.extraction_max_retries", 3)

    await _recover_stuck_files()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "failed"
    assert upload.processing_completed_at is not None
    assert job.status == "failed"
    assert job.stage == "failed"
    assert job.completed_at == upload.processing_completed_at
    assert job.failure["code"] == "local_worker_timeout"


@pytest.mark.asyncio
async def test_stuck_cancelled_strict_job_terminalizes_both_rows(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.upload import _recover_stuck_files

    _headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename="recover-cancelled.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/recover-cancelled.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        cancel_requested=True,
        retry_count=0,
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

    monkeypatch.setattr("app.api.upload.async_session_factory", test_session_factory)
    monkeypatch.setattr("app.api.upload.settings.extraction_timeout_minutes", 1)
    monkeypatch.setattr("app.api.upload.settings.extraction_max_retries", 3)

    await _recover_stuck_files()

    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "cancelled"
    assert upload.processing_completed_at is not None
    assert job.status == "cancelled"
    assert job.stage == "cancelled"
    assert job.cancel_requested is True
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_startup_recovery_terminalizes_cancelled_strict_pair(
    db_session: AsyncSession,
    client: AsyncClient,
) -> None:
    from app.main import _recover_unstructured_jobs_on_startup

    _headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
    upload = UploadedFile(
        user_id=user_id,
        filename="startup-cancelled.pdf",
        mime_type="application/pdf",
        file_hash="d" * 64,
        storage_path="/private/startup-cancelled.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=snapshot,
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        cancel_requested=True,
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
        stage="recovery",
    )
    db_session.add(job)
    await db_session.commit()

    await _recover_unstructured_jobs_on_startup(db_session)
    await db_session.commit()
    await db_session.refresh(upload)
    await db_session.refresh(job)

    assert upload.ingestion_status == "cancelled"
    assert upload.processing_completed_at is not None
    assert job.status == "cancelled"
    assert job.stage == "cancelled"
    assert job.cancel_requested is True
    assert job.completed_at is not None
