"""Slow fidelity test: real clinical-note PDF upload → extraction → re-upload idempotency.

Marks:
    slow    — requires GEMINI_API_KEY and a real note PDF under test_data/
    fidelity — real-data gate (same as CDA / Epic fidelity tests)

Skipped automatically when either the note PDF or the API key is absent.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from tests.conftest import (
    TEST_DB_URL,
    auth_headers,
    create_test_patient,
    private_fixture_root,
)

# ---------------------------------------------------------------------------
# Locate real note PDF via REAL_MEDICAL_FIXTURES_DIR (gitignored, off-repo).
# Originals live under <root>/raw/. No in-repo fallback.
# ---------------------------------------------------------------------------
_FIXROOT = private_fixture_root()
_RAW = (_FIXROOT / "raw") if _FIXROOT else None
_NOTE = next(iter(_RAW.glob("note_*.pdf")), None) if _RAW else None

_SKIP_REASON = (
    "REAL_MEDICAL_FIXTURES_DIR with a real note PDF (raw/note_*.pdf) and "
    "GEMINI_API_KEY are all required"
)
_SHOULD_SKIP = _NOTE is None or not settings.gemini_api_key


# ---------------------------------------------------------------------------
# Test-scoped session factory that points at the *test* database.
# Used to patch app.api.upload.async_session_factory so that
# _process_unstructured writes to the same DB the test fixtures use.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_session_factory():
    """Return a session factory targeting medtimeline_test."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _count_ai_records(factory: async_sessionmaker, user_id: uuid.UUID) -> int:  # type: ignore[type-arg]
    """Count live AI-extracted health records for a user via a fresh session."""
    from app.models.record import HealthRecord

    async with factory() as db:
        result = await db.execute(
            select(func.count())
            .select_from(HealthRecord)
            .where(
                HealthRecord.user_id == user_id,
                HealthRecord.ai_extracted.is_(True),
                HealthRecord.deleted_at.is_(None),
            )
        )
        return result.scalar() or 0


async def _set_upload_status(
    factory: async_sessionmaker,  # type: ignore[type-arg]
    upload_id: uuid.UUID,
    status: str,
) -> None:
    """Update an upload's ingestion_status via a fresh session."""
    from app.models.uploaded_file import UploadedFile

    async with factory() as db:
        await db.execute(
            update(UploadedFile)
            .where(UploadedFile.id == upload_id)
            .values(ingestion_status=status)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.fidelity
@pytest.mark.skipif(_SHOULD_SKIP, reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_real_note_reupload_is_duplicate(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
) -> None:
    """Upload real note PDF, run full extraction, then re-upload → duplicate_file; no new records.

    Flow:
    1. Register user + create Patient (required for auto-confirm path).
    2. POST /upload/unstructured with the real PDF → returns upload_id.
    3. Drive _process_unstructured synchronously (patched to use test DB).
    4. Manually set status → "completed" because _run_dedup_background is
       patched and never runs, leaving the upload at "dedup_scanning" which is
       NOT in PRODUCED_RECORDS_STATUSES.  Setting it to "completed" correctly
       represents the state that would exist after background dedup finished.
    5. Assert n1 > 0 (real note extracted at least one record).
    6. Re-upload identical bytes → status == "duplicate_file", ingestion_progress
       contains "duplicate_of".
    7. Assert n2 == n1 (no new records created).
    """
    from app.api.upload import _process_unstructured
    from app.models.uploaded_file import UploadedFile

    # ── Step 1: auth + patient ────────────────────────────────────────────
    headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)
    await create_test_patient(db_session, user_id)

    content: bytes = _NOTE.read_bytes()  # type: ignore[union-attr]  # skipif guards _NOTE is None

    # ── Step 2: first upload via HTTP API ─────────────────────────────────
    # Patch start_extraction_worker so the background polling loop doesn't start.
    with patch("app.api.upload.start_extraction_worker"):
        r1 = await client.post(
            "/api/v1/upload/unstructured",
            headers=headers,
            files={"file": (_NOTE.name, content, "application/pdf")},  # type: ignore[union-attr]
        )
    assert r1.status_code in (200, 202), f"First upload failed: {r1.text}"
    up1_id = uuid.UUID(r1.json()["upload_id"])

    # ── Step 3: drive extraction synchronously using the test DB ──────────
    # Patch async_session_factory used inside _process_unstructured so it
    # writes to the test DB instead of the production DB.
    # Also patch _run_dedup_background (called via asyncio.create_task inside
    # _process_unstructured) — the mock returns immediately so no background
    # task is left pending on the event loop.
    with (
        patch("app.api.upload.async_session_factory", test_session_factory),
        patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ),
    ):
        # Retrieve the storage_path from the test DB session (written by the API).
        up1 = await db_session.get(UploadedFile, up1_id)
        assert up1 is not None, "Upload record not found in test DB"
        await _process_unstructured(up1_id, Path(up1.storage_path), user_id)

    # Cancel any residual background tasks spawned by _process_unstructured
    # (e.g. a dedup task that slipped through) so pytest-asyncio can close cleanly.
    import asyncio as _asyncio

    current = _asyncio.current_task()
    for task in _asyncio.all_tasks():
        if task is not current and not task.done():
            task.cancel()
            try:
                await task
            except (_asyncio.CancelledError, Exception):
                pass

    # ── Step 4: inspect extraction result, then finalize status ─────────────
    # Check what status _process_unstructured left the upload in.  If it hit an
    # exception it will be "failed"; otherwise it should be "dedup_scanning" or
    # "awaiting_confirmation" (no patient) etc.
    async with test_session_factory() as _check_db:
        await _check_db.execute(
            # expire the identity-map entry so we get a fresh SELECT
            update(UploadedFile).where(UploadedFile.id == up1_id).values()
        )
        result = await _check_db.execute(
            select(UploadedFile).where(UploadedFile.id == up1_id)
        )
        up1_after = result.scalar_one_or_none()
        up1_status_after = up1_after.ingestion_status if up1_after else "NOT FOUND"
        up1_errors_after = up1_after.ingestion_errors if up1_after else []
        up1_entities = up1_after.extraction_entities if up1_after else None
        up1_text_len = len(up1_after.extracted_text or "") if up1_after else 0

    assert up1_status_after not in ("failed", "processing", "pending_extraction"), (
        f"Extraction failed or stalled: status='{up1_status_after}', "
        f"errors={up1_errors_after}, extracted_text_len={up1_text_len}, "
        f"entities_count={len(up1_entities) if up1_entities else 0}"
    )

    # _process_unstructured ends at "dedup_scanning" (dedup task was mocked).
    # find_prior_extracted_upload only matches PRODUCED_RECORDS_STATUSES
    # ("completed", "completed_with_merges", "awaiting_review",
    #  "awaiting_confirmation").  Simulate what the dedup background would do.
    await _set_upload_status(test_session_factory, up1_id, "completed")

    # ── Step 5: count extracted records ───────────────────────────────────
    n1 = await _count_ai_records(test_session_factory, user_id)
    assert n1 > 0, (
        "Real note should have produced at least one AI-extracted record. "
        "Check GEMINI_API_KEY validity and that the note PDF contains clinical text."
    )

    # ── Step 6: re-upload identical bytes ────────────────────────────────
    with patch("app.api.upload.start_extraction_worker"):
        r2 = await client.post(
            "/api/v1/upload/unstructured",
            headers=headers,
            files={"file": (_NOTE.name, content, "application/pdf")},  # type: ignore[union-attr]
        )
    assert r2.status_code in (200, 202), f"Re-upload failed: {r2.text}"
    body2 = r2.json()
    assert body2["status"] == "duplicate_file", (
        f"Expected status='duplicate_file', got '{body2['status']}'. "
        "Re-upload was not detected as a duplicate."
    )

    # Verify the duplicate metadata is present on the UploadedFile row.
    dup_record = await db_session.get(UploadedFile, uuid.UUID(body2["upload_id"]))
    assert dup_record is not None
    assert dup_record.ingestion_progress.get("duplicate_of") == str(up1_id), (
        f"ingestion_progress['duplicate_of'] should be {up1_id}, "
        f"got {dup_record.ingestion_progress}"
    )

    # ── Step 7: record count must not have grown ──────────────────────────
    n2 = await _count_ai_records(test_session_factory, user_id)
    assert n2 == n1, (
        f"Re-upload created new records: count before={n1}, after={n2}. "
        "Idempotency broken — re-upload should not trigger extraction."
    )
