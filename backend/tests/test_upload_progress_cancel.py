"""Tests for extraction-progress batch scoping, cooperative cancel, and
section-level progress (remediation summary items 2a-iii and 2a-iv, backend).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.uploaded_file import UploadedFile
from tests.conftest import TEST_DB_URL, auth_headers


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
async def test_extraction_progress_scoped_to_ids(client: AsyncClient, db_session: AsyncSession):
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
        db_session.add(_mk_upload(user_id, status, record_count=1 if status == "completed" else 0))
    await db_session.commit()

    data = (await client.get("/api/v1/upload/extraction-progress", headers=headers)).json()
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
            f"/api/v1/upload/extraction-progress?ids={mine.id},{theirs.id}", headers=headers
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

    data = (await client.get("/api/v1/upload/extraction-progress", headers=headers)).json()
    assert data["total"] == 1
    assert data["processing"] == 0
    assert data["pending"] == 0
    assert data["completed"] == 1


# ---------------------------------------------------------------------------
# 2a-iv — cooperative cancel endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_sets_flag_on_active_files(client: AsyncClient, db_session: AsyncSession):
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

    with patch("app.api.upload._process_unstructured", new_callable=AsyncMock), patch(
        "app.api.upload.start_extraction_worker"
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
        with patch.object(upload_module, "async_session_factory", factory), patch(
            "app.services.extraction.text_extractor.extract_text", extract_text_mock
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
async def test_worker_writes_section_progress(db_session: AsyncSession):
    """A normal extraction run records progress_stage / progress_detail as it
    advances through the pipeline (verified mid-flight at the entity stage)."""
    from app.api import upload as upload_module
    from app.models.user import User
    from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult

    user = User(
        id=uuid4(),
        email="worker_progress_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    rtf_path = Path("/tmp") / f"progress_{uuid4().hex}.rtf"
    rtf_path.write_bytes(rb"{\rtf1\ansi Patient has hypertension. Plan: continue Lisinopril.}")

    upload = _mk_upload(user.id, "pending_extraction", storage_path=str(rtf_path))
    db_session.add(upload)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    seen_stage: dict[str, str | None] = {}

    async def fake_extract(text, source_file, api_key, progress_callback=None):
        # Capture the DB stage at the moment entity extraction runs.
        async with factory() as s:
            row = (
                await s.execute(select(UploadedFile).where(UploadedFile.id == upload.id))
            ).scalar_one()
            seen_stage["stage"] = row.progress_stage
        if progress_callback is not None:
            progress_callback("extracting_entities", 1, 1)
        return ExtractionResult(
            source_file=source_file,
            source_text=text,
            entities=[ExtractedEntity(entity_class="condition", text="Hypertension",
                                      attributes={"status": "active"})],
        )

    try:
        with patch.object(upload_module, "async_session_factory", factory), patch(
            "app.services.extraction.entity_extractor.extract_entities_async",
            side_effect=fake_extract,
        ), patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ):
            await upload_module._process_unstructured(upload.id, rtf_path, user.id)

        assert seen_stage.get("stage") == "extracting_entities"

        async with factory() as verify:
            row = (
                await verify.execute(select(UploadedFile).where(UploadedFile.id == upload.id))
            ).scalar_one()
            # Terminal-ish: handed off to dedup or completed, never failed.
            assert row.ingestion_status in ("dedup_scanning", "completed", "awaiting_review")
            assert row.progress_detail is not None
            assert row.progress_detail.get("section_total", 0) >= 1
    finally:
        await engine.dispose()
        rtf_path.unlink(missing_ok=True)
