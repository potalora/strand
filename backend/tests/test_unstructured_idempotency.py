from __future__ import annotations

import hashlib
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.uploaded_file import UploadedFile
from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_reupload_identical_file_is_marked_duplicate(
    client: AsyncClient, db_session: AsyncSession
):
    """Same file_hash already completed for the user -> new upload is duplicate_file, not extracted."""
    headers, uid = await auth_headers(client)

    content = b"%PDF-1.4 fake pdf bytes for hashing test"
    h = hashlib.sha256(content).hexdigest()

    # Seed a prior upload in a produced-records status so the idempotency guard triggers.
    prior = UploadedFile(
        id=uuid.uuid4(),
        user_id=uid,
        filename="note.pdf",
        file_hash=h,
        mime_type="application/pdf",
        file_size_bytes=len(content),
        storage_path="/x",
        ingestion_status="completed",
        file_category="unstructured",
        record_count=7,
    )
    db_session.add(prior)
    await db_session.commit()

    resp = await client.post(
        "/api/v1/upload/unstructured",
        headers=headers,
        files={"file": ("note.pdf", content, "application/pdf")},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "duplicate_file"

    rows = (
        await db_session.execute(
            select(UploadedFile).where(
                UploadedFile.file_hash == h,
                UploadedFile.id != prior.id,
            )
        )
    ).scalars().all()
    assert len(rows) == 1
    dup = rows[0]
    assert dup.ingestion_status == "duplicate_file"
    assert dup.ingestion_progress["duplicate_of"] == str(prior.id)
    assert dup.ingestion_progress["record_count"] == 7
