"""TDD for frontend TODO C1: DELETE /upload/:id.

Soft-deletes the uploaded_file AND cascade soft-deletes the health_records it
produced (via source_file_id), writes an audit_log entry, all user-scoped.
Powers the Admin -> Uploads trash button.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from tests.conftest import auth_headers, create_test_patient


async def _make_upload(db, uid: UUID, filename="epic_export.zip") -> UploadedFile:
    up = UploadedFile(
        id=uuid4(),
        user_id=uid,
        filename=filename,
        mime_type="application/zip",
        file_hash=uuid4().hex,
        storage_path=f"/tmp/{uuid4().hex}",
        ingestion_status="completed",
        record_count=2,
    )
    db.add(up)
    return up


async def _make_record(db, uid, pid, source_file_id=None):
    rec = HealthRecord(
        id=uuid4(),
        patient_id=pid,
        user_id=uid,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition"},
        source_format="fhir_r4",
        display_text="x",
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        source_file_id=source_file_id,
    )
    db.add(rec)
    return rec


@pytest.mark.asyncio
async def test_delete_upload_requires_auth(client: AsyncClient):
    resp = await client.delete(f"/api/v1/upload/{uuid4()}")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_delete_nonexistent_upload_returns_404(client: AsyncClient, db_session: AsyncSession):
    headers, _ = await auth_headers(client)
    resp = await client.delete(f"/api/v1/upload/{uuid4()}", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_upload_cascades_to_its_records(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    uid_u, pid = UUID(uid), patient.id
    up = await _make_upload(db_session, uid_u)
    await db_session.flush()
    await _make_record(db_session, uid_u, pid, source_file_id=up.id)
    await _make_record(db_session, uid_u, pid, source_file_id=up.id)
    await _make_record(db_session, uid_u, pid, source_file_id=None)  # unrelated record survives
    await db_session.commit()

    resp = await client.delete(f"/api/v1/upload/{up.id}", headers=headers)
    assert resp.status_code == 204

    # the upload disappears from history
    hist = await client.get("/api/v1/upload/history", headers=headers)
    assert up.id not in {UUID(i["id"]) for i in hist.json()["items"]}

    # its records are soft-deleted; the unrelated record survives
    recs = await client.get("/api/v1/records", headers=headers)
    assert recs.json()["total"] == 1

    # an audit entry was written
    audit = await client.get("/api/v1/audit-log", headers=headers)
    assert any(i["action"] == "upload.delete" for i in audit.json()["items"])


@pytest.mark.asyncio
async def test_delete_upload_is_user_scoped(client: AsyncClient, db_session: AsyncSession):
    headers_a, uid_a = await auth_headers(client, email="a@example.com")
    _, uid_b = await auth_headers(client, email="b@example.com")
    up_b = await _make_upload(db_session, UUID(uid_b))
    await db_session.commit()

    # user A cannot delete user B's upload
    resp = await client.delete(f"/api/v1/upload/{up_b.id}", headers=headers_a)
    assert resp.status_code == 404
