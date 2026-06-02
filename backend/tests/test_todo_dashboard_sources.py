"""TDD for frontend TODO B1: GET /dashboard/sources — provenance breakdown.

Aggregates the user's health_records by source_format into {items:[{source,count}], total}
(total = number of distinct sources). Powers the Overview "Where records come from"
bars and the "Data sources" stat. Excludes soft-deleted + duplicate records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient


async def _add_record(db, uid, pid, source_format, *, deleted=False, duplicate=False):
    db.add(
        HealthRecord(
            id=uuid4(),
            patient_id=pid,
            user_id=uid,
            record_type="observation",
            fhir_resource_type="Observation",
            fhir_resource={"resourceType": "Observation"},
            source_format=source_format,
            display_text="x",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            deleted_at=datetime.now(timezone.utc) if deleted else None,
            is_duplicate=duplicate,
        )
    )


@pytest.mark.asyncio
async def test_sources_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/dashboard/sources")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_sources_aggregates_by_source_format_desc(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    uid_u, pid = UUID(uid), patient.id
    for _ in range(3):
        await _add_record(db_session, uid_u, pid, "fhir_r4")
    for _ in range(2):
        await _add_record(db_session, uid_u, pid, "Epic EHI")
    await _add_record(db_session, uid_u, pid, "CDA XML")
    await db_session.commit()

    resp = await client.get("/api/v1/dashboard/sources", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3  # distinct sources
    assert data["items"][0] == {"source": "fhir_r4", "count": 3}
    assert data["items"][1] == {"source": "Epic EHI", "count": 2}
    assert data["items"][2] == {"source": "CDA XML", "count": 1}


@pytest.mark.asyncio
async def test_sources_excludes_deleted_and_duplicates(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    uid_u, pid = UUID(uid), patient.id
    await _add_record(db_session, uid_u, pid, "fhir_r4")
    await _add_record(db_session, uid_u, pid, "DeletedSource", deleted=True)
    await _add_record(db_session, uid_u, pid, "DuplicateSource", duplicate=True)
    await db_session.commit()

    resp = await client.get("/api/v1/dashboard/sources", headers=headers)
    assert resp.status_code == 200
    sources = {i["source"] for i in resp.json()["items"]}
    assert sources == {"fhir_r4"}
