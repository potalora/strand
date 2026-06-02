"""TDD for frontend TODO B2: GET /records/series?code_value= — the time-series of
one observation code (ascending by date), for the detail-sheet trend sparkline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient


async def _add_obs(db, uid, pid, code_value, value, when, unit="%"):
    db.add(
        HealthRecord(
            id=uuid4(),
            patient_id=pid,
            user_id=uid,
            record_type="observation",
            fhir_resource_type="Observation",
            fhir_resource={
                "resourceType": "Observation",
                "valueQuantity": {"value": value, "unit": unit},
            },
            source_format="fhir_r4",
            code_system="http://loinc.org",
            code_value=code_value,
            code_display="Hemoglobin A1c",
            display_text=f"HbA1c {value}{unit}",
            effective_date=when,
        )
    )


@pytest.mark.asyncio
async def test_series_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/records/series?code_value=4548-4")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_series_returns_points_ascending(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    uid_u, pid = UUID(uid), patient.id
    await _add_obs(db_session, uid_u, pid, "4548-4", 7.8, datetime(2021, 6, 1, tzinfo=timezone.utc))
    await _add_obs(db_session, uid_u, pid, "4548-4", 6.6, datetime(2023, 2, 20, tzinfo=timezone.utc))
    await _add_obs(db_session, uid_u, pid, "4548-4", 7.1, datetime(2022, 1, 15, tzinfo=timezone.utc))
    await db_session.commit()

    resp = await client.get("/api/v1/records/series?code_value=4548-4", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["code_value"] == "4548-4"
    assert data["total"] == 3
    values = [p["value"] for p in data["items"]]
    assert values == [7.8, 7.1, 6.6]  # ascending by date
    assert data["items"][0]["unit"] == "%"
    assert data["items"][0]["effective_date"].startswith("2021-06-01")


@pytest.mark.asyncio
async def test_series_excludes_other_codes(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    uid_u, pid = UUID(uid), patient.id
    await _add_obs(db_session, uid_u, pid, "4548-4", 7.0, datetime(2021, 6, 1, tzinfo=timezone.utc))
    await _add_obs(db_session, uid_u, pid, "13457-7", 120, datetime(2021, 6, 1, tzinfo=timezone.utc), unit="mg/dL")
    await db_session.commit()

    resp = await client.get("/api/v1/records/series?code_value=4548-4", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


@pytest.mark.asyncio
async def test_series_empty_for_unknown_code(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    await create_test_patient(db_session, uid)
    resp = await client.get("/api/v1/records/series?code_value=does-not-exist", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
