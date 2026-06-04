"""TDD for the Snapshot Overview record endpoints (BACKEND-TODOs #3, #7, #8):

  * GET /records/{record_id}/fhir   — raw FHIR resource for single-record export
  * GET /records/recent?limit=5     — most recently ingested records
  * GET /records/stats              — totals + date span + source count
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient, seed_test_records


# --------------------------- /records/{id}/fhir ----------------------------


@pytest.mark.asyncio
async def test_fhir_export_requires_auth(client: AsyncClient):
    resp = await client.get(f"/api/v1/records/{uuid4()}/fhir")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_fhir_export_returns_raw_resource(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=2)
    target = records[0]

    resp = await client.get(f"/api/v1/records/{target.id}/fhir", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body == target.fhir_resource
    assert body.get("resourceType")


@pytest.mark.asyncio
async def test_fhir_export_404_for_unknown(client: AsyncClient, db_session: AsyncSession):
    headers, _ = await auth_headers(client)
    resp = await client.get(f"/api/v1/records/{uuid4()}/fhir", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_fhir_export_404_for_other_users_record(
    client: AsyncClient, db_session: AsyncSession
):
    # Owner creates a record
    headers_a, uid_a = await auth_headers(client, email="owner@example.com")
    patient_a = await create_test_patient(db_session, uid_a)
    records = await seed_test_records(db_session, uid_a, patient_a.id, count=1)

    # A different user must not read it
    headers_b, _ = await auth_headers(client, email="intruder@example.com")
    resp = await client.get(f"/api/v1/records/{records[0].id}/fhir", headers=headers_b)
    assert resp.status_code == 404


# ----------------------------- /records/recent -----------------------------


@pytest.mark.asyncio
async def test_recent_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/records/recent")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_recent_orders_by_created_at_desc(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    now = datetime.now(timezone.utc)
    # Insert with explicit created_at to control ingestion order
    for i, label in enumerate(["oldest", "middle", "newest"]):
        db_session.add(
            HealthRecord(
                id=uuid4(), patient_id=p, user_id=u, record_type="observation",
                fhir_resource_type="Observation",
                fhir_resource={"resourceType": "Observation", "valueQuantity": {"value": i, "unit": "x"}},
                source_format="fhir_r4", code_value=f"c{i}", code_display=label,
                display_text=label, effective_date=now,
                created_at=now - timedelta(hours=10 - i),
            )
        )
    await db_session.commit()

    resp = await client.get("/api/v1/records/recent", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    texts = [it["display_text"] for it in data["items"]]
    assert texts == ["newest", "middle", "oldest"]


@pytest.mark.asyncio
async def test_recent_respects_limit(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=10)

    resp = await client.get("/api/v1/records/recent?limit=3", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 3


@pytest.mark.asyncio
async def test_recent_default_limit_is_5(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=10)

    resp = await client.get("/api/v1/records/recent", headers=headers)
    assert len(resp.json()["items"]) == 5


@pytest.mark.asyncio
async def test_recent_item_shape(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=2)

    resp = await client.get("/api/v1/records/recent", headers=headers)
    item = resp.json()["items"][0]
    assert set(item.keys()) == {
        "id", "record_type", "display_text", "effective_date", "created_at",
        "source", "value", "unit",
    }
    assert item["source"]  # human source label, non-empty


@pytest.mark.asyncio
async def test_recent_value_unit_from_valuequantity(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="observation",
            fhir_resource_type="Observation",
            fhir_resource={"resourceType": "Observation", "valueQuantity": {"value": 6.6, "unit": "%"}},
            source_format="fhir_r4", code_value="4548-4", code_display="A1c",
            display_text="A1c 6.6%", effective_date=datetime.now(timezone.utc),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/recent", headers=headers)
    item = resp.json()["items"][0]
    assert item["value"] == 6.6
    assert item["unit"] == "%"


@pytest.mark.asyncio
async def test_recent_value_null_when_no_valuequantity(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", display_text="cond",
            effective_date=datetime.now(timezone.utc),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/recent", headers=headers)
    item = resp.json()["items"][0]
    assert item["value"] is None
    assert item["unit"] is None


@pytest.mark.asyncio
async def test_recent_excludes_deleted_and_dups(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", display_text="deleted",
            deleted_at=datetime.now(timezone.utc), effective_date=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", display_text="dup", is_duplicate=True,
            effective_date=datetime.now(timezone.utc),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/recent", headers=headers)
    assert resp.json()["total"] == 0


# ------------------------------ /records/stats -----------------------------


@pytest.mark.asyncio
async def test_stats_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/records/stats")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_stats_empty(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    await create_test_patient(db_session, uid)
    resp = await client.get("/api/v1/records/stats", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"total": 0, "first_date": None, "last_date": None, "source_count": 0}


@pytest.mark.asyncio
async def test_stats_counts_and_span(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    early = datetime(2019, 5, 1, tzinfo=timezone.utc)
    late = datetime(2024, 8, 15, tzinfo=timezone.utc)
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="observation",
            fhir_resource_type="Observation", fhir_resource={"resourceType": "Observation"},
            source_format="fhir_r4", display_text="a", effective_date=early,
        )
    )
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="epic_ehi", display_text="b", effective_date=late,
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/stats", headers=headers)
    data = resp.json()
    assert data["total"] == 2
    assert data["first_date"].startswith("2019-05-01")
    assert data["last_date"].startswith("2024-08-15")
    assert data["source_count"] == 2  # fhir_r4 + epic_ehi


@pytest.mark.asyncio
async def test_stats_excludes_deleted_and_dups(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)
    u, p = UUID(uid), patient.id
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="cda_r2", display_text="deleted",
            deleted_at=datetime.now(timezone.utc), effective_date=datetime.now(timezone.utc),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/stats", headers=headers)
    data = resp.json()
    assert data["total"] == 3  # deleted excluded
    assert data["source_count"] == 1  # only fhir_r4 from the seed; cda_r2 was deleted
