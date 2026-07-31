from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient, seed_test_records


@pytest.mark.asyncio
async def test_records_unauthenticated(client: AsyncClient):
    """GET /records without token returns 401."""
    resp = await client.get("/api/v1/records")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_records_empty(client: AsyncClient, db_session: AsyncSession):
    """GET /records with no data returns empty list."""
    headers, uid = await auth_headers(client)
    resp = await client.get("/api/v1/records", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["page_size"] == 20


@pytest.mark.asyncio
async def test_records_with_data(client: AsyncClient, db_session: AsyncSession):
    """GET /records returns seeded records with correct schema."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 5

    # Verify schema fields per handoff
    item = data["items"][0]
    for field in [
        "id", "patient_id", "record_type", "fhir_resource_type",
        "fhir_resource", "source_format", "effective_date",
        "status", "category", "code_system", "code_value",
        "code_display", "display_text", "created_at",
    ]:
        assert field in item, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_records_pagination(client: AsyncClient, db_session: AsyncSession):
    """Pagination: 25 records with page_size=20 gives correct pages."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=25)

    # Page 1
    resp1 = await client.get("/api/v1/records?page=1&page_size=20", headers=headers)
    data1 = resp1.json()
    assert data1["total"] == 25
    assert len(data1["items"]) == 20
    assert data1["page"] == 1

    # Page 2
    resp2 = await client.get("/api/v1/records?page=2&page_size=20", headers=headers)
    data2 = resp2.json()
    assert data2["total"] == 25
    assert len(data2["items"]) == 5
    assert data2["page"] == 2


@pytest.mark.asyncio
async def test_records_filter_by_type(client: AsyncClient, db_session: AsyncSession):
    """Filter by record_type returns only matching records."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records?record_type=condition", headers=headers)
    data = resp.json()
    assert data["total"] >= 1
    for item in data["items"]:
        assert item["record_type"] == "condition"


@pytest.mark.asyncio
async def test_records_search(client: AsyncClient, db_session: AsyncSession):
    """Search on display_text returns matching records."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records?search=diabetes", headers=headers)
    data = resp.json()
    assert data["total"] >= 1
    for item in data["items"]:
        assert "diabetes" in item["display_text"].lower() or "diabetes" in (item["code_display"] or "").lower()


@pytest.mark.asyncio
async def test_records_combined_filters(client: AsyncClient, db_session: AsyncSession):
    """Combined type + search filter."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=10)

    resp = await client.get("/api/v1/records?record_type=observation&search=Hemoglobin", headers=headers)
    data = resp.json()
    for item in data["items"]:
        assert item["record_type"] == "observation"


@pytest.mark.asyncio
async def test_get_record_by_id(client: AsyncClient, db_session: AsyncSession):
    """GET /records/:id returns the specific record."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=1)

    resp = await client.get(f"/api/v1/records/{records[0].id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(records[0].id)
    assert data["display_text"] == records[0].display_text


@pytest.mark.asyncio
async def test_get_record_not_found(client: AsyncClient, db_session: AsyncSession):
    """GET /records/:id with invalid UUID returns 404."""
    headers, _ = await auth_headers(client)
    resp = await client.get("/api/v1/records/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 404
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_get_record_other_user(client: AsyncClient, db_session: AsyncSession):
    """User A cannot access User B's records (returns 404)."""
    _, uid_a = await auth_headers(client, email="usera@test.com")
    headers_b, uid_b = await auth_headers(client, email="userb@test.com")
    patient = await create_test_patient(db_session, uid_a)
    records = await seed_test_records(db_session, uid_a, patient.id, count=1)

    resp = await client.get(f"/api/v1/records/{records[0].id}", headers=headers_b)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_record(client: AsyncClient, db_session: AsyncSession):
    """DELETE /records/:id soft-deletes (204) and excludes from list."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=3)

    # Delete one
    resp = await client.delete(f"/api/v1/records/{records[0].id}", headers=headers)
    assert resp.status_code == 204

    # Verify excluded from list
    list_resp = await client.get("/api/v1/records", headers=headers)
    ids = [item["id"] for item in list_resp.json()["items"]]
    assert str(records[0].id) not in ids
    assert list_resp.json()["total"] == 2


@pytest.mark.asyncio
async def test_delete_record_not_found(client: AsyncClient, db_session: AsyncSession):
    """DELETE /records/:id with invalid UUID returns 404."""
    headers, _ = await auth_headers(client)
    resp = await client.delete("/api/v1/records/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_record_excluded_from_get(client: AsyncClient, db_session: AsyncSession):
    """Soft-deleted records are excluded from GET /records/:id."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=1)

    await client.delete(f"/api/v1/records/{records[0].id}", headers=headers)
    resp = await client.get(f"/api/v1/records/{records[0].id}", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_record_includes_ai_metadata(client: AsyncClient, db_session: AsyncSession):
    """GET /records/:id includes ai_extracted and confidence_score."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    rec = HealthRecord(
        id=uuid4(),
        patient_id=patient.id,
        user_id=uid,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition", "code": {"text": "AI Test Condition"}},
        source_format="ai_extraction",
        display_text="AI Test Condition",
        ai_extracted=True,
        confidence_score=0.85,
    )
    db_session.add(rec)
    await db_session.commit()
    await db_session.refresh(rec)

    resp = await client.get(f"/api/v1/records/{rec.id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ai_extracted"] is True
    assert data["confidence_score"] == 0.85


@pytest.mark.asyncio
async def test_list_records_includes_ai_metadata(client: AsyncClient, db_session: AsyncSession):
    """GET /records list includes ai_extracted and confidence_score on items."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    rec = HealthRecord(
        id=uuid4(),
        patient_id=patient.id,
        user_id=uid,
        record_type="observation",
        fhir_resource_type="Observation",
        fhir_resource={"resourceType": "Observation", "code": {"text": "AI Test Lab"}},
        source_format="ai_extraction",
        display_text="AI Test Lab",
        ai_extracted=True,
        confidence_score=0.72,
    )
    db_session.add(rec)
    await db_session.commit()

    resp = await client.get("/api/v1/records", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    ai_item = next(i for i in data["items"] if i["display_text"] == "AI Test Lab")
    assert ai_item["ai_extracted"] is True
    assert ai_item["confidence_score"] == 0.72
