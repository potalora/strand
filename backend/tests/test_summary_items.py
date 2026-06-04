"""TDD for the "Add to summary" basket (BACKEND-TODO #7):

  * POST   /summary/items          — add a record to the user's summary basket
  * GET    /summary/items          — list the basket (joined to record metadata)
  * DELETE /summary/items/{id}     — remove an item
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import auth_headers, create_test_patient, seed_test_records


@pytest.mark.asyncio
async def test_add_item_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/summary/items", json={"record_id": str(uuid4())})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_add_item_success(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=2)

    resp = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records[0].id)}, headers=headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["record_id"] == str(records[0].id)
    assert "id" in body
    assert "created_at" in body


@pytest.mark.asyncio
async def test_add_item_unknown_record_404(client: AsyncClient, db_session: AsyncSession):
    headers, _ = await auth_headers(client)
    resp = await client.post(
        "/api/v1/summary/items", json={"record_id": str(uuid4())}, headers=headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_add_item_other_users_record_404(client: AsyncClient, db_session: AsyncSession):
    headers_a, uid_a = await auth_headers(client, email="owner2@example.com")
    patient_a = await create_test_patient(db_session, uid_a)
    records = await seed_test_records(db_session, uid_a, patient_a.id, count=1)

    headers_b, _ = await auth_headers(client, email="intruder2@example.com")
    resp = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records[0].id)}, headers=headers_b
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_add_item_idempotent(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=1)

    first = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records[0].id)}, headers=headers
    )
    second = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records[0].id)}, headers=headers
    )
    assert first.status_code == 200
    assert second.status_code == 200
    # same item returned, not a constraint error
    assert first.json()["id"] == second.json()["id"]

    listing = await client.get("/api/v1/summary/items", headers=headers)
    assert listing.json()["total"] == 1


@pytest.mark.asyncio
async def test_list_items_joined_metadata(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=3)
    for r in records[:2]:
        await client.post(
            "/api/v1/summary/items", json={"record_id": str(r.id)}, headers=headers
        )

    resp = await client.get("/api/v1/summary/items", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    item = data["items"][0]
    assert set(item.keys()) == {"id", "record_id", "display_text", "record_type", "created_at"}
    assert item["display_text"]
    assert item["record_type"]


@pytest.mark.asyncio
async def test_list_items_user_scoped(client: AsyncClient, db_session: AsyncSession):
    headers_a, uid_a = await auth_headers(client, email="a3@example.com")
    patient_a = await create_test_patient(db_session, uid_a)
    records_a = await seed_test_records(db_session, uid_a, patient_a.id, count=1)
    await client.post(
        "/api/v1/summary/items", json={"record_id": str(records_a[0].id)}, headers=headers_a
    )

    headers_b, _ = await auth_headers(client, email="b3@example.com")
    resp = await client.get("/api/v1/summary/items", headers=headers_b)
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_delete_item(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    records = await seed_test_records(db_session, uid, patient.id, count=1)
    added = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records[0].id)}, headers=headers
    )
    item_id = added.json()["id"]

    resp = await client.delete(f"/api/v1/summary/items/{item_id}", headers=headers)
    assert resp.status_code == 204

    listing = await client.get("/api/v1/summary/items", headers=headers)
    assert listing.json()["total"] == 0


@pytest.mark.asyncio
async def test_delete_unknown_item_404(client: AsyncClient, db_session: AsyncSession):
    headers, _ = await auth_headers(client)
    resp = await client.delete(f"/api/v1/summary/items/{uuid4()}", headers=headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_other_users_item_404(client: AsyncClient, db_session: AsyncSession):
    headers_a, uid_a = await auth_headers(client, email="a4@example.com")
    patient_a = await create_test_patient(db_session, uid_a)
    records_a = await seed_test_records(db_session, uid_a, patient_a.id, count=1)
    added = await client.post(
        "/api/v1/summary/items", json={"record_id": str(records_a[0].id)}, headers=headers_a
    )
    item_id = added.json()["id"]

    headers_b, _ = await auth_headers(client, email="b4@example.com")
    resp = await client.delete(f"/api/v1/summary/items/{item_id}", headers=headers_b)
    assert resp.status_code == 404
