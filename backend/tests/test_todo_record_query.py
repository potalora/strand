"""TDD for frontend TODOs A1 (?status= filter) and A2 (?sort=&order=) on GET /records.

Seeded records (count=5) cycle the 5 SAMPLE_RECORDS, whose statuses are
[active, final, active, finished, completed] and whose display_texts sort
alphabetically as: Hemoglobin…, Influenza…, Metformin…, Office visit, Type 2…
Effective dates ascend with index (record 0 is the oldest).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import auth_headers, create_test_patient, seed_test_records


# --------------------------- A1: ?status= filter ---------------------------


@pytest.mark.asyncio
async def test_records_status_filter_returns_only_matching(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records?status=active", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    # condition + medication are status "active"
    assert data["total"] == 2
    assert {item["status"] for item in data["items"]} == {"active"}


@pytest.mark.asyncio
async def test_records_status_combines_with_type(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get(
        "/api/v1/records?record_type=condition&status=active", headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["record_type"] == "condition"
    assert data["items"][0]["status"] == "active"


@pytest.mark.asyncio
async def test_records_status_filter_no_match_is_empty(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records?status=entered-in-error", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


# --------------------------- A2: ?sort=&order= -----------------------------


@pytest.mark.asyncio
async def test_records_sort_display_text_ascending(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get(
        "/api/v1/records?sort=display_text&order=asc&page_size=100", headers=headers
    )
    assert resp.status_code == 200
    texts = [item["display_text"] for item in resp.json()["items"]]
    assert texts == sorted(texts)
    assert texts[0].startswith("Hemoglobin")


@pytest.mark.asyncio
async def test_records_sort_date_ascending_oldest_first(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get(
        "/api/v1/records?sort=date&order=asc&page_size=100", headers=headers
    )
    assert resp.status_code == 200
    dates = [item["effective_date"] for item in resp.json()["items"]]
    assert dates == sorted(dates)


@pytest.mark.asyncio
async def test_records_default_sort_is_effective_date_desc(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records?page_size=100", headers=headers)
    assert resp.status_code == 200
    dates = [item["effective_date"] for item in resp.json()["items"]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.asyncio
async def test_records_sort_type_ascending(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get(
        "/api/v1/records?sort=type&order=asc&page_size=100", headers=headers
    )
    assert resp.status_code == 200
    types = [item["record_type"] for item in resp.json()["items"]]
    assert types == sorted(types)
