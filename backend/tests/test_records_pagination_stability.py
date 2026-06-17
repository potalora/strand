from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient


async def _seed_tie_heavy_records(
    db_session: AsyncSession, uid: str, patient_id, count: int = 250
) -> int:
    """Seed records engineered to break OFFSET pagination without a tiebreaker.

    Large tie groups on effective_date (incl. a big NULL group) deliberately
    straddle the page_size=100 boundaries so Postgres' bounded top-N sort can
    return a row on two pages (and drop another) unless the ORDER BY is a total
    order. record_type also has only a few distinct values → big ties for the
    secondary-sort coverage.
    """
    # Date buckets sized so BOTH page boundaries (100, 200) fall INSIDE a tie
    # group: one big dated group of 150 (straddles the 100 boundary) and a NULL
    # group of 100 (straddles the 200 boundary).
    date_a = datetime(2024, 6, 1, tzinfo=timezone.utc)
    buckets: list[tuple[datetime | None, int]] = [(date_a, 150), (None, 100)]
    types = ["condition", "observation", "medication"]

    n = 0
    for eff_date, group_size in buckets:
        for i in range(group_size):
            rec = HealthRecord(
                id=uuid4(),
                patient_id=patient_id,
                user_id=uid,
                record_type=types[n % len(types)],
                fhir_resource_type="Condition",
                fhir_resource={"resourceType": "Condition", "code": {"text": f"rec-{n}"}},
                source_format="fhir_r4",
                effective_date=eff_date,
                status="active",
                category=["condition"],
                display_text=f"Record {n}",
                is_duplicate=False,
            )
            db_session.add(rec)
            n += 1
    await db_session.commit()
    assert n == count
    return n


async def _page_through(client: AsyncClient, headers: dict, sort: str) -> tuple[list[str], int]:
    """Page through all records at page_size=100; return (accumulated_ids, total)."""
    page = 1
    ids: list[str] = []
    total = None
    while True:
        resp = await client.get(
            f"/api/v1/records?page={page}&page_size=100&sort={sort}", headers=headers
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        total = data["total"]
        items = data["items"]
        ids.extend(i["id"] for i in items)
        if len(items) < 100:
            break
        page += 1
        if page > 50:  # safety net against an infinite loop
            break
    return ids, total


@pytest.mark.asyncio
async def test_pagination_no_duplicates_or_drops_default_date_sort(
    client: AsyncClient, db_session: AsyncSession
):
    """Default date sort: paging accumulates every record exactly once."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    total_seeded = await _seed_tie_heavy_records(db_session, uid, patient.id)

    ids, total = await _page_through(client, headers, sort="date")

    assert total == total_seeded
    # (a) no duplicate ids across pages
    assert len(ids) == len(set(ids)), "duplicate id returned across pages"
    # (b) accumulated unique count == reported total
    assert len(set(ids)) == total


@pytest.mark.asyncio
async def test_pagination_no_duplicates_or_drops_type_sort(
    client: AsyncClient, db_session: AsyncSession
):
    """Secondary sort key (type) also yields a stable total order."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    total_seeded = await _seed_tie_heavy_records(db_session, uid, patient.id)

    ids, total = await _page_through(client, headers, sort="type")

    assert total == total_seeded
    assert len(ids) == len(set(ids)), "duplicate id returned across pages"
    assert len(set(ids)) == total


@pytest.mark.asyncio
async def test_pagination_covers_all_ids_no_missing(
    client: AsyncClient, db_session: AsyncSession
):
    """No id present in a direct full query is missing from the paged set."""
    from sqlalchemy import select

    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await _seed_tie_heavy_records(db_session, uid, patient.id)

    expected_ids = {
        str(r) for r in (
            await db_session.execute(
                select(HealthRecord.id).where(
                    HealthRecord.user_id == uid,
                    HealthRecord.deleted_at.is_(None),
                    HealthRecord.is_duplicate.is_(False),
                )
            )
        ).scalars().all()
    }

    ids, total = await _page_through(client, headers, sort="date")

    assert set(ids) == expected_ids, "paged ids differ from full query (dropped/duplicated)"
    assert len(ids) == len(expected_ids)
