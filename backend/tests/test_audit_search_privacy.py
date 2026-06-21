"""TDD for W25 (SEC-PHI-08 / HIPAA AUDIT-08): record-search audit rows must NOT
persist the user's raw free-text query.

``audit_log.details`` is JSONB and is NOT encrypted at rest. The record
search/list endpoints used to write the raw query string into ``details``
(``{"search": search}`` and ``{"query": q}``). A user searching for a patient
name or MRN therefore leaked that PHI into the plaintext audit trail.

The fix stores a privacy-preserving signal instead — a truncated SHA-256 of the
normalized term plus its length and presence — which still supports
"same search repeated" activity analysis without persisting PHI.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from tests.conftest import auth_headers

# A fake "name" that would be PHI if persisted raw.
_PHI_TERM = "Johnson"


def _expected_hash(value: str) -> str:
    return hashlib.sha256(value.strip().casefold().encode("utf-8")).hexdigest()[:16]


async def _fetch_details(db: AsyncSession, action: str) -> dict:
    result = await db.execute(select(AuditLog).where(AuditLog.action == action))
    rows = list(result.scalars().all())
    assert len(rows) == 1, f"expected exactly one {action!r} audit row, got {len(rows)}"
    return rows[0].details or {}


@pytest.mark.asyncio
async def test_list_records_audit_does_not_persist_raw_search(
    client: AsyncClient, db_session: AsyncSession
):
    """GET /records?search=<PHI> must not write the raw term to audit_log."""
    headers, _ = await auth_headers(client)

    resp = await client.get(
        "/api/v1/records", params={"search": _PHI_TERM}, headers=headers
    )
    assert resp.status_code == 200

    details = await _fetch_details(db_session, "records.list")

    # The raw query string must NOT appear anywhere in the serialized details.
    blob = json.dumps(details).lower()
    assert _PHI_TERM.lower() not in blob
    assert "search" not in details or details.get("search") is None  # no raw "search" key value

    # A privacy-preserving signal IS present and useful.
    assert details["search_present"] is True
    assert details["search_hash"] == _expected_hash(_PHI_TERM)
    assert details["search_len"] == len(_PHI_TERM)

    # Unchanged audit fields are still recorded.
    assert "page" in details
    assert "total" in details


@pytest.mark.asyncio
async def test_search_records_audit_does_not_persist_raw_query(
    client: AsyncClient, db_session: AsyncSession
):
    """GET /records/search?q=<PHI> must not write the raw term to audit_log."""
    headers, _ = await auth_headers(client)

    resp = await client.get(
        "/api/v1/records/search", params={"q": _PHI_TERM}, headers=headers
    )
    assert resp.status_code == 200

    details = await _fetch_details(db_session, "records.search")

    blob = json.dumps(details).lower()
    assert _PHI_TERM.lower() not in blob
    assert "query" not in details  # raw "query" key removed entirely

    assert details["search_present"] is True
    assert details["search_hash"] == _expected_hash(_PHI_TERM)
    assert details["search_len"] == len(_PHI_TERM)

    # Unchanged audit field still recorded.
    assert "result_count" in details


@pytest.mark.asyncio
async def test_list_records_audit_signal_absent_when_no_search(
    client: AsyncClient, db_session: AsyncSession
):
    """With no search term, the row records that no search was performed."""
    headers, _ = await auth_headers(client)

    resp = await client.get("/api/v1/records", headers=headers)
    assert resp.status_code == 200

    details = await _fetch_details(db_session, "records.list")
    assert details["search_present"] is False
    assert "search_hash" not in details
