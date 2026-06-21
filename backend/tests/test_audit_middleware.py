"""TDD for W16 (HIPAA AUDIT-02/-03, SEC-AUTH-11): a REQUEST-LEVEL audit safety
net so no authenticated PHI access under ``/api/v1`` can be silently un-logged.

Several PHI-relevant endpoints (``GET /upload/history``, ``GET /upload/{id}/
extraction``, ``GET /auth/me`` …) write no per-endpoint audit row. The middleware
in ``app.middleware.audit`` writes a generic ``api.access`` row for every
authenticated request (decoding the Bearer JWT for ``user_id``; NULL when the
token is missing/invalid so failed-auth is still visible), capturing method,
path and response status in ``details`` — never bodies or query strings (PHI).

The middleware uses a FRESH session (its own commit), independent of the
request's session. In tests that session is bound to ``app.middleware.audit
.async_session_factory``; the ``audit_session_factory`` fixture repoints it at
the test database so the committed row is readable via ``db_session``.
"""

from __future__ import annotations

from uuid import UUID

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.audit import AuditLog
from tests.conftest import TEST_DB_URL, auth_headers


@pytest_asyncio.fixture
async def audit_session_factory(monkeypatch):
    """Repoint the middleware's fresh-session factory at the TEST database.

    In production the middleware opens its own session via
    ``app.database.async_session_factory`` (bound to the non-``_test`` URL).
    The test suite reads/writes ``<db>_test``, so without this the audit row
    would land in a different database and be invisible to ``db_session``.
    """
    import app.middleware.audit as audit_mod

    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(audit_mod, "async_session_factory", factory)
    yield factory
    await engine.dispose()


async def _fetch_access_rows(db: AsyncSession, path: str) -> list[AuditLog]:
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.action == "api.access")
        .where(AuditLog.details["path"].astext == path)
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_unlogged_endpoint_writes_request_audit_row(
    client: AsyncClient, db_session: AsyncSession, audit_session_factory
):
    """An authenticated hit on a previously-UNLOGGED endpoint (GET
    /upload/history has no per-endpoint log_audit_event) now produces an
    ``api.access`` row carrying user_id, method, path and status 200."""
    headers, uid = await auth_headers(client)

    resp = await client.get("/api/v1/upload/history", headers=headers)
    assert resp.status_code == 200

    rows = await _fetch_access_rows(db_session, "/api/v1/upload/history")
    assert len(rows) == 1
    row = rows[0]
    assert row.user_id == UUID(uid)
    assert row.details["method"] == "GET"
    assert row.details["path"] == "/api/v1/upload/history"
    assert row.details["status"] == 200


@pytest.mark.asyncio
async def test_unauthenticated_request_records_access_row_with_null_user(
    client: AsyncClient, db_session: AsyncSession, audit_session_factory
):
    """An UNAUTHENTICATED request to a protected endpoint still records an
    access row (user_id NULL) — failed-auth attempts stay visible in the trail."""
    resp = await client.get("/api/v1/upload/history")
    assert resp.status_code == 401

    rows = await _fetch_access_rows(db_session, "/api/v1/upload/history")
    assert len(rows) == 1
    row = rows[0]
    assert row.user_id is None
    assert row.details["method"] == "GET"
    assert row.details["status"] == 401


@pytest.mark.asyncio
async def test_audit_write_failure_does_not_500_request(
    client: AsyncClient, db_session: AsyncSession, monkeypatch
):
    """A failure inside the request-level audit write must NOT fail the request:
    the middleware is a best-effort safety net (catch + log, never raise)."""
    headers, uid = await auth_headers(client)

    async def _boom(*args, **kwargs):
        raise RuntimeError("audit backend down")

    monkeypatch.setattr(
        "app.middleware.audit.log_request_access", _boom, raising=False
    )

    resp = await client.get("/api/v1/upload/history", headers=headers)
    assert resp.status_code == 200
