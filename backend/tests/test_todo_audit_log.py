"""TDD for frontend TODO A4: GET /audit-log — a paginated, user-scoped read of
the audit_log table (written on every mutation, but previously unreadable).
Powers the Admin → System audit log table.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from tests.conftest import auth_headers


async def _seed_audit(db: AsyncSession, user_id: UUID, action: str, when: datetime, ip: str = "9.9.9.9"):
    db.add(AuditLog(user_id=user_id, action=action, resource_type="health_record", ip_address=ip, created_at=when))
    await db.commit()


@pytest.mark.asyncio
async def test_audit_log_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/audit-log")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_audit_log_lists_user_events_newest_first(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    await _seed_audit(db_session, UUID(uid), "records.view", base)
    await _seed_audit(db_session, UUID(uid), "dashboard.overview", base + timedelta(minutes=5))

    resp = await client.get("/api/v1/audit-log", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    actions = [i["action"] for i in data["items"]]
    # newest first
    assert actions.index("dashboard.overview") < actions.index("records.view")
    first = data["items"][0]
    for key in ("action", "resource_type", "ip_address", "created_at"):
        assert key in first


@pytest.mark.asyncio
async def test_audit_log_is_user_scoped(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    # a second real user (audit_log.user_id is an FK to users)
    _, other_uid = await auth_headers(client, email="other@example.com")
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    await _seed_audit(db_session, UUID(uid), "records.view", base)
    await _seed_audit(db_session, UUID(other_uid), "someone.else.secret", base + timedelta(minutes=1))

    resp = await client.get("/api/v1/audit-log", headers=headers)
    assert resp.status_code == 200
    actions = [i["action"] for i in resp.json()["items"]]
    assert "someone.else.secret" not in actions


@pytest.mark.asyncio
async def test_audit_log_pagination(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    for i in range(5):
        await _seed_audit(db_session, UUID(uid), f"action.{i}", base + timedelta(minutes=i))

    resp = await client.get("/api/v1/audit-log?page=1&limit=2", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    # the 5 seeded rows + the real user.register audit entry
    assert data["total"] >= 5
