"""Auth-hardening regressions (Wave 3).

W21a — an expired lockout must reset the failed-attempt counter so a single
post-lockout failure can't immediately re-lock a known account (a 1-request-per-
15-min permanent-lock DoS).
W21b — reuse of an already-revoked refresh token must invalidate the WHOLE token
family, not just the replayed token, so a thief who already rotated can't keep a
live chain.
W23 — expired ``revoked_tokens`` rows must be purgeable so the blacklist (and the
per-request revocation lookup) doesn't grow without bound.
SEC-PHI-08 — production logging must never drop below INFO (DEBUG logs can carry
extracted entity text / PHI).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.token_blacklist import RevokedToken
from app.models.user import User
from tests.conftest import auth_headers


# ---------------------------------------------------------------------------
# W21a — lockout counter resets once the lockout window has elapsed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lockout_counter_resets_after_expiry(
    client: AsyncClient, db_session: AsyncSession
):
    email = "lockout-reset@example.com"
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!", "display_name": "T"},
    )
    user_id = UUID(reg.json()["id"])

    # Simulate an EXPIRED lockout: counter pinned at the cap, ``locked_until`` in
    # the past. Before the fix the next failed attempt increments 5 -> 6 and
    # re-locks instantly, keeping the account permanently locked.
    user = (
        await db_session.execute(select(User).where(User.id == user_id))
    ).scalar_one()
    user.failed_login_attempts = 5
    user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
    user.last_failed_login_at = datetime.now(timezone.utc) - timedelta(minutes=20)
    await db_session.commit()

    # A single wrong attempt after the window must NOT re-lock at the cap.
    wrong = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "WrongPass123!"},
    )
    assert wrong.status_code == 401

    db_session.expire_all()
    user = (
        await db_session.execute(select(User).where(User.id == user_id))
    ).scalar_one()
    assert user.locked_until is None
    assert user.failed_login_attempts == 1  # reset to 0, then +1 for this attempt

    # ...and a correct login still succeeds — the account is no longer locked.
    ok = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# W21b — refresh-token reuse revokes the entire token family
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_reuse_revokes_family(client: AsyncClient):
    email = "refresh-family@example.com"
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!", "display_name": "T"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "SecurePass123!"},
    )
    r1 = login.json()["refresh_token"]

    # Legit rotation: R1 -> R2 (R1 is now revoked).
    rot = await client.post("/api/v1/auth/refresh", json={"refresh_token": r1})
    assert rot.status_code == 200
    r2 = rot.json()["refresh_token"]

    # Attacker replays the already-revoked R1: reuse detected -> 401.
    reuse = await client.post("/api/v1/auth/refresh", json={"refresh_token": r1})
    assert reuse.status_code == 401

    # Family invalidation: the legitimately-rotated R2 is now ALSO dead.
    r2_attempt = await client.post("/api/v1/auth/refresh", json={"refresh_token": r2})
    assert r2_attempt.status_code == 401

    # Recovery: a fresh interactive login clears the family marker and works again.
    relogin = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert relogin.status_code == 200
    r3 = relogin.json()["refresh_token"]
    ok = await client.post("/api/v1/auth/refresh", json={"refresh_token": r3})
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# W23 — purge expired blacklist rows, keep live ones
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_purge_removes_expired_tokens_keeps_live(
    client: AsyncClient, db_session: AsyncSession
):
    from app.services.auth_service import purge_expired_revoked_tokens

    _, user_id = await auth_headers(client, "purge@example.com")
    uid = UUID(user_id)
    now = datetime.now(timezone.utc)

    db_session.add_all(
        [
            RevokedToken(
                jti="expired-jti",
                user_id=uid,
                token_type="refresh",
                expires_at=now - timedelta(hours=1),
            ),
            RevokedToken(
                jti="live-jti",
                user_id=uid,
                token_type="refresh",
                expires_at=now + timedelta(hours=1),
            ),
        ]
    )
    await db_session.commit()

    removed = await purge_expired_revoked_tokens(db_session)
    assert removed == 1

    remaining = (
        await db_session.execute(select(RevokedToken.jti))
    ).scalars().all()
    assert "live-jti" in remaining
    assert "expired-jti" not in remaining


# ---------------------------------------------------------------------------
# SEC-PHI-08 — production logging never drops below INFO
# ---------------------------------------------------------------------------


def test_log_level_clamped_to_info_in_production():
    from app.main import resolve_log_level

    # DEBUG is suppressed in production (may carry extracted-entity text / PHI).
    assert resolve_log_level("DEBUG", is_production=True) == logging.INFO
    # Non-debug levels pass through unchanged in production.
    assert resolve_log_level("WARNING", is_production=True) == logging.WARNING
    assert resolve_log_level("INFO", is_production=True) == logging.INFO
    # Dev/test can still opt into DEBUG.
    assert resolve_log_level("DEBUG", is_production=False) == logging.DEBUG
    # Unknown level falls back to INFO regardless of environment.
    assert resolve_log_level("bogus", is_production=True) == logging.INFO
