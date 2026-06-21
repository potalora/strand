from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from passlib.context import CryptContext
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.middleware.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.middleware.encryption import blind_index
from app.models.token_blacklist import RevokedToken
from app.models.user import User
from app.schemas.auth import TokenResponse

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


def hash_password(password: str) -> str:
    """Hash a password using bcrypt with cost factor 12."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


async def register_user(
    db: AsyncSession,
    email: str,
    password: str,
    display_name: str | None = None,
) -> User:
    """Register a new user."""
    # ``email`` is encrypted at rest and not directly queryable; look up the
    # deterministic blind index instead.
    email_hmac = blind_index(email)
    existing = await db.execute(select(User).where(User.email_hmac == email_hmac))
    if existing.scalar_one_or_none():
        raise ValueError("Email already registered")

    user = User(
        email=email,
        email_hmac=email_hmac,
        password_hash=hash_password(password),
        display_name=display_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(
    db: AsyncSession,
    email: str,
    password: str,
) -> TokenResponse:
    """Authenticate a user and return JWT tokens."""
    result = await db.execute(select(User).where(User.email_hmac == blind_index(email)))
    user = result.scalar_one_or_none()

    if not user:
        raise ValueError("Invalid email or password")

    now = datetime.now(timezone.utc)

    # Check account lockout.
    if user.locked_until:
        if user.locked_until > now:
            raise ValueError("Account is temporarily locked. Please try again later.")
        # W21a: the lockout window has elapsed. Reset the counter and clear the
        # lock BEFORE processing this attempt. Previously the counter only reset
        # on a *successful* login, so a single post-lockout failure incremented
        # 5 -> 6 and re-locked instantly — an attacker (or a typo) could keep a
        # known account permanently locked with one request every 15 minutes.
        user.failed_login_attempts = 0
        user.locked_until = None

    if not verify_password(password, user.password_hash):
        # Increment failed attempts
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        user.last_failed_login_at = now
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            logger.warning("Account locked for user %s after %d failed attempts", user.id, user.failed_login_attempts)
        await db.commit()
        raise ValueError("Invalid email or password")

    if not user.is_active:
        raise ValueError("Account is disabled")

    # Reset failed attempts on successful login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_failed_login_at = None
    # W21b: a fresh interactive login (which proves possession of the password)
    # re-establishes a trusted session, so clear any token-family revocation
    # marker — previously-compromised refresh chains no longer need to block this
    # user, and new tokens minted below must be honoured.
    await db.execute(
        delete(RevokedToken).where(RevokedToken.jti == _refresh_family_jti(user.id))
    )
    await db.commit()

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


def _refresh_family_jti(user_id: UUID) -> str:
    """Sentinel blacklist key for a user's token-family revocation marker (W21b)."""
    return f"refresh-family:{user_id}"


async def _revoke_refresh_family(db: AsyncSession, user_id: UUID) -> None:
    """Invalidate ALL of a user's outstanding refresh tokens (token-family kill).

    Approach: refresh tokens are stateless JWTs, so we can't enumerate the live
    ones to blacklist each JTI individually. Instead we record a single per-user
    *family marker* row in ``revoked_tokens`` (a sentinel ``jti``,
    ``token_type='refresh_family'``). While the marker is present,
    ``refresh_tokens`` rejects EVERY refresh for that user — killing both the
    thief's stolen chain and the legitimately-rotated chain at once. A successful
    interactive login clears the marker (see ``authenticate_user``); the purge
    job removes it once it can no longer match a live token (``expires_at`` is set
    one refresh-lifetime out). This is per-user, so it also drops other devices'
    sessions — the intended, conservative response to a confirmed token leak.
    """
    sentinel = _refresh_family_jti(user_id)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.jwt_refresh_token_expire_days)

    existing = await db.execute(
        select(RevokedToken).where(RevokedToken.jti == sentinel)
    )
    marker = existing.scalar_one_or_none()
    if marker is not None:
        marker.revoked_at = now
        marker.expires_at = expires_at
    else:
        db.add(
            RevokedToken(
                jti=sentinel,
                user_id=user_id,
                token_type="refresh_family",
                expires_at=expires_at,
            )
        )
    try:
        await db.commit()
    except IntegrityError:
        # Concurrent reuse-detection inserted the marker first — equivalent state.
        await db.rollback()


async def refresh_tokens(
    db: AsyncSession,
    refresh_token_str: str,
) -> TokenResponse:
    """Refresh access token using a valid refresh token (with rotation)."""
    payload = decode_token(refresh_token_str)

    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")

    user_id = UUID(payload["sub"])
    old_jti = payload.get("jti")

    # W21b reuse detection: a refresh token whose JTI is already revoked is being
    # replayed. The legitimate client already rotated it away, so a second
    # presentation means the token leaked — revoke the WHOLE token family, not
    # just this one token, then reject.
    if old_jti:
        result = await db.execute(
            select(RevokedToken).where(RevokedToken.jti == old_jti)
        )
        if result.scalar_one_or_none():
            await _revoke_refresh_family(db, user_id)
            raise ValueError("Refresh token has been revoked")

    # If a prior reuse tripped family revocation, reject every refresh for this
    # user until they log in again (which clears the marker). This is what makes
    # the legitimately-rotated descendant token die too.
    family_marker = await db.execute(
        select(RevokedToken).where(RevokedToken.jti == _refresh_family_jti(user_id))
    )
    if family_marker.scalar_one_or_none():
        raise ValueError("Refresh token has been revoked")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise ValueError("User not found or disabled")

    # Rotate: revoke the presented refresh token, then mint a fresh pair.
    if old_jti:
        exp = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else datetime.now(timezone.utc)
        db.add(
            RevokedToken(
                jti=old_jti,
                user_id=user_id,
                token_type="refresh",
                expires_at=expires_at,
            )
        )
        try:
            await db.commit()
        except IntegrityError:
            # Check-then-insert race: a concurrent refresh revoked this same JTI
            # between our SELECT above and this INSERT. Treat it as reuse and
            # return the same 401, never a 500.
            await db.rollback()
            raise ValueError("Refresh token has been revoked")

    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


async def purge_expired_revoked_tokens(db: AsyncSession) -> int:
    """Delete blacklist rows whose tokens have already expired (W23).

    ``revoked_tokens`` otherwise grows without bound, and the per-request
    revocation lookup slows as it fills. A row is safe to delete once its
    ``expires_at`` is in the past: the underlying JWT can no longer authenticate
    (its own ``exp`` has passed), so it never needs a blacklist entry again. The
    per-user family marker (W21b) carries ``expires_at = now + refresh window``,
    so it is purged only after every refresh token it could match has expired.

    Unlike ``audit_log`` (append-only via a DB trigger), ``revoked_tokens`` has
    no such trigger, so a DELETE here is allowed. Returns the rows removed.
    """
    result = await db.execute(
        delete(RevokedToken).where(
            RevokedToken.expires_at < datetime.now(timezone.utc)
        )
    )
    await db.commit()
    return result.rowcount or 0


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    """Fetch a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
