from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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

    # Check account lockout
    if user.locked_until and user.locked_until > datetime.now(timezone.utc):
        raise ValueError("Account is temporarily locked. Please try again later.")

    if not verify_password(password, user.password_hash):
        # Increment failed attempts
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        user.last_failed_login_at = datetime.now(timezone.utc)
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=LOCKOUT_DURATION_MINUTES
            )
            logger.warning("Account locked for user %s after %d failed attempts", user.id, user.failed_login_attempts)
        await db.commit()
        raise ValueError("Invalid email or password")

    if not user.is_active:
        raise ValueError("Account is disabled")

    # Reset failed attempts on successful login
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_failed_login_at = None
    await db.commit()

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


async def refresh_tokens(
    db: AsyncSession,
    refresh_token_str: str,
) -> TokenResponse:
    """Refresh access token using a valid refresh token."""
    payload = decode_token(refresh_token_str)

    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")

    # Check if refresh token has been revoked
    old_jti = payload.get("jti")
    if old_jti:
        result = await db.execute(
            select(RevokedToken).where(RevokedToken.jti == old_jti)
        )
        if result.scalar_one_or_none():
            raise ValueError("Refresh token has been revoked")

    user_id = UUID(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise ValueError("User not found or disabled")

    # Revoke the old refresh token
    if old_jti:
        exp = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else datetime.now(timezone.utc)
        revoked = RevokedToken(
            jti=old_jti,
            user_id=user_id,
            token_type="refresh",
            expires_at=expires_at,
        )
        db.add(revoked)
        await db.commit()

    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    """Fetch a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
