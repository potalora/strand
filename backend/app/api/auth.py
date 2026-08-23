from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.api.auth_validation import ContentFreeAuthValidationRoute
from app.middleware.audit import log_audit_event
from app.middleware.auth import decode_token, security
from app.middleware.rate_limit import login_limiter, register_limiter
from app.models.token_blacklist import RevokedToken
from app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import (
    authenticate_user,
    get_user_by_id,
    refresh_tokens,
    register_user,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/auth", tags=["auth"], route_class=ContentFreeAuthValidationRoute
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    name="auth_register",
)
async def register(
    body: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a new user account."""
    client_ip = request.client.host if request.client else "unknown"
    if not register_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later.",
        )

    try:
        user = await register_user(
            db, body.login_identifier, body.password, body.display_name
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    await log_audit_event(
        db,
        user_id=user.id,
        action="user.register",
        resource_type="user",
        resource_id=user.id,
        ip_address=client_ip,
    )
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse, name="auth_login")
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate and receive JWT tokens."""
    client_ip = request.client.host if request.client else "unknown"
    if not login_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
        )

    try:
        tokens = await authenticate_user(db, body.login_identifier, body.password)
    except ValueError as e:
        detail = str(e)
        if "locked" not in detail.lower():
            detail = "Invalid account identifier or password"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

    await log_audit_event(
        db,
        user_id=None,
        action="user.login",
        ip_address=client_ip,
    )
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Refresh access token using a refresh token."""
    try:
        tokens = await refresh_tokens(db, body.refresh_token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    return tokens


def _stage_revocation(db: AsyncSession, payload: dict, user_id: UUID) -> None:
    """Add a token's JTI to the blacklist within the current session (no commit)."""
    jti = payload.get("jti")
    if not jti:
        return
    exp = payload.get("exp")
    expires_at = (
        datetime.fromtimestamp(exp, tz=timezone.utc)
        if exp
        else datetime.now(timezone.utc)
    )
    db.add(
        RevokedToken(
            jti=jti,
            user_id=user_id,
            token_type=payload.get("type", "access"),
            expires_at=expires_at,
        )
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(
    request: Request,
    body: LogoutRequest | None = None,
    user_id: UUID = Depends(get_authenticated_user_id),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Logout and revoke the current access token and, if supplied, the refresh token.

    The refresh token is only revoked when it belongs to the authenticated user, so
    a caller cannot revoke someone else's session. Revocation is best-effort: a bad
    token never fails the logout.
    """
    try:
        if credentials is not None:
            _stage_revocation(db, decode_token(credentials.credentials), user_id)
        if body is not None and body.refresh_token:
            refresh_payload = decode_token(body.refresh_token)
            if (
                refresh_payload.get("type") == "refresh"
                and refresh_payload.get("sub") == str(user_id)
            ):
                _stage_revocation(db, refresh_payload, user_id)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning("Failed to revoke token(s) on logout for user %s", user_id)

    await log_audit_event(
        db,
        user_id=user_id,
        action="user.logout",
        ip_address=request.client.host if request.client else None,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get current user profile."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserResponse.model_validate(user)
