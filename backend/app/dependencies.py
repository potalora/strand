from __future__ import annotations

from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.middleware.auth import decode_token, get_current_user_id, security
from app.models.token_blacklist import RevokedToken


async def get_session(
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session dependency."""
    yield db


async def get_authenticated_user_id(
    user_id: Optional[UUID] = Depends(get_current_user_id),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """Require authentication, check token revocation, and return the current user's ID.

    Revocation is checked on the same credentials ``HTTPBearer`` already parsed
    (case-insensitive scheme), not by re-parsing the raw header — so a ``bearer``
    (lowercase) request cannot slip a revoked token past the blacklist.
    """
    if not user_id or credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_token(credentials.credentials)
    jti = payload.get("jti")
    if jti:
        result = await db.execute(select(RevokedToken).where(RevokedToken.jti == jti))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

    return user_id


def is_local_ai_operator(user_id: UUID) -> bool:
    """Return whether this authenticated user can manage the machine-global pack."""
    return user_id in settings.local_ai_operator_ids


async def require_local_ai_operator(
    user_id: UUID = Depends(get_authenticated_user_id),
) -> UUID:
    """Require configured machine-operator authority for pack management."""
    if not is_local_ai_operator(user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local model pack management requires a machine operator.",
        )
    return user_id
