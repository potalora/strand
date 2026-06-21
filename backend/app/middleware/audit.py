from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.database import async_session_factory
from app.middleware.auth import decode_token
from app.models.audit import AuditLog

logger = logging.getLogger(__name__)


async def log_audit_event(
    db: AsyncSession,
    user_id: Optional[UUID],
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[UUID] = None,
    ip_address: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    """Log an audit event to the audit_log table."""
    try:
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            details=details,
        )
        db.add(entry)
        await db.commit()
    except Exception:
        logger.exception("Failed to write audit log entry")
        await db.rollback()


# ---------------------------------------------------------------------------
# Request-level audit safety net (W16 / HIPAA AUDIT-02/-03, SEC-AUTH-11)
# ---------------------------------------------------------------------------
# Several PHI-relevant endpoints (bulk Epic ingest, extracted-text reads,
# upload history/status/errors, /auth/me) write no per-endpoint audit row. The
# middleware below writes a generic ``api.access`` row for EVERY authenticated
# request under ``/api/v1`` so no PHI access can be silently un-logged. It is an
# additive safety net: endpoints with a specific log_audit_event call keep it
# and may carry both rows — that double-logging is acceptable.

# Public endpoints that legitimately take no Bearer access token. Skipped so the
# trail is not flooded with auth-handshake noise; the auth endpoints write their
# own targeted rows (login/register/refresh) where it matters.
_AUDIT_SKIP_PATHS = frozenset(
    {
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/health",
    }
)

_API_PREFIX = "/api/v1"


def _resolve_user_id(request: Request) -> Optional[UUID]:
    """Best-effort decode of the Bearer access token to a user UUID.

    A missing, malformed, expired, or non-access token yields ``None`` so the
    access attempt is still recorded (failed-auth visibility) rather than lost.
    """
    auth = request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    try:
        payload = decode_token(parts[1].strip())
    except Exception:
        return None
    sub = payload.get("sub")
    if not sub:
        return None
    try:
        return UUID(str(sub))
    except (ValueError, TypeError):
        return None


async def log_request_access(
    user_id: Optional[UUID],
    method: str,
    path: str,
    status_code: int,
    ip_address: Optional[str] = None,
) -> None:
    """Write a single ``api.access`` audit row using a FRESH session.

    A dedicated session (its own engine/connection) is used — never the
    request's session — so a rollback elsewhere in the request can't drop the
    audit row, and this commit can't poison the request's transaction. Only
    method/path/status go into ``details``; request bodies and query strings are
    deliberately excluded because they can carry PHI.
    """
    async with async_session_factory() as session:
        entry = AuditLog(
            user_id=user_id,
            action="api.access",
            ip_address=ip_address,
            details={"method": method, "path": path, "status": status_code},
        )
        session.add(entry)
        await session.commit()


class AuditMiddleware(BaseHTTPMiddleware):
    """Log every authenticated ``/api/v1`` request to ``audit_log``.

    Best-effort: a failure to write the audit row is caught and logged, never
    surfaced to the caller. CORS preflight (``OPTIONS``) and the public auth/
    health paths are skipped.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        try:
            if self._should_audit(request):
                user_id = _resolve_user_id(request)
                ip_address = request.client.host if request.client else None
                await log_request_access(
                    user_id=user_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    ip_address=ip_address,
                )
        except Exception:
            logger.exception("Request-level audit logging failed (request unaffected)")
        return response

    @staticmethod
    def _should_audit(request: Request) -> bool:
        if request.method == "OPTIONS":  # CORS preflight — no user context
            return False
        path = request.url.path
        if not path.startswith(_API_PREFIX):
            return False
        if path in _AUDIT_SKIP_PATHS:
            return False
        return True
