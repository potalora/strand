from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.models.audit import AuditLog

router = APIRouter(prefix="/audit-log", tags=["audit"])


@router.get("")
async def list_audit_log(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Read this user's audit trail (most recent first).

    Reading the audit log is intentionally NOT itself audited — it is a benign,
    user-scoped read of the user's own activity, and self-logging would inflate
    and reorder the results on every fetch.
    """
    base = select(AuditLog).where(AuditLog.user_id == user_id)

    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar() or 0

    result = await db.execute(
        base.order_by(AuditLog.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    rows = result.scalars().all()

    return {
        "items": [
            {
                "id": str(r.id),
                "action": r.action,
                "resource_type": r.resource_type,
                "resource_id": str(r.resource_id) if r.resource_id else None,
                "ip_address": r.ip_address,
                "details": r.details,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "total": total,
        "page": page,
        "limit": limit,
    }
