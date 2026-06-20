from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.record import HealthRecord
from app.schemas.timeline import TimelineEvent, TimelineResponse, TimelineStats
from app.services.timeline_preview import build_timeline_preview
from app.services.timeline_service import extract_provider_display

router = APIRouter(prefix="/timeline", tags=["timeline"])


@router.get("", response_model=TimelineResponse)
async def get_timeline(
    request: Request,
    record_type: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> TimelineResponse:
    """Timeline data ordered by date, filterable by type."""
    filters = [
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
        HealthRecord.effective_date.isnot(None),
    ]

    if record_type:
        filters.append(HealthRecord.record_type == record_type)

    # Total count before limit
    count_query = select(func.count()).where(*filters)
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    # Fetch limited results
    query = (
        select(HealthRecord)
        .where(*filters)
        .order_by(HealthRecord.effective_date.desc())
        .limit(limit)
    )
    result = await db.execute(query)
    records = result.scalars().all()

    events = [
        TimelineEvent(
            id=r.id,
            record_type=r.record_type,
            display_text=r.display_text,
            effective_date=r.effective_date,
            code_display=r.code_display,
            category=r.category,
            provider=extract_provider_display(r.fhir_resource, r.record_type),
            preview=build_timeline_preview(r.fhir_resource, r.record_type),
        )
        for r in records
    ]

    await log_audit_event(
        db,
        user_id=user_id,
        action="timeline.view",
        resource_type="timeline",
        ip_address=request.client.host if request.client else None,
        details={"record_type": record_type, "total": total},
    )

    return TimelineResponse(events=events, total=total)


@router.get("/stats", response_model=TimelineStats)
async def get_timeline_stats(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> TimelineStats:
    """Aggregated stats for dashboard."""
    base = select(HealthRecord).where(
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    )

    # Total count
    total_result = await db.execute(
        select(func.count()).select_from(base.subquery())
    )
    total = total_result.scalar() or 0

    # Count by type
    type_result = await db.execute(
        select(HealthRecord.record_type, func.count())
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
        .group_by(HealthRecord.record_type)
    )
    records_by_type = {row[0]: row[1] for row in type_result.all()}

    # Date range
    date_result = await db.execute(
        select(
            func.min(HealthRecord.effective_date),
            func.max(HealthRecord.effective_date),
        ).where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.effective_date.isnot(None),
        )
    )
    date_row = date_result.one()

    await log_audit_event(
        db,
        user_id=user_id,
        action="timeline.stats",
        resource_type="timeline",
        ip_address=request.client.host if request.client else None,
    )

    return TimelineStats(
        total_records=total,
        records_by_type=records_by_type,
        date_range_start=date_row[0],
        date_range_end=date_row[1],
    )
