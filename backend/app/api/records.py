from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.record import HealthRecord
from app.schemas.records import HealthRecordResponse, RecordListResponse
from app.services.utils.source_label import source_label

router = APIRouter(prefix="/records", tags=["records"])


# Server-side sort: maps the ?sort= key to a column. Anything else falls back
# to effective_date so the default ordering is unchanged.
_SORT_COLUMNS = {
    "date": HealthRecord.effective_date,
    "type": HealthRecord.record_type,
    "display_text": HealthRecord.display_text,
    "created": HealthRecord.created_at,
}


@router.get("", response_model=RecordListResponse)
async def list_records(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    record_type: str | None = None,
    category: str | None = None,
    search: str | None = None,
    status: str | None = None,
    sort: str | None = None,
    order: str = Query("desc", pattern="^(asc|desc)$"),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> RecordListResponse:
    """List health records with pagination, filtering, and sorting."""
    query = select(HealthRecord).where(
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    )

    if record_type:
        query = query.where(HealthRecord.record_type == record_type)
    if status:
        query = query.where(HealthRecord.status == status)
    if search:
        query = query.where(
            or_(
                HealthRecord.display_text.ilike(f"%{search}%"),
                HealthRecord.code_display.ilike(f"%{search}%"),
            )
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Fetch page — server-side sort with sensible nulls handling; default is
    # newest-first by effective date.
    sort_col = _SORT_COLUMNS.get(sort or "", HealthRecord.effective_date)
    direction = sort_col.asc() if order == "asc" else sort_col.desc()
    if sort_col is HealthRecord.effective_date:
        direction = direction.nullsfirst() if order == "asc" else direction.nullslast()
    # Append a stable tiebreaker on the primary key so the ordering is a TOTAL
    # order. Without it, large tie groups (shared/NULL effective_date, repeated
    # record_type) leave the row order undefined between page queries, so OFFSET
    # pagination can return the same row on two pages and silently drop another.
    query = query.order_by(direction, HealthRecord.id.asc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    records = result.scalars().all()

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.list",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"record_type": record_type, "search": search, "page": page, "total": total},
    )

    return RecordListResponse(
        items=[HealthRecordResponse.model_validate(r) for r in records],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/search")
async def search_records(
    request: Request,
    q: str = Query("", min_length=1),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Full-text search records."""
    query = (
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            or_(
                HealthRecord.display_text.ilike(f"%{q}%"),
                HealthRecord.code_display.ilike(f"%{q}%"),
            ),
        )
        .order_by(HealthRecord.effective_date.desc().nullslast())
        .limit(50)
    )
    result = await db.execute(query)
    records = result.scalars().all()

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.search",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"query": q, "result_count": len(records)},
    )

    return {
        "items": [HealthRecordResponse.model_validate(r) for r in records],
        "total": len(records),
    }


@router.get("/series")
async def record_series(
    request: Request,
    code_value: str = Query(..., min_length=1),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Time-series of one observation code (ascending by date) for trend charts.

    Declared before /{record_id} so the literal path isn't captured as a UUID.
    Only numeric valueQuantity points are returned.
    """
    result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
            HealthRecord.code_value == code_value,
        )
        .order_by(HealthRecord.effective_date.asc().nullslast())
    )
    records = result.scalars().all()

    items = []
    for r in records:
        value_qty = (r.fhir_resource or {}).get("valueQuantity") or {}
        value = value_qty.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        items.append(
            {
                "id": str(r.id),
                "effective_date": r.effective_date.isoformat() if r.effective_date else None,
                "value": value,
                "unit": value_qty.get("unit", ""),
            }
        )

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.series",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"code_value": code_value, "points": len(items)},
    )

    return {"code_value": code_value, "items": items, "total": len(items)}


@router.get("/export")
async def export_records(
    request: Request,
    format: str = Query("fhir-bundle"),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Bulk-export all of the user's records as a single FHIR R4 collection Bundle.

    Declared before /{record_id} so the literal path isn't captured as a UUID.
    """
    if format != "fhir-bundle":
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")

    result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
        .order_by(HealthRecord.effective_date.desc().nullslast())
    )
    records = result.scalars().all()

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "total": len(records),
        "entry": [{"resource": r.fhir_resource} for r in records],
    }

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.export",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"format": format, "count": len(records)},
    )

    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": 'attachment; filename="medtimeline-fhir-bundle.json"'},
    )


@router.get("/recent")
async def recent_records(
    request: Request,
    limit: int = Query(5, ge=1, le=50),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Most recently ingested records (by created_at), for the "Recently added" feed.

    Declared before /{record_id} so the literal path isn't captured as a UUID.
    """
    result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
        .order_by(HealthRecord.created_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()

    items = []
    for r in records:
        value_qty = (r.fhir_resource or {}).get("valueQuantity") or {}
        raw_value = value_qty.get("value")
        is_num = isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
        items.append(
            {
                "id": str(r.id),
                "record_type": r.record_type,
                "display_text": r.display_text,
                "effective_date": r.effective_date.isoformat() if r.effective_date else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "source": source_label(r.source_format, r.source_system),
                "value": raw_value if is_num else None,
                "unit": value_qty.get("unit") if is_num else None,
            }
        )

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.recent",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"limit": limit, "count": len(items)},
    )

    return {"items": items, "total": len(items)}


@router.get("/stats")
async def record_stats(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate stats for the masthead: total, date span, distinct source count.

    Declared before /{record_id} so the literal path isn't captured as a UUID.
    """
    base_filter = [
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    ]

    agg = await db.execute(
        select(
            func.count(),
            func.min(HealthRecord.effective_date),
            func.max(HealthRecord.effective_date),
            func.count(func.distinct(HealthRecord.source_format)),
        ).where(*base_filter)
    )
    total, first_date, last_date, source_count = agg.one()

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.stats",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"total": total or 0},
    )

    return {
        "total": total or 0,
        "first_date": first_date.isoformat() if first_date else None,
        "last_date": last_date.isoformat() if last_date else None,
        "source_count": source_count or 0,
    }


@router.get("/{record_id}", response_model=HealthRecordResponse)
async def get_record(
    record_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> HealthRecordResponse:
    """Get single record with FHIR resource."""
    result = await db.execute(
        select(HealthRecord).where(
            HealthRecord.id == record_id,
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.view",
        resource_type="health_record",
        resource_id=record_id,
        ip_address=request.client.host if request.client else None,
    )

    return HealthRecordResponse.model_validate(record)


@router.get("/{record_id}/fhir")
async def get_record_fhir(
    record_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return the record's raw FHIR resource as JSON, for single-record export."""
    result = await db.execute(
        select(HealthRecord).where(
            HealthRecord.id == record_id,
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.fhir_export",
        resource_type="health_record",
        resource_id=record_id,
        ip_address=request.client.host if request.client else None,
    )

    return JSONResponse(
        content=record.fhir_resource,
        headers={
            "Content-Disposition": f'attachment; filename="record-{record_id}.json"'
        },
    )


@router.delete("/{record_id}", status_code=204)
async def delete_record(
    record_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Soft delete a health record."""
    from datetime import datetime, timezone

    result = await db.execute(
        select(HealthRecord).where(
            HealthRecord.id == record_id,
            HealthRecord.user_id == user_id,
        )
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    record.deleted_at = datetime.now(timezone.utc)
    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="records.delete",
        resource_type="health_record",
        resource_id=record_id,
        ip_address=request.client.host if request.client else None,
    )
