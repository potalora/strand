from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.middleware.encryption import decrypt_field
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview")
async def get_overview(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Dashboard summary data with real counts."""
    base_filter = [
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    ]

    # Total records
    total_result = await db.execute(
        select(func.count()).where(*base_filter)
    )
    total_records = total_result.scalar() or 0

    # Count by type
    type_result = await db.execute(
        select(HealthRecord.record_type, func.count())
        .where(*base_filter)
        .group_by(HealthRecord.record_type)
    )
    by_type = {row[0]: row[1] for row in type_result.all()}

    # Total patients
    patient_result = await db.execute(
        select(func.count()).where(Patient.user_id == user_id)
    )
    total_patients = patient_result.scalar() or 0

    # Recent records
    recent_result = await db.execute(
        select(HealthRecord)
        .where(*base_filter)
        .order_by(HealthRecord.created_at.desc())
        .limit(10)
    )
    recent = recent_result.scalars().all()
    recent_items = [
        {
            "id": str(r.id),
            "record_type": r.record_type,
            "display_text": r.display_text,
            "effective_date": r.effective_date.isoformat() if r.effective_date else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent
    ]

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

    # Upload count
    upload_result = await db.execute(
        select(func.count()).where(UploadedFile.user_id == user_id)
    )
    total_uploads = upload_result.scalar() or 0

    await log_audit_event(
        db,
        user_id=user_id,
        action="dashboard.overview",
        resource_type="dashboard",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "total_records": total_records,
        "total_patients": total_patients,
        "total_uploads": total_uploads,
        "records_by_type": by_type,
        "recent_records": recent_items,
        "date_range_start": date_row[0].isoformat() if date_row[0] else None,
        "date_range_end": date_row[1].isoformat() if date_row[1] else None,
    }


@router.get("/labs")
async def get_labs_dashboard(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Lab-specific dashboard data with observations (paginated)."""
    base_filter = [
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.record_type == "observation",
    ]

    # Count total
    count_result = await db.execute(
        select(func.count()).where(*base_filter)
    )
    total = count_result.scalar() or 0

    # Paginated fetch
    offset = (page - 1) * page_size
    result = await db.execute(
        select(HealthRecord)
        .where(*base_filter)
        .order_by(HealthRecord.effective_date.desc().nullslast())
        .offset(offset)
        .limit(page_size)
    )
    records = result.scalars().all()

    items = []
    for r in records:
        fhir = r.fhir_resource or {}
        value_qty = fhir.get("valueQuantity", {})
        value_str = fhir.get("valueString", "")
        ref_range = fhir.get("referenceRange", [{}])
        ref = ref_range[0] if ref_range else {}
        interp = fhir.get("interpretation", [{}])
        interp_code = ""
        if interp and interp[0].get("coding"):
            interp_code = interp[0]["coding"][0].get("code", "")

        items.append({
            "id": str(r.id),
            "display_text": r.display_text,
            "effective_date": r.effective_date.isoformat() if r.effective_date else None,
            "value": value_qty.get("value") if value_qty else value_str,
            "unit": value_qty.get("unit", "") if value_qty else "",
            "reference_low": ref.get("low", {}).get("value"),
            "reference_high": ref.get("high", {}).get("value"),
            "interpretation": interp_code,
            "code_display": r.code_display,
            "code_value": r.code_value,
        })

    await log_audit_event(
        db,
        user_id=user_id,
        action="dashboard.labs",
        resource_type="dashboard",
        ip_address=request.client.host if request.client else None,
    )

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/sources")
async def get_sources(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Provenance breakdown: how many records came from each source.

    Powers the Overview "Where records come from" bars and the data-sources stat.
    """
    base_filter = [
        HealthRecord.user_id == user_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    ]
    result = await db.execute(
        select(HealthRecord.source_format, func.count())
        .where(*base_filter)
        .group_by(HealthRecord.source_format)
        .order_by(func.count().desc())
    )
    items = [{"source": row[0], "count": row[1]} for row in result.all()]

    await log_audit_event(
        db,
        user_id=user_id,
        action="dashboard.sources",
        resource_type="dashboard",
        ip_address=request.client.host if request.client else None,
    )

    return {"items": items, "total": len(items)}


@router.get("/patients")
async def get_patients(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """List patients belonging to the current user."""
    result = await db.execute(
        select(Patient).where(Patient.user_id == user_id)
    )
    patients = result.scalars().all()

    await log_audit_event(
        db,
        user_id=user_id,
        action="dashboard.patients",
        resource_type="patient",
        ip_address=request.client.host if request.client else None,
    )

    return {
        "items": [
            {
                "id": str(p.id),
                "fhir_id": p.fhir_id,
                "gender": p.gender,
                "name": _safe_decrypt(p.name_encrypted),
                "birth_date": _safe_decrypt(p.birth_date_encrypted),
            }
            for p in patients
        ],
        "total": len(patients),
    }


def _safe_decrypt(value: bytes | None) -> str | None:
    """Decrypt an encrypted PII field, returning None if absent or undecryptable.

    The record subject's own identity (name/DOB) powers the masthead + account
    card. It is user-scoped data the owner is entitled to see.
    """
    if not value:
        return None
    try:
        return decrypt_field(value)
    except Exception:
        return None
