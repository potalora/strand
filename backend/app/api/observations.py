from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.record import HealthRecord
from app.services.utils.source_label import source_label
from app.services.utils.unit_normalization import normalize_value

router = APIRouter(prefix="/observations", tags=["observations"])


def _is_number(value: object) -> bool:
    """True for real numbers (ints/floats) but not bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _extract_reading(record: HealthRecord) -> dict:
    """Pull the displayable reading from a record's fhir_resource JSONB.

    Mirrors /dashboard/labs reference-range/interpretation extraction. ``value``
    is the numeric valueQuantity when present, else a string (component BP like
    "128/78", a valueString, or the display text). Non-numeric values are kept
    verbatim and excluded from the series upstream.
    """
    fhir = record.fhir_resource or {}
    value_qty = fhir.get("valueQuantity") or {}
    raw_value = value_qty.get("value")
    unit = value_qty.get("unit", "")

    if _is_number(raw_value):
        value: object = raw_value
    else:
        value = _string_value(record, fhir)
        unit = ""

    ref_range_list = fhir.get("referenceRange") or [{}]
    ref = ref_range_list[0] if ref_range_list else {}
    ref_low = (ref.get("low") or {}).get("value")
    ref_high = (ref.get("high") or {}).get("value")

    interp_code = ""
    interp = fhir.get("interpretation") or [{}]
    if interp and interp[0].get("coding"):
        interp_code = interp[0]["coding"][0].get("code", "")

    return {
        "id": str(record.id),
        "value": value,
        "unit": unit,
        "date": record.effective_date.isoformat() if record.effective_date else None,
        "source": source_label(record.source_format, record.source_system),
        "ref_low": ref_low,
        "ref_high": ref_high,
        "interpretation": interp_code,
    }


def _string_value(record: HealthRecord, fhir: dict) -> str:
    """Best-effort string value for non-numeric observations (e.g. BP)."""
    value_string = fhir.get("valueString")
    if value_string:
        return value_string

    # Component-based blood pressure -> "systolic/diastolic"
    components = fhir.get("component") or []
    parts: list[str] = []
    for comp in components:
        cval = (comp.get("valueQuantity") or {}).get("value")
        if _is_number(cval):
            parts.append(str(int(cval) if float(cval).is_integer() else cval))
    if len(parts) >= 2:
        return "/".join(parts[:2])

    value_cc = fhir.get("valueCodeableConcept") or {}
    if value_cc.get("text"):
        return value_cc["text"]
    for coding in value_cc.get("coding") or []:
        disp = coding.get("display") or coding.get("code")
        if disp:
            return disp

    return record.display_text or ""


def _category_label(record: HealthRecord) -> str | None:
    """Prefer source_section, else first element of the category array, else None."""
    if record.source_section:
        return record.source_section
    if record.category:
        return record.category[0]
    return None


@router.get("/by-code")
async def observations_by_code(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """One entry per distinct observation code, recency-sorted.

    Sorted by the latest reading's date (desc); ties broken by higher reading
    count first. Each entry carries latest/prior readings plus a numeric,
    unit-normalized, ascending series for the sparkline/trend.
    """
    result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
            HealthRecord.record_type == "observation",
            HealthRecord.code_value.isnot(None),
        )
        .order_by(HealthRecord.effective_date.asc().nullsfirst())
    )
    records = result.scalars().all()

    # Group by code_value; records arrive ascending by date.
    groups: dict[str, list[HealthRecord]] = {}
    for r in records:
        groups.setdefault(r.code_value, []).append(r)

    items = []
    for code, recs in groups.items():
        # recs ascending by date; latest is last, prior is the one before.
        latest_rec = recs[-1]
        prior_rec = recs[-2] if len(recs) >= 2 else None

        latest = _extract_reading(latest_rec)
        prior = _extract_reading(prior_rec) if prior_rec else None

        # Numeric, unit-normalized series, ascending by date.
        series = []
        for r in recs:
            value_qty = (r.fhir_resource or {}).get("valueQuantity") or {}
            raw = value_qty.get("value")
            if not _is_number(raw):
                continue
            norm_value, _ = normalize_value(code, float(raw), value_qty.get("unit"))
            series.append(
                {
                    "date": r.effective_date.isoformat() if r.effective_date else None,
                    "value": norm_value,
                }
            )

        display = latest_rec.code_display or latest_rec.display_text
        latest_date: datetime | None = latest_rec.effective_date

        items.append(
            {
                "code": code,
                "display": display,
                "category": _category_label(latest_rec),
                "count": len(recs),
                "latest": latest,
                "prior": prior,
                "series": series,
                # sort keys, stripped before returning
                "_latest_ts": latest_date.timestamp() if latest_date else float("-inf"),
            }
        )

    # Recency desc; tie -> higher count first.
    items.sort(key=lambda it: (it["_latest_ts"], it["count"]), reverse=True)
    for it in items:
        it.pop("_latest_ts", None)

    await log_audit_event(
        db,
        user_id=user_id,
        action="observations.by_code",
        resource_type="health_record",
        ip_address=request.client.host if request.client else None,
        details={"code_count": len(items)},
    )

    return {"items": items, "total": len(items)}
