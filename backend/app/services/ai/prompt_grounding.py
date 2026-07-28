"""Grounded prompt-only packages and server-side paste validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import ExtractionEvidence
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.services.ai.grounded_routing import (
    build_deidentified_grounded_transport,
    compose_grounded_routed_system_prompt,
    compose_grounded_routed_user_prompt,
    translate_grounded_transport_response,
    validate_grounded_provider_payload,
)
from app.services.ai.patient_phi import patient_scrub_args
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.grounded_summary import (
    MAX_EVIDENCE,
    MAX_FACTS,
    GroundedSummaryInput,
    RenderedGroundedSummary,
    build_grounded_summary_input,
    validate_and_render_summary,
)
from app.services.local_ai.summary_projection import (
    normalize_summary_record_type,
    project_summary_records,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_FORMATS = frozenset({"natural_language", "json", "both"})


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grounded_input_sha256(summary_input: GroundedSummaryInput) -> str:
    """Return the immutable digest used to reject stale pasted responses."""
    return _canonical_sha256(summary_input.model_dump(mode="json"))


def _validate_prompt_filters(
    *,
    summary_type: str,
    category: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    record_ids: Sequence[UUID] | None,
    record_types: Sequence[str] | None,
) -> None:
    has_dates = date_from is not None or date_to is not None
    if summary_type == "category" and not (category and category != "full"):
        raise ValueError("Category is required for a category summary")
    if summary_type == "date_range" and (date_from is None or date_to is None):
        raise ValueError("Both dates are required for a date range summary")
    if summary_type not in {
        "full",
        "full_health",
        "category",
        "date_range",
        "single_record",
    }:
        raise ValueError("Unsupported prompt-only summary type")
    if has_dates and (date_from is None or date_to is None):
        raise ValueError("Both dates are required for a date range summary")
    selectors = sum(
        (
            bool(category and category != "full"),
            bool(record_ids),
            bool(record_types),
        )
    )
    if selectors > 1:
        raise ValueError("Choose only one record selector for a grounded prompt")
    if has_dates and selectors:
        raise ValueError(
            "Date ranges cannot be combined with another grounded prompt selector"
        )
    if record_ids and len(record_ids) != len(set(record_ids)):
        raise ValueError("Grounded prompt record identifiers must be unique")


def _grounded_scope(
    *,
    summary_type: str,
    category: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    record_ids: Sequence[UUID] | None,
) -> dict[str, object]:
    if category and category != "full":
        return {
            "summary_type": "category",
            "category": normalize_summary_record_type(category),
        }
    if date_from is not None and date_to is not None:
        return {
            "summary_type": "date_range",
            "date_from": date_from.date().isoformat(),
            "date_to": date_to.date().isoformat(),
        }
    if record_ids is not None and len(record_ids) == 1:
        return {
            "summary_type": "single_record",
            "record_ids": [str(record_ids[0])],
        }
    if summary_type not in {"full", "full_health"}:
        raise ValueError("Unsupported prompt-only summary type")
    return {"summary_type": "full_health"}


async def _fetch_selected_records(
    db: AsyncSession,
    *,
    user_id: UUID,
    patient_id: UUID,
    category: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    record_ids: Sequence[UUID] | None,
    record_types: Sequence[str] | None,
) -> list[HealthRecord]:
    query = select(HealthRecord).where(
        HealthRecord.user_id == user_id,
        HealthRecord.patient_id == patient_id,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    )
    if record_ids:
        query = query.where(HealthRecord.id.in_(record_ids))
    elif record_types:
        query = query.where(HealthRecord.record_type.in_(record_types))
    elif category and category != "full":
        query = query.where(HealthRecord.record_type == category)
    if date_from is not None:
        query = query.where(HealthRecord.effective_date >= date_from)
    if date_to is not None:
        query = query.where(HealthRecord.effective_date <= date_to)
    query = query.order_by(
        HealthRecord.effective_date.asc().nullslast(),
        HealthRecord.id.asc(),
    ).limit(MAX_FACTS + 1)
    records = list((await db.execute(query)).scalars().all())
    if len(records) > MAX_FACTS:
        raise ValueError("Too many records for a grounded prompt")
    if not records:
        raise ValueError("No records found matching the criteria")
    if record_ids and {record.id for record in records} != set(record_ids):
        raise ValueError("One or more selected records are unavailable")
    return records


async def _fetch_records_by_snapshot_ids(
    db: AsyncSession,
    *,
    user_id: UUID,
    patient_id: UUID,
    record_ids: Sequence[UUID],
) -> list[HealthRecord]:
    if not record_ids or len(record_ids) > MAX_FACTS:
        raise ValueError("Prompt grounding snapshot is invalid")
    records = list(
        (
            await db.execute(
                select(HealthRecord)
                .where(
                    HealthRecord.user_id == user_id,
                    HealthRecord.patient_id == patient_id,
                    HealthRecord.id.in_(record_ids),
                    HealthRecord.deleted_at.is_(None),
                    HealthRecord.is_duplicate.is_(False),
                )
                .order_by(
                    HealthRecord.effective_date.asc().nullslast(),
                    HealthRecord.id.asc(),
                )
                .limit(MAX_FACTS + 1)
            )
        )
        .scalars()
        .all()
    )
    if {record.id for record in records} != set(record_ids):
        raise ValueError("Prompt records changed; build a new prompt.")
    return records


async def _build_grounding_registry(
    db: AsyncSession,
    *,
    user_id: UUID,
    records: Sequence[HealthRecord],
    scope: Mapping[str, object],
) -> GroundedSummaryInput:
    evidence_rows = list(
        (
            await db.execute(
                select(ExtractionEvidence)
                .where(
                    ExtractionEvidence.user_id == user_id,
                    ExtractionEvidence.health_record_id.in_(
                        [record.id for record in records]
                    ),
                )
                .order_by(ExtractionEvidence.id.asc())
                .limit(MAX_EVIDENCE + 1)
            )
        )
        .scalars()
        .all()
    )
    if len(evidence_rows) > MAX_EVIDENCE:
        raise ValueError("Too much evidence for a grounded prompt")
    evidence_by_record: dict[UUID, list[ExtractionEvidence]] = {}
    for item in evidence_rows:
        if item.health_record_id is not None:
            evidence_by_record.setdefault(item.health_record_id, []).append(item)
    projection = project_summary_records(records, evidence_by_record)
    return build_grounded_summary_input(
        facts=projection.facts,
        evidence=projection.evidence,
        requested_scope=dict(scope),
        uncertainty_labels=projection.uncertainty_labels,
    )


async def build_grounded_prompt_package(
    db: AsyncSession,
    *,
    user_id: UUID,
    patient: Patient,
    summary_type: str,
    category: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    record_ids: Sequence[UUID] | None,
    record_types: Sequence[str] | None,
    output_format: str,
) -> dict[str, object]:
    """Build a de-identified reference-selection prompt and snapshot metadata."""
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError("Unsupported summary output format")
    _validate_prompt_filters(
        summary_type=summary_type,
        category=category,
        date_from=date_from,
        date_to=date_to,
        record_ids=record_ids,
        record_types=record_types,
    )
    records = await _fetch_selected_records(
        db,
        user_id=user_id,
        patient_id=patient.id,
        category=category,
        date_from=date_from,
        date_to=date_to,
        record_ids=record_ids,
        record_types=record_types,
    )
    scope = _grounded_scope(
        summary_type=summary_type,
        category=category,
        date_from=date_from,
        date_to=date_to,
        record_ids=record_ids,
    )
    summary_input = await _build_grounding_registry(
        db,
        user_id=user_id,
        records=records,
        scope=scope,
    )
    transport, _system_preference, _user_preference, report = (
        build_deidentified_grounded_transport(
            summary_input,
            scrub_args=patient_scrub_args(patient),
            custom_system_prompt=None,
            custom_user_prompt=None,
        )
    )
    system_prompt = compose_grounded_routed_system_prompt(None)
    user_prompt = compose_grounded_routed_user_prompt(transport, None)
    validate_grounded_provider_payload(system_prompt, user_prompt)
    return {
        "summary_type": summary_type,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "target_model": settings.prompt_target_model,
        "suggested_config": {
            "temperature": 0,
            "max_output_tokens": settings.prompt_suggested_max_tokens,
            "thinking_level": settings.prompt_suggested_thinking_level,
            "response_format": "json",
        },
        "record_count": len(records),
        "de_identification_report": report,
        "copyable_payload": f"System: {system_prompt}\n\nUser: {user_prompt}",
        "selected_record_ids": [str(record.id) for record in records],
        "grounded_scope": summary_input.requested_scope.model_dump(mode="json"),
        "grounding_input_sha256": grounded_input_sha256(summary_input),
    }


def _snapshot_record_ids(scope_filter: Mapping[str, object]) -> list[UUID]:
    raw_ids = scope_filter.get("selected_record_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or len(raw_ids) > MAX_FACTS
        or any(not isinstance(value, str) for value in raw_ids)
    ):
        raise ValueError("Prompt grounding snapshot is unavailable")
    try:
        record_ids = [UUID(value) for value in raw_ids]
    except (TypeError, ValueError):
        raise ValueError("Prompt grounding snapshot is unavailable") from None
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Prompt grounding snapshot is unavailable")
    return record_ids


async def rebuild_prompt_grounding_registry(
    db: AsyncSession,
    *,
    prompt: AISummaryPrompt,
    user_id: UUID,
) -> GroundedSummaryInput:
    """Rebuild and authenticate the exact registry used by a saved prompt."""
    scope_filter = prompt.scope_filter
    if prompt.processing_mode != "prompt_only" or not isinstance(scope_filter, dict):
        raise ValueError("Prompt grounding snapshot is unavailable")
    expected_digest = scope_filter.get("grounding_input_sha256")
    if (
        not isinstance(expected_digest, str)
        or _SHA256.fullmatch(expected_digest) is None
    ):
        raise ValueError("Prompt grounding snapshot is unavailable")
    record_ids = _snapshot_record_ids(scope_filter)
    records = await _fetch_records_by_snapshot_ids(
        db,
        user_id=user_id,
        patient_id=prompt.patient_id,
        record_ids=record_ids,
    )
    requested_scope = scope_filter.get("grounded_scope")
    if not isinstance(requested_scope, dict):
        raise ValueError("Prompt grounding snapshot is unavailable")
    try:
        summary_input = await _build_grounding_registry(
            db,
            user_id=user_id,
            records=records,
            scope=requested_scope,
        )
    except LocalValidationError:
        raise ValueError("Prompt records changed; build a new prompt.") from None
    if grounded_input_sha256(summary_input) != expected_digest:
        raise ValueError("Prompt records changed; build a new prompt.")
    return summary_input


async def validate_pasted_grounded_response(
    db: AsyncSession,
    *,
    prompt: AISummaryPrompt,
    user_id: UUID,
    raw_response: str,
) -> RenderedGroundedSummary:
    """Accept only exact reference JSON linked to the saved owner-scoped registry."""
    summary_input = await rebuild_prompt_grounding_registry(
        db,
        prompt=prompt,
        user_id=user_id,
    )
    translated_response = translate_grounded_transport_response(
        raw_response,
        summary_input,
    )
    return validate_and_render_summary(
        translated_response,
        facts={item.fact_id: item for item in summary_input.facts},
        evidence={item.evidence_id: item for item in summary_input.evidence},
        uncertainties={
            item.uncertainty_id: item for item in summary_input.uncertainty_labels
        },
    )
