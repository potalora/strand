"""Identity gate for incremental ingestion.

`plan_batch` is a pure classifier: given a map of existing identities and a
batch of incoming records, decide insert / update / skip for each. The DB-bound
`idempotent_insert_records` (added in a later task) executes the plan.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from app.models.record_version import RecordVersion
from app.services.ingestion.content_hash import content_hash
from app.services.ingestion.identity import Identity, extract_identity

logger = logging.getLogger(__name__)

# existing identity -> (row_id, content_hash, version)
ExistingMap = dict[tuple[str, str], tuple[Any, str | None, int]]


@dataclass
class Plan:
    record: dict[str, Any]
    action: str  # insert | update | skip | update_pending
    identity: Identity | None
    content_hash: str | None
    existing_id: Any = None
    new_version: int = 1


def plan_batch(records: list[dict[str, Any]], existing: ExistingMap) -> list[Plan]:
    """Classify each record. `existing` maps identity-key -> (id, hash, version)."""
    plans: list[Plan] = []
    seen_in_batch: dict[tuple[str, str], int] = {}  # key -> index into plans

    for rec in records:
        ident = extract_identity(rec)
        resource = rec.get("fhir_resource") or {}
        chash = content_hash(resource) if isinstance(resource, dict) and resource else None

        if ident is None:
            plans.append(Plan(rec, "insert", None, chash))
            continue

        key = (ident.source_system, ident.external_id)

        # Within-batch duplicate identity.
        if key in seen_in_batch:
            prior_idx = seen_in_batch[key]
            if plans[prior_idx].content_hash == chash:
                plans.append(Plan(rec, "skip", ident, chash))
            else:
                plans.append(Plan(rec, "update_pending", ident, chash, existing_id=prior_idx))
            continue

        seen_in_batch[key] = len(plans)

        if key in existing:
            row_id, old_hash, old_version = existing[key]
            if old_hash == chash:
                plans.append(Plan(rec, "skip", ident, chash, existing_id=row_id, new_version=old_version))
            else:
                plans.append(
                    Plan(rec, "update", ident, chash, existing_id=row_id, new_version=old_version + 1)
                )
        else:
            plans.append(Plan(rec, "insert", ident, chash))

    return plans


async def _load_existing(db: AsyncSession, user_id: Any, identities: list[Identity]) -> ExistingMap:
    if not identities:
        return {}
    keys = list({(i.source_system, i.external_id) for i in identities})
    result = await db.execute(
        select(
            HealthRecord.id,
            HealthRecord.source_system,
            HealthRecord.external_id,
            HealthRecord.content_hash,
            HealthRecord.version,
        ).where(
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
            tuple_(HealthRecord.source_system, HealthRecord.external_id).in_(keys),
        )
    )
    return {
        (r.source_system, r.external_id): (r.id, r.content_hash, r.version) for r in result.all()
    }


def _build_row(rec: dict[str, Any], ident: Identity | None, chash: str | None) -> HealthRecord:
    return HealthRecord(
        id=uuid.uuid4(),
        patient_id=rec["patient_id"],
        user_id=rec["user_id"],
        record_type=rec["record_type"],
        fhir_resource_type=rec["fhir_resource_type"],
        fhir_resource=rec["fhir_resource"],
        source_format=rec["source_format"],
        source_file_id=rec.get("source_file_id"),
        effective_date=rec.get("effective_date"),
        effective_date_end=rec.get("effective_date_end"),
        status=rec.get("status"),
        category=rec.get("category"),
        code_system=rec.get("code_system"),
        code_value=rec.get("code_value"),
        code_display=rec.get("code_display"),
        display_text=rec["display_text"],
        confidence_score=rec.get("confidence_score"),
        ai_extracted=rec.get("ai_extracted", False),
        external_id=ident.external_id if ident else None,
        source_system=ident.source_system if ident else None,
        content_hash=chash,
        version=1,
    )


async def idempotent_insert_records(db: AsyncSession, records: list[dict[str, Any]]) -> dict:
    """Insert new records, update changed ones (snapshotting prior versions),
    skip identical ones. Returns counts + the list of newly inserted record dicts."""
    if not records:
        return {"inserted": 0, "updated": 0, "unchanged": 0, "inserted_records": []}

    user_id = records[0]["user_id"]
    idents = [i for i in (extract_identity(r) for r in records) if i is not None]
    existing = await _load_existing(db, user_id, idents)
    plans = plan_batch(records, existing)

    inserted = updated = unchanged = 0
    inserted_records: list[dict] = []
    pending_rows: dict[int, HealthRecord] = {}  # plan index -> ORM row (for within-batch updates)

    for idx, p in enumerate(plans):
        if p.action == "insert":
            row = _build_row(p.record, p.identity, p.content_hash)
            db.add(row)
            pending_rows[idx] = row
            inserted += 1
            inserted_records.append(p.record)
        elif p.action == "skip":
            unchanged += 1
        elif p.action == "update_pending":
            target = pending_rows.get(p.existing_id)
            if target is not None:
                _snapshot(db, target)
                _apply_update(target, p)
                updated += 1
                inserted -= 1  # the insert it supersedes no longer counts as a fresh insert
        elif p.action == "update":
            row = await db.get(HealthRecord, p.existing_id)
            if row is not None:
                _snapshot(db, row)
                _apply_update(row, p)
                updated += 1

    return {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "inserted_records": inserted_records,
    }


def _snapshot(db: AsyncSession, row: HealthRecord) -> None:
    db.add(
        RecordVersion(
            id=uuid.uuid4(),
            record_id=row.id,
            version=row.version,
            fhir_resource=row.fhir_resource,
            content_hash=row.content_hash,
            changed_fields=None,
            source_file_id=row.source_file_id,
        )
    )


def _apply_update(row: HealthRecord, p: Plan) -> None:
    rec = p.record
    row.fhir_resource = rec["fhir_resource"]
    row.content_hash = p.content_hash
    row.version = p.new_version
    row.status = rec.get("status", row.status)
    row.effective_date = rec.get("effective_date", row.effective_date)
    row.display_text = rec.get("display_text", row.display_text)
    row.source_file_id = rec.get("source_file_id", row.source_file_id)
