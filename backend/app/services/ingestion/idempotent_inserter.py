"""Identity gate for incremental ingestion.

`plan_batch` is a pure classifier: given a map of existing identities and a
batch of incoming records, decide insert / update / skip for each. The DB-bound
`idempotent_insert_records` (added in a later task) executes the plan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

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
