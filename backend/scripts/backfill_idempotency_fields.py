"""Backfill external_id/source_system/content_hash for existing health_records.

Idempotent: safe to re-run. Only touches rows whose target columns are NULL,
so a second run is a clean no-op. Processes in batches to bound memory.

Run:
    cd backend && .venv/bin/python -m scripts.backfill_idempotency_fields
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.database import async_session_factory
from app.models.record import HealthRecord
from app.services.ingestion.content_hash import content_hash
from app.services.ingestion.identity import extract_identity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BATCH = 500


async def _load_existing_identities() -> set[tuple[str, str, str]]:
    """Pre-populate the seen set with identities already stored in the DB.

    This makes guard (b) cross-run safe: a second invocation won't try to
    assign an external_id that was already claimed by a different row on the
    first run, which would violate the partial unique index
    uq_health_records_identity.

    Uses (created_at, id) ordering to guarantee a stable cursor even when many
    rows share the same created_at timestamp (e.g., bulk-inserted batches).
    """
    seen: set[tuple[str, str, str]] = set()
    offset = 0
    async with async_session_factory() as db:
        while True:
            rows = (
                await db.execute(
                    select(
                        HealthRecord.user_id,
                        HealthRecord.source_system,
                        HealthRecord.external_id,
                    )
                    .where(
                        HealthRecord.deleted_at.is_(None),
                        HealthRecord.external_id.is_not(None),
                    )
                    .order_by(HealthRecord.created_at, HealthRecord.id)
                    .offset(offset)
                    .limit(BATCH)
                )
            ).all()
            if not rows:
                break
            for user_id, source_system, external_id in rows:
                seen.add((str(user_id), str(source_system), str(external_id)))
            offset += BATCH
    logger.info("pre-loaded %d existing identities into seen set", len(seen))
    return seen


async def _process_batch(
    offset: int,
    seen: set[tuple[str, str, str]],
) -> tuple[int, int, int, int]:
    """Process a single batch of rows in its own session.

    Returns (rows_in_batch, assigned_id_count, assigned_hash_count, skipped_dupes).

    Uses (created_at, id) ordering for stable offset pagination.
    Uses a fresh session per batch so the ORM identity map never accumulates
    stale state that could cause spurious UPDATE flushes.
    """
    async with async_session_factory() as db:
        rows = (
            await db.execute(
                select(HealthRecord)
                .where(HealthRecord.deleted_at.is_(None))
                .order_by(HealthRecord.created_at, HealthRecord.id)
                .offset(offset)
                .limit(BATCH)
            )
        ).scalars().all()

        if not rows:
            return 0, 0, 0, 0

        assigned_id = 0
        assigned_hash = 0
        skipped = 0

        for row in rows:
            # --- external_id / source_system ---
            # Guard (b): only assign if NULL, and only if this identity key
            # hasn't been claimed yet in the DB or earlier in this run.
            if not row.external_id:
                rec = {
                    "source_format": row.source_format,
                    "fhir_resource": row.fhir_resource,
                    "external_id": row.external_id,
                    "source_system": row.source_system,
                }
                ident = extract_identity(rec)
                if ident:
                    key = (str(row.user_id), ident.source_system, ident.external_id)
                    if key in seen:
                        # Identity already claimed — leave external_id NULL to
                        # avoid violating the partial unique index
                        # uq_health_records_identity.
                        logger.debug(
                            "skipping duplicate identity %s/%s for user %s",
                            ident.source_system,
                            ident.external_id,
                            row.user_id,
                        )
                        skipped += 1
                    else:
                        seen.add(key)
                        row.external_id = ident.external_id
                        row.source_system = ident.source_system
                        assigned_id += 1

            # --- content_hash ---
            # No uniqueness constraint; unconditionally populate if missing.
            if not row.content_hash and row.fhir_resource:
                row.content_hash = content_hash(row.fhir_resource)
                assigned_hash += 1

        await db.commit()
        return len(rows), assigned_id, assigned_hash, skipped


async def main() -> None:
    """Backfill idempotency columns on existing health_records rows."""
    # Pre-populate seen with identities already in the DB so guard (b) works
    # correctly across multiple runs without hitting the unique index.
    seen = await _load_existing_identities()

    offset = 0
    total_processed = 0
    total_assigned_id = 0
    total_assigned_hash = 0
    total_skipped_dupes = 0

    while True:
        rows_done, assigned_id, assigned_hash, skipped = await _process_batch(
            offset, seen
        )
        if rows_done == 0:
            break

        total_processed += rows_done
        total_assigned_id += assigned_id
        total_assigned_hash += assigned_hash
        total_skipped_dupes += skipped
        offset += BATCH

        logger.info(
            "batch offset=%d: processed=%d, assigned_id=%d, assigned_hash=%d, "
            "skipped_dupes=%d",
            offset - BATCH,
            rows_done,
            assigned_id,
            assigned_hash,
            skipped,
        )

    logger.info(
        "done; total rows processed=%d, total assigned_id=%d, "
        "total assigned_hash=%d, skipped_dupes=%d",
        total_processed,
        total_assigned_id,
        total_assigned_hash,
        total_skipped_dupes,
    )


if __name__ == "__main__":
    asyncio.run(main())
