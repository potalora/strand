"""Helpers for idempotent unstructured (AI-extracted) ingestion.

- `find_prior_extracted_upload`: detect an identical file already extracted for a user.
- `soft_delete_prior_extracted`: replace a file's prior extracted records on re-extraction.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provenance import Provenance
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile

PRODUCED_RECORDS_STATUSES = (
    "completed",
    "completed_with_merges",
    "awaiting_review",
    "awaiting_confirmation",
)


async def find_prior_extracted_upload(
    db: AsyncSession, user_id: uuid.UUID, file_hash: str
) -> UploadedFile | None:
    """Return a prior upload for this user with the same file_hash that already
    produced records, or None (failed/pending priors do not count)."""
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.user_id == user_id,
            UploadedFile.file_hash == file_hash,
            UploadedFile.file_category == "unstructured",
            UploadedFile.ingestion_status.in_(PRODUCED_RECORDS_STATUSES),
        )
        .order_by(UploadedFile.created_at)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def soft_delete_prior_extracted(db: AsyncSession, source_file_id: uuid.UUID) -> int:
    """Soft-delete a file's prior live AI-extracted records (replace-on-reextract).

    Writes one provenance row per replaced record. Returns the count replaced.
    Never hard-deletes; structured (non-ai_extracted) records are left untouched.
    """
    rows = (await db.execute(
        select(HealthRecord).where(
            HealthRecord.source_file_id == source_file_id,
            HealthRecord.ai_extracted.is_(True),
            HealthRecord.deleted_at.is_(None),
        )
    )).scalars().all()

    now = datetime.now(timezone.utc)
    for row in rows:
        row.deleted_at = now
        db.add(Provenance(
            record_id=row.id,
            action="reextraction_replace",
            source_file_id=source_file_id,
            agent="extraction_worker",
            details={"reason": "re-extraction replaced prior extracted record"},
        ))
    return len(rows)
