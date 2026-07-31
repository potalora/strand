"""Helpers for idempotent unstructured (AI-extracted) ingestion.

- `find_prior_extracted_upload`: detect an identical file already extracted for a user.
- `soft_delete_prior_extracted`: replace a file's prior extracted records on re-extraction.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import or_, select
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
    db: AsyncSession,
    user_id: uuid.UUID,
    file_hash: str,
    *,
    processing_mode: str = "cloud_assisted",
    processing_manifest: dict | None = None,
    schema_version: str | None = None,
) -> UploadedFile | None:
    """Return a prior result only for the same bytes and processing revision."""
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.user_id == user_id,
            UploadedFile.file_hash == file_hash,
            UploadedFile.file_category == "unstructured",
            UploadedFile.ingestion_status.in_(PRODUCED_RECORDS_STATUSES),
            UploadedFile.processing_mode == processing_mode,
            UploadedFile.processing_manifest == processing_manifest,
            UploadedFile.processing_schema_version == schema_version,
        )
        .order_by(UploadedFile.created_at)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def soft_delete_prior_extracted(
    db: AsyncSession, source_file_id: uuid.UUID
) -> int:
    """Soft-delete a file's prior live AI-extracted records (replace-on-reextract).

    Writes one provenance row per replaced record. Returns the count replaced.
    Never hard-deletes; structured (non-ai_extracted) records are left untouched.
    """
    rows = (
        (
            await db.execute(
                select(HealthRecord).where(
                    HealthRecord.source_file_id == source_file_id,
                    HealthRecord.ai_extracted.is_(True),
                    HealthRecord.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )

    now = datetime.now(timezone.utc)
    for row in rows:
        row.deleted_at = now
        db.add(
            Provenance(
                record_id=row.id,
                action="reextraction_replace",
                source_file_id=source_file_id,
                agent="extraction_worker",
                details={"reason": "re-extraction replaced prior extracted record"},
            )
        )
    return len(rows)


async def soft_delete_lineage_extracted(
    db: AsyncSession,
    canonical_source_file_id: uuid.UUID,
) -> int:
    """Replace live AI records from the complete canonical reprocessing lineage."""
    source = await db.get(UploadedFile, canonical_source_file_id)
    if source is None:
        raise ValueError("Reprocessing source is invalid")
    lineage_ids = select(UploadedFile.id).where(
        UploadedFile.user_id == source.user_id,
        UploadedFile.file_category == "unstructured",
        or_(
            UploadedFile.id == canonical_source_file_id,
            UploadedFile.ingestion_progress["reprocesses_upload_id"].astext
            == str(canonical_source_file_id),
        ),
    )
    rows = (
        (
            await db.execute(
                select(HealthRecord).where(
                    HealthRecord.source_file_id.in_(lineage_ids),
                    HealthRecord.ai_extracted.is_(True),
                    HealthRecord.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    now = datetime.now(timezone.utc)
    for row in rows:
        row.deleted_at = now
        db.add(
            Provenance(
                record_id=row.id,
                action="reextraction_replace",
                source_file_id=row.source_file_id,
                agent="extraction_worker",
                details={"reason": "re-extraction replaced prior extracted record"},
            )
        )
    return len(rows)
