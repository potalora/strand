"""Owner-scoped evidence traversal for deduplicated health-record survivors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.local_ai import ExtractionEvidence
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.types import ProcessingMode


@dataclass(frozen=True)
class EvidenceLineageRow:
    """One immutable evidence row resolved to its active survivor."""

    survivor_id: UUID
    evidence: ExtractionEvidence
    upload: UploadedFile


@dataclass(frozen=True)
class EvidenceLineageResult:
    """Bounded lineage evidence plus an explicit overflow signal."""

    rows: tuple[EvidenceLineageRow, ...]
    overflowed: bool

    def by_survivor(self) -> dict[UUID, list[ExtractionEvidence]]:
        grouped: dict[UUID, list[ExtractionEvidence]] = {}
        for row in self.rows:
            grouped.setdefault(row.survivor_id, []).append(row.evidence)
        return grouped


async def load_strict_local_evidence_lineage(
    db: AsyncSession,
    *,
    user_id: UUID,
    survivor_ids: Sequence[UUID],
    limit: int,
) -> EvidenceLineageResult:
    """Load strict-local evidence from survivors and owner-scoped archived descendants."""

    roots = tuple(dict.fromkeys(survivor_ids))
    if not roots or limit < 1:
        return EvidenceLineageResult(rows=(), overflowed=False)
    lineage = (
        select(
            HealthRecord.id.label("record_id"),
            HealthRecord.id.label("survivor_id"),
        )
        .where(
            HealthRecord.id.in_(roots),
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
        )
        .cte("strict_local_evidence_lineage", recursive=True)
    )
    archived = aliased(HealthRecord)
    lineage = lineage.union(
        select(
            archived.id.label("record_id"),
            lineage.c.survivor_id,
        ).where(
            archived.merged_into_id == lineage.c.record_id,
            archived.user_id == user_id,
            archived.deleted_at.is_(None),
            archived.is_duplicate.is_(True),
        )
    )
    selected = (
        await db.execute(
            select(
                ExtractionEvidence,
                UploadedFile,
                lineage.c.survivor_id,
            )
            .join(
                lineage,
                ExtractionEvidence.health_record_id == lineage.c.record_id,
            )
            .join(
                UploadedFile,
                and_(
                    UploadedFile.id == ExtractionEvidence.upload_id,
                    UploadedFile.user_id == ExtractionEvidence.user_id,
                ),
            )
            .where(
                ExtractionEvidence.user_id == user_id,
                UploadedFile.user_id == user_id,
                UploadedFile.deleted_at.is_(None),
                UploadedFile.processing_mode
                == ProcessingMode.VALIDATED_STRICT_LOCAL.value,
            )
            .order_by(
                lineage.c.survivor_id.asc(),
                ExtractionEvidence.page_number.asc().nullslast(),
                ExtractionEvidence.start_offset.asc().nullslast(),
                ExtractionEvidence.upload_id.asc(),
                ExtractionEvidence.id.asc(),
            )
            .limit(limit + 1)
        )
    ).all()
    overflowed = len(selected) > limit
    return EvidenceLineageResult(
        rows=tuple(
            EvidenceLineageRow(
                survivor_id=survivor_id,
                evidence=evidence,
                upload=upload,
            )
            for evidence, upload, survivor_id in selected[:limit]
        ),
        overflowed=overflowed,
    )
