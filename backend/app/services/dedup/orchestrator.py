from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.deduplication import DedupCandidate
from app.models.provenance import Provenance
from app.models.record import HealthRecord
from app.services.ai.llm import LLMConfig, load_llm_config
from app.services.dedup.detector import detect_upload_duplicates
from app.services.dedup.llm_judge import judge_candidates_batch, JudgmentResult
from app.services.local_ai.types import ProcessingMode

logger = logging.getLogger(__name__)


@dataclass
class DedupSummary:
    """Summary of dedup results for an upload."""

    total_candidates: int = 0
    auto_merged: int = 0
    needs_review: int = 0
    dismissed: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "auto_merged": self.auto_merged,
            "needs_review": self.needs_review,
            "dismissed": self.dismissed,
            "by_type": self.by_type,
        }


async def run_upload_dedup(
    upload_id: UUID,
    patient_id: UUID,
    user_id: UUID,
    db: AsyncSession,
    *,
    processing_mode: ProcessingMode | str = ProcessingMode.CLOUD_ASSISTED,
    commit: bool = True,
) -> DedupSummary:
    """Run the full dedup pipeline for a single upload.

    1. Heuristic filter: scoped comparison (new records vs existing)
    2. Auto-merge exact matches (score >= 0.95)
    3. Persist fuzzy non-cloud matches for manual review, or use the configured
       LLM judge only for cloud-assisted uploads (score 0.6–0.95)
    4. Auto-resolve based on LLM output
    5. Return summary
    """
    summary = DedupSummary()
    llm_enabled = processing_mode == ProcessingMode.CLOUD_ASSISTED

    # Step 1: Heuristic filter
    auto_merged_candidates, needs_llm_candidates = await detect_upload_duplicates(
        db, upload_id, patient_id, user_id
    )

    if not auto_merged_candidates and not needs_llm_candidates:
        return summary

    # Step 2: Auto-merge exact matches
    if auto_merged_candidates:
        for c in auto_merged_candidates:
            c["auto_resolved"] = True
            c["llm_classification"] = "duplicate"
            c["llm_confidence"] = 1.0
            c["llm_explanation"] = "Exact match (heuristic score >= 0.95)"
            c["status"] = "merged"
        await _save_candidates(db, auto_merged_candidates)
        await _apply_auto_merges(db, auto_merged_candidates, upload_id, user_id)
        summary.auto_merged = len(auto_merged_candidates)

    # Step 3: LLM judge on fuzzy matches
    if needs_llm_candidates:
        if not llm_enabled:
            for candidate in needs_llm_candidates:
                candidate["auto_resolved"] = False
                candidate["status"] = "pending"
            await _save_candidates(db, needs_llm_candidates)
            summary.needs_review = len(needs_llm_candidates)
        else:
            # Resolve routing only when an LLM judgment will actually run.
            config = await load_llm_config(db, user_id)
            judgments = await _run_llm_judge(db, needs_llm_candidates, config)

            llm_auto_merge = []
            llm_dismissed = []
            llm_needs_review = []

            for candidate, judgment in zip(needs_llm_candidates, judgments):
                candidate["llm_classification"] = judgment.classification
                candidate["llm_confidence"] = judgment.confidence
                candidate["llm_explanation"] = judgment.explanation
                candidate["field_diff"] = judgment.field_diff

                # Auto-resolution rules
                if (
                    judgment.classification == "duplicate"
                    and judgment.confidence >= 0.8
                ):
                    candidate["auto_resolved"] = True
                    candidate["status"] = "merged"
                    llm_auto_merge.append(candidate)
                elif (
                    judgment.classification == "distinct" and judgment.confidence >= 0.8
                ):
                    candidate["auto_resolved"] = True
                    candidate["status"] = "dismissed"
                    llm_dismissed.append(candidate)
                else:
                    candidate["auto_resolved"] = False
                    candidate["status"] = "pending"
                    llm_needs_review.append(candidate)

            # Save all LLM-judged candidates
            all_llm = llm_auto_merge + llm_dismissed + llm_needs_review
            await _save_candidates(db, all_llm)

            # Apply auto-merges from LLM duplicates
            if llm_auto_merge:
                await _apply_auto_merges(db, llm_auto_merge, upload_id, user_id)
                summary.auto_merged += len(llm_auto_merge)

            summary.dismissed = len(llm_dismissed)
            summary.needs_review = len(llm_needs_review)

    summary.total_candidates = (
        summary.auto_merged + summary.needs_review + summary.dismissed
    )

    # Build by_type counts from all candidates
    all_candidates = auto_merged_candidates + needs_llm_candidates
    for c in all_candidates:
        # Look up record type from record_a
        rec = await db.get(HealthRecord, c["record_a_id"])
        if rec:
            rtype = rec.record_type
            summary.by_type[rtype] = summary.by_type.get(rtype, 0) + 1

    if commit:
        await db.commit()
    else:
        await db.flush()
    return summary


async def _save_candidates(db: AsyncSession, candidates: list[dict]) -> None:
    """Bulk insert dedup candidates."""
    if not candidates:
        return
    for batch_start in range(0, len(candidates), 100):
        batch = candidates[batch_start : batch_start + 100]
        await db.execute(insert(DedupCandidate), batch)
    await db.flush()


async def _apply_auto_merges(
    db: AsyncSession,
    candidates: list[dict],
    upload_id: UUID,
    user_id: UUID,
) -> None:
    """Auto-merge: mark secondary records as duplicates, create provenance."""
    for c in candidates:
        # record_a is existing (primary), record_b is new (secondary)
        await db.execute(
            update(HealthRecord)
            .where(HealthRecord.id == c["record_b_id"])
            .values(
                is_duplicate=True,
                merged_into_id=c["record_a_id"],
                merge_metadata={
                    "merged_from": str(c["record_b_id"]),
                    "merged_at": datetime.now(timezone.utc).isoformat(),
                    "merge_type": "duplicate",
                    "source_upload_id": str(upload_id),
                    "auto_resolved": True,
                },
            )
        )
        # Create provenance
        db.add(
            Provenance(
                record_id=c["record_a_id"],
                action="merge",
                agent="system/auto-merge",
                source_file_id=upload_id,
                details={
                    "merged_record_id": str(c["record_b_id"]),
                    "similarity_score": c["similarity_score"],
                    "classification": c.get("llm_classification", "duplicate"),
                },
            )
        )
    await db.flush()


async def _run_llm_judge(
    db: AsyncSession,
    candidates: list[dict],
    config: LLMConfig | None = None,
) -> list[JudgmentResult]:
    """Fetch FHIR resources and run LLM judge on candidate pairs.

    ``config`` carries the per-user resolved routing/credentials, threaded into
    the batch judge (``None`` => global ``.env``).
    """
    pairs = []
    for c in candidates:
        rec_a = await db.get(HealthRecord, c["record_a_id"])
        rec_b = await db.get(HealthRecord, c["record_b_id"])
        if rec_a and rec_b:
            pairs.append(
                (
                    rec_a.fhir_resource or {},
                    rec_b.fhir_resource or {},
                    rec_a.record_type,
                )
            )
        else:
            pairs.append(({}, {}, "unknown"))

    return await judge_candidates_batch(pairs, settings.gemini_api_key, config=config)
