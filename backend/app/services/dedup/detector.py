from __future__ import annotations

import logging
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deduplication import DedupCandidate
from app.models.record import HealthRecord

logger = logging.getLogger(__name__)


async def detect_duplicates(
    db: AsyncSession,
    user_id: UUID,
    patient_id: UUID,
) -> int:
    """Scan for duplicate records and create dedup candidates.

    Uses hash-based bucketing to reduce comparisons from O(n^2) to
    bucket-scoped pairs, batch existence checks via an in-memory set,
    and bulk inserts for new candidates.

    Returns the number of new candidates found.
    """
    # Fetch all active records for patient
    result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.patient_id == patient_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
        .order_by(HealthRecord.effective_date.asc().nullslast())
    )
    records = result.scalars().all()

    if len(records) < 2:
        return 0

    # Pre-load all existing candidate pairs into a set (batch existence check)
    existing_result = await db.execute(
        select(DedupCandidate.record_a_id, DedupCandidate.record_b_id)
    )
    existing_pairs: set[tuple[UUID, UUID]] = set()
    for r in existing_result.all():
        existing_pairs.add((r[0], r[1]))
        existing_pairs.add((r[1], r[0]))  # both orderings

    # Group records by type + code/text key for bucket-based comparison
    buckets: dict[tuple, list[HealthRecord]] = {}
    for r in records:
        key = (r.record_type, (r.code_value or (r.display_text or "")[:50].lower()))
        buckets.setdefault(key, []).append(r)

    new_candidates: list[dict] = []

    for key, bucket in buckets.items():
        if len(bucket) < 2:
            continue
        for i, a in enumerate(bucket):
            for b in bucket[i + 1 :]:
                score, reasons = _compare_records(a, b)
                if score >= 0.7:
                    if (a.id, b.id) in existing_pairs:
                        continue
                    new_candidates.append({
                        "id": uuid4(),
                        "record_a_id": a.id,
                        "record_b_id": b.id,
                        "similarity_score": score,
                        "match_reasons": reasons,
                        "status": "pending",
                    })
                    # Add to existing_pairs to prevent duplicate inserts within same run
                    existing_pairs.add((a.id, b.id))
                    existing_pairs.add((b.id, a.id))

    if new_candidates:
        from sqlalchemy import insert

        # Insert in batches of 100
        for i in range(0, len(new_candidates), 100):
            batch = new_candidates[i : i + 100]
            await db.execute(insert(DedupCandidate), batch)
        await db.commit()

    candidates_found = len(new_candidates)
    logger.info("Found %d dedup candidates for patient %s", candidates_found, patient_id)
    return candidates_found


async def detect_upload_duplicates(
    db: AsyncSession,
    upload_id: UUID,
    patient_id: UUID,
    user_id: UUID,
) -> tuple[list[dict], list[dict]]:
    """Detect duplicates scoped to a specific upload.

    Compares records from this upload against all other records for the patient.
    Returns (auto_merged, needs_llm_review) — two lists of candidate dicts.
    auto_merged: score >= 0.95
    needs_llm_review: score 0.6–0.95
    silently dropped: score 0.5–0.6 (too low for LLM judge)
    """
    # Fetch records from this upload
    new_result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.patient_id == patient_id,
            HealthRecord.source_file_id == upload_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
    )
    new_records = new_result.scalars().all()

    if not new_records:
        return [], []

    # Fetch existing records (from other uploads)
    existing_result = await db.execute(
        select(HealthRecord)
        .where(
            HealthRecord.user_id == user_id,
            HealthRecord.patient_id == patient_id,
            HealthRecord.source_file_id != upload_id,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
    )
    existing_records = existing_result.scalars().all()

    if not existing_records:
        return [], []

    # Pre-load existing candidate pairs
    existing_pairs_result = await db.execute(
        select(DedupCandidate.record_a_id, DedupCandidate.record_b_id)
    )
    existing_pairs: set[tuple[UUID, UUID]] = set()
    for r in existing_pairs_result.all():
        existing_pairs.add((r[0], r[1]))
        existing_pairs.add((r[1], r[0]))

    # Build lookup by (record_type, code/text) for existing records
    existing_buckets: dict[tuple, list[HealthRecord]] = {}
    for r in existing_records:
        key = (r.record_type, (r.code_value or (r.display_text or "")[:50].lower()))
        existing_buckets.setdefault(key, []).append(r)

    auto_merged: list[dict] = []
    needs_llm_review: list[dict] = []

    for new_rec in new_records:
        key = (new_rec.record_type, (new_rec.code_value or (new_rec.display_text or "")[:50].lower()))
        bucket = existing_buckets.get(key, [])

        for existing_rec in bucket:
            if (new_rec.id, existing_rec.id) in existing_pairs:
                continue

            score, reasons = _compare_records(new_rec, existing_rec)
            if score < 0.5:
                continue

            candidate = {
                "id": uuid4(),
                "record_a_id": existing_rec.id,
                "record_b_id": new_rec.id,
                "similarity_score": score,
                "match_reasons": reasons,
                "status": "pending",
                "source_upload_id": upload_id,
            }

            if score >= 0.95:
                auto_merged.append(candidate)
            elif score >= 0.6:
                needs_llm_review.append(candidate)
            # else: score 0.5-0.6, auto-dismissed (too low for LLM judge)

            existing_pairs.add((new_rec.id, existing_rec.id))
            existing_pairs.add((existing_rec.id, new_rec.id))

    logger.info(
        "Upload %s: %d auto-merged, %d need LLM review",
        upload_id, len(auto_merged), len(needs_llm_review),
    )
    return auto_merged, needs_llm_review


def _compare_records(a: HealthRecord, b: HealthRecord) -> tuple[float, dict]:
    """Compare two records for similarity.

    Returns (score, reasons) where score is 0-1.
    """
    score = 0.0
    reasons = {}

    # Same code = strong match
    if a.code_value and b.code_value and a.code_value == b.code_value:
        score += 0.4
        reasons["code_match"] = True

    # Same display text
    if a.display_text and b.display_text:
        if a.display_text.lower() == b.display_text.lower():
            score += 0.3
            reasons["text_exact_match"] = True
        elif _fuzzy_match(a.display_text, b.display_text) > 0.8:
            score += 0.2
            reasons["text_fuzzy_match"] = True

    # Date relationship. Same-day records strongly support a duplicate. The
    # same code/text on dates more than a week apart are far more likely to be
    # distinct time-series events — repeat labs/vitals, annual immunizations,
    # separate encounters — than a true duplicate, so penalize them below the
    # review threshold instead of flooding the queue. A few days apart is
    # ambiguous (same event with differing cross-system timestamps) and stays
    # neutral. A missing date on either side is not comparable — no adjustment.
    if a.effective_date and b.effective_date:
        delta_days = abs((a.effective_date - b.effective_date).total_seconds()) / 86400.0
        if delta_days < 1:
            score += 0.2
            reasons["date_proximity"] = True
        elif delta_days > 7:
            score -= 0.35
            reasons["date_distant"] = True

    # Same status
    if a.status and b.status and a.status == b.status:
        score += 0.1
        reasons["status_match"] = True

    # Cross-source is a strong signal
    if a.source_format != b.source_format:
        score += 0.1
        reasons["cross_source"] = True

    # Source section match bonus
    if (
        getattr(a, "source_section", None)
        and getattr(b, "source_section", None)
        and a.source_section == b.source_section
    ):
        score += 0.15
        reasons["section_match"] = True

    return max(0.0, min(score, 1.0)), reasons


def _fuzzy_match(a: str, b: str) -> float:
    """Simple fuzzy string matching using character overlap."""
    a_lower = a.lower()
    b_lower = b.lower()
    if a_lower == b_lower:
        return 1.0

    # Use set intersection for quick similarity
    set_a = set(a_lower.split())
    set_b = set(b_lower.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0
