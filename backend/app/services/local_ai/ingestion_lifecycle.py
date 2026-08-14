"""Atomic lifecycle transitions for one strict-local ingestion pair."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.local_ai import LocalAIJob
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from app.services.local_ai.processing_snapshot import (
    fail_legacy_runtime_identity_required,
)

_CANCELLABLE_PAIRS = frozenset(
    {
        ("queued", "pending_extraction"),
        ("processing", "processing"),
    }
)
_STRICT_LOCAL_MODEL_ROLES = frozenset({"ocr", "extraction", "summary"})
_STRICT_LOCAL_PROGRESS_COUNTERS = frozenset(
    {
        "page_index",
        "page_total",
        "worker_current",
        "worker_total",
        "current",
        "total",
        "activity",
        "attempt",
        "attempt_limit",
        "input_tokens",
        "output_tokens",
        "output_token_limit",
        "splits_used",
        "split_limit",
        "active_memory_bytes",
        "peak_memory_bytes",
    }
)
_STRICT_LOCAL_MAX_PROGRESS_COUNTER = 2**63 - 1
_STRICT_INGESTION_SCHEMA = "clinical-document-extraction.v1"


@dataclass(frozen=True)
class StrictIngestionCancellation:
    job: LocalAIJob
    upload: UploadedFile
    worker_cancel_required: bool


@dataclass(frozen=True)
class StrictIngestionClaim:
    """One committed strict pair claimed by this worker attempt."""

    upload_id: UUID
    job_id: UUID
    storage_path: str
    user_id: UUID
    claimed_at: datetime


async def claim_next_strict_ingestion_pair(
    db: AsyncSession,
) -> StrictIngestionClaim | None:
    upload = (
        await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.ingestion_status == "pending_extraction",
                UploadedFile.file_category == "unstructured",
                UploadedFile.manual_extraction_required.is_(False),
                UploadedFile.processing_mode == "validated_strict_local",
            )
            .order_by(UploadedFile.created_at, UploadedFile.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if upload is None:
        return None
    job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.upload_id == upload.id,
                LocalAIJob.user_id == upload.user_id,
                LocalAIJob.kind == "ingestion",
                LocalAIJob.processing_mode == "validated_strict_local",
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if job is None:
        return None
    try:
        _require_matching_strict_ingestion_identity(upload, job)
    except Exception:
        return None
    if (job.status, upload.ingestion_status) != (
        "queued",
        "pending_extraction",
    ) or job.cancel_requested or upload.cancel_requested:
        return None

    claimed_at = datetime.now(timezone.utc)
    upload.ingestion_status = "processing"
    upload.processing_started_at = claimed_at
    upload.progress_stage = "local_preflight"
    upload.progress_detail = None
    job.status = "processing"
    job.stage = "preflight"
    job.started_at = claimed_at
    job.failure = None
    return StrictIngestionClaim(
        upload_id=upload.id,
        job_id=job.id,
        storage_path=upload.storage_path,
        user_id=upload.user_id,
        claimed_at=claimed_at,
    )


async def compensate_unstarted_strict_ingestion_claim(
    db: AsyncSession,
    claim: StrictIngestionClaim,
) -> bool:
    upload = (
        await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.id == claim.upload_id,
                UploadedFile.user_id == claim.user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if upload is None:
        return False
    job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.id == claim.job_id,
                LocalAIJob.upload_id == claim.upload_id,
                LocalAIJob.user_id == claim.user_id,
                LocalAIJob.kind == "ingestion",
                LocalAIJob.processing_mode == "validated_strict_local",
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if job is None:
        return False
    try:
        _require_matching_strict_ingestion_identity(upload, job)
    except Exception:
        return False
    if (
        claim.storage_path != upload.storage_path
        or (job.status, upload.ingestion_status) != ("processing", "processing")
        or job.started_at != claim.claimed_at
        or upload.processing_started_at != claim.claimed_at
    ):
        return False

    if job.cancel_requested or upload.cancel_requested:
        cancelled_at = datetime.now(timezone.utc)
        job.cancel_requested = True
        upload.cancel_requested = True
        job.status = "cancelled"
        job.stage = "cancelled"
        job.progress = _cancelled_progress(job.progress)
        job.failure = None
        job.completed_at = cancelled_at
        upload.ingestion_status = "cancelled"
        upload.progress_stage = None
        upload.progress_detail = None
        upload.processing_completed_at = cancelled_at
        return True

    upload.ingestion_status = "pending_extraction"
    upload.processing_started_at = None
    upload.progress_stage = None
    upload.progress_detail = None
    job.status = "queued"
    job.stage = "queued"
    job.started_at = None
    job.failure = None
    return True


async def lock_active_strict_ingestion_claim(
    db: AsyncSession,
    claim: StrictIngestionClaim,
) -> tuple[UploadedFile, LocalAIJob] | None:
    """Fresh-lock and validate the exact committed pair for one child task."""
    upload = (
        await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.id == claim.upload_id,
                UploadedFile.user_id == claim.user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if upload is None:
        return None
    job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.id == claim.job_id,
                LocalAIJob.upload_id == claim.upload_id,
                LocalAIJob.user_id == claim.user_id,
                LocalAIJob.kind == "ingestion",
                LocalAIJob.processing_mode == "validated_strict_local",
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if job is None:
        return None
    try:
        _require_matching_strict_ingestion_identity(upload, job)
    except Exception:
        return None
    if (
        claim.storage_path != upload.storage_path
        or (job.status, upload.ingestion_status) != ("processing", "processing")
        or job.started_at != claim.claimed_at
        or upload.processing_started_at != claim.claimed_at
    ):
        return None
    return upload, job


def _cancelled_progress(prior: object) -> dict[str, object]:
    """Retain only content-free strict-local telemetry at cancellation."""
    progress: dict[str, object] = {"stage": "cancelled"}
    if not isinstance(prior, dict):
        return progress
    model_role = prior.get("model_role")
    if model_role in _STRICT_LOCAL_MODEL_ROLES:
        progress["model_role"] = model_role
    for key in _STRICT_LOCAL_PROGRESS_COUNTERS:
        value = prior.get(key)
        if type(value) is int and 0 <= value <= _STRICT_LOCAL_MAX_PROGRESS_COUNTER:
            progress[key] = value
    return progress


def _require_matching_strict_ingestion_identity(
    upload: UploadedFile,
    job: LocalAIJob,
) -> None:
    """Require immutable canonical identity without mutating either row."""
    if (
        upload.processing_mode != "validated_strict_local"
        or upload.processing_schema_version != _STRICT_INGESTION_SCHEMA
        or job.kind != "ingestion"
        or job.processing_mode != "validated_strict_local"
    ):
        raise ValueError("strict ingestion identity mismatch")
    snapshot, digest = canonicalize_manifest_snapshot(upload.processing_manifest)
    if (
        snapshot != upload.processing_manifest
        or job.manifest_snapshot != snapshot
        or job.manifest_sha256 != digest
    ):
        raise ValueError("strict ingestion identity mismatch")


async def cancel_strict_ingestion_pair(
    db: AsyncSession,
    *,
    user_id: UUID,
    upload_id: UUID,
    expected_job_id: UUID | None = None,
) -> StrictIngestionCancellation:
    """Lock upload then job and persist one valid cancellation state."""
    upload = (
        await db.execute(
            select(UploadedFile)
            .where(UploadedFile.id == upload_id, UploadedFile.user_id == user_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=404, detail="Local AI job not found.")

    job_query = select(LocalAIJob).where(
        LocalAIJob.upload_id == upload.id,
        LocalAIJob.user_id == user_id,
        LocalAIJob.kind == "ingestion",
        LocalAIJob.processing_mode == "validated_strict_local",
    )
    if expected_job_id is not None:
        job_query = job_query.where(LocalAIJob.id == expected_job_id)
    job = (
        await db.execute(
            job_query.with_for_update().execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=409, detail="This job cannot be cancelled.")

    if fail_legacy_runtime_identity_required(job):
        upload.ingestion_status = "failed"
        upload.progress_stage = None
        upload.progress_detail = None
        upload.ingestion_errors = [{"error_type": "runtime_identity_required"}]
        upload.processing_completed_at = job.completed_at
        return StrictIngestionCancellation(job, upload, False)

    try:
        _require_matching_strict_ingestion_identity(upload, job)
    except Exception:
        raise HTTPException(
            status_code=409,
            detail="This job cannot be cancelled.",
        ) from None

    pair_state = (job.status, upload.ingestion_status)
    if pair_state not in _CANCELLABLE_PAIRS:
        raise HTTPException(status_code=409, detail="This job cannot be cancelled.")

    worker_cancel_required = pair_state == ("processing", "processing")
    job.cancel_requested = True
    upload.cancel_requested = True
    if not worker_cancel_required:
        cancelled_at = datetime.now(timezone.utc)
        job.status = "cancelled"
        job.stage = "cancelled"
        job.progress = _cancelled_progress(job.progress)
        job.failure = None
        job.completed_at = cancelled_at
        upload.ingestion_status = "cancelled"
        upload.progress_stage = None
        upload.progress_detail = None
        upload.processing_completed_at = cancelled_at
    return StrictIngestionCancellation(job, upload, worker_cancel_required)
