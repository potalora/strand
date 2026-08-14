from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.database import async_session_factory, get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.schemas.upload import (
    BatchRejectionCode,
    BatchUploadResponse,
    CancelExtractionRequest,
    CancelExtractionResponse,
    ConfirmExtractionRequest,
    ExtractedEntitySchema,
    ExtractionResultResponse,
    LocalProcessingFailure,
    LocalRunInfo,
    PendingExtractionFile,
    ReprocessUploadRequest,
    RejectedUnstructuredUpload,
    TriggerExtractionRequest,
    TriggerExtractionResponse,
    UnstructuredUploadResponse,
    UploadHistoryResponse,
    UploadResponse,
    UploadStatusResponse,
)
from app.utils.file_utils import EncryptedFileWriter
from app.services.local_ai.processing_snapshot import (
    ProcessingSnapshot,
    build_ingestion_job,
    fail_active_legacy_ingestion_jobs,
    fail_legacy_runtime_identity_required,
    revalidate_strict_snapshot_admission,
    resolve_new_ingestion_snapshot,
)
from app.services.local_ai.types import ProcessingMode

if TYPE_CHECKING:
    from app.services.local_ai.ingestion_lifecycle import StrictIngestionClaim

# Per-event-loop semaphore caches.
#
# An ``asyncio.Semaphore`` binds to the event loop that it first *blocks* on (it
# registers waiter futures against the running loop). A single module-level
# semaphore reused across loops therefore raises
# ``RuntimeError: <Semaphore> is bound to a different event loop`` whenever a
# task must wait on it under a different loop. In production this stays benign
# because uvicorn runs one long-lived loop, but the test harness creates a fresh
# loop per test (pytest-asyncio) and the persisting module global goes stale.
# Keying by the running loop guarantees every loop gets a semaphore bound to it,
# while still capping concurrency at the configured limits.
_gemini_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

# Extraction semaphore to limit concurrent file extractions (layer above Gemini semaphore)
_extraction_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

# A strict-local extraction owns the one allowed MLX model process before its
# upload is claimed. Other extraction modes may still use the general slots.
_strict_extraction_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

_STRICT_LOCAL_PROGRESS_STAGES = frozenset(
    {
        "ocr",
        "extraction",
        "persisting_raw_extraction",
        "validating_extraction",
        "persisting_extraction_checkpoint",
        "persisting_evidence",
        "mapping_fhir",
        "finalizing",
    }
)
_STRICT_LOCAL_MODEL_ROLES = frozenset({"ocr", "extraction"})
_STRICT_LOCAL_PROGRESS_COUNTERS = (
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
)
_STRICT_LOCAL_MAX_PROGRESS_COUNTER = 2**63 - 1

# Worker task reference
_worker_task: asyncio.Task | None = None
_extraction_tasks: set[asyncio.Task[None]] = set()
_extraction_draining = False


def _merge_strict_local_progress(
    prior: object,
    update_value: dict[str, object],
    *,
    stage: str,
) -> dict[str, object]:
    """Carry only allowlisted telemetry into the next content-free stage."""

    merged: dict[str, object] = {"stage": stage}
    if isinstance(prior, dict):
        model_role = prior.get("model_role")
        if model_role in _STRICT_LOCAL_MODEL_ROLES:
            merged["model_role"] = model_role
        for key in _STRICT_LOCAL_PROGRESS_COUNTERS:
            item = prior.get(key)
            if type(item) is int and 0 <= item <= _STRICT_LOCAL_MAX_PROGRESS_COUNTER:
                merged[key] = item
    for key, item in update_value.items():
        if key == "model_role" and item in _STRICT_LOCAL_MODEL_ROLES:
            merged[key] = item
        elif key in _STRICT_LOCAL_PROGRESS_COUNTERS and (
            type(item) is int and 0 <= item <= _STRICT_LOCAL_MAX_PROGRESS_COUNTER
        ):
            merged[key] = item
    return merged


def _prune_closed_loops(
    cache: dict[asyncio.AbstractEventLoop, asyncio.Semaphore],
) -> None:
    """Drop semaphores keyed to event loops that have since closed.

    Keeps the per-loop caches bounded so repeated short-lived loops (e.g. one
    per test) do not accumulate dead entries.
    """
    stale = [loop for loop in cache if loop.is_closed()]
    for loop in stale:
        del cache[loop]


def _get_gemini_semaphore() -> asyncio.Semaphore:
    """Return the Gemini-concurrency semaphore bound to the running event loop."""
    loop = asyncio.get_running_loop()
    sem = _gemini_semaphores.get(loop)
    if sem is None:
        _prune_closed_loops(_gemini_semaphores)
        sem = asyncio.Semaphore(settings.gemini_concurrency_limit)
        _gemini_semaphores[loop] = sem
    return sem


def _get_extraction_semaphore() -> asyncio.Semaphore:
    """Return the file-extraction semaphore bound to the running event loop."""
    loop = asyncio.get_running_loop()
    sem = _extraction_semaphores.get(loop)
    if sem is None:
        _prune_closed_loops(_extraction_semaphores)
        sem = asyncio.Semaphore(settings.extraction_concurrency)
        _extraction_semaphores[loop] = sem
    return sem


def _get_strict_extraction_semaphore() -> asyncio.Semaphore:
    """Return the single strict-local admission slot for the running loop."""
    loop = asyncio.get_running_loop()
    sem = _strict_extraction_semaphores.get(loop)
    if sem is None:
        _prune_closed_loops(_strict_extraction_semaphores)
        sem = asyncio.Semaphore(1)
        _strict_extraction_semaphores[loop] = sem
    return sem


async def _extraction_worker() -> None:
    """DB-polling background worker: claim pending files and process them.

    Uses a rolling window approach: claims one file at a time, acquires a
    semaphore slot, then fires off processing without waiting. This keeps all
    extraction slots busy instead of blocking on the slowest file in a batch.
    """
    sem = _get_extraction_semaphore()
    strict_sem = _get_strict_extraction_semaphore()
    poll_interval = 2
    stuck_check_interval = 60
    last_stuck_check = datetime.now(timezone.utc)

    logger.info(
        "Extraction worker started (concurrency=%d, poll=%ds)",
        settings.extraction_concurrency,
        poll_interval,
    )

    while True:
        try:
            now = datetime.now(timezone.utc)
            if (now - last_stuck_check).total_seconds() >= stuck_check_interval:
                await _recover_stuck_files()
                last_stuck_check = now

            await sem.acquire()
            if not strict_sem.locked():
                await strict_sem.acquire()
                from app.services.local_ai.ingestion_lifecycle import (
                    claim_next_strict_ingestion_pair,
                )

                try:
                    async with async_session_factory() as claim_db:
                        strict_claim = await claim_next_strict_ingestion_pair(claim_db)
                        if strict_claim is None:
                            await claim_db.rollback()
                        else:
                            await claim_db.commit()
                except Exception:
                    strict_sem.release()
                    sem.release()
                    raise
                if strict_claim is not None:
                    scheduled = await _schedule_claimed_extraction(
                        sem,
                        strict_claim,
                        strict_sem=strict_sem,
                    )
                    if not scheduled:
                        await asyncio.sleep(poll_interval)
                    continue
                strict_sem.release()

            claimed = await _claim_pending_files(1)
            if not claimed:
                sem.release()
                await asyncio.sleep(poll_interval)
                continue

            upload_id, file_path, user_id, processing_mode = claimed[0]
            logger.info("Claimed file %s for extraction", upload_id)
            child = asyncio.create_task(
                _process_and_release(
                    sem,
                    upload_id,
                    Path(file_path),
                    user_id,
                )
            )
            _extraction_tasks.add(child)
            child.add_done_callback(_extraction_tasks.discard)

        except Exception:
            logger.error("Extraction worker encountered an error; recovering")
            await asyncio.sleep(5)


async def _claim_pending_files(
    batch_size: int,
) -> list[tuple[str, str, str, str]]:
    """Claim non-strict extraction files using SELECT FOR UPDATE SKIP LOCKED."""
    async with async_session_factory() as db:
        try:
            result = await db.execute(
                text(
                    "SELECT id, storage_path, user_id, processing_mode "
                    "FROM uploaded_files "
                    "WHERE ingestion_status = 'pending_extraction' "
                    "AND file_category = 'unstructured' "
                    "AND manual_extraction_required = false "
                    "AND processing_mode != 'validated_strict_local' "
                    "ORDER BY created_at ASC "
                    "LIMIT :batch_size "
                    "FOR UPDATE SKIP LOCKED"
                ),
                {
                    "batch_size": batch_size,
                },
            )
            rows = result.fetchall()

            if not rows:
                return []

            # Mark claimed files as processing
            ids = [row[0] for row in rows]
            now = datetime.now(timezone.utc)
            await db.execute(
                text(
                    "UPDATE uploaded_files "
                    "SET ingestion_status = 'processing', "
                    "processing_started_at = :now "
                    "WHERE id = ANY(:ids)"
                ),
                {"now": now, "ids": ids},
            )
            await db.commit()

            return [(str(row[0]), row[1], str(row[2]), str(row[3])) for row in rows]

        except Exception:
            logger.error("Failed to claim pending files")
            await db.rollback()
            return []


async def _schedule_claimed_extraction(
    sem: asyncio.Semaphore,
    claim: "StrictIngestionClaim",
    *,
    strict_sem: asyncio.Semaphore,
    create_task_fn: Callable[[object], asyncio.Task[None]] | None = None,
    callback_drain_timeout_seconds: float = 1.0,
) -> bool:
    from app.services.local_ai.ingestion_lifecycle import (
        compensate_unstarted_strict_ingestion_claim,
    )

    released = False

    def release_slots() -> None:
        nonlocal released
        if released:
            return
        released = True
        strict_sem.release()
        sem.release()

    async def compensate() -> None:
        async with async_session_factory() as compensation_db:
            compensated = await compensate_unstarted_strict_ingestion_claim(
                compensation_db,
                claim,
            )
            if compensated:
                await compensation_db.commit()
            else:
                await compensation_db.rollback()

    coroutine = _process_and_release(
        sem,
        claim.upload_id,
        Path(claim.storage_path),
        claim.user_id,
        strict_claim=claim,
        release_slots=release_slots,
    )
    task_factory = create_task_fn or asyncio.create_task
    try:
        child = task_factory(coroutine)
    except Exception:
        coroutine.close()
        release_slots()
        await compensate()
        logger.error("Strict-local task creation failed after durable claim")
        return False

    _extraction_tasks.add(child)
    # A configured eager task factory can enter the child before callback
    # registration. The failure path below therefore always drains the child
    # before compensating the exact durable claim.
    await asyncio.sleep(0)
    try:
        child.add_done_callback(_extraction_tasks.discard)
    except Exception:
        child.cancel()
        terminated = await _wait_for_extraction_tasks(
            [child],
            timeout_seconds=callback_drain_timeout_seconds,
        )
        if not terminated:
            logger.error("Strict-local task did not terminate after callback failure")
            return False
        _extraction_tasks.discard(child)
        release_slots()
        await compensate()
        logger.error("Strict-local task callback registration failed")
        return False
    return True


async def _recover_stuck_files() -> None:
    """Reset files stuck in 'processing' beyond the timeout.

    Increments retry_count. After max retries, marks as 'failed'.
    """
    timeout = timedelta(minutes=settings.extraction_timeout_minutes)
    cutoff = datetime.now(timezone.utc) - timeout
    strict_timeout_seconds = max(
        int(timeout.total_seconds()),
        settings.local_ai_worker_timeout_seconds + 120,
    )
    strict_cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=strict_timeout_seconds
    )
    max_retries = settings.extraction_max_retries
    recovered_at = datetime.now(timezone.utc)

    async with async_session_factory() as db:
        try:
            await fail_active_legacy_ingestion_jobs(
                db,
                completed_at=recovered_at,
            )

            # Cancellation is terminal and takes precedence over retry recovery.
            # Pair the upload/job writes in this one transaction so a strict job
            # can never remain queued behind a cancelled upload.
            await db.execute(
                text(
                    "UPDATE uploaded_files AS u "
                    "SET ingestion_status = 'cancelled', progress_stage = NULL, "
                    "progress_detail = NULL, processing_completed_at = :now "
                    "WHERE u.ingestion_status IN ('pending_extraction', 'processing') "
                    "AND u.file_category = 'unstructured' "
                    "AND u.cancel_requested = true "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM local_ai_jobs AS j "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.manifest_snapshot->>'schema_version' "
                    "IS DISTINCT FROM '2'"
                    ")"
                ),
                {"now": recovered_at},
            )
            await db.execute(
                text(
                    "UPDATE local_ai_jobs AS j "
                    "SET cancel_requested = true, status = 'cancelled', "
                    "stage = 'cancelled', progress = '{\"stage\":\"cancelled\"}'::jsonb, "
                    "failure = NULL, completed_at = :now "
                    "FROM uploaded_files AS u "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.manifest_snapshot->>'schema_version' = '2' "
                    "AND j.status IN ('queued', 'processing') "
                    "AND u.ingestion_status = 'cancelled' "
                    "AND u.cancel_requested = true"
                ),
                {"now": recovered_at},
            )

            # Reset retriable non-strict files back to pending. Strict-local
            # liveness is tracked separately from its job heartbeat below.
            await db.execute(
                text(
                    "UPDATE uploaded_files AS u "
                    "SET ingestion_status = 'pending_extraction', "
                    "processing_started_at = NULL, "
                    "retry_count = COALESCE(retry_count, 0) + 1 "
                    "WHERE u.ingestion_status = 'processing' "
                    "AND u.file_category = 'unstructured' "
                    "AND u.cancel_requested = false "
                    "AND u.processing_started_at < :cutoff "
                    "AND COALESCE(u.retry_count, 0) < :max_retries "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM local_ai_jobs AS j "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.status IN ('queued', 'processing')"
                    ")"
                ),
                {"cutoff": cutoff, "max_retries": max_retries},
            )

            # A progressing strict job updates ``local_ai_jobs.updated_at`` with
            # content-free counters. Its stale window exceeds the worker idle
            # timeout, so recovery cannot race a legal in-flight model call.
            await db.execute(
                text(
                    "UPDATE uploaded_files AS u "
                    "SET ingestion_status = 'pending_extraction', "
                    "processing_started_at = NULL, "
                    "retry_count = COALESCE(u.retry_count, 0) + 1 "
                    "FROM local_ai_jobs AS j "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.status IN ('queued', 'processing') "
                    "AND u.ingestion_status = 'processing' "
                    "AND u.file_category = 'unstructured' "
                    "AND u.cancel_requested = false "
                    "AND COALESCE(u.retry_count, 0) < :max_retries "
                    "AND ("
                    "(j.status = 'processing' AND j.updated_at < :strict_cutoff) "
                    "OR (j.status = 'queued' "
                    "AND u.processing_started_at < :strict_cutoff)"
                    ")"
                ),
                {
                    "strict_cutoff": strict_cutoff,
                    "max_retries": max_retries,
                },
            )

            # Mark non-strict files that exceeded max retries as failed.
            await db.execute(
                text(
                    "UPDATE uploaded_files AS u "
                    "SET ingestion_status = 'failed', "
                    'ingestion_errors = \'[{"error": "Processing timed out after maximum retries.", "error_type": "TimeoutError"}]\'::jsonb, '
                    "processing_completed_at = :now "
                    "WHERE u.ingestion_status = 'processing' "
                    "AND u.file_category = 'unstructured' "
                    "AND u.cancel_requested = false "
                    "AND u.processing_started_at < :cutoff "
                    "AND COALESCE(u.retry_count, 0) >= :max_retries "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM local_ai_jobs AS j "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.status IN ('queued', 'processing')"
                    ")"
                ),
                {
                    "cutoff": cutoff,
                    "max_retries": max_retries,
                    "now": recovered_at,
                },
            )

            await db.execute(
                text(
                    "UPDATE uploaded_files AS u "
                    "SET ingestion_status = 'failed', "
                    'ingestion_errors = \'[{"error": "Processing timed out after maximum retries.", "error_type": "TimeoutError"}]\'::jsonb, '
                    "processing_completed_at = :now "
                    "FROM local_ai_jobs AS j "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.status IN ('queued', 'processing') "
                    "AND u.ingestion_status = 'processing' "
                    "AND u.file_category = 'unstructured' "
                    "AND u.cancel_requested = false "
                    "AND COALESCE(u.retry_count, 0) >= :max_retries "
                    "AND ("
                    "(j.status = 'processing' AND j.updated_at < :strict_cutoff) "
                    "OR (j.status = 'queued' "
                    "AND u.processing_started_at < :strict_cutoff)"
                    ")"
                ),
                {
                    "strict_cutoff": strict_cutoff,
                    "max_retries": max_retries,
                    "now": recovered_at,
                },
            )

            # Strict-local model/page state is resumable. Requeue its immutable
            # job snapshot while leaving encrypted ``local_ai_pages`` untouched.
            await db.execute(
                text(
                    "UPDATE local_ai_jobs AS j "
                    "SET status = 'queued', stage = 'recovery', "
                    "failure = NULL, completed_at = NULL, updated_at = :now "
                    "FROM uploaded_files AS u "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.manifest_snapshot->>'schema_version' = '2' "
                    "AND j.status = 'processing' "
                    "AND u.ingestion_status = 'pending_extraction'"
                ),
                {"now": recovered_at},
            )
            await db.execute(
                text(
                    "UPDATE local_ai_jobs AS j "
                    "SET status = 'failed', stage = 'failed', completed_at = :now, "
                    "updated_at = :now, "
                    "failure = jsonb_build_object("
                    "'stage', j.stage, "
                    "'code', 'local_worker_timeout', "
                    "'message', 'Strict-local processing did not complete.', "
                    "'model_role', j.progress->>'model_role', "
                    "'retryable', false, "
                    "'checkpoint_preserved', EXISTS ("
                    "SELECT 1 FROM local_ai_pages AS p WHERE p.job_id = j.id"
                    "), "
                    "'cloud_fallback_attempted', false"
                    ") "
                    "FROM uploaded_files AS u "
                    "WHERE j.upload_id = u.id "
                    "AND j.processing_mode = 'validated_strict_local' "
                    "AND j.manifest_snapshot->>'schema_version' = '2' "
                    "AND j.status IN ('queued', 'processing') "
                    "AND u.ingestion_status = 'failed'"
                ),
                {"now": recovered_at},
            )

            await db.commit()
        except Exception:
            logger.error("Failed to recover stuck files")
            await db.rollback()


async def _process_and_release(
    sem: asyncio.Semaphore,
    upload_id: UUID | str,
    file_path: Path,
    user_id: UUID | str,
    *,
    strict_sem: asyncio.Semaphore | None = None,
    strict_claim: "StrictIngestionClaim | None" = None,
    release_slots: Callable[[], None] | None = None,
) -> None:
    """Process one file then release the semaphore."""
    try:
        uid = UUID(str(upload_id)) if not isinstance(upload_id, UUID) else upload_id
        uid_user = UUID(str(user_id)) if not isinstance(user_id, UUID) else user_id
        await _process_unstructured(
            uid,
            file_path,
            uid_user,
            strict_claim=strict_claim,
        )
    finally:
        if release_slots is not None:
            release_slots()
        else:
            if strict_sem is not None:
                strict_sem.release()
            sem.release()


def start_extraction_worker() -> None:
    """Start the DB-polling extraction worker. Called from main.py lifespan."""
    global _worker_task
    if _extraction_draining:
        return
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_extraction_worker())


def reset_extraction_worker_shutdown() -> None:
    """Permit polling at the start of a new application lifespan."""
    global _extraction_draining
    _extraction_draining = False


async def stop_extraction_worker() -> None:
    """Stop admissions and drain claimed extraction children for shutdown."""
    global _worker_task, _extraction_draining
    _extraction_draining = True
    worker = _worker_task
    if worker is not None and not worker.done():
        worker.cancel()
        await _drain_extraction_shutdown_tasks([worker])
    _worker_task = None
    children = list(_extraction_tasks)
    for child in children:
        child.cancel()
    if children:
        await _drain_extraction_shutdown_tasks(children)
    for child in children:
        if child.done():
            _extraction_tasks.discard(child)


async def _drain_extraction_shutdown_tasks(tasks: list[asyncio.Task[object]]) -> None:
    """Bound shutdown waits so durable recovery cannot be held by a stuck task."""
    terminated = await _wait_for_extraction_tasks(
        tasks,
        timeout_seconds=settings.local_ai_shutdown_drain_seconds,
    )
    if not terminated:
        logger.warning("Strict-local extraction shutdown drain timed out")


async def _wait_for_extraction_tasks(
    tasks: list[asyncio.Task[object]],
    *,
    timeout_seconds: float,
) -> bool:
    """Wait without registering callbacks on tasks that rejected registration."""
    try:
        async with asyncio.timeout(timeout_seconds):
            while any(not task.done() for task in tasks):
                await asyncio.sleep(0)
    except TimeoutError:
        return False
    for task in tasks:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("Extraction task terminated with an error")
    return True


# Statuses that count as "done" for batch progress. ``cancelled`` is terminal
# (user-initiated), so it is folded into the completed/terminal bucket — never
# left in pending/processing — so a progress bar can reach 100%.
_PROGRESS_DONE_STATUSES = [
    "completed",
    "awaiting_confirmation",
    "awaiting_review",
    "completed_with_merges",
    "cancelled",
]

# A file may only be cancelled while extraction is queued or actively running.
_CANCELLABLE_STATUSES = {"pending_extraction", "processing"}

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


async def _latest_local_jobs(
    db: AsyncSession,
    upload_ids: list[UUID],
    user_id: UUID,
) -> dict[UUID, object]:
    """Return one latest strict-local job per upload without clinical payloads."""
    if not upload_ids:
        return {}
    from app.models.local_ai import LocalAIJob

    jobs = (
        (
            await db.execute(
                select(LocalAIJob)
                .where(
                    LocalAIJob.upload_id.in_(upload_ids),
                    LocalAIJob.user_id == user_id,
                    LocalAIJob.processing_mode == "validated_strict_local",
                )
                .order_by(
                    LocalAIJob.upload_id,
                    LocalAIJob.created_at.desc(),
                    LocalAIJob.id.desc(),
                )
            )
        )
        .scalars()
        .all()
    )
    latest: dict[UUID, object] = {}
    for job in jobs:
        latest.setdefault(job.upload_id, job)
    return latest


def _local_job_views(
    job: object | None,
) -> tuple[LocalRunInfo | None, LocalProcessingFailure | None]:
    """Build typed, non-content local run state from a persisted job."""
    if job is None:
        return None, None
    models: list[dict[str, str]] = []
    try:
        snapshot = job.revalidate_manifest_snapshot()
        for artifact in snapshot["artifacts"]:
            if artifact["role"] not in {"ocr", "extraction"}:
                continue
            models.append(
                {
                    "role": artifact["role"],
                    "repository": artifact["repository"],
                    "revision": artifact["revision"],
                }
            )
        local_run = LocalRunInfo(
            privacy_mode="validated_strict_local",
            models=models,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        local_run = LocalRunInfo(privacy_mode="validated_strict_local", models=[])

    failure = None
    raw_failure = getattr(job, "failure", None)
    if raw_failure is not None:
        try:
            failure = LocalProcessingFailure.model_validate(raw_failure)
        except (TypeError, ValueError):
            failure = LocalProcessingFailure(
                stage="unknown",
                code="local_ai_error",
                message="Strict-local processing did not complete.",
                retryable=False,
                checkpoint_preserved=True,
                cloud_fallback_attempted=False,
            )
    return local_run, failure


# --- Security helpers ---

MAGIC_BYTES = {
    ".pdf": b"%PDF",
    ".rtf": b"{\\rtf",
    ".tif": [b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"],  # LE and BE TIFF
    ".tiff": [b"\x49\x49\x2a\x00", b"\x4d\x4d\x00\x2a"],
}


# SEC-DOS-01: stream uploads to disk in bounded chunks instead of buffering the
# whole body into a single in-memory ``bytes``. A 1 MiB chunk keeps peak RAM per
# in-flight upload tiny regardless of the configured ceiling (up to 5 GB for Epic
# exports), which would otherwise OOM a 16 GB host.
UPLOAD_CHUNK_SIZE = 1024 * 1024


def _reject_if_content_length_exceeds(
    request: Request | None, max_bytes: int, detail: str
) -> None:
    """Reject up front when the declared ``Content-Length`` exceeds the cap.

    A cheap pre-check (no body I/O) so an obviously-oversized upload is refused
    before we stream it. A missing/malformed header falls through to the
    streaming byte-budget enforcement, which is authoritative.
    """
    if request is None:
        return
    declared = request.headers.get("content-length")
    if not declared:
        return
    try:
        declared_bytes = int(declared)
    except (TypeError, ValueError):
        return
    if declared_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=detail)


async def _stream_upload_to_disk(
    file: UploadFile,
    file_path: Path,
    max_bytes: int,
    *,
    request: Request | None = None,
    detail: str = "File too large",
    encrypt: bool = True,
) -> tuple[int, bytes, str]:
    """Stream an ``UploadFile`` to ``file_path`` in chunks, enforcing a hard cap.

    Rejects early on an oversized declared ``Content-Length``, then copies the
    body one ``UPLOAD_CHUNK_SIZE`` chunk at a time while tracking a running total
    — aborting and unlinking the partial file the moment the total exceeds
    ``max_bytes`` (SEC-DOS-01: never materialize the whole body in RAM).

    CRYPTO-02: when ``encrypt`` is True (default) the file is written as a framed
    AES-256-GCM blob — each plaintext chunk is encrypted into its own frame as it
    streams, so the bytes at rest never carry plaintext PHI and the whole file is
    never buffered in memory. Structured uploads (FHIR JSON / ZIP / Epic / CDA)
    pass ``encrypt=False`` because their ingest read path streams the file in
    plaintext immediately after upload (it is owned by the ingestion coordinator);
    the operator migration script encrypts those archives at rest after ingest.

    The size cap and the magic-byte header are always computed on PLAINTEXT bytes
    read from the body, independent of encryption. The returned ``hash`` is the
    SHA-256 of the plaintext, so deduplication-by-hash stays deterministic
    regardless of the random per-frame nonces.

    Returns ``(bytes_written, header, plaintext_sha256_hex)`` where ``header`` is
    the first chunk's leading plaintext bytes, so callers can validate magic
    bytes without re-reading.
    """
    _reject_if_content_length_exceeds(request, max_bytes, detail)

    total = 0
    header = b""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "wb") as f:
            writer = EncryptedFileWriter(f) if encrypt else None
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                if not header:
                    header = chunk[:16]
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=detail)
                hasher.update(chunk)
                if writer is not None:
                    writer.write_chunk(chunk)
                else:
                    f.write(chunk)
            if writer is not None:
                writer.finalize()  # write the header even for an empty body
    except BaseException:
        # Drop any partial file on rejection/error so a too-large upload leaves
        # nothing behind on disk.
        file_path.unlink(missing_ok=True)
        raise
    return total, header, hasher.hexdigest()


def _validate_magic_bytes(content: bytes, ext: str) -> bool:
    """Validate file content matches expected magic bytes for the extension."""
    expected = MAGIC_BYTES.get(ext)
    if expected is None:
        return True  # No magic bytes check for unknown types
    if isinstance(expected, list):
        return any(content[: len(sig)] == sig for sig in expected)
    return content[: len(expected)] == expected


def _safe_file_path(upload_dir: Path, user_id: UUID, original_filename: str) -> Path:
    """Generate a safe file path preventing path traversal attacks."""
    # Preserve original extension only
    ext = Path(original_filename).suffix.lower()
    safe_name = f"{user_id}_{uuid4().hex}{ext}"
    file_path = (upload_dir / safe_name).resolve()

    # Validate the resolved path is within the upload directory
    upload_dir_resolved = upload_dir.resolve()
    if not str(file_path).startswith(str(upload_dir_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return file_path


def _collect_entities(results: list, total_chunks: int) -> tuple[list, int]:
    """Collect entities from asyncio.gather() results; count failed chunks.

    A chunk is considered failed when it raised an Exception OR when its
    ExtractionResult carries a non-empty ``error`` field.  Successful entity
    objects are tagged with ``_source_section`` before being returned.

    Args:
        results: Raw list returned by ``asyncio.gather(..., return_exceptions=True)``.
            Each item is either ``(ExtractionResult, section_type)`` or an ``Exception``.
        total_chunks: Total number of tasks that were submitted (used for logging).

    Returns:
        A 2-tuple ``(entities, failed_chunks)`` where ``entities`` is a flat list of
        ``ExtractedEntity`` objects and ``failed_chunks`` is the count of chunks that
        did not produce usable output.
    """
    from app.services.extraction.entity_extractor import (
        ExtractionResult,
    )  # local to avoid circular import

    entities: list = []
    failed = 0
    for r in results:
        if isinstance(r, Exception):
            failed += 1
            logger.error("Section extraction raised: %s", r)
            continue
        extraction_result: ExtractionResult
        extraction_result, section_type = r
        if extraction_result.error:
            failed += 1
            logger.warning(
                "Extraction error in section %s: %s",
                section_type,
                extraction_result.error,
            )
            continue
        for entity in extraction_result.entities:
            entity.attributes["_source_section"] = section_type
            entities.append(entity)
    return entities, failed


# --- Endpoints ---


async def _resolve_ingestion_snapshot_or_409(
    db: AsyncSession,
    user_id: UUID,
    requested_mode: ProcessingMode | None,
) -> ProcessingSnapshot:
    """Resolve a new upload's immutable mode or return a policy conflict."""

    from app.services.local_ai.errors import LocalPolicyError

    try:
        return await resolve_new_ingestion_snapshot(db, user_id, requested_mode)
    except LocalPolicyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _unstructured_file_type(extension: str) -> str:
    """Return the response file type without importing provider-capable OCR."""

    return "tiff" if extension in {".tif", ".tiff"} else extension.removeprefix(".")


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
    processing_mode: ProcessingMode | None = Form(default=None),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a FHIR JSON or ZIP file for ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    snapshot = await _resolve_ingestion_snapshot_or_409(
        db,
        user_id,
        processing_mode,
    )
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    # CRYPTO-02 (issue #54): structured uploads (FHIR JSON / ZIP) are encrypted at
    # rest as they stream. The ingestion coordinator stream-decrypts the file to a
    # temp plaintext copy at the ingest entry (bounded memory; W9 caps preserved),
    # so the bytes at rest never carry plaintext PHI.
    await _stream_upload_to_disk(
        file,
        file_path,
        settings.max_file_size_mb * 1024 * 1024,
        request=request,
        encrypt=True,
    )

    # Run ingestion synchronously for now (small files)
    from app.services.ingestion.coordinator import ingest_file

    result = await ingest_file(
        db=db,
        user_id=user_id,
        file_path=file_path,
        original_filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        processing_mode=snapshot.mode.value,
        processing_manifest=copy.deepcopy(snapshot.manifest_snapshot),
        processing_schema_version=snapshot.schema_version,
    )

    await log_audit_event(
        db,
        user_id=user_id,
        action="file.upload",
        resource_type="uploaded_file",
        resource_id=UUID(result["upload_id"]),
        details={
            "file_type": Path(file.filename).suffix.lower() or "unknown",
            "file_category": "structured",
            "records": result["records_inserted"],
        },
    )

    return UploadResponse(
        upload_id=result["upload_id"],
        status=result["status"],
        records_inserted=result["records_inserted"],
        errors=result.get("errors", []),
        unstructured_uploads=result.get("unstructured_uploads", []),
    )


@router.post(
    "/epic-export", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_epic_export(
    file: UploadFile,
    request: Request,
    processing_mode: ProcessingMode | None = Form(default=None),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload an Epic EHI Tables export (ZIP)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    snapshot = await _resolve_ingestion_snapshot_or_409(
        db,
        user_id,
        processing_mode,
    )
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # C6 + SEC-DOS-01: stream to disk under the Epic export ceiling instead of
    # buffering up to 5 GB into RAM, rejecting early on Content-Length.
    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    # CRYPTO-02 (issue #54): the Epic EHI ZIP is encrypted at rest as it streams.
    # The coordinator stream-decrypts it to a temp plaintext copy at the ingest
    # entry (bounded memory under the 5 GB ceiling; W9 zip-bomb caps run on the
    # decrypted temp), so the bytes at rest never carry plaintext PHI.
    await _stream_upload_to_disk(
        file,
        file_path,
        settings.max_epic_export_size_mb * 1024 * 1024,
        request=request,
        detail=f"Epic export too large. Maximum size: {settings.max_epic_export_size_mb}MB",
        encrypt=True,
    )

    from app.services.ingestion.coordinator import ingest_file

    result = await ingest_file(
        db=db,
        user_id=user_id,
        file_path=file_path,
        original_filename=file.filename,
        mime_type=file.content_type or "application/zip",
        processing_mode=snapshot.mode.value,
        processing_manifest=copy.deepcopy(snapshot.manifest_snapshot),
        processing_schema_version=snapshot.schema_version,
    )

    return UploadResponse(
        upload_id=result["upload_id"],
        status=result["status"],
        records_inserted=result["records_inserted"],
        errors=result.get("errors", []),
    )


@router.get("/pending-extraction")
async def get_pending_extractions(
    statuses: str | None = None,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List files by extraction status for this user.

    Args:
        statuses: Comma-separated list of statuses to filter by.
                  Defaults to 'pending_extraction'.
    """
    status_list = (
        [s.strip() for s in statuses.split(",")] if statuses else ["pending_extraction"]
    )

    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.user_id == user_id)
        .where(UploadedFile.ingestion_status.in_(status_list))
        .order_by(UploadedFile.created_at.desc())
    )
    files = result.scalars().all()
    local_jobs = await _latest_local_jobs(db, [file.id for file in files], user_id)

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.pending_extraction.list",
        resource_type="uploaded_file",
        resource_id=None,
        details={"count": len(files), "statuses": status_list},
    )

    response_files = []
    for file in files:
        local_run, local_failure = _local_job_views(local_jobs.get(file.id))
        response_files.append(
            PendingExtractionFile(
                id=str(file.id),
                filename=file.filename,
                mime_type=file.mime_type,
                file_category=file.file_category,
                file_size_bytes=file.file_size_bytes,
                created_at=file.created_at.isoformat() if file.created_at else None,
                ingestion_status=file.ingestion_status,
                manual_extraction_required=file.manual_extraction_required,
                progress_stage=file.progress_stage,
                progress_detail=file.progress_detail,
                notices=file.notices or [],
                local_run=local_run,
                local_failure=local_failure,
                local_job_id=(
                    str(local_jobs[file.id].id) if file.id in local_jobs else None
                ),
            ).model_dump()
        )

    return {"files": response_files, "total": len(files)}


@router.get("/extraction-progress")
async def extraction_progress(
    ids: str | None = None,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get extraction progress counts for the user's unstructured files.

    When ``ids`` (comma-separated upload UUIDs) is provided, the counts are
    scoped to just those uploads — so a single-file upload reads "1 of 1", not
    "84 of 85" across the whole history. Still user-scoped. The ``cancelled``
    terminal status is folded into ``completed`` so the bar can reach 100%.
    """
    from sqlalchemy import func, case

    query = (
        select(
            func.count().label("total"),
            func.count()
            .filter(UploadedFile.ingestion_status.in_(_PROGRESS_DONE_STATUSES))
            .label("completed"),
            func.count()
            .filter(
                UploadedFile.ingestion_status.in_(
                    ("processing", "dedup_scanning", "dedup_processing")
                )
            )
            .label("processing"),
            func.count()
            .filter(UploadedFile.ingestion_status == "failed")
            .label("failed"),
            func.count()
            .filter(UploadedFile.ingestion_status == "pending_extraction")
            .label("pending"),
            func.coalesce(
                func.sum(
                    case(
                        (
                            UploadedFile.ingestion_status.in_(_PROGRESS_DONE_STATUSES),
                            UploadedFile.record_count,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("records_created"),
        )
        .where(UploadedFile.user_id == user_id)
        .where(UploadedFile.file_category == "unstructured")
        # Duplicate files are skipped (no extraction work), so they must not
        # inflate the progress denominator ("1 of 2 processed" when 1 is a dup).
        .where(UploadedFile.ingestion_status != "duplicate_file")
    )

    if ids is not None:
        id_list: list[UUID] = []
        for token in ids.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                id_list.append(UUID(token))
            except ValueError:
                continue  # ignore malformed ids rather than 500
        # An explicit (possibly empty) id filter scopes to those uploads only.
        query = query.where(UploadedFile.id.in_(id_list))

    row = (await db.execute(query)).one()

    return {
        "total": row.total,
        "completed": row.completed,
        "processing": row.processing,
        "failed": row.failed,
        "pending": row.pending,
        "records_created": row.records_created,
    }


@router.post("/cancel", response_model=CancelExtractionResponse)
async def cancel_extraction(
    body: CancelExtractionRequest,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> CancelExtractionResponse:
    """Request cancellation of in-flight / queued extractions.

    Sets ``cancel_requested`` on the user's own files that are still
    cancellable (queued or processing). The DB-polling worker checks the flag
    between stages and aborts cleanly, marking the file ``cancelled``. Files
    that are already terminal (or not owned) are returned in ``skipped``.
    """
    parsed: dict[str, UUID] = {}
    for raw in body.upload_ids:
        try:
            parsed[raw] = UUID(raw)
        except (ValueError, AttributeError):
            parsed[raw] = None  # unparseable → always skipped

    valid_uuids = list({u for u in parsed.values() if u is not None})
    discovered: dict[UUID, str] = {}
    if valid_uuids:
        rows = (
            await db.execute(
                select(UploadedFile.id, UploadedFile.processing_mode).where(
                    UploadedFile.id.in_(valid_uuids),
                    UploadedFile.user_id == user_id,
                )
            )
        ).all()
        discovered = {row.id: row.processing_mode for row in rows}

    outcomes: dict[UUID, bool] = {}
    strict_results = []
    strict_ids = sorted(
        (
            upload_id
            for upload_id, processing_mode in discovered.items()
            if processing_mode == "validated_strict_local"
        ),
        key=str,
    )
    if strict_ids:
        from app.services.local_ai.ingestion_lifecycle import (
            cancel_strict_ingestion_pair,
        )

        for upload_id in strict_ids:
            try:
                result = await cancel_strict_ingestion_pair(
                    db,
                    user_id=user_id,
                    upload_id=upload_id,
                )
            except HTTPException as exc:
                if exc.status_code not in {404, 409}:
                    raise
                outcomes[upload_id] = False
            else:
                outcomes[upload_id] = True
                strict_results.append(result)

    non_strict_ids = [
        upload_id
        for upload_id, processing_mode in discovered.items()
        if processing_mode != "validated_strict_local"
    ]
    if non_strict_ids:
        non_strict_uploads = (
            (
                await db.execute(
                    select(UploadedFile)
                    .where(
                        UploadedFile.id.in_(non_strict_ids),
                        UploadedFile.user_id == user_id,
                        UploadedFile.processing_mode != "validated_strict_local",
                    )
                    .order_by(UploadedFile.id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        for upload in non_strict_uploads:
            if upload.ingestion_status in _CANCELLABLE_STATUSES:
                upload.cancel_requested = True
                outcomes[upload.id] = True
            else:
                outcomes[upload.id] = False

    cancelled = [
        raw
        for raw in body.upload_ids
        if (uid := parsed.get(raw)) is not None and outcomes.get(uid, False)
    ]
    skipped = [
        raw
        for raw in body.upload_ids
        if (uid := parsed.get(raw)) is None or not outcomes.get(uid, False)
    ]

    if cancelled:
        await db.commit()

    active_strict_results = [
        result for result in strict_results if result.worker_cancel_required
    ]
    if active_strict_results:
        from app.services.local_ai.errors import LocalWorkerError
        from app.services.local_ai.model_manager import local_model_manager

        for result in active_strict_results:
            try:
                cancelled_registered = await local_model_manager.cancel_registered(
                    str(result.job.id)
                )
                if not cancelled_registered:
                    await local_model_manager.cancel(str(result.job.id))
            except LocalWorkerError:
                logger.warning(
                    "Strict-local worker cancellation could not be confirmed",
                    extra={"job_id": str(result.job.id)},
                )

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.cancel",
        resource_type="uploaded_file",
        resource_id=None,
        details={"cancelled": len(cancelled), "skipped": len(skipped)},
    )

    return CancelExtractionResponse(cancelled=cancelled, skipped=skipped)


@router.post("/trigger-extraction", response_model=TriggerExtractionResponse)
async def trigger_extraction(
    body: TriggerExtractionRequest,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> TriggerExtractionResponse:
    """Trigger text+entity extraction for pending unstructured files."""
    from app.models.local_ai import LocalAIJob

    upload_ids = body.upload_ids

    # Bulk fetch only uploads owned by this user (HIPAA: row-level security)
    result = await db.execute(
        select(UploadedFile)
        .where(
            UploadedFile.id.in_(upload_ids),
            UploadedFile.user_id == user_id,
        )
        .order_by(UploadedFile.id)
        .with_for_update()
    )
    uploads = {u.id: u for u in result.scalars().all()}
    strict_jobs = (
        (
            await db.execute(
                select(LocalAIJob)
                .where(
                    LocalAIJob.upload_id.in_(upload_ids),
                    LocalAIJob.user_id == user_id,
                    LocalAIJob.kind == "ingestion",
                    LocalAIJob.processing_mode
                    == ProcessingMode.VALIDATED_STRICT_LOCAL.value,
                )
                .order_by(LocalAIJob.upload_id, LocalAIJob.id)
                .with_for_update()
            )
        )
        .scalars()
        .all()
    )
    strict_jobs_by_upload = {
        job.upload_id: job for job in strict_jobs if job.upload_id is not None
    }

    triggered = []
    failed = []

    # A 'processing' file is only re-triggerable when it's genuinely STUCK (no
    # live worker): processing longer than the stuck-recovery timeout, or with
    # no start time. Re-triggering an ACTIVELY processing file resets it to
    # pending_extraction, letting the worker re-claim it for a second concurrent
    # extraction pass (wasted Gemini calls, possible duplicate records).
    stale_cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=settings.extraction_timeout_minutes
    )
    strict_stale_cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=max(
            settings.extraction_timeout_minutes * 60,
            settings.local_ai_worker_timeout_seconds + 120,
        )
    )

    for uid in upload_ids:
        upload = uploads.get(uid)
        if not upload:
            failed.append({"upload_id": str(uid), "status": "not_found"})
            continue

        status_ = upload.ingestion_status
        strict_job = (
            strict_jobs_by_upload.get(upload.id)
            if upload.processing_mode == ProcessingMode.VALIDATED_STRICT_LOCAL.value
            else None
        )
        if strict_job is not None and status_ in ("failed", "awaiting_confirmation"):
            # Strict-local retries share the durable job/upload transition used
            # by the dedicated Local AI job API. This preserves encrypted page
            # checkpoints and refuses non-retryable or terminal jobs.
            from app.api.local_ai import _retry_local_ai_job

            try:
                await _retry_local_ai_job(db, job=strict_job, upload=upload)
            except HTTPException:
                failed.append({"upload_id": str(uid), "status": status_})
                continue
            triggered.append(upload)
            continue
        if status_ == "pending_extraction":
            if not upload.manual_extraction_required:
                failed.append(
                    {
                        "upload_id": str(uid),
                        "status": "manual_extraction_not_required",
                    }
                )
                continue
            if upload.processing_mode == ProcessingMode.VALIDATED_STRICT_LOCAL.value:
                if strict_job is None:
                    failed.append(
                        {
                            "upload_id": str(uid),
                            "status": "strict_job_unavailable",
                        }
                    )
                    continue
                if strict_job.status != "queued" or strict_job.cancel_requested:
                    failed.append(
                        {
                            "upload_id": str(uid),
                            "status": "strict_job_not_queued",
                        }
                    )
                    continue
            upload.manual_extraction_required = False
            triggered.append(upload)
            continue
        if status_ == "processing":
            heartbeat = (
                strict_job.updated_at
                if strict_job is not None
                else upload.processing_started_at
            )
            active_cutoff = (
                strict_stale_cutoff if strict_job is not None else stale_cutoff
            )
            if heartbeat is not None and heartbeat > active_cutoff:
                # Actively processing — skip to avoid a duplicate concurrent pass.
                failed.append({"upload_id": str(uid), "status": "processing"})
                continue
            if (
                upload.processing_mode == ProcessingMode.VALIDATED_STRICT_LOCAL.value
                and strict_job is None
            ):
                failed.append(
                    {
                        "upload_id": str(uid),
                        "status": "strict_job_unavailable",
                    }
                )
                continue
            if strict_job is not None:
                # Strict jobs are retried only once their terminal job state
                # marks them retryable; stale active jobs are recovered by the
                # worker's lease-recovery path rather than revived here.
                failed.append({"upload_id": str(uid), "status": "processing"})
                continue
            upload.ingestion_status = "pending_extraction"
        elif status_ in ("failed", "awaiting_confirmation"):
            # The DB-polling worker picks up pending_extraction automatically.
            if status_ in ("failed", "awaiting_confirmation"):
                if (
                    upload.processing_mode
                    == ProcessingMode.VALIDATED_STRICT_LOCAL.value
                    and strict_job is None
                ):
                    failed.append(
                        {
                            "upload_id": str(uid),
                            "status": "strict_job_unavailable",
                        }
                    )
                    continue
                upload.ingestion_status = "pending_extraction"
                upload.manual_extraction_required = False
        else:
            failed.append({"upload_id": str(uid), "status": status_})
            continue
        triggered.append(upload)

    if triggered:
        await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.trigger_extraction",
        resource_type="uploaded_file",
        resource_id=None,
        details={"triggered": len(triggered), "failed": len(failed)},
    )

    return TriggerExtractionResponse(
        triggered=len(triggered),
        failed=len(failed),
        results=[
            {"upload_id": str(upload.id), "status": "pending_extraction"}
            for upload in triggered
        ]
        + failed,
    )


@router.get("/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadStatusResponse:
    """Get ingestion job status."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    local_jobs = await _latest_local_jobs(db, [upload.id], user_id)
    local_run, local_failure = _local_job_views(local_jobs.get(upload.id))

    return UploadStatusResponse(
        upload_id=str(upload.id),
        filename=upload.filename,
        ingestion_status=upload.ingestion_status,
        record_count=upload.record_count,
        total_file_count=upload.total_file_count or 1,
        ingestion_progress=upload.ingestion_progress or {},
        ingestion_errors=upload.ingestion_errors or [],
        manual_extraction_required=upload.manual_extraction_required,
        processing_started_at=upload.processing_started_at,
        processing_completed_at=upload.processing_completed_at,
        progress_stage=upload.progress_stage,
        progress_detail=upload.progress_detail,
        notices=upload.notices or [],
        local_run=local_run,
        local_failure=local_failure,
        local_job_id=str(local_jobs[upload.id].id) if upload.id in local_jobs else None,
    )


@router.get("/{upload_id}/errors")
async def get_upload_errors(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get ingestion errors for a specific upload."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    return {"errors": upload.ingestion_errors or []}


@router.get("/history", response_model=UploadHistoryResponse)
async def get_upload_history(
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UploadHistoryResponse:
    """Upload history with record counts."""
    result = await db.execute(
        select(UploadedFile)
        .where(UploadedFile.user_id == user_id)
        .where(UploadedFile.deleted_at.is_(None))
        .order_by(UploadedFile.created_at.desc())
    )
    uploads = result.scalars().all()
    local_jobs = await _latest_local_jobs(
        db, [upload.id for upload in uploads], user_id
    )

    items = []
    for u in uploads:
        local_run, local_failure = _local_job_views(local_jobs.get(u.id))
        items.append(
            {
                "id": str(u.id),
                "filename": u.filename,
                "ingestion_status": u.ingestion_status,
                "record_count": u.record_count,
                "file_size_bytes": u.file_size_bytes,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "ingestion_progress": u.ingestion_progress or {},
                "ingestion_errors": u.ingestion_errors or [],
                "manual_extraction_required": u.manual_extraction_required,
                "local_run": local_run,
                "local_failure": local_failure,
                "local_job_id": (
                    str(local_jobs[u.id].id) if u.id in local_jobs else None
                ),
            }
        )

    return UploadHistoryResponse(items=items, total=len(items))


@router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_upload(
    upload_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete an upload and cascade soft-delete the records it produced.

    Never hard-deletes. The uploaded_file is marked deleted_at, every
    health_record linked to it via source_file_id is soft-deleted too, and an
    audit entry is written. User-scoped.
    """
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
            UploadedFile.deleted_at.is_(None),
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found"
        )

    now = datetime.now(timezone.utc)

    # Cascade soft-delete the records this upload produced.
    records_result = await db.execute(
        select(HealthRecord).where(
            HealthRecord.source_file_id == upload_id,
            HealthRecord.user_id == user_id,
            HealthRecord.deleted_at.is_(None),
        )
    )
    cascaded = records_result.scalars().all()
    for record in cascaded:
        record.deleted_at = now

    upload.deleted_at = now
    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.delete",
        resource_type="uploaded_file",
        resource_id=upload_id,
        ip_address=request.client.host if request.client else None,
        details={"records_soft_deleted": len(cascaded)},
    )


ALLOWED_UNSTRUCTURED = {".pdf", ".rtf", ".tif", ".tiff"}
MAX_BATCH_REJECTIONS = 50


def _append_batch_rejection(
    rejected: list[RejectedUnstructuredUpload],
    *,
    filename: str,
    code: BatchRejectionCode,
) -> None:
    if len(rejected) < MAX_BATCH_REJECTIONS:
        rejected.append(RejectedUnstructuredUpload(filename=filename, code=code))


async def _ensure_patient(db: AsyncSession, user_id: UUID) -> Patient:
    """Return the user's patient, creating a blank placeholder if none exists.

    Unstructured-first users have no Patient row yet; without one the
    auto-confirm step silently produces zero records and stalls the file at
    awaiting_confirmation (the "Reimagined" frontend has no manual-confirm UI).
    A blank patient (no PHI) is a safe placeholder that a later structured
    upload backfills via ``get_or_create_patient``.
    """
    result = await db.execute(
        select(Patient).where(Patient.user_id == user_id).limit(1)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        patient = Patient(user_id=user_id)
        db.add(patient)
        await db.flush()  # assign id now; caller commits it with the records
    return patient


async def _replacement_source_upload_id(
    db: AsyncSession,
    upload: UploadedFile,
) -> UUID:
    """Resolve the immutable root shared by duplicate and reprocess children."""

    current = upload
    visited: set[UUID] = set()
    while True:
        if current.id in visited:
            raise ValueError("Reprocessing source is invalid")
        visited.add(current.id)
        progress = current.ingestion_progress or {}
        raw_source_id = None
        if isinstance(progress, dict):
            raw_source_id = progress.get("reprocesses_upload_id") or progress.get(
                "duplicate_of"
            )
        if raw_source_id is None:
            return current.id
        try:
            source_id = UUID(str(raw_source_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("Reprocessing source is invalid") from exc
        source = (
            await db.execute(
                select(UploadedFile).where(
                    UploadedFile.id == source_id,
                    UploadedFile.user_id == upload.user_id,
                    UploadedFile.file_category == "unstructured",
                    UploadedFile.file_hash == upload.file_hash,
                )
            )
        ).scalar_one_or_none()
        if source is None:
            raise ValueError("Reprocessing source is invalid")
        current = source


async def _is_cancel_requested(db: AsyncSession, upload_id: UUID) -> bool:
    """Re-read the ``cancel_requested`` flag from the DB (read-committed).

    The cancel endpoint commits the flag on a different connection, so the
    worker must re-query rather than trust its (possibly stale) in-memory row.
    """
    row = await db.execute(
        text("SELECT cancel_requested FROM uploaded_files WHERE id = :id"),
        {"id": upload_id},
    )
    return bool(row.scalar())


async def _strict_cancel_requested(
    db: AsyncSession,
    upload_id: UUID,
    job_id: UUID,
) -> bool:
    """Re-read both strict-local cancellation flags without using ORM cache."""
    row = (
        await db.execute(
            text(
                "SELECT u.cancel_requested, j.cancel_requested "
                "FROM uploaded_files AS u "
                "JOIN local_ai_jobs AS j "
                "ON j.upload_id = u.id AND j.user_id = u.user_id "
                "WHERE u.id = :upload_id AND j.id = :job_id"
            ),
            {"upload_id": upload_id, "job_id": job_id},
        )
    ).one_or_none()
    if row is None:
        return True
    return bool(row[0] or row[1])


async def _refresh_strict_local_job_lease(
    db: AsyncSession,
    *,
    upload_id: UUID,
    job_id: UUID,
    user_id: UUID,
    claim_started_at: datetime | None = None,
) -> None:
    """Durably renew one active strict job without changing visible progress."""

    from app.services.local_ai.errors import LocalPolicyError

    refreshed_at = datetime.now(timezone.utc)
    attempt_clause = (
        "AND j.started_at = :claim_started_at " if claim_started_at is not None else ""
    )
    refreshed = (
        await db.execute(
            text(
                "UPDATE local_ai_jobs AS j "
                "SET updated_at = :refreshed_at "
                "FROM uploaded_files AS u "
                "WHERE j.id = :job_id "
                "AND j.upload_id = :upload_id "
                "AND j.user_id = :user_id "
                "AND j.processing_mode = 'validated_strict_local' "
                "AND j.manifest_snapshot->>'schema_version' = '2' "
                "AND j.status = 'processing' "
                "AND j.cancel_requested = false "
                f"{attempt_clause}"
                "AND u.id = j.upload_id "
                "AND u.user_id = j.user_id "
                "AND u.ingestion_status = 'processing' "
                "AND u.cancel_requested = false "
                "RETURNING j.id"
            ),
            {
                "refreshed_at": refreshed_at,
                "job_id": job_id,
                "upload_id": upload_id,
                "user_id": user_id,
                "claim_started_at": claim_started_at,
            },
        )
    ).scalar_one_or_none()
    if refreshed is None:
        await db.rollback()
        raise LocalPolicyError("Strict-local job was cancelled.")
    await db.commit()


async def _persist_strict_local_progress(
    *,
    runner_db: AsyncSession,
    upload_id: UUID,
    job_id: UUID,
    user_id: UUID,
    stage: str,
    progress: dict[str, object],
    claim_started_at: datetime | None = None,
) -> None:
    """Atomically persist content-free strict-local progress in its own session.

    Model progress can arrive while the runner's content-bearing session has an
    unrelated open or failed transaction. Locking and updating only identifiers,
    status flags, and validated progress keeps callback persistence independent
    without loading encrypted clinical columns into this short-lived session.
    """

    from app.models.local_ai import LocalAIJob
    from app.services.local_ai.errors import LocalPolicyError

    if runner_db.bind is None:
        raise LocalPolicyError("Strict-local job is unavailable.")
    progress_session_factory = async_sessionmaker(
        bind=runner_db.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with progress_session_factory() as progress_db:
        active_upload_id = (
            await progress_db.execute(
                select(UploadedFile.id)
                .where(
                    UploadedFile.id == upload_id,
                    UploadedFile.user_id == user_id,
                    UploadedFile.processing_mode == "validated_strict_local",
                    UploadedFile.ingestion_status == "processing",
                    UploadedFile.cancel_requested.is_(False),
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if active_upload_id is None:
            raise LocalPolicyError("Strict-local job was cancelled.")

        active_job_query = select(LocalAIJob).where(
            LocalAIJob.id == job_id,
            LocalAIJob.upload_id == upload_id,
            LocalAIJob.user_id == user_id,
            LocalAIJob.processing_mode == "validated_strict_local",
            LocalAIJob.manifest_snapshot["schema_version"].as_integer() == 2,
            LocalAIJob.status == "processing",
            LocalAIJob.cancel_requested.is_(False),
        )
        if claim_started_at is not None:
            active_job_query = active_job_query.where(
                LocalAIJob.started_at == claim_started_at
            )
        active_job = (
            await progress_db.execute(active_job_query.with_for_update())
        ).scalar_one_or_none()
        if active_job is None:
            raise LocalPolicyError("Strict-local job was cancelled.")

        merged_progress = _merge_strict_local_progress(
            active_job.progress,
            progress,
            stage=stage,
        )
        update_job_query = (
            update(LocalAIJob)
            .where(
                LocalAIJob.id == active_job.id,
                LocalAIJob.manifest_snapshot["schema_version"].as_integer() == 2,
                LocalAIJob.status == "processing",
                LocalAIJob.cancel_requested.is_(False),
            )
            .values(stage=stage, progress=merged_progress)
            .returning(LocalAIJob.id)
        )
        if claim_started_at is not None:
            update_job_query = update_job_query.where(
                LocalAIJob.started_at == claim_started_at
            )
        updated_job_id = (
            await progress_db.execute(update_job_query)
        ).scalar_one_or_none()
        updated_upload_id = (
            await progress_db.execute(
                update(UploadedFile)
                .where(
                    UploadedFile.id == active_upload_id,
                    UploadedFile.ingestion_status == "processing",
                    UploadedFile.cancel_requested.is_(False),
                )
                .values(
                    progress_stage=f"local_{stage}",
                    progress_detail=merged_progress,
                )
                .returning(UploadedFile.id)
            )
        ).scalar_one_or_none()
        if updated_job_id is None or updated_upload_id is None:
            raise LocalPolicyError("Strict-local job was cancelled.")
        await progress_db.commit()


async def _lock_strict_terminal_state(
    db: AsyncSession,
    upload_id: UUID,
    job_id: UUID,
    *,
    claim_started_at: datetime | None = None,
) -> bool | None:
    """Lock upload then job and return their fresh cancellation decision."""
    terminal_attempt = await _lock_strict_terminal_attempt(
        db,
        upload_id,
        job_id,
        claim_started_at=claim_started_at,
    )
    if claim_started_at is None:
        return terminal_attempt is not False
    return terminal_attempt


async def _lock_strict_terminal_attempt(
    db: AsyncSession,
    upload_id: UUID,
    job_id: UUID,
    *,
    claim_started_at: datetime | None,
) -> bool | None:
    """Lock one terminal pair; return ``None`` when its claim is stale."""
    locked_upload = (
        await db.execute(
            select(UploadedFile)
            .where(UploadedFile.id == upload_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if locked_upload is None:
        return None

    from app.models.local_ai import LocalAIJob

    locked_job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.id == job_id,
                LocalAIJob.upload_id == upload_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()
    if locked_job is None:
        return None
    if claim_started_at is not None and (
        locked_upload.ingestion_status != "processing"
        or locked_job.status != "processing"
        or locked_job.started_at != claim_started_at
    ):
        return None
    return await _strict_cancel_requested(db, upload_id, job_id)


def _strict_evidence_checkpoint_matches(
    row: object,
    evidence: object,
    manifest_sha256: str,
) -> bool:
    """Require exact location, content, and model identity on evidence reuse."""
    metadata = getattr(row, "source_metadata", None)
    return bool(
        isinstance(metadata, dict)
        and getattr(row, "page_number", None) == getattr(evidence, "page_number", None)
        and getattr(row, "excerpt", None) == getattr(evidence, "excerpt", None)
        and getattr(row, "start_offset", None)
        == getattr(evidence, "start_offset", None)
        and getattr(row, "end_offset", None) == getattr(evidence, "end_offset", None)
        and getattr(row, "field_paths", None)
        == list(getattr(evidence, "field_paths", ()))
        and metadata.get("manifest_sha256") == manifest_sha256
        and metadata.get("excerpt_sha256") == getattr(evidence, "excerpt_sha256", None)
        and metadata.get("offset_representation")
        == getattr(evidence, "offset_representation", None)
    )


async def _mark_cancelled(db: AsyncSession, upload: UploadedFile) -> None:
    """Mark a file as cleanly cancelled (terminal). Rolls back any poisoned
    session first so the terminal write always persists."""
    upload_id = upload.id
    user_id = upload.user_id
    strict_local = upload.processing_mode == "validated_strict_local"
    try:
        await db.execute(
            select(UploadedFile.id)
            .where(UploadedFile.id == upload_id)
            .with_for_update()
        )
        completed_at = datetime.now(timezone.utc)
        upload.ingestion_status = "cancelled"
        upload.progress_stage = None
        upload.progress_detail = None
        upload.processing_completed_at = completed_at
        if strict_local:
            from app.models.local_ai import LocalAIJob

            jobs = (
                (
                    await db.execute(
                        select(LocalAIJob)
                        .where(
                            LocalAIJob.upload_id == upload_id,
                            LocalAIJob.user_id == user_id,
                            LocalAIJob.status.in_(("queued", "processing")),
                        )
                        .order_by(LocalAIJob.created_at.desc(), LocalAIJob.id.desc())
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
            legacy_failed = False
            for job in jobs:
                if fail_legacy_runtime_identity_required(
                    job,
                    completed_at=completed_at,
                ):
                    legacy_failed = True
                    continue
                job.cancel_requested = True
                job.status = "cancelled"
                job.stage = "cancelled"
                job.progress = _merge_strict_local_progress(
                    job.progress,
                    {},
                    stage="cancelled",
                )
                job.failure = None
                job.completed_at = completed_at
            if legacy_failed:
                upload.ingestion_status = "failed"
                upload.ingestion_errors = [
                    {"error_type": "runtime_identity_required"}
                ]
        await db.commit()
    except Exception:
        logger.error("Failed to mark %s cancelled; retrying after rollback", upload_id)
        await db.rollback()
        completed_at = datetime.now(timezone.utc)
        await db.execute(
            text(
                "UPDATE uploaded_files SET ingestion_status = 'cancelled', "
                "progress_stage = NULL, progress_detail = NULL, "
                "processing_completed_at = :now WHERE id = :id"
            ),
            {"now": completed_at, "id": upload_id},
        )
        if strict_local:
            from app.models.local_ai import LocalAIJob

            fallback_jobs = (
                (
                    await db.execute(
                        select(LocalAIJob)
                        .where(
                            LocalAIJob.upload_id == upload_id,
                            LocalAIJob.user_id == user_id,
                            LocalAIJob.status.in_(("queued", "processing")),
                        )
                        .order_by(LocalAIJob.created_at.desc(), LocalAIJob.id.desc())
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
            legacy_failed = False
            for fallback_job in fallback_jobs:
                if fail_legacy_runtime_identity_required(
                    fallback_job,
                    completed_at=completed_at,
                ):
                    legacy_failed = True
                    continue
                fallback_job.cancel_requested = True
                fallback_job.status = "cancelled"
                fallback_job.stage = "cancelled"
                fallback_job.progress = _merge_strict_local_progress(
                    fallback_job.progress,
                    {},
                    stage="cancelled",
                )
                fallback_job.failure = None
                fallback_job.completed_at = completed_at
            if legacy_failed:
                await db.execute(
                    text(
                        "UPDATE uploaded_files SET ingestion_status = 'failed', "
                        "progress_stage = NULL, progress_detail = NULL, "
                        "ingestion_errors = "
                        "'[{\"error_type\": \"runtime_identity_required\"}]'::jsonb, "
                        "processing_completed_at = :now WHERE id = :id"
                    ),
                    {"now": completed_at, "id": upload_id},
                )
        await db.commit()


async def _run_gemini_extraction_engine(db, upload, upload_id, user_id, text, sem):
    """Default engine: scrub → Gemini section parse → per-section LangExtract.

    Returns ``(all_entities, parsed_doc)`` or ``(None, None)`` if the upload was
    cancelled mid-run (the file is already marked cancelled). This is the exact
    pre-WS-A behavior, lifted verbatim into a helper so the engine branch can
    select it without changing the default path.
    """
    from app.services.ai.llm import load_llm_config
    from app.services.extraction.entity_extractor import extract_entities_async
    from app.services.ai.phi_scrubber import scrub_phi_async
    from app.services.ai.patient_phi import patient_scrub_args
    from app.models.patient import Patient
    from app.services.extraction.section_parser import (
        ParsedDocument,
        ParsedSection,
        SectionType,
        parse_sections,
        split_large_section,
    )

    # Resolve the user's LLM config once for this run (provider routing/creds for
    # both section parsing and per-section entity extraction). Falls back to .env.
    config = await load_llm_config(db, user_id)

    # Step 2: Scrub PHI before entity extraction — local, no semaphore. The scrub
    # (spaCy PERSON NER especially) is CPU-bound; run it off the event loop so the
    # background worker doesn't freeze concurrent API requests (D3).
    patients = (
        (await db.execute(select(Patient).where(Patient.user_id == user_id)))
        .scalars()
        .all()
    )
    scrubbed_text, _deident_report = await scrub_phi_async(
        text, **patient_scrub_args(list(patients))
    )

    # Step 3: Section parsing (skip Gemini call for small docs)
    if len(scrubbed_text) < settings.small_doc_threshold:
        parsed_doc = ParsedDocument(
            sections=[
                ParsedSection(
                    section_type=SectionType.OTHER,
                    title="Full Document",
                    text=scrubbed_text,
                    char_range=(0, len(scrubbed_text)),
                )
            ],
            document_type="clinical_note",
            primary_visit_date=None,
            provider=None,
            facility=None,
        )
    else:
        async with sem:
            parsed_doc = await parse_sections(
                scrubbed_text, settings.gemini_api_key, config=config
            )

    upload.extraction_sections = {
        "sections": [
            {"type": s.section_type.value, "title": s.title, "char_range": s.char_range}
            for s in parsed_doc.sections
        ]
    }
    upload.document_metadata = {
        "document_type": parsed_doc.document_type,
        "primary_visit_date": parsed_doc.primary_visit_date,
        "provider": parsed_doc.provider,
        "facility": parsed_doc.facility,
        "section_count": len(parsed_doc.sections),
    }
    await db.commit()

    if await _is_cancel_requested(db, upload_id):
        await _mark_cancelled(db, upload)
        return None, None

    # Step 4: Per-section entity extraction (with small-chunk batching)
    all_entities = []
    extraction_tasks = []
    current_batch = ""
    current_section = None

    for section in parsed_doc.sections:
        chunks = split_large_section(section.text)
        for chunk in chunks:
            if current_batch and len(current_batch) + len(chunk) + 1 <= 2000:
                current_batch += "\n" + chunk
            else:
                if current_batch:
                    extraction_tasks.append((current_batch, current_section))
                current_batch = chunk
                current_section = section.section_type.value
    if current_batch:
        extraction_tasks.append((current_batch, current_section))

    section_total = len(extraction_tasks)
    upload.progress_stage = "extracting_entities"
    upload.progress_detail = {"section_index": 0, "section_total": section_total}
    await db.commit()

    section_sem = asyncio.Semaphore(settings.section_extraction_concurrency)

    async def extract_chunk(text_chunk: str, section_type: str):
        async with section_sem:
            async with sem:
                chunk_result = await extract_entities_async(
                    text_chunk, upload.filename, settings.gemini_api_key, config=config
                )
        return chunk_result, section_type

    tasks = [
        asyncio.create_task(extract_chunk(chunk, stype))
        for chunk, stype in extraction_tasks
    ]
    results: list = []
    done_count = 0
    for fut in asyncio.as_completed(tasks):
        try:
            results.append(await fut)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # surfaced to _collect_entities as a failure
            results.append(exc)
        done_count += 1
        upload.progress_detail = {
            "section_index": done_count,
            "section_total": section_total,
        }
        await db.commit()
        if await _is_cancel_requested(db, upload_id):
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await _mark_cancelled(db, upload)
            return None, None

    collected, failed_chunks = _collect_entities(results, len(extraction_tasks))
    all_entities.extend(collected)
    if failed_chunks:
        errs = list(upload.ingestion_errors or [])
        errs.append(
            {
                "stage": "entity_extraction",
                "failed_chunks": failed_chunks,
                "total_chunks": len(extraction_tasks),
                "error_type": "ExtractionChunkFailure",
            }
        )
        upload.ingestion_errors = errs
        logger.warning(
            "Extraction for %s: %d/%d chunks failed",
            upload.id,
            failed_chunks,
            len(extraction_tasks),
        )
    return all_entities, parsed_doc


def _resolve_extraction_engine(requested: str | None) -> str:
    """Resolve the effective extraction engine, degrading ``local``/``hybrid`` to
    ``gemini`` when the OPTIONAL clinical-NLP stack (scispaCy + medspaCy + the NER
    model — the ``.[clinical-nlp]`` extra) isn't installed/available.

    This is what makes the local path strictly opt-in: the flag can be set without
    the deps present ("not install and use") and extraction still works via Gemini
    rather than crashing or under-extracting. Pure decision logic (no I/O beyond
    the cached model probe) so it is unit-testable.
    """
    engine = (requested or "gemini").lower()
    if engine not in ("local", "hybrid"):
        return "gemini"
    try:
        from app.services.extraction.clinical_context import get_clinical_context
        from app.services.extraction.local_ner import get_local_ner

        if get_local_ner().warm_load() and get_clinical_context().warm_load():
            return engine
    except Exception:  # noqa: BLE001 - missing deps must never break extraction
        pass
    logger.warning(
        "EXTRACTION_ENGINE=%s but the local clinical-NLP models/deps are "
        "unavailable; falling back to the Gemini engine. Enable the on-device path "
        'with `pip install -e ".[clinical-nlp]"` plus the scispaCy model.',
        engine,
    )
    return "gemini"


async def _run_local_extraction_engine(
    db, upload, upload_id, user_id, text, engine, sem
):
    """WS-A engine: on-device medspaCy + scispaCy fast-path (``local``/``hybrid``).

    Runs local NER + ConText on the **unscrubbed** text (never leaves the device,
    so no PHI round-trip). In ``hybrid`` mode, hard sections escalate to Gemini —
    and only the escalated section text is scrubbed before that call. Returns
    ``(all_entities, parsed_doc)`` or ``None`` if cancelled.
    """
    from app.services.ai.llm import load_llm_config
    from app.services.ai.phi_scrubber import scrub_phi_async
    from app.services.ai.patient_phi import patient_scrub_args
    from app.services.extraction.entity_extractor import extract_entities_async
    from app.services.extraction.entity_validator import validate_entities
    from app.services.extraction.extraction_engine import run_clinical_extraction
    from app.services.extraction.local_ner import get_local_ner
    from app.services.extraction.clinical_context import get_clinical_context
    from app.services.extraction.section_parser import (
        ParsedDocument,
        split_large_section,
    )
    from app.models.patient import Patient

    # Resolve the user's LLM config once; used by the hybrid Gemini escalation
    # path below. Falls back to .env when the user has no saved rows.
    config = await load_llm_config(db, user_id)

    patients = (
        (await db.execute(select(Patient).where(Patient.user_id == user_id)))
        .scalars()
        .all()
    )

    async def _gemini_section_extract(section_text: str):
        # Rule 2: de-identify before any Gemini call. Only escalated section text
        # is sent; the bulk local path is never scrubbed (it stays on-device).
        # Offload the CPU-bound scrub so the event loop stays responsive (D3).
        scrubbed, _rep = await scrub_phi_async(
            section_text, **patient_scrub_args(list(patients))
        )
        out: list = []
        for chunk in split_large_section(scrubbed):
            async with sem:
                res = await extract_entities_async(
                    chunk, upload.filename, settings.gemini_api_key, config=config
                )
            if res.entities:
                out.extend(res.entities)
        return validate_entities(out)

    upload.progress_stage = "extracting_entities"
    upload.progress_detail = {"section_index": 0, "section_total": 0}
    await db.commit()

    result = await run_clinical_extraction(
        text,
        engine=engine,
        ner=get_local_ner(),
        context=get_clinical_context(),
        gemini_section_extract=_gemini_section_extract,
        confidence_threshold=settings.extraction_local_confidence_threshold,
    )

    if await _is_cancel_requested(db, upload_id):
        await _mark_cancelled(db, upload)
        return None

    # Publish the real section count so the upload UI shows section progress for
    # the local/hybrid engine too (the Gemini path emits live per-section updates;
    # the local pass is near-instant, so we publish the total once sections are
    # processed — never leave it at 0, which reads as "no progress").
    _n_sections = max(len(result.sections), 1)
    upload.progress_detail = {
        "section_index": _n_sections,
        "section_total": _n_sections,
    }
    await db.commit()

    parsed_doc = ParsedDocument(
        sections=result.sections,
        document_type=result.document_metadata.get("document_type", "clinical_note"),
        primary_visit_date=result.document_metadata.get("primary_visit_date"),
        provider=result.document_metadata.get("provider"),
        facility=result.document_metadata.get("facility"),
    )
    upload.extraction_sections = {
        "sections": [
            {"type": s.section_type.value, "title": s.title, "char_range": s.char_range}
            for s in result.sections
        ]
    }
    upload.document_metadata = {
        **result.document_metadata,
        "extraction_stats": result.stats,
    }
    await db.commit()
    return result.entities, parsed_doc


def _build_record_dicts(
    entities, user_id, patient_id, source_file_id, document_date, document_provider
):
    """Map a batch of extracted entities to HealthRecord dicts (pure CPU).

    Runs the terminology lookups (RapidFuzz fuzzy matching), FHIR resource build,
    content hashing and structural validation for every entity — all GIL-bound
    compute with no DB or async I/O. Pulled into a standalone function so the
    whole batch can be offloaded with one ``asyncio.to_thread`` call, keeping the
    event loop free for API requests while the mapping runs (D3).

    Returns ``(entity, record_dict | None)`` pairs in input order — ``None`` for
    entity classes that don't map to a storable record. The DB inserts stay on
    the caller's event-loop thread (the AsyncSession is not thread-safe).
    """
    from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

    built: list[tuple[object, dict | None]] = []
    for entity in entities:
        record_dict = entity_to_health_record_dict(
            entity,
            user_id,
            patient_id,
            source_file_id,
            document_date=document_date,
            document_provider=document_provider,
        )
        built.append((entity, record_dict))
    return built


def _prefer_document_date(built_records, document_date) -> None:
    """A1: prefer the real document date over the de-identified entity dates.

    The de-id pass generalizes dates to year-only BEFORE entity extraction, so the
    dates the LLM returns (and thus ``record_dict["effective_date"]``) collapse to
    year boundaries (e.g. a 10/21/2020 report lands on 2020-01-01). ``document_date``
    is recovered from the ORIGINAL, pre-de-id text — it is internal record data,
    never sent to an LLM, so using the real date does not weaken de-identification.
    Applied to every date-eligible record; dateless types (family_history) are left
    untouched. No-op when no real document date was found.
    """
    if document_date is None:
        return
    from app.services.extraction.entity_to_fhir import _DOCUMENT_DATE_INELIGIBLE

    for entity, record_dict in built_records:
        if record_dict is None:
            continue
        if getattr(entity, "entity_class", None) in _DOCUMENT_DATE_INELIGIBLE:
            continue
        record_dict["effective_date"] = document_date


async def _autoconfirm_and_finish(
    db,
    upload,
    upload_id,
    user_id,
    unique_entities,
    parsed_doc,
    original_text=None,
    *,
    run_dedup: bool = True,
    strict_validated_extraction=None,
    defer_finalization: bool = False,
):
    """Auto-confirm extracted entities into HealthRecords and finalize the upload.

    Shared across engines: ensures a patient exists, maps entities → FHIR records,
    links the encounter, builds A&P cross-references, then kicks off the dedup
    scan. Lifted verbatim from the prior inline tail.
    """
    from app.services.extraction.entity_to_fhir import (
        _find_date_in_text,
        resolve_document_date,
        resolve_document_provider,
    )
    from app.services.extraction.intra_doc_dedup import dedup_within_document

    patient = await _ensure_patient(db, user_id)

    if strict_validated_extraction is None:
        # A1: recover the real document date from ORIGINAL cloud-path text. The
        # entity-derived date has already been de-identified and may be year-only.
        real_document_date = (
            _find_date_in_text(original_text) if original_text else None
        )
        document_date = real_document_date or resolve_document_date(
            unique_entities, parsed_doc.primary_visit_date
        )
        document_provider = resolve_document_provider(unique_entities)
    else:
        # Strict-local facts already carry individually grounded dates/providers.
        # A document-level heuristic must never overwrite that validated evidence.
        real_document_date = None
        document_date = None
        document_provider = None

    created_records = []
    if patient:  # always true — _ensure_patient never returns None (defensive)
        from app.services.ingestion.reextraction import soft_delete_lineage_extracted

        replacement_source_id = await _replacement_source_upload_id(db, upload)
        replaced = await soft_delete_lineage_extracted(db, replacement_source_id)
        if replaced:
            logger.info(
                "Re-extraction replaced %d prior records for %s",
                replaced,
                replacement_source_id,
            )

        encounter_id = None

        # Map all entities → FHIR record dicts off the event loop (CPU-bound:
        # terminology lookups + FHIR build + hashing + validation). The DB adds
        # below stay on the loop thread — the AsyncSession is not thread-safe.
        if strict_validated_extraction is None:
            built_records = await asyncio.to_thread(
                _build_record_dicts,
                unique_entities,
                user_id,
                patient.id,
                upload_id,
                document_date,
                document_provider,
            )
        else:
            from app.services.local_ai.adapters import (
                validated_extraction_to_health_record_dicts,
            )
            from app.services.local_ai.errors import LocalValidationError

            trusted_records = await asyncio.to_thread(
                validated_extraction_to_health_record_dicts,
                strict_validated_extraction,
                user_id,
                patient.id,
                upload_id,
                document_date,
                document_provider,
            )
            if len(trusted_records) != len(unique_entities):
                raise LocalValidationError(
                    "Validated local extraction could not be mapped completely."
                )
            built_records = list(zip(unique_entities, trusted_records, strict=True))

        # A1: replace the de-identified (year-only) entity dates with the real
        # document date recovered from the original text (eligible records only).
        _prefer_document_date(built_records, real_document_date)

        # A5: collapse intra-document over-extraction (encounter fragments from
        # headers/facility/boilerplate; brand+generic medications resolving to
        # the same RxNorm code) before insert. Per-document only — never merges
        # across documents (that is the separate services/dedup pipeline).
        built_records = dedup_within_document(built_records)

        for entity, record_dict in built_records:
            if record_dict is None:
                continue
            record_dict["source_section"] = entity.attributes.get("_source_section")
            record = HealthRecord(**record_dict)
            db.add(record)
            created_records.append((record, entity))

            if entity.entity_class == "encounter":
                await db.flush()
                encounter_id = record.id

        if encounter_id:
            for record, _ in created_records:
                if record.id != encounter_id:
                    record.linked_encounter_id = encounter_id

        ap_records = [
            (r, e) for r, e in created_records if e.entity_class == "assessment_plan"
        ]
        non_ap_records = [
            (r, e) for r, e in created_records if e.entity_class != "assessment_plan"
        ]
        if ap_records and non_ap_records:
            from app.models.cross_reference import RecordCrossReference

            await db.flush()
            for ap_record, _ in ap_records:
                for other_record, other_entity in non_ap_records:
                    if other_entity.entity_class in ("encounter",):
                        continue
                    ref_type = {
                        "medication": "prescribes",
                        "condition": "addresses",
                        "lab_result": "supports",
                        "vital": "supports",
                        "procedure": "addresses",
                        "allergy": "addresses",
                        "imaging_result": "supports",
                        "family_history": "supports",
                        "social_history": "supports",
                    }.get(other_entity.entity_class, "addresses")
                    xref = RecordCrossReference(
                        document_record_id=ap_record.id,
                        referenced_record_id=other_record.id,
                        reference_type=ref_type,
                    )
                    db.add(xref)

        if defer_finalization:
            await db.flush()
            return [record for record, _entity in created_records]

        await db.commit()
        upload.ingestion_status = "dedup_scanning" if run_dedup else "completed"
        upload.record_count = len(created_records)
        upload.progress_stage = None
        await db.commit()

        if run_dedup:
            from app.services.ingestion.coordinator import schedule_dedup_background

            schedule_dedup_background(upload_id, patient.id, user_id)
    else:
        upload.ingestion_status = "awaiting_confirmation"

    upload.progress_stage = None
    upload.processing_completed_at = datetime.now(timezone.utc)
    await db.commit()
    return [record for record, _entity in created_records]


async def _run_strict_local_ingestion_for_upload(
    db: AsyncSession,
    upload: UploadedFile,
    file_path: Path,
    user_id: UUID,
    *,
    strict_claim: "StrictIngestionClaim | None" = None,
) -> None:
    """Run one immutable strict-local job without entering provider routing."""
    from types import SimpleNamespace

    from app.models.local_ai import ExtractionEvidence, LocalAIJob
    from app.services.local_ai.artifact_store import ArtifactStore
    from app.services.local_ai.checkpoint_store import CheckpointStore
    from app.services.local_ai.errors import (
        LOCAL_WORKER_FAILURE_CATEGORIES,
        LocalAIError,
        LocalPolicyError,
        RuntimeIdentityRequiredError,
    )
    from app.services.local_ai.manifest import (
        canonicalize_manifest_snapshot,
        parse_manifest,
    )
    from app.services.local_ai.model_manager import local_model_manager
    from app.services.local_ai.processing_snapshot import (
        fail_legacy_runtime_identity_required,
    )
    from app.services.local_ai.pipeline import StrictLocalPipeline

    upload_id = upload.id
    job_id: UUID | None = None
    claim_started_at: datetime | None = None
    failure_stage = "preflight"

    async def terminalize_failure(exc: Exception) -> bool:
        """Persist paired, content-free failure state after any strict phase."""
        nonlocal job_id
        await db.rollback()
        current_job = await db.get(LocalAIJob, job_id) if job_id is not None else None
        if current_job is None:
            current_job = (
                (
                    await db.execute(
                        select(LocalAIJob)
                        .where(
                            LocalAIJob.upload_id == upload_id,
                            LocalAIJob.user_id == user_id,
                            LocalAIJob.kind == "ingestion",
                            LocalAIJob.processing_mode == "validated_strict_local",
                        )
                        .order_by(
                            LocalAIJob.created_at.desc(),
                            LocalAIJob.id.desc(),
                        )
                        .limit(1)
                        .with_for_update()
                    )
                )
                .scalars()
                .first()
            )
            if current_job is not None:
                job_id = current_job.id
        current_upload = await db.get(UploadedFile, upload_id)

        if current_job is not None and fail_legacy_runtime_identity_required(
            current_job
        ):
            if current_upload is not None:
                current_upload.ingestion_status = "failed"
                current_upload.progress_stage = None
                current_upload.progress_detail = None
                current_upload.ingestion_errors = [
                    {"error_type": "runtime_identity_required"}
                ]
                current_upload.processing_completed_at = current_job.completed_at
            await db.commit()
            return True

        # Never rewrite a terminal job. In particular, an acknowledged cancel
        # must not become a failure merely because the runner was invoked again.
        if current_job is not None and current_job.status in {
            "cancelled",
            "completed",
            "failed",
        }:
            return True

        cancelled_under_lock = False
        if current_job is not None and job_id is not None:
            try:
                terminal_attempt = await _lock_strict_terminal_state(
                    db,
                    upload_id,
                    job_id,
                    claim_started_at=claim_started_at,
                )
                if terminal_attempt is None:
                    await db.rollback()
                    return False
                cancelled_under_lock = terminal_attempt
            except Exception:
                await db.rollback()

        completed_at = datetime.now(timezone.utc)
        cancelled = bool(
            cancelled_under_lock
            or (current_job is not None and current_job.cancel_requested)
            or (current_upload is not None and current_upload.cancel_requested)
        )
        checkpoint_preserved = bool(
            current_job is not None
            and job_id is not None
            and await CheckpointStore(db).count_pages(job_id)
        )

        if cancelled:
            if current_job is not None:
                current_job.status = "cancelled"
                current_job.stage = "cancelled"
                current_job.progress = _merge_strict_local_progress(
                    current_job.progress,
                    {},
                    stage="cancelled",
                )
                current_job.failure = None
                current_job.completed_at = completed_at
            if current_upload is not None:
                current_upload.ingestion_status = "cancelled"
                current_upload.progress_stage = None
                current_upload.progress_detail = None
                current_upload.ingestion_errors = []
                current_upload.processing_completed_at = completed_at
            await db.commit()
            return True

        category = getattr(exc, "category", None)
        error_code = (
            category
            if isinstance(exc, LocalAIError)
            and isinstance(category, str)
            and category in LOCAL_WORKER_FAILURE_CATEGORIES
            else exc.code
            if isinstance(exc, LocalAIError)
            else f"local_ai_{failure_stage}_failed"
        )
        model_role = None
        if current_job is not None and isinstance(current_job.progress, dict):
            model_role = current_job.progress.get("model_role")
        if current_job is not None:
            current_job.status = "failed"
            current_job.stage = "failed"
            current_job.progress = _merge_strict_local_progress(
                current_job.progress,
                {},
                stage="failed",
            )
            current_job.failure = {
                "stage": failure_stage,
                "code": error_code,
                "message": "Strict-local processing did not complete.",
                "model_role": model_role,
                "retryable": bool(getattr(exc, "retryable", False)),
                "checkpoint_preserved": checkpoint_preserved,
                "cloud_fallback_attempted": False,
            }
            current_job.completed_at = completed_at
        if current_upload is not None:
            current_upload.ingestion_status = "failed"
            current_upload.progress_stage = None
            current_upload.progress_detail = None
            current_upload.ingestion_errors = [
                {
                    "error": "Processing failed. Please retry or contact support.",
                    "error_type": error_code,
                }
            ]
            current_upload.processing_completed_at = completed_at
        await db.commit()
        return True

    if strict_claim is not None:
        from app.services.local_ai.ingestion_lifecycle import (
            lock_active_strict_ingestion_claim,
        )

        if (
            upload_id != strict_claim.upload_id
            or user_id != strict_claim.user_id
            or file_path != Path(strict_claim.storage_path)
        ):
            await db.rollback()
            return
        locked_pair = await lock_active_strict_ingestion_claim(db, strict_claim)
        if locked_pair is None:
            await db.rollback()
            return
        upload, job = locked_pair
        upload_id = upload.id
        job_id = job.id
        claim_started_at = strict_claim.claimed_at
        await db.commit()
    else:
        try:
            locked_upload_id = (
                await db.execute(
                    select(UploadedFile.id)
                    .where(UploadedFile.id == upload_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if locked_upload_id is None:
                raise LocalPolicyError("Strict-local upload is unavailable.")

            legacy_candidate = (
                (
                    await db.execute(
                        select(LocalAIJob)
                        .where(
                            LocalAIJob.upload_id == upload_id,
                            LocalAIJob.user_id == user_id,
                            LocalAIJob.kind == "ingestion",
                            LocalAIJob.processing_mode == "validated_strict_local",
                            LocalAIJob.status.in_(("queued", "processing")),
                        )
                        .order_by(LocalAIJob.created_at.desc(), LocalAIJob.id.desc())
                        .limit(1)
                        .with_for_update()
                    )
                )
                .scalars()
                .first()
            )
            if legacy_candidate is not None and fail_legacy_runtime_identity_required(
                legacy_candidate
            ):
                job_id = legacy_candidate.id
                upload.ingestion_status = "failed"
                upload.progress_stage = None
                upload.progress_detail = None
                upload.ingestion_errors = [
                    {"error_type": "runtime_identity_required"}
                ]
                upload.processing_completed_at = legacy_candidate.completed_at
                await db.commit()
                raise RuntimeIdentityRequiredError(
                    "Strict-local worker runtime identity is required."
                )

            try:
                upload_snapshot, upload_digest = canonicalize_manifest_snapshot(
                    upload.processing_manifest
                )
            except LocalAIError as exc:
                raise LocalPolicyError(
                    "Strict-local upload snapshot is invalid."
                ) from exc
            if upload_snapshot != upload.processing_manifest:
                raise LocalPolicyError("Strict-local upload snapshot is invalid.")

            job = (
                await db.execute(
                    select(LocalAIJob)
                    .where(
                        LocalAIJob.upload_id == upload_id,
                        LocalAIJob.user_id == user_id,
                        LocalAIJob.processing_mode == "validated_strict_local",
                        LocalAIJob.manifest_snapshot == upload_snapshot,
                        LocalAIJob.manifest_sha256 == upload_digest,
                    )
                    .limit(1)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if job is None:
                raise LocalPolicyError("Strict-local job snapshot is unavailable.")
            job_id = job.id
            if job.status == "processing":
                await db.rollback()
                return
            if job.status != "queued" or await _strict_cancel_requested(
                db, upload_id, job_id
            ):
                raise LocalPolicyError("Strict-local job was cancelled.")
        except Exception as exc:
            if not await terminalize_failure(exc):
                return
            if isinstance(exc, LocalAIError):
                raise
            raise LocalAIError("Strict-local processing did not complete.") from None

        claim_started_at = datetime.now(timezone.utc)
        job.status = "processing"
        job.stage = failure_stage
        job.started_at = claim_started_at
        job.failure = None
        await db.commit()

    if strict_claim is not None and await _strict_cancel_requested(
        db,
        upload_id,
        job_id,
    ):
        await terminalize_failure(LocalPolicyError("Strict-local job was cancelled."))
        return

    async def publish_progress(value: dict[str, object]) -> None:
        nonlocal failure_stage
        stage = value.get("stage")
        if stage not in _STRICT_LOCAL_PROGRESS_STAGES:
            raise LocalPolicyError("Strict-local progress stage is invalid.")
        safe: dict[str, object] = {"stage": stage}
        model_role = value.get("model_role")
        if model_role is not None:
            if model_role not in _STRICT_LOCAL_MODEL_ROLES:
                raise LocalPolicyError("Strict-local progress role is invalid.")
            safe["model_role"] = model_role
        for key in _STRICT_LOCAL_PROGRESS_COUNTERS:
            if key not in value:
                continue
            item = value[key]
            if (
                type(item) is not int
                or item < 0
                or item > _STRICT_LOCAL_MAX_PROGRESS_COUNTER
            ):
                raise LocalPolicyError("Strict-local progress counter is invalid.")
            safe[key] = item
        safe = _merge_strict_local_progress(job.progress, safe, stage=stage)
        failure_stage = stage
        await _persist_strict_local_progress(
            runner_db=db,
            upload_id=upload_id,
            job_id=job_id,
            user_id=user_id,
            stage=failure_stage,
            progress=safe,
            claim_started_at=claim_started_at,
        )
        # Keep the runner identity map aligned with the durable isolated write.
        # These assignments perform no I/O and intentionally do not commit the
        # content-bearing session. A later encrypted checkpoint/evidence flush
        # therefore still carries the server-owned failure phase, while callback
        # persistence remains independent of this transaction's health.
        job.stage = failure_stage
        job.progress = safe
        upload.progress_stage = f"local_{failure_stage}"
        upload.progress_detail = safe

    async def publish_liveness() -> None:
        # Use an isolated session so cancellation of a slow durable heartbeat
        # cannot poison the pipeline's content-bearing transaction.
        async with async_session_factory() as lease_db:
            await _refresh_strict_local_job_lease(
                lease_db,
                upload_id=upload_id,
                job_id=job_id,
                user_id=user_id,
                claim_started_at=claim_started_at,
            )

    try:
        snapshot = job.revalidate_manifest_snapshot()
        manifest = parse_manifest(snapshot)
        store = ArtifactStore(Path(settings.local_ai_model_dir))
        if store.active_manifest() != manifest:
            raise LocalPolicyError("Validated strict-local model pack is unavailable.")
        pipeline = StrictLocalPipeline(
            manager=local_model_manager,
            checkpoints=CheckpointStore(db),
            manifest=manifest,
            scratch_root=Path(settings.local_ai_scratch_dir),
            model_dir=store.packs_dir / manifest.pack_revision,
            max_page_pixels=settings.local_ai_max_page_pixels,
            on_progress=publish_progress,
            on_liveness=publish_liveness,
        )
        result = await pipeline.run_ingestion(job, upload, file_path)
        if await _strict_cancel_requested(db, upload_id, job_id):
            raise LocalPolicyError("Strict-local job was cancelled.")

        await publish_progress(
            {
                "stage": "persisting_extraction_checkpoint",
                "model_role": "extraction",
            }
        )
        upload.extracted_text = "\n\n".join(
            result.page_markdown[page_number]
            for page_number in sorted(result.page_markdown)
        )
        upload.extraction_entities = [
            {
                "entity_class": entity.entity_class,
                "text": entity.text,
                "attributes": entity.attributes,
                "start_pos": entity.start_pos,
                "end_pos": entity.end_pos,
                "confidence": entity.confidence,
            }
            for entity in result.entities
        ]
        upload.extraction_sections = {
            "pages": [
                {"page_number": page_number}
                for page_number in sorted(result.page_markdown)
            ]
        }
        upload.document_metadata = {
            "schema_version": upload.processing_schema_version,
            "unresolved_fields": result.unresolved_fields,
            "rejected_fields": result.rejected_fields,
        }
        await db.commit()

        await publish_progress(
            {
                "stage": "persisting_evidence",
                "model_role": "extraction",
            }
        )
        prior_evidence_rows = (
            (
                await db.execute(
                    select(ExtractionEvidence).where(
                        ExtractionEvidence.upload_id == upload.id,
                        ExtractionEvidence.user_id == user_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        prior_by_evidence_id = {
            row.source_metadata.get("evidence_id"): row
            for row in prior_evidence_rows
            if isinstance(row.source_metadata, dict)
            and isinstance(row.source_metadata.get("evidence_id"), str)
        }
        evidence_rows = []
        for evidence in result.evidence:
            evidence_row = prior_by_evidence_id.get(evidence.id)
            if evidence_row is not None and not _strict_evidence_checkpoint_matches(
                evidence_row,
                evidence,
                job.manifest_sha256,
            ):
                raise LocalPolicyError(
                    "Stored strict-local evidence checkpoint does not match."
                )
            if evidence_row is None:
                evidence_row = ExtractionEvidence(
                    user_id=user_id,
                    upload_id=upload.id,
                    page_number=evidence.page_number,
                    section=None,
                    excerpt=evidence.excerpt,
                    start_offset=evidence.start_offset,
                    end_offset=evidence.end_offset,
                    field_paths=list(evidence.field_paths),
                    source_metadata={
                        "evidence_id": evidence.id,
                        "manifest_sha256": job.manifest_sha256,
                        "excerpt_sha256": evidence.excerpt_sha256,
                        "offset_representation": evidence.offset_representation,
                    },
                )
                db.add(evidence_row)
            evidence_rows.append(evidence_row)
        await db.commit()

        await publish_progress(
            {
                "stage": "mapping_fhir",
                "model_role": "extraction",
            }
        )
        parsed_doc = SimpleNamespace(primary_visit_date=None)
        records = await _autoconfirm_and_finish(
            db,
            upload,
            upload.id,
            user_id,
            result.entities,
            parsed_doc,
            original_text=upload.extracted_text,
            run_dedup=False,
            strict_validated_extraction=result.validated_extraction,
            defer_finalization=True,
        )
        record_by_evidence_id: dict[str, HealthRecord] = {}
        for record in records:
            metadata = (record.fhir_resource or {}).get("_extraction_metadata", {})
            evidence_ids = metadata.get("_evidence_ids", [])
            if isinstance(evidence_ids, list):
                for evidence_id in evidence_ids:
                    if isinstance(evidence_id, str):
                        record_by_evidence_id.setdefault(evidence_id, record)
        for evidence_row in evidence_rows:
            evidence_id = (evidence_row.source_metadata or {}).get("evidence_id")
            record = record_by_evidence_id.get(evidence_id)
            if record is not None:
                evidence_row.health_record_id = record.id
        await db.flush()

        failure_stage = "finalizing"
        job.stage = failure_stage
        job.progress = _merge_strict_local_progress(
            job.progress,
            {},
            stage=failure_stage,
        )
        if records:
            from app.services.dedup.orchestrator import run_upload_dedup

            dedup_summary = await run_upload_dedup(
                upload.id,
                records[0].patient_id,
                user_id,
                db,
                processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
                commit=False,
            )
            upload.dedup_summary = dedup_summary.to_dict()
        else:
            dedup_summary = None
            upload.dedup_summary = None
        terminal_attempt = await _lock_strict_terminal_state(
            db,
            upload_id,
            job_id,
            claim_started_at=claim_started_at,
        )
        if terminal_attempt is None:
            await db.rollback()
            return
        if terminal_attempt:
            raise LocalPolicyError("Strict-local job was cancelled.")

        completed_at = datetime.now(timezone.utc)
        if dedup_summary is not None and dedup_summary.needs_review > 0:
            upload.ingestion_status = "awaiting_review"
        elif dedup_summary is not None and dedup_summary.auto_merged > 0:
            upload.ingestion_status = "completed_with_merges"
        else:
            upload.ingestion_status = "completed"
        upload.record_count = len(records)
        upload.progress_stage = None
        upload.progress_detail = None
        upload.processing_completed_at = completed_at
        job.status = "completed"
        job.stage = "completed"
        job.progress = _merge_strict_local_progress(
            job.progress,
            {},
            stage="completed",
        )
        job.progress["pages_completed"] = len(result.page_markdown)
        job.completed_at = completed_at
        await db.commit()
    except Exception as exc:
        if not await terminalize_failure(exc):
            return
        if isinstance(exc, LocalAIError):
            raise
        raise LocalAIError("Strict-local processing did not complete.") from None


async def _process_unstructured(
    upload_id: UUID,
    file_path: Path,
    user_id: UUID,
    *,
    strict_claim: "StrictIngestionClaim | None" = None,
) -> None:
    """Background task: extract text then entities from an unstructured file.

    Entity extraction is engine-selectable via ``EXTRACTION_ENGINE`` (WS-A):
    ``gemini`` (default) parses sections + runs LangExtract per section;
    ``local``/``hybrid`` use the on-device medspaCy + scispaCy fast-path
    (escalating hard sections to Gemini in ``hybrid``). The downstream
    validate → dedup → auto-confirm tail is shared across engines.

    Cooperative cancel: ``cancel_requested`` is checked at the start and between
    stages; when set the file is marked ``cancelled`` (terminal) and no further
    work is done. Section-level progress is written to ``progress_stage`` /
    ``progress_detail`` as the pipeline advances.
    """
    async with async_session_factory() as db:
        result = await db.execute(
            select(UploadedFile).where(UploadedFile.id == upload_id)
        )
        upload = result.scalar_one_or_none()
        if not upload:
            return
        is_strict_local = upload.processing_mode == "validated_strict_local"

        if strict_claim is not None:
            if (
                not is_strict_local
                or upload.id != strict_claim.upload_id
                or user_id != strict_claim.user_id
            ):
                await db.rollback()
                return
            await _run_strict_local_ingestion_for_upload(
                db,
                upload,
                file_path,
                user_id,
                strict_claim=strict_claim,
            )
            return

        # Cooperative cancel: if cancellation was requested before the worker
        # got here, abort immediately without doing any extraction work.
        if upload.cancel_requested:
            await _mark_cancelled(db, upload)
            return

        try:
            from app.services.local_ai.errors import LocalAIError, LocalPolicyError

            if upload.processing_mode not in {
                "cloud_assisted",
                "validated_strict_local",
            }:
                raise LocalPolicyError("Upload processing mode is invalid.")
            if upload.processing_mode == "validated_strict_local":
                if (
                    upload.processing_manifest is None
                    or upload.processing_schema_version
                    != "clinical-document-extraction.v1"
                ):
                    raise LocalPolicyError("Strict-local upload snapshot is invalid.")
            elif (
                upload.processing_manifest is not None
                or upload.processing_schema_version is not None
            ):
                raise LocalPolicyError("Cloud-assisted upload snapshot is invalid.")

            upload.processing_started_at = datetime.now(timezone.utc)
            upload.ingestion_status = "processing"
            upload.progress_stage = (
                "local_preflight"
                if upload.processing_mode == "validated_strict_local"
                else "extracting_text"
            )
            upload.progress_detail = None
            await db.commit()

            # This branch intentionally occurs before importing or constructing
            # any provider-capable extraction configuration. A strict-local job
            # can only reach the embedded pipeline and fails locally if it is
            # unavailable; the cloud-assisted path below remains unchanged.
            if upload.processing_mode == "validated_strict_local":
                await _run_strict_local_ingestion_for_upload(
                    db,
                    upload,
                    file_path,
                    user_id,
                )
                return

            from app.services.extraction.entity_validator import (
                normalize_entity_text,
                validate_entities,
            )
            from app.services.extraction.text_extractor import (
                FileType as _FileType,
            )
            from app.services.extraction.text_extractor import (
                detect_file_type as _detect_file_type,
            )
            from app.services.extraction.text_extractor import extract_text

            sem = _get_gemini_semaphore()

            # Resolve the user's LLM config once so OCR (PDF/TIFF) routes through
            # their configured `vision` provider (Gemini fallback). Falls back to .env.
            from app.services.ai.llm import load_llm_config

            config = await load_llm_config(db, user_id)

            # Step 1: Extract text (vision OCR for PDF/TIFF, local for RTF).
            # ocr_trace collects per-provider OCR attempts so a refusal/fallback
            # can be surfaced to the user as a durable per-file notice.
            ocr_trace: list = []
            file_type_enum = _detect_file_type(file_path)
            if file_type_enum == _FileType.RTF:
                extracted_text, file_type = await extract_text(
                    file_path, settings.gemini_api_key, config=config, trace=ocr_trace
                )
            else:
                async with sem:
                    extracted_text, file_type = await extract_text(
                        file_path,
                        settings.gemini_api_key,
                        config=config,
                        trace=ocr_trace,
                    )
            text = extracted_text
            upload.extracted_text = text
            # Surface any OCR provider refusal/fallback as a durable notice
            # (best-effort: never fail extraction over a notice).
            try:
                from app.services.extraction.text_extractor import build_ocr_notice

                _notice = build_ocr_notice(ocr_trace)
                if _notice is not None:
                    upload.notices = (upload.notices or []) + [_notice]
            except Exception:  # noqa: BLE001 - notices are best-effort
                logger.debug("failed to record OCR notice", exc_info=True)
            upload.progress_stage = "scrubbing_phi"
            await db.commit()

            if await _is_cancel_requested(db, upload_id):
                await _mark_cancelled(db, upload)
                return

            # WS-A: clinical-NLP engine selection. The default "gemini" engine
            # runs the LangExtract/Gemini path below unchanged. "local"/"hybrid"
            # run the on-device medspaCy + scispaCy fast-path, escalating only
            # hard sections to Gemini (hybrid). Flag default-OFF until validated.
            # Prefer the user's saved engine choice (Admin -> AI providers),
            # falling back to the global default.
            engine = _resolve_extraction_engine(
                config.extraction_engine or settings.extraction_engine
            )

            if engine in ("local", "hybrid"):
                local_out = await _run_local_extraction_engine(
                    db, upload, upload_id, user_id, text, engine, sem
                )
                if local_out is None:
                    return  # cancelled inside the helper (already marked)
                all_entities, parsed_doc = local_out
            else:
                all_entities, parsed_doc = await _run_gemini_extraction_engine(
                    db, upload, upload_id, user_id, text, sem
                )
                if all_entities is None:
                    return  # cancelled inside the helper (already marked)

            # Precision guards (A1-A5) + within-document dedup are shared across
            # engines (idempotent on already-validated local output).
            all_entities = validate_entities(all_entities)
            seen = set()
            unique_entities = []
            for entity in all_entities:
                key = (entity.entity_class, normalize_entity_text(entity.text))
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)

            upload.progress_stage = "mapping_fhir"
            upload.extraction_entities = [
                {
                    "entity_class": e.entity_class,
                    "text": e.text,
                    "attributes": e.attributes,
                    "start_pos": e.start_pos,
                    "end_pos": e.end_pos,
                    "confidence": e.confidence,
                }
                for e in unique_entities
            ]
            await db.commit()
            await _autoconfirm_and_finish(
                db,
                upload,
                upload_id,
                user_id,
                unique_entities,
                parsed_doc,
                original_text=text,
            )
            return

        except Exception as e:
            error_type = (
                e.code
                if is_strict_local and isinstance(e, LocalAIError)
                else ("local_ai_error" if is_strict_local else type(e).__name__)
            )
            # Strict-local worker exceptions may include OCR, extracted facts,
            # prompts, or model output. Log only fixed, non-content metadata.
            # Cloud-assisted processing retains the existing traceback for
            # operational diagnosis until it receives an equivalent scrubbed
            # error boundary.
            if is_strict_local:
                logger.error(
                    "Strict-local processing failed for %s (%s)",
                    upload_id,
                    error_type,
                )
            else:
                logger.error(
                    "Unstructured processing failed for %s: %s",
                    upload_id,
                    e,
                    exc_info=True,
                )
            # A failed INSERT/commit poisons the session — the next commit would
            # raise PendingRollbackError, so the failed-status write would never
            # persist and the file would stay 'processing'. _recover_stuck_files
            # would then retry it 3x (~30 min + wasted Gemini calls) and finally
            # mislabel it with a generic 'TimeoutError'. Roll back FIRST to clear
            # the session, then re-load the upload row and record the REAL error.
            try:
                await db.rollback()
                result = await db.execute(
                    select(UploadedFile).where(UploadedFile.id == upload_id)
                )
                upload = result.scalar_one_or_none()
                if upload is not None:
                    if is_strict_local and upload.ingestion_status in {
                        "cancelled",
                        "completed",
                        "failed",
                    }:
                        return
                    if is_strict_local and upload.ingestion_status != "processing":
                        return
                    cancelled = (
                        upload.processing_mode == "validated_strict_local"
                        and upload.cancel_requested
                    )
                    upload.ingestion_status = "cancelled" if cancelled else "failed"
                    upload.progress_stage = None
                    upload.ingestion_errors = (
                        []
                        if cancelled
                        else [
                            {
                                "error": (
                                    "Processing failed. Please retry or contact support."
                                ),
                                "error_type": error_type,
                            }
                        ]
                    )
                    upload.processing_completed_at = datetime.now(timezone.utc)
                    await db.commit()
            except Exception:
                logger.error("Failed to record extraction failure for %s", upload_id)
                await db.rollback()


@router.post(
    "/unstructured",
    response_model=UnstructuredUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_unstructured(
    file: UploadFile,
    request: Request,
    processing_mode: ProcessingMode | None = Form(default=None),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UnstructuredUploadResponse:
    """Upload a PDF, RTF, or TIFF for AI-powered text and entity extraction."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UNSTRUCTURED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_UNSTRUCTURED)}",
        )

    snapshot = await _resolve_ingestion_snapshot_or_409(
        db,
        user_id,
        processing_mode,
    )
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # SEC-DOS-01: stream the body to disk in bounded chunks (reject early on an
    # oversized Content-Length), then validate/hash from disk.
    file_path = _safe_file_path(upload_dir, user_id, file.filename)
    # CRYPTO-02: encrypt the unstructured PHI document at rest as it streams.
    file_size, header, file_hash = await _stream_upload_to_disk(
        file, file_path, settings.max_file_size_mb * 1024 * 1024, request=request
    )

    # M1: Validate magic bytes (against the streamed header, not the whole body).
    if not _validate_magic_bytes(header, ext):
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match expected format for {ext}",
        )

    # ``file_hash`` is the PLAINTEXT SHA-256 from the streaming pass — deterministic
    # for re-upload dedup despite the random per-frame encryption nonces.
    from app.services.ingestion.reextraction import find_prior_extracted_upload

    prior = await find_prior_extracted_upload(
        db,
        user_id,
        file_hash,
        processing_mode=snapshot.mode.value,
        processing_manifest=snapshot.manifest_snapshot,
        schema_version=snapshot.schema_version,
    )

    upload_record = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size_bytes=file_size,
        file_hash=file_hash,
        storage_path=str(file_path),
        ingestion_status="duplicate_file" if prior else "pending_extraction",
        file_category="unstructured",
        processing_mode=snapshot.mode.value,
        processing_manifest=copy.deepcopy(snapshot.manifest_snapshot),
        processing_schema_version=snapshot.schema_version,
    )
    if prior:
        upload_record.ingestion_progress = {
            "duplicate_of": str(prior.id),
            "record_count": prior.record_count or 0,
        }
    try:
        db.add(upload_record)
        await db.flush()
        if prior is None:
            job = build_ingestion_job(
                upload_id=upload_record.id,
                user_id=user_id,
                snapshot=snapshot,
            )
            if job is not None:
                db.add(job)
                await revalidate_strict_snapshot_admission(db, snapshot)
        await db.commit()
        await db.refresh(upload_record)
    except Exception:
        await db.rollback()
        file_path.unlink(missing_ok=True)
        raise

    await log_audit_event(
        db,
        user_id=user_id,
        action="file.upload.unstructured",
        resource_type="uploaded_file",
        resource_id=upload_record.id,
        details={"file_type": ext, "file_category": "unstructured"},
    )

    # Worker will pick up the file automatically via DB polling (duplicate_file rows are skipped)

    return UnstructuredUploadResponse(
        upload_id=str(upload_record.id),
        filename=upload_record.filename,
        status=upload_record.ingestion_status,
        file_type=_unstructured_file_type(ext),
    )


@router.post(
    "/unstructured-batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_unstructured_batch(
    files: list[UploadFile],
    processing_mode: ProcessingMode | None = Form(default=None),
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> BatchUploadResponse:
    """Upload multiple unstructured files for concurrent processing."""
    from app.services.ingestion.reextraction import find_prior_extracted_upload

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    snapshot: ProcessingSnapshot | None = None
    results: list[UnstructuredUploadResponse] = []
    rejected: list[RejectedUnstructuredUpload] = []
    for file in files:
        if not file.filename:
            _append_batch_rejection(
                rejected,
                filename="",
                code="missing_filename",
            )
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_UNSTRUCTURED:
            _append_batch_rejection(
                rejected,
                filename=file.filename,
                code="unsupported_type",
            )
            continue

        file_path = _safe_file_path(upload_dir, user_id, file.filename)
        # SEC-DOS-01: stream each file to disk under the size cap.
        # CRYPTO-02: encrypt each unstructured PHI document at rest as it streams.
        try:
            file_size, header, file_hash = await _stream_upload_to_disk(
                file, file_path, settings.max_file_size_mb * 1024 * 1024
            )
        except HTTPException as exc:
            if exc.status_code != status.HTTP_413_CONTENT_TOO_LARGE:
                raise
            _append_batch_rejection(
                rejected,
                filename=file.filename,
                code="file_too_large",
            )
            continue

        if not _validate_magic_bytes(header, ext):
            file_path.unlink(missing_ok=True)
            _append_batch_rejection(
                rejected,
                filename=file.filename,
                code="invalid_signature",
            )
            continue

        if snapshot is None:
            try:
                snapshot = await _resolve_ingestion_snapshot_or_409(
                    db,
                    user_id,
                    processing_mode,
                )
            except BaseException:
                file_path.unlink(missing_ok=True)
                raise

        # Plaintext SHA-256 from the streaming pass (deterministic re-upload dedup).
        prior = await find_prior_extracted_upload(
            db,
            user_id,
            file_hash,
            processing_mode=snapshot.mode.value,
            processing_manifest=snapshot.manifest_snapshot,
            schema_version=snapshot.schema_version,
        )

        upload_record = UploadedFile(
            id=uuid4(),
            user_id=user_id,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            file_size_bytes=file_size,
            file_hash=file_hash,
            storage_path=str(file_path),
            ingestion_status="duplicate_file" if prior else "pending_extraction",
            file_category="unstructured",
            processing_mode=snapshot.mode.value,
            processing_manifest=copy.deepcopy(snapshot.manifest_snapshot),
            processing_schema_version=snapshot.schema_version,
        )
        if prior:
            upload_record.ingestion_progress = {
                "duplicate_of": str(prior.id),
                "record_count": prior.record_count or 0,
            }
        try:
            db.add(upload_record)
            await db.flush()
            if prior is None:
                job = build_ingestion_job(
                    upload_id=upload_record.id,
                    user_id=user_id,
                    snapshot=snapshot,
                )
                if job is not None:
                    db.add(job)
                    await revalidate_strict_snapshot_admission(db, snapshot)
            await db.commit()
        except Exception:
            await db.rollback()
            file_path.unlink(missing_ok=True)
            raise

        await log_audit_event(
            db,
            user_id=user_id,
            action="file.upload.unstructured",
            resource_type="uploaded_file",
            resource_id=upload_record.id,
            details={"file_type": ext, "file_category": "unstructured"},
        )

        results.append(
            UnstructuredUploadResponse(
                upload_id=str(upload_record.id),
                filename=upload_record.filename,
                status=upload_record.ingestion_status,
                file_type=_unstructured_file_type(ext),
            )
        )

    await db.commit()

    # Worker will pick up files automatically via DB polling (duplicate_file rows are skipped)

    return BatchUploadResponse(
        uploads=results,
        rejected=rejected,
        total=len(results),
    )


@router.post(
    "/{upload_id}/reprocess",
    response_model=UnstructuredUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_unstructured_upload(
    upload_id: UUID,
    body: ReprocessUploadRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> UnstructuredUploadResponse:
    """Queue stored ciphertext under a new immutable processing revision."""

    source = (
        await db.execute(
            select(UploadedFile).where(
                UploadedFile.id == upload_id,
                UploadedFile.user_id == user_id,
                UploadedFile.file_category == "unstructured",
                UploadedFile.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    if source.ingestion_status not in {
        "completed",
        "completed_with_merges",
        "awaiting_review",
        "awaiting_confirmation",
        "failed",
        "cancelled",
        "duplicate_file",
    }:
        raise HTTPException(
            status_code=409,
            detail="Upload is not ready for reprocessing.",
        )

    extension = Path(source.filename).suffix.lower()
    if extension not in ALLOWED_UNSTRUCTURED:
        raise HTTPException(
            status_code=409, detail="Stored upload type is unsupported."
        )
    try:
        source_path = Path(source.storage_path).resolve(strict=True)
        upload_root = Path(settings.upload_dir).resolve(strict=True)
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail="Stored encrypted source is unavailable.",
        ) from exc
    if not source_path.is_file() or not source_path.is_relative_to(upload_root):
        raise HTTPException(
            status_code=409,
            detail="Stored encrypted source is unavailable.",
        )

    snapshot = await _resolve_ingestion_snapshot_or_409(
        db,
        user_id,
        body.processing_mode,
    )
    if (
        source.processing_mode == snapshot.mode.value
        and source.processing_manifest == snapshot.manifest_snapshot
        and source.processing_schema_version == snapshot.schema_version
    ):
        raise HTTPException(
            status_code=409,
            detail="Upload already uses the requested processing revision.",
        )

    try:
        canonical_source_id = await _replacement_source_upload_id(db, source)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    locked_source_id = (
        await db.execute(
            select(UploadedFile.id)
            .where(
                UploadedFile.id == canonical_source_id,
                UploadedFile.user_id == user_id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if locked_source_id is None:
        raise HTTPException(status_code=409, detail="Reprocessing source is invalid")
    existing = (
        await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.user_id == user_id,
                UploadedFile.file_category == "unstructured",
                UploadedFile.ingestion_progress["reprocesses_upload_id"].astext
                == str(canonical_source_id),
                UploadedFile.processing_mode == snapshot.mode.value,
                UploadedFile.processing_manifest == snapshot.manifest_snapshot,
                UploadedFile.processing_schema_version == snapshot.schema_version,
            )
            .order_by(UploadedFile.created_at, UploadedFile.id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return UnstructuredUploadResponse(
            upload_id=str(existing.id),
            filename=existing.filename,
            status=existing.ingestion_status,
            file_type=_unstructured_file_type(extension),
        )

    upload_record = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename=source.filename,
        mime_type=source.mime_type,
        file_size_bytes=source.file_size_bytes,
        file_hash=source.file_hash,
        storage_path=source.storage_path,
        ingestion_status="pending_extraction",
        ingestion_progress={"reprocesses_upload_id": str(canonical_source_id)},
        file_category="unstructured",
        processing_mode=snapshot.mode.value,
        processing_manifest=copy.deepcopy(snapshot.manifest_snapshot),
        processing_schema_version=snapshot.schema_version,
    )
    db.add(upload_record)
    await db.flush()
    job = build_ingestion_job(
        upload_id=upload_record.id,
        user_id=user_id,
        snapshot=snapshot,
    )
    if job is not None:
        db.add(job)
        await revalidate_strict_snapshot_admission(db, snapshot)
    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="file.upload.reprocess",
        resource_type="uploaded_file",
        resource_id=upload_record.id,
        ip_address=request.client.host if request.client else None,
        details={
            "source_upload_id": str(source.id),
            "processing_mode": snapshot.mode.value,
        },
    )

    return UnstructuredUploadResponse(
        upload_id=str(upload_record.id),
        filename=upload_record.filename,
        status=upload_record.ingestion_status,
        file_type=_unstructured_file_type(extension),
    )


@router.get("/{upload_id}/extraction", response_model=ExtractionResultResponse)
async def get_extraction_results(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> ExtractionResultResponse:
    """Get extraction results for an unstructured upload."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    entities = []
    if upload.extraction_entities:
        entities = [ExtractedEntitySchema(**e) for e in upload.extraction_entities]

    preview = None
    if upload.extracted_text:
        preview = upload.extracted_text[:500]

    error = None
    if upload.ingestion_errors:
        errors = upload.ingestion_errors
        if errors and isinstance(errors, list) and len(errors) > 0:
            error = (
                errors[0].get("error", str(errors[0]))
                if isinstance(errors[0], dict)
                else str(errors[0])
            )

    return ExtractionResultResponse(
        upload_id=str(upload.id),
        status=upload.ingestion_status,
        extracted_text_preview=preview,
        entities=entities,
        error=error,
    )


@router.post("/{upload_id}/confirm-extraction")
async def confirm_extraction(
    upload_id: UUID,
    body: ConfirmExtractionRequest,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Confirm extracted entities and save them as HealthRecords."""
    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if not body.patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")

    from app.services.extraction.entity_extractor import ExtractedEntity
    from app.services.extraction.entity_to_fhir import (
        entity_to_health_record_dict,
        resolve_document_date,
        resolve_document_provider,
    )

    patient_uuid = UUID(body.patient_id)
    created_count = 0

    from app.services.ingestion.reextraction import soft_delete_lineage_extracted

    replacement_source_id = await _replacement_source_upload_id(db, upload)
    replaced = await soft_delete_lineage_extracted(db, replacement_source_id)
    if replaced:
        logger.info(
            "Manual re-confirm replaced %d prior records for %s",
            replaced,
            replacement_source_id,
        )

    entities = [
        ExtractedEntity(
            entity_class=entity_data.entity_class,
            text=entity_data.text,
            attributes=entity_data.attributes,
            start_pos=entity_data.start_pos,
            end_pos=entity_data.end_pos,
            confidence=entity_data.confidence,
        )
        for entity_data in body.confirmed_entities
    ]

    # Fallback visit date for dateless entities (see _process_unstructured).
    document_date = resolve_document_date(
        entities, (upload.document_metadata or {}).get("primary_visit_date")
    )
    # Note-level provider fallback for records without their own provider (B2).
    document_provider = resolve_document_provider(entities)

    for entity in entities:
        record_dict = entity_to_health_record_dict(
            entity=entity,
            user_id=user_id,
            patient_id=patient_uuid,
            source_file_id=upload_id,
            document_date=document_date,
            document_provider=document_provider,
        )
        if record_dict is None:
            continue

        health_record = HealthRecord(**record_dict)
        db.add(health_record)
        created_count += 1

    await db.commit()

    # Run dedup in background
    upload.ingestion_status = "dedup_scanning"
    upload.record_count = created_count
    await db.commit()

    from app.services.ingestion.coordinator import schedule_dedup_background

    schedule_dedup_background(upload_id, patient_uuid, user_id)

    await log_audit_event(
        db,
        user_id=user_id,
        action="extraction.confirm",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"records_created": created_count, "patient_id": body.patient_id},
    )

    return {
        "upload_id": str(upload_id),
        "records_created": created_count,
        "status": "completed",
    }


@router.get("/{upload_id}/review")
async def get_upload_review(
    upload_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get dedup review data for an upload."""
    from app.models.deduplication import DedupCandidate

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Fetch all candidates for this upload
    candidates_result = await db.execute(
        select(DedupCandidate).where(
            DedupCandidate.source_upload_id == upload_id,
        )
    )
    candidates = candidates_result.scalars().all()

    auto_merged = []
    needs_review: dict[str, list] = {}

    for c in candidates:
        rec_a = await db.get(HealthRecord, c.record_a_id)
        rec_b = await db.get(HealthRecord, c.record_b_id)
        if not rec_a or not rec_b:
            continue

        entry = {
            "candidate_id": str(c.id),
            "primary": {
                "id": str(rec_a.id),
                "display_text": rec_a.display_text or "",
                "record_type": rec_a.record_type,
                "fhir_resource": rec_a.fhir_resource,
            },
            "secondary": {
                "id": str(rec_b.id),
                "display_text": rec_b.display_text or "",
                "record_type": rec_b.record_type,
                "fhir_resource": rec_b.fhir_resource,
            },
            "similarity_score": c.similarity_score,
            "llm_classification": c.llm_classification,
            "llm_confidence": c.llm_confidence,
            "llm_explanation": c.llm_explanation,
            "field_diff": c.field_diff,
            "merged_at": c.resolved_at.isoformat() if c.resolved_at else None,
        }

        if c.status == "merged" and c.auto_resolved:
            auto_merged.append(entry)
        elif c.status == "pending":
            rtype = rec_a.record_type
            needs_review.setdefault(rtype, []).append(entry)

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.view",
        resource_type="uploaded_file",
        resource_id=upload_id,
    )

    return {
        "upload": {
            "id": str(upload.id),
            "filename": upload.filename,
            "uploaded_at": upload.created_at.isoformat() if upload.created_at else None,
            "record_count": upload.record_count,
            "status": upload.ingestion_status,
            "dedup_summary": upload.dedup_summary,
        },
        "auto_merged": auto_merged,
        "needs_review": needs_review,
    }


@router.post("/{upload_id}/review/resolve")
async def resolve_review(
    upload_id: UUID,
    body: dict,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Bulk resolve dedup candidates for an upload."""
    from app.models.deduplication import DedupCandidate
    from app.models.provenance import Provenance
    from app.services.dedup.field_merger import apply_field_update
    from app.services.ingestion.content_hash import content_hash

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    resolutions = body.get("resolutions", [])
    resolved_count = 0

    for resolution in resolutions:
        candidate_id = UUID(resolution["candidate_id"])
        action = resolution["action"]
        field_overrides = resolution.get("field_overrides")

        candidate = await db.get(DedupCandidate, candidate_id)
        if not candidate or candidate.source_upload_id != upload_id:
            continue

        rec_a = await db.get(HealthRecord, candidate.record_a_id)
        rec_b = await db.get(HealthRecord, candidate.record_b_id)
        if not rec_a or not rec_b:
            continue

        now = datetime.now(timezone.utc)

        if action == "merge":
            rec_b.is_duplicate = True
            rec_b.merged_into_id = rec_a.id
            rec_b.merge_metadata = {
                "merged_from": str(rec_b.id),
                "merged_at": now.isoformat(),
                "merge_type": "duplicate",
                "source_upload_id": str(upload_id),
            }
            candidate.status = "merged"
            candidate.resolved_by = user_id
            candidate.resolved_at = now
            db.add(
                Provenance(
                    record_id=rec_a.id,
                    action="merge",
                    agent=f"user/{user_id}",
                    source_file_id=upload_id,
                    details={"merged_record_id": str(rec_b.id), "action": "merge"},
                )
            )

        elif action == "update":
            merge_result = apply_field_update(rec_a, rec_b, field_overrides)
            rec_a.fhir_resource = merge_result["updated_resource"]
            rec_a.content_hash = content_hash(rec_a.fhir_resource)
            rec_a.display_text = merge_result["display_text"]
            rec_a.merge_metadata = merge_result["merge_metadata"]
            rec_b.is_duplicate = True
            rec_b.merged_into_id = rec_a.id
            candidate.status = "merged"
            candidate.resolved_by = user_id
            candidate.resolved_at = now
            db.add(
                Provenance(
                    record_id=rec_a.id,
                    action="field_update",
                    agent=f"user/{user_id}",
                    source_file_id=upload_id,
                    details={
                        "merged_record_id": str(rec_b.id),
                        "fields_updated": merge_result["merge_metadata"].get(
                            "fields_updated", []
                        ),
                    },
                )
            )

        elif action in ("dismiss", "keep_both"):
            candidate.status = "dismissed"
            candidate.resolved_by = user_id
            candidate.resolved_at = now

        resolved_count += 1

    # Check if all candidates are resolved
    pending_result = await db.execute(
        select(DedupCandidate).where(
            DedupCandidate.source_upload_id == upload_id,
            DedupCandidate.status == "pending",
        )
    )
    remaining = pending_result.scalars().all()
    if not remaining:
        upload.ingestion_status = "completed"

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.resolve",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"resolutions_count": resolved_count},
    )

    return {"resolved": resolved_count, "remaining": len(remaining)}


@router.post("/{upload_id}/review/undo-merge")
async def undo_merge(
    upload_id: UUID,
    body: dict,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Undo an auto-merged dedup candidate."""
    from app.models.deduplication import DedupCandidate
    from app.services.dedup.field_merger import revert_field_update

    result = await db.execute(
        select(UploadedFile).where(
            UploadedFile.id == upload_id,
            UploadedFile.user_id == user_id,
        )
    )
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    candidate_id = UUID(body["candidate_id"])
    candidate = await db.get(DedupCandidate, candidate_id)
    if not candidate or candidate.source_upload_id != upload_id:
        raise HTTPException(status_code=404, detail="Candidate not found")

    if candidate.status != "merged":
        raise HTTPException(status_code=400, detail="Candidate is not merged")

    # Restore secondary record
    rec_b = await db.get(HealthRecord, candidate.record_b_id)
    if rec_b:
        rec_b.is_duplicate = False
        rec_b.merged_into_id = None

    # Revert field changes on primary if this was a field update
    rec_a = await db.get(HealthRecord, candidate.record_a_id)
    if rec_a and rec_a.merge_metadata and rec_a.merge_metadata.get("previous_values"):
        revert_field_update(rec_a)

    # Reset candidate
    candidate.status = "pending"
    candidate.resolved_by = None
    candidate.resolved_at = None

    # Update upload status if needed
    if upload.ingestion_status == "completed":
        upload.ingestion_status = "awaiting_review"

    await db.commit()

    await log_audit_event(
        db,
        user_id=user_id,
        action="upload.review.undo",
        resource_type="uploaded_file",
        resource_id=upload_id,
        details={"candidate_id": str(candidate_id)},
    )

    return {"status": "undone", "candidate_id": str(candidate_id)}
