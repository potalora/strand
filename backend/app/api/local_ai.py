"""Validated local model-pack lifecycle endpoints."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_authenticated_user_id
from app.middleware.audit import log_audit_event
from app.models.local_ai import LocalAIJob
from app.models.uploaded_file import UploadedFile
from app.schemas.local_ai import (
    LocalAIJobFailure,
    LocalAIJobProgress,
    LocalAIJobResponse,
    LocalModelArtifactResponse,
    LocalPackOperationCreated,
    LocalPackOperationResponse,
    LocalPackStatusResponse,
)
from app.services.local_ai.artifact_store import ArtifactStore, manifest_sha256
from app.services.local_ai.downloader import download_manifest_to_stage
from app.services.local_ai.errors import LocalAIError, LocalValidationError
from app.services.local_ai.manifest import LocalAIManifest, load_manifest
from app.services.local_ai.lifecycle_lock import acquire_local_ai_lifecycle_lock
from app.services.local_ai.pack_operations import (
    OperationLease,
    PackOperationStore,
    platform_profile,
)
from app.services.local_ai.pack_verifier import verify_pack_candidate
from app.services.local_ai.release_evidence import (
    ReleaseEvidence,
    load_release_evidence,
)
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import RuntimeValidationReceipt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/local-ai", tags=["local-ai"])
_ACTIVE_JOB_STATES = ("queued", "processing")


def _store() -> ArtifactStore:
    return ArtifactStore(Path(settings.local_ai_model_dir))


def _operations(store: ArtifactStore | None = None) -> PackOperationStore:
    return PackOperationStore(store or _store())


def _available_manifest() -> LocalAIManifest:
    try:
        return load_manifest(Path(settings.local_ai_manifest_path))
    except LocalAIError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A locked local model pack is not available.",
        ) from None


def _available_released_manifest() -> LocalAIManifest:
    """Return a lock only when its benchmark and fidelity evidence match."""

    manifest = _available_manifest()
    try:
        load_release_evidence(
            Path(settings.local_ai_release_evidence_path),
            manifest=manifest,
            benchmark_path=Path(settings.local_ai_benchmark_path),
            fidelity_path=Path(settings.local_ai_fidelity_path),
        )
    except LocalAIError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A validated local model pack is not available.",
        ) from None
    return manifest


async def _has_active_jobs(db: AsyncSession) -> bool:
    count = (
        await db.execute(
            select(func.count())
            .select_from(LocalAIJob)
            .where(LocalAIJob.status.in_(_ACTIVE_JOB_STATES))
        )
    ).scalar_one()
    return bool(count)


async def _require_no_active_jobs(db: AsyncSession) -> None:
    if await _has_active_jobs(db):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A local AI job is active.",
        )


async def _acquire_pack_mutation_db_guard(db: AsyncSession) -> None:
    """Serialize pack mutation against strict-local job admission."""

    await acquire_local_ai_lifecycle_lock(db)
    await _require_no_active_jobs(db)


async def _run_globally_locked_pack_mutation(
    mutation: Callable[[], None],
) -> None:
    """Run one filesystem mutation while the PostgreSQL lifecycle lock is held."""

    import app.middleware.audit as audit_module

    async with audit_module.async_session_factory() as session:
        await _acquire_pack_mutation_db_guard(session)
        mutation()
        await session.commit()


def _operation_response(
    operation_store: PackOperationStore,
    operation: dict,
) -> LocalPackOperationResponse:
    return LocalPackOperationResponse.model_validate(
        operation_store.response(operation)
    )


def _bounded_job_state(
    model_type: type[LocalAIJobProgress] | type[LocalAIJobFailure],
    value: object,
) -> LocalAIJobProgress | LocalAIJobFailure | None:
    """Validate only allowlisted, content-free fields from JSON job state."""
    if not isinstance(value, dict):
        return None
    allowed = {
        field_name: value[field_name]
        for field_name in model_type.model_fields
        if field_name in value
    }
    if not allowed:
        return None
    try:
        return model_type.model_validate(allowed)
    except ValueError:
        return None


def _job_response(job: LocalAIJob) -> LocalAIJobResponse:
    """Project a job without prompts, clinical content, or provenance."""

    return LocalAIJobResponse(
        id=job.id,
        upload_id=job.upload_id,
        summary_prompt_id=job.summary_prompt_id,
        kind=job.kind,
        processing_mode=job.processing_mode,
        status=job.status,
        stage=job.stage,
        progress=_bounded_job_state(LocalAIJobProgress, job.progress),
        failure=_bounded_job_state(LocalAIJobFailure, job.failure),
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


async def _retry_local_ai_job(
    db: AsyncSession,
    *,
    job: LocalAIJob,
    upload: UploadedFile | None = None,
) -> None:
    """Atomically requeue one failed, retryable strict-local job."""
    failure = job.failure if isinstance(job.failure, dict) else {}
    if job.status != "failed" or failure.get("retryable") is not True:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")
    if job.cancel_requested or job.processing_mode != "validated_strict_local":
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    if job.kind == "summary":
        if job.summary_prompt_id is None or job.upload_id is not None:
            raise HTTPException(status_code=409, detail="This job cannot be retried.")
        job.status = "queued"
        job.stage = "queued"
        job.progress = {}
        job.failure = None
        job.started_at = None
        job.completed_at = None
        return
    if job.kind != "ingestion" or job.upload_id is None:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    if upload is None:
        upload = (
            await db.execute(
                select(UploadedFile)
                .where(
                    UploadedFile.id == job.upload_id,
                    UploadedFile.user_id == job.user_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
    if (
        upload is None
        or upload.id != job.upload_id
        or upload.user_id != job.user_id
        or upload.processing_mode != "validated_strict_local"
    ):
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    job.status = "queued"
    job.stage = "queued"
    job.progress = {}
    job.failure = None
    job.started_at = None
    job.completed_at = None

    upload.ingestion_status = "pending_extraction"
    upload.cancel_requested = False
    upload.processing_started_at = None
    upload.processing_completed_at = None
    upload.progress_stage = None
    upload.progress_detail = None
    upload.ingestion_errors = []
    upload.retry_count = 0
    upload.manual_extraction_required = False


async def verify_installed_pack(
    manifest: LocalAIManifest,
    pack_path: Path,
) -> RuntimeValidationReceipt:
    """Run the offline native fixture gate for one exact candidate pack."""

    return await verify_pack_candidate(manifest, pack_path)


async def _audit_terminal_operation(
    *,
    user_id: UUID | None,
    operation: dict,
    outcome: str,
) -> None:
    if user_id is None:
        return
    # The request-audit factory is deliberately patched to the isolated test
    # database by the suite and owns a fresh transaction in production.
    import app.middleware.audit as audit_module

    async with audit_module.async_session_factory() as session:
        await log_audit_event(
            session,
            user_id=user_id,
            action=f"local_ai.operation.{outcome}",
            resource_type="local_ai_pack_operation",
            resource_id=UUID(operation["id"]),
            details={
                "action": operation["action"],
                "operation_id": operation["id"],
                "outcome": outcome,
                "pack_revision": operation["pack_revision"],
            },
        )


async def run_operation(
    operation_id: str,
    user_id: UUID | None = None,
    preclaimed_lease: OperationLease | None = None,
) -> None:
    """Claim a cross-process runner lease and execute one queued operation."""

    if preclaimed_lease is not None:
        await _run_claimed_operation(operation_id, user_id)
        return
    operations = _operations()
    with operations.operation_lease(operation_id, blocking=False) as acquired:
        if not acquired:
            return
        await _run_claimed_operation(operation_id, user_id)


async def _run_preclaimed_operation(
    operation_id: str,
    user_id: UUID | None,
    lease: OperationLease,
) -> None:
    """Run an operation under the lease reserved by its request."""

    try:
        await run_operation(operation_id, user_id, lease)
    finally:
        lease.release()


async def _run_claimed_operation(
    operation_id: str,
    user_id: UUID | None = None,
) -> None:
    """Execute one persisted, document-free lifecycle operation."""

    store = _store()
    operations = _operations(store)
    operation = operations.get(operation_id)
    if operation is None or operation["state"] != "queued":
        return
    action = operation["action"]
    try:
        manifest = load_manifest(Path(settings.local_ai_manifest_path))
    except Exception:
        operation = operations.transition(
            operation_id,
            expected_states="queued",
            state="failed",
            bytes_done=0,
            message="Model pack operation failed.",
            retryable=False,
        )
        await _audit_terminal_operation(
            user_id=user_id,
            operation=operation,
            outcome="failed",
        )
        return
    if manifest.pack_revision != operation["pack_revision"] or operation[
        "manifest_sha256"
    ] != manifest_sha256(manifest):
        operation = operations.transition(
            operation_id,
            expected_states="queued",
            state="failed",
            bytes_done=0,
            message="Model pack operation failed.",
            retryable=False,
        )
        await _audit_terminal_operation(
            user_id=user_id,
            operation=operation,
            outcome="failed",
        )
        return

    start_message = {
        "install": "Downloading verified model files.",
        "update": "Downloading verified model files.",
        "verify": "Running local validation fixtures.",
        "rollback": "Restoring previous verified model pack.",
    }[action]
    try:
        operation = operations.transition(
            operation_id,
            expected_states="queued",
            state="running",
            current_role=None,
            message=start_message,
            retryable=False,
        )
    except LocalValidationError:
        return

    try:
        if action in {"install", "update"}:
            role_offsets: dict[str, int] = {}
            offset = 0
            for artifact in manifest.artifacts:
                role_offsets[artifact.role.value] = offset
                offset += sum(file.size for file in artifact.files)

            def progress(value: dict[str, int | str]) -> None:
                role = str(value["role"])
                operations.update(
                    operation_id,
                    current_role=role,
                    bytes_done=role_offsets[role] + int(value["bytes_done"]),
                    bytes_total=operation["bytes_total"],
                )

            staging = await download_manifest_to_stage(manifest, store, progress)
            try:
                operations.update(
                    operation_id,
                    current_role=None,
                    message="Running local validation fixtures.",
                )
                receipt = await verify_installed_pack(manifest, staging)
                await _run_globally_locked_pack_mutation(
                    lambda: store.activate_validated(
                        staging,
                        manifest,
                        receipt,
                    )
                )
            except BaseException:
                try:
                    store.discard_staging(staging)
                except LocalAIError:
                    pass
                raise
        elif action == "verify":
            active = store.active_candidate_manifest_for_validation()
            if active is None or active.pack_revision != manifest.pack_revision:
                raise LocalValidationError("The local model pack is unavailable.")
            receipt = await verify_installed_pack(
                active,
                store.packs_dir / active.pack_revision,
            )
            await _run_globally_locked_pack_mutation(
                lambda: store.refresh_validation_receipt(active, receipt)
            )
        elif action == "rollback":
            previous = store.previous_manifest()
            if previous is None or not operations.is_validated(previous):
                raise LocalValidationError(
                    "No verified previous model pack is available."
                )
            await _run_globally_locked_pack_mutation(store.rollback)
        else:
            raise LocalValidationError("Model pack operation is invalid")
    except asyncio.CancelledError:
        operation = operations.transition(
            operation_id,
            expected_states="running",
            state="paused",
            current_role=None,
            bytes_done=0,
            message="Model pack operation paused.",
            retryable=True,
        )
        await _audit_terminal_operation(
            user_id=user_id,
            operation=operation,
            outcome="paused",
        )
        raise
    except Exception:
        logger.error(
            "Local model pack operation failed",
            extra={"operation_id": operation_id, "action": action},
        )
        operation = operations.transition(
            operation_id,
            expected_states="running",
            state="failed",
            current_role=None,
            bytes_done=0,
            message="Model pack operation failed.",
            retryable=action in {"install", "update", "verify"},
        )
        await _audit_terminal_operation(
            user_id=user_id,
            operation=operation,
            outcome="failed",
        )
        return
    except BaseException:
        operation = operations.transition(
            operation_id,
            expected_states="running",
            state="failed",
            current_role=None,
            bytes_done=0,
            message="Model pack operation failed.",
            retryable=action in {"install", "update", "verify"},
        )
        await _audit_terminal_operation(
            user_id=user_id,
            operation=operation,
            outcome="failed",
        )
        raise

    operation = operations.transition(
        operation_id,
        expected_states="running",
        state="completed",
        current_role=None,
        bytes_done=operation["bytes_total"],
        message=(
            "Previous verified model pack restored."
            if action == "rollback"
            else "Model pack is ready."
        ),
        retryable=False,
    )
    await _audit_terminal_operation(
        user_id=user_id,
        operation=operation,
        outcome="completed",
    )


async def _queue_operation(
    *,
    action: str,
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID,
    db: AsyncSession,
) -> LocalPackOperationCreated:
    await _acquire_pack_mutation_db_guard(db)
    platform_name, compatible = platform_profile()
    if platform_name != "apple_silicon" or not compatible:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This machine is not compatible with the Apple local model pack.",
        )
    manifest = _available_released_manifest()
    store = _store()
    if action == "install" and store.active_revision() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A local model pack is already installed.",
        )
    if action in {"update", "verify", "rollback"} and store.active_revision() is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A local model pack is not installed.",
        )
    if action == "update" and store.active_revision() == manifest.pack_revision:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The available local model pack is already installed.",
        )

    operation_store = _operations(store)
    try:
        operation, lease = operation_store.create_claimed(
            action=action,  # type: ignore[arg-type]
            manifest=manifest,
        )
    except LocalValidationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model pack lifecycle operation is already active.",
        ) from None
    try:
        await log_audit_event(
            db,
            user_id=user_id,
            action=f"local_ai.{action}",
            resource_type="local_ai_pack",
            ip_address=request.client.host if request.client else None,
            details={
                "operation_id": operation["id"],
                "pack_revision": manifest.pack_revision,
                "state": "queued",
            },
        )
        background_tasks.add_task(
            _run_preclaimed_operation,
            operation["id"],
            user_id,
            lease,
        )
    except BaseException:
        try:
            operation_store.transition(
                operation["id"],
                expected_states="queued",
                state="failed",
                current_role=None,
                bytes_done=0,
                message="Model pack operation failed.",
                retryable=action in {"install", "update", "verify"},
            )
        except LocalValidationError:
            pass
        finally:
            lease.release()
        raise
    return LocalPackOperationCreated(
        operation_id=operation["id"],
        state="queued",
    )


@router.get("/status", response_model=LocalPackStatusResponse)
async def get_local_pack_status(
    _user_id: UUID = Depends(get_authenticated_user_id),
) -> LocalPackStatusResponse:
    """Return current pack state without model paths or clinical identifiers."""

    platform_name, compatible = platform_profile()
    store = _store()
    operation_store = _operations(store)
    latest = operation_store.latest()
    try:
        manifest = load_manifest(Path(settings.local_ai_manifest_path))
    except LocalAIError:
        return LocalPackStatusResponse(
            platform=platform_name,  # type: ignore[arg-type]
            compatible=compatible,
            enabled=settings.local_ai_enabled,
            state="failed",
            active_revision=None,
            available_revision=None,
            models=[],
            operation=(
                _operation_response(operation_store, latest) if latest else None
            ),
        )

    active: LocalAIManifest | None = None
    active_invalid = False
    try:
        active = store.active_manifest()
    except LocalAIError:
        active_invalid = True
    active_revision = active.pack_revision if active is not None else None
    exact_active = (
        active is not None
        and active.pack_revision == manifest.pack_revision
        and active == manifest
    )
    evidence: ReleaseEvidence | None = None
    try:
        evidence = load_release_evidence(
            Path(settings.local_ai_release_evidence_path),
            manifest=manifest,
            benchmark_path=Path(settings.local_ai_benchmark_path),
            fidelity_path=Path(settings.local_ai_fidelity_path),
        )
    except LocalAIError:
        evidence = None
    validated = (
        settings.local_ai_enabled
        and exact_active
        and operation_store.is_validated(manifest)
        and evidence is not None
    )
    models = [
        LocalModelArtifactResponse(
            role=artifact.role,
            repository=artifact.repository,
            revision=artifact.revision,
            quantization=artifact.quantization,
            runtime=f"{manifest.runtime['name']} {manifest.runtime['version']}",
            license=artifact.license,
            download_bytes=sum(file.size for file in artifact.files),
            expected_memory_bytes=(
                evidence.expected_memory_bytes(artifact.role)
                if evidence is not None
                else None
            ),
            installed=exact_active,
            validated=validated,
        )
        for artifact in manifest.artifacts
    ]

    if active_invalid:
        pack_state = "failed"
    elif latest and latest["state"] in {"queued", "running", "paused"}:
        pack_state = (
            "verifying"
            if latest["message"] == "Running local validation fixtures."
            else "downloading"
        )
    elif validated:
        pack_state = "ready"
    elif exact_active and operation_store.is_validated(manifest):
        pack_state = "preview"
    elif active is not None:
        pack_state = "update_available"
    elif latest and latest["state"] == "failed":
        pack_state = "failed"
    else:
        pack_state = "not_installed"

    status_reason = None
    if pack_state == "preview":
        status_reason = (
            "feature_disabled"
            if not settings.local_ai_enabled
            else "release_evidence_missing"
        )

    return LocalPackStatusResponse(
        platform=platform_name,  # type: ignore[arg-type]
        compatible=compatible,
        enabled=settings.local_ai_enabled,
        state=pack_state,  # type: ignore[arg-type]
        status_reason=status_reason,
        active_revision=active_revision,
        available_revision=manifest.pack_revision,
        models=models,
        operation=_operation_response(operation_store, latest) if latest else None,
    )


@router.get("/jobs", response_model=list[LocalAIJobResponse])
async def list_local_ai_jobs(
    kind: Literal["ingestion", "summary"] | None = None,
    active_only: bool = True,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> list[LocalAIJobResponse]:
    """List bounded owner-scoped job status so active summaries are discoverable."""
    query = select(LocalAIJob).where(LocalAIJob.user_id == user_id)
    if kind is not None:
        query = query.where(LocalAIJob.kind == kind)
    if active_only:
        manual_upload_gate = exists().where(
            UploadedFile.id == LocalAIJob.upload_id,
            UploadedFile.user_id == user_id,
            UploadedFile.manual_extraction_required.is_(True),
        )
        query = query.where(
            LocalAIJob.status.in_(_ACTIVE_JOB_STATES),
            ~manual_upload_gate,
        )
    jobs = (
        (
            await db.execute(
                query.order_by(
                    LocalAIJob.created_at.desc(), LocalAIJob.id.desc()
                ).limit(50)
            )
        )
        .scalars()
        .all()
    )
    return [_job_response(job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=LocalAIJobResponse)
async def get_local_ai_job(
    job_id: UUID,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalAIJobResponse:
    """Return content-free status for one user-owned local job."""
    job = (
        await db.execute(
            select(LocalAIJob).where(
                LocalAIJob.id == job_id,
                LocalAIJob.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Local AI job not found.")
    return _job_response(job)


@router.post("/jobs/{job_id}/retry", response_model=LocalAIJobResponse)
async def retry_local_ai_job(
    job_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalAIJobResponse:
    """Requeue one failed, retryable strict-local job without diagnostics."""
    target = (
        await db.execute(
            select(
                LocalAIJob.upload_id,
                LocalAIJob.kind,
                LocalAIJob.processing_mode,
            ).where(
                LocalAIJob.id == job_id,
                LocalAIJob.user_id == user_id,
            )
        )
    ).one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Local AI job not found.")
    if target.processing_mode != "validated_strict_local":
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    if target.kind == "summary":
        job = (
            await db.execute(
                select(LocalAIJob)
                .where(
                    LocalAIJob.id == job_id,
                    LocalAIJob.user_id == user_id,
                    LocalAIJob.kind == "summary",
                    LocalAIJob.processing_mode == "validated_strict_local",
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="Local AI job not found.")
        await _retry_local_ai_job(db, job=job)
        await db.commit()
        await db.refresh(job)

        await log_audit_event(
            db,
            user_id=user_id,
            action="local_ai.job_retry",
            resource_type="local_ai_job",
            resource_id=job.id,
            ip_address=request.client.host if request.client else None,
            details={"kind": job.kind, "status": job.status},
        )
        from app.services.local_ai.summary_runner import local_summary_runner

        local_summary_runner.enqueue(job.id)
        return _job_response(job)

    if target.kind != "ingestion" or target.upload_id is None:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    upload = (
        await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.id == target.upload_id,
                UploadedFile.user_id == user_id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if upload is None:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")

    job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.id == job_id,
                LocalAIJob.user_id == user_id,
                LocalAIJob.upload_id == upload.id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Local AI job not found.")

    await _retry_local_ai_job(db, job=job, upload=upload)
    await db.commit()
    await db.refresh(job)

    await log_audit_event(
        db,
        user_id=user_id,
        action="local_ai.job_retry",
        resource_type="local_ai_job",
        resource_id=job.id,
        ip_address=request.client.host if request.client else None,
        details={"kind": job.kind, "status": job.status},
    )

    # The worker is started only after the paired job/upload state is durable.
    from app.api.upload import start_extraction_worker

    start_extraction_worker()
    return _job_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=LocalAIJobResponse)
async def cancel_local_ai_job(
    job_id: UUID,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalAIJobResponse:
    """Persist cancellation and terminate an active embedded worker when present."""
    job = (
        await db.execute(
            select(LocalAIJob)
            .where(
                LocalAIJob.id == job_id,
                LocalAIJob.user_id == user_id,
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Local AI job not found.")

    was_processing = job.status == "processing"
    if job.status in _ACTIVE_JOB_STATES:
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.progress = {"stage": "cancelled"}
            job.failure = None
            job.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(job)

        if was_processing:
            from app.services.local_ai.errors import LocalWorkerError
            from app.services.local_ai.model_manager import local_model_manager

            try:
                registered = await local_model_manager.cancel_registered(str(job.id))
                if not registered:
                    await local_model_manager.cancel(str(job.id))
            except LocalWorkerError:
                logger.warning(
                    "Local worker cancellation could not be confirmed",
                    extra={"job_id": str(job.id)},
                )

    await log_audit_event(
        db,
        user_id=user_id,
        action="local_ai.job_cancel",
        resource_type="local_ai_job",
        resource_id=job.id,
        ip_address=request.client.host if request.client else None,
        details={
            "kind": job.kind,
            "status": job.status,
            "cancel_requested": job.cancel_requested,
        },
    )
    return _job_response(job)


@router.post(
    "/install",
    response_model=LocalPackOperationCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def install_local_pack(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationCreated:
    return await _queue_operation(
        action="install",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.get(
    "/operations/{operation_id}",
    response_model=LocalPackOperationResponse,
)
async def get_local_pack_operation(
    operation_id: UUID,
    _user_id: UUID = Depends(get_authenticated_user_id),
) -> LocalPackOperationResponse:
    operation_store = _operations()
    operation = operation_store.get(str(operation_id))
    if operation is None:
        raise HTTPException(status_code=404, detail="Operation not found.")
    return _operation_response(operation_store, operation)


async def _restart_operation(
    *,
    operation_id: UUID,
    expected_state: str,
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID,
    db: AsyncSession,
) -> LocalPackOperationResponse:
    await _acquire_pack_mutation_db_guard(db)
    operation_store = _operations()
    previous_operation = operation_store.get(str(operation_id))
    if previous_operation is None:
        raise HTTPException(status_code=404, detail="Operation not found.")
    if previous_operation["state"] != expected_state or (
        expected_state == "failed" and not previous_operation["retryable"]
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Operation is not in a restartable state.",
        )
    try:
        operation, lease = operation_store.restart_claimed(
            str(operation_id),
            expected_state=expected_state,  # type: ignore[arg-type]
        )
    except LocalValidationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Operation is not in a restartable state.",
        ) from None
    action = "resume" if expected_state == "paused" else "retry"
    try:
        await log_audit_event(
            db,
            user_id=user_id,
            action=f"local_ai.{action}",
            resource_type="local_ai_pack_operation",
            resource_id=operation_id,
            ip_address=request.client.host if request.client else None,
            details={
                "operation_id": str(operation_id),
                "pack_revision": operation["pack_revision"],
                "state": "queued",
            },
        )
        background_tasks.add_task(
            _run_preclaimed_operation,
            str(operation_id),
            user_id,
            lease,
        )
    except BaseException:
        try:
            operation_store.transition(
                str(operation_id),
                expected_states="queued",
                **{
                    key: value
                    for key, value in previous_operation.items()
                    if key != "id"
                },
            )
        except LocalValidationError:
            pass
        finally:
            lease.release()
        raise
    return _operation_response(operation_store, operation)


@router.post(
    "/operations/{operation_id}/resume",
    response_model=LocalPackOperationResponse,
)
async def resume_local_pack_operation(
    operation_id: UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationResponse:
    return await _restart_operation(
        operation_id=operation_id,
        expected_state="paused",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.post(
    "/operations/{operation_id}/retry",
    response_model=LocalPackOperationResponse,
)
async def retry_local_pack_operation(
    operation_id: UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationResponse:
    return await _restart_operation(
        operation_id=operation_id,
        expected_state="failed",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.post(
    "/verify",
    response_model=LocalPackOperationCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def verify_local_pack(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationCreated:
    return await _queue_operation(
        action="verify",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.post(
    "/update",
    response_model=LocalPackOperationCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def update_local_pack(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationCreated:
    return await _queue_operation(
        action="update",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.post(
    "/rollback",
    response_model=LocalPackOperationCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rollback_local_pack(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> LocalPackOperationCreated:
    return await _queue_operation(
        action="rollback",
        background_tasks=background_tasks,
        request=request,
        user_id=user_id,
        db=db,
    )


@router.delete(
    "/models/{role}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_local_model(
    role: ModelRole,
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> None:
    await _acquire_pack_mutation_db_guard(db)
    store = _store()
    active = store.active_manifest()
    operation_store = _operations(store)
    with operation_store.lifecycle_guard():
        if operation_store.has_nonterminal_locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A model pack lifecycle operation is already active.",
            )
        store.remove(role)
    await log_audit_event(
        db,
        user_id=user_id,
        action="local_ai.remove_role",
        resource_type="local_ai_model",
        ip_address=request.client.host if request.client else None,
        details={
            "role": role.value,
            "pack_revision": active.pack_revision if active else None,
            "status": "removed",
        },
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def remove_local_pack(
    request: Request,
    user_id: UUID = Depends(get_authenticated_user_id),
    db: AsyncSession = Depends(get_db),
) -> None:
    await _acquire_pack_mutation_db_guard(db)
    store = _store()
    active = store.active_manifest()
    operation_store = _operations(store)
    with operation_store.lifecycle_guard():
        if operation_store.has_nonterminal_locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A model pack lifecycle operation is already active.",
            )
        store.remove()
    await log_audit_event(
        db,
        user_id=user_id,
        action="local_ai.remove_pack",
        resource_type="local_ai_pack",
        ip_address=request.client.host if request.client else None,
        details={
            "pack_revision": active.pack_revision if active else None,
            "status": "removed",
        },
    )
