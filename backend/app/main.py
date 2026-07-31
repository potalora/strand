from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.router import api_router
from app.config import settings
from app.database import async_session_factory
from app.middleware.audit import AuditMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.services.local_ai.lifecycle_lock import acquire_local_ai_lifecycle_lock
from app.services.local_ai.model_manager import local_model_manager
from app.services.local_ai.scratch import sweep_stale_scratch
from app.services.ingestion.zip_child_sets import reconcile_zip_child_sets


def resolve_log_level(log_level: str, is_production: bool) -> int:
    """Resolve the effective logging level, clamping below INFO in production.

    DEBUG logs can carry extracted entity text / clinical PHI (SEC-PHI-08), so in
    production we never emit below INFO even if ``LOG_LEVEL=DEBUG`` is configured.
    Development/test keep full DEBUG control for troubleshooting. An unrecognized
    level falls back to INFO.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    if is_production and level < logging.INFO:
        return logging.INFO
    return level


logging.basicConfig(
    level=resolve_log_level(settings.log_level, settings.is_production),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Keep a reference to background fire-and-forget tasks so they aren't GC'd while
# pending (asyncio only holds a weak reference to scheduled tasks).
_background_tasks: set[asyncio.Task] = set()
_STALE_LOCAL_AI_SCRATCH_SECONDS = 24 * 60 * 60
_LOCAL_AI_RECOVERY_BATCH_SIZE = 1_000
_MAX_ACTIVE_LOCAL_AI_SCRATCH_JOBS = 10_000


logger = logging.getLogger(__name__)


def _reconcile_model_pack_operations_on_startup() -> int:
    """Resolve crash-interrupted model-pack operations before job admission."""
    model_root = Path(settings.local_ai_model_dir)
    if not model_root.exists():
        return 0
    from app.services.local_ai.artifact_store import ArtifactStore
    from app.services.local_ai.pack_operations import PackOperationStore

    return PackOperationStore(ArtifactStore(model_root)).reconcile_interrupted()


async def _recover_unstructured_jobs_on_startup(db: AsyncSession) -> int:
    """Pair strict cancellation state, then requeue only uncancelled work."""
    recovered_at = datetime.now(timezone.utc)
    await db.execute(
        text(
            "UPDATE uploaded_files "
            "SET ingestion_status = 'cancelled', progress_stage = NULL, "
            "progress_detail = NULL, processing_completed_at = :now "
            "WHERE ingestion_status IN ('pending_extraction', 'processing') "
            "AND file_category = 'unstructured' "
            "AND cancel_requested = true"
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
            "AND j.status IN ('queued', 'processing') "
            "AND u.ingestion_status = 'cancelled' "
            "AND u.cancel_requested = true"
        ),
        {"now": recovered_at},
    )
    result = await db.execute(
        text(
            "UPDATE uploaded_files SET ingestion_status = 'pending_extraction', "
            "processing_started_at = NULL "
            "WHERE ingestion_status = 'processing' "
            "AND file_category = 'unstructured' "
            "AND cancel_requested = false"
        )
    )
    await db.execute(
        text(
            "UPDATE local_ai_jobs AS j "
            "SET status = 'queued', stage = 'recovery', "
            "failure = NULL, completed_at = NULL "
            "FROM uploaded_files AS u "
            "WHERE j.upload_id = u.id "
            "AND j.processing_mode = 'validated_strict_local' "
            "AND j.status = 'processing' "
            "AND u.ingestion_status = 'pending_extraction'"
        )
    )
    return int(result.rowcount or 0)


async def _recover_strict_local_summary_jobs_on_startup(
    db: AsyncSession,
) -> list[UUID]:
    """Requeue interrupted strict-local summaries and return resumable job IDs."""
    from app.models.local_ai import LocalAIJob

    resumable: list[UUID] = []
    cursor: tuple[datetime, UUID] | None = None
    while True:
        query = select(LocalAIJob).where(
            LocalAIJob.kind == "summary",
            LocalAIJob.processing_mode == "validated_strict_local",
            LocalAIJob.status.in_(("queued", "processing")),
        )
        if cursor is not None:
            created_at, cursor_id = cursor
            query = query.where(
                or_(
                    LocalAIJob.created_at > created_at,
                    (LocalAIJob.created_at == created_at) & (LocalAIJob.id > cursor_id),
                )
            )
        jobs = list(
            (
                await db.execute(
                    query.order_by(
                        LocalAIJob.created_at.asc(),
                        LocalAIJob.id.asc(),
                    ).limit(_LOCAL_AI_RECOVERY_BATCH_SIZE)
                )
            )
            .scalars()
            .all()
        )
        if not jobs:
            break
        cursor = (jobs[-1].created_at, jobs[-1].id)
        recovered_at = datetime.now(timezone.utc)
        for job in jobs:
            if job.cancel_requested:
                job.status = "cancelled"
                job.stage = "cancelled"
                job.progress = {"stage": "cancelled"}
                job.failure = None
                job.completed_at = recovered_at
                continue
            if job.status == "processing":
                job.status = "queued"
                job.stage = "recovery"
                job.progress = {"stage": "recovery"}
                job.failure = None
                job.completed_at = None
            resumable.append(job.id)
        await db.flush()
        if len(jobs) < _LOCAL_AI_RECOVERY_BATCH_SIZE:
            break
    return resumable


async def _active_strict_local_job_ids(db: AsyncSession) -> list[str] | None:
    """Load active IDs boundedly, or return ``None`` when sweep is unsafe."""
    from app.models.local_ai import LocalAIJob

    active_ids: list[str] = []
    cursor: tuple[datetime, UUID] | None = None
    while True:
        query = select(LocalAIJob).where(
            LocalAIJob.processing_mode == "validated_strict_local",
            LocalAIJob.status.in_(("queued", "processing")),
        )
        if cursor is not None:
            created_at, cursor_id = cursor
            query = query.where(
                or_(
                    LocalAIJob.created_at > created_at,
                    (LocalAIJob.created_at == created_at) & (LocalAIJob.id > cursor_id),
                )
            )
        jobs = list(
            (
                await db.execute(
                    query.order_by(
                        LocalAIJob.created_at.asc(),
                        LocalAIJob.id.asc(),
                    ).limit(_LOCAL_AI_RECOVERY_BATCH_SIZE)
                )
            )
            .scalars()
            .all()
        )
        if not jobs:
            break
        active_ids.extend(str(job.id) for job in jobs)
        if len(active_ids) > _MAX_ACTIVE_LOCAL_AI_SCRATCH_JOBS:
            logger.warning(
                "Skipped strict-local scratch inventory because active jobs "
                "exceeded its safety bound"
            )
            return None
        cursor = (jobs[-1].created_at, jobs[-1].id)
        if len(jobs) < _LOCAL_AI_RECOVERY_BATCH_SIZE:
            break
    return active_ids


async def _shutdown_local_ai_workers(*, local_ai_started: bool) -> None:
    """Drain tracked work, recover durable state, then stop model processes."""
    from app.api.upload import stop_extraction_worker

    try:
        await stop_extraction_worker()
    except Exception:
        logger.exception("Failed to drain extraction work during shutdown")
    if not local_ai_started:
        return

    from app.services.local_ai.summary_runner import local_summary_runner

    try:
        await local_summary_runner.stop_and_requeue()
    except Exception:
        logger.exception("Failed to drain summary work during shutdown")
    try:
        async with async_session_factory() as db:
            await acquire_local_ai_lifecycle_lock(db)
            await _recover_unstructured_jobs_on_startup(db)
            await db.commit()
    except Exception:
        logger.exception("Failed to requeue strict-local work during shutdown")
    finally:
        await local_model_manager.stop()


def build_cors_config(cors_origins: str) -> tuple[list[str], bool]:
    """Parse ``CORS_ORIGINS`` into a clean origin list + a safe credentials flag.

    Strips whitespace around each origin (so ``"a, b"`` works) and drops empties.
    The CORS spec forbids the wildcard origin together with credentials — browsers
    reject ``Access-Control-Allow-Origin: *`` when credentials are allowed — so if
    the configured origins include ``*`` we DISABLE credentials rather than emit
    an unusable/insecure combination (SEC-API-05).
    """
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    allow_credentials = "*" not in origins
    return origins, allow_credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    summary_jobs_to_resume: list[UUID] = []
    active_strict_local_job_ids: list[str] | None = None
    strict_local_recovery_succeeded = False
    reconciled_operations = 0
    # Reconcile ZIP child sets independently so one malformed set cannot prevent
    # the globally locked local-AI recovery pass.
    try:
        async with async_session_factory() as db:
            set_recovery = await reconcile_zip_child_sets(
                db,
                Path(settings.upload_dir),
            )
            if set_recovery is not None and (
                set_recovery.recovered_groups
                or set_recovery.failed_groups
                or set_recovery.removed_orphans
            ):
                logger.info(
                    "ZIP child set recovery: recovered=%d failed=%d orphans=%d",
                    set_recovery.recovered_groups,
                    set_recovery.failed_groups,
                    set_recovery.removed_orphans,
                )
            if set_recovery is not None and set_recovery.bounded:
                logger.warning("ZIP child set recovery reached its safety bound")
            await db.commit()
    except Exception:
        logger.exception("Failed to reconcile ZIP child sets on startup")

    # A1: Recover files stuck in 'processing' from previous crash/restart.
    try:
        async with async_session_factory() as db:
            if settings.local_ai_enabled:
                await acquire_local_ai_lifecycle_lock(db)
                reconciled_operations = _reconcile_model_pack_operations_on_startup()
            recovered = await _recover_unstructured_jobs_on_startup(db)
            if recovered:
                logger.info(
                    "Recovered %d stuck files to pending_extraction on startup",
                    recovered,
                )
            summary_jobs_to_resume = (
                await _recover_strict_local_summary_jobs_on_startup(db)
            )
            if settings.local_ai_enabled:
                active_strict_local_job_ids = await _active_strict_local_job_ids(db)
            await db.commit()
            strict_local_recovery_succeeded = settings.local_ai_enabled
    except Exception:
        logger.exception("Failed to recover stuck files on startup")

    # Warm-load the spaCy PHI-NER model at boot (memory free, no GIL contention)
    # so name redaction is a reliable cached singleton — not a first-load that
    # can fail under concurrent extraction and silently disable de-identification.
    if settings.phi_ner_enabled:
        try:
            from app.services.ai.phi_ner import warm_load_ner

            if warm_load_ner():
                logger.info(
                    "PHI-NER spaCy model warm-loaded (%s)", settings.phi_ner_spacy_model
                )
            else:
                logger.warning(
                    "PHI-NER spaCy model %s NOT available at startup; name "
                    "redaction will retry per-call",
                    settings.phi_ner_spacy_model,
                )
        except Exception:
            logger.exception("PHI-NER warm-load raised at startup")

    # WS-A: warm-load the local clinical-NLP models (scispaCy NER + medspaCy
    # ConText/sectionizer) only when the local/hybrid engine is selected, so the
    # default Gemini path never pays the model-load cost. Fail-open, non-latching
    # like PHI-NER: a missing model degrades the local path gracefully (the
    # orchestrator falls back / escalates) and never blocks startup.
    if (settings.extraction_engine or "gemini").lower() in ("local", "hybrid"):
        try:
            from app.services.extraction.clinical_context import (
                warm_load_clinical_context,
            )
            from app.services.extraction.local_ner import warm_load_local_ner

            ner_ok = warm_load_local_ner()
            ctx_ok = warm_load_clinical_context()
            logger.info(
                "WS-A local extraction engine=%s warm-load: scispaCy NER=%s, medspaCy=%s",
                settings.extraction_engine,
                ner_ok,
                ctx_ok,
            )
            if not (ner_ok and ctx_ok):
                logger.warning(
                    "Local extraction models not fully available at startup; "
                    "the local path will retry per-call and hybrid escalates to Gemini"
                )
        except Exception:
            logger.exception("WS-A local-engine warm-load raised at startup")

    # Kick off a NON-BLOCKING, staleness-gated RxNorm medication-index refresh.
    # Fire-and-forget background task: it returns immediately, runs the (rare)
    # rebuild in a worker thread, and fails open — startup is never blocked.
    try:
        from app.services.extraction.terminology import schedule_medication_refresh

        schedule_medication_refresh()
    except Exception:
        logger.exception("medication index refresh scheduling failed at startup")

    # W23: purge expired ``revoked_tokens`` rows so the JWT blacklist (and the
    # per-request revocation lookup) doesn't grow without bound. Fire-and-forget
    # so startup is never delayed; fail-open so a DB hiccup can't block boot. Runs
    # ONCE here — a deployment can add a periodic schedule (cron/arq) on top.
    async def _purge_revoked_tokens() -> None:
        try:
            from app.services.auth_service import purge_expired_revoked_tokens

            async with async_session_factory() as db:
                removed = await purge_expired_revoked_tokens(db)
                if removed:
                    logger.info("Purged %d expired revoked tokens on startup", removed)
        except Exception:
            logger.exception("Failed to purge expired revoked tokens on startup")

    purge_task = asyncio.create_task(_purge_revoked_tokens())
    _background_tasks.add(purge_task)
    purge_task.add_done_callback(_background_tasks.discard)

    local_ai_started = False
    if settings.local_ai_enabled:
        try:
            if reconciled_operations:
                logger.info(
                    "Reconciled %d interrupted model-pack operations",
                    reconciled_operations,
                )
            if (
                strict_local_recovery_succeeded
                and active_strict_local_job_ids is not None
            ):
                removed = sweep_stale_scratch(
                    Path(settings.local_ai_scratch_dir),
                    stale_after_seconds=_STALE_LOCAL_AI_SCRATCH_SECONDS,
                    active_job_ids=active_strict_local_job_ids,
                )
                if removed:
                    logger.info("Removed %d stale strict-local scratch jobs", removed)
            else:
                logger.warning(
                    "Skipped strict-local scratch sweep because active jobs "
                    "could not be verified"
                )
            await local_model_manager.start()
            local_ai_started = True
            from app.services.local_ai.summary_runner import local_summary_runner

            local_summary_runner.start(summary_jobs_to_resume)
        except BaseException:
            await local_model_manager.stop()
            raise

    try:
        # Start the extraction worker
        from app.api.upload import (
            reset_extraction_worker_shutdown,
            start_extraction_worker,
        )

        reset_extraction_worker_shutdown()
        start_extraction_worker()

        import sys

        if any("--reload" in arg for arg in sys.argv):
            logger.warning(
                "Server started with --reload: extraction worker may restart on file changes. "
                "Use without --reload for stable extraction processing."
            )

        yield
    finally:
        await _shutdown_local_ai_workers(local_ai_started=local_ai_started)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not settings.gemini_api_key:
        logger.warning(
            "GEMINI_API_KEY is not set — extraction and summarization will fail"
        )

    app = FastAPI(
        title="AI Web Records API",
        description="Personal health records management API",
        version="0.1.0",
        docs_url="/api/docs" if settings.app_env == "development" else None,
        redoc_url="/api/redoc" if settings.app_env == "development" else None,
        lifespan=lifespan,
    )

    # Request-level audit safety net (W16): a generic api.access row for every
    # authenticated /api/v1 request so no PHI access goes silently un-logged.
    # Added before CORS so it runs INSIDE the CORS layer — CORS short-circuits
    # OPTIONS preflight above it (never audited), and this still sees the final
    # route status for real requests.
    app.add_middleware(AuditMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    cors_origins, cors_allow_credentials = build_cors_config(settings.cors_origins)
    if not cors_allow_credentials:
        logger.warning(
            "CORS_ORIGINS contains '*'; disabling allow_credentials "
            "(a wildcard origin with credentials is rejected by browsers)."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    # Transport security (W19 / CRYPTO-04). PRODUCTION ONLY — gated on
    # is_production so local dev/tests (http on 127.0.0.1) are never redirected or
    # host-rejected, which would break the whole suite. Added LAST so they are the
    # OUTERMOST middleware: an untrusted Host is rejected, and a plain-http request
    # is redirected to https, before any other processing runs.
    #
    # NOTE: HTTPSRedirectMiddleware redirects when the OBSERVED scheme is not
    # https. Behind a TLS-terminating proxy run uvicorn with --proxy-headers (and
    # a trusted --forwarded-allow-ips) so X-Forwarded-Proto rewrites the scheme to
    # https; otherwise it will loop. The redirect target is governed by the proxy.
    if settings.is_production:
        from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        app.add_middleware(HTTPSRedirectMiddleware)
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts_list,
        )

        ssl_warning = settings.database_ssl_warning()
        if ssl_warning:
            logger.warning(ssl_warning)

    app.include_router(api_router)

    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    return app


app = create_app()
