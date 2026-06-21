from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.router import api_router
from app.config import settings
from app.database import async_session_factory
from app.middleware.audit import AuditMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware


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


logger = logging.getLogger(__name__)


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
    # A1: Recover files stuck in 'processing' from previous crash/restart
    try:
        async with async_session_factory() as db:
            result = await db.execute(text(
                "UPDATE uploaded_files SET ingestion_status = 'pending_extraction', "
                "processing_started_at = NULL "
                "WHERE ingestion_status = 'processing' AND file_category = 'unstructured'"
            ))
            if result.rowcount:
                logger.info("Recovered %d stuck files to pending_extraction on startup", result.rowcount)
            await db.commit()
    except Exception:
        logger.exception("Failed to recover stuck files on startup")

    # Warm-load the spaCy PHI-NER model at boot (memory free, no GIL contention)
    # so name redaction is a reliable cached singleton — not a first-load that
    # can fail under concurrent extraction and silently disable de-identification.
    if settings.phi_ner_enabled:
        try:
            from app.services.ai.phi_ner import warm_load_ner

            if warm_load_ner():
                logger.info("PHI-NER spaCy model warm-loaded (%s)", settings.phi_ner_spacy_model)
            else:
                logger.warning(
                    "PHI-NER spaCy model %s NOT available at startup; name "
                    "redaction will retry per-call", settings.phi_ner_spacy_model
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
            from app.services.extraction.clinical_context import warm_load_clinical_context
            from app.services.extraction.local_ner import warm_load_local_ner

            ner_ok = warm_load_local_ner()
            ctx_ok = warm_load_clinical_context()
            logger.info(
                "WS-A local extraction engine=%s warm-load: scispaCy NER=%s, medspaCy=%s",
                settings.extraction_engine, ner_ok, ctx_ok,
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

    # Start the extraction worker
    from app.api.upload import start_extraction_worker
    start_extraction_worker()

    import sys
    if any("--reload" in arg for arg in sys.argv):
        logger.warning(
            "Server started with --reload: extraction worker may restart on file changes. "
            "Use without --reload for stable extraction processing."
        )


    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set — extraction and summarization will fail")

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
