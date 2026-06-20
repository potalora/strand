from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.router import api_router
from app.config import settings
from app.database import async_session_factory
from app.middleware.security_headers import SecurityHeadersMiddleware

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


logger = logging.getLogger(__name__)


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

    # WS-B: warm-load the Presidio engine + clinical LOCATION model when their
    # flags are on (both default OFF, so this is dormant unless opted in). Same
    # rationale as the NER warm-load: build the singleton at boot, not under load.
    if settings.phi_engine == "presidio":
        try:
            from app.services.ai.phi_presidio import warm_load_presidio

            if warm_load_presidio():
                logger.info("Presidio de-identification engine warm-loaded")
            else:
                logger.warning(
                    "Presidio engine NOT available at startup; scrub_phi will "
                    "retry per-call and fall back to legacy on failure"
                )
        except Exception:
            logger.exception("Presidio warm-load raised at startup")

    if settings.phi_location_ner_enabled:
        try:
            from app.services.ai.phi_location_ner import warm_load_location_ner

            if warm_load_location_ner():
                logger.info("Clinical LOCATION de-id model warm-loaded")
            else:
                logger.warning(
                    "Clinical LOCATION model NOT available at startup; the "
                    "LOCATION pass will retry per-call (fail-open, non-latching)"
                )
        except Exception:
            logger.exception("Clinical LOCATION model warm-load raised at startup")

    # Kick off a NON-BLOCKING, staleness-gated RxNorm medication-index refresh.
    # Fire-and-forget background task: it returns immediately, runs the (rare)
    # rebuild in a worker thread, and fails open — startup is never blocked.
    try:
        from app.services.extraction.terminology import schedule_medication_refresh

        schedule_medication_refresh()
    except Exception:
        logger.exception("medication index refresh scheduling failed at startup")

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

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    app.include_router(api_router)

    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}

    return app


app = create_app()
