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
