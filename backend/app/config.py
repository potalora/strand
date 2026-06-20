from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/../..


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Fail fast if production is running with insecure defaults."""
        if self.app_env != "development":
            if self.jwt_secret_key == "change-me-in-production":
                raise ValueError(
                    "JWT_SECRET_KEY must be changed from default in non-development environments"
                )
            if not self.database_encryption_key:
                raise ValueError(
                    "DATABASE_ENCRYPTION_KEY must be set in non-development environments"
                )
        return self

    # Database
    database_url: str = "postgresql+asyncpg://localhost:5432/medtimeline"
    database_encryption_key: str = ""

    # Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # AI Prompt Builder
    prompt_target_model: str = "gemini-3.5-flash"
    prompt_suggested_temperature: float = 0.3
    prompt_suggested_max_tokens: int = 4096
    prompt_suggested_thinking_level: str = "low"

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_extraction_model: str = "gemini-3.5-flash"
    gemini_summary_temperature: float = 0.3
    # max_output_tokens budget. On a thinking model this is shared with reasoning
    # tokens, so it must be large enough that thinking can't starve the visible
    # answer (the prior 8192 left summaries truncated mid-sentence).
    gemini_summary_max_tokens: int = 16384
    # Thinking level for the summary call ("low"/"high"). "low" keeps reasoning
    # tokens small so the full summary fits in the output budget.
    gemini_summary_thinking_level: str = "low"
    gemini_concurrency_limit: int = 10

    # Extraction pipeline
    extraction_concurrency: int = 5
    # Concurrent entity-extraction chunks per upload. Capped by gemini_concurrency_limit.
    # Earlier the value 10 appeared to "drop all records" — root-caused (Phase 2d) to
    # event-loop-bound module semaphores raising under contention, NOT rate limits (paid tier,
    # ~1000 RPM; isolated conc=10 verified clean). Fixed via per-loop semaphore caches, so 10
    # is safe and ~3x faster than 3 (fewer serial waves over the ~22s/call latency floor).
    section_extraction_concurrency: int = 10
    extraction_timeout_minutes: int = 10
    extraction_max_retries: int = 3
    small_doc_threshold: int = 3000

    # PHI scrubbing: NER pass for free-text person names (providers, family,
    # anyone not in the patient record) that the regex patterns can't catch.
    # Complements targeted known-identifier scrubbing. Fails open (skips) if the
    # spaCy model is unavailable, so it never blocks a de-identification call.
    phi_ner_enabled: bool = True
    phi_ner_spacy_model: str = "en_core_web_md"

    # --- OSS adoption (flag-gated; see docs/oss-adoption-design.md) ---
    # WS-A clinical NLP engine. "gemini" = current LangExtract/Gemini path;
    # "local" = medspaCy + local NER (scispaCy en_ner_bc5cdr_md); "hybrid" =
    # local fast-path with Gemini escalation for low-confidence spans/sections.
    extraction_engine: str = "gemini"
    # WS-A: spans/sections below this confidence escalate to Gemini (hybrid).
    extraction_local_confidence_threshold: float = 0.6
    # WS-C: high-threshold RapidFuzz fallback for terminology lookups. Default ON
    # — fires only after exact/token lookups miss, and requires BOTH token_set_ratio
    # AND char-level ratio >= 88 (subset-inflation guard) so a near-miss of nothing
    # known stays uncoded. Preserves "never emit a wrong code"; only adds codes to
    # misspellings of known terms. Validated on the real bundled indexes + real data.
    terminology_fuzzy_enabled: bool = True
    # WS-D FHIR structural validation. "off" | "log" (drift signal, never blocks
    # ingestion; default) | "strict" (never applied to AI-built partial resources).
    fhir_validation: str = "log"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # File Storage
    upload_dir: str = "./data/uploads"
    temp_extract_dir: str = "./data/tmp"
    max_file_size_mb: int = 500
    max_epic_export_size_mb: int = 5000
    ingestion_batch_size: int = 100
    ingestion_worker_concurrency: int = 1

    # Rate limiting
    login_rate_limit: int = 30
    login_rate_window: int = 60
    register_rate_limit: int = 30
    register_rate_window: int = 60

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:3000"


settings = Settings()
