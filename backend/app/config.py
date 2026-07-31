from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Hosts considered loopback / non-networked. A DB reachable only over loopback
# carries no PHI across an untrusted network, so it does not require TLS.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/../..


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        """True unless APP_ENV is an explicit development value.

        Fail-closed: anything other than a recognized dev value (including unset-as-
        unknown, ``staging``, ``production``, typos) is treated as production so a
        deploy that forgets ``APP_ENV`` does not silently run with insecure defaults.
        The published container image also sets ``APP_ENV=production`` in its Dockerfile.
        """
        return self.app_env.strip().lower() not in {
            "development",
            "dev",
            "local",
            "test",
        }

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Fail fast if production is running with weak or default secrets."""
        if not self.is_production:
            return self
        if self.jwt_secret_key == "change-me-in-production":
            raise ValueError(
                "JWT_SECRET_KEY must be changed from the default in production"
            )
        if len(self.jwt_secret_key) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters in production"
            )
        if not self.database_encryption_key:
            raise ValueError("DATABASE_ENCRYPTION_KEY must be set in production")
        try:
            key_bytes = bytes.fromhex(self.database_encryption_key)
        except ValueError as exc:
            raise ValueError(
                "DATABASE_ENCRYPTION_KEY must be valid hex (64 hex chars / 32 bytes)"
            ) from exc
        if len(key_bytes) != 32:
            raise ValueError(
                "DATABASE_ENCRYPTION_KEY must decode to exactly 32 bytes (64 hex chars) "
                "to avoid a silent AES-128 downgrade"
            )
        return self

    @property
    def allowed_hosts_list(self) -> list[str]:
        """Parsed ``ALLOWED_HOSTS`` for Starlette's ``TrustedHostMiddleware``.

        Comma-separated, whitespace-stripped, empties dropped. A blank/empty
        value falls back to ``["*"]`` (permissive) so local dev never blocks on
        Host validation; production should set this to the real hostname(s).
        """
        hosts = [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]
        return hosts or ["*"]

    def database_ssl_warning(self) -> str | None:
        """Advisory (non-breaking) check that production DB traffic is TLS-protected.

        A non-loopback ``DATABASE_URL`` carries PHI across the network and SHOULD
        require ``sslmode=require`` (HIPAA CRYPTO-04). Returning a string the
        caller logs — rather than raising — surfaces the misconfiguration without
        breaking startup. Returns ``None`` in dev, for loopback DBs, or when an
        SSL parameter is already present.
        """
        if not self.is_production:
            return None
        parsed = urlparse(self.database_url)
        host = (parsed.hostname or "").lower()
        if host in _LOOPBACK_HOSTS:
            return None
        query = parse_qs(parsed.query)
        if "sslmode" in query or "ssl" in query:
            return None
        return (
            f"DATABASE_URL points at a non-loopback host ({host}) in production "
            "without sslmode — PHI may traverse the network unencrypted. Add "
            "'?sslmode=require' (or use a loopback/socket connection)."
        )

    # Database
    # NOTE (CRYPTO-04): a non-loopback DATABASE_URL in production should require
    # TLS, e.g. append '?sslmode=require'. Loopback/socket connections do not.
    # database_ssl_warning() surfaces a non-blocking startup warning when this is
    # missing.
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

    # --- Multi-LLM provider routing (see docs/.../multi-llm-provider-design.md) ---
    # Global default provider. Unset/"gemini" preserves current behavior exactly.
    # One of: gemini | openai | anthropic | openrouter | lmstudio | ollama | vertex
    llm_provider: str = "gemini"
    # Optional per-operation overrides (blank => fall back to llm_provider).
    llm_summary_provider: str = ""
    llm_section_provider: str = ""
    llm_dedup_provider: str = ""
    llm_extraction_provider: str = ""

    # OpenAI-compatible family (base_url distinguishes them; all speak the
    # OpenAI Chat Completions API, so one provider class serves all four).
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-5.4-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = ""  # blank => resolved from /v1/models at call time

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Vertex (Gemini-on-Vertex via google-genai vertexai mode)
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-3.5-flash"

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
    # WS-A clinical NLP engine (default "hybrid"). "hybrid" = on-device medspaCy +
    # scispaCy fast-path with Gemini escalation for hard sections; "local" = fully
    # on-device; "gemini" = cloud LangExtract only. hybrid/local need the optional
    # ".[clinical-nlp]" stack — if it's absent the pipeline fail-opens to gemini, so
    # installing the extra IS the opt-in. Off-switch: EXTRACTION_ENGINE=gemini.
    extraction_engine: str = "hybrid"
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

    # Validated strict-local model pack. The shipped profile is opt-in and
    # still requires an installed pack, exact receipt, and release evidence.
    local_ai_enabled: bool = False
    local_ai_model_dir: str = "./data/local-ai/models"
    local_ai_scratch_dir: str = "./data/local-ai/scratch"
    local_ai_manifest_path: str = "./app/model_manifests/apple-m4-16gb-v1.lock.json"
    local_ai_release_evidence_path: str = (
        "./app/model_manifests/apple-m4-16gb-v1.release.json"
    )
    local_ai_benchmark_path: str = "./artifacts/local-ai-benchmark.json"
    local_ai_fidelity_path: str = "./artifacts/local-ai-fidelity.json"
    local_ai_worker_command: str = (
        "../workers/local_ai/apple_mlx/.venv/bin/local-ai-mlx-worker"
    )
    local_ai_max_files: int = 64
    local_ai_max_file_bytes: int = 8 * 1024 * 1024 * 1024
    local_ai_max_pack_bytes: int = 20 * 1024 * 1024 * 1024
    local_ai_worker_timeout_seconds: int = 1800
    local_ai_worker_hard_timeout_seconds: int = 7200
    local_ai_shutdown_drain_seconds: float = Field(default=5.0, gt=0, le=60)
    local_ai_max_page_pixels: int = 40_000_000

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
    # Transport security (W19 / CRYPTO-04). Comma-separated Host headers accepted
    # in production by TrustedHostMiddleware. Default "*" is permissive (dev/
    # loopback); set explicitly to the deployment hostname(s) in production.
    allowed_hosts: str = "*"


settings = Settings()
