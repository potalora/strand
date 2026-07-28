from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.middleware.encryption import decrypt_field
from app.models.llm_settings import LLMProviderConfig, UserLLMPreferences
from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.policy import require_loopback
from app.services.local_ai.types import ProcessingMode

logger = logging.getLogger(__name__)

ROUTING_OPS = ("default", "summary", "section", "dedup", "extraction", "vision")
_KNOWN = ("gemini", "vertex", "openai", "anthropic", "openrouter", "ollama", "lmstudio")


@dataclass
class ProviderCreds:
    """Resolved credentials/config for a single provider.

    Values are merged from the user's DB row over the global ``.env`` defaults.
    """

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = True


@dataclass
class LLMConfig:
    """A fully resolved LLM configuration for one user (or the global default).

    ``routing`` maps an operation key (see :data:`ROUTING_OPS`) to a provider
    name; ``providers`` maps a provider name to its resolved credentials.
    """

    routing: dict[str, str] = field(default_factory=dict)
    providers: dict[str, ProviderCreds] = field(default_factory=dict)
    # On-device-vs-cloud entity extraction: "gemini" | "local" | "hybrid".
    extraction_engine: str = ""
    processing_mode: ProcessingMode = ProcessingMode.CLOUD_ASSISTED

    @classmethod
    def from_settings(cls) -> LLMConfig:
        """Build a config purely from the global ``.env`` settings (no user rows)."""
        routing = {"default": settings.llm_provider or "gemini"}
        for op in ("summary", "section", "dedup", "extraction"):
            routing[op] = (
                getattr(settings, f"llm_{op}_provider", "") or routing["default"]
            )
        routing["vision"] = routing["default"]  # vision pref added by user prefs only
        providers = {
            "gemini": ProviderCreds(settings.gemini_api_key, "", settings.gemini_model),
            "vertex": ProviderCreds("", "", settings.vertex_model),
            "openai": ProviderCreds(
                settings.openai_api_key, settings.openai_base_url, settings.openai_model
            ),
            "anthropic": ProviderCreds(
                settings.anthropic_api_key, "", settings.anthropic_model
            ),
            "openrouter": ProviderCreds(
                settings.openrouter_api_key,
                settings.openrouter_base_url,
                settings.openrouter_model,
            ),
            "ollama": ProviderCreds(
                "", settings.ollama_base_url, settings.ollama_model
            ),
            "lmstudio": ProviderCreds(
                "", settings.lmstudio_base_url, settings.lmstudio_model
            ),
        }
        return cls(
            routing=routing,
            providers=providers,
            extraction_engine=settings.extraction_engine,
        )


def _validate_custom_local_config(config: LLMConfig) -> None:
    """Fail closed unless every effective custom-local route is loopback-only."""
    if config.processing_mode is not ProcessingMode.CUSTOM_LOCAL:
        return
    for operation in ROUTING_OPS:
        provider_name = config.routing.get(operation) or config.routing["default"]
        if provider_name not in {"ollama", "lmstudio"}:
            raise LocalPolicyError(
                "Custom local mode requires Ollama or LM Studio for every operation."
            )
        creds = config.providers.get(provider_name)
        if creds is None or not creds.enabled:
            raise LocalPolicyError("Custom local provider must be enabled.")
        require_loopback(creds.base_url)


async def load_llm_config(db: AsyncSession, user_id: UUID) -> LLMConfig:
    """Merge a user's saved provider config + routing over the global ``.env`` defaults.

    Args:
        db: Async DB session.
        user_id: Owner of the config (row-level scoping).

    Returns:
        A resolved :class:`LLMConfig`. Falls back entirely to ``.env`` when the
        user has no saved rows (back-compat with the no-config path).
    """
    cfg = LLMConfig.from_settings()
    rows = (
        (
            await db.execute(
                select(LLMProviderConfig).where(LLMProviderConfig.user_id == user_id)
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if row.provider not in cfg.providers:
            continue
        base = cfg.providers[row.provider]
        key = base.api_key
        if row.api_key_encrypted:
            try:
                key = decrypt_field(row.api_key_encrypted)
            except Exception:  # noqa: BLE001 - never surface key material in the error
                logger.warning(
                    "failed to decrypt stored key for provider %s", row.provider
                )
        cfg.providers[row.provider] = ProviderCreds(
            api_key=key,
            base_url=row.base_url or base.base_url,
            model=row.model or base.model,
            enabled=row.enabled,
        )
    pref = (
        await db.execute(
            select(UserLLMPreferences).where(UserLLMPreferences.user_id == user_id)
        )
    ).scalar_one_or_none()
    if pref:
        if pref.default_provider:
            cfg.routing["default"] = pref.default_provider
        for op in ("summary", "section", "dedup", "extraction", "vision"):
            val = getattr(pref, f"{op}_provider", None)
            cfg.routing[op] = val or cfg.routing.get(op) or cfg.routing["default"]
        if pref.extraction_engine:
            cfg.extraction_engine = pref.extraction_engine
        try:
            cfg.processing_mode = ProcessingMode(
                pref.processing_mode or ProcessingMode.CLOUD_ASSISTED.value
            )
        except (TypeError, ValueError) as exc:
            raise LocalPolicyError("Custom local processing mode is invalid.") from exc
    _validate_custom_local_config(cfg)
    return cfg
