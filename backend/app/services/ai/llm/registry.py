from __future__ import annotations

import hashlib

from app.config import settings
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.config import LLMConfig, ProviderCreds, load_llm_config
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import LLMBadRequestError

__all__ = [
    "KNOWN_PROVIDERS",
    "LLMConfig",
    "ProviderCreds",
    "available_providers",
    "get_provider",
    "load_llm_config",
    "provider_name_for",
    "reset_cache",
    "resolve_model",
]

KNOWN_PROVIDERS = {"gemini", "openai", "anthropic", "openrouter",
                   "lmstudio", "ollama", "vertex"}
_OPENAI_FAMILY = {"openai", "openrouter", "lmstudio", "ollama"}
_cache: dict[str, LLMProvider] = {}


def reset_cache() -> None:
    """Clear the per-process provider cache (used by tests after re-routing)."""
    _cache.clear()


def provider_name_for(operation: str | None, config: LLMConfig | None = None) -> str:
    """Resolve the provider name for an operation.

    Args:
        operation: Optional operation key ("summary"/"section"/"dedup"/"extraction"/"vision").
        config: Optional resolved config; falls back to the global ``.env`` config.

    Returns:
        The per-operation override when set, else the resolved default provider.
    """
    cfg = config or LLMConfig.from_settings()
    if operation and operation in cfg.routing and cfg.routing[operation]:
        return cfg.routing[operation]
    return cfg.routing.get("default", "gemini")


def resolve_model(provider_name: str) -> str:
    """Return the configured default model for a provider name (or "" if unknown)."""
    return {
        "gemini": settings.gemini_model,
        "vertex": settings.vertex_model,
        "openai": settings.openai_model,
        "openrouter": settings.openrouter_model,
        "ollama": settings.ollama_model,
        "lmstudio": settings.lmstudio_model,
        "anthropic": settings.anthropic_model,
    }.get(provider_name, "")


def _creds(name: str, config: LLMConfig) -> ProviderCreds:
    """Return the resolved credentials for ``name`` (empty creds when absent)."""
    return config.providers.get(name, ProviderCreds())


def _build(name: str, config: LLMConfig | None = None) -> LLMProvider:
    """Construct a provider instance for ``name`` from the resolved config.

    Args:
        name: Provider name.
        config: Optional resolved config; falls back to the global ``.env`` config
            so legacy callers passing only a name keep working.

    Raises:
        LLMBadRequestError: If the provider name is unknown.
    """
    cfg = config or LLMConfig.from_settings()
    creds = _creds(name, cfg)
    if name == "gemini":
        return GeminiProvider(api_key=creds.api_key,
                              model_default=creds.model or settings.gemini_model)
    if name == "vertex":
        return GeminiProvider(vertexai=True, project=settings.vertex_project,
                              location=settings.vertex_location,
                              model_default=creds.model or settings.vertex_model)
    if name == "anthropic":
        return AnthropicProvider(api_key=creds.api_key,
                                 model_default=creds.model or settings.anthropic_model)
    if name in _OPENAI_FAMILY:
        return OpenAICompatProvider(name=name, api_key=creds.api_key or "local",
                                    base_url=creds.base_url,
                                    model_default=creds.model)
    raise LLMBadRequestError(f"Unknown LLM provider: {name!r}")


def _cache_key(name: str, creds: ProviderCreds) -> str:
    """Build a cache key that isolates distinct credentials (no cross-user leak)."""
    h = hashlib.sha256((creds.api_key or "").encode()).hexdigest()[:16]
    return f"{name}|{creds.base_url}|{creds.model}|{h}"


def get_provider(operation: str | None = None, config: LLMConfig | None = None) -> LLMProvider:
    """Return a cached provider for an operation, building it on first use.

    Args:
        operation: Optional operation key for per-operation routing.
        config: Optional resolved per-user config; falls back to the global
            ``.env`` config (back-compat with the no-config path).

    Returns:
        The resolved, cached ``LLMProvider`` instance.

    Raises:
        LLMBadRequestError: If the resolved provider name is unknown.
    """
    cfg = config or LLMConfig.from_settings()
    name = provider_name_for(operation, cfg)
    if name not in KNOWN_PROVIDERS:
        raise LLMBadRequestError(f"Unknown LLM provider: {name!r}")
    key = _cache_key(name, _creds(name, cfg))
    if key not in _cache:
        _cache[key] = _build(name, cfg)
    return _cache[key]


def available_providers(config: LLMConfig | None = None) -> list[dict]:
    """List selectable providers with display metadata (never API keys).

    Args:
        config: Optional resolved config; falls back to the global ``.env`` config.

    Returns:
        A list of dicts with ``name``, ``model``, ``supports_vision``, ``is_local``
        and ``configured`` (key present for cloud providers; local servers always
        selectable).
    """
    cfg = config or LLMConfig.from_settings()
    out: list[dict] = []
    for name in ("gemini", "openai", "anthropic", "openrouter", "vertex", "ollama", "lmstudio"):
        creds = cfg.providers.get(name, ProviderCreds())
        is_local = name in ("ollama", "lmstudio")
        configured = (
            bool(creds.api_key)
            or is_local
            or (name == "vertex" and bool(settings.vertex_project))
        )
        out.append({
            "name": name,
            "model": creds.model or "(default)",
            "supports_vision": name in ("gemini", "vertex", "anthropic", "openai", "openrouter"),
            "is_local": is_local,
            "configured": configured,
        })
    return out
