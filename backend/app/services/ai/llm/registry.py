from __future__ import annotations

from app.config import settings
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import LLMBadRequestError

KNOWN_PROVIDERS = {"gemini", "openai", "anthropic", "openrouter",
                   "lmstudio", "ollama", "vertex"}
_OPENAI_FAMILY = {"openai", "openrouter", "lmstudio", "ollama"}
_OP_SETTING = {
    "summary": "llm_summary_provider",
    "section": "llm_section_provider",
    "dedup": "llm_dedup_provider",
    "extraction": "llm_extraction_provider",
}
_cache: dict[str, LLMProvider] = {}


def reset_cache() -> None:
    """Clear the per-process provider cache (used by tests after re-routing)."""
    _cache.clear()


def provider_name_for(operation: str | None) -> str:
    """Resolve the provider name for an operation.

    Args:
        operation: Optional operation key ("summary"/"section"/"dedup"/"extraction").

    Returns:
        The per-operation override when set, else the global ``llm_provider``.
    """
    if operation:
        override = getattr(settings, _OP_SETTING.get(operation, ""), "")
        if override:
            return override
    return settings.llm_provider or "gemini"


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


def _build(provider_name: str) -> LLMProvider:
    """Construct a provider instance from settings for the given name.

    Raises:
        LLMBadRequestError: If the provider name is unknown.
    """
    if provider_name == "gemini":
        return GeminiProvider(api_key=settings.gemini_api_key,
                              model_default=settings.gemini_model)
    if provider_name == "vertex":
        return GeminiProvider(vertexai=True, project=settings.vertex_project,
                              location=settings.vertex_location,
                              model_default=settings.vertex_model)
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=settings.anthropic_api_key,
                                 model_default=settings.anthropic_model)
    if provider_name in _OPENAI_FAMILY:
        key = {"openai": settings.openai_api_key,
               "openrouter": settings.openrouter_api_key}.get(provider_name, "local")
        base_url = {"openai": settings.openai_base_url,
                    "openrouter": settings.openrouter_base_url,
                    "ollama": settings.ollama_base_url,
                    "lmstudio": settings.lmstudio_base_url}[provider_name]
        return OpenAICompatProvider(name=provider_name, api_key=key,
                                    base_url=base_url,
                                    model_default=resolve_model(provider_name))
    raise LLMBadRequestError(f"Unknown LLM provider: {provider_name!r}")


def get_provider(operation: str | None = None) -> LLMProvider:
    """Return a cached provider for an operation, building it on first use.

    Args:
        operation: Optional operation key for per-operation routing.

    Returns:
        The resolved, cached ``LLMProvider`` instance.

    Raises:
        LLMBadRequestError: If the resolved provider name is unknown.
    """
    name = provider_name_for(operation)
    if name not in KNOWN_PROVIDERS:
        raise LLMBadRequestError(f"Unknown LLM provider: {name!r}")
    if name not in _cache:
        _cache[name] = _build(name)
    return _cache[name]


def available_providers() -> list[dict]:
    """List selectable providers with display metadata (never API keys).

    Returns:
        A list of dicts with ``name``, ``model``, ``supports_vision`` and
        ``configured`` (key present for cloud providers; local servers always True).
    """
    out: list[dict] = []
    cloud_keyed = {
        "gemini": bool(settings.gemini_api_key),
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "vertex": bool(settings.vertex_project),
    }
    for name in ("gemini", "openai", "anthropic", "openrouter", "vertex"):
        out.append({"name": name, "model": resolve_model(name) or "(default)",
                    "supports_vision": name in ("gemini", "vertex", "anthropic"),
                    "configured": cloud_keyed[name]})
    for name in ("ollama", "lmstudio"):  # local servers: always selectable
        out.append({"name": name, "model": resolve_model(name) or "(auto)",
                    "supports_vision": False, "configured": True})
    return out
