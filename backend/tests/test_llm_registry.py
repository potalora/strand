from __future__ import annotations

import pytest

from app.services.ai.llm import registry
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import LLMError


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "gemini")
    monkeypatch.setattr(registry.settings, "llm_summary_provider", "")
    registry.reset_cache()
    assert isinstance(registry.get_provider("summary"), GeminiProvider)


def test_per_operation_override_wins(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "gemini")
    monkeypatch.setattr(registry.settings, "llm_summary_provider", "anthropic")
    monkeypatch.setattr(registry.settings, "anthropic_api_key", "k")
    registry.reset_cache()
    assert isinstance(registry.get_provider("summary"), AnthropicProvider)


def test_openai_family_share_one_class(monkeypatch):
    for name in ("openai", "openrouter", "ollama", "lmstudio"):
        monkeypatch.setattr(registry.settings, "llm_provider", name)
        registry.reset_cache()
        assert isinstance(registry.get_provider(), OpenAICompatProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "bogus")
    monkeypatch.setattr(registry.settings, "llm_dedup_provider", "")
    registry.reset_cache()
    with pytest.raises(LLMError):
        registry.get_provider("dedup")


def test_available_providers_hides_keys(monkeypatch):
    monkeypatch.setattr(registry.settings, "openai_api_key", "secret")
    out = registry.available_providers()
    assert all("secret" not in str(p.values()) for p in out)
    assert any(p["name"] == "ollama" for p in out)
