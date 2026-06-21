from __future__ import annotations

from app.services.ai.llm import registry
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.config import LLMConfig, ProviderCreds


def _cfg(default, providers):
    routing = {"default": default}
    for op in ("summary", "section", "dedup", "extraction", "vision"):
        routing[op] = default
    return LLMConfig(routing=routing, providers=providers)


def test_get_provider_uses_config_routing_and_creds():
    cfg = _cfg("anthropic", {"anthropic": ProviderCreds(api_key="k", model="claude-x")})
    prov = registry.get_provider("summary", cfg)
    assert isinstance(prov, AnthropicProvider)


def test_cache_distinguishes_keys():
    registry.reset_cache()
    a = registry.get_provider("summary", _cfg("openai",
        {"openai": ProviderCreds(api_key="k1", base_url="http://x/v1", model="m")}))
    b = registry.get_provider("summary", _cfg("openai",
        {"openai": ProviderCreds(api_key="k2", base_url="http://x/v1", model="m")}))
    assert a is not b  # different key => different cached client (no cross-user leak)
    c = registry.get_provider("summary", _cfg("openai",
        {"openai": ProviderCreds(api_key="k1", base_url="http://x/v1", model="m")}))
    assert a is c  # identical creds reuse


def test_back_compat_no_config_uses_settings(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "gemini")
    registry.reset_cache()
    prov = registry.get_provider("summary")  # no config
    assert prov.name == "gemini"
