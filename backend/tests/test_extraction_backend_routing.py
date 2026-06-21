from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ai.llm.config import LLMConfig, ProviderCreds
from app.services.extraction import entity_extractor
from app.services.extraction.entity_extractor import ExtractionResult


def _cfg(provider: str, providers: dict) -> LLMConfig:
    return LLMConfig(routing={"default": provider, "extraction": provider}, providers=providers)


def test_langextract_params_openai_ollama_anthropic():
    cfg = LLMConfig(
        routing={},
        providers={
            "openai": ProviderCreds(api_key="o", model="gpt-5.4-mini"),
            "ollama": ProviderCreds(
                api_key="", base_url="http://localhost:11434/v1", model="llama3.2:1b"),
        },
    )
    assert entity_extractor._langextract_params("openai", cfg, "")[:2] == ("gpt-5.4-mini", "o")
    model_id, key, url = entity_extractor._langextract_params("ollama", cfg, "")
    assert model_id == "llama3.2:1b" and key == "" and url == "http://localhost:11434"
    # No native LangExtract backend -> generic facade path.
    assert entity_extractor._langextract_params("anthropic", cfg, "") is None


@pytest.mark.asyncio
async def test_openai_extraction_routes_to_langextract_native():
    cfg = _cfg("openai", {"openai": ProviderCreds(api_key="k", model="gpt-5.4-mini")})
    ee = MagicMock(return_value=ExtractionResult("f", "t"))

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch.object(entity_extractor, "extract_entities", ee), \
         patch.object(entity_extractor.asyncio, "to_thread", new=fake_to_thread):
        await entity_extractor.extract_entities_async("text", "f", "gemkey", config=cfg)
    assert ee.call_args.kwargs["model_id"] == "gpt-5.4-mini"
    assert ee.call_args.kwargs["model_url"] is None


@pytest.mark.asyncio
async def test_ollama_extraction_routes_with_model_url():
    cfg = _cfg("ollama", {
        "ollama": ProviderCreds(api_key="", base_url="http://localhost:11434/v1", model="llama3.2:1b")})
    ee = MagicMock(return_value=ExtractionResult("f", "t"))

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch.object(entity_extractor, "extract_entities", ee), \
         patch.object(entity_extractor.asyncio, "to_thread", new=fake_to_thread):
        await entity_extractor.extract_entities_async("text", "f", "gemkey", config=cfg)
    assert ee.call_args.kwargs["model_id"] == "llama3.2:1b"
    assert ee.call_args.kwargs["model_url"] == "http://localhost:11434"


@pytest.mark.asyncio
async def test_anthropic_extraction_uses_generic_path():
    cfg = _cfg("anthropic", {"anthropic": ProviderCreds(api_key="k", model="claude-x")})
    generic = AsyncMock(return_value=ExtractionResult("f", "t"))
    with patch(
        "app.services.extraction.generic_entity_extractor.generic_extract_entities_async",
        generic,
    ):
        await entity_extractor.extract_entities_async("text", "f", "key", config=cfg)
    generic.assert_awaited_once()
