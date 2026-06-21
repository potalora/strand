from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.types import LLMMessage, LLMRequest, ReasoningConfig


def _fake_response(text="hello", finish="STOP"):
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
    cand = SimpleNamespace(finish_reason=SimpleNamespace(name=finish))
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[cand])


@pytest.mark.asyncio
async def test_gemini_complete_maps_response():
    prov = GeminiProvider(api_key="k", model_default="gemini-3.5-flash")
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _async_return(_fake_response())
    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake_client):
        resp = await prov.complete(LLMRequest(
            messages=[LLMMessage("user", "hi")], model="gemini-3.5-flash",
            system="be brief", json_mode=True, reasoning=ReasoningConfig("low"),
        ))
    assert resp.text == "hello"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 15
    # system hoisted to system_instruction; json + thinking config wired
    cfg = fake_client.aio.models.generate_content.call_args.kwargs["config"]
    assert cfg.system_instruction == "be brief"
    assert cfg.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_gemini_maxtokens_finish_normalized():
    prov = GeminiProvider(api_key="k", model_default="m")
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _async_return(_fake_response(finish="MAX_TOKENS"))
    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake_client):
        resp = await prov.complete(LLMRequest(messages=[LLMMessage("user","x")], model="m"))
    assert resp.finish_reason == "length"


def _async_return(value):
    # AsyncMock is awaitable AND records call_args for the introspection asserts.
    return AsyncMock(return_value=value)
