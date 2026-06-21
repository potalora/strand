# backend/tests/test_llm_anthropic.py
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.types import LLMMessage, LLMRequest


def _fake_msg(text="hello", stop="end_turn"):
    block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=7, output_tokens=3)
    return SimpleNamespace(content=[block], stop_reason=stop, usage=usage, model="claude")


@pytest.mark.asyncio
async def test_anthropic_maps_response_and_system():
    fake = MagicMock()
    # MagicMock(side_effect=...) records call_args while returning the coroutine.
    fake.messages.create = MagicMock(side_effect=_areturn(_fake_msg()))
    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        prov = AnthropicProvider(api_key="k", model_default="claude-haiku-4-5-20251001")
        resp = await prov.complete(LLMRequest(
            messages=[LLMMessage("user", "hi")], model="claude-haiku-4-5-20251001",
            system="be brief"))
    assert resp.text == "hello"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 10
    sent = fake.messages.create.call_args.kwargs
    assert sent["system"] == "be brief"


@pytest.mark.asyncio
async def test_anthropic_json_mode_prefills_brace():
    fake = MagicMock()
    fake.messages.create = MagicMock(side_effect=_areturn(_fake_msg(text='"a": 1}')))
    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        prov = AnthropicProvider(api_key="k", model_default="m")
        resp = await prov.complete(LLMRequest(
            messages=[LLMMessage("user", "give json")], model="m", json_mode=True))
    # Prefill '{' is prepended so callers get valid JSON text.
    assert resp.text.startswith("{")
    sent = fake.messages.create.call_args.kwargs
    assert sent["messages"][-1] == {"role": "assistant", "content": "{"}


def _areturn(value):
    async def _c(*a, **k):
        return value
    return _c
