from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import LLMMessage, LLMRequest


def _fake_completion(content="hi", finish="stop"):
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=4, completion_tokens=6, total_tokens=10)
    return SimpleNamespace(choices=[choice], usage=usage, model="m")


@pytest.mark.asyncio
async def test_openai_compat_maps_response_and_hoists_system():
    fake = MagicMock()
    # MagicMock(side_effect=...) records call_args while returning the coroutine.
    fake.chat.completions.create = MagicMock(side_effect=_areturn(_fake_completion()))
    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        # Construct under the patch so the client is the mock, not a real AsyncOpenAI.
        prov = OpenAICompatProvider(name="openai", api_key="k",
                                    base_url="https://api.openai.com/v1", model_default="m")
        resp = await prov.complete(LLMRequest(
            messages=[LLMMessage("user", "hi")], model="m",
            system="sys", json_mode=True))
    assert resp.text == "hi"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 10
    sent = fake.chat.completions.create.call_args.kwargs
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_compat_finish_length_normalized():
    fake = MagicMock()
    fake.chat.completions.create = MagicMock(side_effect=_areturn(_fake_completion(finish="length")))
    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        prov = OpenAICompatProvider(name="ollama", api_key="x",
                                    base_url="http://localhost:11434/v1", model_default="m")
        resp = await prov.complete(LLMRequest(messages=[LLMMessage("user", "x")], model="m"))
    assert resp.finish_reason == "length"


def _areturn(value):
    async def _c(*a, **k):
        return value
    return _c
