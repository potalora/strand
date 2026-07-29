from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import transform_schema
from openai import BadRequestError

from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import (
    LLMBadRequestError,
    LLMMessage,
    LLMRequest,
    LLMResponseError,
    ReasoningConfig,
)
from app.services.ai.summarizer import _require_complete_routed_response
from app.services.local_ai.grounded_summary import GroundedSummaryDocument


def _openai_response() -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content='{"sections":[],"uncertainties":[]}'),
        finish_reason="stop",
    )
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=7,
            total_tokens=17,
        ),
        model="gpt-5.6-luna",
    )


def _openai_bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


def _anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='{"sections":[],"uncertainties":[]}',
            )
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=7),
        model="claude-sonnet-5",
    )


def _request(model: str) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage("user", "Return the grounded summary selection.")],
        model=model,
        max_output_tokens=4096,
        temperature=0,
        json_mode=True,
        json_schema=GroundedSummaryDocument,
        reasoning=ReasoningConfig(level="low"),
    )


@pytest.mark.asyncio
async def test_gpt_5_6_luna_uses_native_reasoning_and_strict_schema_first() -> None:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=_openai_response())

    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        provider = OpenAICompatProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_default="gpt-5.6-luna",
        )
        await provider.complete(_request("gpt-5.6-luna"))

    sent = fake.chat.completions.create.await_args.kwargs
    assert sent["model"] == "gpt-5.6-luna"
    assert sent["max_completion_tokens"] == 4096
    assert "max_tokens" not in sent
    assert "temperature" not in sent
    assert sent["reasoning_effort"] == "low"
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "GroundedSummaryDocument",
            "schema": GroundedSummaryDocument.model_json_schema(),
            "strict": True,
        },
    }


@pytest.mark.asyncio
async def test_gpt_5_6_luna_never_downgrades_a_rejected_strict_schema() -> None:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(
        side_effect=_openai_bad_request("Invalid response_format JSON schema")
    )

    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        provider = OpenAICompatProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_default="gpt-5.6-luna",
        )
        with pytest.raises(
            LLMBadRequestError,
            match="Invalid response_format JSON schema",
        ):
            await provider.complete(_request("gpt-5.6-luna"))

    assert fake.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_older_openai_model_keeps_json_object_compatibility() -> None:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=_openai_response())

    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        provider = OpenAICompatProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_default="gpt-3.5-turbo",
        )
        await provider.complete(_request("gpt-3.5-turbo"))

    sent = fake.chat.completions.create.await_args.kwargs
    assert sent["model"] == "gpt-3.5-turbo"
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_tokens"] == 4096
    assert sent["temperature"] == 0
    assert "max_completion_tokens" not in sent
    assert "reasoning_effort" not in sent


@pytest.mark.asyncio
async def test_older_openai_model_can_drop_rejected_json_object_mode() -> None:
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(
        side_effect=[
            _openai_bad_request("Unsupported parameter: response_format"),
            _openai_response(),
        ]
    )

    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        provider = OpenAICompatProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_default="gpt-3.5-turbo",
        )
        result = await provider.complete(_request("gpt-3.5-turbo"))

    assert result.text == '{"sections":[],"uncertainties":[]}'
    assert fake.chat.completions.create.await_count == 2
    first, second = fake.chat.completions.create.await_args_list
    assert first.kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in second.kwargs


@pytest.mark.asyncio
async def test_openai_refusal_is_not_a_completed_summary() -> None:
    refusal = _openai_response()
    refusal.choices[0].message.content = None
    refusal.choices[0].message.refusal = "Request declined."
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=refusal)

    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        provider = OpenAICompatProvider(
            name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_default="gpt-5.6-luna",
        )
        result = await provider.complete(_request("gpt-5.6-luna"))

    assert result.text == ""
    assert result.finish_reason == "content_filter"
    with pytest.raises(ValueError, match="did not complete safely"):
        _require_complete_routed_response(result)


@pytest.mark.asyncio
async def test_gemini_3_6_flash_omits_sampling_and_uses_provider_total_tokens() -> None:
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=7,
        thoughts_token_count=11,
        total_token_count=28,
    )
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))
    response = SimpleNamespace(
        text='{"sections":[],"uncertainties":[]}',
        usage_metadata=usage,
        candidates=[candidate],
    )
    fake = MagicMock()
    fake.aio.models.generate_content = AsyncMock(return_value=response)

    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake):
        provider = GeminiProvider(
            api_key="test-key",
            model_default="gemini-3.6-flash",
        )
        result = await provider.complete(_request("gemini-3.6-flash"))

    sent = fake.aio.models.generate_content.await_args.kwargs
    assert sent["model"] == "gemini-3.6-flash"
    assert sent["config"].temperature is None
    assert sent["config"].response_schema is GroundedSummaryDocument
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 28


@pytest.mark.asyncio
async def test_gemini_blocked_prompt_is_not_a_completed_summary() -> None:
    response = SimpleNamespace(
        text=None,
        usage_metadata=None,
        candidates=[],
        prompt_feedback=SimpleNamespace(
            block_reason=SimpleNamespace(name="PROHIBITED_CONTENT")
        ),
    )
    fake = MagicMock()
    fake.aio.models.generate_content = AsyncMock(return_value=response)

    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake):
        provider = GeminiProvider(
            api_key="test-key",
            model_default="gemini-3.6-flash",
        )
        result = await provider.complete(_request("gemini-3.6-flash"))

    assert result.text == ""
    assert result.finish_reason == "content_filter"
    with pytest.raises(ValueError, match="did not complete safely"):
        _require_complete_routed_response(result)


@pytest.mark.asyncio
async def test_gemini_unexplained_empty_response_fails_closed() -> None:
    response = SimpleNamespace(
        text=None,
        usage_metadata=None,
        candidates=[],
        prompt_feedback=None,
    )
    fake = MagicMock()
    fake.aio.models.generate_content = AsyncMock(return_value=response)

    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake):
        provider = GeminiProvider(
            api_key="test-key",
            model_default="gemini-3.6-flash",
        )
        with pytest.raises(LLMResponseError, match="no response candidates"):
            await provider.complete(_request("gemini-3.6-flash"))


@pytest.mark.asyncio
async def test_claude_sonnet_5_uses_parse_for_pydantic_structured_output() -> None:
    fake = MagicMock()
    fake.messages.parse = AsyncMock(return_value=_anthropic_response())
    fake.messages.create = AsyncMock(return_value=_anthropic_response())

    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        provider = AnthropicProvider(
            api_key="test-key",
            model_default="claude-sonnet-5",
        )
        assert provider.capabilities.supports_json_mode
        assert provider.capabilities.supports_reasoning
        result = await provider.complete(_request("claude-sonnet-5"))

    fake.messages.create.assert_not_awaited()
    sent = fake.messages.parse.await_args.kwargs
    assert sent["model"] == "claude-sonnet-5"
    assert sent["messages"] == [
        {
            "role": "user",
            "content": "Return the grounded summary selection.",
        }
    ]
    assert "temperature" not in sent
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["output_format"] is GroundedSummaryDocument
    assert result.text == '{"sections":[],"uncertainties":[]}'


@pytest.mark.asyncio
async def test_claude_sonnet_5_transforms_raw_schema_before_create() -> None:
    raw_schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "minLength": 2,
                "maxLength": 20,
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    request = _request("claude-sonnet-5")
    request.json_schema = raw_schema
    fake = MagicMock()
    fake.messages.parse = AsyncMock(return_value=_anthropic_response())
    fake.messages.create = AsyncMock(return_value=_anthropic_response())

    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        provider = AnthropicProvider(
            api_key="test-key",
            model_default="claude-sonnet-5",
        )
        await provider.complete(request)

    fake.messages.parse.assert_not_awaited()
    sent = fake.messages.create.await_args.kwargs
    assert sent["output_config"] == {
        "effort": "low",
        "format": {
            "type": "json_schema",
            "schema": transform_schema(raw_schema),
        },
    }
    assert (
        "minLength"
        not in sent["output_config"]["format"]["schema"]["properties"]["value"]
    )


@pytest.mark.asyncio
async def test_claude_sonnet_5_rejects_invalid_raw_schema_before_send() -> None:
    request = _request("claude-sonnet-5")
    request.json_schema = {"properties": {"value": {"type": "string"}}}
    fake = MagicMock()
    fake.messages.parse = AsyncMock(return_value=_anthropic_response())
    fake.messages.create = AsyncMock(return_value=_anthropic_response())

    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        provider = AnthropicProvider(
            api_key="test-key",
            model_default="claude-sonnet-5",
        )
        with pytest.raises(LLMBadRequestError, match="Invalid Anthropic JSON schema"):
            await provider.complete(request)

    fake.messages.parse.assert_not_awaited()
    fake.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_older_anthropic_model_keeps_legacy_json_prefill() -> None:
    response = _anthropic_response()
    response.content[0].text = '"sections":[],"uncertainties":[]}'
    response.model = "claude-haiku-4-5-20251001"
    fake = MagicMock()
    fake.messages.create = AsyncMock(return_value=response)

    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
        provider = AnthropicProvider(
            api_key="test-key",
            model_default="claude-haiku-4-5-20251001",
        )
        assert not provider.capabilities.supports_json_mode
        assert not provider.capabilities.supports_reasoning
        result = await provider.complete(_request("claude-haiku-4-5-20251001"))

    sent = fake.messages.create.await_args.kwargs
    assert sent["messages"][-1] == {"role": "assistant", "content": "{"}
    assert "output_config" not in sent
    assert "thinking" not in sent
    assert sent["temperature"] == 0
    assert result.text == '{"sections":[],"uncertainties":[]}'
