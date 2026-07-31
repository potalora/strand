from __future__ import annotations
import pytest
from app.services.ai.llm.types import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMError,
    LLMAuthError,
    LLMRateLimitError,
)
from app.services.ai.llm.base import LLMProvider


def test_request_construction_defaults():
    req = LLMRequest(messages=[LLMMessage("user", "hi")], model="m")
    assert req.system is None
    assert req.json_mode is False
    assert req.temperature is None
    assert req.max_output_tokens > 0


def test_response_and_usage():
    r = LLMResponse(
        text="ok", finish_reason="stop", model="m", usage=LLMUsage(1, 2, 3), raw=None
    )
    assert r.usage.total_tokens == 3


def test_error_hierarchy():
    assert issubclass(LLMAuthError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # cannot instantiate abstract
