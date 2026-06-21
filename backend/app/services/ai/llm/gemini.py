from __future__ import annotations
import logging
from google import genai
from google.genai import types as gtypes
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities, LLMAuthError, LLMError, LLMProviderUnavailableError,
    LLMRateLimitError, LLMRequest, LLMResponse, LLMResponseError, LLMTimeoutError,
    LLMUsage,
)

logger = logging.getLogger(__name__)

_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}


class GeminiProvider(LLMProvider):
    name = "gemini"
    capabilities = Capabilities(supports_vision=True, supports_json_mode=True,
                                supports_reasoning=True)

    def __init__(self, *, api_key: str = "", model_default: str = "",
                 vertexai: bool = False, project: str = "", location: str = ""):
        self._api_key = api_key
        self._model_default = model_default
        self._vertexai = vertexai
        self._project = project
        self._location = location

    def _client(self) -> genai.Client:
        if self._vertexai:
            if not self._project:
                raise LLMProviderUnavailableError("Vertex requires vertex_project")
            return genai.Client(vertexai=True, project=self._project,
                                location=self._location)
        if not self._api_key:
            raise LLMAuthError("GEMINI_API_KEY is not configured")
        return genai.Client(api_key=self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        cfg_kwargs: dict = {}
        if request.system:
            cfg_kwargs["system_instruction"] = request.system
        if request.temperature is not None:
            cfg_kwargs["temperature"] = request.temperature
        cfg_kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                thinking_level=request.reasoning.level)
        if request.json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
            if request.json_schema is not None:
                cfg_kwargs["response_schema"] = request.json_schema
        # Gemini takes a single contents string here (all current callers send one
        # user turn after the system instruction is hoisted out).
        contents = "\n\n".join(m.content for m in request.messages if m.role != "system")
        try:
            client = self._client()
            resp = await client.aio.models.generate_content(
                model=request.model or self._model_default,
                contents=contents,
                config=gtypes.GenerateContentConfig(**cfg_kwargs),
            )
        except LLMError:
            raise
        except Exception as e:  # normalize SDK errors
            raise _map_error(e) from e
        text = resp.text or ""
        finish = "stop"
        try:
            raw_finish = resp.candidates[0].finish_reason.name
            finish = _FINISH.get(raw_finish, "other")
        except Exception:
            pass
        usage = LLMUsage()
        if getattr(resp, "usage_metadata", None):
            p = resp.usage_metadata.prompt_token_count or 0
            c = resp.usage_metadata.candidates_token_count or 0
            usage = LLMUsage(p, c, p + c)
        return LLMResponse(text=text, finish_reason=finish,
                           model=request.model or self._model_default,
                           usage=usage, raw=resp)


def _map_error(e: Exception) -> LLMError:
    s = str(e).lower()
    if any(k in s for k in ("permission", "api key", "unauthenticated", "401", "403")):
        return LLMAuthError(str(e))
    if any(k in s for k in ("429", "quota", "rate", "resource_exhausted")):
        return LLMRateLimitError(str(e))
    if any(k in s for k in ("timeout", "deadline", "connection")):
        return LLMTimeoutError(str(e))
    return LLMResponseError(str(e))
