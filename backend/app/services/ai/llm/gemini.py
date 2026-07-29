from __future__ import annotations
import logging
from google import genai
from google.genai import types as gtypes
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities,
    LLMAuthError,
    LLMError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUsage,
    TextPart,
    as_parts,
)

logger = logging.getLogger(__name__)

_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    # Gemini blocks output for these reasons; normalize them all to
    # ``content_filter`` so callers (e.g. OCR fallback) can detect a block
    # instead of seeing an opaque "other".
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
}


class GeminiProvider(LLMProvider):
    name = "gemini"
    capabilities = Capabilities(
        supports_vision=True, supports_json_mode=True, supports_reasoning=True
    )

    def __init__(
        self,
        *,
        api_key: str = "",
        model_default: str = "",
        vertexai: bool = False,
        project: str = "",
        location: str = "",
    ):
        self._api_key = api_key
        self._model_default = model_default
        self._vertexai = vertexai
        self._project = project
        self._location = location

    def _client(self) -> genai.Client:
        if self._vertexai:
            if not self._project:
                raise LLMProviderUnavailableError("Vertex requires vertex_project")
            return genai.Client(
                vertexai=True, project=self._project, location=self._location
            )
        if not self._api_key:
            raise LLMAuthError("GEMINI_API_KEY is not configured")
        return genai.Client(api_key=self._api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self._model_default
        cfg_kwargs: dict = {}
        if request.system:
            cfg_kwargs["system_instruction"] = request.system
        if request.temperature is not None and not model.startswith("gemini-3.6-"):
            cfg_kwargs["temperature"] = request.temperature
        cfg_kwargs["max_output_tokens"] = request.max_output_tokens
        if request.reasoning is not None:
            cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                thinking_level=request.reasoning.level
            )
        if request.json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
            if request.json_schema is not None:
                cfg_kwargs["response_schema"] = request.json_schema
        # Build contents from message parts: TextPart -> str, Image/DocumentPart ->
        # a Gemini Part built from raw bytes. (System instruction is hoisted out.)
        contents: list = []
        for m in request.messages:
            if m.role == "system":
                continue
            for part in as_parts(m.content):
                if isinstance(part, TextPart):
                    contents.append(part.text)
                else:  # ImagePart | DocumentPart
                    contents.append(
                        gtypes.Part.from_bytes(data=part.data, mime_type=part.mime)
                    )
        # Gemini accepts a single string or a list; keep a string when it's all text
        # so existing text-only behavior is byte-identical.
        if len(contents) == 1 and isinstance(contents[0], str):
            contents = contents[0]
        try:
            client = self._client()
            resp = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**cfg_kwargs),
            )
        except LLMError:
            raise
        except Exception as e:  # normalize SDK errors
            raise _map_error(e) from e
        text = resp.text or ""
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            raw_finish = getattr(candidates[0], "finish_reason", None)
            raw_finish_name = getattr(raw_finish, "name", str(raw_finish or ""))
            finish = _FINISH.get(raw_finish_name, "other")
        else:
            prompt_feedback = getattr(resp, "prompt_feedback", None)
            block_reason = getattr(prompt_feedback, "block_reason", None)
            block_reason_name = getattr(block_reason, "name", str(block_reason or ""))
            if block_reason_name and block_reason_name != "BLOCK_REASON_UNSPECIFIED":
                finish = "content_filter"
            else:
                raise LLMResponseError("Gemini returned no response candidates")
        usage = LLMUsage()
        if getattr(resp, "usage_metadata", None):
            p = resp.usage_metadata.prompt_token_count or 0
            c = resp.usage_metadata.candidates_token_count or 0
            provider_total = getattr(resp.usage_metadata, "total_token_count", None)
            usage = LLMUsage(
                p, c, provider_total if provider_total is not None else p + c
            )
        return LLMResponse(
            text=text, finish_reason=finish, model=model, usage=usage, raw=resp
        )


def _map_error(e: Exception) -> LLMError:
    s = str(e).lower()
    if any(k in s for k in ("permission", "api key", "unauthenticated", "401", "403")):
        return LLMAuthError(str(e))
    if any(k in s for k in ("429", "quota", "rate", "resource_exhausted")):
        return LLMRateLimitError(str(e))
    if any(k in s for k in ("timeout", "deadline", "connection")):
        return LLMTimeoutError(str(e))
    return LLMResponseError(str(e))
