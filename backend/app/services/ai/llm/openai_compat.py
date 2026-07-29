from __future__ import annotations

import base64
import logging
import re
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities,
    DocumentPart,
    ImagePart,
    LLMAuthError,
    LLMBadRequestError,
    LLMMessage,
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
_FINISH = {"stop": "stop", "length": "length", "content_filter": "content_filter"}
_SCHEMA_NAME = re.compile(r"[^A-Za-z0-9_-]")


def _schema_dict(schema: Any) -> dict:
    """Return a JSON-schema dict from a raw schema or Pydantic model class."""
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        generated = model_json_schema()
        if isinstance(generated, dict):
            return generated
    raise TypeError("json_schema must be a JSON-schema dict or Pydantic model")


def _schema_name(schema: Any) -> str:
    """Build an OpenAI-safe structured-output schema name."""
    raw = getattr(schema, "__name__", None)
    if not raw and isinstance(schema, dict):
        raw = schema.get("title")
    cleaned = _SCHEMA_NAME.sub("_", str(raw or "response"))
    return cleaned[:64]


def _supports_native_strict_schema(model: str) -> bool:
    """Return whether the verified OpenAI model family supports strict schemas."""
    return model == "gpt-5.6" or model.startswith("gpt-5.6-")


def _build_content(message: LLMMessage) -> str | list[dict]:
    """Map a message's content to the OpenAI chat shape.

    All-text content (a plain ``str`` or a single ``TextPart``) stays a plain
    string so text-only calls are byte-identical to today's behavior. Image and
    document parts produce the multimodal array shape.

    Args:
        message: The message whose content is mapped.

    Returns:
        A plain string for all-text content, else a list of content-part dicts.
    """
    parts = as_parts(message.content)
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    content: list[dict] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            b64 = base64.standard_b64encode(part.data).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.mime};base64,{b64}"},
                }
            )
        elif isinstance(part, DocumentPart):
            b64 = base64.standard_b64encode(part.data).decode()
            content.append(
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                }
            )
    return content


class OpenAICompatProvider(LLMProvider):
    """Serves any OpenAI-Chat-Completions endpoint (OpenAI/OpenRouter/LM Studio/Ollama)."""

    capabilities = Capabilities(
        supports_vision=False, supports_json_mode=True, supports_reasoning=False
    )

    def __init__(self, *, name: str, api_key: str, base_url: str, model_default: str):
        """Build a provider against an OpenAI-compatible endpoint.

        Args:
            name: Logical provider name (openai/openrouter/lmstudio/ollama).
            api_key: API key; blank is allowed for local servers.
            base_url: OpenAI-compatible base URL ending in ``/v1``.
            model_default: Model used when a request omits an explicit model.
        """
        self.name = name
        # Local servers accept any non-empty key; never send an empty string.
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._model_default = model_default

    async def _create_adaptive(self, kwargs: dict):
        """Create a chat completion, adapting params that some models reject.

        Newer OpenAI models (gpt-5 / o-series) require ``max_completion_tokens``
        instead of ``max_tokens`` and only accept the default temperature; some
        local models reject ``response_format``. On a ``BadRequest`` that names one
        of these, surgically adjust that single param and retry (bounded), so the
        same code serves old and new models without per-model config.
        """
        for _ in range(4):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except AuthenticationError as e:
                raise LLMAuthError(str(e)) from e
            except RateLimitError as e:
                raise LLMRateLimitError(str(e)) from e
            except (APITimeoutError, APIConnectionError) as e:
                raise LLMTimeoutError(str(e)) from e
            except BadRequestError as e:
                msg = str(e).lower()
                if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                response_format = kwargs.get("response_format")
                strict_schema = (
                    isinstance(response_format, dict)
                    and response_format.get("type") == "json_schema"
                )
                if (
                    "response_format" in msg
                    and "response_format" in kwargs
                    and not strict_schema
                ):
                    kwargs.pop("response_format")
                    continue
                raise LLMBadRequestError(str(e)) from e
            except Exception as e:  # noqa: BLE001 - normalized below
                raise LLMResponseError(str(e)) from e
        raise LLMBadRequestError("exhausted OpenAI parameter fallbacks")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a unary chat completion and return a normalized response.

        Args:
            request: Normalized LLM request.

        Returns:
            The normalized response.

        Raises:
            LLMError: A normalized subclass on any provider failure.
        """
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": m.role, "content": _build_content(m)} for m in request.messages
        )
        model = request.model or self._model_default
        is_gpt_5_6 = self.name == "openai" and model.startswith("gpt-5.6")
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if is_gpt_5_6:
            kwargs["max_completion_tokens"] = request.max_output_tokens
        else:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.temperature is not None and not is_gpt_5_6:
            kwargs["temperature"] = request.temperature
        if request.reasoning is not None and is_gpt_5_6:
            kwargs["reasoning_effort"] = request.reasoning.level
        if request.json_mode:
            if (
                request.json_schema is not None
                and self.name == "openai"
                and _supports_native_strict_schema(model)
            ):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _schema_name(request.json_schema),
                        "schema": _schema_dict(request.json_schema),
                        "strict": True,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        resp = await self._create_adaptive(kwargs)
        choice = resp.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        text = "" if refusal else choice.message.content or ""
        finish = (
            "content_filter"
            if refusal
            else _FINISH.get(choice.finish_reason or "", "other")
        )
        u = resp.usage
        usage = (
            LLMUsage(u.prompt_tokens, u.completion_tokens, u.total_tokens)
            if u
            else LLMUsage()
        )
        return LLMResponse(
            text=text,
            finish_reason=finish,
            model=getattr(resp, "model", kwargs["model"]),
            usage=usage,
            raw=resp,
        )
