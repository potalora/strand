# backend/app/services/ai/llm/anthropic.py
from __future__ import annotations
import base64
import logging
from typing import Any
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    transform_schema,
)
from pydantic import BaseModel
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities,
    DocumentPart,
    ImagePart,
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
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
_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}
_JSON_NUDGE = (
    "\n\nReturn ONLY a single valid JSON value. No prose, no markdown "
    "fences. Begin your reply with the opening brace."
)


def _is_pydantic_model(schema: Any) -> bool:
    """Return whether ``schema`` is a Pydantic model class."""
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def _transformed_raw_schema(schema: Any) -> dict:
    """Transform a raw schema with Anthropic's pinned SDK or fail closed."""
    if not isinstance(schema, dict):
        raise LLMBadRequestError(
            "Invalid Anthropic JSON schema: expected a Pydantic model or dict"
        )
    try:
        return transform_schema(schema)
    except (TypeError, ValueError) as exc:
        raise LLMBadRequestError("Invalid Anthropic JSON schema") from exc


def _build_content(content) -> str | list[dict]:
    """Map message content to Anthropic blocks; keep all-text content as a plain string."""
    parts = as_parts(content)
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    blocks: list[dict] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            b64 = base64.standard_b64encode(part.data).decode()
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": part.mime, "data": b64},
                }
            )
        elif isinstance(part, DocumentPart):
            b64 = base64.standard_b64encode(part.data).decode()
            blocks.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64,
                    },
                }
            )
    return blocks


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    capabilities = Capabilities(
        supports_vision=True, supports_json_mode=False, supports_reasoning=False
    )

    def __init__(self, *, api_key: str, model_default: str):
        native_sonnet_5 = model_default == "claude-sonnet-5"
        self.capabilities = Capabilities(
            supports_vision=True,
            supports_json_mode=native_sonnet_5,
            supports_reasoning=native_sonnet_5,
        )
        if not api_key:
            self._client = None
        else:
            self._client = AsyncAnthropic(api_key=api_key)
        self._model_default = model_default

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise LLMAuthError("ANTHROPIC_API_KEY is not configured")
        model = request.model or self._model_default
        is_sonnet_5 = model == "claude-sonnet-5"
        system = request.system or ""
        prefilled = False
        messages = [
            {"role": m.role, "content": _build_content(m.content)}
            for m in request.messages
        ]
        output_config: dict = {}
        output_format: type[BaseModel] | None = None
        if request.json_mode:
            if request.json_schema is not None and is_sonnet_5:
                if _is_pydantic_model(request.json_schema):
                    output_format = request.json_schema
                else:
                    output_config["format"] = {
                        "type": "json_schema",
                        "schema": _transformed_raw_schema(request.json_schema),
                    }
            else:
                system = (system + _JSON_NUDGE).strip()
                if not is_sonnet_5:
                    messages.append({"role": "assistant", "content": "{"})
                    prefilled = True
        if request.reasoning is not None and is_sonnet_5:
            kwargs_thinking = {"type": "adaptive"}
            output_config["effort"] = request.reasoning.level
        else:
            kwargs_thinking = None
        kwargs: dict = {
            "model": model,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None and not is_sonnet_5:
            kwargs["temperature"] = request.temperature
        if kwargs_thinking is not None:
            kwargs["thinking"] = kwargs_thinking
        if output_config:
            kwargs["output_config"] = output_config
        if output_format is not None:
            kwargs["output_format"] = output_format
        try:
            if output_format is not None:
                resp = await self._client.messages.parse(**kwargs)
            else:
                resp = await self._client.messages.create(**kwargs)
        except AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except (APITimeoutError, APIConnectionError) as e:
            raise LLMTimeoutError(str(e)) from e
        except BadRequestError as e:
            raise LLMBadRequestError(str(e)) from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMResponseError(str(e)) from e
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if prefilled:
            text = "{" + text
        finish = _STOP.get(resp.stop_reason or "", "other")
        u = resp.usage
        usage = (
            LLMUsage(u.input_tokens, u.output_tokens, u.input_tokens + u.output_tokens)
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
