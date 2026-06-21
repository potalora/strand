# backend/app/services/ai/llm/anthropic.py
from __future__ import annotations
import logging
from anthropic import (
    APIConnectionError, APITimeoutError, AsyncAnthropic, AuthenticationError,
    BadRequestError, RateLimitError,
)
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities, LLMAuthError, LLMBadRequestError, LLMError, LLMRateLimitError,
    LLMRequest, LLMResponse, LLMResponseError, LLMTimeoutError, LLMUsage,
)

logger = logging.getLogger(__name__)
_STOP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}
_JSON_NUDGE = ("\n\nReturn ONLY a single valid JSON value. No prose, no markdown "
               "fences. Begin your reply with the opening brace.")


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    capabilities = Capabilities(supports_vision=True, supports_json_mode=False,
                                supports_reasoning=False)

    def __init__(self, *, api_key: str, model_default: str):
        if not api_key:
            self._client = None
        else:
            self._client = AsyncAnthropic(api_key=api_key)
        self._model_default = model_default

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise LLMAuthError("ANTHROPIC_API_KEY is not configured")
        system = request.system or ""
        prefilled = False
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.json_mode:
            system = (system + _JSON_NUDGE).strip()
            messages.append({"role": "assistant", "content": "{"})
            prefilled = True
        kwargs: dict = {
            "model": request.model or self._model_default,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
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
        usage = (LLMUsage(u.input_tokens, u.output_tokens, u.input_tokens + u.output_tokens)
                 if u else LLMUsage())
        return LLMResponse(text=text, finish_reason=finish,
                           model=getattr(resp, "model", kwargs["model"]),
                           usage=usage, raw=resp)
