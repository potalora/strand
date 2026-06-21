from __future__ import annotations

import logging

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
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUsage,
)

logger = logging.getLogger(__name__)
_FINISH = {"stop": "stop", "length": "length", "content_filter": "content_filter"}


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
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        kwargs: dict = {
            "model": request.model or self._model_default,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except (APITimeoutError, APIConnectionError) as e:
            raise LLMTimeoutError(str(e)) from e
        except BadRequestError as e:
            # Some local models reject json_mode or temperature; retry once without them.
            if request.json_mode or request.temperature is not None:
                kwargs.pop("response_format", None)
                kwargs.pop("temperature", None)
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMBadRequestError(str(e2)) from e2
            else:
                raise LLMBadRequestError(str(e)) from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMResponseError(str(e)) from e
        choice = resp.choices[0]
        text = choice.message.content or ""
        finish = _FINISH.get(choice.finish_reason or "", "other")
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
