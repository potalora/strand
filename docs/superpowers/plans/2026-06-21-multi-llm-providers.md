# Multi-LLM-Provider Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-agnostic LLM layer so summarization, section parsing, dedup-judge, and entity extraction can run against Gemini, OpenAI, Anthropic, OpenRouter, LM Studio, Ollama, and Vertex — with de-identification always applied first.

**Architecture:** New `backend/app/services/ai/llm/` package: a normalized `LLMProvider` interface + `LLMRequest`/`LLMResponse` types, three provider impls (google-genai, openai SDK, anthropic SDK), and a registry that resolves provider+model from config. The OpenAI provider's `base_url` covers OpenAI/OpenRouter/LM Studio/Ollama. Existing call sites keep their de-id step and swap the raw SDK call for the facade. Default `LLM_PROVIDER=gemini` preserves current behavior byte-for-byte.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, google-genai (existing), openai SDK (new), anthropic SDK (new), pytest/pytest-asyncio, Next.js/TS frontend.

## Global Constraints

- `requires-python = ">=3.11.8,<3.12"` — do not change.
- `[tool.uv] override-dependencies = ["typing-extensions>=4.15"]` — keep; new deps must resolve under it.
- New deps must be MIT/Apache-2.0 (openai = Apache-2.0, anthropic = MIT). No GPL.
- De-identify (scrub_phi) before EVERY provider call. Never send raw PHI. (Absolute Rule #2, all providers.)
- No diagnoses/treatment/advice in any prompt or output (unchanged; prompts are provider-independent).
- API keys via `.env` only; never commit, log, or return them in any response.
- Back-compat: unset `LLM_PROVIDER` ⇒ Gemini with today's models and behavior.
- Type hints + `from __future__ import annotations` everywhere; Google-style docstrings; no bare `except`; `logging` not `print`; Ruff 100-char lines.
- Tests alongside code; API-level verification before frontend (Feature Verification Pattern).

---

### Task 1: Dependencies + config settings

**Files:**
- Modify: `backend/pyproject.toml` (dependencies list)
- Modify: `backend/app/config.py` (new settings block after the Gemini block, ~line 61)
- Modify: `backend/.env.example` (document new vars)
- Test: `backend/tests/test_config_llm.py`

**Interfaces:**
- Produces: `settings.llm_provider`, `settings.llm_summary_provider`, `settings.llm_section_provider`, `settings.llm_dedup_provider`, `settings.llm_extraction_provider`, `settings.openai_api_key`, `settings.openai_model`, `settings.openai_base_url`, `settings.openrouter_api_key`, `settings.openrouter_model`, `settings.openrouter_base_url`, `settings.ollama_base_url`, `settings.ollama_model`, `settings.lmstudio_base_url`, `settings.lmstudio_model`, `settings.anthropic_api_key`, `settings.anthropic_model`, `settings.vertex_project`, `settings.vertex_location`, `settings.vertex_model`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_config_llm.py
from __future__ import annotations
from app.config import settings


def test_llm_provider_defaults_to_gemini():
    assert settings.llm_provider == "gemini"


def test_openai_compatible_defaults_present():
    assert settings.openai_base_url.endswith("/v1")
    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.lmstudio_base_url == "http://localhost:1234/v1"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_per_operation_provider_overrides_blank_by_default():
    assert settings.llm_summary_provider == ""
    assert settings.llm_section_provider == ""
    assert settings.llm_dedup_provider == ""
    assert settings.llm_extraction_provider == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_config_llm.py -v`
Expected: FAIL (AttributeError: 'Settings' object has no attribute 'llm_provider')

- [ ] **Step 3: Add settings**

In `backend/app/config.py`, after `gemini_concurrency_limit` (~line 61) add:

```python
    # --- Multi-LLM provider routing (see docs/.../multi-llm-provider-design.md) ---
    # Global default provider. Unset/"gemini" preserves current behavior exactly.
    # One of: gemini | openai | anthropic | openrouter | lmstudio | ollama | vertex
    llm_provider: str = "gemini"
    # Optional per-operation overrides (blank => fall back to llm_provider).
    llm_summary_provider: str = ""
    llm_section_provider: str = ""
    llm_dedup_provider: str = ""
    llm_extraction_provider: str = ""

    # OpenAI-compatible family (base_url distinguishes them; all speak the
    # OpenAI Chat Completions API, so one provider class serves all four).
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = ""  # blank => resolved from /v1/models at call time

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Vertex (Gemini-on-Vertex via google-genai vertexai mode)
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-3.5-flash"
```

- [ ] **Step 4: Add deps to pyproject.toml**

In `backend/pyproject.toml` `dependencies = [...]`, after the `langextract` line add:

```toml
    "openai>=1.55.0",
    "anthropic>=0.40.0",
```

- [ ] **Step 5: Document in .env.example**

Append a `# --- Multi-LLM providers ---` block to `backend/.env.example` listing each new var with a one-line comment and the local-server URLs. Do NOT include real keys.

- [ ] **Step 6: Install + run test**

Run: `cd backend && uv sync 2>&1 | tail -5 && python -m pytest tests/test_config_llm.py -v`
Expected: deps resolve; tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/config.py backend/.env.example backend/tests/test_config_llm.py
git commit -m "feat(llm): add openai+anthropic deps and multi-provider config settings"
```

---

### Task 2: LLM types + base interface

**Files:**
- Create: `backend/app/services/ai/llm/__init__.py`
- Create: `backend/app/services/ai/llm/types.py`
- Create: `backend/app/services/ai/llm/base.py`
- Test: `backend/tests/test_llm_types.py`

**Interfaces:**
- Produces:
  - `LLMMessage(role: Literal["system","user","assistant"], content: str)`
  - `LLMUsage(prompt_tokens: int, completion_tokens: int, total_tokens: int)`
  - `ReasoningConfig(level: Literal["low","high"])`
  - `LLMRequest(messages, system, model, max_output_tokens, temperature, json_mode, json_schema, reasoning)` (dataclass)
  - `LLMResponse(text: str, finish_reason: Literal["stop","length","content_filter","other"], model: str, usage: LLMUsage, raw: Any)`
  - `Capabilities(supports_vision: bool, supports_json_mode: bool, supports_reasoning: bool)`
  - Errors: `LLMError`, `LLMAuthError`, `LLMRateLimitError`, `LLMTimeoutError`, `LLMBadRequestError`, `LLMResponseError`, `LLMProviderUnavailableError` (all subclass `LLMError`)
  - `class LLMProvider(ABC)` with `name: str`, `capabilities: Capabilities`, `async def complete(self, request: LLMRequest) -> LLMResponse`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_types.py
from __future__ import annotations
import pytest
from app.services.ai.llm.types import (
    LLMMessage, LLMRequest, LLMResponse, LLMUsage, Capabilities,
    ReasoningConfig, LLMError, LLMAuthError, LLMRateLimitError,
)
from app.services.ai.llm.base import LLMProvider


def test_request_construction_defaults():
    req = LLMRequest(messages=[LLMMessage("user", "hi")], model="m")
    assert req.system is None
    assert req.json_mode is False
    assert req.temperature is None
    assert req.max_output_tokens > 0


def test_response_and_usage():
    r = LLMResponse(text="ok", finish_reason="stop", model="m",
                    usage=LLMUsage(1, 2, 3), raw=None)
    assert r.usage.total_tokens == 3


def test_error_hierarchy():
    assert issubclass(LLMAuthError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)


def test_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # cannot instantiate abstract
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && python -m pytest tests/test_llm_types.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement types.py**

```python
# backend/app/services/ai/llm/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter", "other"]


@dataclass
class LLMMessage:
    role: Role
    content: str


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ReasoningConfig:
    level: Literal["low", "high"] = "low"


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str
    system: str | None = None
    max_output_tokens: int = 4096
    temperature: float | None = None
    json_mode: bool = False
    json_schema: Any | None = None  # pydantic model or JSON-schema dict
    reasoning: ReasoningConfig | None = None


@dataclass
class LLMResponse:
    text: str
    finish_reason: FinishReason
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Any = None


@dataclass
class Capabilities:
    supports_vision: bool = False
    supports_json_mode: bool = True
    supports_reasoning: bool = False


class LLMError(Exception):
    """Base for all normalized provider errors."""


class LLMAuthError(LLMError):
    """Auth/permission failure (bad/missing key)."""


class LLMRateLimitError(LLMError):
    """Rate limited / quota exhausted (retryable)."""


class LLMTimeoutError(LLMError):
    """Request timed out / connection error (retryable)."""


class LLMBadRequestError(LLMError):
    """Malformed request (model/param rejected)."""


class LLMResponseError(LLMError):
    """Response present but unparseable / empty when content required."""


class LLMProviderUnavailableError(LLMError):
    """Provider/SDK not installed or local server unreachable."""
```

- [ ] **Step 4: Implement base.py**

```python
# backend/app/services/ai/llm/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from app.services.ai.llm.types import Capabilities, LLMRequest, LLMResponse


class LLMProvider(ABC):
    """Provider-agnostic LLM interface. Implementations normalize one SDK."""

    name: str = "base"
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a unary completion and return a normalized response.

        Raises a subclass of LLMError on failure.
        """
        raise NotImplementedError
```

- [ ] **Step 5: Implement __init__.py (public surface; providers added in later tasks)**

```python
# backend/app/services/ai/llm/__init__.py
from __future__ import annotations
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities, FinishReason, LLMAuthError, LLMBadRequestError, LLMError,
    LLMMessage, LLMProviderUnavailableError, LLMRateLimitError, LLMRequest,
    LLMResponse, LLMResponseError, LLMTimeoutError, LLMUsage, ReasoningConfig,
)

__all__ = [
    "LLMProvider", "LLMRequest", "LLMResponse", "LLMMessage", "LLMUsage",
    "ReasoningConfig", "Capabilities", "FinishReason", "LLMError", "LLMAuthError",
    "LLMRateLimitError", "LLMTimeoutError", "LLMBadRequestError",
    "LLMResponseError", "LLMProviderUnavailableError",
]
```

- [ ] **Step 6: Run tests + commit**

Run: `cd backend && python -m pytest tests/test_llm_types.py -v` → PASS

```bash
git add backend/app/services/ai/llm/ backend/tests/test_llm_types.py
git commit -m "feat(llm): provider-agnostic types and base interface"
```

---

### Task 3: Gemini provider

**Files:**
- Create: `backend/app/services/ai/llm/gemini.py`
- Test: `backend/tests/test_llm_gemini.py`

**Interfaces:**
- Consumes: `LLMProvider`, `LLMRequest`, `LLMResponse`, error types (Task 2); `settings` (Task 1).
- Produces: `class GeminiProvider(LLMProvider)` with `__init__(self, *, api_key: str = "", model_default: str = "", vertexai: bool = False, project: str = "", location: str = "")`. `capabilities = Capabilities(supports_vision=True, supports_json_mode=True, supports_reasoning=True)`.

- [ ] **Step 1: Write the failing test** (mock `genai.Client`)

```python
# backend/tests/test_llm_gemini.py
from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.types import LLMMessage, LLMRequest, ReasoningConfig


def _fake_response(text="hello", finish="STOP"):
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
    cand = SimpleNamespace(finish_reason=SimpleNamespace(name=finish))
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[cand])


@pytest.mark.asyncio
async def test_gemini_complete_maps_response():
    prov = GeminiProvider(api_key="k", model_default="gemini-3.5-flash")
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _async_return(_fake_response())
    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake_client):
        resp = await prov.complete(LLMRequest(
            messages=[LLMMessage("user", "hi")], model="gemini-3.5-flash",
            system="be brief", json_mode=True, reasoning=ReasoningConfig("low"),
        ))
    assert resp.text == "hello"
    assert resp.finish_reason == "stop"
    assert resp.usage.total_tokens == 15
    # system hoisted to system_instruction; json + thinking config wired
    cfg = fake_client.aio.models.generate_content.call_args.kwargs["config"]
    assert cfg.system_instruction == "be brief"
    assert cfg.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_gemini_maxtokens_finish_normalized():
    prov = GeminiProvider(api_key="k", model_default="m")
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = _async_return(_fake_response(finish="MAX_TOKENS"))
    with patch("app.services.ai.llm.gemini.genai.Client", return_value=fake_client):
        resp = await prov.complete(LLMRequest(messages=[LLMMessage("user","x")], model="m"))
    assert resp.finish_reason == "length"


def _async_return(value):
    async def _coro(*a, **k):
        return value
    return _coro
```

- [ ] **Step 2: Run to verify it fails** → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement gemini.py**

```python
# backend/app/services/ai/llm/gemini.py
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
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add backend/app/services/ai/llm/gemini.py backend/tests/test_llm_gemini.py
git commit -m "feat(llm): Gemini provider (api-key + vertex modes)"
```

---

### Task 4: OpenAI-compatible provider (OpenAI / OpenRouter / LM Studio / Ollama)

**Files:**
- Create: `backend/app/services/ai/llm/openai_compat.py`
- Test: `backend/tests/test_llm_openai_compat.py`

**Interfaces:**
- Produces: `class OpenAICompatProvider(LLMProvider)` with `__init__(self, *, name: str, api_key: str, base_url: str, model_default: str)`. `capabilities = Capabilities(supports_vision=False, supports_json_mode=True, supports_reasoning=False)`. Uses `openai.AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_openai_compat.py
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
    prov = OpenAICompatProvider(name="openai", api_key="k",
                                base_url="https://api.openai.com/v1", model_default="m")
    fake = MagicMock()
    fake.chat.completions.create = _areturn(_fake_completion())
    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
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
    prov = OpenAICompatProvider(name="ollama", api_key="x",
                                base_url="http://localhost:11434/v1", model_default="m")
    fake = MagicMock()
    fake.chat.completions.create = _areturn(_fake_completion(finish="length"))
    with patch("app.services.ai.llm.openai_compat.AsyncOpenAI", return_value=fake):
        resp = await prov.complete(LLMRequest(messages=[LLMMessage("user","x")], model="m"))
    assert resp.finish_reason == "length"


def _areturn(value):
    async def _c(*a, **k):
        return value
    return _c
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement openai_compat.py**

```python
# backend/app/services/ai/llm/openai_compat.py
from __future__ import annotations
import logging
from openai import (
    APIConnectionError, APITimeoutError, AsyncOpenAI, AuthenticationError,
    BadRequestError, RateLimitError,
)
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.types import (
    Capabilities, LLMAuthError, LLMBadRequestError, LLMError, LLMRateLimitError,
    LLMRequest, LLMResponse, LLMResponseError, LLMTimeoutError, LLMUsage,
)

logger = logging.getLogger(__name__)
_FINISH = {"stop": "stop", "length": "length", "content_filter": "content_filter"}


class OpenAICompatProvider(LLMProvider):
    """Serves any OpenAI-Chat-Completions endpoint (OpenAI/OpenRouter/LM Studio/Ollama)."""

    capabilities = Capabilities(supports_vision=False, supports_json_mode=True,
                                supports_reasoning=False)

    def __init__(self, *, name: str, api_key: str, base_url: str, model_default: str):
        self.name = name
        # Local servers accept any non-empty key; never send an empty string.
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._model_default = model_default

    async def complete(self, request: LLMRequest) -> LLMResponse:
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
        usage = (LLMUsage(u.prompt_tokens, u.completion_tokens, u.total_tokens)
                 if u else LLMUsage())
        return LLMResponse(text=text, finish_reason=finish,
                           model=getattr(resp, "model", kwargs["model"]),
                           usage=usage, raw=resp)
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add backend/app/services/ai/llm/openai_compat.py backend/tests/test_llm_openai_compat.py
git commit -m "feat(llm): OpenAI-compatible provider (openai/openrouter/lmstudio/ollama)"
```

---

### Task 5: Anthropic provider

**Files:**
- Create: `backend/app/services/ai/llm/anthropic.py`
- Test: `backend/tests/test_llm_anthropic.py`

**Interfaces:**
- Produces: `class AnthropicProvider(LLMProvider)` with `__init__(self, *, api_key: str, model_default: str)`. `capabilities = Capabilities(supports_vision=True, supports_json_mode=False, supports_reasoning=False)`. JSON requested via system-prompt instruction + assistant `{` prefill; response text is `"{" + content` then parsed by callers.

- [ ] **Step 1: Write the failing test**

```python
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
    prov = AnthropicProvider(api_key="k", model_default="claude-haiku-4-5-20251001")
    fake = MagicMock()
    fake.messages.create = _areturn(_fake_msg())
    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
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
    prov = AnthropicProvider(api_key="k", model_default="m")
    fake = MagicMock()
    fake.messages.create = _areturn(_fake_msg(text='"a": 1}'))
    with patch("app.services.ai.llm.anthropic.AsyncAnthropic", return_value=fake):
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
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement anthropic.py**

```python
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
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add backend/app/services/ai/llm/anthropic.py backend/tests/test_llm_anthropic.py
git commit -m "feat(llm): Anthropic provider with JSON-prefill mode"
```

---

### Task 6: Registry + facade

**Files:**
- Create: `backend/app/services/ai/llm/registry.py`
- Modify: `backend/app/services/ai/llm/__init__.py` (export `get_provider`, `resolve_model`, `available_providers`)
- Test: `backend/tests/test_llm_registry.py`

**Interfaces:**
- Produces:
  - `get_provider(operation: str | None = None) -> LLMProvider`
  - `resolve_model(provider_name: str) -> str`
  - `provider_name_for(operation: str | None) -> str`
  - `available_providers() -> list[dict]` — `[{"name","model","supports_vision","configured"}]` (no keys). "configured" = has key (cloud) or is a local server (ollama/lmstudio always listed).
  - `KNOWN_PROVIDERS: set[str]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_registry.py
from __future__ import annotations
import pytest
from app.services.ai.llm import registry
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.types import LLMError


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "gemini")
    monkeypatch.setattr(registry.settings, "llm_summary_provider", "")
    registry.reset_cache()
    assert isinstance(registry.get_provider("summary"), GeminiProvider)


def test_per_operation_override_wins(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "gemini")
    monkeypatch.setattr(registry.settings, "llm_summary_provider", "anthropic")
    monkeypatch.setattr(registry.settings, "anthropic_api_key", "k")
    registry.reset_cache()
    assert isinstance(registry.get_provider("summary"), AnthropicProvider)


def test_openai_family_share_one_class(monkeypatch):
    for name in ("openai", "openrouter", "ollama", "lmstudio"):
        monkeypatch.setattr(registry.settings, "llm_provider", name)
        registry.reset_cache()
        assert isinstance(registry.get_provider(), OpenAICompatProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "bogus")
    monkeypatch.setattr(registry.settings, "llm_dedup_provider", "")
    registry.reset_cache()
    with pytest.raises(LLMError):
        registry.get_provider("dedup")


def test_available_providers_hides_keys(monkeypatch):
    monkeypatch.setattr(registry.settings, "openai_api_key", "secret")
    out = registry.available_providers()
    assert all("secret" not in str(p.values()) for p in out)
    assert any(p["name"] == "ollama" for p in out)
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement registry.py**

```python
# backend/app/services/ai/llm/registry.py
from __future__ import annotations
from app.config import settings
from app.services.ai.llm.anthropic import AnthropicProvider
from app.services.ai.llm.base import LLMProvider
from app.services.ai.llm.gemini import GeminiProvider
from app.services.ai.llm.openai_compat import OpenAICompatProvider
from app.services.ai.llm.types import LLMBadRequestError

KNOWN_PROVIDERS = {"gemini", "openai", "anthropic", "openrouter",
                   "lmstudio", "ollama", "vertex"}
_OPENAI_FAMILY = {"openai", "openrouter", "lmstudio", "ollama"}
_OP_SETTING = {
    "summary": "llm_summary_provider",
    "section": "llm_section_provider",
    "dedup": "llm_dedup_provider",
    "extraction": "llm_extraction_provider",
}
_cache: dict[str, LLMProvider] = {}


def reset_cache() -> None:
    _cache.clear()


def provider_name_for(operation: str | None) -> str:
    if operation:
        override = getattr(settings, _OP_SETTING.get(operation, ""), "")
        if override:
            return override
    return settings.llm_provider or "gemini"


def resolve_model(provider_name: str) -> str:
    return {
        "gemini": settings.gemini_model,
        "vertex": settings.vertex_model,
        "openai": settings.openai_model,
        "openrouter": settings.openrouter_model,
        "ollama": settings.ollama_model,
        "lmstudio": settings.lmstudio_model,
        "anthropic": settings.anthropic_model,
    }.get(provider_name, "")


def _build(provider_name: str) -> LLMProvider:
    if provider_name == "gemini":
        return GeminiProvider(api_key=settings.gemini_api_key,
                              model_default=settings.gemini_model)
    if provider_name == "vertex":
        return GeminiProvider(vertexai=True, project=settings.vertex_project,
                              location=settings.vertex_location,
                              model_default=settings.vertex_model)
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=settings.anthropic_api_key,
                                 model_default=settings.anthropic_model)
    if provider_name in _OPENAI_FAMILY:
        key = {"openai": settings.openai_api_key,
               "openrouter": settings.openrouter_api_key}.get(provider_name, "local")
        base_url = {"openai": settings.openai_base_url,
                    "openrouter": settings.openrouter_base_url,
                    "ollama": settings.ollama_base_url,
                    "lmstudio": settings.lmstudio_base_url}[provider_name]
        return OpenAICompatProvider(name=provider_name, api_key=key,
                                    base_url=base_url,
                                    model_default=resolve_model(provider_name))
    raise LLMBadRequestError(f"Unknown LLM provider: {provider_name!r}")


def get_provider(operation: str | None = None) -> LLMProvider:
    name = provider_name_for(operation)
    if name not in KNOWN_PROVIDERS:
        raise LLMBadRequestError(f"Unknown LLM provider: {name!r}")
    if name not in _cache:
        _cache[name] = _build(name)
    return _cache[name]


def available_providers() -> list[dict]:
    out: list[dict] = []
    cloud_keyed = {
        "gemini": bool(settings.gemini_api_key),
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "vertex": bool(settings.vertex_project),
    }
    for name in ("gemini", "openai", "anthropic", "openrouter", "vertex"):
        out.append({"name": name, "model": resolve_model(name) or "(default)",
                    "supports_vision": name in ("gemini", "vertex", "anthropic"),
                    "configured": cloud_keyed[name]})
    for name in ("ollama", "lmstudio"):  # local servers: always selectable
        out.append({"name": name, "model": resolve_model(name) or "(auto)",
                    "supports_vision": False, "configured": True})
    return out
```

- [ ] **Step 4: Export from __init__.py**

Add to `backend/app/services/ai/llm/__init__.py`:

```python
from app.services.ai.llm.registry import (  # noqa: E402
    available_providers, get_provider, provider_name_for, resolve_model,
    KNOWN_PROVIDERS,
)
```
and extend `__all__` with `"get_provider", "available_providers", "provider_name_for", "resolve_model", "KNOWN_PROVIDERS"`.

- [ ] **Step 5: Run tests** → PASS. **Step 6: Commit**

```bash
git add backend/app/services/ai/llm/registry.py backend/app/services/ai/llm/__init__.py backend/tests/test_llm_registry.py
git commit -m "feat(llm): provider registry with per-operation routing"
```

---

### Task 7: Refactor summarizer to the facade

**Files:**
- Modify: `backend/app/services/ai/summarizer.py:1-214`
- Test: `backend/tests/test_summarizer_provider.py` (new); existing summarizer tests must still pass.

**Interfaces:**
- Consumes: `get_provider`, `LLMRequest`, `LLMMessage`, `ReasoningConfig` (Tasks 2/6).
- `generate_summary(...)` gains two optional params: `provider: str | None = None`, `model: str | None = None`. When `provider` is given it overrides the routed provider for this call.

- [ ] **Step 1: Write the failing test** (provider is selected + receives scrubbed text)

```python
# backend/tests/test_summarizer_provider.py
from __future__ import annotations
from unittest.mock import AsyncMock, patch
import pytest
from app.services.ai.llm.types import LLMResponse, LLMUsage


@pytest.mark.asyncio
async def test_summary_uses_selected_provider_and_scrubbed_text(db_session, seed_test_records):
    # seed_test_records -> (user_id, patient_id) with a record containing a known name
    user_id, patient_id = seed_test_records
    fake_resp = LLMResponse(text="A summary.", finish_reason="stop",
                            model="test-model", usage=LLMUsage(1, 1, 2), raw=None)
    fake_provider = AsyncMock()
    fake_provider.complete.return_value = fake_resp
    fake_provider.name = "anthropic"
    with patch("app.services.ai.summarizer.get_provider", return_value=fake_provider):
        from app.services.ai.summarizer import generate_summary
        out = await generate_summary(db_session, user_id, patient_id,
                                     provider="anthropic")
    assert out["natural_language"] == "A summary."
    assert out["model_used"] == "test-model"
    # The text sent to the provider must already be de-identified.
    sent_request = fake_provider.complete.call_args.args[0]
    assert "REDACTED" in sent_request.messages[0].content or sent_request.system
```

(Use existing `conftest` fixtures; adapt `seed_test_records` to the project's actual signature — see `tests/conftest.py`.)

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Refactor summarizer.py**

Replace the imports `from google import genai` / `from google.genai import types` with:

```python
from app.services.ai.llm import LLMMessage, LLMRequest, ReasoningConfig, get_provider
```

Change the signature to add `provider: str | None = None, model: str | None = None`. Replace the "Call Gemini" block (lines ~153-214) with:

```python
    # Resolve provider: explicit arg > routed summary provider.
    from app.services.ai.llm.registry import get_provider as _routed
    llm = get_provider("summary") if provider is None else _provider_by_name(provider)

    request = LLMRequest(
        messages=[LLMMessage("user", user_prompt)],
        model=model or "",  # blank => provider's configured default
        system=system_prompt,
        max_output_tokens=settings.gemini_summary_max_tokens,
        temperature=settings.gemini_summary_temperature,
        json_mode=output_format in ("json", "both"),
        reasoning=ReasoningConfig(level=settings.gemini_summary_thinking_level),
    )
    response = await llm.complete(request)
    response_text = response.text or ""
    # ... existing parse-by-format block unchanged ...
    tokens_used = response.usage.total_tokens or None
    # model_used = response.model
```

Add a tiny helper at module level:

```python
def _provider_by_name(name: str):
    """Build a one-off provider for an explicit per-request override."""
    from app.services.ai.llm.registry import _build, KNOWN_PROVIDERS
    from app.services.ai.llm.types import LLMBadRequestError
    if name not in KNOWN_PROVIDERS:
        raise LLMBadRequestError(f"Unknown provider: {name!r}")
    return _build(name)
```

Keep the `if not settings.gemini_api_key` guard ONLY when the resolved provider is gemini (move it: raise the existing ValueError only if `provider in (None,"gemini")` and no key — otherwise the provider itself raises a normalized auth error). Update the keys in the returned dict: `"model_used": response.model`. The `generate_content` import of `types`/`genai` is now unused — remove it.

- [ ] **Step 4: Run new + existing summarizer tests**

Run: `cd backend && python -m pytest tests/test_summarizer_provider.py tests/test_summarizer.py -v` (use the real summarizer test filename if different)
Expected: PASS. Fix any fixture-signature mismatches.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ai/summarizer.py backend/tests/test_summarizer_provider.py
git commit -m "refactor(summary): route summarization through LLM provider facade"
```

---

### Task 8: Refactor section_parser to the facade

**Files:**
- Modify: `backend/app/services/extraction/section_parser.py:1-192`
- Test: `backend/tests/test_section_parser_provider.py` (existing `test_section_parser*.py` must still pass)

**Interfaces:**
- `_call_gemini_for_sections(text, api_key)` becomes `_call_llm_for_sections(text)` using `get_provider("section")`. `parse_sections(text, api_key)` keeps its signature (api_key now unused but retained for caller compatibility; document it).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_section_parser_provider.py
from __future__ import annotations
import json
from unittest.mock import AsyncMock, patch
import pytest
from app.services.ai.llm.types import LLMResponse, LLMUsage
from app.services.extraction.section_parser import parse_sections, SectionType


@pytest.mark.asyncio
async def test_section_parser_uses_facade():
    payload = {"document_type": "note", "primary_visit_date": None,
               "provider": None, "facility": None,
               "sections": [{"type": "medications", "anchor": "MEDICATIONS"}]}
    resp = LLMResponse(text=json.dumps(payload), finish_reason="stop",
                       model="m", usage=LLMUsage(1, 1, 2), raw=None)
    prov = AsyncMock(); prov.complete.return_value = resp
    with patch("app.services.extraction.section_parser.get_provider", return_value=prov):
        doc = await parse_sections("MEDICATIONS\nlisinopril 10mg", api_key="unused")
    assert any(s.section_type == SectionType.MEDICATIONS for s in doc.sections)
    req = prov.complete.call_args.args[0]
    assert req.json_mode is True
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Refactor**

Replace `from google import genai` with `from app.services.ai.llm import LLMMessage, LLMRequest, get_provider`. Rewrite `_call_gemini_for_sections`:

```python
async def _call_llm_for_sections(text: str) -> dict:
    """Parse document sections (type + verbatim anchor) via the LLM facade."""
    llm = get_provider("section")
    request = LLMRequest(
        messages=[LLMMessage("user",
            f"{_SECTION_PARSER_PROMPT}\n\n---\n\nDOCUMENT TEXT:\n{text}")],
        model="", max_output_tokens=settings.gemini_summary_max_tokens,
        temperature=0.1, json_mode=True, json_schema=_SectionParseSchema,
    )
    response = await llm.complete(request)
    return json.loads(response.text)
```

Update `parse_sections` to call `await _call_llm_for_sections(text)` (drop the `api_key` arg internally; keep it in the public signature). The `_SectionParseSchema` is passed as `json_schema`; the Gemini provider forwards it to `response_schema`, other providers ignore it (json_object mode) — both yield parseable JSON.

- [ ] **Step 4: Run tests** → PASS (`tests/test_section_parser*.py`). **Step 5: Commit**

```bash
git add backend/app/services/extraction/section_parser.py backend/tests/test_section_parser_provider.py
git commit -m "refactor(section): route section parsing through LLM facade"
```

---

### Task 9: Refactor llm_judge to the facade

**Files:**
- Modify: `backend/app/services/dedup/llm_judge.py:1-139`
- Test: `backend/tests/test_llm_judge.py` (update patch target), `backend/tests/test_llm_judge_provider.py` (new)

**Interfaces:**
- `judge_candidate_pair(fhir_a, fhir_b, record_type, api_key)` keeps its signature (`api_key` retained, unused internally; routing via `get_provider("dedup")`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm_judge_provider.py
from __future__ import annotations
import json
from unittest.mock import AsyncMock, patch
import pytest
from app.services.ai.llm.types import LLMResponse, LLMUsage
from app.services.dedup.llm_judge import judge_candidate_pair


@pytest.mark.asyncio
async def test_judge_uses_facade_and_strips_patient_fields():
    payload = {"classification": "duplicate", "confidence": 0.9,
               "explanation": "same", "field_diff": None}
    resp = LLMResponse(text=json.dumps(payload), finish_reason="stop",
                       model="m", usage=LLMUsage(1, 1, 2), raw=None)
    prov = AsyncMock(); prov.complete.return_value = resp
    with patch("app.services.dedup.llm_judge.get_provider", return_value=prov):
        out = await judge_candidate_pair(
            {"resourceType": "Condition", "subject": {"display": "Jane Doe"}},
            {"resourceType": "Condition"}, "condition", api_key="unused")
    assert out.classification == "duplicate"
    sent = prov.complete.call_args.args[0].messages[0].content
    assert "Jane Doe" not in sent  # subject stripped before send
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Refactor llm_judge.py**

Replace `from google import genai` with `from app.services.ai.llm import LLMMessage, LLMRequest, get_provider`. Rewrite the body of `judge_candidate_pair` (keep the try/except → `error_fallback()`):

```python
        llm = get_provider("dedup")
        content = (
            f"{_JUDGE_PROMPT}\n\nRecord type: {record_type}\n\n"
            f"Record A:\n{json.dumps(_strip_patient_fields(fhir_a), indent=2)}\n\n"
            f"Record B:\n{json.dumps(_strip_patient_fields(fhir_b), indent=2)}"
        )
        response = await llm.complete(LLMRequest(
            messages=[LLMMessage("user", content)], model="",
            max_output_tokens=2048, temperature=0.1, json_mode=True))
        data = json.loads(response.text)
        return JudgmentResult.from_llm_response(data)
```

- [ ] **Step 4: Update existing test patch target**

In `backend/tests/test_llm_judge.py`, replace patches of `app.services.dedup.llm_judge.genai.Client` with patching `app.services.dedup.llm_judge.get_provider` to return an `AsyncMock` whose `complete` returns an `LLMResponse` with the canned JSON (mirror the new-test pattern). Update each of the ~5 call sites.

- [ ] **Step 5: Run tests** → PASS (`tests/test_llm_judge.py tests/test_llm_judge_provider.py`). **Step 6: Commit**

```bash
git add backend/app/services/dedup/llm_judge.py backend/tests/test_llm_judge.py backend/tests/test_llm_judge_provider.py
git commit -m "refactor(dedup): route LLM judge through provider facade"
```

---

### Task 10: Generic entity extraction path + provider branch

**Files:**
- Create: `backend/app/services/extraction/generic_entity_extractor.py`
- Modify: `backend/app/services/extraction/entity_extractor.py` (add async provider branch)
- Test: `backend/tests/test_generic_entity_extract.py`

**Interfaces:**
- Produces: `async def generic_extract_entities_async(text, source_file, *, progress_callback=None) -> ExtractionResult` — uses `get_provider("extraction")` with `json_mode`, parses a JSON entities array into `ExtractedEntity` objects (same shape LangExtract yields).
- Modify `extract_entities_async`: when `provider_name_for("extraction") == "gemini"` (or "vertex") use LangExtract as today; else delegate to `generic_extract_entities_async`. Keep `api_key` param.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_generic_entity_extract.py
from __future__ import annotations
import json
from unittest.mock import AsyncMock, patch
import pytest
from app.services.ai.llm.types import LLMResponse, LLMUsage
from app.services.extraction.generic_entity_extractor import generic_extract_entities_async


@pytest.mark.asyncio
async def test_generic_extract_parses_entities():
    payload = {"entities": [
        {"entity_class": "medication", "text": "lisinopril",
         "attributes": {"dosage": "10mg", "confidence": 0.9}},
        {"entity_class": "condition", "text": "hypertension", "attributes": {}},
    ]}
    resp = LLMResponse(text=json.dumps(payload), finish_reason="stop",
                       model="m", usage=LLMUsage(1, 1, 2), raw=None)
    prov = AsyncMock(); prov.complete.return_value = resp
    with patch("app.services.extraction.generic_entity_extractor.get_provider", return_value=prov):
        result = await generic_extract_entities_async("lisinopril 10mg for hypertension", "f.txt")
    classes = {e.entity_class for e in result.entities}
    assert {"medication", "condition"} <= classes
    med = next(e for e in result.entities if e.entity_class == "medication")
    assert med.text == "lisinopril" and med.confidence == 0.9


@pytest.mark.asyncio
async def test_generic_extract_handles_bad_json():
    resp = LLMResponse(text="not json", finish_reason="stop", model="m",
                       usage=LLMUsage(1, 1, 2), raw=None)
    prov = AsyncMock(); prov.complete.return_value = resp
    with patch("app.services.extraction.generic_entity_extractor.get_provider", return_value=prov):
        result = await generic_extract_entities_async("x", "f.txt")
    assert result.entities == [] and result.error is not None
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement generic_entity_extractor.py**

```python
# backend/app/services/extraction/generic_entity_extractor.py
from __future__ import annotations
import json
import logging
from collections.abc import Callable
from app.services.ai.llm import LLMMessage, LLMRequest, get_provider
from app.services.extraction.clinical_examples import CLINICAL_EXTRACTION_PROMPT
from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult

logger = logging.getLogger(__name__)

_SCHEMA_HINT = """
Return ONLY JSON of the form:
{"entities": [{"entity_class": "<medication|condition|procedure|lab|vital|allergy|provider>",
  "text": "<verbatim span>", "attributes": {"<k>": "<v>", "confidence": 0.0-1.0}}]}
Extract only entities explicitly present. Do not infer. Omit negated/family-history items
as performed/active. If none, return {"entities": []}.
"""


async def generic_extract_entities_async(
    text: str, source_file: str,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ExtractionResult:
    """Provider-agnostic clinical entity extraction via JSON-mode completion."""
    llm = get_provider("extraction")
    prompt = f"{CLINICAL_EXTRACTION_PROMPT}\n{_SCHEMA_HINT}\n\nTEXT:\n{text}"
    try:
        resp = await llm.complete(LLMRequest(
            messages=[LLMMessage("user", prompt)], model="",
            max_output_tokens=4096, temperature=0.0, json_mode=True))
        data = json.loads(resp.text)
        raw = data.get("entities", []) if isinstance(data, dict) else []
        entities: list[ExtractedEntity] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            attrs = item.get("attributes") or {}
            conf = 0.8
            try:
                conf = max(0.0, min(1.0, float(attrs.get("confidence", 0.8))))
            except (ValueError, TypeError):
                pass
            entities.append(ExtractedEntity(
                entity_class=str(item.get("entity_class", "other")),
                text=str(item["text"]), attributes=attrs, confidence=conf))
        if progress_callback is not None:
            try:
                progress_callback("extracting_entities", 1, len(entities))
            except Exception:
                logger.debug("progress_callback raised; ignoring", exc_info=True)
        return ExtractionResult(source_file=source_file, source_text=text,
                                entities=entities)
    except Exception as e:
        logger.error("Generic entity extraction failed for %s: %s", source_file, e)
        return ExtractionResult(source_file=source_file, source_text=text, error=str(e))
```

- [ ] **Step 4: Branch in entity_extractor.extract_entities_async**

```python
async def extract_entities_async(text, source_file, api_key, progress_callback=None):
    """Async entity extraction. Gemini/Vertex => LangExtract; else generic JSON path."""
    from app.services.ai.llm.registry import provider_name_for
    if provider_name_for("extraction") in ("gemini", "vertex"):
        return await asyncio.to_thread(
            extract_entities, text, source_file, api_key, progress_callback)
    from app.services.extraction.generic_entity_extractor import (
        generic_extract_entities_async)
    return await generic_extract_entities_async(text, source_file, progress_callback)
```

- [ ] **Step 5: Run tests** → PASS. **Step 6: Commit**

```bash
git add backend/app/services/extraction/generic_entity_extractor.py backend/app/services/extraction/entity_extractor.py backend/tests/test_generic_entity_extract.py
git commit -m "feat(extraction): provider-agnostic JSON entity path for non-Gemini providers"
```

---

### Task 11: Vision/OCR capability guard

**Files:**
- Modify: `backend/app/services/extraction/text_extractor.py` (PDF/TIFF Gemini fallback)
- Test: `backend/tests/test_text_extractor_guard.py`

**Interfaces:**
- OCR stays on Gemini. Add `_vision_client_or_error()` that uses the Gemini key directly (independent of `LLM_PROVIDER`), raising `ValueError("Vision OCR requires GEMINI_API_KEY ...")` when absent — so a non-Gemini global provider doesn't break OCR as long as a Gemini key exists.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_text_extractor_guard.py
from __future__ import annotations
import pytest
from pathlib import Path
from app.services.extraction import text_extractor


@pytest.mark.asyncio
async def test_tiff_ocr_requires_gemini_key(monkeypatch, tmp_path):
    monkeypatch.setattr(text_extractor.settings, "gemini_api_key", "")
    # api_key="" simulates no Gemini available
    f = tmp_path / "x.tiff"; f.write_bytes(b"II*\x00")  # minimal tiff magic
    with pytest.raises(ValueError, match="Vision OCR requires"):
        await text_extractor.extract_text_from_tiff(f, api_key="")
```

- [ ] **Step 2: Run to verify it fails** → FAIL (currently raises a genai error, not ValueError).

- [ ] **Step 3: Add the guard**

At the top of `_extract_text_from_pdf_gemini` and `extract_text_from_tiff`, before constructing the client:

```python
    if not api_key:
        raise ValueError(
            "Vision OCR requires GEMINI_API_KEY (vision is Gemini-only regardless "
            "of LLM_PROVIDER).")
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add backend/app/services/extraction/text_extractor.py backend/tests/test_text_extractor_guard.py
git commit -m "feat(extraction): explicit Gemini-key guard for vision OCR"
```

---

### Task 12: API — provider/model on generate + GET /summary/providers

**Files:**
- Modify: `backend/app/schemas/summary.py` (`GenerateSummaryRequest`: add `provider`, `model`)
- Modify: `backend/app/api/summary.py` (thread params into `generate_summary`; add `GET /summary/providers`)
- Test: `backend/tests/test_summary_providers_api.py`

**Interfaces:**
- `GenerateSummaryRequest` gains `provider: str | None = None`, `model: str | None = None`.
- `GET /summary/providers` → `{"providers": available_providers(), "default": provider_name_for("summary")}` (auth-required, audited, no keys).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_summary_providers_api.py
from __future__ import annotations
import pytest


@pytest.mark.asyncio
async def test_list_providers(async_client, auth_headers):
    r = await async_client.get("/api/v1/summary/providers", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    names = {p["name"] for p in body["providers"]}
    assert {"gemini", "openai", "anthropic", "ollama", "lmstudio"} <= names
    assert body["default"]  # a default provider name
    # never leak keys
    assert "api_key" not in r.text and "sk-" not in r.text
```

(Adapt `async_client`/`auth_headers` to the project's conftest fixture names.)

- [ ] **Step 2: Run to verify it fails** → FAIL (404).

- [ ] **Step 3: Implement**

In `schemas/summary.py` `GenerateSummaryRequest`, add:

```python
    provider: str | None = None
    model: str | None = None
```

In `api/summary.py`, thread into the existing `generate_summary(...)` call (the route ~line 296): add `provider=request.provider, model=request.model`. Add the new route (place it with the other `@router.get` routes):

```python
@router.get("/providers")
async def list_providers(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """List configured LLM providers (names + capabilities only; never keys)."""
    from app.services.ai.llm.registry import available_providers, provider_name_for
    await log_audit(db, current_user.id, "summary.providers.list", "summary", None)
    return {"providers": available_providers(), "default": provider_name_for("summary")}
```

(Use the project's actual audit/log helper + dependency names — match the neighbouring routes in the file.)

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add backend/app/schemas/summary.py backend/app/api/summary.py backend/tests/test_summary_providers_api.py
git commit -m "feat(api): provider/model on summary generate + GET /summary/providers"
```

---

### Task 13: Frontend — provider selector on Summarize

**Files:**
- Modify: `frontend/src/lib/api.ts` (add `getSummaryProviders`, extend `generateSummary` body with provider/model)
- Modify: `frontend/src/types/*` (provider type)
- Modify: the Summarize page/component that calls generate (find via `generateSummary` usage)
- Test: `frontend/e2e/multi-provider-summary.spec.ts` (Task 14 covers e2e)

**Interfaces:**
- `getSummaryProviders(): Promise<{providers: ProviderInfo[]; default: string}>`
- `generateSummary` accepts optional `provider?: string; model?: string`.

- [ ] **Step 1:** Add `getSummaryProviders` to `lib/api.ts` (GET `/summary/providers`) and a `ProviderInfo` type `{name: string; model: string; supports_vision: boolean; configured: boolean}`.
- [ ] **Step 2:** Extend the `generateSummary` request body type/param with optional `provider`, `model`; pass through.
- [ ] **Step 3:** In the Summarize UI, add a compact `<select>` (shadcn Select) populated from `getSummaryProviders`, default to `default`; disable options where `configured === false` (except local). Store choice in component state; pass to `generateSummary`. When a cloud provider is selected, render a one-line note: "De-identified data will be sent to {provider}." Keep the existing AI disclaimer.
- [ ] **Step 4:** `cd frontend && npm run build` → typechecks. Manual: selector renders, defaults correctly.
- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/types frontend/src/app frontend/src/components
git commit -m "feat(ui): provider selector on Summarize page"
```

---

### Task 14: Live smoke tests + e2e

**Files:**
- Create: `backend/tests/test_llm_live_smoke.py` (marked `slow`, gated on key/server presence)
- Create: `frontend/e2e/multi-provider-summary.spec.ts`

**Interfaces:** none (test-only).

- [ ] **Step 1: Live smoke (skip when unconfigured)**

```python
# backend/tests/test_llm_live_smoke.py
from __future__ import annotations
import httpx
import pytest
from app.config import settings
from app.services.ai.llm.registry import _build
from app.services.ai.llm.types import LLMMessage, LLMRequest

pytestmark = pytest.mark.slow


def _ollama_up() -> bool:
    try:
        httpx.get(settings.ollama_base_url.replace("/v1", "") + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


async def _smoke(name: str):
    prov = _build(name)
    resp = await prov.complete(LLMRequest(
        messages=[LLMMessage("user", "Reply with the single word: ok")],
        model="", max_output_tokens=20, temperature=0.0))
    assert resp.text.strip()


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.openai_api_key, reason="no OpenAI key")
async def test_openai_smoke():
    await _smoke("openai")


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.anthropic_api_key, reason="no Anthropic key")
async def test_anthropic_smoke():
    await _smoke("anthropic")


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.gemini_api_key, reason="no Gemini key")
async def test_gemini_smoke():
    await _smoke("gemini")


@pytest.mark.asyncio
@pytest.mark.skipif(not _ollama_up(), reason="ollama not running")
async def test_ollama_smoke():
    await _smoke("ollama")
```

(Add an `lmstudio` smoke guarded by a `_lmstudio_up()` probe of `lmstudio_base_url + "/models"`.)

- [ ] **Step 2: Run live smoke** (after keys are in `.env` and local servers started)

Run: `cd backend && python -m pytest tests/test_llm_live_smoke.py -v -m slow`
Expected: configured providers PASS; unconfigured SKIP.

- [ ] **Step 3: E2E spec**

`frontend/e2e/multi-provider-summary.spec.ts` — import `test`/`expect` from the console-gate fixture; seed records via the existing helper; navigate to Summarize; for each available non-Gemini provider in the selector, generate and assert a summary + AI disclaimer + de-id report render; skip a provider if it isn't in the selector. Keep within the suite's worker/retry conventions.

- [ ] **Step 4: Run e2e**

Run: `cd frontend && npx playwright test multi-provider-summary --reporter=list`
Expected: PASS (or graceful skips).

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_llm_live_smoke.py frontend/e2e/multi-provider-summary.spec.ts
git commit -m "test(llm): live provider smoke + multi-provider summary e2e"
```

---

### Task 15: Full regression + docs

**Files:**
- Modify: `CLAUDE.local.md` (Rules #3/#16; new AI Providers section; gotchas) — at ship time
- Verify: full fast suite green

- [ ] **Step 1: Backend fast suite**

Run: `cd backend && python -m pytest -m "not slow" -x -q`
Expected: all pass (prior ~1214 + new tests). Fix regressions (esp. any other `genai.Client` patch targets in older tests).

- [ ] **Step 2: Grep for stale direct genai calls**

Run: `cd backend && grep -rn "genai.Client" app/ | grep -v "services/ai/llm/gemini.py\|services/extraction/text_extractor.py"`
Expected: empty (all text-gen call sites now route through the facade; only the LLM Gemini provider + vision OCR construct a client directly).

- [ ] **Step 3:** Update `CLAUDE.local.md` (handled by the `ship` skill / revise-claude-md at the end). Commit any doc-only changes.

```bash
git commit -am "docs: note multi-provider LLM support"
```

---

## Self-Review

**Spec coverage:** providers (Tasks 3-5), registry/routing (6), all four text ops (7-10), vision guard (11), API + UI (12-13), de-id-before-send guarded by tests in 7/9/10, testing incl. live + e2e (14), back-compat default + regression (1,6,15), licenses/deps (1). Vertex code-complete via GeminiProvider vertexai mode (3,6), live-untested by design. All spec sections map to a task.

**Placeholders:** none — concrete code/tests in each code step. Frontend steps (13) are description-level by necessity (selector wiring varies with the actual Summarize component); the executor must locate the component via `generateSummary` usage.

**Type consistency:** `LLMRequest`/`LLMResponse`/`LLMMessage`/`ReasoningConfig`/`LLMUsage` names consistent across Tasks 2-14; `get_provider(operation)`, `_build(name)`, `provider_name_for`, `available_providers`, `resolve_model` consistent in Tasks 6-12; `ExtractedEntity`/`ExtractionResult` reused unchanged from `entity_extractor` in Task 10.
