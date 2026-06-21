from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter", "other"]


@dataclass
class TextPart:
    """A plain-text segment of a message."""

    text: str


@dataclass
class ImagePart:
    """An inline image (e.g. a scanned page) — raw bytes + MIME type."""

    data: bytes
    mime: str


@dataclass
class DocumentPart:
    """An inline document (e.g. a PDF) — raw bytes + MIME type."""

    data: bytes
    mime: str


ContentPart = "TextPart | ImagePart | DocumentPart"


def as_parts(content: "str | list[TextPart | ImagePart | DocumentPart]") -> list:
    """Normalize message content to a list of parts (a ``str`` becomes one TextPart)."""
    if isinstance(content, str):
        return [TextPart(content)]
    return list(content)


@dataclass
class LLMMessage:
    role: Role
    content: "str | list[TextPart | ImagePart | DocumentPart]"


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
