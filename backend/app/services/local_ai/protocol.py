"""Versioned, bounded JSON-Lines protocol for local model workers."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from app.services.local_ai.types import ModelRole

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
WorkerCommand = Literal["health", "ocr", "extract", "summarize", "cancel", "shutdown"]
ResponseKind = Literal["ready", "progress", "result", "error"]
ProgressStage = Literal["starting", "loading", "processing", "finalizing", "cancelling"]
ErrorCode = Literal[
    "cancelled",
    "invalid_request",
    "protocol_error",
    "unavailable",
    "worker_failed",
]
ProtocolVersion = Annotated[StrictInt, Field(ge=PROTOCOL_VERSION, le=PROTOCOL_VERSION)]

SAFE_ERROR_MESSAGES: dict[str, str] = {
    "cancelled": "Local worker cancelled.",
    "invalid_request": "Local worker request was rejected.",
    "protocol_error": "Local worker protocol failed.",
    "unavailable": "Local worker is unavailable.",
    "worker_failed": "Local worker failed.",
}


class ProtocolViolation(ValueError):
    """A safe protocol validation failure with no worker-supplied detail."""


class _StrictMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WorkerRequest(_StrictMessage):
    """One bounded command sent to a fresh worker process."""

    version: ProtocolVersion
    request_id: Identifier
    job_id: Identifier
    command: WorkerCommand
    payload: dict[str, JsonValue]


class ReadyPayload(_StrictMessage):
    """Non-content worker readiness metadata."""

    role: ModelRole | None = None


class ProgressPayload(_StrictMessage):
    """Allowlisted non-content progress metadata."""

    role: ModelRole
    stage: ProgressStage
    current: Annotated[StrictInt, Field(ge=0)]
    total: Annotated[StrictInt, Field(ge=0)]

    @model_validator(mode="after")
    def current_does_not_exceed_total(self) -> ProgressPayload:
        if self.current > self.total:
            raise ValueError("current cannot exceed total")
        return self


class ResultPayload(_StrictMessage):
    """Content-bearing terminal worker result kept inside owner-only IPC."""

    data: JsonValue


class ErrorPayload(_StrictMessage):
    """Stable, non-content terminal worker error."""

    code: ErrorCode
    message: str

    @model_validator(mode="after")
    def message_is_fixed_for_code(self) -> ErrorPayload:
        if self.message != SAFE_ERROR_MESSAGES[self.code]:
            raise ValueError("error message must match its safe code")
        return self


class WorkerResponse(_StrictMessage):
    """One validated worker response."""

    version: ProtocolVersion
    request_id: Identifier
    kind: ResponseKind
    payload: ReadyPayload | ProgressPayload | ResultPayload | ErrorPayload

    @model_validator(mode="after")
    def payload_matches_kind(self) -> WorkerResponse:
        expected_type: dict[str, type[_StrictMessage]] = {
            "ready": ReadyPayload,
            "progress": ProgressPayload,
            "result": ResultPayload,
            "error": ErrorPayload,
        }
        if not isinstance(self.payload, expected_type[self.kind]):
            raise ValueError("response payload does not match response kind")
        return self


def encode_message(message: WorkerRequest | WorkerResponse) -> bytes:
    """Encode exactly one bounded JSON-Line message."""
    encoded = message.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolViolation("Worker protocol message exceeds the size limit.")
    return encoded + b"\n"


def _bounded_line(line: bytes) -> bytes:
    if len(line) > MAX_MESSAGE_BYTES + 1:
        raise ProtocolViolation("Worker protocol message exceeds the size limit.")
    if not line.endswith(b"\n"):
        raise ProtocolViolation("Worker protocol message is malformed.")
    content = line[:-1]
    if not content or len(content) > MAX_MESSAGE_BYTES:
        raise ProtocolViolation("Worker protocol message exceeds the size limit.")
    return content


def _parse_strict_json(content: bytes) -> object:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite numeric constant")

    return json.loads(content, parse_constant=reject_constant)


_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


def validate_identifier(value: object) -> str:
    """Validate a manager-facing identifier before it reaches state maps."""
    try:
        return _IDENTIFIER_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        raise ProtocolViolation("Worker protocol identifier is invalid.") from None


def parse_request_line(line: bytes) -> WorkerRequest:
    """Parse a request only after enforcing the byte ceiling."""
    content = _bounded_line(line)
    try:
        return WorkerRequest.model_validate(_parse_strict_json(content))
    except (
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
        UnicodeDecodeError,
        ValueError,
    ):
        raise ProtocolViolation("Worker protocol message is malformed.") from None


def parse_response_line(line: bytes) -> WorkerResponse:
    """Parse a response only after enforcing the byte ceiling."""
    content = _bounded_line(line)
    try:
        return WorkerResponse.model_validate(_parse_strict_json(content))
    except (
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
        UnicodeDecodeError,
        ValueError,
    ):
        raise ProtocolViolation("Worker protocol message is malformed.") from None
