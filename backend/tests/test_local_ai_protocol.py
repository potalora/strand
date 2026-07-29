from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.services.local_ai.protocol import (
    MAX_MESSAGE_BYTES,
    ProtocolViolation,
    WorkerRequest,
    WorkerResponse,
    encode_message,
    parse_request_line,
    parse_response_line,
)


def _response(kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "request_id": "request-1",
        "kind": kind,
        "payload": payload,
    }


def test_protocol_rejects_unknown_version() -> None:
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(
            {
                "version": 2,
                "request_id": "request-1",
                "job_id": "job-1",
                "command": "ocr",
                "payload": {},
            }
        )


@pytest.mark.parametrize("version", [True, 1.0])
def test_protocol_version_is_a_strict_integer(version: object) -> None:
    request = {
        "version": version,
        "request_id": "request-1",
        "job_id": "job-1",
        "command": "ocr",
        "payload": {},
    }
    response = _response("ready", {})
    response["version"] = version

    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(request)
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(response)


@pytest.mark.parametrize("field", ["request_id", "job_id"])
def test_protocol_bounds_identifiers(field: str) -> None:
    message = {
        "version": 1,
        "request_id": "request-1",
        "job_id": "job-1",
        "command": "ocr",
        "payload": {},
    }
    message[field] = "x" * 129

    with pytest.raises(ValidationError):
        WorkerRequest.model_validate(message)


def test_protocol_rejects_unknown_command_and_extra_request_fields() -> None:
    base = {
        "version": 1,
        "request_id": "request-1",
        "job_id": "job-1",
        "payload": {},
    }
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({**base, "command": "diagnose"})
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({**base, "command": "ocr", "secret": "no"})


def test_progress_response_accepts_only_non_content_fields() -> None:
    response = WorkerResponse.model_validate(
        _response(
            "progress",
            {
                "role": "ocr",
                "stage": "processing",
                "current": 1,
                "total": 2,
            },
        )
    )

    assert response.kind == "progress"
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "progress",
                {
                    "role": "ocr",
                    "stage": "processing",
                    "current": 1,
                    "total": 2,
                    "excerpt": "patient content",
                },
            )
        )
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "progress",
                {"role": "ocr", "stage": "patient content", "current": 1, "total": 2},
            )
        )


def test_progress_response_allows_bounded_non_content_memory_metrics() -> None:
    response = WorkerResponse.model_validate(
        _response(
            "progress",
            {
                "role": "summary",
                "stage": "finalizing",
                "current": 1,
                "total": 1,
                "active_memory_bytes": 512 * 1024**2,
                "peak_memory_bytes": 6 * 1024**3,
            },
        )
    )

    assert response.payload.active_memory_bytes == 512 * 1024**2
    assert response.payload.peak_memory_bytes == 6 * 1024**3
    for invalid in (-1, True, 1.5, "1024"):
        payload = {
            "role": "summary",
            "stage": "finalizing",
            "current": 1,
            "total": 1,
            "active_memory_bytes": invalid,
            "peak_memory_bytes": 1024,
        }
        with pytest.raises(ValidationError):
            WorkerResponse.model_validate(_response("progress", payload))


def test_progress_activity_is_bounded_integer_only() -> None:
    response = WorkerResponse.model_validate(
        _response(
            "progress",
            {
                "role": "extraction",
                "stage": "processing",
                "current": 0,
                "total": 3,
                "activity": 2,
            },
        )
    )

    assert response.payload.activity == 2
    for invalid in (-1, True, 1.5, "2", 2**63):
        payload = {
            "role": "extraction",
            "stage": "processing",
            "current": 0,
            "total": 3,
            "activity": invalid,
        }
        with pytest.raises(ValidationError):
            WorkerResponse.model_validate(_response("progress", payload))


def test_health_ready_response_needs_no_model_role() -> None:
    response = WorkerResponse.model_validate(_response("ready", {}))

    assert response.kind == "ready"
    assert response.payload.role is None


def test_error_response_cannot_contain_raw_detail_or_arbitrary_message() -> None:
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {
                    "code": "worker_failed",
                    "message": "Local worker failed.",
                    "raw": "PHI",
                },
            )
        )
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {"code": "worker_failed", "message": "Traceback: patient content"},
            )
        )


def test_input_limit_error_uses_only_its_fixed_non_content_message() -> None:
    response = WorkerResponse.model_validate(
        _response(
            "error",
            {
                "code": "input_limit_exceeded",
                "message": ("Local worker input exceeds supported limits."),
            },
        )
    )

    assert response.payload.code == "input_limit_exceeded"
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {
                    "code": "input_limit_exceeded",
                    "message": "53676 patient tokens exceeded 32768",
                },
            )
        )


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("generation_failed", "Local worker generation failed."),
        ("runtime_failed", "Local worker runtime failed."),
        ("resource_exhausted", "Local worker resources were exhausted."),
    ],
)
def test_runtime_error_codes_use_only_fixed_non_content_messages(
    code: str,
    message: str,
) -> None:
    response = WorkerResponse.model_validate(
        _response("error", {"code": code, "message": message})
    )

    assert response.payload.code == code
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {
                    "code": code,
                    "message": "private runtime detail",
                },
            )
        )


@pytest.mark.parametrize("category", ["output_limit", "invalid_structured_output"])
def test_generation_failure_accepts_only_allowlisted_content_free_categories(
    category: str,
) -> None:
    response = WorkerResponse.model_validate(
        _response(
            "error",
            {
                "code": "generation_failed",
                "message": "Local worker generation failed.",
                "category": category,
            },
        )
    )

    assert response.payload.category == category
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {
                    "code": "generation_failed",
                    "message": "Local worker generation failed.",
                    "category": "patient-content",
                },
            )
        )


def test_generation_failure_category_is_optional_and_code_scoped() -> None:
    legacy = WorkerResponse.model_validate(
        _response(
            "error",
            {
                "code": "generation_failed",
                "message": "Local worker generation failed.",
            },
        )
    )

    assert legacy.payload.category is None
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "error",
                {
                    "code": "runtime_failed",
                    "message": "Local worker runtime failed.",
                    "category": "output_limit",
                },
            )
        )


def test_response_kind_must_match_its_bounded_payload_schema() -> None:
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "result", {"code": "worker_failed", "message": "Local worker failed."}
            )
        )
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(_response("unknown", {"role": "ocr"}))


def test_parser_rejects_malformed_and_oversized_lines_before_json_parsing() -> None:
    with pytest.raises(ProtocolViolation, match="malformed"):
        parse_response_line(b"{not-json}\n")

    oversized = b"{" + (b"x" * MAX_MESSAGE_BYTES) + b"}\n"
    with pytest.raises(ProtocolViolation, match="size"):
        parse_response_line(oversized)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_non_json_numeric_constants(constant: str) -> None:
    line = (
        '{"version":1,"request_id":"request-1","kind":"result",'
        f'"payload":{{"data":{constant}}}}}\n'
    ).encode()

    with pytest.raises(ProtocolViolation, match="malformed"):
        parse_response_line(line)


@pytest.mark.parametrize("parser", [parse_request_line, parse_response_line])
def test_parser_normalizes_recursion_errors(parser: object) -> None:
    deeply_nested = b"[" * 2_000 + b"]" * 2_000 + b"\n"

    with pytest.raises(ProtocolViolation, match="malformed"):
        parser(deeply_nested)  # type: ignore[operator]


@pytest.mark.parametrize("counter", [True, 1.0, "1"])
def test_progress_counters_are_strict_integers(counter: object) -> None:
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response(
                "progress",
                {
                    "role": "ocr",
                    "stage": "processing",
                    "current": counter,
                    "total": 2,
                },
            )
        )


def test_encoder_outputs_one_bounded_json_line() -> None:
    request = WorkerRequest(
        version=1,
        request_id="request-1",
        job_id="job-1",
        command="health",
        payload={},
    )

    encoded = encode_message(request)

    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert len(encoded) <= MAX_MESSAGE_BYTES + 1
    assert json.loads(encoded) == request.model_dump(mode="json")
