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


def test_response_kind_must_match_its_bounded_payload_schema() -> None:
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate(
            _response("result", {"code": "worker_failed", "message": "Local worker failed."})
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
