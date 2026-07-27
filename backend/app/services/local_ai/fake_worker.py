"""Deterministic protocol worker used by public tests without model dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from app.services.local_ai.protocol import (
    MAX_MESSAGE_BYTES,
    ErrorPayload,
    ProgressPayload,
    ProtocolViolation,
    ReadyPayload,
    ResultPayload,
    WorkerRequest,
    WorkerResponse,
    encode_message,
    parse_request_line,
)
from app.services.local_ai.types import ModelRole


def _write(response: WorkerResponse) -> None:
    sys.stdout.buffer.write(encode_message(response))
    sys.stdout.buffer.flush()


def _response(
    request_id: str,
    kind: str,
    payload: ReadyPayload | ProgressPayload | ResultPayload | ErrorPayload,
) -> WorkerResponse:
    return WorkerResponse(
        version=1,
        request_id=request_id,
        kind=kind,  # type: ignore[arg-type]
        payload=payload,
    )


def _role_for(request: WorkerRequest) -> ModelRole:
    if request.command == "ocr":
        return ModelRole.OCR
    if request.command == "extract":
        return ModelRole.EXTRACTION
    return ModelRole.SUMMARY


def _fixed_result(role: ModelRole, payload: dict[str, object]) -> dict[str, object]:
    if payload.get("inspect_env") is True:
        forbidden_names = {
            "ANTHROPIC_API_KEY",
            "DATABASE_URL",
            "GEMINI_API_KEY",
            "HF_TOKEN",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "OPENAI_API_KEY",
            "REDIS_URL",
        }
        return {
            "home_matches": Path.home() == Path(os.environ["HOME"]),
            "cwd_matches": Path.cwd() == Path(os.environ["HOME"]),
            "offline": (
                os.environ.get("HF_HUB_OFFLINE") == "1"
                and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
                and os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
            ),
            "path_is_fixed": os.environ.get("PATH") == os.defpath,
            "secret_names_present": sorted(forbidden_names.intersection(os.environ)),
        }
    if role is ModelRole.OCR:
        result: dict[str, object] = {
            "markdown": "# Synthetic OCR",
            "page_number": 1,
        }
        descendant_pid = payload.get("_descendant_pid")
        if isinstance(descendant_pid, int):
            result["descendant_pid"] = descendant_pid
        return result
    if role is ModelRole.EXTRACTION:
        return {"entities": [], "evidence": []}
    return {"sections": [{"facts": [], "title": "Synthetic summary"}]}


def _wait_for_shutdown() -> None:
    while True:
        line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
        if not line:
            return
        try:
            request = parse_request_line(line)
        except ProtocolViolation:
            return
        if request.command in {"cancel", "shutdown"}:
            return


def _spawn_descendant(ignore_term: bool) -> int:
    setup = (
        "import signal,time;"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN);" if ignore_term else "")
        + "time.sleep(3600)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", setup],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    pid_path = Path(os.environ["HOME"]) / "descendant.pid"
    pid_path.write_text(str(child.pid))
    os.chmod(pid_path, 0o600)
    return child.pid


def _main() -> int:
    line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
    try:
        request = parse_request_line(line)
    except ProtocolViolation:
        return 2

    payload = dict(request.payload)
    if request.command == "shutdown":
        return 0
    if request.command == "health":
        _write(_response(request.request_id, "ready", ReadyPayload()))
        _write(
            _response(
                request.request_id,
                "result",
                ResultPayload(data={"status": "healthy"}),
            )
        )
        _wait_for_shutdown()
        return 0
    if request.command == "cancel":
        _write(_response(request.request_id, "ready", ReadyPayload()))
        _write(
            _response(
                request.request_id,
                "error",
                ErrorPayload(code="cancelled", message="Local worker cancelled."),
            )
        )
        _wait_for_shutdown()
        return 0

    role = _role_for(request)
    response_id = "mismatched-request" if payload.get("mismatched_id") is True else request.request_id

    _write(_response(response_id, "ready", ReadyPayload(role=role)))

    if payload.get("spawn_descendant") is True:
        payload["_descendant_pid"] = _spawn_descendant(
            payload.get("descendant_ignores_term") is True
        )
    if payload.get("stderr_noise") is True:
        sys.stderr.write("synthetic worker noise\n" * 16_384)
        sys.stderr.flush()
    if payload.get("crash") is True:
        os._exit(7)
    if payload.get("malformed") is True:
        sys.stdout.buffer.write(b"{not-json}\n")
        sys.stdout.buffer.flush()
        _wait_for_shutdown()
        return 0
    if payload.get("oversized") is True:
        sys.stdout.buffer.write((b"x" * (MAX_MESSAGE_BYTES + 1)) + b"\n")
        sys.stdout.buffer.flush()
        _wait_for_shutdown()
        return 0

    _write(
        _response(
            response_id,
            "progress",
            ProgressPayload(role=role, stage="processing", current=0, total=1),
        )
    )
    delay_ms = payload.get("delay_ms", 0)
    if isinstance(delay_ms, int) and not isinstance(delay_ms, bool) and delay_ms > 0:
        time.sleep(min(delay_ms, 60_000) / 1000)
    if payload.get("block") is True:
        _wait_for_shutdown()
        return 0

    if payload.get("safe_error") is True:
        terminal = _response(
            response_id,
            "error",
            ErrorPayload(code="worker_failed", message="Local worker failed."),
        )
    else:
        terminal = _response(
            response_id,
            "result",
            ResultPayload(data=_fixed_result(role, payload)),
        )
    _write(terminal)

    if payload.get("duplicate_terminal") is True:
        _write(terminal)
    if payload.get("post_terminal") is True:
        _write(
            _response(
                response_id,
                "progress",
                ProgressPayload(role=role, stage="finalizing", current=1, total=1),
            )
        )

    _wait_for_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
