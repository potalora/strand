"""Version-1 JSON-Lines entry point for the isolated MLX worker."""

from __future__ import annotations

import json
import os
import re
import select
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Literal

from .common import (
    GENERATION_FAILURE_CATEGORIES,
    ArtifactUnavailableError,
    GenerationError,
    WorkerInputError,
    WorkerInputLimitError,
)
from .nuextract3 import run_extraction
from .ovisocr2 import run_ocr
from .qwen_summary import run_summary

PROTOCOL_VERSION = 1
RUNTIME = "mlx-vlm-0.5.0"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_EXTRACTION_ACTIVITY = 2**63 - 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
COMMANDS = frozenset({"health", "ocr", "extract", "summarize", "cancel", "shutdown"})
SAFE_MESSAGES = {
    "cancelled": "Local worker cancelled.",
    "generation_failed": "Local worker generation failed.",
    "input_limit_exceeded": "Local worker input exceeds supported limits.",
    "invalid_request": "Local worker request was rejected.",
    "protocol_error": "Local worker protocol failed.",
    "resource_exhausted": "Local worker resources were exhausted.",
    "runtime_failed": "Local worker runtime failed.",
    "unavailable": "Local worker is unavailable.",
    "worker_failed": "Local worker failed.",
}
Role = Literal["ocr", "extraction", "summary"]


class ProtocolError(ValueError):
    """A malformed protocol frame."""


@dataclass(frozen=True)
class Request:
    """One strictly validated worker request."""

    request_id: str
    job_id: str
    command: str
    payload: dict[str, object]


def _parse_line(line: bytes) -> Request:
    if not line or not line.endswith(b"\n") or len(line) > MAX_MESSAGE_BYTES + 1:
        raise ProtocolError

    def reject_constant(_value: str) -> None:
        raise ValueError

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        raw = json.loads(
            line[:-1],
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError):
        raise ProtocolError from None
    if not isinstance(raw, dict) or set(raw) != {
        "version",
        "request_id",
        "job_id",
        "command",
        "payload",
    }:
        raise ProtocolError
    request_id = raw.get("request_id")
    job_id = raw.get("job_id")
    command = raw.get("command")
    payload = raw.get("payload")
    if (
        raw.get("version") != PROTOCOL_VERSION
        or isinstance(raw.get("version"), bool)
        or not isinstance(request_id, str)
        or IDENTIFIER.fullmatch(request_id) is None
        or not isinstance(job_id, str)
        or IDENTIFIER.fullmatch(job_id) is None
        or not isinstance(command, str)
        or command not in COMMANDS
        or not isinstance(payload, dict)
    ):
        raise ProtocolError
    return Request(
        request_id=request_id,
        job_id=job_id,
        command=command,
        payload=payload,
    )


def _write(request_id: str, kind: str, payload: dict[str, object]) -> None:
    response = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "kind": kind,
        "payload": payload,
    }
    encoded = json.dumps(
        response,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolError
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()


def _ready(request_id: str, role: Role | None) -> None:
    _write(request_id, "ready", {"role": role})


def _result(request_id: str, data: object) -> None:
    _write(request_id, "result", {"data": data})


def _progress(request_id: str, payload: dict[str, object]) -> None:
    _write(request_id, "progress", payload)


def _write_to_descriptor(
    descriptor: int,
    request_id: str,
    kind: str,
    payload: dict[str, object],
) -> None:
    """Write one bounded protocol frame through a preserved protocol descriptor."""

    response = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "kind": kind,
        "payload": payload,
    }
    encoded = json.dumps(
        response,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolError
    remaining = memoryview(encoded + b"\n")
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise ProtocolError
        remaining = remaining[written:]


def _error(
    request_id: str,
    code: str,
    *,
    category: str | None = None,
) -> None:
    if category is not None and (
        code != "generation_failed" or category not in GENERATION_FAILURE_CATEGORIES
    ):
        raise ProtocolError
    payload = {"code": code, "message": SAFE_MESSAGES[code]}
    if category is not None:
        payload["category"] = category
    _write(request_id, "error", payload)
    sys.stderr.write(f"local_ai_worker event=terminal_error code={code}\n")
    sys.stderr.flush()


def _role(command: str) -> Role:
    if command == "ocr":
        return "ocr"
    if command == "extract":
        return "extraction"
    return "summary"


def _safe_runtime_error_code(exc: Exception) -> str:
    """Reduce runtime failures to a fixed code without retaining exception text."""

    if isinstance(exc, GenerationError):
        return "generation_failed"
    if isinstance(exc, MemoryError):
        return "resource_exhausted"
    if isinstance(exc, RuntimeError):
        return "runtime_failed"
    return "worker_failed"


def _safe_runtime_error_category(exc: Exception) -> str | None:
    """Return only an allowlisted content-free generation failure category."""

    if not isinstance(exc, GenerationError):
        return None
    category = exc.category
    if category not in GENERATION_FAILURE_CATEGORIES:
        return None
    return category


def _dispatch(
    request: Request,
    *,
    extraction_progress: Callable[[int, int], None] | None = None,
    extraction_heartbeat: Callable[[], None] | None = None,
    extraction_lifecycle: Callable[[], None] | None = None,
) -> object:
    if request.command == "ocr":
        return run_ocr(request.payload)
    if request.command == "extract":
        return run_extraction(
            request.payload,
            progress_fn=extraction_progress,
            attempt_progress_fn=extraction_heartbeat,
            lifecycle_progress_fn=extraction_lifecycle,
        )
    if request.command == "summarize":
        return run_summary(request.payload)
    raise WorkerInputError("Local worker command is invalid.")


def _quiet_dispatch(request: Request) -> object:
    """Suppress all model/runtime output while retaining only fixed worker events."""

    sys.stdout.flush()
    sys.stderr.flush()
    stdout_copy = os.dup(1)
    stderr_copy = os.dup(2)
    try:
        activity = 0
        page_current = 0
        page_total = 0
        processing_started = False

        def publish_extraction_progress(stage: str, current: int, total: int) -> None:
            nonlocal activity
            if (
                type(current) is not int
                or type(total) is not int
                or current < 0
                or total < current
                or total > MAX_EXTRACTION_ACTIVITY
                or activity >= MAX_EXTRACTION_ACTIVITY
            ):
                raise WorkerInputError("Local worker progress is invalid.")
            activity += 1
            _write_to_descriptor(
                stdout_copy,
                request.request_id,
                "progress",
                {
                    "role": "extraction",
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "activity": activity,
                },
            )

        def publish_page_progress(current: int, total: int) -> None:
            nonlocal page_current, page_total, processing_started
            if (
                type(current) is not int
                or type(total) is not int
                or current < page_current
                or total < current
                or (processing_started and total != page_total)
            ):
                raise WorkerInputError("Local worker progress is invalid.")
            page_current = current
            page_total = total
            processing_started = True
            publish_extraction_progress("processing", page_current, page_total)

        def publish_activity() -> None:
            stage = "processing" if processing_started else "loading"
            publish_extraction_progress(stage, page_current, page_total)

        dispatch_options: dict[str, object] = {}
        if request.command == "extract":
            dispatch_options = {
                "extraction_progress": publish_page_progress,
                "extraction_heartbeat": publish_activity,
                "extraction_lifecycle": publish_activity,
            }
        with (
            open(os.devnull, "w", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
        ):
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            return _dispatch(request, **dispatch_options)
    finally:
        os.dup2(stdout_copy, 1)
        os.dup2(stderr_copy, 2)
        os.close(stdout_copy)
        os.close(stderr_copy)


def _reset_mlx_peak_memory() -> None:
    """Reset the role-local MLX peak counter without affecting inference."""

    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        return


def _mlx_memory_counters() -> tuple[int, int] | None:
    """Return non-content MLX allocator counters when the runtime exposes them."""

    try:
        import mlx.core as mx

        active = int(mx.get_active_memory())
        peak = int(mx.get_peak_memory())
    except Exception:
        return None
    if active < 0 or peak < 0 or active > 2**63 - 1 or peak > 2**63 - 1:
        return None
    return active, peak


def _memory_progress_payload(role: Role) -> dict[str, object] | None:
    counters = _mlx_memory_counters()
    if counters is None:
        return None
    active, peak = counters
    return {
        "role": role,
        "stage": "finalizing",
        "current": 1,
        "total": 1,
        "active_memory_bytes": active,
        "peak_memory_bytes": peak,
    }


def _wait_for_shutdown() -> bool:
    line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
    if not line:
        return True
    try:
        request = _parse_line(line)
    except ProtocolError:
        return False
    return request.command == "shutdown"


def _start_parent_watchdog() -> None:
    parent_value = os.environ.get("LOCAL_AI_PARENT_PID")
    lock_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_FD")
    device_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_DEVICE")
    inode_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_INODE")
    if all(value is None for value in (parent_value, lock_value, device_value, inode_value)):
        return
    try:
        parent_pid = int(parent_value)
        lock_descriptor = int(lock_value)
        expected_device = int(device_value)
        expected_inode = int(inode_value)
        lock_metadata = os.fstat(lock_descriptor)
    except (OSError, TypeError, ValueError):
        os._exit(126)
    worker_pid = os.getpid()
    if (
        parent_pid != os.getppid()
        or parent_pid <= 1
        or lock_descriptor < 3
        or not stat.S_ISREG(lock_metadata.st_mode)
        or stat.S_IMODE(lock_metadata.st_mode) != 0o600
        or lock_metadata.st_nlink != 1
        or (hasattr(os, "getuid") and lock_metadata.st_uid != os.getuid())
        or expected_device < 0
        or expected_inode <= 0
        or lock_metadata.st_dev != expected_device
        or lock_metadata.st_ino != expected_inode
    ):
        os._exit(126)
    read_descriptor, write_descriptor = os.pipe()
    try:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_ai_mlx_worker.parent_watchdog",
                str(parent_pid),
                str(worker_pid),
                str(os.getpgrp()),
                str(write_descriptor),
                str(lock_descriptor),
                str(expected_device),
                str(expected_inode),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_descriptor, lock_descriptor),
            start_new_session=True,
        )
    except OSError:
        os.close(read_descriptor)
        os.close(write_descriptor)
        os._exit(126)
    os.close(write_descriptor)
    try:
        readable, _, _ = select.select([read_descriptor], [], [], 5)
        ready = os.read(read_descriptor, 1) if readable else b""
    finally:
        os.close(read_descriptor)
    if ready != b"1":
        watchdog.kill()
        watchdog.wait()
        os._exit(126)


def main() -> int:
    """Read one request, emit strict frames, and exit after shutdown."""

    _start_parent_watchdog()
    line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
    try:
        request = _parse_line(line)
    except ProtocolError:
        return 2
    if request.command == "shutdown":
        return 0
    if request.command == "health":
        _ready(request.request_id, None)
        _result(
            request.request_id,
            {
                "status": "ready",
                "runtime": RUNTIME,
            },
        )
        return 0 if _wait_for_shutdown() else 2
    if request.command == "cancel":
        _ready(request.request_id, None)
        _error(request.request_id, "cancelled")
        return 0 if _wait_for_shutdown() else 2

    role = _role(request.command)
    _ready(request.request_id, role)
    try:
        _reset_mlx_peak_memory()
        data = _quiet_dispatch(request)
    except WorkerInputLimitError:
        _error(request.request_id, "input_limit_exceeded")
    except WorkerInputError:
        _error(request.request_id, "invalid_request")
    except ArtifactUnavailableError:
        _error(request.request_id, "unavailable")
    except Exception as exc:
        _error(
            request.request_id,
            _safe_runtime_error_code(exc),
            category=_safe_runtime_error_category(exc),
        )
    else:
        try:
            memory_progress = _memory_progress_payload(role)
            if memory_progress is not None:
                _progress(request.request_id, memory_progress)
            _result(request.request_id, data)
        except (ProtocolError, TypeError, ValueError, RecursionError):
            _error(request.request_id, "worker_failed")
    return 0 if _wait_for_shutdown() else 2


if __name__ == "__main__":
    raise SystemExit(main())
