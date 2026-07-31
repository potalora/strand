"""Deterministic protocol worker used by public tests without model dependencies."""

from __future__ import annotations

import json
import os
import select
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
)
from app.services.local_ai.grounded_summary import GroundedSummaryDocument
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

_PIPELINE_VALID_FLAG = "--pipeline-valid"
_PIPELINE_VALID_EVIDENCE = "Fixturemed 10 mg daily is active."
_PIPELINE_VALID_OCR_MARKDOWN = f"# Synthetic OCR\n\n{_PIPELINE_VALID_EVIDENCE}"


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


def _pipeline_valid_summary(payload: dict[str, object]) -> dict[str, object]:
    """Select only supplied fact/evidence references for E2E pipeline tests."""

    raw_facts = payload.get("facts")
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_facts, list) or not isinstance(raw_evidence, list):
        return {"sections": [], "uncertainties": []}

    evidence_by_id = {
        evidence["evidence_id"]: evidence
        for evidence in raw_evidence
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_id"), str)
    }
    claims: list[dict[str, object]] = []
    for fact in raw_facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("fact_id"), str):
            continue
        fields = fact.get("fields")
        linked_evidence = fact.get("evidence_ids")
        if not isinstance(fields, list) or not isinstance(linked_evidence, list):
            continue
        fact_paths = [
            field["path"]
            for field in fields
            if isinstance(field, dict) and isinstance(field.get("path"), str)
        ]
        for evidence_id in linked_evidence:
            if not isinstance(evidence_id, str):
                continue
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            evidence_paths = evidence.get("field_paths")
            if not isinstance(evidence_paths, list):
                continue
            supported_paths = [
                path
                for path in fact_paths
                if path in evidence_paths and isinstance(path, str)
            ]
            if not supported_paths:
                continue
            claims.append(
                {
                    "fact_id": fact["fact_id"],
                    "field_paths": supported_paths,
                    "evidence_ids": [evidence_id],
                }
            )
            break

    sections = [{"heading": "Overview", "claims": claims}] if claims else []
    return {"sections": sections, "uncertainties": []}


def _pipeline_valid_summary_token_count(
    payload: dict[str, object],
) -> dict[str, object]:
    """Validate and conservatively estimate tokens without model dependencies."""

    reference = GroundedSummaryDocument.model_validate(
        payload.get("reference_document")
    )
    compact = json.dumps(
        reference.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    byte_count = len(compact.encode("utf-8"))
    return {"token_count": (byte_count + 1) // 2}


def _pipeline_valid_extraction(payload: dict[str, object]) -> dict[str, object]:
    """Return one grounded fictional fact only for synthetic OCR pages."""

    result: dict[str, object] = {
        "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
        "patient": None,
        **{category: [] for category in FACT_CATEGORY_NAMES},
        "unresolved_fields": [],
        "rejected_fields": [],
    }
    raw_pages = payload.get("page_markdown")
    if not isinstance(raw_pages, list):
        return result
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page_number = raw_page.get("page_number")
        markdown = raw_page.get("markdown")
        if (
            type(page_number) is not int
            or page_number < 1
            or not isinstance(markdown, str)
            or _PIPELINE_VALID_EVIDENCE not in markdown
        ):
            continue
        result["medications"] = [
            {
                "fact_id": "synthetic-medication-1",
                "name": "Fixturemed",
                "dose_value": "10",
                "dose_unit": "mg",
                "frequency": "daily",
                "status": "active",
                "verbatim": _PIPELINE_VALID_EVIDENCE,
                "page_number": page_number,
                "evidence_excerpt": _PIPELINE_VALID_EVIDENCE,
            }
        ]
        break
    return result


def _pipeline_valid_result(
    role: ModelRole,
    payload: dict[str, object],
) -> dict[str, object]:
    """Return validator-compatible deterministic pipeline output."""

    if role is ModelRole.OCR:
        return {
            "markdown": _PIPELINE_VALID_OCR_MARKDOWN,
            "page_number": payload.get("page_number"),
        }
    if role is ModelRole.EXTRACTION:
        return _pipeline_valid_extraction(payload)
    return _pipeline_valid_summary(payload)


def _fixed_result(
    role: ModelRole,
    payload: dict[str, object],
    *,
    command: str | None = None,
    pipeline_valid: bool = False,
) -> dict[str, object]:
    if payload.get("fake_token_count") is not None:
        return {"token_count": payload["fake_token_count"]}
    if payload.get("inspect_lock_fd") is True:
        raw_descriptor = os.environ.get("LOCAL_AI_PROCESS_LOCK_FD", "")
        try:
            metadata = os.fstat(int(raw_descriptor))
            expected_device = int(os.environ.get("LOCAL_AI_PROCESS_LOCK_DEVICE", ""))
            expected_inode = int(os.environ.get("LOCAL_AI_PROCESS_LOCK_INODE", ""))
        except (OSError, ValueError):
            return {
                "lock_fd_inherited": False,
                "lock_identity_matches": False,
                "lock_path_exposed": "LOCAL_AI_PROCESS_LOCK_PATH" in os.environ,
            }
        return {
            "lock_fd_inherited": stat.S_ISREG(metadata.st_mode),
            "lock_identity_matches": (
                metadata.st_dev == expected_device and metadata.st_ino == expected_inode
            ),
            "lock_path_exposed": "LOCAL_AI_PROCESS_LOCK_PATH" in os.environ,
        }
    unix_socket_path = payload.get("probe_unix_socket_path")
    if payload.get("probe_network") is True or isinstance(unix_socket_path, str):
        if isinstance(unix_socket_path, str):
            probe = socket.socket(socket.AF_UNIX)
            address: tuple[str, int] | str = unix_socket_path
        else:
            probe = socket.socket()
            address = ("127.0.0.1", 9)
        probe.settimeout(0.2)
        try:
            error_number = probe.connect_ex(address)
        finally:
            probe.close()
        return {
            "network_denied": error_number in {1, 13},
            "network_errno": error_number,
        }
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
    if pipeline_valid and command == "count_summary_tokens":
        return _pipeline_valid_summary_token_count(payload)
    if pipeline_valid:
        return _pipeline_valid_result(role, payload)
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


def _start_parent_watchdog() -> None:
    parent_value = os.environ.get("LOCAL_AI_PARENT_PID")
    lock_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_FD")
    device_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_DEVICE")
    inode_value = os.environ.get("LOCAL_AI_PROCESS_LOCK_INODE")
    if all(
        value is None for value in (parent_value, lock_value, device_value, inode_value)
    ):
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
    backend_root = Path(__file__).resolve().parents[3]
    bootstrap = (
        "import sys;"
        f"sys.path.insert(0,{str(backend_root)!r});"
        "from app.services.local_ai.parent_watchdog import main;"
        "raise SystemExit(main())"
    )
    read_descriptor, write_descriptor = os.pipe()
    try:
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-c",
                bootstrap,
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


def _main(*, pipeline_valid: bool = False) -> int:
    _start_parent_watchdog()
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
    response_id = (
        "mismatched-request"
        if payload.get("mismatched_id") is True
        else request.request_id
    )

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

    progress_steps = payload.get("progress_steps")
    progress_delay_ms = payload.get("progress_delay_ms", 0)
    if (
        not isinstance(progress_delay_ms, int)
        or isinstance(progress_delay_ms, bool)
        or progress_delay_ms < 0
    ):
        progress_delay_ms = 0
    if (
        isinstance(progress_steps, int)
        and not isinstance(progress_steps, bool)
        and 0 < progress_steps <= 1_000
    ):
        progress_stage = "generating" if role is ModelRole.SUMMARY else "processing"
        _write(
            _response(
                response_id,
                "progress",
                ProgressPayload(
                    role=role,
                    stage=progress_stage,
                    current=0,
                    total=progress_steps,
                ),
            )
        )
        for current in range(1, progress_steps + 1):
            time.sleep(min(progress_delay_ms, 60_000) / 1000)
            _write(
                _response(
                    response_id,
                    "progress",
                    ProgressPayload(
                        role=role,
                        stage=progress_stage,
                        current=current,
                        total=progress_steps,
                    ),
                )
            )
    elif payload.get("repeat_progress") is True:
        progress_stage = "generating" if role is ModelRole.SUMMARY else "processing"
        repeated = _response(
            response_id,
            "progress",
            ProgressPayload(role=role, stage=progress_stage, current=0, total=1),
        )
        _write(repeated)
        while True:
            time.sleep(min(progress_delay_ms, 60_000) / 1000)
            _write(repeated)
    else:
        progress_stage = "generating" if role is ModelRole.SUMMARY else "processing"
        _write(
            _response(
                response_id,
                "progress",
                ProgressPayload(role=role, stage=progress_stage, current=0, total=1),
            )
        )
    guard_path_value = payload.get("concurrency_guard_path")
    guard_path = (
        Path(guard_path_value)
        if isinstance(guard_path_value, str) and guard_path_value
        else None
    )
    guard_fd: int | None = None
    if guard_path is not None:
        try:
            guard_fd = os.open(
                guard_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            payload["_overlap_detected"] = False
        except FileExistsError:
            payload["_overlap_detected"] = True
    started_marker_value = payload.get("started_marker_path")
    if isinstance(started_marker_value, str) and started_marker_value:
        started_marker = Path(started_marker_value)
        started_marker.write_text("started")
        os.chmod(started_marker, 0o600)
    delay_ms = payload.get("delay_ms", 0)
    if isinstance(delay_ms, int) and not isinstance(delay_ms, bool) and delay_ms > 0:
        time.sleep(min(delay_ms, 60_000) / 1000)
    if payload.get("block") is True:
        _wait_for_shutdown()
        return 0

    if payload.get("input_limit_error") is True:
        terminal = _response(
            response_id,
            "error",
            ErrorPayload(
                code="input_limit_exceeded",
                message="Local worker input exceeds supported limits.",
            ),
        )
    elif payload.get("safe_error") is True:
        terminal = _response(
            response_id,
            "error",
            ErrorPayload(code="worker_failed", message="Local worker failed."),
        )
    else:
        if "_overlap_detected" in payload:
            terminal_data: dict[str, object] = {
                "overlap_detected": payload["_overlap_detected"],
            }
        else:
            terminal_data = _fixed_result(
                role,
                payload,
                command=request.command,
                pipeline_valid=pipeline_valid,
            )
        terminal = _response(
            response_id,
            "result",
            ResultPayload(data=terminal_data),
        )
    _write(terminal)

    if guard_fd is not None:
        os.close(guard_fd)
        try:
            guard_path.unlink()
        except FileNotFoundError:
            pass

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fake worker, optionally emitting pipeline-validator-compatible output."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return _main()
    if arguments == [_PIPELINE_VALID_FLAG]:
        return _main(pipeline_valid=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
