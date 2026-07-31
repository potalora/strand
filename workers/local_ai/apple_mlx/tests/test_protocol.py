from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

PROTOCOL_VERSION = 1
SAFETY_RULES = [
    "Select only server-supplied fact fields and their linked evidence.",
    "Do not emit free-text clinical claims, headings, or uncertainties.",
    "Do not create, correct, infer, or repair clinical facts.",
    "Do not provide diagnoses, treatment recommendations, medical advice, "
    "or clinical decision support.",
]


class _Tokenizer:
    model_max_length = 65_536

    def encode(self, value: str, **_kwargs: object) -> list[str]:
        return list(value)


class _Processor:
    tokenizer = _Tokenizer()


class _Config:
    max_position_embeddings = 65_536


class _Model:
    config = _Config()


def _loaded(role: str, *, max_input_tokens: int = 32_768) -> object:
    from local_ai_mlx_worker.common import LoadedRole

    return LoadedRole(
        role=role,  # type: ignore[arg-type]
        model=_Model(),
        processor=_Processor(),
        model_path=f"/verified-pack/{role}",
        decode_limits={
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": 8192 if role == "extraction" else 4096,
        },
        repository_files_used=frozenset({"config.json", "model.safetensors"}),
    )


def _png(scratch: Path, name: str = "page.png", *, size: tuple[int, int] = (10, 10)) -> Path:
    path = scratch / name
    Image.new("RGB", size, color="white").save(path, format="PNG")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_payload() -> dict[str, object]:
    content_json = json.dumps(
        {"name": "Metformin", "record_type": "medication"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_snapshot = json.dumps(
        {
            "excerpt": "Metformin is listed as active.",
            "field_paths": ["/name"],
            "page_number": 1,
            "section": "Medications",
            "source_id": "source-evidence-1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    evidence_id = "evidence1_" + hashlib.sha256(evidence_snapshot.encode()).hexdigest()[:40]
    fact_snapshot = json.dumps(
        {
            "content_json": content_json,
            "evidence_ids": [evidence_id],
            "record_id": "record-1",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fact_id = "fact1_" + hashlib.sha256(fact_snapshot.encode()).hexdigest()[:40]
    uncertainty_snapshot = json.dumps(
        {
            "evidence_ids": [evidence_id],
            "fact_ids": [fact_id],
            "template_id": "medication_end_date_missing",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    uncertainty_id = (
        "uncertainty1_" + hashlib.sha256(uncertainty_snapshot.encode()).hexdigest()[:40]
    )
    return {
        "requested_scope": {
            "summary_type": "full_health",
            "category": None,
            "date_from": None,
            "date_to": None,
            "record_ids": [],
        },
        "facts": [
            {
                "fact_id": fact_id,
                "record_id": "record-1",
                "content_json": content_json,
                "fields": [
                    {"path": "/name", "value_json": '"Metformin"'},
                    {"path": "/record_type", "value_json": '"medication"'},
                ],
                "evidence_ids": [evidence_id],
            }
        ],
        "evidence": [
            {
                "evidence_id": evidence_id,
                "source_id": "source-evidence-1",
                "excerpt": "Metformin is listed as active.",
                "page_number": 1,
                "section": "Medications",
                "fact_ids": [fact_id],
                "field_paths": ["/name"],
            }
        ],
        "uncertainty_labels": [
            {
                "uncertainty_id": uncertainty_id,
                "template_id": "medication_end_date_missing",
                "label": "Medication end date is not available.",
                "fact_ids": [fact_id],
                "evidence_ids": [evidence_id],
            }
        ],
        "safety_rules": SAFETY_RULES,
    }


def _maximal_summary_reference() -> dict[str, object]:
    payload = _summary_payload()
    fact = payload["facts"][0]  # type: ignore[index]
    evidence = payload["evidence"][0]  # type: ignore[index]
    uncertainty = payload["uncertainty_labels"][0]  # type: ignore[index]
    return {
        "sections": [
            {
                "heading": "Medications",
                "claims": [
                    {
                        "fact_id": fact["fact_id"],
                        "field_paths": ["/name"],
                        "evidence_ids": [evidence["evidence_id"]],
                    }
                ],
            }
        ],
        "uncertainties": [
            {
                "uncertainty_id": uncertainty["uncertainty_id"],
                "fact_ids": uncertainty["fact_ids"],
                "evidence_ids": uncertainty["evidence_ids"],
            }
        ],
    }


def test_summary_reference_token_count_uses_processor_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import qwen_summary

    model_loader = Mock(side_effect=AssertionError("model weights loaded"))
    processor_loader = Mock(return_value=_Processor())
    monkeypatch.setattr(qwen_summary, "load_role_from_payload", model_loader)
    reference = _maximal_summary_reference()

    result = qwen_summary.count_summary_reference_tokens(
        {"reference_document": reference},
        processor_loader=processor_loader,
    )

    compact = json.dumps(
        reference,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result == {"token_count": len(compact)}
    processor_loader.assert_called_once()
    model_loader.assert_not_called()


def test_malformed_summary_reference_never_loads_tokenizer() -> None:
    from local_ai_mlx_worker.qwen_summary import count_summary_reference_tokens

    processor_loader = Mock(side_effect=AssertionError("tokenizer loaded"))
    reference = _maximal_summary_reference()
    del reference["sections"][0]["claims"][0]["field_paths"]  # type: ignore[index]

    with pytest.raises(Exception, match="invalid"):
        count_summary_reference_tokens(
            {"reference_document": reference},
            processor_loader=processor_loader,
        )

    processor_loader.assert_not_called()


def _replace_summary_fact_content(
    payload: dict[str, object],
    content: dict[str, object],
) -> None:
    content_json = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fact = payload["facts"][0]  # type: ignore[index]
    old_fact_id = fact["fact_id"]
    fact_snapshot = json.dumps(
        {
            "content_json": content_json,
            "evidence_ids": sorted(fact["evidence_ids"]),
            "record_id": fact["record_id"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fact_id = "fact1_" + hashlib.sha256(fact_snapshot.encode()).hexdigest()[:40]
    fact["content_json"] = content_json
    fact["fact_id"] = fact_id
    fields: list[dict[str, str]] = []

    def visit(value: object, path: str) -> None:
        if type(value) is dict and value:
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        fields.append(
            {
                "path": path,
                "value_json": json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    for key, value in sorted(content.items()):
        visit(value, f"/{key}")
    fact["fields"] = fields
    evidence = payload["evidence"][0]  # type: ignore[index]
    evidence["fact_ids"] = [fact_id]
    uncertainty = payload["uncertainty_labels"][0]  # type: ignore[index]
    uncertainty["fact_ids"] = [fact_id]
    uncertainty_snapshot = json.dumps(
        {
            "evidence_ids": uncertainty["evidence_ids"],
            "fact_ids": [fact_id],
            "template_id": uncertainty["template_id"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    uncertainty["uncertainty_id"] = (
        "uncertainty1_" + hashlib.sha256(uncertainty_snapshot.encode()).hexdigest()[:40]
    )
    assert old_fact_id != fact_id


@pytest.fixture
def worker_process() -> Iterator[subprocess.Popen[str]]:
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "local_ai_mlx_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _send(
    process: subprocess.Popen[str],
    *,
    command: str,
    payload: dict[str, object],
    request_id: str = "test-1",
) -> None:
    assert process.stdin is not None
    process.stdin.write(
        json.dumps(
            {
                "version": PROTOCOL_VERSION,
                "request_id": request_id,
                "job_id": "job-1",
                "command": command,
                "payload": payload,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    process.stdin.flush()


def _read(process: subprocess.Popen[str]) -> dict[str, object]:
    assert process.stdout is not None
    line = process.stdout.readline()
    assert line
    return json.loads(line)


def test_worker_stdout_is_protocol_only(worker_process: subprocess.Popen[str]) -> None:
    _send(worker_process, command="health", payload={})

    assert _read(worker_process) == {
        "version": 1,
        "request_id": "test-1",
        "kind": "ready",
        "payload": {"role": None},
    }
    assert _read(worker_process) == {
        "version": 1,
        "request_id": "test-1",
        "kind": "result",
        "payload": {
            "data": {
                "status": "ready",
                "runtime": "mlx-vlm-0.5.0",
            }
        },
    }

    _send(worker_process, command="shutdown", payload={})
    assert worker_process.stdin is not None
    worker_process.stdin.close()
    assert worker_process.wait(timeout=5) == 0
    assert worker_process.stdout is not None
    assert worker_process.stdout.read() == ""


def test_invalid_role_payload_maps_to_fixed_non_content_error(
    worker_process: subprocess.Popen[str],
) -> None:
    sentinel = "patient-content-must-not-be-logged"
    _send(
        worker_process,
        command="ocr",
        payload={"prompt": sentinel},
    )

    assert _read(worker_process) == {
        "version": 1,
        "request_id": "test-1",
        "kind": "ready",
        "payload": {"role": "ocr"},
    }
    assert _read(worker_process) == {
        "version": 1,
        "request_id": "test-1",
        "kind": "error",
        "payload": {
            "code": "invalid_request",
            "message": "Local worker request was rejected.",
        },
    }

    _send(worker_process, command="shutdown", payload={})
    assert worker_process.stdin is not None
    worker_process.stdin.close()
    assert worker_process.wait(timeout=5) == 0
    assert worker_process.stderr is not None
    assert sentinel not in worker_process.stderr.read()


def test_runtime_failures_map_to_fixed_non_content_codes() -> None:
    from local_ai_mlx_worker import __main__ as worker_main
    from local_ai_mlx_worker.common import GenerationError

    assert worker_main._safe_runtime_error_code(GenerationError("private")) == ("generation_failed")
    assert worker_main._safe_runtime_error_code(MemoryError("private")) == ("resource_exhausted")
    assert worker_main._safe_runtime_error_code(RuntimeError("private")) == ("runtime_failed")
    assert worker_main._safe_runtime_error_code(Exception("private")) == ("worker_failed")


@pytest.mark.parametrize("category", ["output_limit", "invalid_structured_output"])
def test_generation_failure_category_survives_worker_terminal_frame(
    category: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from local_ai_mlx_worker import __main__ as worker_main
    from local_ai_mlx_worker.common import GenerationError

    sentinel = "clinical-content-must-not-cross-worker-boundary"
    emitted: list[tuple[str, str, dict[str, object]]] = []
    error = GenerationError(sentinel, category=category)

    monkeypatch.setattr(
        worker_main,
        "_write",
        lambda request_id, kind, payload: emitted.append((request_id, kind, payload)),
    )

    worker_main._error(
        "request-1",
        worker_main._safe_runtime_error_code(error),
        category=worker_main._safe_runtime_error_category(error),
    )

    assert emitted == [
        (
            "request-1",
            "error",
            {
                "code": "generation_failed",
                "message": "Local worker generation failed.",
                "category": category,
            },
        )
    ]
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_worker_terminal_rejects_arbitrary_generation_failure_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import __main__ as worker_main

    monkeypatch.setattr(worker_main, "_write", lambda *_args, **_kwargs: None)

    with pytest.raises(worker_main.ProtocolError):
        worker_main._error(
            "request-1",
            "generation_failed",
            category="patient-content",
        )


def test_role_runtime_output_is_suppressed_from_protocol_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from local_ai_mlx_worker import __main__ as worker_main

    sentinel = "clinical-content-must-not-leak"

    def noisy_dispatch(_request: object) -> dict[str, object]:
        print(sentinel)
        print(sentinel, file=sys.stderr)
        os.write(1, f"{sentinel}\n".encode())
        os.write(2, f"{sentinel}\n".encode())
        return {"ok": True}

    monkeypatch.setattr(worker_main, "_dispatch", noisy_dispatch)
    request = worker_main.Request(
        request_id="request-1",
        job_id="job-1",
        command="summarize",
        payload={},
    )

    assert worker_main._quiet_dispatch(request) == {"ok": True}
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_quiet_extraction_dispatch_preserves_page_and_budget_counters_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress frames distinguish completed pages from content-free activity."""

    from local_ai_mlx_worker import __main__ as worker_main

    emitted: list[dict[str, object]] = []
    sentinel = "clinical-content-must-not-enter-progress"

    def extraction_dispatch(
        _request: object,
        *,
        extraction_progress: object = None,
        extraction_heartbeat: object = None,
        extraction_lifecycle: object = None,
        extraction_budget_progress: object = None,
    ) -> dict[str, object]:
        assert callable(extraction_lifecycle)
        assert callable(extraction_progress)
        assert callable(extraction_heartbeat)
        assert callable(extraction_budget_progress)
        extraction_lifecycle()
        extraction_progress(0, 3)
        extraction_budget_progress(
            {
                "attempt": 1,
                "attempt_limit": 12,
                "output_tokens": 17,
                "output_token_limit": 16_384,
                "splits_used": 0,
                "split_limit": 7,
            }
        )
        extraction_heartbeat()
        extraction_progress(1, 3)
        return {"ok": True, "private": sentinel}

    monkeypatch.setattr(worker_main, "_dispatch", extraction_dispatch)
    monkeypatch.setattr(
        worker_main,
        "_write_to_descriptor",
        lambda _descriptor, _request_id, kind, payload: emitted.append(
            {"kind": kind, "payload": payload}
        ),
    )
    request = worker_main.Request(
        request_id="request-1",
        job_id="job-1",
        command="extract",
        payload={},
    )

    assert worker_main._quiet_dispatch(request) == {"ok": True, "private": sentinel}
    assert emitted == [
        {
            "kind": "progress",
            "payload": {
                "role": "extraction",
                "stage": "loading",
                "current": 0,
                "total": 0,
                "activity": 1,
            },
        },
        {
            "kind": "progress",
            "payload": {
                "role": "extraction",
                "stage": "processing",
                "current": 0,
                "total": 3,
                "activity": 2,
            },
        },
        {
            "kind": "progress",
            "payload": {
                "role": "extraction",
                "stage": "processing",
                "current": 0,
                "total": 3,
                "activity": 3,
                "attempt": 1,
                "attempt_limit": 12,
                "output_tokens": 17,
                "output_token_limit": 16_384,
                "splits_used": 0,
                "split_limit": 7,
            },
        },
        {
            "kind": "progress",
            "payload": {
                "role": "extraction",
                "stage": "processing",
                "current": 0,
                "total": 3,
                "activity": 4,
                "attempt": 1,
                "attempt_limit": 12,
                "output_tokens": 17,
                "output_token_limit": 16_384,
                "splits_used": 0,
                "split_limit": 7,
            },
        },
        {
            "kind": "progress",
            "payload": {
                "role": "extraction",
                "stage": "processing",
                "current": 1,
                "total": 3,
                "activity": 5,
                "attempt": 1,
                "attempt_limit": 12,
                "output_tokens": 17,
                "output_token_limit": 16_384,
                "splits_used": 0,
                "split_limit": 7,
            },
        },
    ]
    assert sentinel not in json.dumps(emitted)


def test_memory_progress_payload_contains_only_bounded_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import __main__ as worker_main

    monkeypatch.setattr(
        worker_main,
        "_mlx_memory_counters",
        lambda: (512 * 1024**2, 6 * 1024**3),
    )

    assert worker_main._memory_progress_payload("summary") == {
        "role": "summary",
        "stage": "finalizing",
        "current": 1,
        "total": 1,
        "active_memory_bytes": 512 * 1024**2,
        "peak_memory_bytes": 6 * 1024**3,
    }


def test_ocr_uses_fixed_greedy_decode_options(tmp_path: Path) -> None:
    from local_ai_mlx_worker.ovisocr2 import run_ocr

    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return "faithful markdown"

    loaded = _loaded("ocr")
    image = _png(tmp_path, "page-2.png")
    result = run_ocr(
        {
            "page_number": 2,
            "scratch_dir": str(tmp_path),
            "image_path": str(image),
            "image_sha256": _sha256(image),
            "max_output_tokens": 9000,
        },
        loaded=loaded,  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert result == {"markdown": "faithful markdown", "page_number": 2}
    assert calls == [
        {
            "model": loaded.model,
            "processor": loaded.processor,
            "prompt": (
                "Extract this page faithfully as Markdown. "
                "Preserve reading order, tables, and formulas."
            ),
            "images": [str(image)],
            "max_tokens": 4096,
            "temperature": 0.0,
            "do_sample": False,
            "input_token_limit": 32768,
        }
    ]


def test_extraction_retries_duplicate_json_keys_once_with_non_thinking_template(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    outputs = iter(
        [
            '{"schema_version":"bad","schema_version":"duplicate"}',
            '{"schema_version":"clinical-document-extraction.v1"}',
        ]
    )
    calls: list[dict[str, object]] = []
    attempt_progress: list[int] = []
    lifecycle_progress: list[int] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return next(outputs)

    loaded = _loaded("extraction")
    image = _png(tmp_path, "selected-page.png")
    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
            "scratch_dir": str(tmp_path),
            "image_paths": {"1": str(image)},
            "schema": {
                "type": "object",
                "required": ["schema_version"],
            },
        },
        loaded=loaded,  # type: ignore[arg-type]
        generate_fn=generate,
        attempt_progress_fn=lambda: attempt_progress.append(1),
        lifecycle_progress_fn=lambda: lifecycle_progress.append(1),
    )

    assert result == {"schema_version": "clinical-document-extraction.v1"}
    assert len(calls) == 2
    assert attempt_progress == [1, 1]
    assert lifecycle_progress == [1] * 4
    assert all(call["temperature"] == 0.0 for call in calls)
    assert all(call["do_sample"] is False for call in calls)
    assert all(call["max_tokens"] == 4096 for call in calls)
    assert all(call["enable_thinking"] is False for call in calls)
    assert all(call["mode"] == "structured" for call in calls)
    assert all(
        json.loads(str(call["template"]))
        == {
            "required": ["schema_version"],
            "type": "object",
        }
        for call in calls
    )
    assert all(call["images"] == [str(image)] for call in calls)
    assert all(call["input_token_limit"] == 4_096 for call in calls)
    assert all(
        "Never use JSON null for enum-valued fields." in str(call["instructions"]) for call in calls
    )
    assert all("Use null, empty lists" not in str(call["instructions"]) for call in calls)
    assert all(
        "Set assertion to negated for explicit no, denies, absent, or negative evidence."
        in str(call["instructions"])
        for call in calls
    )
    assert all(
        "Set assertion to family_history only for explicit family-history context."
        in str(call["instructions"])
        for call in calls
    )


def test_extraction_converts_schema_leaf_constraints_to_native_nuextract_template(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "medications": [],
            }
        )

    run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "medications": [
                    {
                        "name": {"type": "verbatim-string"},
                        "page_number": {"type": "integer", "minimum": 1},
                        "status": {
                            "type": "string",
                            "enum": ["active", "stopped", "unknown"],
                        },
                    }
                ],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert len(calls) == 1
    assert json.loads(str(calls[0]["template"])) == {
        "schema_version": "clinical-document-extraction.v1",
        "medications": [
            {
                "name": "verbatim-string",
                "page_number": "integer",
                "status": ["active", "stopped", "unknown"],
            }
        ],
    }


def test_extraction_batches_long_documents_without_reloading_the_model(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [
                    f"page-{item['page_number']}" for item in source["source_pages"]
                ],
            }
        )

    loaded = _loaded("extraction", max_input_tokens=2_000)
    first_image = _png(tmp_path, "page-1.png")
    last_image = _png(tmp_path, "page-6.png")
    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": page_number,
                    "markdown": "bounded local OCR " * 30,
                }
                for page_number in range(1, 7)
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {
                "1": str(first_image),
                "6": str(last_image),
            },
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=loaded,  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert len(calls) > 1
    assert all(call["model"] is loaded.model for call in calls)
    for call in calls:
        source = json.loads(str(call["prompt"]).split("INPUT_JSON=", 1)[1])
        page_numbers = {item["page_number"] for item in source["source_pages"]}
        assert call["images"] == [
            path
            for page_number, path in (
                (1, str(first_image)),
                (6, str(last_image)),
            )
            if page_number in page_numbers
        ]
    assert result["result_type"] == "chunked_clinical_extraction.v1"
    assert [page_number for chunk in result["chunks"] for page_number in chunk["page_numbers"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [
        value for chunk in result["chunks"] for value in chunk["extraction"]["unresolved_fields"]
    ] == [
        "page-1",
        "page-2",
        "page-3",
        "page-4",
        "page-5",
        "page-6",
    ]


def test_extraction_isolates_selected_image_pages_without_splitting_text_batches(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[tuple[list[int], list[str]]] = []

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        page_numbers = [int(item["page_number"]) for item in source["source_pages"]]
        calls.append((page_numbers, list(kwargs["images"])))  # type: ignore[arg-type]
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [f"page-{page_number}" for page_number in page_numbers],
            }
        )

    page_two_image = _png(tmp_path, "page-2.png")
    page_five_image = _png(tmp_path, "page-5.png")
    progress: list[tuple[int, int]] = []
    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": page_number,
                    "markdown": "bounded local OCR",
                }
                for page_number in range(1, 8)
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {
                "2": str(page_two_image),
                "5": str(page_five_image),
            },
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
        progress_fn=lambda current, total: progress.append((current, total)),
    )

    assert calls == [
        ([1], []),
        ([2], [str(page_two_image)]),
        ([3, 4], []),
        ([5], [str(page_five_image)]),
        ([6, 7], []),
    ]
    assert progress == [(0, 7), (1, 7), (2, 7), (4, 7), (5, 7), (7, 7)]
    assert result["result_type"] == "chunked_clinical_extraction.v1"
    assert [page_number for chunk in result["chunks"] for page_number in chunk["page_numbers"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert [
        value for chunk in result["chunks"] for value in chunk["extraction"]["unresolved_fields"]
    ] == [f"page-{page_number}" for page_number in range(1, 8)]


def test_extraction_soft_caps_batch_input_for_the_16gb_profile(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[list[int]] = []

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        page_numbers = [int(item["page_number"]) for item in source["source_pages"]]
        calls.append(page_numbers)
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            }
        )

    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": page_number,
                    "markdown": "x" * 1_000,
                }
                for page_number in range(1, 5)
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert len(calls) > 1
    assert [page for batch in calls for page in batch] == [1, 2, 3, 4]
    assert result["result_type"] == "chunked_clinical_extraction.v1"


def test_extraction_recursively_splits_a_formatted_batch_over_the_runtime_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import nuextract3
    from local_ai_mlx_worker.common import WorkerInputLimitError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[list[int]] = []
    runtime_batches: list[list[int]] = []

    original_validate = nuextract3.validate_token_budget

    def validate(loaded: object, values: list[object], **kwargs: object) -> int:
        prompt = str(values[0])
        source = json.loads(prompt.split("INPUT_JSON=", 1)[1])
        page_numbers = [int(item["page_number"]) for item in source["source_pages"]]
        if "The prior response violated" not in prompt:
            runtime_batches.append(page_numbers)
            if len(page_numbers) > 2:
                raise WorkerInputLimitError(
                    "Formatted multimodal request exceeds the runtime limit."
                )
        return original_validate(loaded, values, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(nuextract3, "validate_token_budget", validate)

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        page_numbers = [int(item["page_number"]) for item in source["source_pages"]]
        calls.append(page_numbers)
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [f"page-{page_number}" for page_number in page_numbers],
            }
        )

    loaded = _loaded("extraction")
    progress: list[tuple[int, int]] = []
    budget_progress: list[dict[str, int]] = []
    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": page_number,
                    "markdown": "bounded local OCR",
                }
                for page_number in range(1, 7)
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=loaded,  # type: ignore[arg-type]
        generate_fn=generate,
        progress_fn=lambda current, total: progress.append((current, total)),
        budget_progress_fn=budget_progress.append,
    )

    assert runtime_batches == [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3],
        [1],
        [2, 3],
        [4, 5, 6],
        [4],
        [5, 6],
    ]
    assert calls == [[1], [2, 3], [4], [5, 6]]
    assert result["result_type"] == "chunked_clinical_extraction.v1"
    assert [page_number for chunk in result["chunks"] for page_number in chunk["page_numbers"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert all(
        chunk["extraction"]["schema_version"] == "clinical-document-extraction.v1"
        for chunk in result["chunks"]
    )
    assert progress == [(0, 6), (1, 6), (3, 6), (4, 6), (6, 6)]
    assert budget_progress[-1]["splits_used"] == 3


def test_extraction_recursively_splits_a_batch_that_cannot_finish_valid_json(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[list[int]] = []

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        page_numbers = [int(item["page_number"]) for item in source["source_pages"]]
        calls.append(page_numbers)
        if len(page_numbers) > 1:
            return "{"
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            }
        )

    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": page_number,
                    "markdown": "bounded local OCR",
                }
                for page_number in range(1, 5)
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert calls[:2] == [[1, 2, 3, 4], [1, 2, 3, 4]]
    assert result["result_type"] == "chunked_clinical_extraction.v1"
    assert [page_number for chunk in result["chunks"] for page_number in chunk["page_numbers"]] == [
        1,
        2,
        3,
        4,
    ]
    assert all(len(chunk["page_numbers"]) == 1 for chunk in result["chunks"])


def test_split_markdown_overlaps_a_fact_crossing_the_old_midpoint() -> None:
    from local_ai_mlx_worker.nuextract3 import _split_markdown

    fact = "Metformin 500 mg twice daily."
    prefix = "ordinary history " * 18
    suffix = " continued detail" * 18
    markdown = f"{prefix}{fact}{suffix}"
    midpoint = len(markdown) // 2
    assert markdown[:midpoint].find(fact) < 0
    assert markdown[midpoint:].find(fact) < 0

    left, right = _split_markdown(markdown)

    assert fact in left or fact in right
    assert len(left) < len(markdown)
    assert len(right) < len(markdown)


def test_extraction_splits_one_long_page_and_merges_page_grounded_facts(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    first = "MedicationAlpha is active.\n" + ("first section detail " * 20)
    second = "MedicationBeta is active.\n" + ("second section detail " * 20)
    markdown = f"{first}\n\n{second}"
    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        assert len(source["source_pages"]) == 1
        page = source["source_pages"][0]
        facts = []
        for name in ("MedicationAlpha", "MedicationBeta"):
            if name in page["markdown"]:
                facts.append(
                    {
                        "name": name,
                        "status": "active",
                        "verbatim": f"{name} is active.",
                        "page_number": 1,
                        "evidence_excerpt": f"{name} is active.",
                    }
                )
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "medications": facts,
            },
        )

    image = _png(tmp_path, "long-page.png")
    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": markdown}],
            "scratch_dir": str(tmp_path),
            "image_paths": {"1": str(image)},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "medications": [
                    {
                        "name": {"type": "verbatim-string"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "stopped", "historical", "unknown"],
                        },
                        "verbatim": {"type": "verbatim-string"},
                        "page_number": {"type": "integer", "minimum": 1},
                        "evidence_excerpt": {"type": "verbatim-string"},
                    }
                ],
            },
        },
        loaded=_loaded(  # type: ignore[arg-type]
            "extraction",
            max_input_tokens=2_000,
        ),
        generate_fn=generate,
    )

    fragments = [
        json.loads(str(call["prompt"]).split("INPUT_JSON=", 1)[1])["source_pages"][0]
        for call in calls
    ]
    assert len(fragments) >= 2
    assert all(call["images"] == [] for call in calls)
    assert all(str(fragment["markdown"]) in markdown for fragment in fragments)
    assert all(
        any(name in str(fragment["markdown"]) for fragment in fragments)
        for name in ("MedicationAlpha", "MedicationBeta")
    )
    assert {fragment["page_number"] for fragment in fragments} == {1}
    assert result["schema_version"] == "clinical-document-extraction.v1"
    assert [fact["name"] for fact in result["medications"]] == [
        "MedicationAlpha",
        "MedicationBeta",
    ]
    assert {fact["page_number"] for fact in result["medications"]} == {1}


def test_fragment_filter_rejects_full_page_image_fact_outside_supplied_fragment(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    first = "MedicationAlpha is active.\n" + ("first section detail " * 20)
    second = "MedicationBeta is active.\n" + ("second section detail " * 20)
    markdown = f"{first}\n\n{second}"
    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "medications": [
                    {
                        "name": "MedicationBeta",
                        "status": "active",
                        "verbatim": "MedicationBeta is active.",
                        "page_number": 1,
                        "evidence_excerpt": "MedicationBeta is active.",
                    }
                ],
            }
        )

    image = _png(tmp_path, "fragment-source.png")
    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": markdown}],
            "scratch_dir": str(tmp_path),
            "image_paths": {"1": str(image)},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "medications": [
                    {
                        "name": {"type": "verbatim-string"},
                        "status": {
                            "type": "string",
                            "enum": ["active", "stopped", "historical", "unknown"],
                        },
                        "verbatim": {"type": "verbatim-string"},
                        "page_number": {"type": "integer", "minimum": 1},
                        "evidence_excerpt": {"type": "verbatim-string"},
                    }
                ],
            },
        },
        loaded=_loaded(  # type: ignore[arg-type]
            "extraction",
            max_input_tokens=2_000,
        ),
        generate_fn=generate,
    )

    assert len(calls) >= 2
    assert all(call["images"] == [] for call in calls)
    assert [fact["name"] for fact in result["medications"]] == ["MedicationBeta"]


def test_extraction_splits_immediately_after_cap_length_invalid_output(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    markdown = ("first bounded section " * 20) + "\n\n" + ("second bounded section " * 20)
    calls: list[str] = []

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        source_markdown = str(source["source_pages"][0]["markdown"])
        calls.append(source_markdown)
        if source_markdown == markdown:
            return "{" + ("x" * 4_095)
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            }
        )

    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": markdown}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert calls.count(markdown) == 1
    assert all(call in markdown for call in calls[1:])
    assert sum(len(call) for call in calls[1:]) >= len(markdown)
    assert result == {
        "schema_version": "clinical-document-extraction.v1",
        "unresolved_fields": [],
    }


def test_extraction_uses_exact_stream_token_count_to_split_truncated_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real decode-cap hit cannot be hidden by decoded-text re-tokenization."""

    import mlx_vlm

    from local_ai_mlx_worker.nuextract3 import run_extraction

    markdown = ("first bounded section " * 20) + "\n\n" + ("second bounded section " * 20)
    calls: list[str] = []

    class StreamResult:
        def __init__(self, text: str, generation_tokens: int) -> None:
            self.text = text
            self.generation_tokens = generation_tokens

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda _processor, _config, prompt, **_kwargs: prompt,
    )

    def stream_generate(
        _model: object,
        _processor: object,
        prompt: str,
        **_kwargs: object,
    ) -> Iterator[StreamResult]:
        source = json.loads(prompt.split("INPUT_JSON=", 1)[1])
        source_markdown = str(source["source_pages"][0]["markdown"])
        calls.append(source_markdown)
        if source_markdown == markdown:
            yield StreamResult("{", 4_096)
            return
        yield StreamResult(
            json.dumps(
                {
                    "schema_version": "clinical-document-extraction.v1",
                    "unresolved_fields": [],
                }
            ),
            32,
        )

    monkeypatch.setattr(mlx_vlm, "stream_generate", stream_generate)

    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": markdown}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
    )

    assert calls.count(markdown) == 1
    assert all(call in markdown for call in calls[1:])
    assert result == {
        "schema_version": "clinical-document-extraction.v1",
        "unresolved_fields": [],
    }


def test_extraction_stops_at_fragment_depth_after_repeated_invalid_output(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    with pytest.raises(GenerationError) as error:
        run_extraction(
            {
                "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"schema_version": "clinical-document-extraction.v1"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "{",
        )

    assert error.value.category == "fragment_depth_limit"


def test_extraction_splits_after_two_under_cap_invalid_structured_outputs(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    markdown = ("first bounded section " * 20) + "\n\n" + ("second bounded section " * 20)
    calls: list[str] = []

    def generate(**kwargs: object) -> str:
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        source_markdown = str(source["source_pages"][0]["markdown"])
        calls.append(source_markdown)
        if source_markdown == markdown:
            return "{"
        return json.dumps(
            {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            }
        )

    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": markdown}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {
                "schema_version": "clinical-document-extraction.v1",
                "unresolved_fields": [],
            },
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert calls.count(markdown) == 2
    assert all(call in markdown for call in calls[2:])
    assert result == {
        "schema_version": "clinical-document-extraction.v1",
        "unresolved_fields": [],
    }


def test_extraction_does_not_split_non_structured_generation_failures(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls = 0

    def generate(**_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise GenerationError("Local generation failed.", category="stream_contract")

    with pytest.raises(GenerationError) as error:
        run_extraction(
            {
                "page_markdown": [
                    {"page_number": 1, "markdown": "first bounded OCR"},
                    {"page_number": 2, "markdown": "second bounded OCR"},
                ],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"schema_version": "clinical-document-extraction.v1"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=generate,
        )

    assert calls == 1
    assert error.value.category == "stream_contract"


def test_extraction_does_not_split_runtime_input_failures(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import WorkerInputLimitError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls = 0
    budget_progress: list[dict[str, int]] = []

    def generate(**_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise WorkerInputLimitError("Runtime rejected a generated request.")

    with pytest.raises(WorkerInputLimitError):
        run_extraction(
            {
                "page_markdown": [
                    {"page_number": 1, "markdown": "first bounded OCR"},
                    {"page_number": 2, "markdown": "second bounded OCR"},
                ],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"schema_version": "clinical-document-extraction.v1"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=generate,
            budget_progress_fn=budget_progress.append,
        )

    assert calls == 1
    assert budget_progress[-1]["attempt"] == 1
    assert budget_progress[-1]["output_tokens"] == 4_096
    assert budget_progress[-1]["splits_used"] == 0


def test_unsplittable_invalid_structured_output_preserves_bounded_category(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls = 0

    def generate(**_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return "{"

    with pytest.raises(GenerationError) as error:
        run_extraction(
            {
                "page_markdown": [{"page_number": 1, "markdown": "x"}],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"schema_version": "clinical-document-extraction.v1"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=generate,
        )

    assert calls == 2
    assert error.value.category == "invalid_structured_output"


def test_generation_preserves_exact_cap_metadata_without_rejecting_valid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mlx_vlm

    from local_ai_mlx_worker.common import generate_content
    from local_ai_mlx_worker.nuextract3 import run_extraction

    class StreamResult:
        text = '{"schema_version":"clinical-document-extraction.v1"}'
        generation_tokens = 4_096

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: iter([StreamResult()]),
    )

    generated = generate_content(
        model=_Model(),
        processor=_Processor(),
        prompt="source",
        images=[],
        max_tokens=4_096,
        temperature=0.0,
        do_sample=False,
        input_token_limit=100,
    )

    assert generated.generation_tokens == 4_096
    assert run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"schema_version": "clinical-document-extraction.v1"},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: generated,
    ) == {"schema_version": "clinical-document-extraction.v1"}


def test_extraction_work_budget_enforces_exact_independent_limits() -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import (
        MAX_EXTRACTION_GENERATION_ATTEMPTS,
        MAX_EXTRACTION_RUNTIME_SPLITS,
        _ExtractionWorkBudget,
    )

    token_budget = _ExtractionWorkBudget()
    for _ in range(4):
        assert token_budget.next_output_cap(8_192) == 4_096
        token_budget.reserve_attempt()
        token_budget.charge_generation(4_096)
    with pytest.raises(GenerationError) as token_error:
        token_budget.next_output_cap(8_192)
    assert token_error.value.category == "work_token_limit"

    attempt_budget = _ExtractionWorkBudget()
    for _ in range(MAX_EXTRACTION_GENERATION_ATTEMPTS):
        attempt_budget.reserve_attempt()
    with pytest.raises(GenerationError) as attempt_error:
        attempt_budget.reserve_attempt()
    assert attempt_error.value.category == "work_attempt_limit"
    assert attempt_budget.attempts == MAX_EXTRACTION_GENERATION_ATTEMPTS

    split_budget = _ExtractionWorkBudget()
    for _ in range(MAX_EXTRACTION_RUNTIME_SPLITS):
        split_budget.reserve_runtime_split()
    with pytest.raises(GenerationError) as split_error:
        split_budget.reserve_runtime_split()
    assert split_error.value.category == "work_split_limit"
    assert split_budget.runtime_splits == MAX_EXTRACTION_RUNTIME_SPLITS


def test_extraction_fragment_depth_three_is_private_and_depth_four_fails() -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import _prompt_pages, _split_page

    fragment = {"page_number": 1, "markdown": "bounded clinical text " * 128}
    for _ in range(3):
        fragment, _right = _split_page(fragment)

    assert set(_prompt_pages([fragment])[0]) == {"page_number", "markdown"}
    with pytest.raises(GenerationError) as error:
        _split_page(fragment)
    assert error.value.category == "fragment_depth_limit"


def test_extraction_shrinks_call_cap_and_stops_at_exact_token_budget(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GeneratedText, GenerationError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    call_caps: list[int] = []

    def generate(**kwargs: object) -> GeneratedText:
        call_cap = int(kwargs["max_tokens"])
        call_caps.append(call_cap)
        source = json.loads(str(kwargs["prompt"]).split("INPUT_JSON=", 1)[1])
        if len(source["source_pages"]) > 1:
            return GeneratedText("{", generation_tokens=call_cap - 1)
        return GeneratedText(
            '{"schema_version":"clinical-document-extraction.v1"}',
            generation_tokens=call_cap,
        )

    with pytest.raises(GenerationError) as error:
        run_extraction(
            {
                "page_markdown": [
                    {"page_number": page, "markdown": f"bounded OCR page {page}"}
                    for page in range(1, 5)
                ],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"schema_version": "clinical-document-extraction.v1"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=generate,
        )

    assert error.value.category == "work_token_limit"
    assert call_caps == [4_096, 4_096, 4_096, 4_096, 4]


def test_extraction_budget_progress_is_monotonic_and_content_free(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GeneratedText
    from local_ai_mlx_worker.nuextract3 import run_extraction

    events: list[dict[str, int]] = []
    outputs = iter(
        [
            GeneratedText("{", generation_tokens=7),
            GeneratedText(
                '{"schema_version":"clinical-document-extraction.v1"}',
                generation_tokens=11,
            ),
        ]
    )

    assert run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "private clinical OCR"}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"schema_version": "clinical-document-extraction.v1"},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: next(outputs),
        budget_progress_fn=events.append,
    ) == {"schema_version": "clinical-document-extraction.v1"}

    assert events
    assert all(
        set(event)
        == {
            "attempt",
            "attempt_limit",
            "output_tokens",
            "output_token_limit",
            "splits_used",
            "split_limit",
        }
        for event in events
    )
    assert [event["attempt"] for event in events] == sorted(event["attempt"] for event in events)
    assert [event["output_tokens"] for event in events] == sorted(
        event["output_tokens"] for event in events
    )
    assert events[-1]["attempt"] == 2
    assert events[-1]["output_tokens"] == 18
    assert "private clinical OCR" not in json.dumps(events)


def test_extraction_conservatively_charges_oversized_exact_metadata(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import GeneratedText
    from local_ai_mlx_worker.nuextract3 import run_extraction

    events: list[dict[str, int]] = []
    assert run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"schema_version": "clinical-document-extraction.v1"},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: GeneratedText(
            '{"schema_version":"clinical-document-extraction.v1"}',
            generation_tokens=99_999,
        ),
        budget_progress_fn=events.append,
    ) == {"schema_version": "clinical-document-extraction.v1"}
    assert events[-1]["output_tokens"] == 4_096


def test_generation_builds_locked_json_schema_logits_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm
    from mlx_vlm import structured

    from local_ai_mlx_worker.common import generate_content

    schema = {
        "type": "object",
        "properties": {"sections": {"type": "array"}},
        "required": ["sections"],
        "additionalProperties": False,
    }
    logits_processor = object()
    builder_calls: list[tuple[object, object]] = []
    stream_calls: list[dict[str, object]] = []

    def build(tokenizer: object, received_schema: object) -> object:
        builder_calls.append((tokenizer, received_schema))
        return logits_processor

    def stream(*_args: object, **kwargs: object) -> Iterator[str]:
        stream_calls.append(kwargs)
        return iter(['{"sections":[]}'])

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(structured, "build_json_schema_logits_processor", build)
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream)

    assert (
        generate_content(
            model=_Model(),
            processor=_Processor(),
            prompt="source",
            images=[],
            max_tokens=256,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
            json_schema=schema,
        )
        == '{"sections":[]}'
    )
    assert builder_calls == [(_Processor.tokenizer, schema)]
    assert stream_calls[0]["logits_processors"] == [logits_processor]


def test_generation_schema_compile_failure_never_starts_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm
    from mlx_vlm import structured

    from local_ai_mlx_worker.common import generate_content

    stream = Mock(side_effect=AssertionError("generation started"))
    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        structured,
        "build_json_schema_logits_processor",
        Mock(side_effect=RuntimeError("schema compile failed")),
    )
    monkeypatch.setattr(mlx_vlm, "stream_generate", stream)

    with pytest.raises(RuntimeError, match="schema compile failed"):
        generate_content(
            model=_Model(),
            processor=_Processor(),
            prompt="source",
            images=[],
            max_tokens=256,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
            json_schema={"type": "object"},
        )

    stream.assert_not_called()


def test_extraction_deterministically_grounds_explicit_assertion_phrases(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "medications": [
                    {
                        "name": "Warfarin",
                        "status": "active",
                        "verbatim": "Warfarin 2.5 mg was stopped.",
                        "page_number": 1,
                        "evidence_excerpt": "Warfarin 2.5 mg was stopped.",
                    }
                ],
                "conditions": [
                    {
                        "name": "pneumonia",
                        "assertion": "present",
                        "verbatim": "No evidence of pneumonia.",
                        "page_number": 1,
                        "evidence_excerpt": "No evidence of pneumonia.",
                    },
                    {
                        "name": "colon cancer",
                        "assertion": "present",
                        "verbatim": "FHx: colon cancer.",
                        "page_number": 1,
                        "evidence_excerpt": "FHx: colon cancer.",
                    },
                    {
                        "name": "pulmonary embolism",
                        "assertion": "present",
                        "verbatim": "Possible pulmonary embolism.",
                        "page_number": 1,
                        "evidence_excerpt": "Possible pulmonary embolism.",
                    },
                    {
                        "name": "dizziness",
                        "assertion": "present",
                        "verbatim": "Denles dizziness.",
                        "page_number": 1,
                        "evidence_excerpt": "Denles dizziness.",
                    },
                ],
                "procedures": [
                    {
                        "name": "Colonoscopy",
                        "assertion": "present",
                        "verbatim": "Colonoscopy cancelled.",
                        "page_number": 1,
                        "evidence_excerpt": "Colonoscopy cancelled.",
                    }
                ],
                "diagnostic_reports": [
                    {
                        "name": "Laboratory report",
                        "findings": "A1c 6.8%",
                        "assertion": "present",
                        "status": "final",
                        "verbatim": (
                            "Laboratory report findings: A1c 6.8%. No evidence of pneumonia."
                        ),
                        "page_number": 1,
                        "evidence_excerpt": (
                            "Laboratory report findings: A1c 6.8%. No evidence of pneumonia."
                        ),
                    }
                ],
            }
        )

    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": "No evidence of pneumonia."}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"conditions": []},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert result["medications"][0]["status"] == "stopped"  # type: ignore[index]
    assert result["conditions"] == [  # type: ignore[index]
        {
            "name": "colon cancer",
            "assertion": "family_history",
            "verbatim": "FHx: colon cancer.",
            "page_number": 1,
            "evidence_excerpt": "FHx: colon cancer.",
        }
    ]
    assert result["procedures"] == []
    assert result["diagnostic_reports"][0]["assertion"] == "present"  # type: ignore[index]
    assert result["diagnostic_reports"][0]["status"] == "unknown"  # type: ignore[index]
    assert result["rejected_fields"] == [
        "conditions[0]",
        "conditions[2]",
        "conditions[3]",
        "procedures[0]",
    ]
    assert len(calls) == 1


def test_extraction_binds_missing_table_evidence_to_exact_source_row(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    row = "<tr><td>Potassium</td><td>4.1</td><td>mmol/L</td></tr>"
    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": 1,
                    "markdown": f"<table><tbody>{row}</tbody></table>",
                }
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"labs": []},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "labs": [
                    {
                        "name": "Potassium",
                        "value": "4.1",
                        "unit": "mmol/L",
                        "verbatim": None,
                        "page_number": 1,
                        "evidence_excerpt": None,
                    }
                ]
            }
        ),
    )

    assert result["labs"][0]["verbatim"] == row  # type: ignore[index]
    assert result["labs"][0]["evidence_excerpt"] == row  # type: ignore[index]


def test_extraction_discards_fact_with_missing_required_subject(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": 1,
                    "markdown": "Insulin glargine 12 units nightly.",
                }
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"care_plans": []},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "care_plans": [
                    {
                        "title": None,
                        "plan_items": [{"type": "Insulin glargine nightly"}],
                        "status": "active",
                        "verbatim": "Insulin glargine 12 units nightly.",
                        "page_number": 1,
                        "evidence_excerpt": "Insulin glargine 12 units nightly.",
                    }
                ],
                "rejected_fields": [],
            }
        ),
    )

    assert result["care_plans"] == []
    assert result["rejected_fields"] == ["care_plans[0].title"]


def test_extraction_binds_missing_adjacent_numeric_unit(tmp_path: Path) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    sentence = "Insulin glargine 12 units nightly."
    result = run_extraction(
        {
            "page_markdown": [{"page_number": 1, "markdown": sentence}],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"medications": []},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "medications": [
                    {
                        "name": "Insulin glargine",
                        "dose_value": "12",
                        "dose_unit": None,
                        "status": "active",
                        "verbatim": sentence,
                        "page_number": 1,
                        "evidence_excerpt": sentence,
                    }
                ]
            }
        ),
    )

    assert result["medications"][0]["dose_unit"] == "units"  # type: ignore[index]


def test_extraction_filters_wrappers_and_prefers_vital_category(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.nuextract3 import run_extraction

    table_row = "Blood pressure | 120/80 | mmHg"
    ct_evidence = "CT chest report findings: stable pulmonary nodule."
    result = run_extraction(
        {
            "page_markdown": [
                {
                    "page_number": 1,
                    "markdown": f"{table_row}\n{ct_evidence}",
                }
            ],
            "scratch_dir": str(tmp_path),
            "image_paths": {},
            "schema": {"labs": [], "vital_signs": [], "diagnostic_reports": []},
        },
        loaded=_loaded("extraction"),  # type: ignore[arg-type]
        generate_fn=lambda **_kwargs: json.dumps(
            {
                "labs": [
                    {
                        "name": "Blood pressure",
                        "value": "120/80",
                        "unit": "mmHg",
                        "verbatim": table_row,
                        "page_number": 1,
                        "evidence_excerpt": table_row,
                    }
                ],
                "vital_signs": [
                    {
                        "name": "Blood pressure",
                        "value": "120/80",
                        "unit": "mmHg",
                        "verbatim": table_row,
                        "page_number": 1,
                        "evidence_excerpt": table_row,
                    }
                ],
                "diagnostic_reports": [
                    {
                        "name": "Blood pressure",
                        "findings": "120/80 mmHg",
                        "status": "final",
                        "assertion": "present",
                        "verbatim": table_row,
                        "page_number": 1,
                        "evidence_excerpt": table_row,
                    },
                    {
                        "name": "CT chest",
                        "findings": "stable pulmonary nodule",
                        "status": "final",
                        "assertion": "present",
                        "verbatim": ct_evidence,
                        "page_number": 1,
                        "evidence_excerpt": ct_evidence,
                    },
                ],
            }
        ),
    )

    assert result["labs"] == []
    assert len(result["vital_signs"]) == 1  # type: ignore[arg-type]
    assert [item["name"] for item in result["diagnostic_reports"]] == ["CT chest"]  # type: ignore[union-attr]
    assert result["rejected_fields"] == [
        "labs[0]",
        "diagnostic_reports[0]",
    ]


def test_summary_accepts_only_validated_fact_and_evidence_inputs() -> None:
    from local_ai_mlx_worker.qwen_summary import run_summary

    calls: list[dict[str, object]] = []
    payload = _summary_payload()
    payload["max_output_tokens"] = 301
    fact = payload["facts"][0]  # type: ignore[index]

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "sections": [
                    {
                        "heading": "Medications",
                        "claims": [
                            {
                                "fact_id": fact["fact_id"],
                                "field_paths": ["/name"],
                                "evidence_ids": fact["evidence_ids"],
                            }
                        ],
                    }
                ],
                "uncertainties": [],
            },
            separators=(",", ":"),
        )

    loaded = _loaded("summary")
    result = run_summary(
        payload,
        loaded=loaded,  # type: ignore[arg-type]
        generate_fn=generate,
    )

    assert result["sections"][0]["claims"][0] == {  # type: ignore[index]
        "fact_id": fact["fact_id"],
        "field_paths": ["/name"],
        "evidence_ids": fact["evidence_ids"],
    }
    assert len(calls) == 1
    assert calls[0]["images"] == []
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["do_sample"] is False
    assert calls[0]["input_token_limit"] == 32768
    assert calls[0]["max_tokens"] == 301
    schema = calls[0]["json_schema"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    section_schema = next(
        item
        for item in schema["properties"]["sections"]["items"]["oneOf"]
        if item["properties"]["heading"] == {"const": "Medications"}
    )
    assert section_schema["properties"]["heading"] == {"const": "Medications"}
    claim_schema = section_schema["properties"]["claims"]["items"]["oneOf"][0]
    assert claim_schema["properties"]["fact_id"] == {"const": fact["fact_id"]}
    assert claim_schema["properties"]["field_paths"]["items"] == {"enum": ["/name"]}
    assert claim_schema["properties"]["evidence_ids"] == {
        "type": "array",
        "items": {"enum": fact["evidence_ids"]},
        "minItems": 1,
        "maxItems": 1,
        "uniqueItems": True,
    }
    evidence_id = fact["evidence_ids"][0]
    assert {
        "if": {
            "properties": {
                "field_paths": {"contains": {"const": "/name"}},
            },
            "required": ["field_paths"],
        },
        "then": {
            "properties": {
                "evidence_ids": {"contains": {"enum": [evidence_id]}},
            }
        },
    } in claim_schema["allOf"]
    assert {
        "if": {
            "properties": {
                "evidence_ids": {"contains": {"const": evidence_id}},
            },
            "required": ["evidence_ids"],
        },
        "then": {
            "properties": {
                "field_paths": {"contains": {"enum": ["/name"]}},
            }
        },
    } in claim_schema["allOf"]
    uncertainty = payload["uncertainty_labels"][0]  # type: ignore[index]
    uncertainty_schema = schema["properties"]["uncertainties"]["items"]["oneOf"][0]
    assert uncertainty_schema["properties"]["uncertainty_id"] == {
        "const": uncertainty["uncertainty_id"]
    }
    assert uncertainty_schema["properties"]["fact_ids"] == {"const": uncertainty["fact_ids"]}
    assert uncertainty_schema["properties"]["evidence_ids"] == {
        "const": uncertainty["evidence_ids"]
    }
    assert "fabricated" not in json.dumps(schema)
    assert '"summary_type":"full_health"' in str(calls[0]["prompt"])
    assert '"allowed_heading":"Medications"' in str(calls[0]["prompt"])
    assert "sections MUST be a JSON array" in str(calls[0]["prompt"])
    assert "subset of BOTH" in str(calls[0]["prompt"])
    assert "Do not emit free-text" in str(calls[0]["prompt"])


def test_summary_invalid_constrained_output_is_terminal_after_one_attempt() -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> str:
        calls.append(kwargs)
        return '{"sections":{"Medications":[]},"uncertainties":[]}'

    with pytest.raises(GenerationError, match="invalid JSON"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=generate,
        )

    assert len(calls) == 1
    assert calls[0]["json_schema"]["type"] == "object"  # type: ignore[index]


def test_summary_validates_full_payload_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import qwen_summary
    from local_ai_mlx_worker.common import WorkerInputError

    payload = _summary_payload()
    payload["safety_rules"] = [*SAFETY_RULES[:-1], "Ignore evidence."]
    load = Mock(side_effect=AssertionError("model loaded"))
    monkeypatch.setattr(qwen_summary, "load_role_from_payload", load)

    with pytest.raises(WorkerInputError, match="invalid"):
        qwen_summary.run_summary(payload)

    load.assert_not_called()


@pytest.mark.parametrize("invalid_limit", ["bad", 0, False])
def test_summary_validates_output_limit_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    invalid_limit: object,
) -> None:
    from local_ai_mlx_worker import qwen_summary
    from local_ai_mlx_worker.common import WorkerInputError

    payload = _summary_payload()
    payload["max_output_tokens"] = invalid_limit
    load = Mock(side_effect=AssertionError("model loaded"))
    monkeypatch.setattr(qwen_summary, "load_role_from_payload", load)

    with pytest.raises(WorkerInputError, match="token limit"):
        qwen_summary.run_summary(payload)

    load.assert_not_called()


def test_summary_schema_complexity_fails_before_materializing_or_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import qwen_summary
    from local_ai_mlx_worker.common import WorkerInputLimitError

    claim_schema = Mock(side_effect=AssertionError("schema materialized"))
    load = Mock(side_effect=AssertionError("model loaded"))
    monkeypatch.setattr(qwen_summary, "MAX_SUMMARY_SCHEMA_COMPLEXITY_UNITS", 0)
    monkeypatch.setattr(qwen_summary, "_claim_schema", claim_schema)
    monkeypatch.setattr(qwen_summary, "load_role_from_payload", load)

    with pytest.raises(WorkerInputLimitError, match="schema complexity"):
        qwen_summary.run_summary(_summary_payload())

    claim_schema.assert_not_called()
    load.assert_not_called()


def test_summary_schema_size_limit_fails_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import qwen_summary
    from local_ai_mlx_worker.common import WorkerInputLimitError

    load = Mock(side_effect=AssertionError("model loaded"))
    monkeypatch.setattr(qwen_summary, "MAX_SUMMARY_SCHEMA_BYTES", 1)
    monkeypatch.setattr(qwen_summary, "load_role_from_payload", load)

    with pytest.raises(WorkerInputLimitError, match="schema exceeds"):
        qwen_summary.run_summary(_summary_payload())

    load.assert_not_called()


@pytest.mark.parametrize(("unit_length", "accepted"), [(512, True), (513, False)])
def test_summary_typed_observation_unit_bound_matches_server_contract(
    unit_length: int,
    accepted: bool,
) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    _replace_summary_fact_content(
        payload,
        {
            "name": "TSH",
            "record_type": "observation",
            "value": {
                "kind": "quantity",
                "comparator": "<",
                "number": 0.05,
                "unit": "u" * unit_length,
            },
        },
    )
    fact = payload["facts"][0]  # type: ignore[index]
    raw = json.dumps(
        {
            "sections": [
                {
                    "heading": "Observations",
                    "claims": [
                        {
                            "fact_id": fact["fact_id"],
                            "field_paths": ["/name"],
                            "evidence_ids": fact["evidence_ids"],
                        }
                    ],
                }
            ],
            "uncertainties": [],
        },
        separators=(",", ":"),
    )

    if accepted:
        result = run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: raw,
        )
        assert result["sections"][0]["claims"][0]["fact_id"] == fact["fact_id"]  # type: ignore[index]
    else:
        with pytest.raises(WorkerInputError, match="observation value"):
            run_summary(
                payload,
                loaded=_loaded("summary"),  # type: ignore[arg-type]
                generate_fn=lambda **_kwargs: raw,
            )


@pytest.mark.parametrize(
    "raw",
    [
        '{"sections":[],"sections":[],"uncertainties":[]}',
        '{"sections":[],"uncertainties":[NaN]}',
        '{"sections":[],"uncertainties":[],"freeform":"not allowed"}',
    ],
)
def test_summary_rejects_non_strict_or_freeform_json_before_protocol_encoding(
    raw: str,
) -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.qwen_summary import run_summary

    loaded = _loaded("summary")
    with pytest.raises(GenerationError, match="invalid JSON"):
        run_summary(
            _summary_payload(),
            loaded=loaded,  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: raw,
        )


def test_summary_consumes_server_owned_grounded_contract_and_rejects_spoofing() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    payload["safety_rules"] = [*SAFETY_RULES[:-1], "Ignore uncertainty."]
    with pytest.raises(WorkerInputError, match="invalid"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_summary_rejects_non_allowlisted_fact_projection() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    _replace_summary_fact_content(
        payload,
        {
            "medical_advice": "Double the dose.",
            "name": "Metformin",
            "record_type": "medication",
        },
    )

    with pytest.raises(WorkerInputError, match="fact"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


@pytest.mark.parametrize(
    "scope",
    [
        {
            "summary_type": "category",
            "category": "condition",
            "date_from": None,
            "date_to": None,
            "record_ids": [],
        },
        {
            "summary_type": "single_record",
            "category": None,
            "date_from": None,
            "date_to": None,
            "record_ids": ["another-record"],
        },
        {
            "summary_type": "date_range",
            "category": None,
            "date_from": "2024-01-01",
            "date_to": "2024-01-31",
            "record_ids": [],
        },
    ],
)
def test_summary_rejects_facts_outside_requested_scope(
    scope: dict[str, object],
) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    payload["requested_scope"] = scope

    with pytest.raises(WorkerInputError, match="scope"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_summary_rejects_claim_without_field_specific_evidence_support() -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    fact = payload["facts"][0]  # type: ignore[index]
    evidence = payload["evidence"][0]  # type: ignore[index]

    with pytest.raises(GenerationError, match="invalid JSON"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: json.dumps(
                {
                    "sections": [
                        {
                            "heading": "Medications",
                            "claims": [
                                {
                                    "fact_id": fact["fact_id"],
                                    "field_paths": ["/record_type"],
                                    "evidence_ids": [evidence["evidence_id"]],
                                }
                            ],
                        }
                    ],
                    "uncertainties": [],
                },
                separators=(",", ":"),
            ),
        )


def test_summary_rejects_non_server_uncertainty_label() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    payload["uncertainty_labels"][0]["label"] = "Double the dose."  # type: ignore[index]

    with pytest.raises(WorkerInputError, match="uncertainty"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_summary_rejects_raw_payload_fields_and_invented_uncertainty() -> None:
    from local_ai_mlx_worker.common import GenerationError, WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    payload["raw_upload"] = "must not enter the worker"
    with pytest.raises(WorkerInputError, match="invalid"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )

    with pytest.raises(GenerationError, match="invalid JSON"):
        uncertainty = _summary_payload()["uncertainty_labels"][0]  # type: ignore[index]
        run_summary(
            _summary_payload(),
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: (
                '{"sections":[],"uncertainties":[{'
                '"uncertainty_id":"uncertainty1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
                f'"fact_ids":{json.dumps(uncertainty["fact_ids"])},'
                f'"evidence_ids":{json.dumps(uncertainty["evidence_ids"])}'
                "}]}",
            ),
        )


def test_summary_rejects_control_characters_inside_fact_content() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    payload = _summary_payload()
    payload["facts"][0]["content_json"] = '{"name":"Metformin\\u0000hidden"}'  # type: ignore[index]
    with pytest.raises(WorkerInputError, match="invalid"):
        run_summary(
            payload,
            loaded=_loaded("summary"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_summary_uses_tokenizer_and_model_context_limits() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    with pytest.raises(WorkerInputError, match="token"):
        run_summary(
            _summary_payload(),
            loaded=_loaded("summary", max_input_tokens=32),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_summary_reserves_output_inside_model_context_window() -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.qwen_summary import run_summary

    class SmallContextConfig:
        max_position_embeddings = 1024

    class SmallContextModel:
        config = SmallContextConfig()

    loaded = replace(_loaded("summary"), model=SmallContextModel())  # type: ignore[arg-type]
    with pytest.raises(WorkerInputError, match="context token"):
        run_summary(
            _summary_payload(),
            loaded=loaded,
            generate_fn=lambda **_kwargs: '{"sections":[],"uncertainties":[]}',
        )


def test_generation_checks_formatted_prompt_plus_output_against_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm

    from local_ai_mlx_worker.common import WorkerInputError, generate_content

    class TinyConfig:
        max_position_embeddings = 10

    class TinyModel:
        config = TinyConfig()

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "12345",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "generate",
        lambda *_args, **_kwargs: "must not run",
    )

    with pytest.raises(WorkerInputError, match="context token"):
        generate_content(
            model=TinyModel(),
            processor=_Processor(),
            prompt="12345",
            images=[],
            max_tokens=6,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
        )


def test_generation_strips_only_a_declared_terminal_eos_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm

    from local_ai_mlx_worker.common import generate_content

    class TerminalTokenizer(_Tokenizer):
        eos_token = "<|im_end|>"

    class TerminalProcessor:
        tokenizer = TerminalTokenizer()

    generated = ['{"labs":[]}<|im_end|>\n', '{"text":"<|im_end|> inside"}']
    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: iter([generated.pop(0)]),
    )

    first = generate_content(
        model=_Model(),
        processor=TerminalProcessor(),
        prompt="source",
        images=[],
        max_tokens=100,
        temperature=0.0,
        do_sample=False,
        input_token_limit=100,
    )
    second = generate_content(
        model=_Model(),
        processor=TerminalProcessor(),
        prompt="source",
        images=[],
        max_tokens=100,
        temperature=0.0,
        do_sample=False,
        input_token_limit=100,
    )

    assert first == '{"labs":[]}'
    assert second == '{"text":"<|im_end|> inside"}'


def test_generation_streams_identical_text_and_emits_content_free_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heartbeat is driven by stream events, never generated text."""

    import mlx_vlm

    from local_ai_mlx_worker.common import generate_content

    class StreamItem:
        def __init__(self, text: str) -> None:
            self.text = text

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "generate",
        lambda *_args, **_kwargs: "must-not-use-non-streaming-generation",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: iter([StreamItem("{"), StreamItem("}")]),
    )
    activity: list[int] = []

    result = generate_content(
        model=_Model(),
        processor=_Processor(),
        prompt="source",
        images=[],
        max_tokens=100,
        temperature=0.0,
        do_sample=False,
        input_token_limit=100,
        activity_fn=lambda: activity.append(1),
    )

    assert result == "{}"
    assert activity == [1, 1]


def test_generation_throttles_stream_activity_without_changing_generated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activity is bounded even when MLX yields many text fragments."""

    import mlx_vlm

    from local_ai_mlx_worker.common import generate_content

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: iter(["x"] * 65),
    )
    activity: list[int] = []

    result = generate_content(
        model=_Model(),
        processor=_Processor(),
        prompt="source",
        images=[],
        max_tokens=100,
        temperature=0.0,
        do_sample=False,
        input_token_limit=100,
        activity_fn=lambda: activity.append(1),
    )

    assert result == "x" * 65
    assert activity == [1, 1, 1]


def test_generation_closes_stream_before_collecting_and_clearing_mlx_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed generation releases stream frames before allocator cleanup."""

    import mlx_vlm

    from local_ai_mlx_worker import common

    events: list[str] = []

    class Stream:
        def __iter__(self) -> Iterator[str]:
            return iter(["{", "}"])

        def close(self) -> None:
            events.append("close")

    class FakeMlx:
        @staticmethod
        def device_info() -> dict[str, int]:
            return {"max_recommended_working_set_size": 12 * 1024**3}

        @staticmethod
        def set_memory_limit(_limit: int) -> None:
            events.append("memory-limit")

        @staticmethod
        def set_cache_limit(_limit: int) -> None:
            events.append("cache-limit")

        @staticmethod
        def clear_cache() -> None:
            events.append("clear")

    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: Stream(),
    )
    monkeypatch.setattr(common, "_load_mlx_core", lambda: FakeMlx())
    monkeypatch.setattr(common.gc, "collect", lambda: events.append("collect"))

    assert (
        common.generate_content(
            model=_Model(),
            processor=_Processor(),
            prompt="source",
            images=[],
            max_tokens=100,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
        )
        == "{}"
    )
    assert events[-3:] == ["close", "collect", "clear"]


def test_generation_cleans_mlx_after_stream_error_and_subsequent_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed attempt cannot retain stream frames or cache into its retry."""

    import mlx_vlm

    from local_ai_mlx_worker import common

    events: list[str] = []

    class Stream:
        def __init__(self, *, fail: bool) -> None:
            self._fail = fail

        def __iter__(self) -> Iterator[str]:
            yield "{"
            if self._fail:
                raise RuntimeError("private generation failure")
            yield "}"

        def close(self) -> None:
            events.append("close")

    class FakeMlx:
        @staticmethod
        def device_info() -> dict[str, int]:
            return {"max_recommended_working_set_size": 12 * 1024**3}

        @staticmethod
        def set_memory_limit(_limit: int) -> None:
            return None

        @staticmethod
        def set_cache_limit(_limit: int) -> None:
            return None

        @staticmethod
        def clear_cache() -> None:
            events.append("clear")

    attempts = iter([Stream(fail=True), Stream(fail=False)])
    monkeypatch.setattr(
        mlx_vlm,
        "apply_chat_template",
        lambda *_args, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        mlx_vlm,
        "stream_generate",
        lambda *_args, **_kwargs: next(attempts),
    )
    monkeypatch.setattr(common, "_load_mlx_core", lambda: FakeMlx())
    monkeypatch.setattr(common.gc, "collect", lambda: events.append("collect"))

    with pytest.raises(RuntimeError, match="private generation failure"):
        common.generate_content(
            model=_Model(),
            processor=_Processor(),
            prompt="source",
            images=[],
            max_tokens=100,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
        )
    assert events == ["close", "collect", "clear"]

    assert (
        common.generate_content(
            model=_Model(),
            processor=_Processor(),
            prompt="source",
            images=[],
            max_tokens=100,
            temperature=0.0,
            do_sample=False,
            input_token_limit=100,
        )
        == "{}"
    )
    assert events == [
        "close",
        "collect",
        "clear",
        "close",
        "collect",
        "clear",
    ]


def test_mlx_memory_configuration_uses_working_set_and_bounded_cache() -> None:
    """The 16 GB runtime avoids MLX's larger default allocator allowance."""

    from local_ai_mlx_worker.common import (
        MLX_FREE_CACHE_LIMIT_BYTES,
        _configure_mlx_memory,
    )

    recommended = 12 * 1024**3
    calls: list[tuple[str, int]] = []

    class FakeMlx:
        @staticmethod
        def device_info() -> dict[str, int]:
            return {"max_recommended_working_set_size": recommended}

        @staticmethod
        def set_memory_limit(limit: int) -> None:
            calls.append(("memory", limit))

        @staticmethod
        def set_cache_limit(limit: int) -> None:
            calls.append(("cache", limit))

    _configure_mlx_memory(FakeMlx())

    assert calls == [
        ("memory", recommended),
        ("cache", MLX_FREE_CACHE_LIMIT_BYTES),
    ]
    assert MLX_FREE_CACHE_LIMIT_BYTES <= 256 * 1024**2


def test_mlx_memory_configuration_tolerates_missing_runtime_apis() -> None:
    """Non-Mac and test fakes remain importable when MLX APIs are absent."""

    from local_ai_mlx_worker.common import _configure_mlx_memory

    _configure_mlx_memory(object())


def test_ocr_rejects_image_outside_job_scratch(tmp_path: Path) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.ovisocr2 import run_ocr

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = _png(tmp_path, "outside.png")
    with pytest.raises(WorkerInputError, match="scratch"):
        run_ocr(
            {
                "page_number": 1,
                "scratch_dir": str(scratch),
                "image_path": str(outside),
                "image_sha256": _sha256(outside),
            },
            loaded=_loaded("ocr"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "",
        )


def test_ocr_rejects_image_digest_mismatch_and_unexpected_payload(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.ovisocr2 import run_ocr

    image = _png(tmp_path)
    payload: dict[str, object] = {
        "page_number": 1,
        "scratch_dir": str(tmp_path),
        "image_path": str(image),
        "image_sha256": "0" * 64,
    }
    with pytest.raises(WorkerInputError, match="digest"):
        run_ocr(
            payload,
            loaded=_loaded("ocr"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "",
        )

    payload["image_sha256"] = _sha256(image)
    payload["raw_upload"] = "forbidden"
    with pytest.raises(WorkerInputError, match="invalid"):
        run_ocr(
            payload,
            loaded=_loaded("ocr"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "",
        )


def test_scratch_image_enforces_regular_file_byte_and_dimension_bounds(
    tmp_path: Path,
) -> None:
    from local_ai_mlx_worker.common import WorkerInputError, validate_scratch_image

    image = _png(tmp_path)
    with pytest.raises(WorkerInputError, match="invalid"):
        validate_scratch_image(
            str(image),
            str(tmp_path),
            max_bytes=image.stat().st_size - 1,
        )
    with pytest.raises(WorkerInputError, match="dimension"):
        validate_scratch_image(
            str(image),
            str(tmp_path),
            max_pixels=99,
        )


def test_extraction_rejects_more_than_bounded_selected_images(tmp_path: Path) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.nuextract3 import MAX_SELECTED_IMAGES, run_extraction

    pages = []
    image_paths = {}
    for page in range(1, MAX_SELECTED_IMAGES + 2):
        path = _png(tmp_path, f"page-{page}.png")
        pages.append({"page_number": page, "markdown": f"page {page}"})
        image_paths[str(page)] = str(path)

    with pytest.raises(WorkerInputError, match="image"):
        run_extraction(
            {
                "page_markdown": pages,
                "scratch_dir": str(tmp_path),
                "image_paths": image_paths,
                "schema": {"type": "object"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "{}",
        )


def test_extraction_rejects_selected_images_over_aggregate_byte_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import nuextract3
    from local_ai_mlx_worker.common import WorkerInputError

    first = _png(tmp_path, "aggregate-bytes-1.png")
    second = _png(tmp_path, "aggregate-bytes-2.png")
    monkeypatch.setattr(
        nuextract3,
        "MAX_SELECTED_IMAGE_BYTES",
        first.stat().st_size + second.stat().st_size - 1,
    )

    with pytest.raises(WorkerInputError, match="aggregate byte"):
        nuextract3.run_extraction(
            {
                "page_markdown": [
                    {"page_number": 1, "markdown": "page one"},
                    {"page_number": 2, "markdown": "page two"},
                ],
                "scratch_dir": str(tmp_path),
                "image_paths": {"1": str(first), "2": str(second)},
                "schema": {"type": "object"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "{}",
        )


def test_extraction_rejects_selected_images_over_aggregate_pixel_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import nuextract3
    from local_ai_mlx_worker.common import WorkerInputError

    first = _png(tmp_path, "aggregate-pixels-1.png")
    second = _png(tmp_path, "aggregate-pixels-2.png")
    monkeypatch.setattr(nuextract3, "MAX_SELECTED_IMAGE_PIXELS", 199)

    with pytest.raises(WorkerInputError, match="aggregate pixel"):
        nuextract3.run_extraction(
            {
                "page_markdown": [
                    {"page_number": 1, "markdown": "page one"},
                    {"page_number": 2, "markdown": "page two"},
                ],
                "scratch_dir": str(tmp_path),
                "image_paths": {"1": str(first), "2": str(second)},
                "schema": {"type": "object"},
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "{}",
        )


def test_extraction_rejects_unexpected_raw_payload_field(tmp_path: Path) -> None:
    from local_ai_mlx_worker.common import WorkerInputError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    with pytest.raises(WorkerInputError, match="invalid"):
        run_extraction(
            {
                "page_markdown": [{"page_number": 1, "markdown": "bounded OCR"}],
                "scratch_dir": str(tmp_path),
                "image_paths": {},
                "schema": {"type": "object"},
                "raw_upload": "forbidden",
            },
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=lambda **_kwargs: "{}",
        )


def test_summary_progress_precedes_loader_and_reports_one_bounded_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summary telemetry is content-free, bounded, and has no repair call."""
    from local_ai_mlx_worker import qwen_summary
    from local_ai_mlx_worker.common import GeneratedText

    payload = _summary_payload()
    payload["max_output_tokens"] = 301
    fact = payload["facts"][0]  # type: ignore[index]
    events: list[dict[str, int | str]] = []
    calls = 0

    def loader(_role: str, _payload: object) -> object:
        assert events == [{"stage": "loading", "current": 0, "total": 1}]
        return _loaded("summary")

    def generate(**_kwargs: object) -> GeneratedText:
        nonlocal calls
        calls += 1
        return GeneratedText(
            json.dumps(
                {
                    "sections": [
                        {
                            "heading": "Medications",
                            "claims": [
                                {
                                    "fact_id": fact["fact_id"],
                                    "field_paths": ["/name"],
                                    "evidence_ids": fact["evidence_ids"],
                                }
                            ],
                        }
                    ],
                    "uncertainties": [],
                }
            ),
            generation_tokens=7,
        )

    monkeypatch.setattr(qwen_summary, "load_role_from_payload", loader)
    qwen_summary.run_summary(payload, generate_fn=generate, progress_fn=events.append)

    assert calls == 1
    assert [event["stage"] for event in events] == [
        "loading",
        "generating",
        "generating",
        "validating",
    ]
    assert events[-1]["output_tokens"] == 7
    assert events[-1]["output_token_limit"] == 301
