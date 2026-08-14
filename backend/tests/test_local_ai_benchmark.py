"""Deterministic worker-attestation boundaries for the local benchmark."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from app.services.local_ai.errors import LocalValidationError, LocalWorkerError
from app.services.local_ai.runtime_identity import WorkerRuntimeIdentity
from app.services.local_ai.types import ModelRole
from tests.test_local_ai_pack_verifier import _manifest


class _AttestedBenchmarkWorker:
    active_pid = None

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.metrics = type(
            "Metrics",
            (),
            {
                "max_live_processes": 0,
                "roles_started": [],
            },
        )()
        self.runtime_identity = WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="a" * 64,
        )
        self.calls: list[tuple[object, ModelRole, dict[str, Any]]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run_attested(
        self,
        manifest: object,
        role: ModelRole,
        payload: dict[str, Any],
        *,
        on_progress: object,
    ) -> object:
        self.calls.append((manifest, role, payload))
        if self.fail:
            raise LocalWorkerError("secret path /private/worker escaped")
        assert callable(on_progress)
        on_progress(
            {
                "stage": "finalizing",
                "active_memory_bytes": 0,
                "peak_memory_bytes": 1,
            }
        )
        return {"sections": []}


@pytest.mark.asyncio
async def test_benchmark_wrapper_forwards_exact_manifest_to_attested_worker() -> None:
    import scripts.benchmark_local_ai as benchmark

    worker = _AttestedBenchmarkWorker()
    manager = benchmark._BenchmarkingManager(manager=worker)
    manifest = _manifest()

    await manager.run_attested(
        manifest,
        ModelRole.SUMMARY,
        {"job_id": "bench-summary"},
    )

    assert manager.runtime_identity == worker.runtime_identity
    assert worker.calls == [(manifest, ModelRole.SUMMARY, {"job_id": "bench-summary"})]


@pytest.mark.asyncio
async def test_benchmark_wrapper_translates_attested_worker_failure_without_leak() -> (
    None
):
    import scripts.benchmark_local_ai as benchmark

    worker = _AttestedBenchmarkWorker(fail=True)
    manager = benchmark._BenchmarkingManager(manager=worker)

    with pytest.raises(benchmark.BenchmarkGateError) as captured:
        await manager.run_attested(
            _manifest(),
            ModelRole.SUMMARY,
            {"job_id": "bench-summary"},
        )

    assert "/private/worker" not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.asyncio
async def test_benchmark_candidate_failure_is_bounded_before_any_role_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import scripts.benchmark_local_ai as benchmark

    manifest = _manifest()

    class _Candidate:
        def __init__(self) -> None:
            self.pack_path = tmp_path
            self.manifest = manifest

        def revalidate(self) -> None:
            raise AssertionError("failed model gate must not report success")

    async def reject_candidate(*_args, **_kwargs) -> None:
        raise LocalValidationError("secret path /private/candidate escaped")

    class _Manager:
        runtime_identity = WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="a" * 64,
        )

    monkeypatch.setattr(benchmark, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(benchmark, "_physical_memory_bytes", lambda: 16 * benchmark.GIB)
    monkeypatch.setattr(
        benchmark,
        "resolve_retained_candidate_pack",
        lambda **_kwargs: _Candidate(),
    )
    monkeypatch.setattr(benchmark, "_BenchmarkingManager", _Manager)
    monkeypatch.setattr(benchmark, "verify_pack_candidate", reject_candidate)

    with pytest.raises(benchmark.BenchmarkGateError) as captured:
        await benchmark.run_benchmark(runs=1)

    assert "/private/candidate" not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    ("worker_command", "worker_project_dir"),
    [
        ("'unterminated /private/command", "/private/project"),
        ("/usr/bin/python3", "/private/missing-worker-project"),
    ],
)
@pytest.mark.asyncio
async def test_benchmark_execute_bounds_invalid_runtime_configuration(
    worker_command: str,
    worker_project_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import scripts.benchmark_local_ai as benchmark

    monkeypatch.setattr(
        benchmark.settings,
        "local_ai_worker_command",
        worker_command,
    )
    monkeypatch.setattr(
        benchmark.settings,
        "local_ai_worker_project_dir",
        worker_project_dir,
    )

    async def construct_manager(*, runs: int, output: Path) -> dict[str, Any]:
        del runs, output
        benchmark._BenchmarkingManager()
        raise AssertionError("invalid runtime configuration was accepted")

    monkeypatch.setattr(benchmark, "run_benchmark", construct_manager)

    result = await benchmark._execute(
        Namespace(runs=1, output=tmp_path / "benchmark.json")
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "ERROR: Benchmark local worker is unavailable.\n"
    assert "/private/" not in captured.err


@pytest.mark.asyncio
async def test_benchmark_execute_bounds_pack_store_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import scripts.benchmark_local_ai as benchmark

    def reject_private_path(*_args: object, **_kwargs: object) -> object:
        raise LocalValidationError("private path /private/model-pack escaped")

    monkeypatch.setattr(benchmark, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(benchmark, "_physical_memory_bytes", lambda: 16 * benchmark.GIB)
    monkeypatch.setattr(
        benchmark,
        "resolve_retained_candidate_pack",
        reject_private_path,
    )

    with pytest.raises(benchmark.BenchmarkGateError) as captured_error:
        await benchmark.run_benchmark(runs=1)

    assert str(captured_error.value) == "Validated local model pack is unavailable."
    assert captured_error.value.__cause__ is None
    assert "/private/" not in str(captured_error.value)

    result = await benchmark._execute(
        Namespace(runs=1, output=tmp_path / "benchmark.json")
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "ERROR: Validated local model pack is unavailable.\n"
    assert "/private/" not in captured.err


@pytest.mark.asyncio
async def test_benchmark_report_binds_v2_manifest_over_retained_v1_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import scripts.benchmark_local_ai as benchmark

    manifest = _manifest()

    class _Candidate:
        def __init__(self) -> None:
            self.pack_path = tmp_path / "packs" / "apple-m4-16gb-v1"
            self.revalidated = False
            self.manifest = manifest

        def revalidate(self) -> None:
            self.revalidated = True

    candidate = _Candidate()

    class _Metrics:
        max_live_processes = 1
        roles_started = ["ocr", "extraction", "summary"]

    class _Manager:
        def __init__(self) -> None:
            self.manager = type("Worker", (), {"metrics": _Metrics()})()
            self.samples: dict[str, dict[str, int | float]] = {}
            self.observed_terminal_mlx_roles = set(benchmark._ROLE_NAMES)

    async def verify(
        observed_manifest: object,
        observed_path: Path,
        *,
        manager: _Manager,
    ) -> None:
        assert observed_manifest == manifest
        assert observed_path == candidate.pack_path
        manager.samples = {
            role: {
                "cold_start_seconds": 0.1,
                "duration_seconds": 1.0,
                "peak_rss_bytes": 1,
                "mlx_peak_memory_bytes": 1,
                "mlx_active_memory_after_bytes": 0,
                "throughput_per_second": 1.0,
            }
            for role in benchmark._ROLE_NAMES
        }

    monkeypatch.setattr(benchmark, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(benchmark, "_physical_memory_bytes", lambda: 16 * benchmark.GIB)
    monkeypatch.setattr(
        benchmark,
        "resolve_retained_candidate_pack",
        lambda **_kwargs: candidate,
    )
    monkeypatch.setattr(benchmark, "_BenchmarkingManager", _Manager)
    monkeypatch.setattr(benchmark, "verify_pack_candidate", verify)
    monkeypatch.setattr(benchmark, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(
        benchmark,
        "memory_pressure_termination_detected",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        benchmark,
        "measured_reclamation",
        lambda *_args, **_kwargs: {
            "baseline_active_memory_bytes": 0,
            "final_active_memory_bytes": 0,
            "peak_active_memory_bytes": 1,
            "final_active_memory_ratio": 0.0,
        },
    )
    monkeypatch.setattr(benchmark, "sustained_swap_thrashing", lambda _values: False)
    monkeypatch.setattr(benchmark, "validate_acceptance", lambda *_a, **_k: None)

    report = await benchmark.run_benchmark(runs=1)

    assert report["manifest"]["pack_revision"] == "apple-m4-16gb-v2"
    assert report["manifest"]["sha256"] == benchmark.manifest_sha256(manifest)
    assert candidate.revalidated is True
