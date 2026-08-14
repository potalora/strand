"""Release-gate tests for content-free 16 GB Apple benchmark reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.types import ModelRole


def _runtime_manifest() -> LocalAIManifest:
    return LocalAIManifest(
        schema_version=2,
        pack_revision="apple-m4-16gb-v2",
        platform="apple_silicon",
        runtime={
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "a" * 64,
        },
        validation_suite_version="local-ai-fixtures-v1",
        artifacts=tuple(
            ManifestArtifact(
                role=role,
                repository=f"owner/{role.value}",
                revision=str(index) * 40,
                quantization="4bit",
                license="apache-2.0",
                attribution=f"https://huggingface.co/owner/{role.value}",
                decode_limits={
                    "max_input_tokens": 32768,
                    "max_output_tokens": 4096,
                },
                files=(
                    ManifestFile(
                        path="model.safetensors",
                        sha256=str(index) * 64,
                        size=index,
                    ),
                ),
            )
            for index, role in enumerate(ModelRole, start=1)
        ),
    )


def _passing_report() -> dict[str, object]:
    role_run = {
        "cold_start_seconds": 12.5,
        "duration_seconds": 18.0,
        "peak_rss_bytes": 8 * 1024**3,
        "mlx_peak_memory_bytes": 6 * 1024**3,
        "mlx_active_memory_after_bytes": 256 * 1024**2,
        "throughput_per_second": 1.25,
    }
    return {
        "schema_version": 1,
        "content_free": True,
        "machine": {
            "platform_profile": "apple_silicon",
            "model": "Mac16,1",
            "physical_memory_bytes": 16 * 1024**3,
            "os_version": "27.0",
        },
        "manifest": {
            "pack_revision": "apple-m4-16gb-v2",
            "sha256": "a" * 64,
            "runtime_name": "mlx-vlm",
            "runtime_version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "a" * 64,
        },
        "processes": {
            "max_live_models": 1,
            "roles_started": 9,
        },
        "roles": {
            role: [{**role_run, "run": run} for run in range(1, 4)]
            for role in ("ocr", "extraction", "summary")
        },
        "system": {
            "swap_before_bytes": 0,
            "swap_after_bytes": 128 * 1024**2,
            "swap_delta_bytes": 128 * 1024**2,
            "memory_pressure_termination": False,
            "sustained_swap_thrashing": False,
        },
        "reclamation": {
            "baseline_active_memory_bytes": 256 * 1024**2,
            "final_active_memory_bytes": 300 * 1024**2,
            "peak_active_memory_bytes": 6 * 1024**3,
            "final_active_memory_ratio": 0.05,
        },
    }


def test_benchmark_report_acceptance_is_strict_and_content_free() -> None:
    from scripts.benchmark_local_ai import validate_acceptance

    validate_acceptance(_passing_report(), required_runs=3)


def test_benchmark_report_requires_worker_bundle_digest() -> None:
    from scripts.benchmark_local_ai import BenchmarkGateError, validate_acceptance

    report = _passing_report()
    del report["manifest"]["worker_bundle_sha256"]

    with pytest.raises(BenchmarkGateError, match="structure"):
        validate_acceptance(report, required_runs=3)


@pytest.mark.parametrize(
    "identity_scheme",
    [None, "local-ai-worker-bundle.v2"],
)
def test_benchmark_report_requires_exact_worker_identity_scheme(
    identity_scheme: str | None,
) -> None:
    from scripts.benchmark_local_ai import BenchmarkGateError, validate_acceptance

    report = _passing_report()
    if identity_scheme is None:
        del report["manifest"]["worker_identity_scheme"]
    else:
        report["manifest"]["worker_identity_scheme"] = identity_scheme

    with pytest.raises(BenchmarkGateError, match="structure"):
        validate_acceptance(report, required_runs=3)


def test_benchmark_output_identity_contains_exact_worker_binding() -> None:
    from scripts.benchmark_local_ai import _benchmark_manifest_identity

    manifest = _runtime_manifest()

    assert _benchmark_manifest_identity(manifest) == {
        "pack_revision": "apple-m4-16gb-v2",
        "sha256": ANY,
        "runtime_name": "mlx-vlm",
        "runtime_version": "0.5.0",
        "worker_identity_scheme": "local-ai-worker-bundle.v1",
        "worker_bundle_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("machine", "physical_memory_bytes"), 15 * 1024**3, "16 GiB"),
        (("processes", "max_live_models"), 2, "one live model"),
        (("system", "memory_pressure_termination"), True, "memory pressure"),
        (("system", "sustained_swap_thrashing"), True, "swap"),
        (("reclamation", "final_active_memory_ratio"), 0.21, "reclaim"),
    ],
)
def test_benchmark_report_rejects_each_hard_resource_gate(
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    from scripts.benchmark_local_ai import BenchmarkGateError, validate_acceptance

    report = _passing_report()
    section = report[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value

    with pytest.raises(BenchmarkGateError, match=message):
        validate_acceptance(report, required_runs=3)


def test_benchmark_writer_rejects_unknown_content_bearing_fields(
    tmp_path: Path,
) -> None:
    from scripts.benchmark_local_ai import BenchmarkGateError, write_report

    report = _passing_report()
    report["ocr_text"] = "Patient Canary should never enter a benchmark"

    with pytest.raises(BenchmarkGateError, match="structure"):
        write_report(tmp_path / "report.json", report, required_runs=3)
    assert not (tmp_path / "report.json").exists()


def test_benchmark_writer_is_atomic_and_canonical(tmp_path: Path) -> None:
    from scripts.benchmark_local_ai import write_report

    output = tmp_path / "artifacts" / "local-ai-benchmark.json"
    report = _passing_report()

    write_report(output, report, required_runs=3)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert list(output.parent.glob("*.tmp")) == []


def test_sustained_swap_thrashing_requires_two_large_run_over_run_increases() -> None:
    from scripts.benchmark_local_ai import sustained_swap_thrashing

    threshold = 512 * 1024**2
    assert sustained_swap_thrashing([0, threshold, threshold + 1, threshold * 2 + 1])
    assert not sustained_swap_thrashing([0, threshold, threshold + 1])
    assert not sustained_swap_thrashing([threshold * 4, threshold * 3, threshold * 2])


def test_reclamation_uses_observed_terminal_mlx_memory() -> None:
    from scripts.benchmark_local_ai import measured_reclamation

    samples = {
        role: [
            {
                "mlx_peak_memory_bytes": 8 * 1024**3,
                "mlx_active_memory_after_bytes": 128 * 1024**2,
            }
        ]
        for role in ("ocr", "extraction", "summary")
    }

    assert measured_reclamation(
        samples,
        observed_roles={"ocr", "extraction", "summary"},
    ) == {
        "baseline_active_memory_bytes": 0,
        "final_active_memory_bytes": 128 * 1024**2,
        "peak_active_memory_bytes": 8 * 1024**3,
        "final_active_memory_ratio": 1 / 64,
    }


def test_reclamation_fails_closed_when_terminal_mlx_telemetry_is_missing() -> None:
    from scripts.benchmark_local_ai import BenchmarkGateError, measured_reclamation

    samples = {
        role: [
            {
                "mlx_peak_memory_bytes": 8 * 1024**3,
                "mlx_active_memory_after_bytes": 0,
            }
        ]
        for role in ("ocr", "extraction", "summary")
    }

    with pytest.raises(BenchmarkGateError, match="MLX telemetry"):
        measured_reclamation(samples, observed_roles={"ocr", "extraction"})


def test_unexpected_worker_exit_is_a_memory_pressure_gate_failure() -> None:
    from scripts.benchmark_local_ai import memory_pressure_termination_detected

    clean = SimpleNamespace(
        live_processes=0,
        completed_runs=3,
        failed_runs=0,
        cancelled_runs=0,
    )
    assert not memory_pressure_termination_detected(clean, expected_runs=3)

    assert memory_pressure_termination_detected(
        SimpleNamespace(
            live_processes=0,
            completed_runs=2,
            failed_runs=1,
            cancelled_runs=0,
        ),
        expected_runs=3,
    )


@pytest.mark.asyncio
async def test_benchmark_role_fails_closed_without_final_mlx_progress(
) -> None:
    import scripts.benchmark_local_ai as benchmark

    class MissingTelemetryManager:
        active_pid = None
        runtime_identity = benchmark.WorkerRuntimeIdentity(
            scheme="local-ai-worker-bundle.v1",
            bundle_sha256="a" * 64,
        )

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def run_attested(
            self,
            _manifest: object,
            _role: object,
            _payload: object,
            *,
            on_progress: object,
        ) -> object:
            assert callable(on_progress)
            return {"sections": []}

    manager = benchmark._BenchmarkingManager(manager=MissingTelemetryManager())
    await manager.start()

    with pytest.raises(benchmark.BenchmarkGateError, match="MLX telemetry"):
        await manager.run_attested(
            _runtime_manifest(),
            benchmark.ModelRole.SUMMARY,
            {"job_id": "bench-summary"},
        )


@pytest.mark.local_model
@pytest.mark.hardware
@pytest.mark.timeout(900)
@pytest.mark.asyncio
async def test_installed_pack_passes_16gb_apple_resource_gate(
    tmp_path: Path,
) -> None:
    """Run only when the operator explicitly enables the real-model gate."""
    if os.environ.get("LOCAL_AI_ENABLED", "").casefold() != "true":
        pytest.skip("set LOCAL_AI_ENABLED=true to run the installed-pack hardware gate")

    from scripts.benchmark_local_ai import run_benchmark, validate_acceptance

    report = await run_benchmark(runs=3, output=tmp_path / "benchmark.json")
    validate_acceptance(report, required_runs=3)
