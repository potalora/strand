"""Run the content-free 16 GB Apple strict-local release benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.local_ai.artifact_store import ArtifactStore, manifest_sha256
from app.services.local_ai.errors import LocalAIError
from app.services.local_ai.model_manager import LocalModelManager
from app.services.local_ai.pack_operations import PackOperationStore, platform_profile
from app.services.local_ai.pack_verifier import verify_pack_candidate
from app.services.local_ai.types import ModelRole

GIB = 1024**3
_ROLE_NAMES = tuple(role.value for role in ModelRole)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "content_free",
        "machine",
        "manifest",
        "processes",
        "roles",
        "system",
        "reclamation",
    }
)
_MACHINE_KEYS = frozenset(
    {"platform_profile", "model", "physical_memory_bytes", "os_version"}
)
_MANIFEST_KEYS = frozenset(
    {"pack_revision", "sha256", "runtime_name", "runtime_version"}
)
_PROCESS_KEYS = frozenset({"max_live_models", "roles_started"})
_ROLE_RUN_KEYS = frozenset(
    {
        "run",
        "cold_start_seconds",
        "duration_seconds",
        "peak_rss_bytes",
        "mlx_peak_memory_bytes",
        "mlx_active_memory_after_bytes",
        "throughput_per_second",
    }
)
_SYSTEM_KEYS = frozenset(
    {
        "swap_before_bytes",
        "swap_after_bytes",
        "swap_delta_bytes",
        "memory_pressure_termination",
        "sustained_swap_thrashing",
    }
)
_RECLAMATION_KEYS = frozenset(
    {
        "baseline_active_memory_bytes",
        "final_active_memory_bytes",
        "peak_active_memory_bytes",
        "final_active_memory_ratio",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SWAP_USED = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b")


class BenchmarkGateError(RuntimeError):
    """A content-free release benchmark failed a hard gate."""


def _plain_dict(
    value: object,
    keys: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise BenchmarkGateError(f"Benchmark report structure is invalid ({context}).")
    return value


def _strict_int(value: object, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BenchmarkGateError("Benchmark report structure is invalid.")
    return value


def _strict_number(value: object, *, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not float(value) >= minimum
        or not float(value) < float("inf")
    ):
        raise BenchmarkGateError("Benchmark report structure is invalid.")
    return float(value)


def _validate_structure(
    report: object,
    *,
    required_runs: int,
) -> dict[str, Any]:
    if not isinstance(required_runs, int) or isinstance(required_runs, bool):
        raise BenchmarkGateError("Benchmark run count is invalid.")
    if required_runs <= 0 or required_runs > 20:
        raise BenchmarkGateError("Benchmark run count is invalid.")
    root = _plain_dict(report, _TOP_LEVEL_KEYS, context="root")
    if root["schema_version"] != 1 or root["content_free"] is not True:
        raise BenchmarkGateError("Benchmark report structure is invalid.")

    machine = _plain_dict(root["machine"], _MACHINE_KEYS, context="machine")
    if (
        machine["platform_profile"] != "apple_silicon"
        or not isinstance(machine["model"], str)
        or not machine["model"]
        or len(machine["model"]) > 128
        or not isinstance(machine["os_version"], str)
        or not machine["os_version"]
        or len(machine["os_version"]) > 128
    ):
        raise BenchmarkGateError("Benchmark report structure is invalid.")
    _strict_int(machine["physical_memory_bytes"], minimum=1)

    manifest = _plain_dict(root["manifest"], _MANIFEST_KEYS, context="manifest")
    if (
        not all(
            isinstance(manifest[key], str)
            and bool(manifest[key])
            and len(manifest[key]) <= 256
            for key in ("pack_revision", "runtime_name", "runtime_version")
        )
        or not isinstance(manifest["sha256"], str)
        or _SHA256.fullmatch(manifest["sha256"]) is None
    ):
        raise BenchmarkGateError("Benchmark report structure is invalid.")

    processes = _plain_dict(root["processes"], _PROCESS_KEYS, context="processes")
    _strict_int(processes["max_live_models"])
    _strict_int(processes["roles_started"])

    roles = _plain_dict(root["roles"], frozenset(_ROLE_NAMES), context="roles")
    for role in _ROLE_NAMES:
        runs = roles[role]
        if not isinstance(runs, list) or len(runs) != required_runs:
            raise BenchmarkGateError("Benchmark report structure is invalid.")
        for expected_run, item in enumerate(runs, start=1):
            run = _plain_dict(item, _ROLE_RUN_KEYS, context="role run")
            if _strict_int(run["run"], minimum=1) != expected_run:
                raise BenchmarkGateError("Benchmark report structure is invalid.")
            for key in (
                "cold_start_seconds",
                "duration_seconds",
                "throughput_per_second",
            ):
                _strict_number(run[key])
            for key in (
                "peak_rss_bytes",
                "mlx_peak_memory_bytes",
                "mlx_active_memory_after_bytes",
            ):
                _strict_int(run[key])

    system = _plain_dict(root["system"], _SYSTEM_KEYS, context="system")
    for key in ("swap_before_bytes", "swap_after_bytes", "swap_delta_bytes"):
        _strict_int(system[key])
    for key in ("memory_pressure_termination", "sustained_swap_thrashing"):
        if not isinstance(system[key], bool):
            raise BenchmarkGateError("Benchmark report structure is invalid.")

    reclamation = _plain_dict(
        root["reclamation"],
        _RECLAMATION_KEYS,
        context="reclamation",
    )
    for key in (
        "baseline_active_memory_bytes",
        "final_active_memory_bytes",
        "peak_active_memory_bytes",
    ):
        _strict_int(reclamation[key])
    _strict_number(reclamation["final_active_memory_ratio"])
    return root


def validate_acceptance(report: object, *, required_runs: int) -> None:
    """Raise when a report misses any advertised 16 GB Apple hard gate."""

    root = _validate_structure(report, required_runs=required_runs)
    if root["machine"]["physical_memory_bytes"] < 16 * GIB:
        raise BenchmarkGateError("Benchmark host has less than 16 GiB of memory.")
    if root["processes"]["max_live_models"] != 1:
        raise BenchmarkGateError("Benchmark did not preserve one live model at a time.")
    if root["processes"]["roles_started"] != required_runs * len(_ROLE_NAMES):
        raise BenchmarkGateError("Benchmark did not execute every required role run.")
    if root["system"]["memory_pressure_termination"]:
        raise BenchmarkGateError("Benchmark encountered a memory pressure termination.")
    if root["system"]["sustained_swap_thrashing"]:
        raise BenchmarkGateError("Benchmark encountered sustained swap thrashing.")
    if root["reclamation"]["final_active_memory_ratio"] > 0.20:
        raise BenchmarkGateError("Benchmark did not reclaim local model memory.")
    for role in _ROLE_NAMES:
        for run in root["roles"][role]:
            if (
                run["duration_seconds"] <= 0
                or run["peak_rss_bytes"] <= 0
                or run["mlx_peak_memory_bytes"] <= 0
                or run["throughput_per_second"] <= 0
            ):
                raise BenchmarkGateError(
                    "Benchmark role resource measurements are incomplete."
                )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_report(
    output: Path,
    report: object,
    *,
    required_runs: int,
) -> None:
    """Atomically persist only the strict allowlisted, content-free report."""

    root = _validate_structure(report, required_runs=required_runs)
    destination = Path(output)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    encoded = (
        json.dumps(
            root,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = -1
    try:
        descriptor = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temp.unlink(missing_ok=True)


def _sysctl(name: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _physical_memory_bytes() -> int:
    value = _sysctl("hw.memsize")
    try:
        return int(value)
    except ValueError:
        return 0


def _swap_used_bytes() -> int:
    value = _sysctl("vm.swapusage")
    match = _SWAP_USED.search(value)
    if match is None:
        return 0
    multiplier = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def sustained_swap_thrashing(samples: list[int]) -> bool:
    """Return true only for repeated large increases across consecutive runs."""

    threshold = 512 * 1024**2
    large_increases = sum(
        current - previous >= threshold
        for previous, current in zip(samples, samples[1:])
    )
    return large_increases >= 2


def memory_pressure_termination_detected(
    metrics: object, *, expected_runs: int
) -> bool:
    """Treat any non-clean isolated-worker lifecycle as a failed pressure gate.

    ``LocalModelManager`` only increments ``completed_runs`` after it has received a
    terminal response, requested shutdown, and reaped an exit status of zero. A
    signal-killed MLX worker therefore cannot be reported as a clean run. macOS
    does not expose a trustworthy, per-process "memory pressure killed this"
    attribution API, so the benchmark intentionally treats every unexpected
    worker termination as a pressure-gate failure rather than guessing a cause.
    """

    if not isinstance(expected_runs, int) or isinstance(expected_runs, bool):
        return True
    if expected_runs <= 0:
        return True
    values = {
        name: getattr(metrics, name, None)
        for name in (
            "live_processes",
            "completed_runs",
            "failed_runs",
            "cancelled_runs",
        )
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values.values()
    ):
        return True
    return (
        values["live_processes"] != 0
        or values["completed_runs"] != expected_runs
        or values["failed_runs"] != 0
        or values["cancelled_runs"] != 0
    )


def measured_reclamation(
    role_samples: Mapping[str, list[Mapping[str, object]]],
    *,
    observed_roles: set[str],
) -> dict[str, int | float]:
    """Build reclamation metrics from final MLX allocator counters only.

    The worker emits the counter after role dispatch returns, when the
    role-local model reference has been released. A missing terminal counter is
    not interchangeable with a measured zero, so release benchmarking fails
    closed instead of presenting an optimistic reclamation result.
    """

    expected_roles = set(_ROLE_NAMES)
    if observed_roles != expected_roles:
        raise BenchmarkGateError("Benchmark MLX telemetry is unavailable.")
    if set(role_samples) != expected_roles:
        raise BenchmarkGateError("Benchmark role measurements are incomplete.")

    active_samples: list[int] = []
    peak_samples: list[int] = []
    for role in _ROLE_NAMES:
        samples = role_samples[role]
        if not samples:
            raise BenchmarkGateError("Benchmark role measurements are incomplete.")
        for sample in samples:
            active = sample.get("mlx_active_memory_after_bytes")
            peak = sample.get("mlx_peak_memory_bytes")
            if (
                not isinstance(active, int)
                or isinstance(active, bool)
                or active < 0
                or not isinstance(peak, int)
                or isinstance(peak, bool)
                or peak < 0
            ):
                raise BenchmarkGateError("Benchmark MLX telemetry is unavailable.")
            active_samples.append(active)
            peak_samples.append(peak)

    peak_active = max(peak_samples, default=0)
    if peak_active <= 0:
        raise BenchmarkGateError("Benchmark MLX telemetry is unavailable.")
    final_active = max(active_samples, default=0)
    return {
        # There is no MLX allocator in the benchmark parent process. Each
        # model worker is fresh and isolated, so its measured terminal counter
        # is the conservative post-dispatch value used for reclamation.
        "baseline_active_memory_bytes": 0,
        "final_active_memory_bytes": final_active,
        "peak_active_memory_bytes": peak_active,
        "final_active_memory_ratio": final_active / peak_active,
    }


def _rss_bytes(pid: int) -> int:
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return int(result.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def _work_units(role: ModelRole, result: object) -> int:
    if role is ModelRole.OCR and isinstance(result, dict):
        markdown = result.get("markdown")
        return max(len(markdown), 1) if isinstance(markdown, str) else 1
    if role is ModelRole.EXTRACTION and isinstance(result, dict):
        count = sum(len(value) for value in result.values() if isinstance(value, list))
        return max(count, 1)
    if role is ModelRole.SUMMARY and isinstance(result, dict):
        sections = result.get("sections")
        if isinstance(sections, list):
            count = sum(
                len(section.get("claims", []))
                for section in sections
                if isinstance(section, dict)
                and isinstance(section.get("claims", []), list)
            )
            return max(count, 1)
    return 1


class _BenchmarkingManager:
    """Wrap the production manager with non-content process/resource sampling."""

    def __init__(self) -> None:
        self.manager = LocalModelManager()
        self.samples: dict[str, dict[str, int | float]] = {}
        self.observed_terminal_mlx_roles: set[str] = set()

    async def start(self) -> None:
        await self.manager.start()

    async def stop(self) -> None:
        await self.manager.stop()

    async def run(self, role: ModelRole, payload: dict[str, Any]) -> Any:
        started = time.monotonic()
        first_process_seen: float | None = None
        peak_rss = 0
        mlx_active = 0
        mlx_peak = 0

        def on_progress(progress: dict[str, object]) -> None:
            nonlocal mlx_active, mlx_peak
            active = progress.get("active_memory_bytes")
            peak = progress.get("peak_memory_bytes")
            if (
                progress.get("stage") == "finalizing"
                and isinstance(active, int)
                and not isinstance(active, bool)
                and active >= 0
                and isinstance(peak, int)
                and not isinstance(peak, bool)
                and peak >= 0
            ):
                # This frame is emitted after role dispatch returns, not while
                # generation still owns a model reference.
                mlx_active = active
                mlx_peak = max(mlx_peak, peak)
                self.observed_terminal_mlx_roles.add(role.value)

        run_task = asyncio.create_task(
            self.manager.run(role, payload, on_progress=on_progress)
        )
        try:
            while not run_task.done():
                pid = self.manager.active_pid
                if pid is not None:
                    if first_process_seen is None:
                        first_process_seen = time.monotonic()
                    peak_rss = max(
                        peak_rss,
                        await asyncio.to_thread(_rss_bytes, pid),
                    )
                await asyncio.sleep(0.05)
            result = await run_task
        finally:
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
        if role.value not in self.observed_terminal_mlx_roles:
            raise BenchmarkGateError("Benchmark MLX telemetry is unavailable.")
        duration = time.monotonic() - started
        self.samples[role.value] = {
            "cold_start_seconds": max(
                (first_process_seen or time.monotonic()) - started,
                0.0,
            ),
            "duration_seconds": duration,
            "peak_rss_bytes": peak_rss,
            "mlx_peak_memory_bytes": mlx_peak,
            "mlx_active_memory_after_bytes": mlx_active,
            "throughput_per_second": _work_units(role, result) / max(duration, 1e-9),
        }
        return result


async def run_benchmark(
    *,
    runs: int,
    output: Path | None = None,
) -> dict[str, Any]:
    """Execute validated cold role runs and return a content-free report."""

    if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0 or runs > 20:
        raise BenchmarkGateError("Benchmark run count is invalid.")
    profile, compatible = platform_profile()
    memory_bytes = _physical_memory_bytes()
    if profile != "apple_silicon" or not compatible or memory_bytes < 16 * GIB:
        raise BenchmarkGateError(
            "Benchmark requires Apple Silicon with at least 16 GiB of memory."
        )

    store = ArtifactStore(Path(settings.local_ai_model_dir))
    operations = PackOperationStore(store)
    try:
        manifest = store.active_manifest()
    except LocalAIError as exc:
        raise BenchmarkGateError("Validated local model pack is unavailable.") from exc
    if manifest is None or not operations.is_validated(manifest):
        raise BenchmarkGateError("Validated local model pack is unavailable.")
    pack_path = store.packs_dir / manifest.pack_revision

    swap_samples = [_swap_used_bytes()]
    role_samples: dict[str, list[dict[str, int | float]]] = {
        role: [] for role in _ROLE_NAMES
    }
    max_live_models = 0
    roles_started = 0
    observed_terminal_mlx_roles: set[str] = set()
    memory_pressure_termination = False
    for run_number in range(1, runs + 1):
        manager = _BenchmarkingManager()
        await verify_pack_candidate(
            manifest,
            pack_path,
            manager=manager,
        )
        max_live_models = max(
            max_live_models,
            manager.manager.metrics.max_live_processes,
        )
        roles_started += len(manager.manager.metrics.roles_started)
        memory_pressure_termination = (
            memory_pressure_termination
            or memory_pressure_termination_detected(
                manager.manager.metrics,
                expected_runs=len(_ROLE_NAMES),
            )
        )
        if memory_pressure_termination:
            raise BenchmarkGateError(
                "Benchmark encountered a memory pressure termination."
            )
        observed_terminal_mlx_roles.update(manager.observed_terminal_mlx_roles)
        for role in _ROLE_NAMES:
            sample = manager.samples.get(role)
            if sample is None:
                raise BenchmarkGateError("Benchmark role measurements are incomplete.")
            role_samples[role].append({"run": run_number, **sample})
        swap_samples.append(_swap_used_bytes())

    swap_before = swap_samples[0]
    swap_after = swap_samples[-1]
    swap_delta = max(swap_after - swap_before, 0)
    reclamation = measured_reclamation(
        role_samples,
        observed_roles=observed_terminal_mlx_roles,
    )
    runtime = manifest.runtime
    report: dict[str, Any] = {
        "schema_version": 1,
        "content_free": True,
        "machine": {
            "platform_profile": profile,
            "model": _sysctl("hw.model") or "unknown-apple-silicon",
            "physical_memory_bytes": memory_bytes,
            "os_version": platform.mac_ver()[0] or platform.release(),
        },
        "manifest": {
            "pack_revision": manifest.pack_revision,
            "sha256": manifest_sha256(manifest),
            "runtime_name": runtime["name"],
            "runtime_version": runtime["version"],
        },
        "processes": {
            "max_live_models": max_live_models,
            "roles_started": roles_started,
        },
        "roles": role_samples,
        "system": {
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
            "swap_delta_bytes": swap_delta,
            "memory_pressure_termination": memory_pressure_termination,
            "sustained_swap_thrashing": sustained_swap_thrashing(swap_samples),
        },
        "reclamation": reclamation,
    }
    _validate_structure(report, required_runs=runs)
    # Never persist a report that could be mistaken for a passing release gate.
    validate_acceptance(report, required_runs=runs)
    if output is not None:
        write_report(Path(output), report, required_runs=runs)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the installed validated strict-local Apple model pack.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local-ai-benchmark.json"),
    )
    return parser


async def _execute(args: argparse.Namespace) -> int:
    try:
        report = await run_benchmark(runs=args.runs, output=args.output)
        validate_acceptance(report, required_runs=args.runs)
    except BenchmarkGateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Local AI benchmark passed: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI."""

    return asyncio.run(_execute(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
