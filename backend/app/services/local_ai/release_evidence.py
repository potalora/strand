"""Verify the source-controlled evidence that promotes a local model pack."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.local_ai.artifact_store import manifest_sha256
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.fidelity_metrics import (
    FidelityGateError,
    release_thresholds,
)
from app.services.local_ai.fidelity_runner import (
    RELEASE_SYNTHETIC_DOCUMENT_COUNT,
    FidelityRunReport,
    parse_fidelity_report,
    parse_fidelity_report_bytes,
)
from app.services.local_ai.manifest import LocalAIManifest
from app.services.local_ai.types import ModelRole

GIB = 1024**3
MIB = 1024**2
_KEYS = frozenset(
    {
        "schema_version",
        "manifest_sha256",
        "benchmark_sha256",
        "fidelity_sha256",
        "fidelity",
        "roles",
    }
)
_FIDELITY_KEYS = frozenset(
    {"fixture_suite_version", "fixture_suite_sha256", "required_thresholds"}
)
_ROLE_NAMES = frozenset(role.value for role in ModelRole)
_MAX_EVIDENCE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReleaseEvidence:
    """Content-free release metadata bound to benchmark and fidelity artifacts."""

    manifest_sha256: str
    benchmark_sha256: str
    fidelity_sha256: str
    fidelity_suite_version: str
    fidelity_corpus_sha256: str
    fidelity_thresholds: dict[str, int | float]
    roles: dict[str, int]

    def expected_memory_bytes(self, role: ModelRole | str) -> int:
        """Return the release-measured resident memory for one model role."""

        key = role.value if isinstance(role, ModelRole) else role
        return self.roles[key]


def _read_small_regular(path: Path) -> bytes:
    """Read one bounded regular file without following symlinks."""

    descriptor = -1
    try:
        path_status = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or path_status.st_size > _MAX_EVIDENCE_BYTES
        ):
            raise OSError("unsafe evidence")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or opened_status.st_size > _MAX_EVIDENCE_BYTES
        ):
            raise OSError("unsafe evidence")
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _MAX_EVIDENCE_BYTES + 1 - bytes_read),
            )
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _MAX_EVIDENCE_BYTES:
                raise OSError("oversize evidence")
        final_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final_status.st_mode)
            or final_status.st_size != opened_status.st_size
            or bytes_read != final_status.st_size
        ):
            raise OSError("changed evidence")
        value = b"".join(chunks)
        if len(value) > _MAX_EVIDENCE_BYTES:
            raise OSError("oversize evidence")
        return value
    except OSError as exc:
        raise LocalValidationError(
            "Local model release evidence is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def expected_memory_bytes(runs: list[dict[str, Any]]) -> int:
    """Return ceil(110% of maximum RSS/MLX peak, rounded to 64 MiB."""

    peak = 0
    for run in runs:
        if type(run) is not dict:
            raise LocalValidationError("Local model release evidence is invalid")
        rss = run.get("peak_rss_bytes")
        mlx = run.get("mlx_peak_memory_bytes")
        if (
            not isinstance(rss, int)
            or isinstance(rss, bool)
            or rss < 0
            or not isinstance(mlx, int)
            or isinstance(mlx, bool)
            or mlx < 0
        ):
            raise LocalValidationError("Local model release evidence is invalid")
        peak = max(peak, rss, mlx)
    if peak <= 0:
        raise LocalValidationError("Local model release evidence is invalid")
    expected = math.ceil((peak * 1.10) / (64 * MIB)) * 64 * MIB
    if expected > 16 * GIB:
        raise LocalValidationError("Local model release evidence is invalid")
    return expected


def build_release_evidence(
    manifest: LocalAIManifest,
    benchmark: object,
    fidelity: FidelityRunReport | object,
) -> dict[str, object]:
    """Build canonical evidence after benchmark and fidelity gates pass."""

    if type(benchmark) is not dict or type(benchmark.get("roles")) is not dict:
        raise LocalValidationError("Local model release evidence is invalid")
    report = (
        fidelity
        if isinstance(fidelity, FidelityRunReport)
        else parse_fidelity_report(fidelity)
    )
    report.assert_release_thresholds()
    if (
        report.content_free is not True
        or report.synthetic_documents != RELEASE_SYNTHETIC_DOCUMENT_COUNT
        or report.private_documents != 0
        or report.private_metrics is not None
    ):
        raise LocalValidationError("Local model release evidence is invalid")
    digest = manifest_sha256(manifest)
    if report.manifest_sha256 != digest:
        raise LocalValidationError("Local model release evidence is invalid")
    roles = benchmark["roles"]
    if set(roles) != _ROLE_NAMES:
        raise LocalValidationError("Local model release evidence is invalid")
    return {
        "schema_version": 2,
        "manifest_sha256": digest,
        "benchmark_sha256": "",  # populated only after canonical benchmark bytes exist
        "fidelity_sha256": "",  # populated only after exact fidelity bytes exist
        "fidelity": {
            "fixture_suite_version": report.fixture_suite_version,
            "fixture_suite_sha256": report.fixture_suite_sha256,
            "required_thresholds": release_thresholds(),
        },
        "roles": {
            role: expected_memory_bytes(runs) for role, runs in sorted(roles.items())
        },
    }


def load_release_evidence(
    path: Path,
    *,
    manifest: LocalAIManifest,
    benchmark_path: Path,
    fidelity_path: Path,
) -> ReleaseEvidence:
    """Load exact, bounded benchmark and fidelity evidence."""

    from scripts.benchmark_local_ai import BenchmarkGateError, validate_acceptance

    try:
        evidence_bytes = _read_small_regular(path)
        benchmark_bytes = _read_small_regular(benchmark_path)
        fidelity_bytes = _read_small_regular(fidelity_path)
        raw = json.loads(evidence_bytes.decode("utf-8"))
        benchmark = json.loads(benchmark_bytes.decode("utf-8"))
        fidelity = parse_fidelity_report_bytes(fidelity_bytes)
        fidelity.assert_release_thresholds()
        validate_acceptance(benchmark, required_runs=3)
    except (
        LocalValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        BenchmarkGateError,
        FidelityGateError,
    ) as exc:
        raise LocalValidationError(
            "Local model release evidence is unavailable"
        ) from exc
    if type(raw) is not dict or set(raw) != _KEYS or raw.get("schema_version") != 2:
        raise LocalValidationError("Local model release evidence is invalid")
    manifest_digest = raw.get("manifest_sha256")
    benchmark_digest = raw.get("benchmark_sha256")
    fidelity_digest = raw.get("fidelity_sha256")
    fidelity_identity = raw.get("fidelity")
    roles = raw.get("roles")
    if (
        not isinstance(manifest_digest, str)
        or not isinstance(benchmark_digest, str)
        or not isinstance(fidelity_digest, str)
        or len(manifest_digest) != 64
        or len(benchmark_digest) != 64
        or len(fidelity_digest) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in manifest_digest + benchmark_digest + fidelity_digest
        )
        or type(fidelity_identity) is not dict
        or set(fidelity_identity) != _FIDELITY_KEYS
        or type(roles) is not dict
        or set(roles) != _ROLE_NAMES
        or any(
            type(value) is not int or value <= 0 or value > 16 * GIB
            for value in roles.values()
        )
        or manifest_digest != manifest_sha256(manifest)
        or benchmark_digest != hashlib.sha256(benchmark_bytes).hexdigest()
        or fidelity_digest != hashlib.sha256(fidelity_bytes).hexdigest()
    ):
        raise LocalValidationError("Local model release evidence is invalid")
    report_manifest = benchmark.get("manifest") if type(benchmark) is dict else None
    if (
        type(report_manifest) is not dict
        or report_manifest.get("sha256") != manifest_digest
        or report_manifest.get("runtime_name") != manifest.runtime["name"]
        or report_manifest.get("runtime_version") != manifest.runtime["version"]
        or report_manifest.get("worker_identity_scheme")
        != manifest.runtime["worker_identity_scheme"]
        or report_manifest.get("worker_bundle_sha256")
        != manifest.runtime["worker_bundle_sha256"]
        or fidelity.manifest_sha256 != manifest_digest
        or fidelity_identity.get("fixture_suite_version")
        != fidelity.fixture_suite_version
        or fidelity_identity.get("fixture_suite_sha256")
        != fidelity.fixture_suite_sha256
        or fidelity_identity.get("required_thresholds") != release_thresholds()
        or build_release_evidence(manifest, benchmark, fidelity)["roles"] != roles
    ):
        raise LocalValidationError("Local model release evidence is invalid")
    return ReleaseEvidence(
        manifest_sha256=manifest_digest,
        benchmark_sha256=benchmark_digest,
        fidelity_sha256=fidelity_digest,
        fidelity_suite_version=fidelity.fixture_suite_version,
        fidelity_corpus_sha256=fidelity.fixture_suite_sha256,
        fidelity_thresholds=release_thresholds(),
        roles=dict(roles),
    )
