"""Create source-controlled release evidence from an accepted benchmark report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.artifact_store import manifest_sha256
from app.services.local_ai.fidelity_metrics import FidelityGateError
from app.services.local_ai.fidelity_runner import parse_fidelity_report_bytes
from app.services.local_ai.manifest import load_manifest
from app.services.local_ai.release_evidence import (
    _read_small_regular,
    build_release_evidence,
    load_release_evidence,
)
from scripts.benchmark_local_ai import BenchmarkGateError, validate_acceptance


def promote(
    *,
    manifest_path: Path,
    benchmark_path: Path,
    fidelity_path: Path,
    output: Path,
) -> None:
    """Bind accepted benchmark and fidelity artifacts to one exact local pack."""

    try:
        manifest = load_manifest(manifest_path)
        benchmark_bytes = _read_small_regular(benchmark_path)
        fidelity_bytes = _read_small_regular(fidelity_path)
        benchmark = json.loads(benchmark_bytes.decode("utf-8"))
        fidelity = parse_fidelity_report_bytes(fidelity_bytes)
        fidelity.assert_release_thresholds()
        validate_acceptance(benchmark, required_runs=3)
        report_manifest = benchmark.get("manifest") if type(benchmark) is dict else None
        if (
            type(report_manifest) is not dict
            or report_manifest.get("sha256") != manifest_sha256(manifest)
            or report_manifest.get("runtime_name") != manifest.runtime["name"]
            or report_manifest.get("runtime_version") != manifest.runtime["version"]
        ):
            raise LocalValidationError("Local model release evidence is invalid")
        evidence = build_release_evidence(manifest, benchmark, fidelity)
        evidence["benchmark_sha256"] = hashlib.sha256(benchmark_bytes).hexdigest()
        evidence["fidelity_sha256"] = hashlib.sha256(fidelity_bytes).hexdigest()
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        LocalValidationError,
        RecursionError,
        ValueError,
        BenchmarkGateError,
        FidelityGateError,
    ) as exc:
        raise LocalValidationError("Local model release promotion failed") from exc
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            json.dump(evidence, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_release_evidence(
            temporary,
            manifest=manifest,
            benchmark_path=benchmark_path,
            fidelity_path=fidelity_path,
        )
        os.replace(temporary, output)
    except LocalValidationError as exc:
        raise LocalValidationError("Local model release promotion failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote accepted local-AI benchmark and fidelity evidence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        promote(
            manifest_path=args.manifest,
            benchmark_path=args.benchmark,
            fidelity_path=args.fidelity,
            output=args.output,
        )
    except LocalValidationError:
        print("ERROR: local model release promotion failed.", file=sys.stderr)
        return 1
    print(f"Local model release evidence written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
