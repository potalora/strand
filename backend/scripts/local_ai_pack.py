"""Install, verify, or remove the optional validated local model pack."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.local_ai import (
    _run_globally_locked_pack_mutation,
    run_operation,
)
from app.config import settings
from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalAIError, LocalValidationError
from app.services.local_ai.manifest import LocalAIManifest, load_manifest
from app.services.local_ai.pack_operations import (
    PackOperationStore,
    platform_profile,
)
from app.services.local_ai.release_evidence import load_release_evidence


def _load_locked_manifest() -> LocalAIManifest:
    """Load the release-generated immutable manifest selected by configuration."""

    return load_manifest(Path(settings.local_ai_manifest_path))


def _load_released_manifest() -> LocalAIManifest:
    """Require detached benchmark and fidelity evidence for lifecycle actions."""

    manifest = _load_locked_manifest()
    load_release_evidence(
        Path(settings.local_ai_release_evidence_path),
        manifest=manifest,
        benchmark_path=Path(settings.local_ai_benchmark_path),
        fidelity_path=Path(settings.local_ai_fidelity_path),
    )
    return manifest


def _write(stream: TextIO, message: str) -> None:
    stream.write(f"{message}\n")
    stream.flush()


def _write_candidate_only_message(stream: TextIO) -> None:
    """Explain why a catalog-only checkout cannot install a strict-local pack."""

    _write(
        stream,
        (
            "ERROR: no validated locked local model pack is shipped; "
            "the candidate catalog cannot be installed."
        ),
    )


def _validated_active_pack(
    store: ArtifactStore,
    operations: PackOperationStore,
    manifest: LocalAIManifest,
) -> bool:
    try:
        active = store.active_manifest()
    except LocalAIError:
        return False
    return active == manifest and operations.is_validated(manifest)


async def _run_lifecycle(
    action: str,
    *,
    stdout: TextIO,
    stderr: TextIO,
    require_release: bool = True,
) -> int:
    platform_name, compatible = platform_profile()
    if platform_name != "apple_silicon" or not compatible:
        _write(
            stderr,
            "ERROR: the validated pack requires Apple Silicon with at least 16 GB.",
        )
        return 1

    try:
        manifest = (
            _load_released_manifest() if require_release else _load_locked_manifest()
        )
    except LocalAIError:
        _write_candidate_only_message(stderr)
        return 1

    store = ArtifactStore(Path(settings.local_ai_model_dir))
    operations = PackOperationStore(store)
    operations.reconcile_interrupted()
    try:
        active_revision = store.active_revision()
    except LocalAIError:
        _write(stderr, "ERROR: local model pack state is invalid.")
        return 1

    if action == "install" and active_revision is not None:
        _write(stderr, "ERROR: a local model pack is already installed.")
        return 1
    if action == "verify" and active_revision is None:
        _write(stderr, "ERROR: no local model pack is installed.")
        return 1

    try:
        operation, lease = operations.create_claimed(
            action=action,  # type: ignore[arg-type]
            manifest=manifest,
        )
    except LocalValidationError:
        _write(stderr, "ERROR: a model pack lifecycle operation is already active.")
        return 1

    try:
        await run_operation(operation["id"], None, lease)
    except Exception:
        _write(
            stderr,
            (
                "ERROR: local model pack installation failed."
                if action == "install"
                else "ERROR: local model pack verification failed."
            ),
        )
        return 1
    finally:
        lease.release()

    final = operations.get(operation["id"])
    if (
        final is None
        or final["state"] != "completed"
        or not _validated_active_pack(store, operations, manifest)
    ):
        _write(
            stderr,
            (
                "ERROR: local model pack installation failed."
                if action == "install"
                else "ERROR: local model pack verification failed."
            ),
        )
        return 1

    if require_release:
        verb = "installed and validated" if action == "install" else "verified"
    else:
        verb = (
            "installed and runtime-verified for benchmarking"
            if action == "install"
            else "runtime-verified for benchmarking"
        )
    _write(stdout, f"Local model pack {verb}: {manifest.pack_revision}")
    return 0


async def _remove(*, stdout: TextIO, stderr: TextIO) -> int:
    store = ArtifactStore(Path(settings.local_ai_model_dir))
    operations = PackOperationStore(store)

    def mutation() -> None:
        with operations.lifecycle_guard():
            if operations.has_nonterminal_locked():
                raise LocalValidationError(
                    "A model pack lifecycle operation is already active"
                )
            store.remove()

    try:
        await _run_globally_locked_pack_mutation(mutation)
    except Exception:
        _write(stderr, "ERROR: local model pack removal failed.")
        return 1
    _write(stdout, "Local model pack removed.")
    return 0


async def _preflight(*, stdout: TextIO, stderr: TextIO) -> int:
    """Check whether this checkout contains a release-generated model lock."""

    try:
        _load_released_manifest()
    except LocalAIError:
        _write_candidate_only_message(stderr)
        return 1
    _write(stdout, "Validated local model pack lock is available.")
    return 0


async def execute(
    action: str,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Execute one content-free operator action and return a process exit code."""

    selected_stdout = stdout or sys.stdout
    selected_stderr = stderr or sys.stderr
    if action in {"install", "verify"}:
        return await _run_lifecycle(
            action,
            stdout=selected_stdout,
            stderr=selected_stderr,
        )
    if action == "remove":
        return await _remove(stdout=selected_stdout, stderr=selected_stderr)
    if action == "preflight":
        return await _preflight(stdout=selected_stdout, stderr=selected_stderr)
    _write(selected_stderr, "ERROR: unsupported local model pack action.")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the optional validated local model pack.",
    )
    parser.add_argument("action", choices=("install", "verify", "remove", "preflight"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    args = _parser().parse_args(argv)
    try:
        return asyncio.run(execute(args.action))
    except KeyboardInterrupt:
        _write(sys.stderr, "ERROR: local model pack command interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
