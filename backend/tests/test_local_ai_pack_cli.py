"""Operator CLI tests for the optional validated local model pack."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.pack_operations import PackOperationStore
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import (
    _issue_runtime_validation_receipt,
)

import scripts.local_ai_pack as pack_cli


def test_script_entrypoint_imports_backend_package() -> None:
    backend_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/local_ai_pack.py", "--help"],
        cwd=backend_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    assert "install" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.asyncio
async def test_preflight_explains_that_a_candidate_catalog_is_not_installable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators get a deterministic message before installing an unavailable pack."""

    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(
        pack_cli,
        "_load_locked_manifest",
        lambda: (_ for _ in ()).throw(LocalValidationError("missing lock")),
    )

    result = await pack_cli.execute("preflight", stdout=stdout, stderr=stderr)

    assert result == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "ERROR: no validated locked local model pack is shipped; "
        "the candidate catalog cannot be installed.\n"
    )


@pytest.mark.asyncio
async def test_install_reports_candidate_only_when_no_lock_is_shipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal task-runner install path gives the same deterministic preflight."""

    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(
        pack_cli,
        "_load_locked_manifest",
        lambda: (_ for _ in ()).throw(LocalValidationError("missing lock")),
    )

    result = await pack_cli.execute("install", stdout=stdout, stderr=stderr)

    assert result == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == (
        "ERROR: no validated locked local model pack is shipped; "
        "the candidate catalog cannot be installed.\n"
    )


def _manifest() -> LocalAIManifest:
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository={
                ModelRole.OCR: "sahilchachra/ovisocr2-int4-mlx",
                ModelRole.EXTRACTION: "numind/NuExtract3-mlx-4bits",
                ModelRole.SUMMARY: "mlx-community/Qwen3.5-9B-MLX-4bit",
            }[role],
            revision=str(index) * 40,
            quantization="int4" if role is ModelRole.OCR else "4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/model/{role.value}",
            decode_limits={"max_input_tokens": 32768, "max_output_tokens": 4096},
            files=(
                ManifestFile(
                    path="model.safetensors",
                    sha256=hashlib.sha256(role.value.encode()).hexdigest(),
                    size=len(role.value),
                ),
            ),
        )
        for index, role in enumerate(ModelRole, start=1)
    )
    return LocalAIManifest(
        schema_version=1,
        pack_revision="apple-m4-16gb-v1",
        platform="apple_silicon",
        runtime={"name": "mlx-vlm", "version": "0.5.0"},
        validation_suite_version="local-ai-fixtures-v1",
        artifacts=artifacts,
    )


@pytest.mark.parametrize(
    ("require_release", "expected_message"),
    [
        (True, "installed and validated"),
        (False, "installed and runtime-verified for benchmarking"),
    ],
)
@pytest.mark.asyncio
async def test_install_uses_persisted_lifecycle_and_requires_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_release: bool,
    expected_message: str,
) -> None:
    manifest = _manifest()
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "_load_locked_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "_load_released_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))

    async def complete(operation_id: str, _user_id=None, _lease=None) -> None:
        store = ArtifactStore(tmp_path)
        operations = PackOperationStore(store)
        operations.transition(
            operation_id,
            expected_states="queued",
            state="running",
            message="Downloading verified model files.",
        )
        staging = store.stage(manifest.pack_revision)
        for artifact in manifest.artifacts:
            for model_file in artifact.files:
                destination = staging / artifact.role.value / model_file.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(artifact.role.value.encode())
        store.activate_validated(
            staging,
            manifest,
            _issue_runtime_validation_receipt(manifest),
        )
        operation = operations.get(operation_id)
        assert operation is not None
        operations.transition(
            operation_id,
            expected_states="running",
            state="completed",
            bytes_done=operation["bytes_total"],
            message="Model pack is ready.",
        )

    monkeypatch.setattr(pack_cli, "run_operation", complete)

    result = await pack_cli._run_lifecycle(
        "install",
        stdout=stdout,
        stderr=stderr,
        require_release=require_release,
    )

    assert result == 0
    assert stderr.getvalue() == ""
    assert expected_message in stdout.getvalue()
    latest = PackOperationStore(ArtifactStore(tmp_path)).latest()
    assert latest is not None
    assert latest["action"] == "install"
    assert latest["state"] == "completed"


@pytest.mark.asyncio
async def test_lifecycle_failure_is_content_free_and_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "_load_locked_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "_load_released_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))

    async def fail(operation_id: str, _user_id=None, _lease=None) -> None:
        operations = PackOperationStore(ArtifactStore(tmp_path))
        operations.transition(
            operation_id,
            expected_states="queued",
            state="running",
            message="Downloading verified model files.",
        )
        operations.transition(
            operation_id,
            expected_states="running",
            state="failed",
            message="Model pack operation failed.",
            retryable=True,
        )

    monkeypatch.setattr(pack_cli, "run_operation", fail)

    result = await pack_cli.execute("install", stdout=stdout, stderr=stderr)

    assert result == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "ERROR: local model pack installation failed.\n"


@pytest.mark.asyncio
async def test_install_reconciles_a_stale_crashed_cli_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "_load_locked_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "_load_released_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))
    operations = PackOperationStore(ArtifactStore(tmp_path))
    stale = operations.create(action="install", manifest=manifest)
    invoked: list[str] = []

    async def fail_new_operation(
        operation_id: str,
        _user_id=None,
        _lease=None,
    ) -> None:
        invoked.append(operation_id)
        current = operations.transition(
            operation_id,
            expected_states="queued",
            state="running",
            message="Downloading verified model files.",
        )
        operations.transition(
            operation_id,
            expected_states="running",
            state="failed",
            bytes_done=0,
            bytes_total=current["bytes_total"],
            message="Model pack operation failed.",
            retryable=True,
        )

    monkeypatch.setattr(pack_cli, "run_operation", fail_new_operation)

    result = await pack_cli.execute("install", stdout=stdout, stderr=stderr)

    assert result == 1
    assert len(invoked) == 1
    assert invoked[0] != stale["id"]
    assert operations.get(stale["id"])["state"] == "failed"
    assert stderr.getvalue() == "ERROR: local model pack installation failed.\n"


@pytest.mark.asyncio
async def test_install_reconciliation_does_not_interrupt_a_live_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "_load_locked_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "_load_released_manifest", lambda: manifest)
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))
    operations = PackOperationStore(ArtifactStore(tmp_path))
    live = operations.create(action="install", manifest=manifest)
    lease = operations.acquire_operation_lease(live["id"], blocking=False)
    assert lease is not None

    try:
        result = await pack_cli.execute("install", stdout=stdout, stderr=stderr)
    finally:
        lease.release()

    assert result == 1
    assert operations.get(live["id"])["state"] == "queued"
    assert stdout.getvalue() == ""
    assert (
        stderr.getvalue()
        == "ERROR: a model pack lifecycle operation is already active.\n"
    )


@pytest.mark.asyncio
async def test_verify_refuses_when_no_pack_is_installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "_load_locked_manifest", _manifest)
    monkeypatch.setattr(pack_cli, "_load_released_manifest", _manifest)
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("apple_silicon", True))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))

    result = await pack_cli.execute("verify", stdout=stdout, stderr=stderr)

    assert result == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "ERROR: no local model pack is installed.\n"


@pytest.mark.asyncio
async def test_remove_uses_global_job_guard_and_lifecycle_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    guarded = False
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))

    async def globally_locked(mutation) -> None:
        nonlocal guarded
        guarded = True
        mutation()

    monkeypatch.setattr(
        pack_cli,
        "_run_globally_locked_pack_mutation",
        globally_locked,
    )

    result = await pack_cli.execute("remove", stdout=stdout, stderr=stderr)

    assert result == 0
    assert guarded is True
    assert stdout.getvalue() == "Local model pack removed.\n"
    assert stderr.getvalue() == ""


@pytest.mark.asyncio
async def test_install_rejects_an_incompatible_host_before_creating_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(pack_cli, "platform_profile", lambda: ("unsupported", False))
    monkeypatch.setattr(pack_cli.settings, "local_ai_model_dir", str(tmp_path))

    result = await pack_cli.execute("install", stdout=stdout, stderr=stderr)

    assert result == 1
    assert not any(tmp_path.iterdir())
    assert stdout.getvalue() == ""
    assert (
        stderr.getvalue()
        == "ERROR: the validated pack requires Apple Silicon with at least 16 GB.\n"
    )
