"""Regressions for cross-process local-model lifecycle safety."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

import app.services.local_ai.pack_operations as pack_operations_module
from app.config import settings
from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.fidelity_metrics import FidelityMetrics
from app.services.local_ai.fidelity_runner import (
    FIDELITY_CORPUS_SHA256,
    FIDELITY_SUITE_VERSION,
    RELEASE_SYNTHETIC_DOCUMENT_COUNT,
    build_fidelity_report,
)
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.pack_operations import PackOperationStore
from app.services.local_ai.release_evidence import build_release_evidence
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import (
    _issue_runtime_validation_receipt,
)


def _manifest(
    revision: str = "audit-v1",
) -> tuple[LocalAIManifest, dict[ModelRole, bytes]]:
    contents = {
        ModelRole.OCR: b"ocr",
        ModelRole.EXTRACTION: b"extract",
        ModelRole.SUMMARY: b"summary",
    }
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository=f"owner/{role.value}",
            revision=str(index) * 40,
            quantization="4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 4096, "max_output_tokens": 1024},
            files=(
                ManifestFile(
                    path="model.safetensors",
                    sha256=hashlib.sha256(contents[role]).hexdigest(),
                    size=len(contents[role]),
                ),
            ),
        )
        for index, role in enumerate(ModelRole, start=1)
    )
    return (
        LocalAIManifest(
            schema_version=1,
            pack_revision=revision,
            platform="apple_silicon",
            runtime={"name": "mlx-vlm", "version": "0.5.0"},
            validation_suite_version="fixtures-v1",
            artifacts=artifacts,
        ),
        contents,
    )


def _write_manifest(path: Path, manifest: LocalAIManifest) -> None:
    path.write_text(json.dumps(asdict(manifest), sort_keys=True), encoding="utf-8")


def _write_release_evidence(
    root: Path,
    manifest: LocalAIManifest,
) -> tuple[Path, Path, Path]:
    run = {
        "cold_start_seconds": 1.0,
        "duration_seconds": 1.0,
        "peak_rss_bytes": 1024**3,
        "mlx_peak_memory_bytes": 1024**3,
        "mlx_active_memory_after_bytes": 1,
        "throughput_per_second": 1.0,
    }
    from app.services.local_ai.artifact_store import manifest_sha256

    benchmark_value = {
        "schema_version": 1,
        "content_free": True,
        "machine": {
            "platform_profile": manifest.platform,
            "model": "Test Mac",
            "physical_memory_bytes": 16 * 1024**3,
            "os_version": "test",
        },
        "manifest": {
            "pack_revision": manifest.pack_revision,
            "sha256": manifest_sha256(manifest),
            "runtime_name": manifest.runtime["name"],
            "runtime_version": manifest.runtime["version"],
        },
        "processes": {"max_live_models": 1, "roles_started": 9},
        "roles": {
            role.value: [{**run, "run": index} for index in range(1, 4)]
            for role in ModelRole
        },
        "system": {
            "swap_before_bytes": 0,
            "swap_after_bytes": 0,
            "swap_delta_bytes": 0,
            "memory_pressure_termination": False,
            "sustained_swap_thrashing": False,
        },
        "reclamation": {
            "baseline_active_memory_bytes": 0,
            "final_active_memory_bytes": 1,
            "peak_active_memory_bytes": 1024**3,
            "final_active_memory_ratio": 1 / 1024**3,
        },
    }
    benchmark_path = root / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark_value, sort_keys=True),
        encoding="utf-8",
    )
    fidelity = build_fidelity_report(
        metrics=FidelityMetrics(
            critical_numeric_exact=1.0,
            critical_precision=1.0,
            critical_recall=1.0,
            accepted_output_schema_validity=1.0,
            forbidden_extraction_facts=0,
            unsupported_summary_facts=0,
            accepted_facts_without_evidence=0,
            summary_fact_recall=1.0,
            summary_typed_field_recall=1.0,
        ),
        fixture_suite_version=FIDELITY_SUITE_VERSION,
        fixture_suite_sha256=FIDELITY_CORPUS_SHA256,
        manifest_sha256=manifest_sha256(manifest),
        synthetic_documents=RELEASE_SYNTHETIC_DOCUMENT_COUNT,
        private_documents=0,
    )
    fidelity_path = root / "fidelity.json"
    fidelity_path.write_text(
        json.dumps(fidelity.as_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    release_value = build_release_evidence(manifest, benchmark_value, fidelity)
    release_value["benchmark_sha256"] = hashlib.sha256(
        benchmark_path.read_bytes()
    ).hexdigest()
    release_value["fidelity_sha256"] = hashlib.sha256(
        fidelity_path.read_bytes()
    ).hexdigest()
    release_path = root / "release.json"
    release_path.write_text(
        json.dumps(release_value, sort_keys=True),
        encoding="utf-8",
    )
    return release_path, benchmark_path, fidelity_path


def _install(
    root: Path,
    manifest: LocalAIManifest,
    contents: dict[ModelRole, bytes],
) -> ArtifactStore:
    store = ArtifactStore(root)
    staging = store.stage(manifest.pack_revision)
    for artifact in manifest.artifacts:
        for model_file in artifact.files:
            destination = staging / artifact.role.value / model_file.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents[artifact.role])
    store.activate_validated(
        staging,
        manifest,
        _issue_runtime_validation_receipt(manifest),
    )
    return store


@pytest.fixture
def lifecycle_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, contents = _manifest()
    manifest_path = tmp_path / "pack.lock.json"
    model_root = tmp_path / "models"
    _write_manifest(manifest_path, manifest)
    release_path, benchmark_path, fidelity_path = _write_release_evidence(
        tmp_path, manifest
    )
    monkeypatch.setattr(settings, "local_ai_enabled", True)
    monkeypatch.setattr(settings, "local_ai_manifest_path", str(manifest_path))
    monkeypatch.setattr(settings, "local_ai_model_dir", str(model_root))
    monkeypatch.setattr(
        settings,
        "local_ai_release_evidence_path",
        str(release_path),
    )
    monkeypatch.setattr(settings, "local_ai_benchmark_path", str(benchmark_path))
    monkeypatch.setattr(settings, "local_ai_fidelity_path", str(fidelity_path))
    monkeypatch.setattr(
        "app.api.local_ai.platform_profile",
        lambda: ("apple_silicon", True),
    )
    return manifest, contents, model_root


@pytest.mark.asyncio
async def test_retry_refuses_to_create_a_second_nonterminal_operation(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, _contents, model_root = lifecycle_paths

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    operation_store = PackOperationStore(ArtifactStore(model_root))
    failed = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        failed["id"],
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )
    active = operation_store.create(action="verify", manifest=manifest)

    request = Request({"type": "http", "method": "POST", "path": "/"})
    with pytest.raises(HTTPException) as error:
        await local_ai_module._restart_operation(
            operation_id=UUID(failed["id"]),
            expected_state="failed",
            background_tasks=BackgroundTasks(),
            request=request,
            user_id=UUID(int=1),
            db=object(),
        )

    assert error.value.status_code == 409
    assert operation_store.get(failed["id"])["state"] == "failed"
    assert operation_store.get(active["id"])["state"] == "queued"


@pytest.mark.asyncio
async def test_retry_sweeps_orphan_staging_before_requeue(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, _contents, model_root = lifecycle_paths

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    store = ArtifactStore(model_root)
    operation_store = PackOperationStore(store)
    failed = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        failed["id"],
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )
    orphan = store.stage(manifest.pack_revision)
    (orphan / "partial.bin").write_bytes(b"partial")

    response = await local_ai_module._restart_operation(
        operation_id=UUID(failed["id"]),
        expected_state="failed",
        background_tasks=BackgroundTasks(),
        request=Request({"type": "http", "method": "POST", "path": "/"}),
        user_id=UUID(int=1),
        db=object(),
    )

    assert response.state == "queued"
    assert list(store.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_ready_pack_remains_ready_after_failed_maintenance_operation(
    lifecycle_paths,
) -> None:
    from app.api.local_ai import get_local_pack_status

    manifest, contents, model_root = lifecycle_paths
    store = _install(model_root, manifest, contents)
    operation_store = PackOperationStore(store)
    failed = operation_store.create(action="verify", manifest=manifest)
    operation_store.transition(
        failed["id"],
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )

    response = await get_local_pack_status(UUID(int=1))
    body = response.model_dump(mode="json")

    assert body["state"] == "ready"
    assert body["operation"]["state"] == "failed"
    assert all(model["validated"] for model in body["models"])


def test_live_peer_operation_is_not_reconciled_as_failed(tmp_path: Path) -> None:
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))
    operation = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Downloading verified model files.",
    )
    lease_path = operation_store.directory / f".{operation['id']}.lease"
    script = (
        "import fcntl, os, sys\n"
        "fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT, 0o600)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "print('ready', flush=True)\n"
        "sys.stdin.read()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lease_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"

        assert operation_store.reconcile_interrupted() == 0
        assert operation_store.get(operation["id"])["state"] == "running"
    finally:
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=5)

    assert operation_store.reconcile_interrupted() == 1
    assert operation_store.get(operation["id"])["state"] == "failed"


@pytest.mark.asyncio
async def test_operation_runner_holds_peer_visible_lease(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, _contents, model_root = lifecycle_paths
    operation_store = PackOperationStore(ArtifactStore(model_root))
    operation = operation_store.create(action="install", manifest=manifest)
    peer_acquired: bool | None = None

    async def inspect_lease(_operation_id: str, _user_id: UUID | None) -> None:
        nonlocal peer_acquired
        peer = PackOperationStore(ArtifactStore(model_root))
        with peer.operation_lease(operation["id"], blocking=False) as acquired:
            peer_acquired = acquired

    monkeypatch.setattr(local_ai_module, "_run_claimed_operation", inspect_lease)

    await local_ai_module.run_operation(operation["id"])

    assert peer_acquired is False


@pytest.mark.asyncio
async def test_queued_operation_is_leased_before_background_runner_starts(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    _manifest_value, _contents, model_root = lifecycle_paths

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    monkeypatch.setattr(local_ai_module, "_run_claimed_operation", leave_queued)
    background_tasks = BackgroundTasks()

    response = await local_ai_module._queue_operation(
        action="install",
        background_tasks=background_tasks,
        request=Request({"type": "http", "method": "POST", "path": "/"}),
        user_id=UUID(int=1),
        db=object(),
    )
    operation_store = PackOperationStore(ArtifactStore(model_root))

    assert operation_store.reconcile_interrupted() == 0
    assert operation_store.get(str(response.operation_id))["state"] == "queued"

    await background_tasks()

    assert operation_store.reconcile_interrupted() == 1
    assert operation_store.get(str(response.operation_id))["state"] == "failed"


@pytest.mark.asyncio
async def test_failed_queue_handoff_marks_operation_failed_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    _manifest_value, _contents, model_root = lifecycle_paths

    async def no_guard(_db: object) -> None:
        return None

    async def fail_audit(*_args, **_kwargs) -> None:
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", fail_audit)

    with pytest.raises(RuntimeError, match="audit unavailable"):
        await local_ai_module._queue_operation(
            action="install",
            background_tasks=BackgroundTasks(),
            request=Request({"type": "http", "method": "POST", "path": "/"}),
            user_id=UUID(int=1),
            db=object(),
        )

    operation_store = PackOperationStore(ArtifactStore(model_root))
    operation = operation_store.latest()
    assert operation is not None
    assert operation["state"] == "failed"
    assert operation["retryable"] is True
    with operation_store.operation_lease(operation["id"], blocking=False) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_failed_restart_handoff_restores_previous_restartable_state(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, _contents, model_root = lifecycle_paths

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    class RejectingBackgroundTasks:
        def add_task(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("background handoff unavailable")

    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    operation_store = PackOperationStore(ArtifactStore(model_root))
    failed = operation_store.create(action="install", manifest=manifest)
    failed = operation_store.transition(
        failed["id"],
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )

    with pytest.raises(RuntimeError, match="background handoff unavailable"):
        await local_ai_module._restart_operation(
            operation_id=UUID(failed["id"]),
            expected_state="failed",
            background_tasks=RejectingBackgroundTasks(),  # type: ignore[arg-type]
            request=Request({"type": "http", "method": "POST", "path": "/"}),
            user_id=UUID(int=1),
            db=object(),
        )

    assert operation_store.get(failed["id"]) == failed
    with operation_store.operation_lease(failed["id"], blocking=False) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_dropped_background_tasks_releases_preclaimed_operation_lease(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    _manifest_value, _contents, model_root = lifecycle_paths

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    background_tasks = BackgroundTasks()
    response = await local_ai_module._queue_operation(
        action="install",
        background_tasks=background_tasks,
        request=Request({"type": "http", "method": "POST", "path": "/"}),
        user_id=UUID(int=1),
        db=object(),
    )
    operation_store = PackOperationStore(ArtifactStore(model_root))

    with operation_store.operation_lease(
        str(response.operation_id),
        blocking=False,
    ) as acquired:
        assert acquired is False

    del background_tasks
    gc.collect()

    with operation_store.operation_lease(
        str(response.operation_id),
        blocking=False,
    ) as acquired:
        assert acquired is True
    assert operation_store.reconcile_interrupted() == 1
    assert operation_store.get(str(response.operation_id))["state"] == "failed"


@pytest.mark.asyncio
async def test_restart_rejects_operation_while_prior_runner_holds_lease(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_paths,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, _contents, model_root = lifecycle_paths

    async def no_guard(_db: object) -> None:
        return None

    async def no_audit(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(local_ai_module, "_acquire_pack_mutation_db_guard", no_guard)
    monkeypatch.setattr(local_ai_module, "log_audit_event", no_audit)
    operation_store = PackOperationStore(ArtifactStore(model_root))
    failed = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        failed["id"],
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )

    with operation_store.operation_lease(failed["id"], blocking=False) as acquired:
        assert acquired is True
        with pytest.raises(HTTPException) as error:
            await local_ai_module._restart_operation(
                operation_id=UUID(failed["id"]),
                expected_state="failed",
                background_tasks=BackgroundTasks(),
                request=Request({"type": "http", "method": "POST", "path": "/"}),
                user_id=UUID(int=1),
                db=object(),
            )

    assert error.value.status_code == 409
    assert operation_store.get(failed["id"])["state"] == "failed"


@pytest.mark.asyncio
async def test_active_maintenance_precedes_ready_pack_status(
    lifecycle_paths,
) -> None:
    from app.api.local_ai import get_local_pack_status

    manifest, contents, model_root = lifecycle_paths
    store = _install(model_root, manifest, contents)
    operation_store = PackOperationStore(store)
    operation = operation_store.create(action="verify", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Running local validation fixtures.",
    )

    response = await get_local_pack_status(UUID(int=1))

    assert response.state == "verifying"
    assert response.operation is not None
    assert response.operation.state == "running"


def test_startup_reconciliation_sweeps_orphan_staging(tmp_path: Path) -> None:
    manifest, _contents = _manifest()
    store = ArtifactStore(tmp_path)
    orphan = store.stage(manifest.pack_revision)
    (orphan / "partial.bin").write_bytes(b"partial")

    assert PackOperationStore(store).reconcile_interrupted() == 0

    assert list(store.staging_dir.iterdir()) == []


def test_orphan_sweep_never_follows_a_staging_symlink(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "models")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    (store.staging_dir / "orphan-link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(LocalValidationError, match="staging path"):
        store.sweep_orphan_staging()

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_operation_json_replace_fsyncs_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _contents = _manifest()
    directory_fsyncs: list[int] = []
    original_fsync = os.fsync

    def observe_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(pack_operations_module.os, "fsync", observe_fsync)

    PackOperationStore(ArtifactStore(tmp_path)).create(
        action="install",
        manifest=manifest,
    )

    assert directory_fsyncs


def test_operation_lease_explicit_release_and_finalizer_close_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []
    monkeypatch.setattr(
        pack_operations_module,
        "_release_lease_descriptor",
        released.append,
    )
    lease = pack_operations_module.OperationLease(123)

    lease.release()
    lease.release()
    del lease
    gc.collect()

    assert released == [123]


@pytest.mark.asyncio
async def test_startup_reconciles_pack_and_strict_jobs_under_database_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.main as main_module
    import app.services.auth_service as auth_service
    import app.services.extraction.terminology as terminology
    from app.api import upload

    events: list[str] = []

    class FakeResult:
        rowcount = 0

    class FakeSession:
        async def execute(self, _query: object, *_args, **_kwargs) -> FakeResult:
            return FakeResult()

        async def commit(self) -> None:
            events.append("commit")

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class FakeManager:
        async def start(self) -> None:
            events.append("manager-start")

        async def stop(self) -> None:
            events.append("manager-stop")

    async def acquire(_db: object) -> None:
        events.append("db-lock")

    async def recover_uploads(_db: object) -> int:
        events.append("recover-ingestion")
        return 0

    async def recover_summaries(_db: object) -> list[UUID]:
        events.append("recover-summary")
        return []

    async def reconcile_zip(_db: object, _root: Path) -> None:
        events.append("zip")

    async def no_purge(_db: object) -> int:
        return 0

    monkeypatch.setattr(main_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        main_module.settings, "local_ai_model_dir", str(tmp_path / "models")
    )
    monkeypatch.setattr(
        main_module.settings,
        "local_ai_scratch_dir",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setattr(main_module.settings, "phi_ner_enabled", False)
    monkeypatch.setattr(main_module.settings, "extraction_engine", "gemini")
    monkeypatch.setattr(main_module, "async_session_factory", FakeSessionContext)
    monkeypatch.setattr(main_module, "local_model_manager", FakeManager())
    monkeypatch.setattr(main_module, "reconcile_zip_child_sets", reconcile_zip)
    monkeypatch.setattr(
        main_module,
        "_recover_unstructured_jobs_on_startup",
        recover_uploads,
    )
    monkeypatch.setattr(
        main_module,
        "_recover_strict_local_summary_jobs_on_startup",
        recover_summaries,
    )
    monkeypatch.setattr(
        main_module,
        "_reconcile_model_pack_operations_on_startup",
        lambda: events.append("reconcile-pack") or 0,
    )
    monkeypatch.setattr(
        main_module,
        "acquire_local_ai_lifecycle_lock",
        acquire,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "sweep_stale_scratch",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(terminology, "schedule_medication_refresh", lambda: None)
    monkeypatch.setattr(auth_service, "purge_expired_revoked_tokens", no_purge)
    monkeypatch.setattr(upload, "start_extraction_worker", lambda: None)

    async with main_module.lifespan(FastAPI()):
        pass

    assert events.index("db-lock") < events.index("reconcile-pack")
    assert events.index("db-lock") < events.index("recover-ingestion")
    assert events.index("db-lock") < events.index("recover-summary")
    assert events.index("commit", events.index("recover-summary")) > events.index(
        "recover-summary"
    )
