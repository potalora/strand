"""Authenticated, document-free model-pack lifecycle API tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

import app.services.local_ai.pack_operations as pack_operations_module
from app.config import settings
from app.models.ai_summary import AISummaryPrompt
from app.models.audit import AuditLog
from app.models.local_ai import LocalAIJob
from app.models.uploaded_file import UploadedFile
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
from tests.conftest import auth_headers, create_test_patient


def _manifest(
    revision: str = "apple-m4-16gb-v1",
) -> tuple[LocalAIManifest, dict[ModelRole, bytes]]:
    contents = {
        ModelRole.OCR: b"ocr-weights",
        ModelRole.EXTRACTION: b"extraction-weights",
        ModelRole.SUMMARY: b"summary-weights",
    }
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
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 32768, "max_output_tokens": 4096},
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
            validation_suite_version="local-ai-fixtures-v1",
            artifacts=artifacts,
        ),
        contents,
    )


def _write_manifest(path: Path, manifest: LocalAIManifest) -> None:
    path.write_text(
        json.dumps(asdict(manifest), sort_keys=True),
        encoding="utf-8",
    )


def _manifest_dict(manifest: LocalAIManifest) -> dict:
    return json.loads(json.dumps(asdict(manifest)))


def _install_pack(
    root: Path,
    manifest: LocalAIManifest,
    contents: dict[ModelRole, bytes],
) -> ArtifactStore:
    store = ArtifactStore(root)
    staging = store.stage(manifest.pack_revision)
    for artifact in manifest.artifacts:
        for model_file in artifact.files:
            path = staging / artifact.role.value / model_file.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[artifact.role])
    store.activate_validated(
        staging,
        manifest,
        _issue_runtime_validation_receipt(manifest),
    )
    return store


@pytest.mark.asyncio
async def test_retrying_failed_strict_upload_requeues_its_existing_job(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="strict-retry@example.com",
    )
    manifest, _contents = _manifest()
    upload = UploadedFile(
        id=uuid4(),
        user_id=UUID(user_id),
        filename="fixture-001.pdf",
        mime_type="application/pdf",
        file_size_bytes=100,
        file_hash="a" * 64,
        storage_path="/tmp/fixture-001.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
        processing_started_at=datetime.now(timezone.utc),
        processing_completed_at=datetime.now(timezone.utc),
        retry_count=3,
    )
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=UUID(user_id),
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="failed",
        stage="failed",
        progress={"stage": "extraction"},
        failure={"code": "local_worker_error"},
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(job)
    await db_session.commit()

    with patch(
        "app.api.upload.start_extraction_worker",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["triggered"] == 1
    await db_session.refresh(upload)
    await db_session.refresh(job)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.processing_started_at is None
    assert upload.processing_completed_at is None
    assert upload.retry_count == 0
    assert job.status == "queued"
    assert job.stage == "queued"
    assert job.progress == {}
    assert job.failure is None
    assert job.started_at is None
    assert job.completed_at is None


@pytest.mark.asyncio
async def test_summary_job_status_and_cancel_are_owner_scoped_and_content_free(
    client,
    db_session,
) -> None:
    owner_headers, owner_id = await auth_headers(
        client,
        email="summary-job-owner@example.com",
    )
    other_headers, _other_id = await auth_headers(
        client,
        email="summary-job-other@example.com",
    )
    patient = await create_test_patient(db_session, owner_id)
    manifest, _contents = _manifest()
    prompt = AISummaryPrompt(
        id=uuid4(),
        user_id=UUID(owner_id),
        patient_id=patient.id,
        summary_type="full",
        processing_mode="validated_strict_local",
        scope_filter={},
        system_prompt="sensitive system prompt",
        user_prompt="sensitive patient content",
        target_model="locked-local-summary",
        suggested_config={},
        record_count=1,
        model_provenance={"sensitive": "must not be returned"},
        generated_at=datetime.now(timezone.utc),
    )
    job = LocalAIJob(
        user_id=UUID(owner_id),
        summary_prompt_id=prompt.id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=_manifest_dict(manifest),
        status="processing",
        stage="summary",
    )
    db_session.add_all([prompt, job])
    await db_session.commit()

    with patch(
        "app.services.local_ai.model_manager.local_model_manager.cancel_registered",
        new_callable=AsyncMock,
        return_value=True,
    ) as cancel_registered:
        other_list = await client.get(
            "/api/v1/local-ai/jobs?kind=summary&active_only=true",
            headers=other_headers,
        )
        other_get = await client.get(
            f"/api/v1/local-ai/jobs/{job.id}",
            headers=other_headers,
        )
        other_cancel = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=other_headers,
        )
        owner_list = await client.get(
            "/api/v1/local-ai/jobs?kind=summary&active_only=true",
            headers=owner_headers,
        )
        owner_cancel = await client.post(
            f"/api/v1/local-ai/jobs/{job.id}/cancel",
            headers=owner_headers,
        )

    assert other_list.status_code == 200
    assert other_list.json() == []
    assert other_get.status_code == 404
    assert other_cancel.status_code == 404
    assert owner_list.status_code == 200
    assert owner_list.json() == [
        {
            "id": str(job.id),
            "kind": "summary",
            "status": "processing",
            "stage": "summary",
            "cancel_requested": False,
            "created_at": job.created_at.isoformat().replace("+00:00", "Z"),
            "started_at": None,
            "completed_at": None,
        }
    ]
    assert owner_cancel.status_code == 200
    assert owner_cancel.json()["cancel_requested"] is True
    assert set(owner_cancel.json()) == {
        "id",
        "kind",
        "status",
        "stage",
        "cancel_requested",
        "created_at",
        "started_at",
        "completed_at",
    }
    cancel_registered.assert_awaited_once_with(str(job.id))
    await db_session.refresh(job)
    assert job.cancel_requested is True


def test_lifecycle_operations_take_owner_only_cross_process_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[int] = []

    class FakeFcntl:
        LOCK_EX = 2
        LOCK_UN = 8

        @staticmethod
        def flock(_descriptor: int, operation: int) -> None:
            events.append(operation)

    monkeypatch.setattr(
        pack_operations_module,
        "fcntl",
        FakeFcntl,
        raising=False,
    )
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))

    operation_store.create(action="install", manifest=manifest)

    assert events == [FakeFcntl.LOCK_EX, FakeFcntl.LOCK_UN]
    assert (
        operation_store.directory / ".lifecycle.lock"
    ).stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_pack_mutation_guard_locks_before_rechecking_active_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.local_ai as local_ai_module

    events: list[str] = []

    async def acquire(_db: object) -> None:
        events.append("lock")

    async def recheck(_db: object) -> None:
        events.append("recheck")

    monkeypatch.setattr(
        local_ai_module,
        "acquire_local_ai_lifecycle_lock",
        acquire,
    )
    monkeypatch.setattr(local_ai_module, "_require_no_active_jobs", recheck)

    await local_ai_module._acquire_pack_mutation_db_guard(object())

    assert events == ["lock", "recheck"]


def test_lifecycle_transition_is_compare_and_swap(tmp_path: Path) -> None:
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))
    operation = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Downloading verified model files.",
    )

    with pytest.raises(LocalValidationError, match="state changed"):
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state="paused",
            message="Model pack operation paused.",
            retryable=True,
        )

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "running"


def test_startup_reconciliation_completes_only_exact_validated_activation(
    tmp_path: Path,
) -> None:
    manifest, contents = _manifest()
    store = _install_pack(tmp_path, manifest, contents)
    operation_store = PackOperationStore(store)
    operation = operation_store.create(action="install", manifest=manifest)
    operation_store.transition(
        operation["id"],
        expected_states="queued",
        state="running",
        message="Running local validation fixtures.",
    )

    assert operation_store.reconcile_interrupted() == 1

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "completed"
    assert persisted["bytes_done"] == persisted["bytes_total"]
    assert persisted["retryable"] is False


@pytest.mark.parametrize(
    ("action", "initial_state", "retryable"),
    [
        ("install", "queued", True),
        ("verify", "paused", True),
        ("rollback", "running", False),
    ],
)
def test_startup_reconciliation_never_strands_nonterminal_operations(
    tmp_path: Path,
    action: str,
    initial_state: str,
    retryable: bool,
) -> None:
    manifest, _contents = _manifest()
    operation_store = PackOperationStore(ArtifactStore(tmp_path))
    operation = operation_store.create(action=action, manifest=manifest)
    if initial_state != "queued":
        operation_store.transition(
            operation["id"],
            expected_states="queued",
            state=initial_state,
            message=(
                "Model pack operation paused."
                if initial_state == "paused"
                else "Restoring previous verified model pack."
            ),
            retryable=initial_state == "paused",
        )

    assert operation_store.reconcile_interrupted() == 1

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "failed"
    assert persisted["bytes_done"] == 0
    assert persisted["retryable"] is retryable


@pytest.fixture
def local_ai_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, contents = _manifest()
    manifest_path = tmp_path / "pack.lock.json"
    model_root = tmp_path / "models"
    _write_manifest(manifest_path, manifest)
    monkeypatch.setattr(settings, "local_ai_manifest_path", str(manifest_path))
    monkeypatch.setattr(settings, "local_ai_model_dir", str(model_root))
    monkeypatch.setattr(
        "app.api.local_ai._available_released_manifest", lambda: manifest
    )
    monkeypatch.setattr(
        "app.api.local_ai.platform_profile",
        lambda: ("apple_silicon", True),
    )
    return manifest, contents, model_root


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("install", "update", "verify"))
async def test_public_lifecycle_rejects_missing_release_evidence(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
    action: str,
) -> None:
    import app.api.local_ai as local_ai_module

    manifest, contents, model_root = local_ai_paths
    if action != "install":
        _install_pack(model_root, manifest, contents)
    monkeypatch.setattr(
        local_ai_module,
        "_available_released_manifest",
        lambda: (_ for _ in ()).throw(
            HTTPException(
                status_code=409, detail="A validated local model pack is not available."
            )
        ),
    )
    headers, _ = await auth_headers(client)

    response = await client.post(f"/api/v1/local-ai/{action}", headers=headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "A validated local model pack is not available."


@pytest.mark.asyncio
async def test_install_returns_document_free_operation(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"operation_id", "state"}
    assert body["state"] == "queued"
    assert all(
        forbidden not in json.dumps(body).lower()
        for forbidden in ("upload", "document", "prompt", "excerpt", "patient")
    )


@pytest.mark.asyncio
async def test_operation_status_is_persisted_and_document_free(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]
    response = await client.get(
        f"/api/v1/local-ai/operations/{operation_id}",
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "id",
        "action",
        "state",
        "current_role",
        "bytes_done",
        "bytes_total",
        "message",
        "retryable",
    }
    assert body["action"] == "install"
    assert body["state"] == "queued"
    assert "pack" not in body
    assert "manifest" not in body


@pytest.mark.asyncio
async def test_second_lifecycle_mutation_is_refused_while_one_is_nonterminal(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    first = await client.post("/api/v1/local-ai/install", headers=headers)
    second = await client.post("/api/v1/local-ai/install", headers=headers)

    assert first.status_code == 202
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_resume_and_retry_enforce_operation_state(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]

    resume = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/resume",
        headers=headers,
    )
    retry = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )

    assert resume.status_code == 409
    assert retry.status_code == 409


@pytest.mark.asyncio
async def test_resume_and_retry_restart_from_zero_after_staging_cleanup(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    _manifest_value, _contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    created = await client.post("/api/v1/local-ai/install", headers=headers)
    operation_id = created.json()["operation_id"]
    operation_store = PackOperationStore(ArtifactStore(model_root))

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="paused",
        message="Model pack operation paused.",
        retryable=True,
        bytes_done=5,
    )
    resumed = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/resume",
        headers=headers,
    )
    assert resumed.status_code == 200
    assert resumed.json()["state"] == "queued"
    assert resumed.json()["bytes_done"] == 0

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=True,
    )
    retried = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )
    assert retried.status_code == 200
    assert retried.json()["state"] == "queued"
    assert retried.json()["bytes_done"] == 0

    operation_store.transition(
        operation_id,
        expected_states="queued",
        state="failed",
        message="Model pack operation failed.",
        retryable=False,
    )
    refused = await client.post(
        f"/api/v1/local-ai/operations/{operation_id}/retry",
        headers=headers,
    )
    assert refused.status_code == 409


@pytest.mark.asyncio
async def test_status_reflects_exact_validation_receipt_without_fake_memory(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    monkeypatch.setattr(settings, "local_ai_enabled", False)
    monkeypatch.setattr(
        settings,
        "local_ai_release_evidence_path",
        str(model_root.parent / "missing.release.json"),
    )
    headers, _ = await auth_headers(client)
    _install_pack(model_root, manifest, contents)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["platform"] == "apple_silicon"
    assert body["compatible"] is True
    assert body["enabled"] is False
    assert body["active_revision"] == manifest.pack_revision
    assert body["state"] == "preview"
    assert body["status_reason"] == "feature_disabled"
    assert body["available_revision"] == manifest.pack_revision
    assert len(body["models"]) == 3
    assert all(model["installed"] is True for model in body["models"])
    assert all(model["validated"] is False for model in body["models"])
    assert all(model["expected_memory_bytes"] is None for model in body["models"])
    assert [model["download_bytes"] for model in body["models"]] == [
        sum(file.size for file in artifact.files) for artifact in manifest.artifacts
    ]


@pytest.mark.asyncio
async def test_status_distinguishes_missing_release_evidence_when_enabled(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    monkeypatch.setattr(settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        settings,
        "local_ai_release_evidence_path",
        str(model_root.parent / "missing.release.json"),
    )
    headers, _ = await auth_headers(client)
    _install_pack(model_root, manifest, contents)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["state"] == "preview"
    assert body["status_reason"] == "release_evidence_missing"


@pytest.mark.asyncio
async def test_status_marks_a_different_installed_pack_as_update_available(
    client,
    local_ai_paths,
) -> None:
    _available, _contents, model_root = local_ai_paths
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    _install_pack(model_root, prior, prior_contents)
    headers, _ = await auth_headers(client)

    response = await client.get("/api/v1/local-ai/status", headers=headers)

    assert response.status_code == 200
    assert response.json()["state"] == "update_available"


@pytest.mark.asyncio
async def test_verify_marks_only_a_successfully_verified_active_pack_ready(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)
    _install_pack(model_root, manifest, contents)

    async def verified(*_args, **_kwargs):
        return _issue_runtime_validation_receipt(manifest)

    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", verified)
    response = await client.post("/api/v1/local-ai/verify", headers=headers)
    assert response.status_code == 202

    status = await client.get("/api/v1/local-ai/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["state"] == "preview"
    assert all(not model["validated"] for model in status.json()["models"])


@pytest.mark.asyncio
async def test_install_validates_staging_before_atomic_activation(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)
    store = ArtifactStore(model_root)
    validation_saw_inactive_pack = False

    async def staged_download(
        selected_manifest,
        selected_store,
        progress_callback,
    ):
        assert selected_manifest == manifest
        staging = selected_store.stage(manifest.pack_revision)
        for artifact in manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
            progress_callback(
                {
                    "role": artifact.role.value,
                    "bytes_done": sum(file.size for file in artifact.files),
                    "bytes_total": sum(file.size for file in artifact.files),
                }
            )
        selected_store.verify(staging, manifest)
        return staging

    async def verify_candidate(selected_manifest, candidate_path):
        nonlocal validation_saw_inactive_pack
        assert selected_manifest == manifest
        assert candidate_path.parent == store.staging_dir
        validation_saw_inactive_pack = store.active_revision() is None
        return _issue_runtime_validation_receipt(manifest)

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", verify_candidate)

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert validation_saw_inactive_pack is True
    assert store.active_revision() == manifest.pack_revision
    assert PackOperationStore(store).is_validated(manifest)
    terminal_audit = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.action == "local_ai.operation.completed")
        )
    ).scalar_one()
    assert terminal_audit.details == {
        "action": "install",
        "operation_id": response.json()["operation_id"],
        "outcome": "completed",
        "pack_revision": manifest.pack_revision,
    }


@pytest.mark.asyncio
async def test_job_admitted_during_validation_prevents_pack_activation(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, user_id = await auth_headers(
        client,
        email="activation-race@example.com",
    )
    store = ArtifactStore(model_root)

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        staging = selected_store.stage(selected_manifest.pack_revision)
        for artifact in selected_manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
        return staging

    async def admit_job_before_activation(_manifest, _candidate_path):
        upload = UploadedFile(
            user_id=UUID(user_id),
            filename="already-admitted.pdf",
            mime_type="application/pdf",
            file_hash="b" * 64,
            storage_path="/private/already-admitted.pdf",
            ingestion_status="pending_extraction",
            file_category="unstructured",
            processing_mode="validated_strict_local",
            processing_manifest=_manifest_dict(manifest),
            processing_schema_version="clinical-document-extraction.v1",
        )
        db_session.add(upload)
        await db_session.flush()
        db_session.add(
            LocalAIJob(
                user_id=UUID(user_id),
                upload_id=upload.id,
                kind="ingestion",
                processing_mode="validated_strict_local",
                manifest_snapshot=_manifest_dict(manifest),
                status="queued",
                stage="queued",
            )
        )
        await db_session.commit()
        return _issue_runtime_validation_receipt(manifest)

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr(
        "app.api.local_ai.verify_installed_pack",
        admit_job_before_activation,
    )

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() is None
    operation = PackOperationStore(store).get(response.json()["operation_id"])
    assert operation is not None
    assert operation["state"] == "failed"
    assert list(store.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_runtime_validation_never_activates_staged_pack(
    client,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)
    store = ArtifactStore(model_root)

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        assert selected_manifest == manifest
        staging = selected_store.stage(manifest.pack_revision)
        for artifact in manifest.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[artifact.role])
        selected_store.verify(staging, manifest)
        return staging

    async def reject_candidate(_selected_manifest, _candidate_path) -> None:
        raise RuntimeError("synthetic fixture failure")

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", reject_candidate)

    response = await client.post("/api/v1/local-ai/install", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() is None
    assert list(store.staging_dir.iterdir()) == []
    status_response = await client.get("/api/v1/local-ai/status", headers=headers)
    assert status_response.json()["state"] == "failed"
    assert all(not model["validated"] for model in status_response.json()["models"])
    assert "synthetic fixture failure" not in caplog.text


@pytest.mark.asyncio
async def test_failed_update_validation_preserves_prior_verified_active_pointer(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    available, available_contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    store = _install_pack(model_root, prior, prior_contents)
    PackOperationStore(store).mark_validated(
        prior,
        _issue_runtime_validation_receipt(prior),
    )

    async def staged_download(selected_manifest, selected_store, _progress_callback):
        assert selected_manifest == available
        staging = selected_store.stage(available.pack_revision)
        for artifact in available.artifacts:
            for model_file in artifact.files:
                path = staging / artifact.role.value / model_file.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(available_contents[artifact.role])
        selected_store.verify(staging, available)
        return staging

    async def reject_candidate(_selected_manifest, _candidate_path) -> None:
        raise RuntimeError("synthetic fixture failure")

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        staged_download,
    )
    monkeypatch.setattr("app.api.local_ai.verify_installed_pack", reject_candidate)

    response = await client.post("/api/v1/local-ai/update", headers=headers)

    assert response.status_code == 202
    assert store.active_revision() == prior.pack_revision
    assert PackOperationStore(store).is_validated(prior)
    assert not (store.packs_dir / available.pack_revision).exists()
    assert list(store.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_download_persists_paused_retryable_operation(
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    manifest, _contents, model_root = local_ai_paths
    operation_store = PackOperationStore(ArtifactStore(model_root))
    operation = operation_store.create(action="install", manifest=manifest)

    async def cancel_download(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "app.api.local_ai.download_manifest_to_stage",
        cancel_download,
    )
    from app.api.local_ai import run_operation

    with pytest.raises(asyncio.CancelledError):
        await run_operation(operation["id"])

    persisted = operation_store.get(operation["id"])
    assert persisted is not None
    assert persisted["state"] == "paused"
    assert persisted["message"] == "Model pack operation paused."
    assert persisted["retryable"] is True


@pytest.mark.asyncio
async def test_rollback_never_points_at_an_unvalidated_previous_pack(
    client,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    current, current_contents, model_root = local_ai_paths
    headers, _ = await auth_headers(client)
    prior, prior_contents = _manifest("apple-m4-16gb-prior")
    store = _install_pack(model_root, prior, prior_contents)
    _install_pack(model_root, current, current_contents)
    store.remove_validation_receipt(prior)
    calls = 0
    original_rollback = ArtifactStore.rollback

    def observed_rollback(selected_store) -> None:
        nonlocal calls
        calls += 1
        original_rollback(selected_store)

    monkeypatch.setattr(ArtifactStore, "rollback", observed_rollback)

    response = await client.post("/api/v1/local-ai/rollback", headers=headers)

    assert response.status_code == 202
    assert calls == 0
    assert store.active_revision() == current.pack_revision
    operation = PackOperationStore(store).get(response.json()["operation_id"])
    assert operation is not None
    assert operation["state"] == "failed"


@pytest.mark.asyncio
async def test_remove_refuses_any_active_local_job(
    client,
    db_session,
    local_ai_paths,
) -> None:
    manifest, _contents, _model_root = local_ai_paths
    headers, user_id = await auth_headers(client)
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="note.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/note.pdf",
        processing_mode="validated_strict_local",
        processing_manifest=_manifest_dict(manifest),
        processing_schema_version="1",
    )
    db_session.add(upload)
    await db_session.flush()
    db_session.add(
        LocalAIJob(
            user_id=UUID(user_id),
            upload_id=upload.id,
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=_manifest_dict(manifest),
            status="queued",
            stage="preflight",
        )
    )
    await db_session.commit()

    response = await client.delete("/api/v1/local-ai", headers=headers)

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_pack_audit_details_are_content_free(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
    local_ai_paths,
) -> None:
    headers, _ = await auth_headers(client)

    async def leave_queued(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("app.api.local_ai.run_operation", leave_queued)
    response = await client.post("/api/v1/local-ai/install", headers=headers)
    assert response.status_code == 202

    rows = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.action == "local_ai.install")
        )
    ).scalars()
    details = [row.details for row in rows]
    assert details
    serialized = json.dumps(details).lower()
    assert all(
        forbidden not in serialized
        for forbidden in (
            "excerpt",
            "patient",
            "prompt",
            "document",
            "upload",
            "clinical",
        )
    )
