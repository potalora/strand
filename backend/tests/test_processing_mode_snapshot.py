from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.local_ai import LocalAIJob
from app.models.llm_settings import LLMProviderConfig
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.schemas.llm_settings import RoutingUpdate
from app.schemas.summary import GenerateSummaryRequest
from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.errors import RuntimeIdentityRequiredError
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.processing_snapshot import (
    EXTRACTION_SCHEMA_VERSION,
    ProcessingSnapshot,
    build_ingestion_job,
    revalidate_strict_snapshot_admission,
    resolve_new_ingestion_snapshot,
    resolve_new_job_snapshot,
)
from app.services.local_ai.types import ModelRole
from app.services.local_ai.types import ProcessingMode
from app.services.local_ai.runtime_identity import (
    WorkerRuntimeIdentity,
    resolve_worker_runtime_identity,
)
from app.services.ai.llm.config import load_llm_config
from tests.test_local_ai_runtime_identity import build_worker_project
from app.utils.file_utils import encrypt_stream
from tests.conftest import auth_headers


def _load_prompt_only_default_migration() -> ModuleType:
    migration_path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "b0c1d2e3f4a5_default_processing_prompt_only.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_prompt_only_default_migration",
        migration_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ScalarResult:
    def __init__(self, value: object | None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object | None:
        return self._value


class _FakeDB:
    def __init__(self, stored_mode: str | None) -> None:
        self._stored_mode = stored_mode

    async def execute(self, _statement: object) -> _ScalarResult:
        preference = (
            None
            if self._stored_mode is None
            else SimpleNamespace(processing_mode=self._stored_mode)
        )
        return _ScalarResult(preference)


def _manifest() -> LocalAIManifest:
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
        validation_suite_version="fixtures-v1",
        artifacts=tuple(
            ManifestArtifact(
                role=role,
                repository=f"owner/{role.value}",
                revision=str(index) * 40,
                quantization="4bit",
                license="apache-2.0",
                attribution=f"https://huggingface.co/owner/{role.value}",
                decode_limits={
                    "max_input_tokens": 4096,
                    "max_output_tokens": 1024,
                },
                files=(
                    ManifestFile(
                        path="weights/model.safetensors",
                        sha256=str(index) * 64,
                        size=1,
                    ),
                ),
            )
            for index, role in enumerate(ModelRole, start=1)
        ),
    )


def _enable_strict_pack(
    monkeypatch: pytest.MonkeyPatch,
    manifest: LocalAIManifest | None = None,
) -> LocalAIManifest:
    import app.services.local_ai.processing_snapshot as snapshot_module

    manifest = manifest or _manifest()
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )

    class _Store:
        def __init__(self, _path: Path) -> None:
            pass

        def active_manifest(self) -> LocalAIManifest:
            return manifest

        def has_validation_receipt(self, _manifest: LocalAIManifest) -> bool:
            return True

    monkeypatch.setattr(snapshot_module, "ArtifactStore", _Store)
    monkeypatch.setattr(
        snapshot_module,
        "PackOperationStore",
        lambda _store: SimpleNamespace(has_nonterminal=lambda: False),
    )
    monkeypatch.setattr(
        snapshot_module,
        "load_release_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    _stub_runtime_identity(monkeypatch, manifest)
    return manifest


def _stub_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    manifest: LocalAIManifest,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    command = ("/synthetic/worker",)
    project = Path("/synthetic/project")
    monkeypatch.setattr(
        snapshot_module,
        "normalize_worker_runtime_binding",
        lambda _command, _project: (command, project),
    )
    monkeypatch.setattr(
        snapshot_module,
        "resolve_worker_runtime_identity",
        lambda observed_command, observed_project: (
            WorkerRuntimeIdentity(
                scheme=manifest.runtime["worker_identity_scheme"],
                bundle_sha256=manifest.runtime["worker_bundle_sha256"],
            )
            if observed_command == command and observed_project == project
            else pytest.fail("unexpected worker runtime binding")
        ),
    )


@pytest.mark.asyncio
async def test_strict_admission_rejects_missing_release_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    _enable_strict_pack(monkeypatch)
    monkeypatch.setattr(
        snapshot_module,
        "load_release_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LocalValidationError("missing")
        ),
    )
    with pytest.raises(LocalPolicyError, match="unavailable"):
        await resolve_new_job_snapshot(
            _FakeDB(None), uuid4(), ProcessingMode.VALIDATED_STRICT_LOCAL
        )


@pytest.mark.asyncio
async def test_strict_admission_rejects_v1_before_release_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    legacy_manifest = replace(
        _manifest(),
        schema_version=1,
        pack_revision="apple-m4-16gb-v1",
        runtime={"name": "mlx-vlm", "version": "0.5.0"},
    )
    manifest_path = tmp_path / "legacy-manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(legacy_manifest), sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_manifest_path",
        str(manifest_path),
    )
    release_calls = 0

    def _release_evidence(*_args: object, **_kwargs: object) -> None:
        nonlocal release_calls
        release_calls += 1

    monkeypatch.setattr(snapshot_module, "load_release_evidence", _release_evidence)

    with pytest.raises(RuntimeIdentityRequiredError):
        await resolve_new_job_snapshot(
            _FakeDB(None), uuid4(), ProcessingMode.VALIDATED_STRICT_LOCAL
        )

    assert release_calls == 0


@pytest.mark.parametrize("drifted_file", ["source", "lock"])
@pytest.mark.asyncio
async def test_strict_admission_rejects_worker_drift_before_pack_evidence_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drifted_file: str,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    project = build_worker_project(tmp_path / "worker")
    observed = resolve_worker_runtime_identity(project.command, project.project_dir)
    manifest = replace(
        _manifest(),
        runtime={
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": observed.scheme,
            "worker_bundle_sha256": observed.bundle_sha256,
        },
    )
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_worker_command",
        project.command,
    )
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_worker_project_dir",
        str(project.project_dir),
    )
    monkeypatch.setattr(
        snapshot_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )
    if drifted_file == "source":
        (project.effective_package / "common.py").write_text(
            "# drifted source\n",
            encoding="utf-8",
        )
    else:
        (project.project_dir / "uv.lock").write_text(
            "version = 1\n# drifted lock\n",
            encoding="utf-8",
        )

    pack_reads = 0

    class _UnexpectedStore:
        def __init__(self, _path: Path) -> None:
            nonlocal pack_reads
            pack_reads += 1

    monkeypatch.setattr(snapshot_module, "ArtifactStore", _UnexpectedStore)
    monkeypatch.setattr(
        snapshot_module,
        "load_release_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "release evidence loaded before identity"
        ),
    )

    with pytest.raises(LocalPolicyError, match="unavailable"):
        await resolve_new_job_snapshot(
            _FakeDB(None),
            uuid4(),
            ProcessingMode.VALIDATED_STRICT_LOCAL,
        )

    assert pack_reads == 0


def test_routing_update_accepts_only_explicit_processing_modes() -> None:
    update = RoutingUpdate(
        processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
    )

    assert update.processing_mode is ProcessingMode.VALIDATED_STRICT_LOCAL
    with pytest.raises(ValidationError):
        RoutingUpdate(processing_mode="fallback-to-cloud")


def test_generate_summary_defaults_prompt_only() -> None:
    request = GenerateSummaryRequest(patient_id=uuid4())

    assert request.processing_mode is ProcessingMode.PROMPT_ONLY


def test_prompt_only_default_migration_never_rewrites_existing_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _load_prompt_only_default_migration()
    altered: list[tuple[str, str, object]] = []
    statements: list[str] = []
    monkeypatch.setattr(
        migration.op,
        "alter_column",
        lambda table, column, **kwargs: altered.append(
            (table, column, kwargs["server_default"])
        ),
    )
    monkeypatch.setattr(
        migration.op,
        "execute",
        lambda statement: statements.append(str(statement)),
    )

    migration.upgrade()

    assert len(statements) == 1
    assert statements[0].lstrip().startswith("CREATE OR REPLACE FUNCTION")
    assert "'prompt_only'" in statements[0]
    assert re.search(r"(?im)^\s*(UPDATE|INSERT|DELETE)\s+", statements[0]) is None
    assert altered == [
        ("ai_summary_prompts", "processing_mode", "prompt_only"),
        ("uploaded_files", "processing_mode", "prompt_only"),
        ("user_llm_preferences", "processing_mode", "prompt_only"),
    ]

    altered.clear()
    statements.clear()
    migration.downgrade()

    assert len(statements) == 1
    assert statements[0].lstrip().startswith("CREATE OR REPLACE FUNCTION")
    assert "'prompt_only'" not in statements[0]
    assert re.search(r"(?im)^\s*(UPDATE|INSERT|DELETE)\s+", statements[0]) is None
    assert altered == [
        ("ai_summary_prompts", "processing_mode", "cloud_assisted"),
        ("uploaded_files", "processing_mode", "cloud_assisted"),
        ("user_llm_preferences", "processing_mode", None),
    ]


def test_ingestion_job_factory_is_strict_only_and_snapshot_bound() -> None:
    upload_id = uuid4()
    user_id = uuid4()
    manifest = json.loads(json.dumps(asdict(_manifest())))
    strict = ProcessingSnapshot(
        mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
        manifest_snapshot=manifest,
        manifest_sha256=None,
        schema_version=EXTRACTION_SCHEMA_VERSION,
    )
    from app.services.local_ai.manifest import canonicalize_manifest_snapshot

    canonical, digest = canonicalize_manifest_snapshot(manifest)
    strict = ProcessingSnapshot(
        mode=strict.mode,
        manifest_snapshot=canonical,
        manifest_sha256=digest,
        schema_version=strict.schema_version,
    )
    cloud = ProcessingSnapshot(
        mode=ProcessingMode.CLOUD_ASSISTED,
        manifest_snapshot=None,
        manifest_sha256=None,
        schema_version=None,
    )

    job = build_ingestion_job(
        upload_id=upload_id,
        user_id=user_id,
        snapshot=strict,
    )

    assert job is not None
    assert job.upload_id == upload_id
    assert job.user_id == user_id
    assert job.manifest_snapshot == canonical
    assert job.manifest_sha256 == digest
    assert job.status == "queued"
    assert (
        build_ingestion_job(
            upload_id=upload_id,
            user_id=user_id,
            snapshot=cloud,
        )
        is None
    )


@pytest.mark.asyncio
async def test_processing_mode_preference_round_trips_through_settings(client) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="processing-mode-settings@example.com",
    )

    update = await client.put(
        "/api/v1/settings/llm/routing",
        json={"processing_mode": "validated_strict_local"},
        headers=headers,
    )
    response = await client.get("/api/v1/settings/llm", headers=headers)

    assert update.status_code == 200
    assert response.status_code == 200
    assert response.json()["routing"]["processing_mode"] == "validated_strict_local"


@pytest.mark.asyncio
async def test_custom_local_mode_rejects_nonlocal_effective_routes(client) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="custom-local-cloud-route@example.com",
    )

    update = await client.put(
        "/api/v1/settings/llm/routing",
        json={"processing_mode": "custom_local"},
        headers=headers,
    )
    response = await client.get("/api/v1/settings/llm", headers=headers)

    assert update.status_code == 409
    assert response.status_code == 200
    assert response.json()["routing"]["processing_mode"] == "prompt_only"


@pytest.mark.asyncio
async def test_custom_local_mode_accepts_loopback_only_routes(client) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="custom-local-loopback@example.com",
    )
    provider = await client.put(
        "/api/v1/settings/llm/providers/ollama",
        json={"base_url": "http://127.0.0.1:11434/v1", "enabled": True},
        headers=headers,
    )

    update = await client.put(
        "/api/v1/settings/llm/routing",
        json={
            "default": "ollama",
            "summary": "ollama",
            "section": "ollama",
            "dedup": "ollama",
            "extraction": "ollama",
            "vision": "ollama",
            "processing_mode": "custom_local",
        },
        headers=headers,
    )
    response = await client.get("/api/v1/settings/llm", headers=headers)

    assert provider.status_code == 200
    assert update.status_code == 200
    assert response.status_code == 200
    assert response.json()["routing"]["processing_mode"] == "custom_local"


@pytest.mark.asyncio
async def test_custom_local_provider_rejects_remote_endpoint(client) -> None:
    headers, _user_id = await auth_headers(
        client,
        email="custom-local-remote-endpoint@example.com",
    )

    response = await client.put(
        "/api/v1/settings/llm/providers/lmstudio",
        json={"base_url": "https://models.example.com/v1"},
        headers=headers,
    )

    assert response.status_code == 400
    assert "loopback" in response.json()["detail"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("processing_mode", "validated_strict_local"),
        ("processing_manifest", {"pack_revision": "different"}),
        ("processing_schema_version", "clinical-document-extraction.v2"),
    ],
)
async def test_persisted_upload_processing_snapshot_is_immutable(
    db_session: AsyncSession,
    field: str,
    changed_value: object,
) -> None:
    user = User(
        email=f"immutable-upload-{field}@example.com",
        password_hash="test",
    )
    db_session.add(user)
    await db_session.flush()
    upload = UploadedFile(
        user_id=user.id,
        filename="snapshot.pdf",
        mime_type="application/pdf",
        file_hash="f" * 64,
        storage_path="/encrypted/snapshot.pdf",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.flush()

    setattr(upload, field, changed_value)
    with pytest.raises(LocalValidationError, match="immutable"):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_database_rejects_raw_upload_processing_snapshot_mutation(
    db_session: AsyncSession,
) -> None:
    user = User(
        email="raw-immutable-upload@example.com",
        password_hash="test",
    )
    db_session.add(user)
    await db_session.flush()
    upload = UploadedFile(
        user_id=user.id,
        filename="snapshot.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/encrypted/snapshot.pdf",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    with pytest.raises(IntegrityError, match="processing snapshot is immutable"):
        await db_session.execute(
            update(UploadedFile)
            .where(UploadedFile.id == upload.id)
            .values(processing_mode="validated_strict_local")
        )
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_explicit_processing_mode_overrides_stored_preference() -> None:
    snapshot = await resolve_new_job_snapshot(
        _FakeDB("prompt_only"),
        uuid4(),
        ProcessingMode.CLOUD_ASSISTED,
    )

    assert snapshot.mode is ProcessingMode.CLOUD_ASSISTED
    assert snapshot.manifest_snapshot is None
    assert snapshot.schema_version is None


@pytest.mark.asyncio
async def test_stored_processing_mode_precedes_prompt_only_default() -> None:
    stored = await resolve_new_job_snapshot(_FakeDB("custom_local"), uuid4())
    defaulted = await resolve_new_job_snapshot(_FakeDB(None), uuid4())

    assert stored.mode is ProcessingMode.CUSTOM_LOCAL
    assert defaulted.mode is ProcessingMode.PROMPT_ONLY


@pytest.mark.asyncio
async def test_no_preference_ingestion_rejects_prompt_only_default() -> None:
    with pytest.raises(
        LocalPolicyError,
        match="prompt_only is not available for document ingestion",
    ):
        await resolve_new_ingestion_snapshot(_FakeDB(None), uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["custom_local", "prompt_only"])
@pytest.mark.parametrize("source", ["explicit", "stored"])
async def test_ingestion_snapshot_never_downgrades_unsupported_modes(
    mode: str,
    source: str,
) -> None:
    with pytest.raises(
        LocalPolicyError,
        match=f"{mode} is not available for document ingestion",
    ):
        await resolve_new_ingestion_snapshot(
            _FakeDB(mode if source == "stored" else None),
            uuid4(),
            mode if source == "explicit" else None,
        )


@pytest.mark.asyncio
async def test_strict_snapshot_requires_enabled_exact_active_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    manifest = _manifest()
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_manifest_path",
        "/locked/manifest.json",
    )
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_model_dir",
        "/verified/models",
    )
    monkeypatch.setattr(
        snapshot_module,
        "load_manifest",
        lambda path, **_kwargs: (
            manifest
            if path == Path("/locked/manifest.json")
            else pytest.fail("unexpected manifest path")
        ),
    )

    class _VerifiedStore:
        def __init__(self, path: Path) -> None:
            assert path == Path("/verified/models")

        def active_manifest(self) -> LocalAIManifest:
            return manifest

        def has_validation_receipt(self, _manifest: LocalAIManifest) -> bool:
            return True

    monkeypatch.setattr(snapshot_module, "ArtifactStore", _VerifiedStore)
    monkeypatch.setattr(
        snapshot_module,
        "PackOperationStore",
        lambda _store: SimpleNamespace(has_nonterminal=lambda: False),
    )
    monkeypatch.setattr(
        snapshot_module,
        "load_release_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    _stub_runtime_identity(monkeypatch, manifest)

    snapshot = await resolve_new_job_snapshot(
        _FakeDB(None),
        uuid4(),
        ProcessingMode.VALIDATED_STRICT_LOCAL,
    )

    assert snapshot.mode is ProcessingMode.VALIDATED_STRICT_LOCAL
    assert snapshot.manifest_snapshot == json.loads(json.dumps(asdict(manifest)))
    assert snapshot.manifest_sha256 is not None
    assert snapshot.schema_version == EXTRACTION_SCHEMA_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "active_matches"),
    [(False, True), (True, False)],
)
async def test_strict_snapshot_fails_closed_when_pack_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    active_matches: bool,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    manifest = _manifest()
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", enabled)
    monkeypatch.setattr(
        snapshot_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )

    class _Store:
        def __init__(self, _path: Path) -> None:
            pass

        def active_manifest(self) -> LocalAIManifest | None:
            return manifest if active_matches else None

        def has_validation_receipt(self, _manifest: LocalAIManifest) -> bool:
            return True

    monkeypatch.setattr(snapshot_module, "ArtifactStore", _Store)
    monkeypatch.setattr(
        snapshot_module,
        "PackOperationStore",
        lambda _store: SimpleNamespace(has_nonterminal=lambda: False),
    )

    with pytest.raises(LocalPolicyError, match="unavailable"):
        await resolve_new_job_snapshot(
            _FakeDB(None),
            uuid4(),
            ProcessingMode.VALIDATED_STRICT_LOCAL,
        )


@pytest.mark.asyncio
async def test_strict_snapshot_is_revalidated_under_lifecycle_lock_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    _enable_strict_pack(monkeypatch)
    db = _FakeDB(None)
    snapshot = await resolve_new_job_snapshot(
        db,
        uuid4(),
        ProcessingMode.VALIDATED_STRICT_LOCAL,
    )
    lock_calls = 0

    async def acquire_lock(_db: object) -> None:
        nonlocal lock_calls
        lock_calls += 1

    monkeypatch.setattr(
        snapshot_module,
        "acquire_local_ai_lifecycle_lock",
        acquire_lock,
    )

    await revalidate_strict_snapshot_admission(db, snapshot)

    assert lock_calls == 1


@pytest.mark.asyncio
async def test_strict_snapshot_rejects_hash_valid_active_pack_without_runtime_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    manifest = _manifest()
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module,
        "load_manifest",
        lambda _path, **_kwargs: manifest,
    )

    class _HashOnlyStore:
        def __init__(self, _path: Path) -> None:
            pass

        def active_manifest(self) -> LocalAIManifest:
            return manifest

        def has_validation_receipt(self, _manifest: LocalAIManifest) -> bool:
            return False

    monkeypatch.setattr(snapshot_module, "ArtifactStore", _HashOnlyStore)
    monkeypatch.setattr(
        snapshot_module,
        "PackOperationStore",
        lambda _store: SimpleNamespace(has_nonterminal=lambda: False),
    )

    with pytest.raises(LocalPolicyError, match="unavailable"):
        await resolve_new_job_snapshot(
            _FakeDB(None),
            uuid4(),
            ProcessingMode.VALIDATED_STRICT_LOCAL,
        )


@pytest.mark.asyncio
async def test_strict_snapshot_rejects_nonterminal_pack_lifecycle_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.processing_snapshot as snapshot_module

    _enable_strict_pack(monkeypatch)
    monkeypatch.setattr(
        snapshot_module,
        "PackOperationStore",
        lambda _store: SimpleNamespace(has_nonterminal=lambda: True),
    )

    with pytest.raises(LocalPolicyError, match="unavailable"):
        await resolve_new_job_snapshot(
            _FakeDB(None),
            uuid4(),
            ProcessingMode.VALIDATED_STRICT_LOCAL,
        )


@pytest.mark.asyncio
async def test_unstructured_upload_locks_strict_snapshot_before_enqueue(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="strict-upload-snapshot@example.com",
    )
    manifest = _enable_strict_pack(monkeypatch)

    response = await client.post(
        "/api/v1/upload/unstructured",
        headers=headers,
        data={"processing_mode": "validated_strict_local"},
        files={
            "file": (
                "record.pdf",
                b"%PDF-1.7\n1 0 obj\n%%EOF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 202
    upload_id = UUID(response.json()["upload_id"])
    upload = (
        await db_session.execute(
            select(UploadedFile).where(
                UploadedFile.id == upload_id,
                UploadedFile.user_id == UUID(user_id),
            )
        )
    ).scalar_one()
    job = (
        await db_session.execute(
            select(LocalAIJob).where(LocalAIJob.upload_id == upload.id)
        )
    ).scalar_one()
    assert upload.processing_mode == "validated_strict_local"
    assert upload.processing_manifest == json.loads(json.dumps(asdict(manifest)))
    assert upload.processing_schema_version == EXTRACTION_SCHEMA_VERSION
    assert job.manifest_snapshot == upload.processing_manifest
    assert job.status == "queued"

    preference_change = await client.put(
        "/api/v1/settings/llm/routing",
        headers=headers,
        json={"processing_mode": "cloud_assisted"},
    )
    await db_session.refresh(upload)
    await db_session.refresh(job)

    assert preference_change.status_code == 200
    assert upload.processing_mode == "validated_strict_local"
    assert job.processing_mode == "validated_strict_local"
    assert job.manifest_snapshot == upload.processing_manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["custom_local", "prompt_only"])
async def test_unstructured_upload_rejects_non_ingestion_modes_without_cloud_fallback(
    client,
    mode: str,
) -> None:
    headers, _user_id = await auth_headers(
        client,
        email=f"unsupported-ingestion-{mode}@example.com",
    )

    response = await client.post(
        "/api/v1/upload/unstructured",
        headers=headers,
        data={"processing_mode": mode},
        files={
            "file": (
                "record.pdf",
                b"%PDF-1.7\n1 0 obj\n%%EOF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        f"{mode} is not available for document ingestion."
    )


@pytest.mark.asyncio
async def test_batch_upload_locks_one_strict_job_per_file(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers, user_id = await auth_headers(
        client,
        email="strict-batch-snapshot@example.com",
    )
    manifest = _enable_strict_pack(monkeypatch)

    response = await client.post(
        "/api/v1/upload/unstructured-batch",
        headers=headers,
        data={"processing_mode": "validated_strict_local"},
        files=[
            (
                "files",
                (
                    "first.pdf",
                    b"%PDF-1.7\n1 0 obj\n%%EOF",
                    "application/pdf",
                ),
            ),
            (
                "files",
                (
                    "second.pdf",
                    b"%PDF-1.7\n2 0 obj\n%%EOF",
                    "application/pdf",
                ),
            ),
        ],
    )

    assert response.status_code == 202
    upload_ids = [UUID(item["upload_id"]) for item in response.json()["uploads"]]
    uploads = (
        (
            await db_session.execute(
                select(UploadedFile).where(
                    UploadedFile.id.in_(upload_ids),
                    UploadedFile.user_id == UUID(user_id),
                )
            )
        )
        .scalars()
        .all()
    )
    jobs = (
        (
            await db_session.execute(
                select(LocalAIJob).where(LocalAIJob.upload_id.in_(upload_ids))
            )
        )
        .scalars()
        .all()
    )

    assert len(uploads) == 2
    assert len(jobs) == 2
    assert all(upload.processing_mode == "validated_strict_local" for upload in uploads)
    assert all(
        upload.processing_manifest == json.loads(json.dumps(asdict(manifest)))
        for upload in uploads
    )
    assert all(job.status == "queued" for job in jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "filename", "content_type"),
    [
        ("/api/v1/upload", "bundle.json", "application/fhir+json"),
        ("/api/v1/upload/epic-export", "epic.zip", "application/zip"),
    ],
)
async def test_structured_upload_passes_strict_snapshot_to_coordinator(
    client,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    filename: str,
    content_type: str,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    headers, _user_id = await auth_headers(
        client,
        email=f"strict-structured-{filename}@example.com",
    )
    manifest = _enable_strict_pack(monkeypatch)
    captured: dict[str, object] = {}

    async def _fake_ingest_file(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "upload_id": str(uuid4()),
            "status": "completed",
            "records_inserted": 0,
            "errors": [],
            "unstructured_uploads": [],
        }

    monkeypatch.setattr(coordinator, "ingest_file", _fake_ingest_file)

    response = await client.post(
        route,
        headers=headers,
        data={"processing_mode": "validated_strict_local"},
        files={"file": (filename, b"synthetic structured payload", content_type)},
    )

    assert response.status_code == 202
    assert captured["processing_mode"] == "validated_strict_local"
    assert captured["processing_manifest"] == json.loads(json.dumps(asdict(manifest)))
    assert captured["processing_schema_version"] == EXTRACTION_SCHEMA_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["custom_local", "prompt_only"])
async def test_structured_upload_rejects_unsupported_processing_mode_before_ingestion(
    client,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    headers, _user_id = await auth_headers(
        client,
        email=f"unsupported-structured-{mode}@example.com",
    )

    async def _unexpected_ingest(**_kwargs: object) -> dict[str, object]:
        pytest.fail("unsupported processing mode reached ingestion")

    monkeypatch.setattr(coordinator, "ingest_file", _unexpected_ingest)

    response = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": mode},
        files={
            "file": (
                "bundle.json",
                b'{"resourceType":"Bundle","entry":[]}',
                "application/fhir+json",
            )
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        f"{mode} is not available for document ingestion."
    )


@pytest.mark.asyncio
async def test_reprocess_queues_stored_ciphertext_under_new_strict_revision(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers, user_id_text = await auth_headers(
        client,
        email="reprocess-new-revision@example.com",
    )
    user_id = UUID(user_id_text)
    manifest = _enable_strict_pack(monkeypatch)
    upload_root = Path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    source_path = upload_root / f"{uuid4()}.pdf"
    source_path.write_bytes(b"".join(encrypt_stream([b"%PDF-1.7\nsynthetic\n%%EOF"])))
    source = UploadedFile(
        user_id=user_id,
        filename="record.pdf",
        mime_type="application/pdf",
        file_size_bytes=27,
        file_hash="b" * 64,
        storage_path=str(source_path),
        ingestion_status="completed",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(source)
    await db_session.commit()

    response = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        headers=headers,
        json={"processing_mode": "validated_strict_local"},
    )

    assert response.status_code == 202
    reprocessed_id = UUID(response.json()["upload_id"])
    reprocessed = await db_session.get(UploadedFile, reprocessed_id)
    job = (
        await db_session.execute(
            select(LocalAIJob).where(LocalAIJob.upload_id == reprocessed_id)
        )
    ).scalar_one()
    assert reprocessed is not None
    assert reprocessed.id != source.id
    assert reprocessed.storage_path == source.storage_path
    assert reprocessed.file_hash == source.file_hash
    assert reprocessed.ingestion_status == "pending_extraction"
    assert reprocessed.ingestion_progress == {"reprocesses_upload_id": str(source.id)}
    assert reprocessed.processing_mode == "validated_strict_local"
    assert reprocessed.processing_manifest == json.loads(json.dumps(asdict(manifest)))
    assert reprocessed.processing_schema_version == EXTRACTION_SCHEMA_VERSION
    assert job.status == "queued"
    assert job.manifest_snapshot == reprocessed.processing_manifest


@pytest.mark.asyncio
async def test_reprocess_rejects_same_processing_revision(
    client,
    db_session: AsyncSession,
) -> None:
    headers, user_id_text = await auth_headers(
        client,
        email="reprocess-same-revision@example.com",
    )
    upload_root = Path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    source_path = upload_root / f"{uuid4()}.pdf"
    source_path.write_bytes(b"".join(encrypt_stream([b"%PDF-1.7\n%%EOF"])))
    source = UploadedFile(
        user_id=UUID(user_id_text),
        filename="record.pdf",
        mime_type="application/pdf",
        file_size_bytes=15,
        file_hash="c" * 64,
        storage_path=str(source_path),
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(source)
    await db_session.commit()

    response = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        headers=headers,
        json={"processing_mode": "cloud_assisted"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Upload already uses the requested processing revision."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_source", [False, True])
async def test_reprocess_reuses_one_child_for_each_canonical_target_revision(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    duplicate_source: bool,
) -> None:
    """Repeated requests cannot queue sibling jobs for one source/revision."""
    headers, user_id_text = await auth_headers(
        client,
        email=f"canonical-reprocess-{duplicate_source}@example.com",
    )
    user_id = UUID(user_id_text)
    _enable_strict_pack(monkeypatch)
    upload_root = Path(settings.upload_dir)
    upload_root.mkdir(parents=True, exist_ok=True)
    root_path = upload_root / f"{uuid4()}.pdf"
    root_path.write_bytes(b"".join(encrypt_stream([b"%PDF-1.7\n%%EOF"])))
    root = UploadedFile(
        user_id=user_id,
        filename="record.pdf",
        mime_type="application/pdf",
        file_size_bytes=15,
        file_hash="canonical-reprocess-hash",
        storage_path=str(root_path),
        ingestion_status="completed",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(root)
    await db_session.flush()
    source = root
    if duplicate_source:
        duplicate_path = upload_root / f"{uuid4()}.pdf"
        duplicate_path.write_bytes(b"".join(encrypt_stream([b"%PDF-1.7\n%%EOF"])))
        source = UploadedFile(
            user_id=user_id,
            filename="record.pdf",
            mime_type="application/pdf",
            file_size_bytes=15,
            file_hash=root.file_hash,
            storage_path=str(duplicate_path),
            ingestion_status="duplicate_file",
            ingestion_progress={"duplicate_of": str(root.id)},
            file_category="unstructured",
            processing_mode="cloud_assisted",
        )
        db_session.add(source)
    await db_session.commit()

    first = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        headers=headers,
        json={"processing_mode": "validated_strict_local"},
    )
    second = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        headers=headers,
        json={"processing_mode": "validated_strict_local"},
    )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["upload_id"] == first.json()["upload_id"]
    children = (
        (
            await db_session.execute(
                select(UploadedFile).where(
                    UploadedFile.user_id == user_id,
                    UploadedFile.ingestion_progress["reprocesses_upload_id"].astext
                    == str(root.id),
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(children) == 1
    jobs = (
        (
            await db_session.execute(
                select(LocalAIJob).where(LocalAIJob.upload_id == children[0].id)
            )
        )
        .scalars()
        .all()
    )
    assert len(jobs) == 1


@pytest.mark.asyncio
async def test_custom_local_delete_rejects_remote_environment_fallback(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the local override may not silently expose a remote endpoint."""
    headers, user_id_text = await auth_headers(
        client,
        email="custom-local-delete-fallback@example.com",
    )
    monkeypatch.setattr(settings, "ollama_base_url", "https://remote.example/v1")
    for name in ("ollama", "lmstudio"):
        response = await client.put(
            f"/api/v1/settings/llm/providers/{name}",
            headers=headers,
            json={"base_url": "http://127.0.0.1:11434/v1", "enabled": True},
        )
        assert response.status_code == 200
    routing = await client.put(
        "/api/v1/settings/llm/routing",
        headers=headers,
        json={
            "default": "ollama",
            "summary": "ollama",
            "section": "ollama",
            "dedup": "ollama",
            "extraction": "ollama",
            "vision": "ollama",
            "processing_mode": "custom_local",
        },
    )
    assert routing.status_code == 200

    deleted = await client.delete(
        "/api/v1/settings/llm/providers/ollama",
        headers=headers,
    )

    assert deleted.status_code == 409
    row = (
        await db_session.execute(
            select(LLMProviderConfig).where(
                LLMProviderConfig.user_id == UUID(user_id_text),
                LLMProviderConfig.provider == "ollama",
            )
        )
    ).scalar_one()
    assert row.base_url == "http://127.0.0.1:11434/v1"


@pytest.mark.asyncio
async def test_runtime_custom_local_config_fails_closed_for_remote_environment(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime config loading must reject a persisted custom-local remote route."""
    user = User(email="runtime-custom-local@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    from app.models.llm_settings import UserLLMPreferences

    db_session.add(
        UserLLMPreferences(
            user_id=user.id,
            default_provider="ollama",
            summary_provider="ollama",
            section_provider="ollama",
            dedup_provider="ollama",
            extraction_provider="ollama",
            vision_provider="ollama",
            processing_mode="custom_local",
        )
    )
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "https://remote.example/v1")
    await db_session.commit()

    with pytest.raises(LocalPolicyError, match="loopback"):
        await load_llm_config(db_session, user.id)


@pytest.mark.asyncio
async def test_custom_local_rejects_disabling_an_effective_provider(client) -> None:
    """Provider updates cannot leave persisted custom-local routing disabled."""
    headers, _user_id = await auth_headers(
        client,
        email="custom-local-disable-provider@example.com",
    )
    configured = await client.put(
        "/api/v1/settings/llm/providers/ollama",
        headers=headers,
        json={"base_url": "http://127.0.0.1:11434/v1", "enabled": True},
    )
    routing = await client.put(
        "/api/v1/settings/llm/routing",
        headers=headers,
        json={
            "default": "ollama",
            "summary": "ollama",
            "section": "ollama",
            "dedup": "ollama",
            "extraction": "ollama",
            "vision": "ollama",
            "processing_mode": "custom_local",
        },
    )
    disabled = await client.put(
        "/api/v1/settings/llm/providers/ollama",
        headers=headers,
        json={"enabled": False},
    )

    assert configured.status_code == 200
    assert routing.status_code == 200
    assert disabled.status_code == 409
