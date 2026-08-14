"""Regression tests for strict-local Alembic migrations."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_ATTESTED_MIGRATION = (
    _BACKEND_ROOT
    / "alembic"
    / "versions"
    / "c5d6e7f8a9b0_require_attested_local_ai_manifest.py"
)
_CREATE_ALL_DDL = _BACKEND_ROOT / "app" / "models" / "local_ai_ddl.py"
_MIGRATION_DATABASE = "medtimeline_runtime_attestation_migrations_ci"
_LEGACY_USER_ID = uuid.UUID("00000000-0000-0000-0000-00000000d001")
_LEGACY_JOB_IDS = {
    name: uuid.UUID(f"00000000-0000-0000-0000-{index:012x}")
    for index, name in enumerate(
        (
            "exact_queued",
            "exact_processing",
            "requeue",
            "promotion",
            "completed",
            "cancelled",
            "changed_manifest",
            "changed_digest",
            "altered_progress",
            "altered_failure",
            "malformed_role",
            "malformed_files",
            "malformed_artifact",
        ),
        start=0xD201,
    )
}
_LEGACY_UPLOAD_IDS = {
    name: uuid.UUID(f"00000000-0000-0000-0000-{index:012x}")
    for index, name in enumerate(_LEGACY_JOB_IDS, start=0xD101)
}


def _python_string_containing(path: Path, marker: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and marker in node.value
    ]
    assert len(matches) == 1
    return matches[0]


def _assigned_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                spec = importlib.util.spec_from_file_location(
                    f"_{path.stem}_{name}",
                    path,
                )
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                value = getattr(module, name)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing string assignment: {name}")


def _normalized_sql(value: str, *, sqlalchemy_ddl: bool = False) -> str:
    if sqlalchemy_ddl:
        value = value.replace("%%", "%")
    value = " ".join(value.split())
    return re.sub(r"\s*([(),])\s*", r"\1", value)


def _artifact(role: str) -> dict:
    marker = {"ocr": "a", "extraction": "b", "summary": "c"}[role]
    return {
        "role": role,
        "repository": f"owner/{role}",
        "revision": "0" * 40,
        "quantization": "4bit",
        "license": "apache-2.0",
        "attribution": f"synthetic {role}",
        "decode_limits": {"max_input_tokens": 4096, "max_output_tokens": 1024},
        "files": [
            {
                "path": f"{role}/model.safetensors",
                "sha256": marker * 64,
                "size": 10,
            }
        ],
    }


def _manifest(*, schema_version: int = 2) -> dict:
    runtime = {"name": "mlx-vlm", "version": "0.5.0"}
    if schema_version == 2:
        runtime.update(
            worker_identity_scheme="local-ai-worker-bundle.v1",
            worker_bundle_sha256="d" * 64,
        )
    return {
        "schema_version": schema_version,
        "pack_revision": f"apple-m4-16gb-v{schema_version}",
        "platform": "apple_silicon",
        "runtime": runtime,
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": [_artifact(role) for role in ("ocr", "extraction", "summary")],
    }


def _digest(manifest: dict) -> str:
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


async def _database_manifest_is_valid(db_session, manifest: dict) -> bool:
    return bool(
        (
            await db_session.execute(
                text("SELECT local_ai_manifest_is_valid(CAST(:payload AS jsonb))"),
                {"payload": json.dumps(manifest)},
            )
        ).scalar_one()
    )


async def _database_legacy_manifest_is_valid(db_session, manifest: dict) -> bool:
    return bool(
        (
            await db_session.execute(
                text(
                    "SELECT local_ai_legacy_manifest_is_valid("
                    "CAST(:payload AS jsonb))"
                ),
                {"payload": json.dumps(manifest)},
            )
        ).scalar_one()
    )


def _literal_migration_database_url() -> str:
    value = os.environ["DATABASE_URL"]
    if make_url(value).database != _MIGRATION_DATABASE:
        raise AssertionError("migration tests require the literal disposable DB")
    return value


@pytest_asyncio.fixture
async def migration_engine() -> AsyncIterator[AsyncEngine]:
    value = os.environ.get("DATABASE_URL")
    if value is None or make_url(value).database != _MIGRATION_DATABASE:
        pytest.skip("requires the literal disposable migration database")
    engine = create_async_engine(value, echo=False)
    try:
        yield engine
    finally:
        await engine.dispose()


async def seed_preupgrade_legacy_rows() -> None:
    """Seed only the literal disposable migration DB while the v1 guard is active."""

    engine = create_async_engine(_literal_migration_database_url(), echo=False)
    canonical = _manifest(schema_version=1)
    manifests = {name: deepcopy(canonical) for name in _LEGACY_JOB_IDS}
    manifests["malformed_role"]["artifacts"][0]["role"] = "summary"
    manifests["malformed_files"]["artifacts"][0]["files"] = []
    manifests["malformed_artifact"]["artifacts"][0]["extra"] = True
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO users (id, email, email_hmac, password_hash)
                    VALUES (:id, decode('00', 'hex'), :email_hmac, 'x')
                    """
                ),
                {"id": _LEGACY_USER_ID, "email_hmac": "d" * 64},
            )
            for name, upload_id in _LEGACY_UPLOAD_IDS.items():
                await connection.execute(
                    text(
                        """
                        INSERT INTO uploaded_files (
                            id, user_id, filename, mime_type, file_hash,
                            storage_path, processing_mode, processing_manifest,
                            processing_schema_version
                        ) VALUES (
                            :id, :user_id, :filename, 'application/pdf',
                            :file_hash, :storage_path, 'validated_strict_local',
                            CAST(:processing_manifest AS jsonb), '1'
                        )
                        """
                    ),
                    {
                        "id": upload_id,
                        "user_id": _LEGACY_USER_ID,
                        "filename": f"{name}.pdf",
                        "file_hash": f"{len(name):064x}",
                        "storage_path": f"/synthetic/{name}.pdf",
                        "processing_manifest": json.dumps(
                            {"pack_revision": "apple-m4-16gb-v1"}
                        ),
                    },
                )

            for name in _LEGACY_JOB_IDS:
                if name == "malformed_role":
                    await connection.execute(
                        text(
                            "ALTER TABLE local_ai_jobs DISABLE TRIGGER "
                            "trg_local_ai_jobs_immutable_identity"
                        )
                    )
                await connection.execute(
                    text(
                        """
                        INSERT INTO local_ai_jobs (
                            id, user_id, upload_id, kind, processing_mode,
                            manifest_snapshot, manifest_sha256, status, stage
                        ) VALUES (
                            :id, :user_id, :upload_id, 'ingestion',
                            'validated_strict_local', CAST(:manifest AS jsonb),
                            :digest, :status, 'preflight'
                        )
                        """
                    ),
                    {
                        "id": _LEGACY_JOB_IDS[name],
                        "user_id": _LEGACY_USER_ID,
                        "upload_id": _LEGACY_UPLOAD_IDS[name],
                        "manifest": json.dumps(manifests[name]),
                        "digest": _digest(manifests[name]),
                        "status": (
                            "processing" if name == "exact_processing" else "queued"
                        ),
                    },
                )
            await connection.execute(
                text(
                    "ALTER TABLE local_ai_jobs ENABLE TRIGGER "
                    "trg_local_ai_jobs_immutable_identity"
                )
            )
    finally:
        await engine.dispose()


def _load_unique_job_migration() -> ModuleType:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "a8b9c0d1e2f3_unique_ingestion_local_ai_job.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_a8_unique_ingestion_local_ai_job",
        migration_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manual_zip_migration() -> ModuleType:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "b4c5d6e7f8a9_add_manual_zip_extraction_flag.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_b4_manual_zip_extraction_flag",
        migration_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unique_ingestion_job_migration_tolerates_create_all_index(
    monkeypatch,
) -> None:
    """The create_all-managed test schema may already contain the model index."""

    migration = _load_unique_job_migration()
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    assert len(statements) == 1
    assert "CREATE UNIQUE INDEX IF NOT EXISTS" in statements[0]
    assert "uq_local_ai_jobs_ingestion_upload" in statements[0]


def test_manual_zip_migration_backfills_unreleased_children(monkeypatch) -> None:
    """Pre-upgrade staged/pending ZIP children retain their manual hold."""

    migration = _load_manual_zip_migration()
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "add_column", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration.upgrade()

    assert len(statements) == 1
    statement = statements[0]
    assert "manual_extraction_required = true" in statement
    assert "staging_extraction" in statement
    assert "pending_extraction" in statement
    assert "medtimeline-zip-set-" in statement


def test_attested_manifest_guard_is_an_exact_migration_create_all_clone() -> None:
    migration_sql = _assigned_string(
        _ATTESTED_MIGRATION,
        "_ATTESTED_MANIFEST_FUNCTION",
    )
    create_all_sql = _python_string_containing(
        _CREATE_ALL_DDL,
        "CREATE OR REPLACE FUNCTION local_ai_manifest_is_valid(payload jsonb)",
    )

    assert _normalized_sql(migration_sql) == _normalized_sql(
        create_all_sql,
        sqlalchemy_ddl=True,
    )
    normalized = _normalized_sql(migration_sql)
    assert "payload->>'schema_version' <> '2'" in normalized
    assert (
        "payload->'runtime',ARRAY[ 'name','version',"
        "'worker_identity_scheme','worker_bundle_sha256' ]"
    ) in normalized
    assert (
        "payload->'runtime'->>'worker_identity_scheme' "
        "<> 'local-ai-worker-bundle.v1'"
    ) in normalized
    assert "worker_bundle_sha256' !~ '^[0-9a-f]{64}$'" in normalized
    assert "file_path LIKE '%//%'" in normalized
    assert "file_path LIKE '%%//%%'" in create_all_sql


def test_attested_job_trigger_is_an_exact_migration_create_all_clone() -> None:
    migration_sql = _assigned_string(
        _ATTESTED_MIGRATION,
        "_ATTESTED_JOB_IDENTITY_FUNCTION",
    )
    create_all_sql = _python_string_containing(
        _CREATE_ALL_DDL,
        "CREATE OR REPLACE FUNCTION enforce_local_ai_job_identity()",
    )

    assert _normalized_sql(migration_sql) == _normalized_sql(create_all_sql)
    normalized = _normalized_sql(migration_sql)
    required_legacy_predicates = (
        "TG_OP = 'UPDATE'",
        "local_ai_legacy_manifest_is_valid(OLD.manifest_snapshot)",
        "OLD.status IN ('queued', 'processing')",
        "NEW.status = 'failed'",
        "NEW.stage = 'failed'",
        "NEW.progress = '{\"stage\": \"failed\"}'::jsonb",
        "NEW.failure = '{\"stage\": \"failed\", \"code\": "
        "\"runtime_identity_required\", \"retryable\": false, "
        "\"checkpoint_preserved\": false, "
        "\"cloud_fallback_attempted\": false}'::jsonb",
        "NEW.completed_at IS NOT NULL",
        "NEW.manifest_snapshot IS NOT DISTINCT FROM OLD.manifest_snapshot",
        "NEW.manifest_sha256 IS NOT DISTINCT FROM OLD.manifest_sha256",
        "NEW.cancel_requested IS NOT DISTINCT FROM OLD.cancel_requested",
        "NEW.audit_metadata IS NOT DISTINCT FROM OLD.audit_metadata",
        "NEW.started_at IS NOT DISTINCT FROM OLD.started_at",
    )
    for predicate in required_legacy_predicates:
        assert _normalized_sql(predicate) in normalized


def test_legacy_validator_is_a_full_migration_create_all_clone() -> None:
    migration_sql = _assigned_string(
        _ATTESTED_MIGRATION,
        "_LEGACY_MANIFEST_FUNCTION",
    )
    create_all_sql = _python_string_containing(
        _CREATE_ALL_DDL,
        "CREATE OR REPLACE FUNCTION local_ai_legacy_manifest_is_valid(",
    )

    assert _normalized_sql(migration_sql) == _normalized_sql(create_all_sql)
    normalized = _normalized_sql(migration_sql)
    assert "payload->>'schema_version' <> '1'" in normalized
    assert "payload->'runtime',ARRAY['name','version']" in normalized
    assert "RETURN local_ai_manifest_is_valid(attested_payload)" in normalized


def test_attested_migration_downgrade_restores_exact_v1_guards() -> None:
    source = _ATTESTED_MIGRATION.read_text(encoding="utf-8")
    v1_manifest = _normalized_sql(
        _assigned_string(_ATTESTED_MIGRATION, "_V1_MANIFEST_FUNCTION")
    )
    v1_trigger = _normalized_sql(
        _assigned_string(_ATTESTED_MIGRATION, "_V1_JOB_IDENTITY_FUNCTION")
    )

    assert 'revision: str = "c5d6e7f8a9b0"' in source
    assert 'down_revision: str = "b0c1d2e3f4a5"' in source
    assert "payload->>'schema_version' <> '1'" in v1_manifest
    assert "payload->'runtime',ARRAY['name','version']" in v1_manifest
    assert "worker_identity_scheme" not in v1_manifest
    assert "runtime_identity_required" not in v1_trigger

    tree = ast.parse(source, filename=str(_ATTESTED_MIGRATION))
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "op.execute(_ATTESTED_MANIFEST_FUNCTION)" in functions["upgrade"]
    assert "op.execute(_LEGACY_MANIFEST_FUNCTION)" in functions["upgrade"]
    assert "op.execute(_ATTESTED_JOB_IDENTITY_FUNCTION)" in functions["upgrade"]
    assert "op.execute(_V1_MANIFEST_FUNCTION)" in functions["downgrade"]
    assert "op.execute(_V1_JOB_IDENTITY_FUNCTION)" in functions["downgrade"]


async def test_migrated_guard_accepts_only_exact_attested_v2_runtime(
    migration_engine: AsyncEngine,
) -> None:
    valid = _manifest()
    invalid = [_manifest(schema_version=1)]
    missing_digest = deepcopy(valid)
    del missing_digest["runtime"]["worker_bundle_sha256"]
    invalid.append(missing_digest)
    uppercase_digest = deepcopy(valid)
    uppercase_digest["runtime"]["worker_bundle_sha256"] = "D" * 64
    invalid.append(uppercase_digest)
    extra_runtime = deepcopy(valid)
    extra_runtime["runtime"]["extra"] = "rejected"
    invalid.append(extra_runtime)

    async with migration_engine.connect() as connection:
        assert await _database_manifest_is_valid(connection, valid) is True
        for manifest in invalid:
            assert await _database_manifest_is_valid(connection, manifest) is False

        legacy = _manifest(schema_version=1)
        for mutation in ("role", "files", "artifact_shape"):
            malformed = deepcopy(legacy)
            if mutation == "role":
                malformed["artifacts"][0]["role"] = "summary"
            elif mutation == "files":
                malformed["artifacts"][0]["files"] = []
            else:
                malformed["artifacts"][0]["extra"] = True
            assert (
                await _database_legacy_manifest_is_valid(connection, malformed)
                is False
            )


@pytest.mark.parametrize("initial_status", ("queued", "processing"))
async def test_migrated_guard_allows_only_exact_legacy_terminal_failure(
    migration_engine: AsyncEngine,
    initial_status: str,
) -> None:
    legacy = _manifest(schema_version=1)
    digest = _digest(legacy)
    job_id = _LEGACY_JOB_IDS[f"exact_{initial_status}"]
    async with migration_engine.begin() as connection:
        await connection.execute(
            text(
                """
                UPDATE local_ai_jobs
                SET status = 'failed', stage = 'failed',
                    progress = '{"stage": "failed"}'::jsonb,
                    failure = CAST(:failure AS jsonb), completed_at = now()
                WHERE id = :job_id
                """
            ),
            {
                "job_id": job_id,
                "failure": json.dumps(
                    {
                        "stage": "failed",
                        "code": "runtime_identity_required",
                        "retryable": False,
                        "checkpoint_preserved": False,
                        "cloud_fallback_attempted": False,
                    }
                ),
            },
        )
    async with migration_engine.connect() as connection:
        stored = (
            await connection.execute(
                text(
                    "SELECT status, stage, manifest_snapshot, manifest_sha256 "
                    "FROM local_ai_jobs WHERE id = :job_id"
                ),
                {"job_id": job_id},
            )
        ).one()
    assert (stored.status, stored.stage) == ("failed", "failed")
    assert stored.manifest_snapshot == legacy
    assert stored.manifest_sha256 == digest


_REJECTED_LEGACY_UPDATES = (
    ("requeue", "status = 'queued'"),
    ("promotion", "status = 'processing', stage = 'ocr'"),
    ("completed", "status = 'completed', stage = 'completed', completed_at = now()"),
    ("cancelled", "status = 'cancelled', stage = 'cancelled', completed_at = now()"),
    (
        "changed_manifest",
        "manifest_snapshot = jsonb_set(manifest_snapshot, '{pack_revision}', "
        "'\"tampered\"'::jsonb)",
    ),
    ("changed_digest", "manifest_sha256 = repeat('e', 64)"),
    (
        "altered_progress",
        "status = 'failed', stage = 'failed', "
        "progress = '{\"stage\": \"failed\", \"extra\": 1}'::jsonb, "
        "failure = '{\"stage\": \"failed\", "
        "\"code\": \"runtime_identity_required\", \"retryable\": false, "
        "\"checkpoint_preserved\": false, "
        "\"cloud_fallback_attempted\": false}'::jsonb, completed_at = now()",
    ),
    (
        "altered_failure",
        "status = 'failed', stage = 'failed', "
        "progress = '{\"stage\": \"failed\"}'::jsonb, "
        "failure = '{\"stage\": \"failed\", "
        "\"code\": \"runtime_identity_required\", \"retryable\": true, "
        "\"checkpoint_preserved\": false, "
        "\"cloud_fallback_attempted\": false}'::jsonb, completed_at = now()",
    ),
)


@pytest.mark.parametrize(("seed_name", "set_clause"), _REJECTED_LEGACY_UPDATES)
async def test_migrated_guard_rejects_every_other_legacy_update_unchanged(
    migration_engine: AsyncEngine,
    seed_name: str,
    set_clause: str,
) -> None:
    job_id = _LEGACY_JOB_IDS[seed_name]
    before = await _stored_legacy_row(migration_engine, job_id)
    with pytest.raises(IntegrityError):
        async with migration_engine.begin() as connection:
            await connection.execute(
                text(f"UPDATE local_ai_jobs SET {set_clause} WHERE id = :job_id"),
                {"job_id": job_id},
            )
    assert await _stored_legacy_row(migration_engine, job_id) == before


@pytest.mark.parametrize(
    "seed_name",
    ("malformed_role", "malformed_files", "malformed_artifact"),
)
async def test_migrated_guard_rejects_malformed_legacy_terminal_unchanged(
    migration_engine: AsyncEngine,
    seed_name: str,
) -> None:
    job_id = _LEGACY_JOB_IDS[seed_name]
    before = await _stored_legacy_row(migration_engine, job_id)
    with pytest.raises(IntegrityError):
        async with migration_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE local_ai_jobs
                    SET status = 'failed', stage = 'failed',
                        progress = '{"stage": "failed"}'::jsonb,
                        failure = '{"stage": "failed", "code":
                            "runtime_identity_required", "retryable": false,
                            "checkpoint_preserved": false,
                            "cloud_fallback_attempted": false}'::jsonb,
                        completed_at = now()
                    WHERE id = :job_id
                    """
                ),
                {"job_id": job_id},
            )
    assert await _stored_legacy_row(migration_engine, job_id) == before


async def _stored_legacy_row(engine: AsyncEngine, job_id: uuid.UUID):
    async with engine.connect() as connection:
        return (
            await connection.execute(
                text(
                    "SELECT manifest_snapshot, manifest_sha256, status, stage, "
                    "progress, failure, completed_at FROM local_ai_jobs "
                    "WHERE id = :job_id"
                ),
                {"job_id": job_id},
            )
        ).one()


def _load_attested_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_c5_attested_local_ai_manifest",
        _ATTESTED_MIGRATION,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
