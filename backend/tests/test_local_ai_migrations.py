"""Regression tests for strict-local Alembic migrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
