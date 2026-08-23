"""Byte-preservation and downgrade-safety tests for the identifier rename."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from cryptography.exceptions import InvalidTag
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import EmailStr, TypeAdapter
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.auth import router as auth_router
from app.config import settings
from app.database import get_db
from app.middleware.encryption import blind_index, encrypt_field
from app.models.base import Base
from app.services.auth_service import hash_password

import app.models  # noqa: F401


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION_PATH = (
    _BACKEND_ROOT / "alembic" / "versions" / "d6e7f8a9b0c1_general_login_identifier.py"
)
_MIGRATION_DATABASE = "medtimeline_issue67_migrations_ci"
_TEST_ENCRYPTION_KEY = "19" * 32
_WRONG_ENCRYPTION_KEY = "91" * 32
_LEGACY_IDENTIFIER = "legacy-migration@example.com"
_DECOMPOSED_LEGACY_IDENTIFIER = "u\u0308ser@example.com"
_NORMALIZED_LEGACY_IDENTIFIER = "üser@example.com"
_NON_EMAIL_IDENTIFIER = "synthetic account name sentinel"
_PASSWORD = "SecurePass123!"
_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000006701")

DOWNGRADE_BLOCKED = (
    "Downgrade blocked: account identifiers are not compatible with the legacy schema."
)
_DOWNGRADE_LOCK = "LOCK TABLE users IN SHARE ROW EXCLUSIVE MODE"
_EMAIL_ADAPTER = TypeAdapter(EmailStr)


class FakeScalarResult:
    def __init__(self, rows: list[tuple[bytes, str]]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[bytes]:
        return iter(row[0] for row in self._rows)


class FakeResult:
    def __init__(self, rows: list[tuple[bytes, str]]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[tuple[bytes, str]]:
        return iter(self._rows)

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self._rows)


class FakeConnection:
    def __init__(self, rows: list[tuple[bytes, str]], events: list[str]) -> None:
        self._rows = rows
        self._events = events

    def execute(self, statement: object) -> FakeResult:
        self._events.append(str(statement))
        return FakeResult(self._rows)


@pytest.fixture
def migration_module() -> ModuleType | None:
    if not _MIGRATION_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_d6_general_login_identifier",
        _MIGRATION_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "database_encryption_key", _TEST_ENCRYPTION_KEY)


@pytest.fixture
def migration_database_url(
    configured_test_key: None,
) -> str:
    del configured_test_key
    value = os.environ.get("DATABASE_URL")
    if value is None or make_url(value).database != _MIGRATION_DATABASE:
        pytest.skip("requires the literal disposable migration database")
    return value


def _require_migration(module: ModuleType | None) -> ModuleType:
    assert module is not None, f"missing migration revision: {_MIGRATION_PATH.name}"
    return module


def _patch_fake_operations(
    migration: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[tuple[bytes, str]],
) -> tuple[list[str], list[tuple[str, tuple[object, ...], dict[str, object]]]]:
    connection_events: list[str] = []
    ddl_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    connection = FakeConnection(rows, connection_events)
    monkeypatch.setattr(migration.op, "get_bind", lambda: connection)
    monkeypatch.setattr(
        migration.op,
        "alter_column",
        lambda *args, **kwargs: ddl_calls.append(("alter_column", args, kwargs)),
    )
    monkeypatch.setattr(
        migration.op,
        "execute",
        lambda *args, **kwargs: ddl_calls.append(("execute", args, kwargs)),
    )
    return connection_events, ddl_calls


def test_upgrade_is_rename_only_and_does_not_read_or_rewrite_identifiers(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _require_migration(migration_module)
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        migration.op,
        "get_bind",
        lambda: pytest.fail("rename-only upgrade must not read account rows"),
    )
    monkeypatch.setattr(
        migration.op,
        "alter_column",
        lambda *args, **kwargs: calls.append(("alter_column", args, kwargs)),
    )
    monkeypatch.setattr(
        migration.op,
        "execute",
        lambda *args, **kwargs: calls.append(("execute", args, kwargs)),
    )

    migration.upgrade()

    assert calls == [
        (
            "alter_column",
            ("users", "email"),
            {"new_column_name": "login_identifier"},
        ),
        (
            "alter_column",
            ("users", "email_hmac"),
            {"new_column_name": "login_identifier_hmac"},
        ),
        (
            "execute",
            (
                "ALTER INDEX ix_users_email_hmac "
                "RENAME TO ix_users_login_identifier_hmac",
            ),
            {},
        ),
    ]
    assert all("UPDATE" not in str(call).upper() for call in calls)


def test_downgrade_preflight_failure_is_content_free_and_precedes_ddl(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    configured_test_key: None,
) -> None:
    del configured_test_key
    migration = _require_migration(migration_module)
    sentinel = "not-an-email-identifier-sentinel"
    connection_events, ddl_calls = _patch_fake_operations(
        migration,
        monkeypatch,
        [(encrypt_field(sentinel), blind_index(sentinel))],
    )

    with pytest.raises(RuntimeError) as exc_info:
        migration.downgrade()

    assert str(exc_info.value) == DOWNGRADE_BLOCKED
    assert ddl_calls == []
    assert connection_events == [
        _DOWNGRADE_LOCK,
        "SELECT login_identifier, login_identifier_hmac FROM users",
    ]
    assert all("UPDATE" not in statement.upper() for statement in connection_events)
    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("missing-key-sentinel"),
        ValueError("wrong-key-sentinel"),
        InvalidTag("decryption-failure-sentinel"),
    ),
)
def test_downgrade_key_or_decryption_failure_is_content_free_and_precedes_ddl(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    migration = _require_migration(migration_module)
    connection_events, ddl_calls = _patch_fake_operations(
        migration,
        monkeypatch,
        [(b"synthetic-ciphertext", "synthetic-hmac")],
    )
    failure_text = str(failure)

    def fail_decryption(_ciphertext: bytes) -> str:
        raise failure

    monkeypatch.setattr(migration, "decrypt_field", fail_decryption)

    with pytest.raises(RuntimeError) as exc_info:
        migration.downgrade()

    assert str(exc_info.value) == DOWNGRADE_BLOCKED
    assert ddl_calls == []
    assert connection_events == [
        _DOWNGRADE_LOCK,
        "SELECT login_identifier, login_identifier_hmac FROM users",
    ]
    assert all("UPDATE" not in statement.upper() for statement in connection_events)
    assert failure_text not in str(exc_info.value)
    assert failure_text not in caplog.text


def test_downgrade_does_not_broaden_the_approved_exception_set(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = _require_migration(migration_module)
    connection_events, ddl_calls = _patch_fake_operations(
        migration,
        monkeypatch,
        [(b"synthetic-ciphertext", "synthetic-hmac")],
    )

    def raise_unexpected(_ciphertext: bytes) -> str:
        raise KeyError("unexpected-programming-error")

    monkeypatch.setattr(migration, "decrypt_field", raise_unexpected)

    with pytest.raises(KeyError, match="unexpected-programming-error"):
        migration.downgrade()

    assert ddl_calls == []
    assert connection_events == [
        _DOWNGRADE_LOCK,
        "SELECT login_identifier, login_identifier_hmac FROM users",
    ]


def test_downgrade_refuses_email_validator_normalization_mismatch_before_ddl(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    configured_test_key: None,
) -> None:
    del configured_test_key
    migration = _require_migration(migration_module)
    normalized = str(_EMAIL_ADAPTER.validate_python(_DECOMPOSED_LEGACY_IDENTIFIER))
    assert normalized == _NORMALIZED_LEGACY_IDENTIFIER
    stored_hmac = blind_index(_DECOMPOSED_LEGACY_IDENTIFIER)
    normalized_hmac = blind_index(normalized)
    assert stored_hmac != normalized_hmac
    connection_events, ddl_calls = _patch_fake_operations(
        migration,
        monkeypatch,
        [(encrypt_field(_DECOMPOSED_LEGACY_IDENTIFIER), stored_hmac)],
    )

    with pytest.raises(RuntimeError) as exc_info:
        migration.downgrade()

    assert str(exc_info.value) == DOWNGRADE_BLOCKED
    assert ddl_calls == []
    assert connection_events == [
        _DOWNGRADE_LOCK,
        "SELECT login_identifier, login_identifier_hmac FROM users",
    ]
    assert all("UPDATE" not in statement.upper() for statement in connection_events)
    for sensitive_value in (
        _DECOMPOSED_LEGACY_IDENTIFIER,
        normalized,
        stored_hmac,
        normalized_hmac,
    ):
        assert sensitive_value not in str(exc_info.value)
        assert sensitive_value not in caplog.text


def test_email_compatible_downgrade_runs_only_inverse_renames_in_order(
    migration_module: ModuleType | None,
    monkeypatch: pytest.MonkeyPatch,
    configured_test_key: None,
) -> None:
    del configured_test_key
    migration = _require_migration(migration_module)
    connection_events, ddl_calls = _patch_fake_operations(
        migration,
        monkeypatch,
        [(encrypt_field(_LEGACY_IDENTIFIER), blind_index(_LEGACY_IDENTIFIER))],
    )

    migration.downgrade()

    assert connection_events == [
        _DOWNGRADE_LOCK,
        "SELECT login_identifier, login_identifier_hmac FROM users",
    ]
    assert ddl_calls == [
        (
            "execute",
            (
                "ALTER INDEX ix_users_login_identifier_hmac "
                "RENAME TO ix_users_email_hmac",
            ),
            {},
        ),
        (
            "alter_column",
            ("users", "login_identifier_hmac"),
            {"new_column_name": "email_hmac"},
        ),
        (
            "alter_column",
            ("users", "login_identifier"),
            {"new_column_name": "email"},
        ),
    ]
    assert all("UPDATE" not in str(call).upper() for call in ddl_calls)


def _literal_database_url(value: str) -> str:
    if make_url(value).database != _MIGRATION_DATABASE:
        raise AssertionError("migration tests require the literal disposable DB")
    return value


def _alembic_environment(database_url: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "test",
            "DATABASE_URL": _literal_database_url(database_url),
            "DATABASE_ENCRYPTION_KEY": _TEST_ENCRYPTION_KEY,
        }
    )
    return environment


def _run_alembic(
    database_url: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=_BACKEND_ROOT,
        env=_alembic_environment(database_url),
        check=False,
        capture_output=True,
        text=True,
    )


async def _reset_database(database_url: str) -> None:
    engine = create_async_engine(_literal_database_url(database_url), echo=False)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


async def _prepare_previous_head(database_url: str) -> AsyncEngine:
    await _reset_database(database_url)
    result = _run_alembic(database_url, "upgrade", "c5d6e7f8a9b0")
    assert result.returncode == 0, result.stdout + result.stderr
    return create_async_engine(database_url, echo=False)


async def _seed_legacy_user(engine: AsyncEngine, identifier: str) -> tuple[bytes, str]:
    ciphertext = encrypt_field(identifier)
    identifier_hmac = blind_index(identifier)
    password_hash = hash_password(_PASSWORD)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                """
                INSERT INTO users (
                    id, email, email_hmac, password_hash, display_name, is_active
                ) VALUES (
                    :id, :email, :email_hmac, :password_hash, 'Synthetic', true
                )
                """
            ),
            {
                "id": _USER_ID,
                "email": ciphertext,
                "email_hmac": identifier_hmac,
                "password_hash": password_hash,
            },
        )
    return ciphertext, identifier_hmac


async def _raw_identifier_values(
    engine: AsyncEngine,
    identifier_column: str,
    hmac_column: str,
) -> tuple[bytes, str]:
    async with engine.connect() as connection:
        row = (
            await connection.execute(
                text(
                    f"SELECT {identifier_column}, {hmac_column} "
                    "FROM users WHERE id = :id"
                ),
                {"id": _USER_ID},
            )
        ).one()
    return bytes(row[0]), row[1]


async def _user_schema_signature(engine: AsyncEngine) -> dict[str, object]:
    def inspect_schema(sync_connection) -> dict[str, object]:
        inspector = inspect(sync_connection)
        columns = inspector.get_columns("users")
        indexes = inspector.get_indexes("users")
        return {
            "column_names": tuple(sorted(column["name"] for column in columns)),
            "identifier_columns": {
                column["name"]: (
                    column["nullable"],
                    column["type"].__class__.__name__,
                )
                for column in columns
                if column["name"]
                in {"login_identifier", "login_identifier_hmac", "email", "email_hmac"}
            },
            "identifier_indexes": tuple(
                sorted(
                    (
                        index["name"],
                        tuple(index["column_names"]),
                        bool(index["unique"]),
                    )
                    for index in indexes
                    if set(index["column_names"])
                    & {"login_identifier_hmac", "email_hmac"}
                )
            ),
        }

    async with engine.connect() as connection:
        return await connection.run_sync(inspect_schema)


async def _invoke_revision(
    engine: AsyncEngine,
    migration: ModuleType,
    function_name: str,
    statements: list[str] | None = None,
) -> list[str]:
    if statements is None:
        statements = []

    def record_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    try:
        async with engine.begin() as connection:

            def invoke(sync_connection) -> None:
                operations = Operations(MigrationContext.configure(sync_connection))
                original_operations = migration.op
                migration.op = operations
                try:
                    getattr(migration, function_name)()
                finally:
                    migration.op = original_operations

            await connection.run_sync(invoke)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
    return statements


async def _run_downgrade_with_validation_barrier(
    database_url: str,
    migration: ModuleType,
    validation_started: threading.Event,
    allow_ddl: threading.Event,
    ddl_completed: threading.Event,
) -> None:
    engine = create_async_engine(database_url, echo=False)
    ddl_statement_count = 0

    def coordinate_downgrade(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        nonlocal ddl_statement_count
        normalized = " ".join(statement.split()).upper()
        if normalized == ("SELECT LOGIN_IDENTIFIER, LOGIN_IDENTIFIER_HMAC FROM USERS"):
            validation_started.set()
            if not allow_ddl.wait(timeout=15):
                raise TimeoutError("test barrier did not release downgrade DDL")
        elif normalized.startswith("ALTER "):
            ddl_statement_count += 1
            if ddl_statement_count == 3:
                ddl_completed.set()

    event.listen(engine.sync_engine, "after_cursor_execute", coordinate_downgrade)
    try:
        await _invoke_revision(engine, migration, "downgrade")
    finally:
        event.remove(
            engine.sync_engine,
            "after_cursor_execute",
            coordinate_downgrade,
        )
        await engine.dispose()


async def _attempt_concurrent_identifier_update(
    database_url: str,
    writer_ready: threading.Event,
    writer_finished: threading.Event,
    ddl_completed: threading.Event,
    writer_pid: list[int],
    outcome: dict[str, object],
) -> None:
    engine = create_async_engine(database_url, echo=False)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET LOCAL statement_timeout = '15s'"))
            writer_pid.append(
                int(await connection.scalar(text("SELECT pg_backend_pid()")))
            )
            writer_ready.set()
            try:
                await connection.execute(
                    text(
                        "UPDATE users "
                        "SET login_identifier = :identifier, "
                        "login_identifier_hmac = :identifier_hmac "
                        "WHERE id = :id"
                    ),
                    {
                        "identifier": encrypt_field(_NON_EMAIL_IDENTIFIER),
                        "identifier_hmac": blind_index(_NON_EMAIL_IDENTIFIER),
                        "id": _USER_ID,
                    },
                )
                outcome["status"] = "succeeded"
            except Exception as exc:  # PostgreSQL may invalidate the canonical SQL.
                outcome["status"] = "failed"
                outcome["error_type"] = type(exc).__name__
            finally:
                outcome["ddl_completed_when_finished"] = ddl_completed.is_set()
    finally:
        writer_ready.set()
        writer_finished.set()
        await engine.dispose()


async def _wait_for_thread_event(
    event_to_wait: threading.Event,
    description: str,
) -> None:
    observed = await asyncio.to_thread(event_to_wait.wait, 15)
    assert observed, f"timed out waiting for {description}"


async def _observe_writer_lock_wait(
    database_url: str,
    writer_pid: int,
    writer_finished: threading.Event,
) -> tuple[list[int], set[tuple[str, bool]]]:
    engine = create_async_engine(database_url, echo=False)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    try:
        async with engine.connect() as connection:
            while loop.time() < deadline:
                if writer_finished.is_set():
                    raise AssertionError(
                        "concurrent identifier update completed before downgrade DDL"
                    )
                row = (
                    await connection.execute(
                        text(
                            "SELECT wait_event_type, pg_blocking_pids(pid) AS blockers "
                            "FROM pg_stat_activity WHERE pid = :writer_pid"
                        ),
                        {"writer_pid": writer_pid},
                    )
                ).one_or_none()
                if row is not None and row.wait_event_type == "Lock" and row.blockers:
                    blocker_locks = {
                        (lock_row.mode, bool(lock_row.granted))
                        for lock_row in (
                            await connection.execute(
                                text(
                                    "SELECT mode, granted FROM pg_locks "
                                    "WHERE pid = ANY(CAST(:blockers AS integer[])) "
                                    "AND relation = to_regclass('public.users')"
                                ),
                                {"blockers": list(row.blockers)},
                            )
                        )
                    }
                    return list(row.blockers), blocker_locks
                await asyncio.sleep(0.01)
    finally:
        await engine.dispose()
    raise AssertionError("concurrent identifier update never entered a table-lock wait")


@asynccontextmanager
async def _auth_client(database_url: str) -> AsyncIterator[AsyncClient]:
    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def test_upgrade_preserves_ciphertext_and_hmac_bytes_and_both_login_payloads(
    migration_module: ModuleType | None,
    migration_database_url: str,
) -> None:
    _require_migration(migration_module)
    engine = await _prepare_previous_head(migration_database_url)
    try:
        seeded_values = await _seed_legacy_user(engine, _LEGACY_IDENTIFIER)
        legacy_values = await _raw_identifier_values(engine, "email", "email_hmac")
        assert legacy_values == seeded_values
    finally:
        await engine.dispose()

    result = _run_alembic(migration_database_url, "upgrade", "d6e7f8a9b0c1")
    assert result.returncode == 0, result.stdout + result.stderr
    engine = create_async_engine(migration_database_url, echo=False)
    try:
        assert (
            await _raw_identifier_values(
                engine,
                "login_identifier",
                "login_identifier_hmac",
            )
            == seeded_values
        )
    finally:
        await engine.dispose()

    async with _auth_client(migration_database_url) as client:
        canonical = await client.post(
            "/api/v1/auth/login",
            json={"login_identifier": _LEGACY_IDENTIFIER, "password": _PASSWORD},
        )
        legacy = await client.post(
            "/api/v1/auth/login",
            json={"email": _LEGACY_IDENTIFIER, "password": _PASSWORD},
        )

    assert canonical.status_code == 200
    assert legacy.status_code == 200
    assert canonical.json()["access_token"]
    assert legacy.json()["access_token"]


async def test_email_only_downgrade_preserves_ciphertext_and_hmac_bytes(
    migration_module: ModuleType | None,
    migration_database_url: str,
) -> None:
    migration = _require_migration(migration_module)
    engine = await _prepare_previous_head(migration_database_url)
    try:
        seeded_values = await _seed_legacy_user(engine, _LEGACY_IDENTIFIER)
    finally:
        await engine.dispose()
    result = _run_alembic(migration_database_url, "upgrade", "d6e7f8a9b0c1")
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_async_engine(migration_database_url, echo=False)
    try:
        assert (
            await _raw_identifier_values(
                engine,
                "login_identifier",
                "login_identifier_hmac",
            )
            == seeded_values
        )
        statements = await _invoke_revision(engine, migration, "downgrade")
        assert (
            await _raw_identifier_values(engine, "email", "email_hmac") == seeded_values
        )
        signature = await _user_schema_signature(engine)
    finally:
        await engine.dispose()

    assert any(
        " ".join(statement.split())
        == "SELECT login_identifier, login_identifier_hmac FROM users"
        for statement in statements
    )
    assert all(
        not statement.lstrip().upper().startswith("UPDATE") for statement in statements
    )
    assert "login_identifier" not in signature["column_names"]
    assert "login_identifier_hmac" not in signature["column_names"]
    assert signature["identifier_indexes"] == (
        ("ix_users_email_hmac", ("email_hmac",), True),
    )


async def test_concurrent_identifier_update_waits_for_downgrade_ddl_after_validation(
    migration_module: ModuleType | None,
    migration_database_url: str,
) -> None:
    migration = _require_migration(migration_module)
    engine = await _prepare_previous_head(migration_database_url)
    try:
        await _seed_legacy_user(engine, _LEGACY_IDENTIFIER)
    finally:
        await engine.dispose()
    result = _run_alembic(migration_database_url, "upgrade", "d6e7f8a9b0c1")
    assert result.returncode == 0, result.stdout + result.stderr

    validation_started = threading.Event()
    allow_ddl = threading.Event()
    ddl_completed = threading.Event()
    writer_ready = threading.Event()
    writer_finished = threading.Event()
    writer_pid: list[int] = []
    writer_outcome: dict[str, object] = {}
    migration_task = asyncio.create_task(
        asyncio.to_thread(
            lambda: asyncio.run(
                _run_downgrade_with_validation_barrier(
                    migration_database_url,
                    migration,
                    validation_started,
                    allow_ddl,
                    ddl_completed,
                )
            )
        )
    )
    writer_task: asyncio.Task[None] | None = None
    task_results: list[object] = []
    try:
        await _wait_for_thread_event(validation_started, "downgrade validation")
        writer_task = asyncio.create_task(
            _attempt_concurrent_identifier_update(
                migration_database_url,
                writer_ready,
                writer_finished,
                ddl_completed,
                writer_pid,
                writer_outcome,
            )
        )
        await _wait_for_thread_event(writer_ready, "concurrent writer backend")
        assert writer_pid, "concurrent writer failed before publishing its backend PID"

        blockers, blocker_locks = await _observe_writer_lock_wait(
            migration_database_url,
            writer_pid[0],
            writer_finished,
        )

        assert blockers
        assert ("ShareRowExclusiveLock", True) in blocker_locks
        assert not writer_finished.is_set()
    finally:
        allow_ddl.set()
        tasks = [migration_task]
        if writer_task is not None:
            tasks.append(writer_task)
        task_results = list(
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=20,
            )
        )

    assert task_results == [None, None]
    assert ddl_completed.is_set()
    assert writer_finished.is_set()
    assert writer_outcome["status"] in {"succeeded", "failed"}
    assert writer_outcome["ddl_completed_when_finished"] is True
    assert writer_outcome.get("error_type") != "QueryCanceledError"


@pytest.mark.parametrize(
    ("identifier", "wrong_key"),
    (
        pytest.param(_NON_EMAIL_IDENTIFIER, False, id="non-email"),
        pytest.param(_LEGACY_IDENTIFIER, True, id="wrong-key"),
        pytest.param(
            _DECOMPOSED_LEGACY_IDENTIFIER,
            False,
            id="email-validator-normalization-mismatch",
        ),
    ),
)
async def test_refused_downgrade_preserves_canonical_schema_and_bytes_before_any_write(
    migration_module: ModuleType | None,
    migration_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    identifier: str,
    wrong_key: bool,
) -> None:
    migration = _require_migration(migration_module)
    engine = await _prepare_previous_head(migration_database_url)
    try:
        seeded_values = await _seed_legacy_user(engine, identifier)
    finally:
        await engine.dispose()
    sensitive_values = {identifier, seeded_values[1]}
    if identifier == _DECOMPOSED_LEGACY_IDENTIFIER:
        sensitive_values.update(
            {
                _NORMALIZED_LEGACY_IDENTIFIER,
                blind_index(_NORMALIZED_LEGACY_IDENTIFIER),
            }
        )
    result = _run_alembic(migration_database_url, "upgrade", "d6e7f8a9b0c1")
    assert result.returncode == 0, result.stdout + result.stderr

    if wrong_key:
        monkeypatch.setattr(settings, "database_encryption_key", _WRONG_ENCRYPTION_KEY)

    engine = create_async_engine(migration_database_url, echo=False)
    statements: list[str] = []
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await _invoke_revision(engine, migration, "downgrade", statements)
        assert str(exc_info.value) == DOWNGRADE_BLOCKED
        assert (
            await _raw_identifier_values(
                engine,
                "login_identifier",
                "login_identifier_hmac",
            )
            == seeded_values
        )
        signature = await _user_schema_signature(engine)
    finally:
        await engine.dispose()

    assert any(
        " ".join(statement.split())
        == "SELECT login_identifier, login_identifier_hmac FROM users"
        for statement in statements
    )
    assert all(
        not statement.lstrip().upper().startswith("UPDATE") for statement in statements
    )
    assert all("ALTER " not in statement.upper() for statement in statements)
    assert "login_identifier" in signature["column_names"]
    assert "login_identifier_hmac" in signature["column_names"]
    assert "email" not in signature["column_names"]
    assert "email_hmac" not in signature["column_names"]
    assert signature["identifier_indexes"] == (
        ("ix_users_login_identifier_hmac", ("login_identifier_hmac",), True),
    )
    for sensitive_value in sensitive_values:
        assert sensitive_value not in str(exc_info.value)
        assert sensitive_value not in caplog.text
    assert str(_USER_ID) not in str(exc_info.value)
    assert str(_USER_ID) not in caplog.text


async def test_create_all_and_alembic_head_have_equivalent_user_identifier_schema(
    migration_module: ModuleType | None,
    migration_database_url: str,
) -> None:
    _require_migration(migration_module)
    await _reset_database(migration_database_url)
    create_all_engine = create_async_engine(migration_database_url, echo=False)
    try:
        async with create_all_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        create_all_signature = await _user_schema_signature(create_all_engine)
    finally:
        await create_all_engine.dispose()

    await _reset_database(migration_database_url)
    result = _run_alembic(migration_database_url, "upgrade", "d6e7f8a9b0c1")
    assert result.returncode == 0, result.stdout + result.stderr
    migration_engine = create_async_engine(migration_database_url, echo=False)
    try:
        migration_signature = await _user_schema_signature(migration_engine)
    finally:
        await migration_engine.dispose()

    assert migration_signature == create_all_signature
    assert migration_signature["identifier_columns"] == {
        "login_identifier": (False, "BYTEA"),
        "login_identifier_hmac": (False, "VARCHAR"),
    }
    assert migration_signature["identifier_indexes"] == (
        ("ix_users_login_identifier_hmac", ("login_identifier_hmac",), True),
    )
