"""Performance-regression guards for the record/timeline read paths (D4).

The clinical FHIR payload (`health_records.fhir_resource`) is encrypted at rest
(AES-256-GCM via :class:`EncryptedJSON`), so **every loaded row triggers a
decrypt**. Two properties must therefore hold for the list endpoints to scale
with the *page* rather than the *table*:

1. ``GET /records`` decrypts only the page's rows — LIMIT/OFFSET is pushed into
   SQL and the total-count query decrypts nothing (no ``HealthRecord`` row
   crosses into Python for the count).
2. The default newest-first ordering is served by a partial index whose
   predicate **matches the query** (``is_duplicate IS FALSE``). Postgres will
   not match an ``IS FALSE`` query against a ``= false`` partial-index
   predicate, so a mismatched index is silently unused and the query
   full-scans + sorts the whole table (the original D4 slowdown).
"""

from __future__ import annotations

import contextlib
from uuid import UUID

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.models.encrypted_types as encrypted_types
from app.api.records import list_records
from app.models.record import HealthRecord
from tests.conftest import (
    TEST_DB_URL,
    auth_headers,
    create_test_patient,
    seed_test_records,
)


class _FakeClient:
    host = "127.0.0.1"


class _FakeRequest:
    client = _FakeClient()


@contextlib.asynccontextmanager
async def _fresh_session():
    """A session on a brand-new engine so its ORM identity map is empty.

    Reusing the seeding session would let already-loaded rows skip the result
    processor, hiding how many rows the page query actually decrypts. A fresh
    engine guarantees each returned row is decrypted afresh — so the spy count
    equals the number of rows the SQL really fetched.
    """
    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory() as session:
            yield session
    finally:
        await engine.dispose()


class _DecryptSpy:
    """Count calls to ``decrypt_field`` (patched where it is actually invoked)."""

    def __init__(self, monkeypatch):
        self.n = 0
        self._orig = encrypted_types.decrypt_field

        def _wrapped(data):
            self.n += 1
            return self._orig(data)

        monkeypatch.setattr(encrypted_types, "decrypt_field", _wrapped)

    def reset(self):
        self.n = 0


async def _seed(client, db_session, count: int):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=count)
    return UUID(uid)


@pytest.mark.asyncio
async def test_list_page_decrypts_only_page_rows(client, db_session, monkeypatch):
    """A page_size=20 request decrypts exactly 20 fhir_resource values, not all 300."""
    uid = await _seed(client, db_session, count=120)
    spy = _DecryptSpy(monkeypatch)

    async with _fresh_session() as sess:
        spy.reset()
        resp = await list_records(
            request=_FakeRequest(), page=1, page_size=20, record_type=None,
            category=None, search=None, status=None, sort=None, order="desc",
            user_id=uid, db=sess,
        )

    assert resp.total == 120
    assert len(resp.items) == 20
    # Page-bounded: only the 20 returned rows are decrypted, not the whole table.
    assert spy.n == 20, f"expected 20 decrypts for a 20-row page, got {spy.n}"


@pytest.mark.asyncio
async def test_single_row_page_and_count_decrypt_minimally(client, db_session, monkeypatch):
    """page_size=1 decrypts exactly 1 row — the count contributes 0 decrypts."""
    uid = await _seed(client, db_session, count=120)
    spy = _DecryptSpy(monkeypatch)

    async with _fresh_session() as sess:
        spy.reset()
        resp = await list_records(
            request=_FakeRequest(), page=1, page_size=1, record_type=None,
            category=None, search=None, status=None, sort=None, order="desc",
            user_id=uid, db=sess,
        )

    assert resp.total == 120
    assert len(resp.items) == 1
    # 1 decrypt total proves the count (over 250 rows) decrypted nothing.
    assert spy.n == 1, f"expected 1 decrypt (count must add none), got {spy.n}"


@pytest.mark.asyncio
async def test_deep_page_still_page_bounded(client, db_session, monkeypatch):
    """A deep page (offset 180) still decrypts only its own rows."""
    uid = await _seed(client, db_session, count=120)
    spy = _DecryptSpy(monkeypatch)

    async with _fresh_session() as sess:
        spy.reset()
        resp = await list_records(
            request=_FakeRequest(), page=5, page_size=20, record_type=None,
            category=None, search=None, status=None, sort=None, order="desc",
            user_id=uid, db=sess,
        )

    assert len(resp.items) == 20
    assert spy.n == 20, f"deep page should decrypt 20 rows, got {spy.n}"


@pytest.mark.asyncio
async def test_count_query_runs_no_decrypts(client, db_session, monkeypatch):
    """The standalone count statement the endpoint issues decrypts nothing."""
    uid = await _seed(client, db_session, count=90)
    spy = _DecryptSpy(monkeypatch)

    conditions = [
        HealthRecord.user_id == uid,
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    ]
    async with _fresh_session() as sess:
        spy.reset()
        total = (await sess.execute(select(func.count()).where(*conditions))).scalar()

    assert total == 90
    assert spy.n == 0, f"count must not decrypt any row, got {spy.n}"


def test_fetch_statement_pushes_limit_to_sql():
    """The page fetch carries LIMIT/OFFSET in SQL (work bounded by page size)."""
    conditions = [
        HealthRecord.user_id == UUID("00000000-0000-0000-0000-000000000001"),
        HealthRecord.deleted_at.is_(None),
        HealthRecord.is_duplicate.is_(False),
    ]
    stmt = (
        select(HealthRecord)
        .where(*conditions)
        .order_by(HealthRecord.effective_date.desc().nullslast(), HealthRecord.id.asc())
        .offset(40)
        .limit(20)
    )
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
    assert "limit 20" in sql
    assert "offset 40" in sql


# --- the structural fix: an IS FALSE ordering index serves the page ----------
#
# These tests CREATE/DROP an index on health_records, which needs SHARE /
# ACCESS EXCLUSIVE locks. The shared ``db_session``/audit-middleware connections
# hold conflicting locks and would deadlock the DDL, so these run fully isolated
# on their own engine (every step committed in sequence, no concurrent holders).

_ORDER_BY = (HealthRecord.effective_date.desc().nullslast(), HealthRecord.id.asc())


@contextlib.asynccontextmanager
async def _isolated_records(count: int = 60):
    """Own-engine fixture: a user + patient + ``count`` active records, with a
    sessionmaker and the user id. Cleans up its own rows on exit so it doesn't
    rely on (or contend with) the shared ``db_session`` truncation."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from app.models.base import Base
    from app.models.patient import Patient
    from app.models.user import User

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    uid, pid = uuid4(), uuid4()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with factory() as db:
            db.add(User(id=uid, email=f"perf-{uid}@x.com", email_hmac=f"bi-{uid}",
                        password_hash="x"))
            db.add(Patient(id=pid, user_id=uid, fhir_id=f"pp-{uid}", gender="male"))
            await db.commit()
            base = datetime(2024, 1, 1, tzinfo=timezone.utc)
            for i in range(count):
                db.add(HealthRecord(
                    id=uuid4(), patient_id=pid, user_id=uid, record_type="observation",
                    fhir_resource_type="Observation", fhir_resource={"n": i},
                    source_format="fhir_r4", effective_date=base + timedelta(days=i),
                    status="final", code_value="x", code_display="A1c",
                    display_text=f"r{i}"))
            await db.commit()
        yield factory, uid
    finally:
        async with factory() as db:
            await db.execute(text("DELETE FROM audit_log WHERE user_id = :u"), {"u": uid})
            await db.execute(HealthRecord.__table__.delete().where(HealthRecord.user_id == uid))
            await db.execute(Patient.__table__.delete().where(Patient.id == pid))
            await db.execute(User.__table__.delete().where(User.id == uid))
            await db.commit()
        await engine.dispose()


async def _explain_ordered_fetch(factory, uid: UUID) -> str:
    """EXPLAIN the default ordered page with seqscan disabled, so the planner
    must use a usable index if one matches — revealing whether the Sort drops."""
    stmt = (
        select(HealthRecord)
        .where(
            HealthRecord.user_id == uid,
            HealthRecord.deleted_at.is_(None),
            HealthRecord.is_duplicate.is_(False),
        )
        .order_by(*_ORDER_BY)
        .limit(20)
    )
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    async with factory() as sess:
        await sess.execute(text("SET enable_seqscan = off"))
        rows = (await sess.execute(text("EXPLAIN " + compiled))).all()
    return "\n".join(r[0] for r in rows)


@pytest.mark.asyncio
async def test_isfalse_ordering_index_eliminates_the_sort():
    """With the migration's IS FALSE index present, the ordered page is an index
    range scan with NO Sort node — the work is bounded by the page, not the table."""
    idx = "idx_perf_eff_isfalse"
    async with _isolated_records() as (factory, uid):
        async with factory() as sess:
            await sess.execute(text(
                f"CREATE INDEX IF NOT EXISTS {idx} ON health_records "
                "(user_id, effective_date DESC NULLS LAST, id) "
                "WHERE deleted_at IS NULL AND is_duplicate IS FALSE"
            ))
            await sess.commit()
        try:
            plan = await _explain_ordered_fetch(factory, uid)
            assert idx in plan, f"IS FALSE index should serve the query:\n{plan}"
            assert "Sort" not in plan, f"ordering index must remove the Sort:\n{plan}"
        finally:
            async with factory() as sess:
                await sess.execute(text(f"DROP INDEX IF EXISTS {idx}"))
                await sess.commit()


@pytest.mark.asyncio
async def test_eqfalse_predicate_index_is_not_matched():
    """Regression guard for the original D4 cause: an otherwise-identical index
    with a ``= false`` predicate is NOT matched by the ``IS FALSE`` query, so the
    planner cannot use it and falls back to a Sort. This is why shipping the
    index with ``= false`` left every read full-scanning."""
    idx = "idx_perf_eff_eqfalse"
    async with _isolated_records() as (factory, uid):
        async with factory() as sess:
            await sess.execute(text(
                f"CREATE INDEX IF NOT EXISTS {idx} ON health_records "
                "(user_id, effective_date DESC NULLS LAST, id) "
                "WHERE deleted_at IS NULL AND is_duplicate = false"
            ))
            await sess.commit()
        try:
            plan = await _explain_ordered_fetch(factory, uid)
            # The mismatched-predicate index must NOT be chosen for the IS FALSE query.
            assert idx not in plan, (
                "a = false predicate index must NOT match an IS FALSE query "
                f"(predicate mismatch is the D4 bug):\n{plan}"
            )
        finally:
            async with factory() as sess:
                await sess.execute(text(f"DROP INDEX IF EXISTS {idx}"))
                await sess.commit()
