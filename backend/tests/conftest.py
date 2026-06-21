from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.main import app as fastapi_app
from app.database import get_db
from app.models.base import Base
from app.models.patient import Patient
from app.models.record import HealthRecord

# Import all models so metadata is populated
import app.models  # noqa: F401

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Private real-data fixtures (gitignored; never committed to any repo).
# ---------------------------------------------------------------------------
# Load machine-local test env so REAL_MEDICAL_FIXTURES_DIR is available to the
# fidelity/extraction tests. ``.env.test.local`` sits at the repo root next to
# ``.env`` and is gitignored. Loaded here (not in app.config) because it is
# test-only plumbing, and conftest is imported before any test module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env.test.local", override=False)


def private_fixture_root() -> Path | None:
    """Return the real-medical-fixtures root, or ``None``.

    Reads ``REAL_MEDICAL_FIXTURES_DIR`` (expanding ``~``). Returns ``None`` when
    the var is unset or the directory is missing, so callers can compute a clean
    module-level skip. There is intentionally NO fallback to in-repo paths —
    real PHI never lives in the repository. Original exports live under
    ``<root>/raw/``.
    """
    root = os.environ.get("REAL_MEDICAL_FIXTURES_DIR")
    if not root:
        return None
    path = Path(root).expanduser()
    return path if path.exists() else None


# Slow/fidelity tests drive real Gemini calls over real data and legitimately
# take minutes (e.g. the section-parsing wall-clock test runs the full pipeline
# and asserts <450s itself). The fast-suite ``timeout = 120`` would falsely kill
# them, so they get a generous BUT BOUNDED override — enough headroom for real
# API latency, while still backstopping a genuine hang (a frozen await never
# returns; the internal perf assertions can't fire until it does).
_SLOW_TEST_TIMEOUT_S = 900


def pytest_collection_modifyitems(config, items):
    """Give slow/fidelity tests a generous bounded timeout instead of the 120s
    fast-suite default. Never disable the timeout entirely (timeout=0) — that
    reintroduces the infinite-hang failure mode this whole change exists to
    prevent."""
    for item in items:
        if item.get_closest_marker("slow") or item.get_closest_marker("fidelity"):
            item.add_marker(pytest.mark.timeout(_SLOW_TEST_TIMEOUT_S))

# Use a dedicated test database to avoid destroying production data.
# Derive from the production URL by appending "_test" to the database name.
_prod_url = settings.database_url
if "_test" not in _prod_url:
    TEST_DB_URL = _prod_url.rsplit("/", 1)[0] + "/" + _prod_url.rsplit("/", 1)[1] + "_test"
else:
    TEST_DB_URL = _prod_url


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session with table creation and cleanup."""
    engine = create_async_engine(TEST_DB_URL, echo=False)

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clean up any leftover data from prior runs (CASCADE handles FK deps)
    async with engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE revoked_tokens, provenance, dedup_candidates, record_cross_references, health_records, "
            "ai_summary_prompts, uploaded_files, patients, audit_log, users CASCADE"
        ))

    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session

    # Clean up all data after each test (CASCADE handles FK deps)
    async with engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE revoked_tokens, provenance, dedup_candidates, record_cross_references, health_records, "
            "ai_summary_prompts, uploaded_files, patients, audit_log, users CASCADE"
        ))

    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP test client with DB dependency override."""

    async def override_get_db():
        yield db_session

    fastapi_app.dependency_overrides[get_db] = override_get_db

    # The request-level audit middleware (W16) opens its OWN session via
    # app.middleware.audit.async_session_factory (deliberately not get_db, so a
    # request rollback can't drop the audit row). Point that factory at the test
    # DB for the duration of the test, otherwise it writes api.access rows to the
    # real database during the suite.
    import app.middleware.audit as _audit

    _audit_engine = create_async_engine(TEST_DB_URL, echo=False)
    _orig_audit_factory = _audit.async_session_factory
    _audit.async_session_factory = async_sessionmaker(
        _audit_engine, class_=AsyncSession, expire_on_commit=False
    )

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()
    _audit.async_session_factory = _orig_audit_factory
    await _audit_engine.dispose()


@pytest.fixture
def fhir_bundle():
    """Load user-provided FHIR JSON, fall back to synthetic."""
    user_file = FIXTURES_DIR / "user_provided_fhir.json"
    synthetic_file = FIXTURES_DIR / "sample_fhir_bundle.json"
    path = user_file if user_file.exists() else synthetic_file
    return json.loads(path.read_text())


@pytest.fixture
def epic_export_dir():
    """Load user-provided Epic export dir, fall back to synthetic."""
    user_dir = FIXTURES_DIR / "epic_export"
    synthetic_dir = FIXTURES_DIR / "sample_epic_tsv"
    return user_dir if user_dir.exists() else synthetic_dir


# ---------------------------------------------------------------------------
# Rate limiter cleanup (singleton state persists between tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiter state before each test."""
    from app.middleware.rate_limit import login_limiter, register_limiter
    login_limiter._requests.clear()
    register_limiter._requests.clear()
    yield
    login_limiter._requests.clear()
    register_limiter._requests.clear()


@pytest.fixture(autouse=True)
def reset_extraction_worker():
    """Stop the DB-polling extraction worker from leaking across tests.

    ``upload._worker_task`` is a module global. A worker started in one test is
    bound to that test's event loop; once pytest-asyncio closes that loop the
    task is dead but still referenced, so the *next* test's
    ``start_extraction_worker`` sees a "not done" task and refuses to spawn a
    fresh one. Extraction then never runs and any test that waits on it hangs
    forever (the whole-suite hang). Nulling the global before each test — and
    clearing the per-loop semaphore caches — guarantees each test gets a worker
    bound to its own loop.
    """
    import app.api.upload as upload_module

    def _reset() -> None:
        task = getattr(upload_module, "_worker_task", None)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass
        upload_module._worker_task = None
        upload_module._extraction_semaphores.clear()
        upload_module._gemini_semaphores.clear()

    _reset()
    yield
    _reset()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def auth_headers(client: AsyncClient, email: str = "test@example.com") -> tuple[dict, str]:
    """Register a user, log in, return (headers_dict, user_id_str)."""
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!", "display_name": "Test"},
    )
    user_id = reg.json()["id"]
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "SecurePass123!"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, user_id


async def create_test_patient(db_session: AsyncSession, user_id: str | UUID) -> Patient:
    """Insert a Patient row and return it."""
    uid = UUID(user_id) if isinstance(user_id, str) else user_id
    patient = Patient(id=uuid4(), user_id=uid, fhir_id="test-patient-001", gender="male")
    db_session.add(patient)
    await db_session.commit()
    await db_session.refresh(patient)
    return patient


SAMPLE_RECORDS = [
    {
        "record_type": "condition",
        "fhir_resource_type": "Condition",
        "fhir_resource": {
            "resourceType": "Condition",
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes"}]
            },
            "clinicalStatus": {"coding": [{"code": "active"}]},
        },
        "source_format": "fhir_r4",
        "status": "active",
        "category": ["encounter-diagnosis"],
        "code_system": "http://snomed.info/sct",
        "code_value": "44054006",
        "code_display": "Type 2 diabetes",
        "display_text": "Type 2 diabetes mellitus",
    },
    {
        "record_type": "observation",
        "fhir_resource_type": "Observation",
        "fhir_resource": {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{"code": "laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
            "valueQuantity": {"value": 6.8, "unit": "%"},
            "referenceRange": [{"low": {"value": 4.0}, "high": {"value": 5.6}}],
            "interpretation": [{"coding": [{"code": "H"}]}],
        },
        "source_format": "fhir_r4",
        "status": "final",
        "category": ["laboratory"],
        "code_system": "http://loinc.org",
        "code_value": "4548-4",
        "code_display": "Hemoglobin A1c",
        "display_text": "Hemoglobin A1c: 6.8%",
    },
    {
        "record_type": "medication",
        "fhir_resource_type": "MedicationRequest",
        "fhir_resource": {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Metformin 500mg"},
        },
        "source_format": "fhir_r4",
        "status": "active",
        "category": ["medication"],
        "code_system": None,
        "code_value": None,
        "code_display": "Metformin 500mg",
        "display_text": "Metformin 500mg — Take twice daily",
    },
    {
        "record_type": "encounter",
        "fhir_resource_type": "Encounter",
        "fhir_resource": {
            "resourceType": "Encounter",
            "status": "finished",
            "class": {"code": "AMB"},
            "type": [{"text": "Office visit"}],
        },
        "source_format": "fhir_r4",
        "status": "finished",
        "category": ["encounter"],
        "code_system": None,
        "code_value": None,
        "code_display": "Office visit",
        "display_text": "Office visit",
    },
    {
        "record_type": "immunization",
        "fhir_resource_type": "Immunization",
        "fhir_resource": {
            "resourceType": "Immunization",
            "status": "completed",
            "vaccineCode": {"coding": [{"display": "Influenza vaccine"}]},
        },
        "source_format": "fhir_r4",
        "status": "completed",
        "category": ["immunization"],
        "code_system": None,
        "code_value": None,
        "code_display": "Influenza vaccine",
        "display_text": "Influenza vaccine",
    },
]


async def seed_test_records(
    db_session: AsyncSession,
    user_id: str | UUID,
    patient_id: str | UUID,
    count: int = 5,
) -> list[HealthRecord]:
    """Insert varied HealthRecord rows. Returns the created records."""
    uid = UUID(user_id) if isinstance(user_id, str) else user_id
    pid = UUID(patient_id) if isinstance(patient_id, str) else patient_id

    records = []
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        sample = SAMPLE_RECORDS[i % len(SAMPLE_RECORDS)]
        rec = HealthRecord(
            id=uuid4(),
            patient_id=pid,
            user_id=uid,
            effective_date=base_date + timedelta(days=i),
            **sample,
        )
        db_session.add(rec)
        records.append(rec)

    await db_session.commit()
    for r in records:
        await db_session.refresh(r)
    return records
