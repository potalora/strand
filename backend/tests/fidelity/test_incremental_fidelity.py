"""Real-data fidelity tests for idempotent incremental XDM ingestion.

Headline validation: two REAL IHE XDM HealthSummary extracts of the SAME
patient ("Pedro1"), taken ~7 weeks apart, prove that re-ingesting overlapping
clinical data is recognized by the identity gate (idempotent_insert_records)
rather than blindly re-inserting.

These tests are CI-safe: they skip when the gitignored real extracts are absent.

Measured behavior (2026-05-29, this machine) — used to calibrate the bounds below:

  Same extract (MAY ingested twice):
    1st ingest:  inserted=124, skipped(intra-upload dedup)=354  -> 124 live rows
    2nd ingest:  inserted=2,  updated=0,  unchanged=121         -> 126 live rows
    Re-insert fraction on a perfect re-ingest: 2/124 ~= 1.6%

  Cumulative (APR then MAY):
    APR ingest:  inserted=123                                   -> 123 live rows
    MAY ingest:  inserted=2,  updated=15, unchanged=106         -> 125 live rows

  Re-insert by resource type (no-stable-identity types fall through the gate):
    AllergyIntolerance: +1   DocumentReference: +2
    Condition / Observation / Encounter / Immunization /
    MedicationStatement / DiagnosticReport: +0  (all recognized)

KNOWN LIMITATION: the CDA->FHIR renderer emits resource.identifier (-> stable
id) for most clinical resource types but NOT for AllergyIntolerance and
DocumentReference here, so those few records lack a stable id and re-insert on
every ingest. The downstream content-dedup pipeline (not exercised by
_ingest_xdm alone) would catch them later. Hence we MEASURE and bound, never
assert a perfect zero-insert no-op.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.ingestion.coordinator import _ingest_xdm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from tests.conftest import auth_headers, create_test_patient, private_fixture_root

# Real IHE-XDM extracts resolve via REAL_MEDICAL_FIXTURES_DIR (gitignored,
# off-repo); originals live under <root>/raw/. No in-repo fallback.
# NOTE: the previous MAY_DIR pointed at REPO_ROOT (missing the test_data
# segment), so it silently skipped on every run — now both extracts resolve
# consistently under raw/.
_FIXROOT = private_fixture_root()
_RAW = (_FIXROOT / "raw") if _FIXROOT else None
MAY_DIR = (_RAW / "HealthSummary_May_29_2026" / "IHE_XDM" / "Pedro1") if _RAW else None
APR_DIR = (_RAW / "HealthSummary_Apr_05_2026" / "IHE_XDM" / "Pedro1") if _RAW else None

pytestmark = pytest.mark.fidelity

skip_if_no_may = pytest.mark.skipif(
    not (MAY_DIR and (MAY_DIR / "METADATA.XML").exists()),
    reason="REAL_MEDICAL_FIXTURES_DIR unset or MAY XDM extract (raw/HealthSummary_May_29_2026) missing",
)
skip_if_no_apr = pytest.mark.skipif(
    not (APR_DIR and (APR_DIR / "METADATA.XML").exists()),
    reason="REAL_MEDICAL_FIXTURES_DIR unset or APRIL XDM extract (raw/HealthSummary_Apr_05_2026) missing",
)


async def _make_upload(db: AsyncSession, user_id: UUID) -> UUID:
    """Create a real uploaded_files row (health_records.source_file_id FKs to it)."""
    uf = UploadedFile(
        user_id=user_id,
        filename="HealthSummary.zip",
        mime_type="application/zip",
        file_hash=uuid4().hex + uuid4().hex,
        storage_path=f"/tmp/{uuid4().hex}.zip",
        file_category="structured",
    )
    db.add(uf)
    await db.flush()
    return uf.id


async def _ingest_xdm_package(
    db: AsyncSession, user_id: UUID, patient_id: UUID, xdm_dir: Path
) -> dict:
    """Ingest one IHE XDM package for a patient and return the coordinator stats."""
    upload_id = await _make_upload(db, user_id)
    return await _ingest_xdm(
        db, user_id, patient_id, upload_id, xdm_dir, xdm_dir / "METADATA.XML"
    )


async def _count_live(db: AsyncSession, patient_id: UUID) -> int:
    """Count a patient's live (non-soft-deleted) HealthRecords."""
    return (
        await db.execute(
            select(func.count())
            .select_from(HealthRecord)
            .where(
                HealthRecord.patient_id == patient_id,
                HealthRecord.deleted_at.is_(None),
            )
        )
    ).scalar_one()


@skip_if_no_may
@pytest.mark.asyncio
async def test_same_extract_reingest_recognizes_overlap(client, db_session):
    """Re-ingesting the identical MAY extract recognizes the bulk as unchanged.

    Measured: 1st ingest -> 124 rows; 2nd ingest -> inserted=2, unchanged=121.
    Only the no-identity minority (AllergyIntolerance + DocumentReference, ~1.6%)
    re-inserts. We assert the second ingest recognizes most records as unchanged
    and re-inserts well under 20% of the corpus (measured ~1.6%), so a regression
    in identity extraction would spike inserts and fail this test.
    """
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    await _ingest_xdm_package(db_session, uid, patient.id, MAY_DIR)
    n1 = await _count_live(db_session, patient.id)
    assert n1 > 0, "First MAY ingest produced no records"

    s2 = await _ingest_xdm_package(db_session, uid, patient.id, MAY_DIR)

    # Bulk recognized as already present, not re-inserted.
    assert s2["records_unchanged"] > 0
    # Most records recognized; only the no-identity minority re-inserts.
    # Measured re-insert fraction ~1.6% (AllergyIntolerance + DocumentReference).
    assert s2["records_inserted"] < n1 * 0.2, (
        f"Re-ingest inserted {s2['records_inserted']} of {n1} "
        f"({s2['records_inserted'] / n1:.1%}); expected ~1.6%. "
        "Identity-gate regression?"
    )
    # The recognized records dominate the corpus.
    assert s2["records_unchanged"] > n1 * 0.8


@skip_if_no_apr
@skip_if_no_may
@pytest.mark.asyncio
async def test_cumulative_apr_then_may_recognizes_overlap(client, db_session):
    """Ingesting APRIL then the (overlapping) MAY extract converges, not doubles.

    Measured: APR -> 123 rows; MAY -> inserted=2, updated=15, unchanged=106,
    final 125 rows. The MAY ingest recognizes the April corpus (unchanged +
    updated) instead of re-inserting it, so the corpus stays ~flat (does not
    double) and far fewer than the whole April corpus is inserted as new.
    """
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    await _ingest_xdm_package(db_session, uid, patient.id, APR_DIR)
    n_apr = await _count_live(db_session, patient.id)
    assert n_apr > 0, "April ingest produced no records"

    s_may = await _ingest_xdm_package(db_session, uid, patient.id, MAY_DIR)
    n_final = await _count_live(db_session, patient.id)

    # April records are recognized when the overlapping MAY extract lands.
    assert s_may["records_unchanged"] > 0
    # Corpus converged — it did not double.
    assert n_final < 2 * n_apr
    # Far fewer new inserts than the whole April corpus (measured: 2 vs 123).
    assert s_may["records_inserted"] < n_apr
    # Tight, honest bound from measured ~1.6% re-insert fraction.
    assert s_may["records_inserted"] < n_apr * 0.2, (
        f"MAY ingest inserted {s_may['records_inserted']} new vs April corpus "
        f"{n_apr}; expected ~2. Identity-gate regression?"
    )
