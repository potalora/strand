from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.models.provenance import Provenance
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile


async def _make_record(db, patient, source_file_id, *, ai_extracted: bool):
    rec = HealthRecord(
        id=uuid.uuid4(), patient_id=patient.id, user_id=patient.user_id,
        record_type="condition", fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition", "code": {"text": "x"}},
        source_format="ai_extracted" if ai_extracted else "fhir_r4",
        source_file_id=source_file_id, display_text="x", ai_extracted=ai_extracted,
    )
    db.add(rec)
    return rec


@pytest.mark.asyncio
async def test_soft_delete_prior_extracted_replaces_only_ai_records(db_session, client):
    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    from tests.conftest import auth_headers, create_test_patient
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    # HealthRecord.source_file_id has a FK to uploaded_files; create a real row first
    upload = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="doc.pdf",
        file_hash="FAKEABC", storage_path="/tmp/x", ingestion_status="completed",
        file_category="unstructured", mime_type="application/pdf",
    )
    db_session.add(upload)
    await db_session.flush()
    sfid = upload.id
    await _make_record(db_session, patient, sfid, ai_extracted=True)
    await _make_record(db_session, patient, sfid, ai_extracted=True)
    structured = await _make_record(db_session, patient, sfid, ai_extracted=False)
    await db_session.commit()

    n = await soft_delete_prior_extracted(db_session, sfid)
    await db_session.commit()
    assert n == 2

    live = (await db_session.execute(
        select(HealthRecord).where(HealthRecord.source_file_id == sfid, HealthRecord.deleted_at.is_(None))
    )).scalars().all()
    assert len(live) == 1 and live[0].id == structured.id

    prov = (await db_session.execute(
        select(Provenance).where(Provenance.action == "reextraction_replace")
    )).scalars().all()
    assert len(prov) == 2


@pytest.mark.asyncio
async def test_soft_delete_prior_extracted_noop_when_none(db_session):
    from app.services.ingestion.reextraction import soft_delete_prior_extracted
    assert await soft_delete_prior_extracted(db_session, uuid.uuid4()) == 0


@pytest.mark.asyncio
async def test_find_prior_extracted_upload_matches_completed(db_session, client):
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    prior = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="n.pdf", file_hash="HASH1",
        storage_path="/x", ingestion_status="completed", file_category="unstructured",
        record_count=5, mime_type="application/pdf",
    )
    db_session.add(prior)
    await db_session.commit()
    found = await find_prior_extracted_upload(db_session, uuid.UUID(uid), "HASH1")
    assert found is not None and found.id == prior.id


@pytest.mark.asyncio
async def test_find_prior_extracted_upload_ignores_failed_and_unknown(db_session, client):
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    failed = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="n.pdf", file_hash="HASH2",
        storage_path="/x", ingestion_status="failed", file_category="unstructured",
        mime_type="application/pdf",
    )
    db_session.add(failed)
    await db_session.commit()
    assert await find_prior_extracted_upload(db_session, uuid.UUID(uid), "HASH2") is None
    assert await find_prior_extracted_upload(db_session, uuid.UUID(uid), "NOPE") is None


@pytest.mark.asyncio
async def test_find_prior_extracted_upload_ignores_structured_uploads(db_session, client):
    """A structured upload sharing a file_hash must NOT be treated as an unstructured duplicate."""
    from app.services.ingestion.reextraction import find_prior_extracted_upload
    from tests.conftest import auth_headers
    _, uid = await auth_headers(client)
    structured = UploadedFile(
        id=uuid.uuid4(), user_id=uuid.UUID(uid), filename="bundle.json", file_hash="HASH3",
        storage_path="/x", ingestion_status="completed", file_category="structured",
        record_count=9, mime_type="application/json",
    )
    db_session.add(structured)
    await db_session.commit()
    assert await find_prior_extracted_upload(db_session, uuid.UUID(uid), "HASH3") is None
