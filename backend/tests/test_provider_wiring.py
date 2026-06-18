"""B2 wiring: the note-level provider is attached to AI-extracted records.

Agent B's `entity_to_health_record_dict` accepts an optional `document_provider`
and `resolve_document_provider` derives it from a `provider` entity. This test
covers the wiring Agent A owns: both `confirm_extraction` (manual) and
`_process_unstructured` (auto-confirm) must compute and pass it so encounters /
observations / procedures get a participant/performer even when the individual
entity carries no provider of its own.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from tests.conftest import TEST_DB_URL, auth_headers, create_test_patient


def _provider_of(record: HealthRecord) -> str | None:
    """Pull the provider display from whichever FHIR path the type uses."""
    res = record.fhir_resource or {}
    if record.record_type == "encounter":
        parts = res.get("participant") or []
        return parts[0]["individual"]["display"] if parts else None
    if record.record_type == "procedure":
        perfs = res.get("performer") or []
        return perfs[0]["actor"]["display"] if perfs else None
    # observation / others
    perfs = res.get("performer") or []
    return perfs[0]["display"] if perfs else None


@pytest.mark.asyncio
async def test_confirm_extraction_attaches_document_provider(
    client: AsyncClient, db_session: AsyncSession
):
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="note.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash=f"hash_{uuid4().hex}",
        storage_path="/tmp/note.rtf",
        ingestion_status="awaiting_confirmation",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    resp = await client.post(
        f"/api/v1/upload/{upload.id}/confirm-extraction",
        json={
            "patient_id": str(patient.id),
            "confirmed_entities": [
                {"entity_class": "provider", "text": "Dr. Lee",
                 "attributes": {"specialty": "GI"}, "confidence": 0.9},
                {"entity_class": "encounter", "text": "Office visit",
                 "attributes": {"visit_type": "office", "date": "03/15/2025"}, "confidence": 0.9},
                {"entity_class": "lab_result", "text": "Glucose 95 mg/dL",
                 "attributes": {"test": "Glucose", "value": "95", "unit": "mg/dL"}, "confidence": 0.9},
                {"entity_class": "procedure", "text": "Appendectomy",
                 "attributes": {"date": "01/2020"}, "confidence": 0.9},
            ],
        },
        headers=headers,
    )
    assert resp.status_code == 200
    # provider entity is non-storable → 3 records created
    assert resp.json()["records_created"] == 3

    records = (
        await db_session.execute(
            select(HealthRecord).where(HealthRecord.user_id == user_id)
        )
    ).scalars().all()
    by_type = {r.record_type: r for r in records}
    assert _provider_of(by_type["encounter"]) == "Dr. Lee"
    assert _provider_of(by_type["observation"]) == "Dr. Lee"
    assert _provider_of(by_type["procedure"]) == "Dr. Lee"


@pytest.mark.asyncio
async def test_auto_confirm_attaches_document_provider(db_session: AsyncSession):
    """The auto-confirm path (_process_unstructured) also derives + passes the
    note provider so AI records aren't 0% provider."""
    from app.api import upload as upload_module
    from app.models.patient import Patient
    from app.models.user import User
    from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult

    user = User(
        id=uuid4(),
        email="prov_wiring_enc",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(Patient(id=uuid4(), user_id=user.id, fhir_id="p-prov", gender="female"))
    await db_session.flush()

    rtf_path = Path("/tmp") / f"prov_{uuid4().hex}.rtf"
    rtf_path.write_bytes(rb"{\rtf1\ansi Seen by Dr. Vance. Office visit.}")
    upload = UploadedFile(
        id=uuid4(),
        user_id=user.id,
        filename="prov.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash=f"hash_{uuid4().hex}",
        storage_path=str(rtf_path),
        ingestion_status="pending_extraction",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    engine = create_async_engine(TEST_DB_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def fake_extract(text, source_file, api_key, progress_callback=None):
        return ExtractionResult(
            source_file=source_file,
            source_text=text,
            entities=[
                ExtractedEntity("provider", "Dr. Vance", {"specialty": "GI"}),
                ExtractedEntity("encounter", "Office visit",
                                {"visit_type": "office", "date": "03/15/2025"}),
            ],
        )

    try:
        with patch.object(upload_module, "async_session_factory", factory), patch(
            "app.services.extraction.entity_extractor.extract_entities_async",
            side_effect=fake_extract,
        ), patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ):
            await upload_module._process_unstructured(upload.id, rtf_path, user.id)

        async with factory() as verify:
            enc = (
                await verify.execute(
                    select(HealthRecord).where(
                        HealthRecord.user_id == user.id,
                        HealthRecord.record_type == "encounter",
                    )
                )
            ).scalar_one()
            assert _provider_of(enc) == "Dr. Vance"
    finally:
        await engine.dispose()
        rtf_path.unlink(missing_ok=True)
