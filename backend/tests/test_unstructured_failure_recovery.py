"""Fast integration test: a DB INSERT failure during unstructured extraction
must leave the upload cleanly 'failed' with the REAL error — not stuck in
'processing' (which makes _recover_stuck_files retry 3x and finally mislabel it
as a generic TimeoutError).

Gemini is fully mocked, so this is a fast test (no GEMINI_API_KEY needed).
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.uploaded_file import UploadedFile
from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult
from app.services.extraction.text_extractor import FileType
from tests.conftest import TEST_DB_URL, auth_headers


@pytest_asyncio.fixture
async def test_session_factory():
    """Session factory targeting medtimeline_test (what conftest truncates)."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _poison_record_dict(entity, user_id, patient_id, source_file_id=None):
    """Stand-in for entity_to_health_record_dict that yields a record Postgres
    rejects on INSERT: a non-datetime string in the timestamptz column. This
    reproduces the real-world failure (a bad extracted date reaching the DB)
    that poisons the session and made the original except-handler commit throw."""
    return {
        "id": uuid.uuid4(),
        "patient_id": patient_id,
        "user_id": user_id,
        "record_type": "condition",
        "fhir_resource_type": "Condition",
        "fhir_resource": {"resourceType": "Condition", "code": {"text": "x"}},
        "content_hash": "deadbeef",
        "source_format": "ai_extracted",
        "source_file_id": source_file_id,
        "effective_date": "this-is-not-a-timestamptz",  # asyncpg DataError on INSERT
        "status": "active",
        "category": ["condition"],
        "code_display": "x",
        "display_text": "x",
        "is_duplicate": False,
        "confidence_score": 0.8,
        "ai_extracted": True,
    }


@pytest.mark.asyncio
async def test_insert_failure_marks_failed_with_real_error_not_timeout(
    db_session: AsyncSession,
    client: AsyncClient,
    test_session_factory: async_sessionmaker,  # type: ignore[type-arg]
):
    from app.api.upload import _process_unstructured

    # ── Arrange: user + a pending unstructured upload row in the test DB ──
    headers, uid_str = await auth_headers(client)
    user_id = uuid.UUID(uid_str)

    upload_id = uuid.uuid4()
    upload = UploadedFile(
        id=upload_id,
        user_id=user_id,
        filename="note.pdf",
        mime_type="application/pdf",
        file_size_bytes=1234,
        file_hash="hash-" + upload_id.hex,
        storage_path="/tmp/does-not-need-to-exist.pdf",
        ingestion_status="processing",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    one_entity = ExtractionResult(
        source_file="note.pdf",
        source_text="t",
        entities=[ExtractedEntity(entity_class="condition", text="hypertension")],
    )

    # ── Act: drive extraction with Gemini mocked + a poisoned record dict ──
    raised: Exception | None = None
    with (
        patch("app.api.upload.async_session_factory", test_session_factory),
        patch(
            "app.services.extraction.text_extractor.extract_text",
            new=AsyncMock(return_value=("Patient has hypertension.", "pdf")),
        ),
        patch(
            "app.services.extraction.text_extractor.detect_file_type",
            return_value=FileType.PDF,
        ),
        patch(
            "app.services.extraction.entity_extractor.extract_entities_async",
            new=AsyncMock(return_value=one_entity),
        ),
        patch(
            "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
            new=_poison_record_dict,
        ),
        patch(
            "app.services.ingestion.coordinator._run_dedup_background",
            new_callable=AsyncMock,
        ),
    ):
        try:
            await _process_unstructured(upload_id, Path(upload.storage_path), user_id)
        except Exception as exc:  # noqa: BLE001 — we assert on clean handling below
            raised = exc

    # ── Assert: handler recovered the session and recorded the real failure ──
    assert raised is None, (
        f"_process_unstructured should swallow the INSERT failure, but it "
        f"raised {type(raised).__name__}: {raised}"
    )

    async with test_session_factory() as check:
        row = (
            await check.execute(select(UploadedFile).where(UploadedFile.id == upload_id))
        ).scalar_one()
        status = row.ingestion_status
        completed_at = row.processing_completed_at
        errors = row.ingestion_errors or []

    assert status == "failed", f"expected status 'failed', got '{status}'"
    assert completed_at is not None, "processing_completed_at must be set on failure"

    # The stored error must reflect the REAL DB failure, never the misleading
    # generic timeout that _recover_stuck_files would later stamp.
    error_types = {e.get("error_type") for e in errors}
    assert error_types, f"no error recorded: {errors}"
    assert "TimeoutError" not in error_types, (
        f"failure was mislabeled as a timeout: {errors}"
    )
    assert any(
        et and ("Error" in et or "Exception" in et) for et in error_types
    ), f"expected a real DB error type, got {error_types}"
