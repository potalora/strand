"""Integration test for the local extraction-engine helper (WS-A).

Drives ``_run_local_extraction_engine`` against a real DB upload row with the
real medspaCy/scispaCy models (no Gemini), proving the pipeline wiring:
section detection → local NER → ConText → parsed_doc + document_metadata.
Model-gated: skips when scispaCy/medspaCy are not installed.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.uploaded_file import UploadedFile
from app.models.user import User

pytestmark = pytest.mark.asyncio


def _models_available() -> bool:
    from app.services.extraction.clinical_context import get_clinical_context
    from app.services.extraction.local_ner import get_local_ner

    return get_local_ner().available and get_clinical_context().available


_NOTE = (
    "Past Medical History:\n"
    "Type 2 diabetes mellitus, hypertension, asthma.\n\n"
    "Medications:\n"
    "metformin 500mg twice daily\n"
    "lisinopril 10mg daily\n"
)


@pytest.mark.skipif(not _models_available(), reason="scispaCy/medspaCy not installed")
async def test_local_engine_helper_end_to_end(db_session: AsyncSession):
    from app.api.upload import _run_local_extraction_engine

    user = User(id=uuid4(), login_identifier=f"wsa_{uuid4().hex[:8]}@example.com", password_hash="x")
    db_session.add(user)
    await db_session.commit()
    user_id = user.id
    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="note.rtf",
        mime_type="application/rtf",
        file_size_bytes=len(_NOTE),
        file_hash=f"hash_{uuid4().hex}",
        storage_path=f"/tmp/{uuid4().hex}.rtf",
        ingestion_status="processing",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    sem = asyncio.Semaphore(5)
    out = await _run_local_extraction_engine(
        db_session, upload, upload.id, user_id, _NOTE, "local", sem
    )
    assert out is not None
    entities, parsed_doc = out

    meds = {e.text.lower() for e in entities if e.entity_class == "medication"}
    conds = {e.text.lower() for e in entities if e.entity_class == "condition"}
    assert {"metformin", "lisinopril"} <= meds
    assert "hypertension" in conds and "asthma" in conds

    # parsed_doc + upload bookkeeping populated for the downstream auto-confirm.
    assert parsed_doc.sections
    assert upload.document_metadata["extraction_engine"] == "local"
    assert "extraction_stats" in upload.document_metadata
    assert upload.extraction_sections["sections"]
