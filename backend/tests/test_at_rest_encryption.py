"""At-rest encryption (W7 / CRYPTO-01) + email blind index (W17) tests.

Verifies that clinical PHI columns (``health_records.fhir_resource``,
``uploaded_files.extracted_text`` & extraction JSON, AI-summary prompts/response,
``users.email``) are encrypted at rest via the AES-256-GCM ``EncryptedJSON`` /
``EncryptedText`` SQLAlchemy ``TypeDecorator``s, while remaining transparent to
the ORM (read back identical), and that email lookups go through a deterministic
HMAC blind index so login still works without a queryable plaintext column.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from app.middleware.encryption import blind_index
from app.models.ai_summary import AISummaryPrompt
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.auth_service import authenticate_user, register_user
from tests.conftest import create_test_patient

# asyncio_mode = "auto" (pyproject) auto-marks the async tests; the lone sync
# test (blind index) needs no marker.

# A recognizable PHI marker we plant inside the clinical payloads, then assert is
# absent from the raw on-disk bytes.
PHI_MARKER = "ZacharyQuirkePHImarker"


async def _make_user(db_session, email: str = "enc-user@example.com"):
    return await register_user(db_session, email=email, password="SecurePass123!")


# ---------------------------------------------------------------------------
# (1) Transparent round-trip: a dict written reads back as the same dict.
# ---------------------------------------------------------------------------
async def test_fhir_resource_roundtrips_as_same_dict(db_session):
    user = await _make_user(db_session)
    patient = await create_test_patient(db_session, user.id)

    payload = {
        "resourceType": "Condition",
        "subject": {"display": PHI_MARKER},
        "code": {"coding": [{"display": "Type 2 diabetes"}]},
        "note": [{"text": "patient reports symptoms"}],
    }
    rec = HealthRecord(
        id=uuid.uuid4(),
        patient_id=patient.id,
        user_id=user.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource=payload,
        source_format="fhir_r4",
        display_text="Type 2 diabetes",
    )
    db_session.add(rec)
    await db_session.commit()
    db_session.expunge_all()

    fetched = (
        await db_session.execute(
            text("SELECT id FROM health_records WHERE id = :id"), {"id": rec.id}
        )
    ).first()
    assert fetched is not None
    reloaded = await db_session.get(HealthRecord, rec.id)
    assert reloaded.fhir_resource == payload


# ---------------------------------------------------------------------------
# (2) The raw stored bytes are ciphertext — the plaintext PHI is NOT present.
# ---------------------------------------------------------------------------
async def test_fhir_resource_stored_as_ciphertext(db_session):
    user = await _make_user(db_session, "enc-cipher@example.com")
    patient = await create_test_patient(db_session, user.id)

    rec = HealthRecord(
        id=uuid.uuid4(),
        patient_id=patient.id,
        user_id=user.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition", "subject": {"display": PHI_MARKER}},
        source_format="fhir_r4",
        display_text="x",
    )
    db_session.add(rec)
    await db_session.commit()

    raw = (
        await db_session.execute(
            text("SELECT fhir_resource FROM health_records WHERE id = :id"),
            {"id": rec.id},
        )
    ).scalar_one()
    raw_bytes = bytes(raw)
    assert PHI_MARKER.encode() not in raw_bytes
    # Sanity: it is genuinely opaque bytes, not a JSON string column.
    assert b"resourceType" not in raw_bytes


async def test_extracted_text_and_entities_encrypted(db_session):
    user = await _make_user(db_session, "enc-text@example.com")

    uf = UploadedFile(
        id=uuid.uuid4(),
        user_id=user.id,
        filename="note.pdf",
        mime_type="application/pdf",
        file_hash="abc",
        storage_path="/tmp/x",
        extracted_text=f"Visit note for {PHI_MARKER}",
        extraction_entities=[{"type": "condition", "text": PHI_MARKER}],
        extraction_sections={"history": PHI_MARKER},
        document_metadata={"author": PHI_MARKER},
    )
    db_session.add(uf)
    await db_session.commit()

    reloaded = await db_session.get(UploadedFile, uf.id)
    assert reloaded.extracted_text == f"Visit note for {PHI_MARKER}"
    assert reloaded.extraction_entities == [{"type": "condition", "text": PHI_MARKER}]
    assert reloaded.extraction_sections == {"history": PHI_MARKER}
    assert reloaded.document_metadata == {"author": PHI_MARKER}

    for col in ("extracted_text", "extraction_entities", "extraction_sections", "document_metadata"):
        raw = (
            await db_session.execute(
                text(f"SELECT {col} FROM uploaded_files WHERE id = :id"), {"id": uf.id}
            )
        ).scalar_one()
        assert PHI_MARKER.encode() not in bytes(raw), f"{col} leaked plaintext"


async def test_ai_summary_prompt_columns_encrypted(db_session):
    user = await _make_user(db_session, "enc-summary@example.com")
    patient = await create_test_patient(db_session, user.id)

    summary = AISummaryPrompt(
        id=uuid.uuid4(),
        user_id=user.id,
        patient_id=patient.id,
        summary_type="full",
        system_prompt=f"system {PHI_MARKER}",
        user_prompt=f"user {PHI_MARKER}",
        target_model="gemini-3.5-flash",
        suggested_config={"temperature": 0.2},
        record_count=3,
        response_text=f"response {PHI_MARKER}",
    )
    db_session.add(summary)
    await db_session.commit()

    reloaded = await db_session.get(AISummaryPrompt, summary.id)
    assert reloaded.system_prompt == f"system {PHI_MARKER}"
    assert reloaded.user_prompt == f"user {PHI_MARKER}"
    assert reloaded.response_text == f"response {PHI_MARKER}"

    for col in ("system_prompt", "user_prompt", "response_text"):
        raw = (
            await db_session.execute(
                text(f"SELECT {col} FROM ai_summary_prompts WHERE id = :id"),
                {"id": summary.id},
            )
        ).scalar_one()
        assert PHI_MARKER.encode() not in bytes(raw), f"{col} leaked plaintext"


# ---------------------------------------------------------------------------
# (3) Login / get-by-email works through the blind index, not a plaintext col.
# ---------------------------------------------------------------------------
async def test_login_through_blind_index(db_session):
    email = "blind-login@example.com"
    await register_user(db_session, email=email, password="SecurePass123!")

    tokens = await authenticate_user(db_session, email=email, password="SecurePass123!")
    assert tokens.access_token
    assert tokens.refresh_token

    # The encrypted email column must not be queryable as plaintext, while the
    # blind-index column resolves the row.
    by_hmac = (
        await db_session.execute(
            text("SELECT email FROM users WHERE email_hmac = :h"),
            {"h": blind_index(email)},
        )
    ).first()
    assert by_hmac is not None
    raw_email = bytes(by_hmac[0])
    assert email.encode() not in raw_email  # ciphertext, not plaintext


async def test_email_stored_as_ciphertext_and_decrypts(db_session):
    email = "enc-email@example.com"
    user = await register_user(db_session, email=email, password="SecurePass123!")
    # ORM read decrypts transparently.
    reloaded = await db_session.get(type(user), user.id)
    assert reloaded.email == email
    raw = (
        await db_session.execute(
            text("SELECT email FROM users WHERE id = :id"), {"id": user.id}
        )
    ).scalar_one()
    assert email.encode() not in bytes(raw)


# ---------------------------------------------------------------------------
# (4) Blind index: deterministic, normalized, collision-free across emails.
# ---------------------------------------------------------------------------
def test_blind_index_deterministic_and_distinct():
    a = blind_index("Person.A@Example.com")
    b = blind_index("person.b@example.com")
    assert a != b
    # Same email (case + surrounding whitespace normalized) -> same index.
    assert blind_index("Person.A@Example.com") == blind_index("  person.a@example.com  ")
    # Hex digest of HMAC-SHA256 -> 64 hex chars.
    assert len(a) == 64
    int(a, 16)  # parses as hex
