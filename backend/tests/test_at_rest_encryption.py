"""At-rest encryption (W7 / CRYPTO-01) + login blind index (W17) tests.

Verifies that clinical PHI columns (``health_records.fhir_resource``,
``uploaded_files.extracted_text`` & extraction JSON, AI-summary prompts/response,
``users.login_identifier``) are encrypted at rest via the AES-256-GCM ``EncryptedJSON`` /
``EncryptedText`` SQLAlchemy ``TypeDecorator``s, while remaining transparent to
the ORM (read back identical), and that login lookups go through a deterministic
HMAC blind index so login still works without a queryable plaintext column.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from app.middleware.encryption import blind_index
from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import ExtractionEvidence, LocalAIJob, LocalAIPage
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.auth_service import authenticate_user, register_user
from tests.conftest import create_test_patient

# asyncio_mode = "auto" (pyproject) auto-marks the async tests; the lone sync
# test (blind index) needs no marker.

# A recognizable PHI marker we plant inside the clinical payloads, then assert is
# absent from the raw on-disk bytes.
PHI_MARKER = "ZacharyQuirkePHImarker"
OCR_MARKDOWN_MARKER = "OcrCheckpointCanary-RQ83V"
EVIDENCE_EXCERPT_MARKER = "EvidenceExcerptCanary-FM29K"
EVIDENCE_FIELD_PATH_MARKER = "EvidenceFieldPathCanary-HX47P"
EVIDENCE_SOURCE_MARKER = "EvidenceSourceCanary-NC61J"
TYPED_SUMMARY_MARKER = "TypedSummaryCanary-BD95W"


async def _make_user(db_session, login_identifier: str = "enc-user@example.com"):
    return await register_user(
        db_session,
        login_identifier=login_identifier,
        password="SecurePass123!",
    )


def _local_ai_manifest() -> dict:
    artifacts = []
    for role, hash_character in (
        ("ocr", "a"),
        ("extraction", "b"),
        ("summary", "c"),
    ):
        artifacts.append(
            {
                "role": role,
                "repository": f"owner/{role}",
                "revision": "0" * 40,
                "quantization": "4bit",
                "license": "apache-2.0",
                "attribution": f"https://huggingface.co/owner/{role}",
                "decode_limits": {
                    "max_input_tokens": 4096,
                    "max_output_tokens": 1024,
                },
                "files": [
                    {
                        "path": f"{role}/model.safetensors",
                        "sha256": hash_character * 64,
                        "size": 10,
                    }
                ],
            }
        )
    return {
        "schema_version": 2,
        "pack_revision": "apple-m4-16gb-v2",
        "platform": "apple_silicon",
        "runtime": {
            "name": "mlx-vlm",
            "version": "0.5.0",
            "worker_identity_scheme": "local-ai-worker-bundle.v1",
            "worker_bundle_sha256": "d" * 64,
        },
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": artifacts,
    }


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
        processing_mode="validated_strict_local",
        model_provenance={
            "role": "summary",
            "revision": "0123456789abcdef0123456789abcdef01234567",
        },
        typed_response={
            "sections": [
                {
                    "kind": "record_overview",
                    "text": TYPED_SUMMARY_MARKER,
                    "evidence_ids": ["evidence-1"],
                }
            ]
        },
    )
    db_session.add(summary)
    await db_session.commit()

    reloaded = await db_session.get(AISummaryPrompt, summary.id)
    assert reloaded.system_prompt == f"system {PHI_MARKER}"
    assert reloaded.user_prompt == f"user {PHI_MARKER}"
    assert reloaded.response_text == f"response {PHI_MARKER}"
    assert reloaded.typed_response["sections"][0]["text"] == TYPED_SUMMARY_MARKER

    for col, marker in (
        ("system_prompt", PHI_MARKER),
        ("user_prompt", PHI_MARKER),
        ("response_text", PHI_MARKER),
        ("typed_response", TYPED_SUMMARY_MARKER),
    ):
        raw = (
            await db_session.execute(
                text(f"SELECT {col} FROM ai_summary_prompts WHERE id = :id"),
                {"id": summary.id},
            )
        ).scalar_one()
        assert marker.encode() not in bytes(raw), f"{col} leaked plaintext"


async def test_local_ai_checkpoints_and_evidence_are_encrypted(db_session):
    user = await _make_user(db_session, "enc-local-ai@example.com")
    upload = UploadedFile(
        id=uuid.uuid4(),
        user_id=user.id,
        filename="strict-local.pdf",
        mime_type="application/pdf",
        file_hash="b" * 64,
        storage_path="/tmp/strict-local.pdf",
        processing_mode="validated_strict_local",
        processing_manifest={"pack_revision": "apple-m4-16gb-v1"},
        processing_schema_version="1",
    )
    db_session.add(upload)
    await db_session.flush()

    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_local_ai_manifest(),
        status="running",
        stage="ocr",
        progress={"pages_completed": 0, "pages_total": 1},
        audit_metadata={"attempt_count": 1},
    )
    db_session.add(job)
    await db_session.flush()

    page = LocalAIPage(
        job_id=job.id,
        page_number=1,
        checkpoint_key="c" * 64,
        image_sha256="d" * 64,
        ocr_result={
            "markdown": f"# Clinical note\n\n{OCR_MARKDOWN_MARKER}",
            "width": 1275,
            "height": 1650,
        },
        warnings=["ocr_layout_uncertain"],
    )
    evidence = ExtractionEvidence(
        user_id=user.id,
        upload_id=upload.id,
        page_number=1,
        section="medications",
        excerpt=EVIDENCE_EXCERPT_MARKER,
        start_offset=10,
        end_offset=42,
        field_paths=[f"$.medications[0].{EVIDENCE_FIELD_PATH_MARKER}"],
        source_metadata={
            "source_kind": "ocr_page",
            "anchor": EVIDENCE_SOURCE_MARKER,
        },
    )
    db_session.add_all([page, evidence])
    await db_session.commit()

    reloaded_page = await db_session.get(LocalAIPage, page.id)
    reloaded_evidence = await db_session.get(ExtractionEvidence, evidence.id)
    assert reloaded_page.ocr_result["markdown"].endswith(OCR_MARKDOWN_MARKER)
    assert reloaded_evidence.excerpt == EVIDENCE_EXCERPT_MARKER
    assert reloaded_evidence.field_paths == [
        f"$.medications[0].{EVIDENCE_FIELD_PATH_MARKER}"
    ]
    assert reloaded_evidence.source_metadata["anchor"] == EVIDENCE_SOURCE_MARKER

    raw_page = (
        await db_session.execute(
            text("SELECT ocr_result FROM local_ai_pages WHERE id = :id"),
            {"id": page.id},
        )
    ).scalar_one()
    assert OCR_MARKDOWN_MARKER.encode() not in bytes(raw_page)

    for column, marker in (
        ("excerpt", EVIDENCE_EXCERPT_MARKER),
        ("field_paths", EVIDENCE_FIELD_PATH_MARKER),
        ("source_metadata", EVIDENCE_SOURCE_MARKER),
    ):
        raw_evidence = (
            await db_session.execute(
                text(f"SELECT {column} FROM extraction_evidence WHERE id = :id"),
                {"id": evidence.id},
            )
        ).scalar_one()
        assert marker.encode() not in bytes(raw_evidence), f"{column} leaked plaintext"


# ---------------------------------------------------------------------------
# (3) Login works through the blind index, not a plaintext column.
# ---------------------------------------------------------------------------
async def test_login_through_blind_index(db_session):
    login_identifier = "blind-login@example.com"
    await register_user(
        db_session,
        login_identifier=login_identifier,
        password="SecurePass123!",
    )

    tokens = await authenticate_user(
        db_session,
        login_identifier=login_identifier,
        password="SecurePass123!",
    )
    assert tokens.access_token
    assert tokens.refresh_token

    # The encrypted identifier column must not be queryable as plaintext, while the
    # blind-index column resolves the row.
    by_hmac = (
        await db_session.execute(
            text(
                "SELECT login_identifier FROM users "
                "WHERE login_identifier_hmac = :h"
            ),
            {"h": blind_index(login_identifier)},
        )
    ).first()
    assert by_hmac is not None
    raw_identifier = bytes(by_hmac[0])
    assert login_identifier.encode() not in raw_identifier


async def test_login_identifier_stored_as_ciphertext_and_decrypts(db_session):
    login_identifier = "encrypted private account name"
    user = await register_user(
        db_session,
        login_identifier=login_identifier,
        password="SecurePass123!",
    )
    # ORM read decrypts transparently.
    reloaded = await db_session.get(type(user), user.id)
    assert reloaded.login_identifier == login_identifier
    raw = (
        await db_session.execute(
            text("SELECT login_identifier FROM users WHERE id = :id"),
            {"id": user.id},
        )
    ).scalar_one()
    assert login_identifier.encode() not in bytes(raw)


# ---------------------------------------------------------------------------
# (4) Blind index: deterministic, normalized, collision-free across identifiers.
# ---------------------------------------------------------------------------
def test_blind_index_deterministic_and_distinct():
    a = blind_index("Person.A@Example.com")
    b = blind_index("person.b@example.com")
    assert a != b
    # Same identifier (case + surrounding whitespace normalized) -> same index.
    assert blind_index("Person.A@Example.com") == blind_index("  person.a@example.com  ")
    # Hex digest of HMAC-SHA256 -> 64 hex chars.
    assert len(a) == 64
    int(a, 16)  # parses as hex
