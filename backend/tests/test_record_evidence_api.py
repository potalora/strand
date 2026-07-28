"""User-scoped, bounded extraction-evidence API tests."""

from __future__ import annotations

from dataclasses import asdict
from uuid import UUID

import pytest
from sqlalchemy import select

from app.models.audit import AuditLog
from app.models.local_ai import ExtractionEvidence
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
)
from app.services.local_ai.types import ModelRole
from tests.conftest import auth_headers, create_test_patient


def _manifest() -> LocalAIManifest:
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository=f"owner/{role.value}",
            revision=str(index) * 40,
            quantization="4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 32768, "max_output_tokens": 4096},
            files=(
                ManifestFile(
                    path="model.safetensors",
                    sha256=str(index) * 64,
                    size=10,
                ),
            ),
        )
        for index, role in enumerate(ModelRole, start=1)
    )
    return LocalAIManifest(
        schema_version=1,
        pack_revision="apple-m4-16gb-v1",
        platform="apple_silicon",
        runtime={"name": "mlx-vlm", "version": "0.5.0"},
        validation_suite_version="local-ai-fixtures-v1",
        artifacts=artifacts,
    )


async def _evidenced_record(db_session, user_id: str) -> tuple[HealthRecord, str]:
    patient = await create_test_patient(db_session, user_id)
    manifest = _manifest()
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename="note.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/note.pdf",
        processing_mode="validated_strict_local",
        processing_manifest=asdict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
        document_metadata={
            "unresolved_fields": ["labs[1].unit"],
            "rejected_fields": ["conditions[2].date"],
        },
    )
    db_session.add(upload)
    await db_session.flush()
    record = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="observation",
        fhir_resource_type="Observation",
        fhir_resource={"resourceType": "Observation"},
        source_format="local_ai",
        source_file_id=upload.id,
        display_text="A1c",
        ai_extracted=True,
    )
    db_session.add(record)
    await db_session.flush()
    excerpt = "Hemoglobin A1c 6.8 %"
    db_session.add(
        ExtractionEvidence(
            user_id=UUID(user_id),
            upload_id=upload.id,
            health_record_id=record.id,
            page_number=2,
            section="Laboratory",
            excerpt=excerpt,
            start_offset=11,
            end_offset=33,
            field_paths=["$.labs[0].value", "$.labs[0].unit"],
            source_metadata={
                "evidence_id": "evidence-1",
                "manifest_sha256": "f" * 64,
                "excerpt_sha256": "e" * 64,
                "offset_representation": "unicode-codepoint",
            },
        )
    )
    await db_session.commit()
    return record, excerpt


@pytest.mark.asyncio
async def test_user_cannot_read_another_users_evidence(client, db_session) -> None:
    _headers_a, user_a = await auth_headers(client, email="evidence-a@example.com")
    headers_b, _user_b = await auth_headers(client, email="evidence-b@example.com")
    record, _ = await _evidenced_record(db_session, user_a)

    response = await client.get(
        f"/api/v1/records/{record.id}/evidence",
        headers=headers_b,
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_record_evidence_is_bounded_and_contains_only_ingestion_provenance(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client)
    record, excerpt = await _evidenced_record(db_session, user_id)

    response = await client.get(
        f"/api/v1/records/{record.id}/evidence",
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["record_id"] == str(record.id)
    assert body["processing_mode"] == "validated_strict_local"
    assert body["schema_version"] == "clinical-document-extraction.v1"
    assert body["unresolved_fields"] == ["labs[1].unit"]
    assert body["rejected_fields"] == ["conditions[2].date"]
    assert body["evidence"] == [
        {
            "id": "evidence-1",
            "page_number": 2,
            "section": "Laboratory",
            "excerpt": excerpt,
            "start_offset": 11,
            "end_offset": 33,
            "field_paths": ["$.labs[0].value", "$.labs[0].unit"],
        }
    ]
    assert {item["role"] for item in body["models"]} == {
        "ocr",
        "extraction",
    }
    assert all(
        set(item) == {"role", "repository", "revision", "quantization", "runtime"}
        for item in body["models"]
    )
    assert "response_text" not in body
    assert "summary" not in body


@pytest.mark.asyncio
async def test_evidence_audit_never_contains_excerpt_or_field_content(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client)
    record, excerpt = await _evidenced_record(db_session, user_id)

    response = await client.get(
        f"/api/v1/records/{record.id}/evidence",
        headers=headers,
    )
    assert response.status_code == 200

    row = (
        await db_session.execute(
            select(AuditLog).where(AuditLog.action == "records.evidence")
        )
    ).scalar_one()
    assert row.resource_id == record.id
    assert row.details == {
        "evidence_count": 1,
        "processing_mode": "validated_strict_local",
    }
    assert excerpt not in str(row.details)
    assert "labs[0].value" not in str(row.details)
