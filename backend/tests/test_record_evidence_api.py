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
from app.services.local_ai.evidence_lineage import (
    EvidenceLineageResult,
    load_strict_local_evidence_lineage,
)
from app.services.local_ai.types import ModelRole
from tests.conftest import auth_headers, create_test_patient


def _manifest(*, revision_offset: int = 0) -> LocalAIManifest:
    artifacts = tuple(
        ManifestArtifact(
            role=role,
            repository=f"owner/{role.value}",
            revision=str(index + revision_offset) * 40,
            quantization="4bit",
            license="apache-2.0",
            attribution=f"https://huggingface.co/owner/{role.value}",
            decode_limits={"max_input_tokens": 32768, "max_output_tokens": 4096},
            files=(
                ManifestFile(
                    path="model.safetensors",
                    sha256=str(index + revision_offset) * 64,
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


async def _archived_evidenced_record(
    db_session,
    user_id: str,
    survivor: HealthRecord,
    *,
    suffix: str,
    revision_offset: int = 0,
) -> tuple[HealthRecord, str]:
    manifest = _manifest(revision_offset=revision_offset)
    upload = UploadedFile(
        user_id=UUID(user_id),
        filename=f"note-{suffix}.pdf",
        mime_type="application/pdf",
        file_hash=(suffix[0] if suffix else "b") * 64,
        storage_path=f"/private/note-{suffix}.pdf",
        processing_mode="validated_strict_local",
        processing_manifest=asdict(manifest),
        processing_schema_version="clinical-document-extraction.v1",
        document_metadata={
            "unresolved_fields": [f"labs[{suffix}].unit"],
            "rejected_fields": [f"conditions[{suffix}].date"],
        },
    )
    db_session.add(upload)
    await db_session.flush()
    archived = HealthRecord(
        user_id=UUID(user_id),
        patient_id=survivor.patient_id,
        record_type=survivor.record_type,
        fhir_resource_type=survivor.fhir_resource_type,
        fhir_resource={"resourceType": survivor.fhir_resource_type},
        source_format="local_ai",
        source_file_id=upload.id,
        display_text=f"Archived {suffix}",
        ai_extracted=True,
        is_duplicate=True,
        merged_into_id=survivor.id,
    )
    db_session.add(archived)
    await db_session.flush()
    excerpt = f"Corroborating evidence {suffix}"
    db_session.add(
        ExtractionEvidence(
            user_id=UUID(user_id),
            upload_id=upload.id,
            health_record_id=archived.id,
            page_number=3 + revision_offset,
            section="Laboratory",
            excerpt=excerpt,
            start_offset=10,
            end_offset=10 + len(excerpt),
            field_paths=["$.labs[0].value"],
            source_metadata={
                "evidence_id": f"evidence-{suffix}",
                "manifest_sha256": str(7 + revision_offset) * 64,
                "excerpt_sha256": str(8 + revision_offset) * 64,
                "offset_representation": "unicode-codepoint",
            },
        )
    )
    await db_session.commit()
    return archived, excerpt


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


@pytest.mark.asyncio
async def test_survivor_exposes_archived_strict_evidence_and_undo_restores_scope(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client, email="evidence-merge@example.com")
    survivor, survivor_excerpt = await _evidenced_record(db_session, user_id)
    archived, archived_excerpt = await _archived_evidenced_record(
        db_session,
        user_id,
        survivor,
        suffix="b",
    )

    response = await client.get(
        f"/api/v1/records/{survivor.id}/evidence",
        headers=headers,
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["evidence"]] == [
        "evidence-1",
        "evidence-b",
    ]
    assert {item["excerpt"] for item in response.json()["evidence"]} == {
        survivor_excerpt,
        archived_excerpt,
    }

    archived.is_duplicate = False
    archived.merged_into_id = None
    await db_session.commit()

    survivor_response = await client.get(
        f"/api/v1/records/{survivor.id}/evidence",
        headers=headers,
    )
    archived_response = await client.get(
        f"/api/v1/records/{archived.id}/evidence",
        headers=headers,
    )
    assert [item["id"] for item in survivor_response.json()["evidence"]] == [
        "evidence-1"
    ]
    assert [item["id"] for item in archived_response.json()["evidence"]] == [
        "evidence-b"
    ]


@pytest.mark.asyncio
async def test_non_strict_survivor_unions_strict_child_manifests_and_diagnostics(
    client,
    db_session,
) -> None:
    headers, user_id = await auth_headers(client, email="evidence-union@example.com")
    patient = await create_test_patient(db_session, user_id)
    survivor = HealthRecord(
        user_id=UUID(user_id),
        patient_id=patient.id,
        record_type="observation",
        fhir_resource_type="Observation",
        fhir_resource={"resourceType": "Observation"},
        source_format="fhir",
        source_file_id=None,
        display_text="Structured survivor",
        ai_extracted=False,
    )
    db_session.add(survivor)
    await db_session.flush()
    await _archived_evidenced_record(
        db_session,
        user_id,
        survivor,
        suffix="b",
    )
    await _archived_evidenced_record(
        db_session,
        user_id,
        survivor,
        suffix="c",
        revision_offset=3,
    )

    response = await client.get(
        f"/api/v1/records/{survivor.id}/evidence",
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["evidence"]] == [
        "evidence-b",
        "evidence-c",
    ]
    assert body["unresolved_fields"] == ["labs[b].unit", "labs[c].unit"]
    assert body["rejected_fields"] == [
        "conditions[b].date",
        "conditions[c].date",
    ]
    assert {(item["role"], item["revision"]) for item in body["models"]} == {
        ("ocr", "1" * 40),
        ("extraction", "2" * 40),
        ("ocr", "4" * 40),
        ("extraction", "5" * 40),
    }


@pytest.mark.asyncio
async def test_survivor_lineage_never_traverses_another_users_archived_record(
    client,
    db_session,
) -> None:
    headers_a, user_a = await auth_headers(client, email="evidence-owner-a@example.com")
    _headers_b, user_b = await auth_headers(
        client, email="evidence-owner-b@example.com"
    )
    survivor, _ = await _evidenced_record(db_session, user_a)
    patient_b = await create_test_patient(db_session, user_b)
    foreign_record = HealthRecord(
        user_id=UUID(user_b),
        patient_id=patient_b.id,
        record_type="observation",
        fhir_resource_type="Observation",
        fhir_resource={"resourceType": "Observation"},
        source_format="fhir",
        source_file_id=None,
        display_text="Foreign archived record",
        ai_extracted=False,
        is_duplicate=True,
        merged_into_id=survivor.id,
    )
    db_session.add(foreign_record)
    await db_session.commit()

    response = await client.get(
        f"/api/v1/records/{survivor.id}/evidence",
        headers=headers_a,
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["evidence"]] == ["evidence-1"]


@pytest.mark.asyncio
async def test_shared_lineage_loader_is_owner_scoped_and_signals_global_bound(
    client,
    db_session,
) -> None:
    _headers_a, user_a = await auth_headers(
        client,
        email="evidence-loader-a@example.com",
    )
    _headers_b, user_b = await auth_headers(
        client,
        email="evidence-loader-b@example.com",
    )
    survivor, _ = await _evidenced_record(db_session, user_a)
    await _archived_evidenced_record(
        db_session,
        user_a,
        survivor,
        suffix="b",
    )
    await _archived_evidenced_record(
        db_session,
        user_a,
        survivor,
        suffix="c",
        revision_offset=3,
    )
    await _archived_evidenced_record(
        db_session,
        user_b,
        survivor,
        suffix="d",
    )

    bounded = await load_strict_local_evidence_lineage(
        db_session,
        user_id=UUID(user_a),
        survivor_ids=[survivor.id],
        limit=1,
    )
    complete = await load_strict_local_evidence_lineage(
        db_session,
        user_id=UUID(user_a),
        survivor_ids=[survivor.id],
        limit=10,
    )

    assert bounded.overflowed is True
    assert len(bounded.rows) == 1
    assert complete.overflowed is False
    assert {row.evidence.source_metadata["evidence_id"] for row in complete.rows} == {
        "evidence-1",
        "evidence-b",
        "evidence-c",
    }
    assert all(row.evidence.user_id == UUID(user_a) for row in complete.rows)
    assert set(complete.by_survivor()) == {survivor.id}


@pytest.mark.asyncio
async def test_record_evidence_overflow_is_an_explicit_conflict(
    client,
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.api.records as records_api

    headers, user_id = await auth_headers(
        client,
        email="evidence-overflow@example.com",
    )
    survivor, _ = await _evidenced_record(db_session, user_id)

    async def overflow(*_args, **_kwargs) -> EvidenceLineageResult:
        return EvidenceLineageResult(rows=(), overflowed=True)

    monkeypatch.setattr(records_api, "load_strict_local_evidence_lineage", overflow)
    response = await client.get(
        f"/api/v1/records/{survivor.id}/evidence",
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Record extraction evidence exceeds the supported limit"
    }
