from __future__ import annotations

import hashlib
import json
import re
import uuid

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, text
from sqlalchemy.exc import IntegrityError

from app.models.ai_summary import AISummaryPrompt
from app.models.encrypted_types import EncryptedJSON, EncryptedText
from app.models.llm_settings import UserLLMPreferences
from app.models.local_ai import ExtractionEvidence, LocalAIJob, LocalAIPage
from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.services.local_ai.errors import LocalValidationError


def _artifact(role: str) -> dict:
    hash_character = {"ocr": "a", "extraction": "b", "summary": "c"}[role]
    return {
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


def _valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": "local-ai-fixtures-v1",
        "artifacts": [_artifact(role) for role in ("ocr", "extraction", "summary")],
    }


def _upload_for(user: User) -> UploadedFile:
    return UploadedFile(
        user_id=user.id,
        filename="strict-local-note.pdf",
        mime_type="application/pdf",
        file_hash="a" * 64,
        storage_path="/private/strict-local-note.pdf",
        processing_mode="validated_strict_local",
        processing_manifest={
            "pack_revision": "apple-m4-16gb-v1",
            "artifact_count": 3,
        },
        processing_schema_version="1",
    )


def _patient_for(user: User, suffix: str) -> Patient:
    return Patient(
        user_id=user.id,
        fhir_id=f"local-ai-{suffix}",
        gender="unknown",
    )


def _summary_for(user: User, patient: Patient) -> AISummaryPrompt:
    return AISummaryPrompt(
        user_id=user.id,
        patient_id=patient.id,
        summary_type="full",
        system_prompt="server-owned safety instructions",
        user_prompt="de-identified record facts",
        target_model="local-summary",
        suggested_config={"temperature": 0},
        record_count=1,
        processing_mode="validated_strict_local",
    )


def _record_for(
    user: User,
    patient: Patient,
    upload: UploadedFile,
) -> HealthRecord:
    return HealthRecord(
        user_id=user.id,
        patient_id=patient.id,
        record_type="condition",
        fhir_resource_type="Condition",
        fhir_resource={"resourceType": "Condition"},
        source_format="local_ai",
        source_file_id=upload.id,
        display_text="synthetic condition",
    )


async def test_local_job_locks_mode_and_manifest_when_preferences_change(db_session) -> None:
    user = User(email="local-job-models@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    preference = UserLLMPreferences(
        user_id=user.id,
        processing_mode="validated_strict_local",
    )
    db_session.add_all([upload, preference])
    await db_session.flush()

    manifest_snapshot = _valid_manifest()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode=preference.processing_mode,
        manifest_snapshot=manifest_snapshot,
        status="queued",
        stage="preflight",
        progress={"pages_completed": 0, "pages_total": 1},
        audit_metadata={"attempt_count": 0},
    )
    db_session.add(job)
    await db_session.commit()

    preference.processing_mode = "cloud_assisted"
    await db_session.commit()
    await db_session.refresh(job)

    assert job.processing_mode == "validated_strict_local"
    assert job.manifest_snapshot == manifest_snapshot


def test_local_job_requires_kind_matched_target() -> None:
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in LocalAIJob.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert constraints["ck_local_ai_jobs_kind_target"] == (
        "(kind = 'ingestion' AND upload_id IS NOT NULL AND summary_prompt_id IS NULL) "
        "OR (kind = 'summary' AND upload_id IS NULL AND summary_prompt_id IS NOT NULL)"
    )


@pytest.mark.parametrize(
    "case",
    (
        "summary_kind_with_upload",
        "ingestion_kind_with_summary",
        "cross_user_upload",
        "cross_user_summary",
    ),
)
async def test_local_job_rejects_kind_and_owner_mismatches(db_session, case: str) -> None:
    user_a = User(email=f"{case}-a@example.com", password_hash="x")
    user_b = User(email=f"{case}-b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    await db_session.flush()
    upload_a = _upload_for(user_a)
    upload_b = _upload_for(user_b)
    patient_a = _patient_for(user_a, f"{case}-a")
    patient_b = _patient_for(user_b, f"{case}-b")
    db_session.add_all([upload_a, upload_b, patient_a, patient_b])
    await db_session.flush()
    summary_a = _summary_for(user_a, patient_a)
    summary_b = _summary_for(user_b, patient_b)
    db_session.add_all([summary_a, summary_b])
    await db_session.flush()

    target = {
        "summary_kind_with_upload": {
            "kind": "summary",
            "upload_id": upload_a.id,
        },
        "ingestion_kind_with_summary": {
            "kind": "ingestion",
            "summary_prompt_id": summary_a.id,
        },
        "cross_user_upload": {
            "kind": "ingestion",
            "upload_id": upload_b.id,
        },
        "cross_user_summary": {
            "kind": "summary",
            "summary_prompt_id": summary_b.id,
        },
    }[case]
    job = LocalAIJob(
        user_id=user_a.id,
        processing_mode="validated_strict_local",
        manifest_snapshot=_valid_manifest(),
        status="queued",
        stage="preflight",
        **target,
    )
    db_session.add(job)

    try:
        with pytest.raises(IntegrityError):
            await db_session.commit()
    finally:
        await db_session.rollback()


async def test_evidence_rejects_user_upload_mismatch(db_session) -> None:
    user_a = User(email="evidence-upload-a@example.com", password_hash="x")
    user_b = User(email="evidence-upload-b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    await db_session.flush()
    upload_b = _upload_for(user_b)
    db_session.add(upload_b)
    await db_session.flush()
    evidence = ExtractionEvidence(
        user_id=user_a.id,
        upload_id=upload_b.id,
        excerpt="synthetic excerpt",
        field_paths=["$.conditions[0]"],
        source_metadata={"source_kind": "ocr_page"},
    )
    db_session.add(evidence)

    try:
        with pytest.raises(IntegrityError):
            await db_session.commit()
    finally:
        await db_session.rollback()


@pytest.mark.parametrize("case", ("record_owner", "record_upload"))
async def test_evidence_rejects_health_record_scope_mismatch(
    db_session,
    case: str,
) -> None:
    user_a = User(email=f"evidence-record-{case}-a@example.com", password_hash="x")
    user_b = User(email=f"evidence-record-{case}-b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    await db_session.flush()
    upload_a = _upload_for(user_a)
    upload_b = _upload_for(user_b if case == "record_owner" else user_a)
    patient_a = _patient_for(user_a, f"record-{case}-a")
    patient_b = _patient_for(user_b, f"record-{case}-b")
    db_session.add_all([upload_a, upload_b, patient_a, patient_b])
    await db_session.flush()
    record = _record_for(
        user_b if case == "record_owner" else user_a,
        patient_b if case == "record_owner" else patient_a,
        upload_b,
    )
    db_session.add(record)
    await db_session.flush()
    evidence = ExtractionEvidence(
        user_id=user_a.id,
        upload_id=upload_a.id,
        health_record_id=record.id,
        excerpt="synthetic excerpt",
        field_paths=["$.conditions[0]"],
        source_metadata={"source_kind": "ocr_page"},
    )
    db_session.add(evidence)

    try:
        with pytest.raises(IntegrityError):
            await db_session.commit()
    finally:
        await db_session.rollback()


async def test_deleting_health_record_preserves_evidence_with_null_record_link(
    db_session,
) -> None:
    user = User(email="evidence-delete-record@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    patient = _patient_for(user, "delete-record")
    db_session.add_all([upload, patient])
    await db_session.flush()
    record = _record_for(user, patient, upload)
    db_session.add(record)
    await db_session.flush()
    evidence = ExtractionEvidence(
        user_id=user.id,
        upload_id=upload.id,
        health_record_id=record.id,
        excerpt="synthetic excerpt",
        field_paths=["$.conditions[0]"],
        source_metadata={"source_kind": "ocr_page"},
    )
    db_session.add(evidence)
    await db_session.commit()

    await db_session.delete(record)
    await db_session.commit()
    await db_session.refresh(evidence)

    assert evidence.health_record_id is None


def test_local_page_derives_upload_scope_from_job() -> None:
    assert "upload_id" not in LocalAIPage.__table__.c


def test_local_ai_parent_scope_constraints_are_named() -> None:
    def unique_names(model: type) -> set[str | None]:
        return {
            constraint.name
            for constraint in model.__table__.constraints
            if isinstance(constraint, UniqueConstraint)
        }

    assert "uq_uploaded_files_id_user_id" in unique_names(UploadedFile)
    assert "uq_ai_summary_prompts_id_user_id" in unique_names(AISummaryPrompt)
    assert "uq_health_records_id_source_file_user" in unique_names(HealthRecord)


def test_local_job_rejects_incomplete_manifest_snapshot() -> None:
    incomplete = _valid_manifest()
    incomplete["artifacts"] = incomplete["artifacts"][:1]

    with pytest.raises(LocalValidationError, match="manifest"):
        LocalAIJob(
            user_id=uuid.uuid4(),
            upload_id=uuid.uuid4(),
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=incomplete,
            status="queued",
            stage="preflight",
        )


def test_local_job_rejects_invalid_processing_mode() -> None:
    with pytest.raises(LocalValidationError, match="mode"):
        LocalAIJob(
            user_id=uuid.uuid4(),
            upload_id=uuid.uuid4(),
            kind="ingestion",
            processing_mode="sometimes_local",
            manifest_snapshot=_valid_manifest(),
            status="queued",
            stage="preflight",
        )


@pytest.mark.parametrize("digest", (b"0" * 64, "f" * 64))
def test_local_job_rejects_invalid_supplied_manifest_digest(digest: object) -> None:
    with pytest.raises(LocalValidationError, match="digest"):
        LocalAIJob(
            user_id=uuid.uuid4(),
            upload_id=uuid.uuid4(),
            kind="ingestion",
            processing_mode="validated_strict_local",
            manifest_snapshot=_valid_manifest(),
            manifest_sha256=digest,
            status="queued",
            stage="preflight",
        )


def test_local_job_canonicalizes_copies_and_hashes_manifest() -> None:
    source = _valid_manifest()
    job = LocalAIJob(
        user_id=uuid.uuid4(),
        upload_id=uuid.uuid4(),
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=source,
        status="queued",
        stage="preflight",
    )
    source["artifacts"][0]["revision"] = "f" * 40

    canonical_json = json.dumps(
        job.manifest_snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    assert job.manifest_snapshot["artifacts"][0]["revision"] == "0" * 40
    assert job.manifest_sha256 == hashlib.sha256(canonical_json).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", job.manifest_sha256)


async def _persist_local_job(db_session) -> LocalAIJob:
    user = User(email=f"immutable-job-{uuid.uuid4()}@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_valid_manifest(),
        status="queued",
        stage="preflight",
    )
    db_session.add(job)
    await db_session.commit()
    return job


@pytest.mark.parametrize(
    "attribute",
    ("processing_mode", "manifest_snapshot", "manifest_sha256"),
)
async def test_persisted_local_job_identity_is_immutable_in_orm(
    db_session,
    attribute: str,
) -> None:
    job = await _persist_local_job(db_session)
    if attribute == "processing_mode":
        value: object = "custom_local"
    elif attribute == "manifest_snapshot":
        value = _valid_manifest()
        value["pack_revision"] = "apple-m4-16gb-v2"
    else:
        value = "f" * 64
    setattr(job, attribute, value)

    try:
        with pytest.raises(LocalValidationError, match="immutable"):
            await db_session.commit()
    finally:
        await db_session.rollback()


@pytest.mark.parametrize(
    "assignment",
    (
        "processing_mode = 'custom_local'",
        "manifest_snapshot = jsonb_set("
        "manifest_snapshot, '{pack_revision}', '\"tampered\"'::jsonb)",
        "manifest_sha256 = repeat('f', 64)",
    ),
)
async def test_database_rejects_raw_local_job_identity_mutation(
    db_session,
    assignment: str,
) -> None:
    job = await _persist_local_job(db_session)

    try:
        with pytest.raises(IntegrityError):
            await db_session.execute(
                text(f"UPDATE local_ai_jobs SET {assignment} WHERE id = :job_id"),
                {"job_id": job.id},
            )
    finally:
        await db_session.rollback()


def test_local_job_revalidates_manifest_and_digest_before_use() -> None:
    job = LocalAIJob(
        user_id=uuid.uuid4(),
        upload_id=uuid.uuid4(),
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_valid_manifest(),
        status="queued",
        stage="preflight",
    )
    assert job.revalidate_manifest_snapshot() == job.manifest_snapshot

    job.manifest_snapshot["pack_revision"] = "tampered"
    with pytest.raises(LocalValidationError, match="manifest"):
        job.revalidate_manifest_snapshot()


def test_page_evidence_and_typed_summary_payload_columns_use_encrypted_types() -> None:
    assert isinstance(LocalAIPage.__table__.c.ocr_result.type, EncryptedJSON)
    assert isinstance(ExtractionEvidence.__table__.c.excerpt.type, EncryptedText)
    assert isinstance(ExtractionEvidence.__table__.c.field_paths.type, EncryptedJSON)
    assert isinstance(ExtractionEvidence.__table__.c.source_metadata.type, EncryptedJSON)
    assert isinstance(AISummaryPrompt.__table__.c.typed_response.type, EncryptedJSON)


def test_local_ai_indexes_match_checkpoint_and_lookup_paths() -> None:
    assert {index.name for index in LocalAIJob.__table__.indexes} == {
        "ix_local_ai_jobs_user_status",
        "ix_local_ai_jobs_upload_id",
        "ix_local_ai_jobs_summary_prompt_id",
    }
    assert {index.name for index in LocalAIPage.__table__.indexes} == {
        "ix_local_ai_pages_job_page"
    }
    assert {index.name for index in ExtractionEvidence.__table__.indexes} == {
        "ix_extraction_evidence_health_record_id"
    }


def test_processing_mode_defaults_match_database_defaults() -> None:
    upload_mode = UploadedFile.__table__.c.processing_mode
    summary_mode = AISummaryPrompt.__table__.c.processing_mode

    assert upload_mode.default.arg == "cloud_assisted"
    assert upload_mode.server_default.arg == "cloud_assisted"
    assert summary_mode.default.arg == "cloud_assisted"
    assert summary_mode.server_default.arg == "cloud_assisted"
