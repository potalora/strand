from __future__ import annotations

import hashlib
import json
import re
import uuid
from copy import deepcopy

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
        "artifacts": [_artifact(role) for role in ("ocr", "extraction", "summary")],
    }


def _manifest_digest(manifest: dict) -> str:
    canonical_json = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical_json).hexdigest()


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


async def test_local_job_locks_mode_and_manifest_when_preferences_change(
    db_session,
) -> None:
    user = User(login_identifier="local-job-models@example.com", password_hash="x")
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
async def test_local_job_rejects_kind_and_owner_mismatches(
    db_session, case: str
) -> None:
    user_a = User(login_identifier=f"{case}-a@example.com", password_hash="x")
    user_b = User(login_identifier=f"{case}-b@example.com", password_hash="x")
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
    user_a = User(login_identifier="evidence-upload-a@example.com", password_hash="x")
    user_b = User(login_identifier="evidence-upload-b@example.com", password_hash="x")
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
    user_a = User(
        login_identifier=f"evidence-record-{case}-a@example.com", password_hash="x"
    )
    user_b = User(
        login_identifier=f"evidence-record-{case}-b@example.com", password_hash="x"
    )
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
    user = User(
        login_identifier="evidence-delete-record@example.com", password_hash="x"
    )
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
    user = User(
        login_identifier=f"immutable-job-{uuid.uuid4()}@example.com", password_hash="x"
    )
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
    (
        "user_id",
        "kind",
        "upload_id",
        "summary_prompt_id",
        "processing_mode",
        "manifest_snapshot",
        "manifest_sha256",
    ),
)
async def test_persisted_local_job_identity_is_immutable_in_orm(
    db_session,
    attribute: str,
) -> None:
    job = await _persist_local_job(db_session)
    if attribute in {"user_id", "upload_id", "summary_prompt_id"}:
        value: object = uuid.uuid4()
    elif attribute == "kind":
        value = "summary"
    elif attribute == "processing_mode":
        value: object = "custom_local"
    elif attribute == "manifest_snapshot":
        value = _valid_manifest()
        value["pack_revision"] = "apple-m4-16gb-v2-drift"
    else:
        value = "f" * 64
    setattr(job, attribute, value)

    try:
        with pytest.raises(LocalValidationError, match="immutable"):
            await db_session.commit()
    finally:
        await db_session.rollback()


@pytest.mark.parametrize(
    "case",
    (
        "atomic_owner_and_upload",
        "kind_and_target",
        "upload_target",
        "summary_target",
    ),
)
async def test_database_rejects_raw_local_job_target_reassignment(
    db_session,
    case: str,
) -> None:
    user_a = User(login_identifier=f"job-move-{case}-a@example.com", password_hash="x")
    user_b = User(login_identifier=f"job-move-{case}-b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    await db_session.flush()
    upload_a = _upload_for(user_a)
    upload_a_other = _upload_for(user_a)
    upload_a_other.filename = "strict-local-note-other.pdf"
    upload_a_other.file_hash = "b" * 64
    upload_a_other.storage_path = "/private/strict-local-note-other.pdf"
    upload_b = _upload_for(user_b)
    patient_a = _patient_for(user_a, f"job-move-{case}-a")
    db_session.add_all([upload_a, upload_a_other, upload_b, patient_a])
    await db_session.flush()
    summary_a = _summary_for(user_a, patient_a)
    summary_a_other = _summary_for(user_a, patient_a)
    db_session.add_all([summary_a, summary_a_other])
    await db_session.flush()

    is_summary = case == "summary_target"
    job = LocalAIJob(
        user_id=user_a.id,
        upload_id=None if is_summary else upload_a.id,
        summary_prompt_id=summary_a.id if is_summary else None,
        kind="summary" if is_summary else "ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot=_valid_manifest(),
        status="queued",
        stage="preflight",
    )
    db_session.add(job)
    await db_session.commit()

    statement, parameters = {
        "atomic_owner_and_upload": (
            "UPDATE local_ai_jobs SET user_id = :user_id, upload_id = :upload_id "
            "WHERE id = :job_id",
            {"user_id": user_b.id, "upload_id": upload_b.id},
        ),
        "kind_and_target": (
            "UPDATE local_ai_jobs SET kind = 'summary', upload_id = NULL, "
            "summary_prompt_id = :summary_prompt_id WHERE id = :job_id",
            {"summary_prompt_id": summary_a.id},
        ),
        "upload_target": (
            "UPDATE local_ai_jobs SET upload_id = :upload_id WHERE id = :job_id",
            {"upload_id": upload_a_other.id},
        ),
        "summary_target": (
            "UPDATE local_ai_jobs SET summary_prompt_id = :summary_prompt_id "
            "WHERE id = :job_id",
            {"summary_prompt_id": summary_a_other.id},
        ),
    }[case]
    parameters["job_id"] = job.id

    try:
        with pytest.raises(IntegrityError):
            await db_session.execute(text(statement), parameters)
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


async def test_persisted_evidence_scope_is_immutable_in_orm(db_session) -> None:
    user = User(
        login_identifier="evidence-immutable-orm@example.com", password_hash="x"
    )
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    evidence = ExtractionEvidence(
        user_id=user.id,
        upload_id=upload.id,
        excerpt="synthetic excerpt",
        field_paths=["$.conditions[0]"],
        source_metadata={"source_kind": "ocr_page"},
    )
    db_session.add(evidence)
    await db_session.commit()

    evidence.user_id = uuid.uuid4()
    evidence.upload_id = uuid.uuid4()
    try:
        with pytest.raises(LocalValidationError, match="immutable"):
            await db_session.commit()
    finally:
        await db_session.rollback()


async def test_database_rejects_raw_evidence_scope_reassignment(db_session) -> None:
    user_a = User(login_identifier="evidence-move-a@example.com", password_hash="x")
    user_b = User(login_identifier="evidence-move-b@example.com", password_hash="x")
    db_session.add_all([user_a, user_b])
    await db_session.flush()
    upload_a = _upload_for(user_a)
    upload_b = _upload_for(user_b)
    db_session.add_all([upload_a, upload_b])
    await db_session.flush()
    evidence = ExtractionEvidence(
        user_id=user_a.id,
        upload_id=upload_a.id,
        excerpt="synthetic excerpt",
        field_paths=["$.conditions[0]"],
        source_metadata={"source_kind": "ocr_page"},
    )
    db_session.add(evidence)
    await db_session.commit()

    try:
        with pytest.raises(IntegrityError):
            await db_session.execute(
                text(
                    "UPDATE extraction_evidence "
                    "SET user_id = :user_id, upload_id = :upload_id "
                    "WHERE id = :evidence_id"
                ),
                {
                    "user_id": user_b.id,
                    "upload_id": upload_b.id,
                    "evidence_id": evidence.id,
                },
            )
    finally:
        await db_session.rollback()


async def _raw_insert_local_job(
    db_session,
    *,
    user_id: uuid.UUID,
    upload_id: uuid.UUID,
    processing_mode: str,
    manifest: dict | str,
    digest: str,
) -> uuid.UUID:
    manifest_json = json.dumps(manifest) if isinstance(manifest, dict) else manifest
    result = await db_session.execute(
        text(
            """
            INSERT INTO local_ai_jobs (
                user_id,
                upload_id,
                kind,
                processing_mode,
                manifest_snapshot,
                manifest_sha256,
                status,
                stage
            )
            VALUES (
                :user_id,
                :upload_id,
                'ingestion',
                :processing_mode,
                CAST(:manifest AS jsonb),
                :digest,
                'queued',
                'preflight'
            )
            RETURNING id
            """
        ),
        {
            "user_id": user_id,
            "upload_id": upload_id,
            "processing_mode": processing_mode,
            "manifest": manifest_json,
            "digest": digest,
        },
    )
    return result.scalar_one()


async def _database_manifest_digest(db_session, manifest_json: str) -> str:
    return (
        await db_session.execute(
            text(
                """
                SELECT encode(
                    sha256(
                        convert_to(
                            local_ai_canonical_json(CAST(:manifest AS jsonb)),
                            'UTF8'
                        )
                    ),
                    'hex'
                )
                """
            ),
            {"manifest": manifest_json},
        )
    ).scalar_one()


async def _database_manifest_is_valid(db_session, manifest: dict) -> bool:
    return bool(
        (
            await db_session.execute(
                text("SELECT local_ai_manifest_is_valid(CAST(:payload AS jsonb))"),
                {"payload": json.dumps(manifest)},
            )
        ).scalar_one()
    )


async def test_fresh_database_accepts_only_exact_attested_v2_runtime(
    db_session,
) -> None:
    valid = _valid_manifest()
    legacy = deepcopy(valid)
    legacy["schema_version"] = 1
    legacy["pack_revision"] = "apple-m4-16gb-v1"
    legacy["runtime"] = {"name": "mlx-vlm", "version": "0.5.0"}
    missing_digest = deepcopy(valid)
    del missing_digest["runtime"]["worker_bundle_sha256"]
    uppercase_digest = deepcopy(valid)
    uppercase_digest["runtime"]["worker_bundle_sha256"] = "D" * 64
    extra_runtime = deepcopy(valid)
    extra_runtime["runtime"]["extra"] = "rejected"

    assert await _database_manifest_is_valid(db_session, valid) is True
    for manifest in (legacy, missing_digest, uppercase_digest, extra_runtime):
        assert await _database_manifest_is_valid(db_session, manifest) is False


async def test_fresh_database_rejects_legacy_raw_job_insert(db_session) -> None:
    user = User(login_identifier="fresh-v1-rejected@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    legacy = _valid_manifest()
    legacy["schema_version"] = 1
    legacy["pack_revision"] = "apple-m4-16gb-v1"
    legacy["runtime"] = {"name": "mlx-vlm", "version": "0.5.0"}

    try:
        with pytest.raises(IntegrityError):
            await _raw_insert_local_job(
                db_session,
                user_id=user.id,
                upload_id=upload.id,
                processing_mode="validated_strict_local",
                manifest=legacy,
                digest=_manifest_digest(legacy),
            )
    finally:
        await db_session.rollback()


@pytest.mark.parametrize("case", ("empty_manifest", "bad_digest", "invalid_mode"))
async def test_database_rejects_invalid_raw_local_job_insert(
    db_session,
    case: str,
) -> None:
    user = User(login_identifier=f"raw-job-{case}@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    manifest = {} if case == "empty_manifest" else _valid_manifest()
    digest = "f" * 64 if case == "bad_digest" else _manifest_digest(manifest)
    mode = "sometimes_local" if case == "invalid_mode" else "validated_strict_local"

    try:
        with pytest.raises(IntegrityError):
            await _raw_insert_local_job(
                db_session,
                user_id=user.id,
                upload_id=upload.id,
                processing_mode=mode,
                manifest=manifest,
                digest=digest,
            )
    finally:
        await db_session.rollback()


async def test_database_rejects_second_raw_ingestion_job_for_one_upload(
    db_session,
) -> None:
    """A raw SQL insert cannot create a competing strict-local ingestion job."""
    user = User(
        login_identifier="duplicate-raw-local-job@example.com", password_hash="x"
    )
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    manifest = _valid_manifest()
    digest = _manifest_digest(manifest)
    await _raw_insert_local_job(
        db_session,
        user_id=user.id,
        upload_id=upload.id,
        processing_mode="validated_strict_local",
        manifest=manifest,
        digest=digest,
    )
    await db_session.commit()

    try:
        with pytest.raises(IntegrityError):
            await _raw_insert_local_job(
                db_session,
                user_id=user.id,
                upload_id=upload.id,
                processing_mode="validated_strict_local",
                manifest=manifest,
                digest=digest,
            )
    finally:
        await db_session.rollback()


@pytest.mark.parametrize(
    ("field", "numeric_value"),
    (
        ("schema_version", 1.0),
        ("decode_limit", 4096.0),
        ("file_size", 10.0),
    ),
)
async def test_database_rejects_float_manifest_integer_fields(
    db_session,
    field: str,
    numeric_value: float,
) -> None:
    user = User(
        login_identifier=f"raw-job-float-{field}@example.com", password_hash="x"
    )
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    manifest = _valid_manifest()
    if field == "schema_version":
        manifest["schema_version"] = numeric_value
    elif field == "decode_limit":
        manifest["artifacts"][0]["decode_limits"]["max_input_tokens"] = numeric_value
    else:
        manifest["artifacts"][0]["files"][0]["size"] = numeric_value

    try:
        with pytest.raises(IntegrityError):
            await _raw_insert_local_job(
                db_session,
                user_id=user.id,
                upload_id=upload.id,
                processing_mode="validated_strict_local",
                manifest=manifest,
                digest=_manifest_digest(manifest),
            )
    finally:
        await db_session.rollback()


@pytest.mark.parametrize(
    ("field", "old_token", "exponent_token"),
    (
        ("schema_version", '"schema_version":2', '"schema_version":2.00e0'),
        ("decode_limit", '"max_input_tokens":4096', '"max_input_tokens":4.0960e3'),
        ("file_size", '"size":10', '"size":1.00e1'),
    ),
)
async def test_database_rejects_noncanonical_exponent_manifest_integer_fields(
    db_session,
    field: str,
    old_token: str,
    exponent_token: str,
) -> None:
    user = User(
        login_identifier=f"raw-job-exponent-{field}@example.com", password_hash="x"
    )
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    manifest_json = json.dumps(
        _valid_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).replace(old_token, exponent_token, 1)
    digest = await _database_manifest_digest(db_session, manifest_json)

    try:
        with pytest.raises(IntegrityError):
            await _raw_insert_local_job(
                db_session,
                user_id=user.id,
                upload_id=upload.id,
                processing_mode="validated_strict_local",
                manifest=manifest_json,
                digest=digest,
            )
    finally:
        await db_session.rollback()


async def test_database_accepts_canonical_raw_local_job_insert(db_session) -> None:
    user = User(login_identifier="raw-job-canonical@example.com", password_hash="x")
    db_session.add(user)
    await db_session.flush()
    upload = _upload_for(user)
    db_session.add(upload)
    await db_session.flush()
    manifest = _valid_manifest()
    manifest["artifacts"][0]["attribution"] = "Cliníc model 😀"
    digest = _manifest_digest(manifest)

    job_id = await _raw_insert_local_job(
        db_session,
        user_id=user.id,
        upload_id=upload.id,
        processing_mode="validated_strict_local",
        manifest=manifest,
        digest=digest,
    )
    await db_session.commit()

    stored = (
        await db_session.execute(
            text(
                "SELECT manifest_snapshot, manifest_sha256 "
                "FROM local_ai_jobs WHERE id = :job_id"
            ),
            {"job_id": job_id},
        )
    ).one()
    assert stored.manifest_snapshot == manifest
    assert stored.manifest_sha256 == digest


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
    assert isinstance(
        ExtractionEvidence.__table__.c.source_metadata.type, EncryptedJSON
    )
    assert isinstance(AISummaryPrompt.__table__.c.typed_response.type, EncryptedJSON)


def test_local_ai_indexes_match_checkpoint_and_lookup_paths() -> None:
    assert {index.name for index in LocalAIJob.__table__.indexes} == {
        "ix_local_ai_jobs_user_status",
        "ix_local_ai_jobs_upload_id",
        "ix_local_ai_jobs_summary_prompt_id",
        "uq_local_ai_jobs_ingestion_upload",
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

    assert upload_mode.default.arg == "prompt_only"
    assert upload_mode.server_default.arg == "prompt_only"
    assert summary_mode.default.arg == "prompt_only"
    assert summary_mode.server_default.arg == "prompt_only"
