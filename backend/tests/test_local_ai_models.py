from __future__ import annotations

from sqlalchemy import CheckConstraint

from app.models.ai_summary import AISummaryPrompt
from app.models.encrypted_types import EncryptedJSON, EncryptedText
from app.models.llm_settings import UserLLMPreferences
from app.models.local_ai import ExtractionEvidence, LocalAIJob, LocalAIPage
from app.models.uploaded_file import UploadedFile
from app.models.user import User


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

    manifest_snapshot = {
        "pack_revision": "apple-m4-16gb-v1",
        "artifacts": [
            {
                "role": "ocr",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "sha256": "b" * 64,
            }
        ],
    }
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


def test_local_job_requires_exactly_one_target() -> None:
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in LocalAIJob.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert constraints["ck_local_ai_jobs_exactly_one_target"] == (
        "(upload_id IS NOT NULL AND summary_prompt_id IS NULL) "
        "OR (upload_id IS NULL AND summary_prompt_id IS NOT NULL)"
    )


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
