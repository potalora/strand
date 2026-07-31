from __future__ import annotations

import hmac
import re
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, validates

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedJSON, EncryptedText
from app.models.local_ai_ddl import (
    EXTRACTION_EVIDENCE_DATABASE_GUARDS,
    LOCAL_AI_JOB_DATABASE_GUARDS,
)
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from app.services.local_ai.types import ProcessingMode

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_JOB_FIELDS = (
    "user_id",
    "kind",
    "upload_id",
    "summary_prompt_id",
    "processing_mode",
    "manifest_snapshot",
    "manifest_sha256",
)
_IMMUTABLE_EVIDENCE_FIELDS = ("user_id", "upload_id")


class LocalAIJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Immutable processing-mode and model-pack snapshot for one local job."""

    __tablename__ = "local_ai_jobs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    upload_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    summary_prompt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    processing_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    # Model identities, fixed revisions, hashes, and runtime metadata only.
    manifest_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    # Stable stage codes and non-content counters only.
    progress: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    failure: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    audit_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __init__(self, **kwargs: Any) -> None:
        raw_manifest = kwargs.pop("manifest_snapshot", None)
        supplied_digest = kwargs.pop("manifest_sha256", None)
        if raw_manifest is None:
            raise LocalValidationError("Local AI job manifest is invalid")
        super().__init__(**kwargs)
        self.manifest_snapshot = raw_manifest
        if supplied_digest is not None:
            if not isinstance(supplied_digest, str) or not hmac.compare_digest(
                supplied_digest,
                self.manifest_sha256,
            ):
                raise LocalValidationError("Local AI job manifest digest is invalid")

    @validates("processing_mode")
    def _validate_processing_mode(self, _key: str, value: Any) -> str:
        try:
            return ProcessingMode(value).value
        except (TypeError, ValueError) as exc:
            raise LocalValidationError(
                "Local AI job processing mode is invalid"
            ) from exc

    @validates("manifest_snapshot")
    def _validate_manifest_snapshot(self, _key: str, value: Any) -> dict:
        try:
            snapshot, digest = canonicalize_manifest_snapshot(value)
        except LocalValidationError as exc:
            raise LocalValidationError("Local AI job manifest is invalid") from exc
        self.manifest_sha256 = digest
        return snapshot

    @validates("manifest_sha256")
    def _validate_manifest_sha256(self, _key: str, value: Any) -> str:
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise LocalValidationError("Local AI job manifest digest is invalid")
        return value

    def revalidate_manifest_snapshot(self) -> dict:
        """Revalidate and detach the stored snapshot before worker use."""

        try:
            snapshot, digest = canonicalize_manifest_snapshot(self.manifest_snapshot)
        except LocalValidationError as exc:
            raise LocalValidationError(
                "Stored local AI job manifest is invalid"
            ) from exc
        if (
            snapshot != self.manifest_snapshot
            or not isinstance(self.manifest_sha256, str)
            or not hmac.compare_digest(digest, self.manifest_sha256)
        ):
            raise LocalValidationError("Stored local AI job manifest is invalid")
        return snapshot

    __table_args__ = (
        CheckConstraint(
            "(kind = 'ingestion' AND upload_id IS NOT NULL "
            "AND summary_prompt_id IS NULL) "
            "OR (kind = 'summary' AND upload_id IS NULL "
            "AND summary_prompt_id IS NOT NULL)",
            name="ck_local_ai_jobs_kind_target",
        ),
        ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_local_ai_jobs_upload_owner",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["summary_prompt_id", "user_id"],
            ["ai_summary_prompts.id", "ai_summary_prompts.user_id"],
            name="fk_local_ai_jobs_summary_owner",
            ondelete="CASCADE",
        ),
        Index("ix_local_ai_jobs_user_status", "user_id", "status"),
        Index("ix_local_ai_jobs_upload_id", "upload_id"),
        Index(
            "uq_local_ai_jobs_ingestion_upload",
            "upload_id",
            unique=True,
            postgresql_where=text("kind = 'ingestion' AND upload_id IS NOT NULL"),
        ),
        Index("ix_local_ai_jobs_summary_prompt_id", "summary_prompt_id"),
    )


class LocalAIPage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Encrypted OCR checkpoint for one page of an upload."""

    __tablename__ = "local_ai_pages"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("local_ai_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_key: Mapped[str] = mapped_column(String(64), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ocr_result: Mapped[dict] = mapped_column(EncryptedJSON, nullable=False)
    # Stable warning codes only; OCR content belongs in ``ocr_result``.
    warnings: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "page_number",
            name="uq_local_ai_pages_job_page",
        ),
        Index("ix_local_ai_pages_job_page", "job_id", "page_number"),
    )


class LocalAIExtractionCheckpoint(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Encrypted, resumable clinical extraction for one immutable local job."""

    __tablename__ = "local_ai_extraction_checkpoints"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("local_ai_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    checkpoint_key: Mapped[str] = mapped_column(String(64), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ocr_text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_bindings_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_result_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_extraction_result: Mapped[Any | None] = mapped_column(
        EncryptedJSON,
        nullable=True,
    )
    extraction_result: Mapped[dict | None] = mapped_column(
        EncryptedJSON,
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "job_id",
            name="uq_local_ai_extraction_checkpoints_job",
        ),
        Index(
            "ix_local_ai_extraction_checkpoints_job",
            "job_id",
        ),
    )


class ExtractionEvidence(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Encrypted source evidence that grounds an extracted health record."""

    __tablename__ = "extraction_evidence"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    health_record_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "health_records.id",
            name="fk_extraction_evidence_health_record",
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    excerpt: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    field_paths: Mapped[list] = mapped_column(EncryptedJSON, nullable=False)
    source_metadata: Mapped[dict] = mapped_column(EncryptedJSON, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_extraction_evidence_upload_owner",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["health_record_id", "upload_id", "user_id"],
            [
                "health_records.id",
                "health_records.source_file_id",
                "health_records.user_id",
            ],
            name="fk_extraction_evidence_record_lineage",
            deferrable=True,
            initially="DEFERRED",
        ),
        Index("ix_extraction_evidence_health_record_id", "health_record_id"),
    )


def _validate_new_job_identity(
    _mapper: Any,
    _connection: Any,
    target: LocalAIJob,
) -> None:
    target.revalidate_manifest_snapshot()


def _reject_persisted_job_identity_changes(
    _mapper: Any,
    _connection: Any,
    target: LocalAIJob,
) -> None:
    state = inspect(target)
    if any(state.attrs[field].history.has_changes() for field in _IMMUTABLE_JOB_FIELDS):
        raise LocalValidationError("Local AI job identity is immutable")
    target.revalidate_manifest_snapshot()


def _reject_persisted_evidence_scope_changes(
    _mapper: Any,
    _connection: Any,
    target: ExtractionEvidence,
) -> None:
    state = inspect(target)
    if any(
        state.attrs[field].history.has_changes() for field in _IMMUTABLE_EVIDENCE_FIELDS
    ):
        raise LocalValidationError("Extraction evidence scope is immutable")


event.listen(LocalAIJob, "before_insert", _validate_new_job_identity)
event.listen(LocalAIJob, "before_update", _reject_persisted_job_identity_changes)
event.listen(
    ExtractionEvidence,
    "before_update",
    _reject_persisted_evidence_scope_changes,
)
for database_guard in LOCAL_AI_JOB_DATABASE_GUARDS:
    event.listen(LocalAIJob.__table__, "after_create", database_guard)
for database_guard in EXTRACTION_EVIDENCE_DATABASE_GUARDS:
    event.listen(ExtractionEvidence.__table__, "after_create", database_guard)
