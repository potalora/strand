from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedJSON, EncryptedText


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
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=True,
    )
    summary_prompt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_summary_prompts.id", ondelete="CASCADE"),
        nullable=True,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    processing_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    # Model identities, fixed revisions, hashes, and runtime metadata only.
    manifest_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
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

    __table_args__ = (
        CheckConstraint(
            "(upload_id IS NOT NULL AND summary_prompt_id IS NULL) "
            "OR (upload_id IS NULL AND summary_prompt_id IS NOT NULL)",
            name="ck_local_ai_jobs_exactly_one_target",
        ),
        Index("ix_local_ai_jobs_user_status", "user_id", "status"),
        Index("ix_local_ai_jobs_upload_id", "upload_id"),
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
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
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
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    health_record_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("health_records.id", ondelete="SET NULL"),
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
        Index("ix_extraction_evidence_health_record_id", "health_record_id"),
    )
