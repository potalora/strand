from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UploadedFile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "uploaded_files"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    ingestion_status: Mapped[str] = mapped_column(
        Text, default="pending", server_default="pending"
    )
    ingestion_progress: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    ingestion_errors: Mapped[list] = mapped_column(JSONB, server_default="[]")
    record_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_file_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    file_category: Mapped[str] = mapped_column(
        Text, default="structured", server_default="structured"
    )
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_entities: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    extraction_sections: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    document_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    dedup_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Cooperative cancel: the API sets this; the extraction worker checks it
    # between stages and aborts cleanly, marking the file ``cancelled``.
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    # Section-level progress for the unstructured extraction pipeline. The
    # worker writes the current stage (extracting_text / scrubbing_phi /
    # extracting_entities / mapping_fhir) and a {section_index, section_total}
    # detail so the frontend can show "section 3 of 8".
    progress_stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Durable per-file user-facing notices (e.g. an OCR provider refusal +
    # fallback, or a document no provider could read). Each entry:
    # {type, level, message, detail}. Surfaced in upload history + Extractions.
    notices: Mapped[list] = mapped_column(JSONB, server_default="[]")
