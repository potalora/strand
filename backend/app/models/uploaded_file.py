from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    DDL,
    event,
    ForeignKey,
    inspect,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedJSON, EncryptedText
from app.services.local_ai.errors import LocalValidationError

_IMMUTABLE_PROCESSING_SNAPSHOT_FIELDS = (
    "processing_mode",
    "processing_manifest",
    "processing_schema_version",
)


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
    total_file_count: Mapped[int] = mapped_column(
        Integer, default=1, server_default="1"
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    file_category: Mapped[str] = mapped_column(
        Text, default="structured", server_default="structured"
    )
    # Mixed-ZIP children are staged for explicit user confirmation. The
    # extraction worker excludes them until trigger-extraction clears this
    # durable flag in the same locked transaction.
    manual_extraction_required: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    processing_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="prompt_only",
        server_default="prompt_only",
    )
    # Immutable model identities, fixed revisions, hashes, and schema metadata.
    processing_manifest: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    processing_schema_version: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    # Encrypted at rest (AES-256-GCM). These hold raw extracted document text
    # and the entities/sections/metadata derived from it — all clinical PHI,
    # fetch-and-render only, never SQL-queried.
    extracted_text: Mapped[str | None] = mapped_column(EncryptedText, nullable=True)
    extraction_entities: Mapped[list | None] = mapped_column(
        EncryptedJSON, nullable=True
    )
    extraction_sections: Mapped[dict | None] = mapped_column(
        EncryptedJSON, nullable=True
    )
    document_metadata: Mapped[dict | None] = mapped_column(EncryptedJSON, nullable=True)
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

    __table_args__ = (
        UniqueConstraint(
            "id",
            "user_id",
            name="uq_uploaded_files_id_user_id",
        ),
    )


def _reject_persisted_processing_snapshot_changes(
    _mapper: Any,
    _connection: Any,
    target: UploadedFile,
) -> None:
    """Keep the privacy mode and processing revision fixed after enqueue."""

    state = inspect(target)
    if any(
        state.attrs[field].history.has_changes()
        for field in _IMMUTABLE_PROCESSING_SNAPSHOT_FIELDS
    ):
        raise LocalValidationError("Upload processing snapshot is immutable")


event.listen(
    UploadedFile,
    "before_update",
    _reject_persisted_processing_snapshot_changes,
)
event.listen(
    UploadedFile.__table__,
    "after_create",
    DDL(
        """
        CREATE OR REPLACE FUNCTION enforce_uploaded_file_processing_snapshot()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.processing_mode NOT IN (
                'prompt_only',
                'cloud_assisted',
                'validated_strict_local'
            ) THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'upload processing mode is invalid';
            END IF;
            IF TG_OP = 'UPDATE'
               AND (
                    NEW.processing_mode IS DISTINCT FROM OLD.processing_mode
                    OR NEW.processing_manifest IS DISTINCT FROM OLD.processing_manifest
                    OR NEW.processing_schema_version
                       IS DISTINCT FROM OLD.processing_schema_version
               ) THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'upload processing snapshot is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    ).execute_if(dialect="postgresql"),
)
event.listen(
    UploadedFile.__table__,
    "after_create",
    DDL(
        """
        CREATE TRIGGER trg_uploaded_files_immutable_processing_snapshot
        BEFORE INSERT OR UPDATE
        ON uploaded_files
        FOR EACH ROW
        EXECUTE FUNCTION enforce_uploaded_file_processing_snapshot()
        """
    ).execute_if(dialect="postgresql"),
)
