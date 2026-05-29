from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class HealthRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "health_records"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    record_type: Mapped[str] = mapped_column(Text, nullable=False)
    fhir_resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    fhir_resource: Mapped[dict] = mapped_column(JSONB, nullable=False)
    source_format: Mapped[str] = mapped_column(Text, nullable=False)
    source_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("uploaded_files.id"), nullable=True
    )
    effective_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    effective_date_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    code_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    code_display: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("health_records.id"), nullable=True
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_extracted: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    source_section: Mapped[str | None] = mapped_column(Text, nullable=True)
    linked_encounter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("health_records.id"), nullable=True
    )
    merge_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_system: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    patient: Mapped[Patient] = relationship("Patient", back_populates="health_records")

    __table_args__ = (
        Index("idx_health_records_patient_date", "patient_id", effective_date.desc()),
        Index("idx_health_records_type", "record_type"),
        Index("idx_health_records_code", "code_system", "code_value"),
    )


from app.models.patient import Patient  # noqa: E402
