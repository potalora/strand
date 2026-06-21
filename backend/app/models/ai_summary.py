from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedText


class AISummaryPrompt(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "ai_summary_prompts"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False
    )
    summary_type: Mapped[str] = mapped_column(Text, nullable=False)
    scope_filter: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Encrypted at rest (AES-256-GCM). Prompts + the model response can embed
    # clinical context; stored ciphertext, fetch-and-render only.
    system_prompt: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    user_prompt: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    target_model: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="gemini-3-flash-preview"
    )
    suggested_config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    de_identification_log: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    response_text: Mapped[str | None] = mapped_column(EncryptedText, nullable=True)
    response_pasted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    response_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_format: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_model_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_tokens_used: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        nullable=False,
    )
