from __future__ import annotations

import uuid

from sqlalchemy import Boolean, ForeignKey, LargeBinary, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LLMProviderConfig(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Per-user credentials/config for a single LLM provider.

    The API key is stored encrypted (AES-256-GCM via ``encrypt_field``) and
    is decrypted only server-side at call time. One row per (user, provider).
    """

    __tablename__ = "llm_provider_configs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    api_key_encrypted: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )

    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_llm_user_provider"),
    )


class UserLLMPreferences(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Per-user LLM routing preferences (one row per user).

    Each ``*_provider`` column names the provider to use for that operation,
    overriding the global ``.env`` routing. ``None`` falls back to the default.
    """

    __tablename__ = "user_llm_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    default_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    section_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dedup_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    extraction_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vision_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    extraction_engine: Mapped[str | None] = mapped_column(String(16), nullable=True)
