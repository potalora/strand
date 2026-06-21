from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.middleware.encryption import blind_index
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedText


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    # ``email`` is encrypted at rest (AES-256-GCM via EncryptedText), so it is
    # NOT directly queryable. ``email_hmac`` is a deterministic blind index
    # (HMAC-SHA256 of the normalized email) that carries the uniqueness
    # constraint and backs login / get-by-email lookups. The two are kept in
    # sync by the before_insert/before_update listener below.
    email: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    email_hmac: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failed_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patients: Mapped[list[Patient]] = relationship("Patient", back_populates="user")


def _sync_email_hmac(mapper, connection, target: User) -> None:
    """Keep ``email_hmac`` consistent with ``email`` on insert/update.

    Derives the blind index from the plaintext email so every write path —
    registration, fixtures, scripts — gets a correct, lookup-able index without
    having to set it manually.
    """
    if target.email is not None:
        target.email_hmac = blind_index(target.email)


event.listen(User, "before_insert", _sync_email_hmac)
event.listen(User, "before_update", _sync_email_hmac)


from app.models.patient import Patient  # noqa: E402
