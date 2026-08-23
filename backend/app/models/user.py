from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.middleware.encryption import blind_index
from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedText


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    # The private login identifier is encrypted at rest (AES-256-GCM via
    # EncryptedText), so it is NOT directly queryable. The deterministic blind
    # index carries uniqueness and backs login lookups. The two values are kept
    # in sync by the before_insert/before_update listener below.
    login_identifier: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    login_identifier_hmac: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failed_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patients: Mapped[list[Patient]] = relationship("Patient", back_populates="user")


def _sync_login_identifier_hmac(mapper, connection, target: User) -> None:
    """Keep the identifier blind index consistent on insert/update.

    Derives the blind index from the plaintext identifier so every write path —
    registration, fixtures, scripts — gets a correct, lookup-able index without
    having to set it manually.
    """
    if target.login_identifier is not None:
        target.login_identifier_hmac = blind_index(target.login_identifier)


event.listen(User, "before_insert", _sync_login_identifier_hmac)
event.listen(User, "before_update", _sync_login_identifier_hmac)


from app.models.patient import Patient  # noqa: E402
