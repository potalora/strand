from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class SummaryItem(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A record the user has staged into their "Add to summary" basket.

    One row per (user, record). Used by the detail sheet's "Add to summary"
    action to build a curated set of records for an AI summary prompt.
    """

    __tablename__ = "summary_items"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    record_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("health_records.id"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("user_id", "record_id", name="uq_summary_items_user_record"),
    )
