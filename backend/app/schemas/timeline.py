from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class TimelineEvent(BaseModel):
    id: UUID
    record_type: str
    display_text: str
    effective_date: datetime | None
    code_display: str | None
    category: list[str] | None
    provider: str | None = None

    model_config = {"from_attributes": True}


class TimelineResponse(BaseModel):
    events: list[TimelineEvent]
    total: int


class TimelineStats(BaseModel):
    total_records: int
    records_by_type: dict[str, int]
    date_range_start: datetime | None
    date_range_end: datetime | None
