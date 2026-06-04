from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class BuildPromptRequest(BaseModel):
    patient_id: UUID
    summary_type: str = "full"
    category: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    record_ids: list[UUID] | None = None
    record_types: list[str] | None = None


class PromptResponse(BaseModel):
    id: UUID
    summary_type: str
    system_prompt: str
    user_prompt: str
    target_model: str
    suggested_config: dict
    record_count: int
    de_identification_report: dict | None
    copyable_payload: str
    generated_at: datetime


class PasteResponseRequest(BaseModel):
    prompt_id: UUID
    response_text: str


class GenerateSummaryRequest(BaseModel):
    patient_id: UUID
    summary_type: str = "full"
    category: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    output_format: str = "natural_language"  # "natural_language", "json", "both"
    custom_system_prompt: str | None = None
    custom_user_prompt: str | None = None


class DuplicateWarning(BaseModel):
    total_records: int
    deduped_records: int
    duplicates_excluded: int
    message: str | None = None


class GenerateSummaryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: UUID
    natural_language: str | None = None
    json_data: dict | None = None
    record_count: int
    duplicate_warning: DuplicateWarning | None = None
    de_identification_report: dict | None = None
    model_used: str
    generated_at: datetime


class SummaryItemCreate(BaseModel):
    record_id: UUID


class SummaryItemResponse(BaseModel):
    id: UUID
    record_id: UUID
    created_at: datetime
