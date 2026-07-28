from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.local_ai.types import ProcessingMode


class LocalModelInfo(BaseModel):
    role: Literal["ocr", "extraction"]
    repository: str
    revision: str


class LocalRunInfo(BaseModel):
    privacy_mode: Literal["validated_strict_local"]
    models: list[LocalModelInfo] = Field(default_factory=list, max_length=2)


class LocalProcessingFailure(BaseModel):
    stage: str
    code: str
    message: str
    model_role: str | None = None
    repository: str | None = None
    revision: str | None = None
    retryable: bool
    checkpoint_preserved: bool
    cloud_fallback_attempted: Literal[False] = False


class UploadResponse(BaseModel):
    upload_id: str
    status: str
    records_inserted: int
    errors: list[Any] = []
    unstructured_uploads: list[dict] = []


class UploadStatusResponse(BaseModel):
    upload_id: str
    filename: str
    ingestion_status: str
    record_count: int
    total_file_count: int = 1
    ingestion_progress: dict = {}
    ingestion_errors: list[Any] = []
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    # Section-level extraction progress (unstructured pipeline).
    progress_stage: str | None = None
    progress_detail: dict | None = None
    # Durable per-file notices (e.g. an OCR provider refusal + fallback).
    notices: list[Any] = []
    local_run: LocalRunInfo | None = None
    local_failure: LocalProcessingFailure | None = None


class UploadHistoryItem(BaseModel):
    id: str
    filename: str
    ingestion_status: str
    record_count: int
    file_size_bytes: int | None = None
    created_at: str | None = None
    ingestion_progress: dict = {}
    ingestion_errors: list[Any] = []
    local_run: LocalRunInfo | None = None
    local_failure: LocalProcessingFailure | None = None


class UploadHistoryResponse(BaseModel):
    items: list[UploadHistoryItem]
    total: int


class UnstructuredUploadResponse(BaseModel):
    upload_id: str
    status: str
    file_type: str


class ExtractedEntitySchema(BaseModel):
    entity_class: str
    text: str
    attributes: dict = {}
    start_pos: int | None = None
    end_pos: int | None = None
    confidence: float = 0.8


class ExtractionResultResponse(BaseModel):
    upload_id: str
    status: str
    extracted_text_preview: str | None = None
    entities: list[ExtractedEntitySchema] = []
    error: str | None = None


class BatchUploadResponse(BaseModel):
    uploads: list[UnstructuredUploadResponse]
    total: int


class ReprocessUploadRequest(BaseModel):
    processing_mode: ProcessingMode | None = None


class ConfirmExtractionRequest(BaseModel):
    confirmed_entities: list[ExtractedEntitySchema]
    patient_id: str


class TriggerExtractionRequest(BaseModel):
    upload_ids: list[str]


class PendingExtractionFile(BaseModel):
    id: str
    filename: str
    mime_type: str
    file_category: str
    file_size_bytes: int | None = None
    created_at: str | None = None
    ingestion_status: str | None = None
    # Section-level extraction progress (unstructured pipeline).
    progress_stage: str | None = None
    progress_detail: dict | None = None
    # Durable per-file notices (e.g. an OCR provider refusal + fallback).
    notices: list[Any] = []
    local_run: LocalRunInfo | None = None
    local_failure: LocalProcessingFailure | None = None


class CancelExtractionRequest(BaseModel):
    upload_ids: list[str]


class CancelExtractionResponse(BaseModel):
    cancelled: list[str]
    skipped: list[str]


class TriggerExtractionResult(BaseModel):
    upload_id: str
    status: str


class TriggerExtractionResponse(BaseModel):
    triggered: int
    failed: int
    results: list[TriggerExtractionResult]


class PendingExtractionResponse(BaseModel):
    files: list[PendingExtractionFile]
    total: int


class ExtractionProgressResponse(BaseModel):
    total: int
    completed: int
    processing: int
    failed: int
    pending: int
    records_created: int
