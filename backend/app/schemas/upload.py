from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


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


class UploadHistoryItem(BaseModel):
    id: str
    filename: str
    ingestion_status: str
    record_count: int
    file_size_bytes: int | None = None
    created_at: str | None = None
    ingestion_progress: dict = {}
    ingestion_errors: list[Any] = []


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
