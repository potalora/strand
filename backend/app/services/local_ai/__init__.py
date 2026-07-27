"""Platform-neutral contracts for local AI processing."""

from __future__ import annotations

from app.services.local_ai.types import (
    ClinicalExtraction,
    EvidenceReference,
    ExtractionBackend,
    ExtractionRequest,
    GroundedSummary,
    ModelIdentity,
    ModelRole,
    OCRBackend,
    OCRPageRequest,
    OCRPageResult,
    ProcessingMode,
    SummaryBackend,
    SummaryRequest,
)

__all__ = [
    "ClinicalExtraction",
    "EvidenceReference",
    "ExtractionBackend",
    "ExtractionRequest",
    "GroundedSummary",
    "ModelIdentity",
    "ModelRole",
    "OCRBackend",
    "OCRPageRequest",
    "OCRPageResult",
    "ProcessingMode",
    "SummaryBackend",
    "SummaryRequest",
]
