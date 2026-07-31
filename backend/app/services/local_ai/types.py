"""Serializable, worker-safe local AI contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ProcessingMode(StrEnum):
    """Explicit routing modes for document processing."""

    VALIDATED_STRICT_LOCAL = "validated_strict_local"
    CUSTOM_LOCAL = "custom_local"
    CLOUD_ASSISTED = "cloud_assisted"
    PROMPT_ONLY = "prompt_only"


class ModelRole(StrEnum):
    """The supported roles in a local AI model pack."""

    OCR = "ocr"
    EXTRACTION = "extraction"
    SUMMARY = "summary"


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable identity of the exact model used by a worker."""

    role: ModelRole
    repository: str
    revision: str
    quantization: str
    runtime: str


@dataclass(frozen=True)
class EvidenceReference:
    """A worker-serializable reference to source evidence."""

    id: str
    upload_id: str
    page_number: int | None
    section: str | None
    excerpt: str
    start_offset: int | None
    end_offset: int | None
    field_paths: list[str]


@dataclass(frozen=True)
class OCRPageRequest:
    """One rasterized page submitted to the embedded OCR worker."""

    job_id: str
    page_number: int
    image_path: str
    image_sha256: str
    model: ModelIdentity
    max_output_tokens: int


@dataclass
class OCRPageResult:
    """Serializable OCR result for one input page."""

    page_number: int
    markdown: str
    width: int
    height: int
    warnings: list[str]
    content_sha256: str
    model: ModelIdentity


@dataclass(frozen=True)
class ExtractionRequest:
    """Worker request for structured clinical entity extraction."""

    job_id: str
    upload_id: str
    page_markdown: list[dict[str, Any]]
    image_paths: dict[int, str]
    schema_version: str
    model: ModelIdentity


@dataclass
class ClinicalExtraction:
    """Serializable clinical extraction output with source evidence."""

    entities: list[dict[str, Any]]
    evidence: list[EvidenceReference]
    unresolved_fields: list[str] = field(default_factory=list)
    rejected_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryRequest:
    """Worker request for a grounded health-record summary."""

    job_id: str
    summary_type: str
    facts: list[dict[str, Any]]
    evidence: list[EvidenceReference]
    model: ModelIdentity


@dataclass
class GroundedSummary:
    """Serializable summary output produced from supplied evidence."""

    sections: list[dict[str, Any]]
    model: ModelIdentity


class OCRBackend(Protocol):
    """Async interface implemented by local OCR worker adapters."""

    async def parse_page(self, request: OCRPageRequest) -> OCRPageResult: ...


class ExtractionBackend(Protocol):
    """Async interface implemented by local extraction worker adapters."""

    async def extract(self, request: ExtractionRequest) -> ClinicalExtraction: ...


class SummaryBackend(Protocol):
    """Async interface implemented by local summary worker adapters."""

    async def summarize(self, request: SummaryRequest) -> GroundedSummary: ...
