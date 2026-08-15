"""Fail-closed strict-local ingestion orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from striprtf.striprtf import rtf_to_text

from app.services.local_ai.adapters import (
    to_extracted_entities,
    validated_extraction_to_health_record_dicts,
)
from app.services.local_ai.checkpoint_store import (
    ExtractionCheckpoint,
    OCRCheckpoint,
    RawExtractionCheckpoint,
    extraction_checkpoint_identity,
    raw_extraction_result_sha256,
)
from app.services.local_ai.checkpoints import (
    extraction_checkpoint_key,
    ocr_checkpoint_key,
)
from app.services.local_ai.errors import LocalPolicyError, LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
    MAX_FACTS_PER_CATEGORY,
    MAX_FIELD_LIST_ITEMS,
    MAX_LONG_TEXT,
    MAX_SHORT_TEXT,
    NUEXTRACT_TEMPLATE_V1,
    ClinicalDocumentExtraction,
)
from app.services.local_ai.extraction_validator import (
    MAX_EXTRACTION_JSON_BYTES,
    clinical_fact_duplicate_signature,
    validate_clinical_extraction,
)
from app.services.local_ai.manifest import (
    LocalAIManifest,
    ManifestArtifact,
    canonicalize_manifest_snapshot,
)
from app.services.local_ai.rasterizer import RasterizedPage, iter_rasterized_pages
from app.services.local_ai.scratch import ScratchJob
from app.services.local_ai.types import ModelRole, ProcessingMode

_RASTER_VERSION = "pdfium-pillow-2x.v1"
_RTF_TEXT_VERSION = "striprtf-text.v1"
_EXTRACTION_PROMPT_VERSION = "nuextract3-clinical.v3"
_MAX_PAGES = 500
_MAX_OCR_MARKDOWN_BYTES = 4 * 1024 * 1024
_MAX_DOCUMENT_MARKDOWN_BYTES = 16 * 1024 * 1024
_MAX_RTF_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_EXTRACTION_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_SELECTED_IMAGES = 8
_MAX_SELECTED_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_SELECTED_IMAGE_PIXELS = 40_000_000
_CHUNKED_EXTRACTION_RESULT_TYPE = "chunked_clinical_extraction.v1"
_MAX_EXTRACTION_CHUNKS = 500
_EXTRACTION_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "patient",
        *FACT_CATEGORY_NAMES,
        "unresolved_fields",
        "rejected_fields",
    }
)
_REJECTION_OVERFLOW_PATH = "extraction.rejections:limit_exceeded"
_VALIDATION_WORK_OVERFLOW_PATH = "extraction.validation_work:limit_exceeded"
_MAX_RAW_ITEMS_PER_DOCUMENT = len(FACT_CATEGORY_NAMES) * MAX_FACTS_PER_CATEGORY + (
    2 * MAX_FIELD_LIST_ITEMS
)
_MAX_EVIDENCE_SOURCE_SEGMENTS_PER_PAGE = 4_096
_MAX_EVIDENCE_SOURCE_SEGMENT_WORK_PER_DOCUMENT = 16_384
_FACT_SUBJECT_FIELDS = {
    "medications": "name",
    "conditions": "name",
    "procedures": "name",
    "labs": "name",
    "allergies": "substance",
    "encounters": "name",
    "immunizations": "name",
    "vital_signs": "name",
    "diagnostic_reports": "name",
    "care_plans": "title",
}
_FACT_NUMERIC_FIELDS = {
    "medications": ("dose_value",),
    "labs": ("value",),
    "immunizations": ("dose",),
    "vital_signs": ("value",),
}
_HTML_TABLE_ROW_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_FHIR_VALIDATION_UUID = uuid.UUID(int=0)
_TABLE_SEPARATOR_RE = re.compile(r"(?m)^\s*\|?(?:\s*:?-{3,}:?\s*\|){2,}\s*$")
_IMAGE_ESCALATION_MARKERS = (
    "[illegible]",
    "[unreadable]",
    "<table",
    "<image",
    "handwritten",
)


class WorkerManager(Protocol):
    """The subset of the isolated manager used by ingestion."""

    async def run_attested(
        self,
        manifest: LocalAIManifest,
        role: ModelRole,
        payload: dict[str, Any],
        on_progress: Callable[[dict[str, object]], object] | None = None,
        *,
        on_liveness: Callable[[], object] | None = None,
    ) -> Any: ...


class LocalAICheckpointStore(Protocol):
    """Persistence boundary needed by the pipeline."""

    async def get_ocr_page(
        self,
        job_id: str | uuid.UUID,
        page_number: int,
        checkpoint_key: str,
        image_sha256: str,
    ) -> OCRCheckpoint | None: ...

    async def put_ocr_page(
        self,
        job_id: str | uuid.UUID,
        checkpoint: OCRCheckpoint,
    ) -> OCRCheckpoint: ...

    async def get_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint | None: ...

    async def put_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint: ...

    async def get_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint | None: ...

    async def put_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint: ...


@dataclass
class _ValidationWorkBudget:
    """Document-global bound for validation of model-authored clinical items."""

    remaining: int

    def claim(self, requested: int) -> int:
        allowed = min(max(self.remaining, 0), max(requested, 0))
        self.remaining -= allowed
        return allowed


@dataclass(frozen=True)
class _EvidenceSourceSegment:
    """One bounded, exact row or line from a claimed OCR page."""

    text: str
    start: int
    end: int
    priority: int


@dataclass(frozen=True)
class _EvidenceSourceScan:
    """Bounded source-index result with explicit overflow state."""

    segments: tuple[_EvidenceSourceSegment, ...]
    candidates_inspected: int
    overflowed: bool


def _locator_text(value: str) -> str:
    """Mirror the strict validator's evidence-locator representation."""

    return " ".join(value.casefold().split())


def _locator_is_valid(
    raw_fact: Mapping[str, object],
    normalized_page_text: str,
) -> bool:
    verbatim = raw_fact.get("verbatim")
    excerpt = raw_fact.get("evidence_excerpt")
    if not isinstance(verbatim, str) or not isinstance(excerpt, str):
        return False
    normalized_verbatim = _locator_text(verbatim)
    normalized_excerpt = _locator_text(excerpt)
    if not normalized_verbatim or not normalized_excerpt:
        return False
    return (
        normalized_excerpt in normalized_page_text
        and normalized_verbatim in normalized_excerpt
    )


def _source_segments(source: str, *, limit: int) -> _EvidenceSourceScan:
    """Collect a bounded source index without retaining partial overflow data."""

    bounded_limit = max(limit, 0)
    candidates: list[_EvidenceSourceSegment] = []
    seen_spans: set[tuple[int, int]] = set()
    candidates_inspected = 0

    def append_candidate(start: int, end: int, *, priority: int) -> bool:
        nonlocal candidates_inspected
        candidates_inspected += 1
        if candidates_inspected > bounded_limit:
            return False
        span = (start, end)
        if end - start > MAX_LONG_TEXT or span in seen_spans:
            return True
        seen_spans.add(span)
        candidates.append(
            _EvidenceSourceSegment(
                text=source[start:end],
                start=start,
                end=end,
                priority=priority,
            )
        )
        return True

    for match in _HTML_TABLE_ROW_RE.finditer(source):
        if not append_candidate(
            match.start(),
            match.end(),
            priority=0,
        ):
            return _EvidenceSourceScan(
                segments=(),
                candidates_inspected=candidates_inspected,
                overflowed=True,
            )
    for match in re.finditer(r"[^\r\n]+", source):
        start, end = match.span()
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        if not append_candidate(start, end, priority=1):
            return _EvidenceSourceScan(
                segments=(),
                candidates_inspected=candidates_inspected,
                overflowed=True,
            )
    return _EvidenceSourceScan(
        segments=tuple(candidates),
        candidates_inspected=candidates_inspected,
        overflowed=False,
    )


def _exact_term_pattern(value: str, *, numeric: bool = False) -> re.Pattern[str] | None:
    if len(value) > MAX_SHORT_TEXT:
        return None
    normalized = _locator_text(value)
    if not normalized:
        return None
    boundary = r"[\w.,]" if numeric else r"\w"
    return re.compile(rf"(?<!{boundary}){re.escape(normalized)}(?!{boundary})")


def _numeric_locator_value(
    raw_fact: Mapping[str, object],
    category: str,
) -> str | None:
    for field_name in _FACT_NUMERIC_FIELDS.get(category, ()):
        value = raw_fact.get(field_name)
        if (
            isinstance(value, str)
            and value.strip()
            and any(character.isdigit() for character in value)
        ):
            return value
    return None


def _find_rebind_segment(
    segments: tuple[_EvidenceSourceSegment, ...],
    raw_fact: Mapping[str, object],
    category: str,
) -> str | None:
    """Find one unambiguous exact source row or line for a raw fact."""

    subject_field = _FACT_SUBJECT_FIELDS.get(category)
    if subject_field is None:
        return None
    subject = raw_fact.get(subject_field)
    if not isinstance(subject, str) or len(subject) > MAX_SHORT_TEXT:
        return None
    numeric_value = _numeric_locator_value(raw_fact, category)
    if numeric_value is not None and len(numeric_value) > MAX_SHORT_TEXT:
        return None
    subject_pattern = _exact_term_pattern(subject)
    if subject_pattern is None:
        return None
    numeric_pattern = (
        _exact_term_pattern(numeric_value, numeric=True)
        if numeric_value is not None
        else None
    )

    matches: list[_EvidenceSourceSegment] = []
    for segment in segments:
        normalized_segment = _locator_text(segment.text)
        subject_matches = list(subject_pattern.finditer(normalized_segment))
        numeric_matches = (
            [True]
            if numeric_pattern is None
            else list(numeric_pattern.finditer(normalized_segment))
        )
        if len(subject_matches) > 1 and numeric_matches:
            return None
        if len(subject_matches) == 1 and numeric_matches:
            matches.append(segment)
    if not matches:
        return None

    groups: list[list[_EvidenceSourceSegment]] = []
    for segment in sorted(matches, key=lambda item: (item.start, item.end)):
        if not groups or segment.start >= max(item.end for item in groups[-1]):
            groups.append([segment])
        else:
            groups[-1].append(segment)
    if len(groups) != 1:
        return None
    return min(
        groups[0],
        key=lambda item: (item.priority, len(item.text), item.start),
    ).text


class _EvidenceLocatorRebinder:
    """Cache bounded page indexes and enforce document-global repair work."""

    def __init__(self) -> None:
        self._remaining_work = _MAX_EVIDENCE_SOURCE_SEGMENT_WORK_PER_DOCUMENT
        self._normalized_pages: dict[int, str] = {}
        self._page_segments: dict[
            int,
            tuple[_EvidenceSourceSegment, ...] | None,
        ] = {}

    def _normalized_page(self, page_number: int, page_text: str) -> str:
        if page_number not in self._normalized_pages:
            self._normalized_pages[page_number] = _locator_text(page_text)
        return self._normalized_pages[page_number]

    def _segments_for_page(
        self,
        page_number: int,
        page_text: str,
    ) -> tuple[_EvidenceSourceSegment, ...] | None:
        if page_number in self._page_segments:
            return self._page_segments[page_number]
        if self._remaining_work <= 0:
            self._page_segments[page_number] = None
            return None
        scan = _source_segments(
            page_text,
            limit=min(
                _MAX_EVIDENCE_SOURCE_SEGMENTS_PER_PAGE,
                self._remaining_work,
            ),
        )
        self._remaining_work = max(
            0,
            self._remaining_work - scan.candidates_inspected,
        )
        segments = None if scan.overflowed else scan.segments
        self._page_segments[page_number] = segments
        return segments

    def rebind(
        self,
        raw_fact: object,
        pages: Mapping[int, str],
        category: str,
    ) -> object:
        """Repair only an invalid locator with unique evidence on its claimed page."""

        if not isinstance(raw_fact, dict):
            return raw_fact
        page_number = raw_fact.get("page_number")
        if type(page_number) is not int:
            return raw_fact
        page_text = pages.get(page_number)
        if page_text is None:
            return raw_fact
        normalized_page = self._normalized_page(page_number, page_text)
        if _locator_is_valid(raw_fact, normalized_page):
            return raw_fact

        subject_field = _FACT_SUBJECT_FIELDS.get(category)
        if subject_field is None:
            return raw_fact
        subject = raw_fact.get(subject_field)
        if not isinstance(subject, str) or len(subject) > MAX_SHORT_TEXT:
            return raw_fact
        numeric_value = _numeric_locator_value(raw_fact, category)
        if numeric_value is not None and len(numeric_value) > MAX_SHORT_TEXT:
            return raw_fact

        segments = self._segments_for_page(page_number, page_text)
        if not segments or len(segments) > self._remaining_work:
            return raw_fact
        self._remaining_work -= len(segments)
        segment = _find_rebind_segment(segments, raw_fact, category)
        if segment is None:
            return raw_fact
        rebound = dict(raw_fact)
        rebound["verbatim"] = segment
        rebound["evidence_excerpt"] = segment
        return rebound


class StrictLocalJob(Protocol):
    """Immutable job fields consumed by the strict-local pipeline."""

    id: uuid.UUID
    processing_mode: str
    manifest_snapshot: dict[str, Any]
    manifest_sha256: str

    def revalidate_manifest_snapshot(self) -> dict[str, Any]: ...


class StrictLocalUpload(Protocol):
    """Immutable upload routing fields consumed by the pipeline."""

    id: uuid.UUID
    user_id: uuid.UUID
    file_hash: str
    processing_mode: str
    processing_manifest: dict[str, Any] | None
    processing_schema_version: str | None


@dataclass(frozen=True)
class PersistableEvidence:
    """Encrypted evidence material ready for ``ExtractionEvidence`` rows."""

    id: str
    page_number: int
    excerpt: str
    start_offset: int
    end_offset: int
    field_paths: tuple[str, ...]
    excerpt_sha256: str
    offset_representation: str


@dataclass(frozen=True)
class StrictLocalIngestionResult:
    """Validated output ready for encrypted persistence and FHIR mapping."""

    page_markdown: dict[int, str]
    validated_extraction: ClinicalDocumentExtraction
    entities: list[Any]
    evidence: list[PersistableEvidence]
    unresolved_fields: list[str]
    rejected_fields: list[str]


class MemoryCheckpointStore:
    """Small deterministic store for fake-worker tests and local smoke checks."""

    def __init__(self) -> None:
        self._pages: dict[tuple[str, int], OCRCheckpoint] = {}
        self._extractions: dict[tuple[str, str], ExtractionCheckpoint] = {}
        self._raw_extractions: dict[tuple[str, str], RawExtractionCheckpoint] = {}

    async def get_ocr_page(
        self,
        job_id: str | uuid.UUID,
        page_number: int,
        checkpoint_key: str,
        image_sha256: str,
    ) -> OCRCheckpoint | None:
        value = self._pages.get((str(job_id), page_number))
        if (
            value is None
            or value.checkpoint_key != checkpoint_key
            or value.image_sha256 != image_sha256
        ):
            return None
        return deepcopy(value)

    async def put_ocr_page(
        self,
        job_id: str | uuid.UUID,
        checkpoint: OCRCheckpoint,
    ) -> OCRCheckpoint:
        detached = deepcopy(checkpoint)
        self._pages[(str(job_id), checkpoint.page_number)] = detached
        return deepcopy(detached)

    async def count_pages(self, job_id: str | uuid.UUID) -> int:
        return sum(key[0] == str(job_id) for key in self._pages)

    async def get_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint | None:
        value = self._extractions.get((str(job_id), str(upload_id)))
        if value is None or extraction_checkpoint_identity(
            value
        ) != extraction_checkpoint_identity(expected):
            return None
        return deepcopy(value)

    async def put_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint:
        detached = deepcopy(checkpoint)
        self._extractions[(str(job_id), checkpoint.upload_id)] = detached
        return deepcopy(detached)

    async def get_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint | None:
        value = self._raw_extractions.get((str(job_id), str(upload_id)))
        if value is None or extraction_checkpoint_identity(
            value
        ) != extraction_checkpoint_identity(expected):
            return None
        if not hmac.compare_digest(
            value.raw_result_sha256,
            raw_extraction_result_sha256(value.raw_extraction_result),
        ):
            raise LocalValidationError("Raw extraction checkpoint result is invalid.")
        return deepcopy(value)

    async def put_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint:
        if not hmac.compare_digest(
            checkpoint.raw_result_sha256,
            raw_extraction_result_sha256(checkpoint.raw_extraction_result),
        ):
            raise LocalValidationError("Raw extraction checkpoint result is invalid.")
        detached = deepcopy(checkpoint)
        self._raw_extractions[(str(job_id), checkpoint.upload_id)] = detached
        return deepcopy(detached)


Rasterizer = Callable[..., Iterator[RasterizedPage]]
ProgressCallback = Callable[[dict[str, object]], object]
LivenessCallback = Callable[[], object]
SourceDigest = Callable[[Path, ScratchJob], str]


def _plaintext_source_digest(encrypted_path: Path, scratch: ScratchJob) -> str:
    """Stream-decrypt one source and hash its plaintext in secure scratch."""
    source = scratch.decrypt_to_file(encrypted_path)
    digest = hashlib.sha256()
    try:
        with scratch.open_file(source.name) as handle:
            handle.seek(0)
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    finally:
        scratch.remove_file(source.name)
    return digest.hexdigest()


class StrictLocalPipeline:
    """Run page OCR and grounded clinical extraction without provider routing."""

    def __init__(
        self,
        *,
        manager: WorkerManager,
        checkpoints: LocalAICheckpointStore,
        manifest: LocalAIManifest,
        scratch_root: Path,
        model_dir: Path,
        rasterize: Rasterizer = iter_rasterized_pages,
        source_digest: SourceDigest = _plaintext_source_digest,
        max_page_pixels: int = 40_000_000,
        on_progress: ProgressCallback | None = None,
        on_liveness: LivenessCallback | None = None,
    ) -> None:
        self.manager = manager
        self.checkpoints = checkpoints
        self.manifest = manifest
        self.scratch_root = Path(scratch_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.rasterize = rasterize
        self.source_digest = source_digest
        self.max_page_pixels = max_page_pixels
        self.on_progress = on_progress
        self.on_liveness = on_liveness
        manifest_payload = json.loads(
            json.dumps(asdict(manifest), ensure_ascii=True, allow_nan=False)
        )
        snapshot, digest = canonicalize_manifest_snapshot(manifest_payload)
        self._manifest_snapshot = snapshot
        self._manifest_digest = digest
        self._artifacts = {artifact.role: artifact for artifact in manifest.artifacts}

    async def run_ingestion(
        self,
        job: StrictLocalJob,
        upload: StrictLocalUpload,
        encrypted_path: Path,
    ) -> StrictLocalIngestionResult:
        """Run the validated local path or fail without any alternate route."""
        self._preflight(job, upload)
        job_id = str(job.id)
        pages: dict[int, str] = {}
        page_checkpoints: dict[int, OCRCheckpoint] = {}
        selected_images: dict[int, Path] = {}
        selected_image_bytes = 0
        selected_image_pixels = 0
        total_markdown_bytes = 0
        with ScratchJob(self.scratch_root, job_id) as scratch:
            source_digest = await asyncio.to_thread(
                self.source_digest,
                Path(encrypted_path),
                scratch,
            )
            if not hmac.compare_digest(source_digest, upload.file_hash):
                raise LocalValidationError(
                    "Strict-local source digest does not match the upload."
                )
            locked_manifest_path = scratch.create_file(
                "locked-manifest.json",
                json.dumps(
                    self._manifest_snapshot,
                    allow_nan=False,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            if Path(encrypted_path).suffix.lower() == ".rtf":
                checkpoint = await self._rtf_page(
                    job_id,
                    upload,
                    Path(encrypted_path),
                    scratch,
                )
                total_markdown_bytes += len(checkpoint.markdown.encode("utf-8"))
                pages[checkpoint.page_number] = checkpoint.markdown
                page_checkpoints[checkpoint.page_number] = checkpoint
            else:
                for page in self.rasterize(
                    encrypted_path,
                    scratch,
                    max_pages=_MAX_PAGES,
                    max_pixels=self.max_page_pixels,
                ):
                    checkpoint = await self._ocr_page(
                        job_id,
                        upload,
                        page,
                        locked_manifest_path,
                    )
                    total_markdown_bytes += len(checkpoint.markdown.encode("utf-8"))
                    if total_markdown_bytes > _MAX_DOCUMENT_MARKDOWN_BYTES:
                        raise LocalValidationError(
                            "Local OCR document exceeds the content limit."
                        )
                    if checkpoint.page_number in pages:
                        raise LocalValidationError(
                            "Local OCR returned a duplicate page."
                        )
                    pages[checkpoint.page_number] = checkpoint.markdown
                    page_checkpoints[checkpoint.page_number] = checkpoint
                    candidate_bytes = page.path.stat().st_size
                    candidate_pixels = page.width * page.height
                    if (
                        len(selected_images) < _MAX_SELECTED_IMAGES
                        and selected_image_bytes + candidate_bytes
                        <= _MAX_SELECTED_IMAGE_BYTES
                        and selected_image_pixels + candidate_pixels
                        <= _MAX_SELECTED_IMAGE_PIXELS
                        and self._needs_image_escalation(checkpoint.markdown)
                    ):
                        selected_images[checkpoint.page_number] = (
                            self._retain_selected_image(page, scratch)
                        )
                        selected_image_bytes += candidate_bytes
                        selected_image_pixels += candidate_pixels
            if not pages:
                raise LocalValidationError("Local OCR returned no pages.")
            expected_extraction = self._extraction_checkpoint(
                upload,
                pages,
                page_checkpoints,
            )
            cached_extraction = await self.checkpoints.get_extraction(
                job.id,
                upload.id,
                expected_extraction,
            )
            if cached_extraction is not None:
                await self._extraction_stage_progress("validating_extraction")
                validated = self._validate_extraction_result(
                    cached_extraction.extraction_result,
                    pages,
                    upload_id=str(upload.id),
                )
            else:
                expected_raw = RawExtractionCheckpoint(
                    upload_id=expected_extraction.upload_id,
                    checkpoint_key=expected_extraction.checkpoint_key,
                    source_sha256=expected_extraction.source_sha256,
                    ocr_text_sha256=expected_extraction.ocr_text_sha256,
                    page_bindings_sha256=expected_extraction.page_bindings_sha256,
                    manifest_sha256=expected_extraction.manifest_sha256,
                    schema_version=expected_extraction.schema_version,
                    prompt_version=expected_extraction.prompt_version,
                    page_count=expected_extraction.page_count,
                    raw_result_sha256="",
                    raw_extraction_result=None,
                )
                cached_raw = await self.checkpoints.get_raw_extraction(
                    job.id,
                    upload.id,
                    expected_raw,
                )
                if cached_raw is not None:
                    raw_extraction = cached_raw.raw_extraction_result
                else:
                    await self._progress(
                        None,
                        ModelRole.EXTRACTION,
                        page_total=len(pages),
                    )

                    async def extraction_progress(
                        value: dict[str, object],
                    ) -> None:
                        if self.on_progress is None:
                            return
                        allowed = {
                            "current",
                            "total",
                            "activity",
                            "attempt",
                            "attempt_limit",
                            "input_tokens",
                            "output_tokens",
                            "output_token_limit",
                            "splits_used",
                            "split_limit",
                            "active_memory_bytes",
                            "peak_memory_bytes",
                        }
                        safe = {
                            key: item
                            for key, item in value.items()
                            if key in allowed and type(item) is int and item >= 0
                        }
                        current = safe.get("current")
                        total = safe.get("total")
                        if type(current) is int and type(total) is int:
                            safe["worker_current"] = safe.pop("current")
                            safe["worker_total"] = safe.pop("total")
                        progress_value = self.on_progress(
                            {
                                "stage": ModelRole.EXTRACTION.value,
                                "model_role": ModelRole.EXTRACTION.value,
                                **safe,
                            }
                        )
                        if inspect.isawaitable(progress_value):
                            await progress_value

                    extraction_payload = self._extraction_payload(
                        job_id,
                        pages,
                        selected_images,
                        locked_manifest_path,
                    )
                    if self.on_liveness is None:
                        raw_extraction = await self.manager.run_attested(
                            self.manifest,
                            ModelRole.EXTRACTION,
                            extraction_payload,
                            on_progress=extraction_progress,
                        )
                    else:
                        raw_extraction = await self.manager.run_attested(
                            self.manifest,
                            ModelRole.EXTRACTION,
                            extraction_payload,
                            on_progress=extraction_progress,
                            on_liveness=self.on_liveness,
                        )
                    raw_checkpoint = RawExtractionCheckpoint(
                        **{
                            **asdict(expected_raw),
                            "raw_result_sha256": raw_extraction_result_sha256(
                                raw_extraction
                            ),
                            "raw_extraction_result": raw_extraction,
                        }
                    )
                    await self._extraction_stage_progress("persisting_raw_extraction")
                    await self.checkpoints.put_raw_extraction(
                        job.id,
                        raw_checkpoint,
                    )
                await self._extraction_stage_progress("validating_extraction")
                validated = self._validate_extraction_result(
                    raw_extraction,
                    pages,
                    upload_id=str(upload.id),
                )
                checkpoint = ExtractionCheckpoint(
                    **{
                        **asdict(expected_extraction),
                        "extraction_result": validated.model_dump(mode="json"),
                    }
                )
                await self._persisting_extraction_checkpoint_progress()
                await self.checkpoints.put_extraction(job.id, checkpoint)

        try:
            entities = to_extracted_entities(validated)
        except LocalValidationError:
            raise
        except Exception:
            raise LocalValidationError("Local extraction adapter is invalid.") from None
        evidence = self._persistable_evidence(validated)
        return StrictLocalIngestionResult(
            page_markdown=pages,
            validated_extraction=validated,
            entities=entities,
            evidence=evidence,
            unresolved_fields=list(validated.unresolved_fields),
            rejected_fields=list(validated.rejected_fields),
        )

    @staticmethod
    def _persistable_evidence(validated: Any) -> list[PersistableEvidence]:
        try:
            facts = [
                fact
                for category in FACT_CATEGORY_NAMES
                for fact in getattr(validated, category)
            ]
            if len(facts) != len(validated.evidence):
                raise LocalValidationError("Local extraction evidence is incomplete.")
            output: list[PersistableEvidence] = []
            for fact, evidence in zip(facts, validated.evidence, strict=True):
                output.append(
                    PersistableEvidence(
                        id=evidence.id,
                        page_number=evidence.page_number,
                        excerpt=fact.evidence_excerpt,
                        start_offset=evidence.start_offset,
                        end_offset=evidence.end_offset,
                        field_paths=tuple(evidence.field_paths),
                        excerpt_sha256=evidence.excerpt_sha256,
                        offset_representation=evidence.offset_representation,
                    )
                )
            return output
        except LocalValidationError:
            raise
        except Exception:
            raise LocalValidationError(
                "Local extraction evidence is invalid."
            ) from None

    def _preflight(self, job: StrictLocalJob, upload: StrictLocalUpload) -> None:
        try:
            job_mode = ProcessingMode(job.processing_mode)
            upload_mode = ProcessingMode(upload.processing_mode)
        except (TypeError, ValueError):
            raise LocalPolicyError("Strict-local processing mode is invalid.") from None
        if (
            job_mode is not ProcessingMode.VALIDATED_STRICT_LOCAL
            or upload_mode is not ProcessingMode.VALIDATED_STRICT_LOCAL
        ):
            raise LocalPolicyError("Strict-local processing was not selected.")
        stored = job.revalidate_manifest_snapshot()
        if (
            stored != self._manifest_snapshot
            or job.manifest_sha256 != self._manifest_digest
            or upload.processing_manifest != self._manifest_snapshot
            or upload.processing_schema_version != CLINICAL_EXTRACTION_SCHEMA_VERSION
        ):
            raise LocalPolicyError("Strict-local manifest snapshot does not match.")
        if (
            not isinstance(upload.file_hash, str)
            or len(upload.file_hash) != 64
            or any(
                character not in "0123456789abcdef" for character in upload.file_hash
            )
        ):
            raise LocalValidationError("Strict-local upload digest is invalid.")
        if not self.model_dir.is_absolute():
            raise LocalPolicyError("Strict-local model paths must be absolute.")

    async def _ocr_page(
        self,
        job_id: str,
        upload: StrictLocalUpload,
        page: RasterizedPage,
        locked_manifest_path: Path,
    ) -> OCRCheckpoint:
        key = ocr_checkpoint_key(
            upload.file_hash,
            page.page_number,
            _RASTER_VERSION,
            self._manifest_digest,
        )
        await self._progress(page.page_number, ModelRole.OCR)
        cached = await self.checkpoints.get_ocr_page(
            job_id,
            page.page_number,
            key,
            page.sha256,
        )
        if cached is not None:
            return cached
        payload = self._ocr_payload(job_id, page, locked_manifest_path)
        if self.on_liveness is None:
            raw_result = await self.manager.run_attested(
                self.manifest,
                ModelRole.OCR,
                payload,
            )
        else:
            raw_result = await self.manager.run_attested(
                self.manifest,
                ModelRole.OCR,
                payload,
                on_liveness=self.on_liveness,
            )
        checkpoint = self._parse_ocr_result(raw_result, page, key)
        stored = await self.checkpoints.put_ocr_page(job_id, checkpoint)
        return stored

    async def _rtf_page(
        self,
        job_id: str,
        upload: StrictLocalUpload,
        encrypted_path: Path,
        scratch: ScratchJob,
    ) -> OCRCheckpoint:
        key = ocr_checkpoint_key(
            upload.file_hash,
            1,
            _RTF_TEXT_VERSION,
            self._manifest_digest,
        )
        cached = await self.checkpoints.get_ocr_page(
            job_id,
            1,
            key,
            upload.file_hash,
        )
        if cached is not None:
            return cached

        source = scratch.decrypt_to_file(encrypted_path)
        with scratch.open_file(source.name) as handle:
            raw = handle.read(_MAX_RTF_SOURCE_BYTES + 1)
        if len(raw) > _MAX_RTF_SOURCE_BYTES:
            raise LocalValidationError("Local RTF exceeds the content limit.")
        decoded = raw.decode("utf-8", errors="replace")
        if not decoded.lstrip("\ufeff \t\r\n").startswith(r"{\rtf"):
            raise LocalValidationError("Local RTF document is malformed.")
        try:
            markdown = rtf_to_text(decoded)
        except Exception:
            raise LocalValidationError("Local RTF document is malformed.") from None
        if not markdown.strip():
            raise LocalValidationError("Local RTF returned no text.")
        if len(markdown.encode("utf-8")) > _MAX_DOCUMENT_MARKDOWN_BYTES:
            raise LocalValidationError("Local RTF exceeds the content limit.")

        checkpoint = OCRCheckpoint(
            page_number=1,
            checkpoint_key=key,
            image_sha256=upload.file_hash,
            markdown=markdown,
            width=1,
            height=1,
            warnings=("text_only",),
        )
        return await self.checkpoints.put_ocr_page(job_id, checkpoint)

    def _ocr_payload(
        self,
        job_id: str,
        page: RasterizedPage,
        locked_manifest_path: Path,
    ) -> dict[str, object]:
        artifact = self._artifact(ModelRole.OCR)
        return {
            "job_id": job_id,
            "manifest_path": str(locked_manifest_path),
            "manifest_identity": self._manifest_identity(ModelRole.OCR),
            "model_dir": str(self.model_dir),
            "scratch_dir": str(page.path.parent),
            "page_number": page.page_number,
            "image_path": str(page.path),
            "image_sha256": page.sha256,
            "max_output_tokens": artifact.decode_limits["max_output_tokens"],
        }

    def _extraction_checkpoint(
        self,
        upload: StrictLocalUpload,
        pages: Mapping[int, str],
        page_checkpoints: Mapping[int, OCRCheckpoint],
    ) -> ExtractionCheckpoint:
        if set(pages) != set(page_checkpoints):
            raise LocalValidationError("Local OCR checkpoint set is incomplete.")
        page_bindings = [
            {
                "page_number": page_number,
                "checkpoint_key": checkpoint.checkpoint_key,
                "image_sha256": checkpoint.image_sha256,
                "markdown_sha256": hashlib.sha256(
                    pages[page_number].encode("utf-8")
                ).hexdigest(),
            }
            for page_number, checkpoint in sorted(page_checkpoints.items())
        ]
        page_bindings_sha256 = hashlib.sha256(
            json.dumps(
                page_bindings,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        ocr_text_sha256 = combined_markdown_hash(pages)
        return ExtractionCheckpoint(
            upload_id=str(upload.id),
            checkpoint_key=extraction_checkpoint_key(
                ocr_text_sha256,
                CLINICAL_EXTRACTION_SCHEMA_VERSION,
                _EXTRACTION_PROMPT_VERSION,
                self._manifest_digest,
            ),
            source_sha256=upload.file_hash,
            ocr_text_sha256=ocr_text_sha256,
            page_bindings_sha256=page_bindings_sha256,
            manifest_sha256=self._manifest_digest,
            schema_version=CLINICAL_EXTRACTION_SCHEMA_VERSION,
            prompt_version=_EXTRACTION_PROMPT_VERSION,
            page_count=len(pages),
            extraction_result={},
        )

    def _extraction_payload(
        self,
        job_id: str,
        pages: Mapping[int, str],
        selected_images: Mapping[int, Path],
        locked_manifest_path: Path,
    ) -> dict[str, object]:
        artifact = self._artifact(ModelRole.EXTRACTION)
        page_markdown = [
            {"page_number": page_number, "markdown": markdown}
            for page_number, markdown in sorted(pages.items())
        ]
        try:
            canonical_source = json.dumps(
                {"source_pages": page_markdown},
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            raise LocalValidationError(
                "Local extraction document is invalid."
            ) from None
        if len(canonical_source) > _MAX_EXTRACTION_SOURCE_BYTES:
            raise LocalValidationError(
                "Local extraction document exceeds the content limit."
            )
        return {
            "job_id": job_id,
            "manifest_path": str(locked_manifest_path),
            "manifest_identity": self._manifest_identity(ModelRole.EXTRACTION),
            "model_dir": str(self.model_dir),
            "scratch_dir": str(locked_manifest_path.parent),
            "page_markdown": page_markdown,
            "image_paths": {
                str(page_number): str(path)
                for page_number, path in sorted(selected_images.items())
            },
            "schema": deepcopy(NUEXTRACT_TEMPLATE_V1),
            "max_output_tokens": artifact.decode_limits["max_output_tokens"],
        }

    @classmethod
    def _validate_extraction_result(
        cls,
        raw: object,
        pages: Mapping[int, str],
        *,
        upload_id: str,
    ) -> ClinicalDocumentExtraction:
        work_budget = _ValidationWorkBudget(_MAX_RAW_ITEMS_PER_DOCUMENT)
        locator_rebinder = _EvidenceLocatorRebinder()
        if not (
            isinstance(raw, dict)
            and raw.get("result_type") == _CHUNKED_EXTRACTION_RESULT_TYPE
        ):
            return cls._validate_chunk_facts(
                raw,
                pages,
                upload_id=upload_id,
                chunk_index=0,
                work_budget=work_budget,
                locator_rebinder=locator_rebinder,
            )
        if set(raw) != {"result_type", "chunks"}:
            raise LocalValidationError("Local extraction chunk envelope is invalid.")
        raw_chunks = raw.get("chunks")
        if (
            not isinstance(raw_chunks, list)
            or not raw_chunks
            or len(raw_chunks) > _MAX_EXTRACTION_CHUNKS
        ):
            raise LocalValidationError("Local extraction chunk envelope is invalid.")

        expected_pages = sorted(pages)
        observed_pages: list[int] = []
        validated_chunks: list[ClinicalDocumentExtraction] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict) or set(raw_chunk) != {
                "page_numbers",
                "extraction",
            }:
                raise LocalValidationError("Local extraction chunk is invalid.")
            page_numbers = raw_chunk.get("page_numbers")
            if (
                not isinstance(page_numbers, list)
                or not page_numbers
                or any(
                    isinstance(page_number, bool)
                    or not isinstance(page_number, int)
                    or page_number <= 0
                    for page_number in page_numbers
                )
                or page_numbers != sorted(set(page_numbers))
            ):
                raise LocalValidationError("Local extraction chunk pages are invalid.")
            start = len(observed_pages)
            end = start + len(page_numbers)
            if page_numbers != expected_pages[start:end]:
                raise LocalValidationError(
                    "Local extraction chunks do not match the document."
                )
            observed_pages.extend(page_numbers)
            chunk_pages = {
                page_number: pages[page_number] for page_number in page_numbers
            }
            validated_chunks.append(
                cls._validate_chunk_facts(
                    raw_chunk.get("extraction"),
                    chunk_pages,
                    upload_id=upload_id,
                    chunk_index=len(validated_chunks),
                    work_budget=work_budget,
                    locator_rebinder=locator_rebinder,
                )
            )

        if observed_pages != expected_pages:
            raise LocalValidationError(
                "Local extraction chunks do not cover the document."
            )
        if len(validated_chunks) == 1:
            return validated_chunks[0]
        try:
            merged = cls._merge_extraction_chunks(validated_chunks)
        except LocalValidationError:
            raise
        except Exception:
            raise LocalValidationError("Local extraction merge is invalid.") from None
        return cls._validate_final_extraction(
            merged,
            pages,
            upload_id=upload_id,
        )

    @classmethod
    def _validate_chunk_facts(
        cls,
        raw: object,
        pages: Mapping[int, str],
        *,
        upload_id: str,
        chunk_index: int,
        work_budget: _ValidationWorkBudget,
        locator_rebinder: _EvidenceLocatorRebinder,
    ) -> ClinicalDocumentExtraction:
        """Keep independently valid facts while quarantining model-local failures."""
        if not isinstance(raw, dict):
            raise LocalValidationError("Local extraction chunk is invalid.")
        try:
            encoded = json.dumps(
                raw,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise LocalValidationError("Local extraction chunk is invalid.") from None
        if len(encoded) > MAX_EXTRACTION_JSON_BYTES:
            raise LocalValidationError("Local extraction chunk exceeds the size limit.")
        if set(raw) - _EXTRACTION_RESULT_KEYS:
            raise LocalValidationError("Local extraction chunk is invalid.")
        if (
            raw.get("schema_version", CLINICAL_EXTRACTION_SCHEMA_VERSION)
            != CLINICAL_EXTRACTION_SCHEMA_VERSION
        ):
            raise LocalValidationError("Local extraction schema is invalid.")

        accepted: dict[str, object] = {
            "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
            "patient": None,
            **{category: [] for category in FACT_CATEGORY_NAMES},
            "unresolved_fields": [],
            "rejected_fields": [],
        }
        server_rejections: list[str] = []
        seen_facts: dict[str, set[str]] = {
            category: set() for category in FACT_CATEGORY_NAMES
        }
        raw_patient = raw.get("patient")
        if raw_patient is not None:
            patient: dict[str, str | None] = {}
            if not isinstance(raw_patient, dict):
                cls._append_rejection(
                    server_rejections,
                    f"chunks[{chunk_index}].patient:fact_validation_failed",
                )
            else:
                if set(raw_patient) - {"name", "date_of_birth"}:
                    cls._append_rejection(
                        server_rejections,
                        f"chunks[{chunk_index}].patient:fact_validation_failed",
                    )
                for field_name in ("name", "date_of_birth"):
                    value = raw_patient.get(field_name)
                    if value is None:
                        continue
                    if work_budget.claim(1) == 0:
                        cls._append_rejection(
                            server_rejections,
                            _VALIDATION_WORK_OVERFLOW_PATH,
                        )
                        continue
                    try:
                        validated_patient = validate_clinical_extraction(
                            {
                                "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
                                "patient": {field_name: value},
                            },
                            pages,
                            upload_id=upload_id,
                            strict_local=True,
                        )
                    except LocalValidationError:
                        cls._append_rejection(
                            server_rejections,
                            (
                                f"chunks[{chunk_index}].patient.{field_name}"
                                ":fact_validation_failed"
                            ),
                        )
                        continue
                    except Exception:
                        cls._append_rejection(
                            server_rejections,
                            (
                                f"chunks[{chunk_index}].patient.{field_name}"
                                ":fact_validation_failed"
                            ),
                        )
                        continue
                    if validated_patient.patient is not None:
                        patient[field_name] = getattr(
                            validated_patient.patient,
                            field_name,
                        )
            if patient:
                accepted["patient"] = patient

        for category in FACT_CATEGORY_NAMES:
            raw_facts = raw.get(category, [])
            if not isinstance(raw_facts, list):
                cls._append_rejection(
                    server_rejections,
                    f"chunks[{chunk_index}].{category}:fact_validation_failed",
                )
                continue
            target = accepted[category]
            if not isinstance(target, list):
                raise LocalValidationError("Local extraction merge is invalid.")
            facts_to_validate = work_budget.claim(len(raw_facts))
            if facts_to_validate < len(raw_facts):
                cls._append_rejection(
                    server_rejections,
                    _VALIDATION_WORK_OVERFLOW_PATH,
                )
            for fact_index in range(facts_to_validate):
                raw_fact = raw_facts[fact_index]
                path = f"chunks[{chunk_index}].{category}[{fact_index}]"
                try:
                    raw_fact = locator_rebinder.rebind(
                        raw_fact,
                        pages,
                        category,
                    )
                    candidate = validate_clinical_extraction(
                        {
                            "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
                            category: [raw_fact],
                        },
                        pages,
                        upload_id=upload_id,
                        strict_local=True,
                    )
                except LocalValidationError:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fact_validation_failed",
                    )
                    continue
                except Exception:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fact_validation_failed",
                    )
                    continue
                try:
                    adapted_entities = to_extracted_entities(candidate)
                except LocalValidationError:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:adapter_rejected",
                    )
                    continue
                except Exception:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:adapter_rejected",
                    )
                    continue
                if len(adapted_entities) != 1:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:adapter_rejected",
                    )
                    continue
                try:
                    mapped_records = validated_extraction_to_health_record_dicts(
                        candidate,
                        _FHIR_VALIDATION_UUID,
                        _FHIR_VALIDATION_UUID,
                        _FHIR_VALIDATION_UUID,
                    )
                except LocalValidationError:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fhir_mapping_rejected",
                    )
                    continue
                except Exception:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fhir_mapping_rejected",
                    )
                    continue
                if len(mapped_records) != 1:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fhir_mapping_rejected",
                    )
                    continue
                try:
                    fact = getattr(candidate, category)[0]
                    signature = clinical_fact_duplicate_signature(fact)
                    fact_payload = fact.model_dump(mode="json")
                except Exception:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fact_validation_failed",
                    )
                    continue
                if signature in seen_facts[category]:
                    continue
                if len(target) >= MAX_FACTS_PER_CATEGORY:
                    cls._append_rejection(
                        server_rejections,
                        f"{path}:fact_limit_exceeded",
                    )
                    continue
                seen_facts[category].add(signature)
                fact_payload["fact_id"] = None
                target.append(fact_payload)

        for field_name in ("unresolved_fields", "rejected_fields"):
            values = raw.get(field_name, [])
            if not isinstance(values, list):
                cls._append_rejection(
                    server_rejections,
                    f"chunks[{chunk_index}].{field_name}:fact_validation_failed",
                )

        # These lists are model-authored metadata, not grounded facts. Strict-local
        # quarantine never carries their contents into server-owned diagnostics.
        accepted["unresolved_fields"] = []
        accepted["rejected_fields"] = server_rejections

        # This final pass is deliberately outside the quarantine catches. If a
        # server-owned merge invariant fails, the strict-local job remains fatal.
        return cls._validate_final_extraction(
            accepted,
            pages,
            upload_id=upload_id,
        )

    @staticmethod
    def _validate_final_extraction(
        raw: dict[str, object],
        pages: Mapping[int, str],
        *,
        upload_id: str,
    ) -> ClinicalDocumentExtraction:
        try:
            return validate_clinical_extraction(
                raw,
                pages,
                upload_id=upload_id,
                strict_local=True,
            )
        except LocalValidationError:
            raise
        except Exception:
            raise LocalValidationError(
                "Local extraction invariant is invalid."
            ) from None

    @staticmethod
    def _append_rejection(values: list[str], path: str) -> None:
        """Append one server-owned path while preserving the schema's hard cap."""
        if path in values or _REJECTION_OVERFLOW_PATH in values:
            return
        if len(values) < MAX_FIELD_LIST_ITEMS - 1:
            values.append(path)
            return
        values.append(_REJECTION_OVERFLOW_PATH)

    @staticmethod
    def _merge_extraction_chunks(
        chunks: list[ClinicalDocumentExtraction],
    ) -> dict[str, object]:
        if not chunks:
            raise LocalValidationError("Local extraction returned no validated chunks.")
        merged: dict[str, object] = {
            "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
            "patient": None,
            **{category: [] for category in FACT_CATEGORY_NAMES},
            "unresolved_fields": [],
            "rejected_fields": [],
        }
        patient_fields: dict[str, tuple[str, str]] = {}
        seen_facts: dict[str, set[str]] = {
            category: set() for category in FACT_CATEGORY_NAMES
        }
        seen_fields = {
            "unresolved_fields": set(),
            "rejected_fields": set(),
        }
        conflicted_patient_fields: set[str] = set()

        for chunk in chunks:
            patient = chunk.patient
            if patient is not None:
                for field_name in ("name", "date_of_birth"):
                    value = getattr(patient, field_name)
                    if value is None or field_name in conflicted_patient_fields:
                        continue
                    normalized = " ".join(value.split()).casefold()
                    prior = patient_fields.get(field_name)
                    if prior is not None and prior[1] != normalized:
                        patient_fields.pop(field_name, None)
                        conflicted_patient_fields.add(field_name)
                        rejected = merged["rejected_fields"]
                        if not isinstance(rejected, list):
                            raise LocalValidationError(
                                "Local extraction merge is invalid."
                            )
                        StrictLocalPipeline._append_rejection(
                            rejected,
                            f"patient.{field_name}:conflict",
                        )
                    if prior is None:
                        patient_fields[field_name] = (
                            value,
                            normalized,
                        )

            for category in FACT_CATEGORY_NAMES:
                facts = getattr(chunk, category)
                target = merged[category]
                if not isinstance(target, list):
                    raise LocalValidationError("Local extraction merge is invalid.")
                for fact in facts:
                    identity = clinical_fact_duplicate_signature(fact)
                    if identity in seen_facts[category]:
                        continue
                    if len(target) >= MAX_FACTS_PER_CATEGORY:
                        rejected = merged["rejected_fields"]
                        if not isinstance(rejected, list):
                            raise LocalValidationError(
                                "Local extraction merge is invalid."
                            )
                        StrictLocalPipeline._append_rejection(
                            rejected,
                            f"{category}:fact_limit_exceeded",
                        )
                        continue
                    seen_facts[category].add(identity)
                    fact_payload = fact.model_dump(mode="json")
                    fact_payload["fact_id"] = None
                    target.append(fact_payload)

            for field_name in (
                "unresolved_fields",
                "rejected_fields",
            ):
                values = getattr(chunk, field_name)
                target = merged[field_name]
                if not isinstance(target, list):
                    raise LocalValidationError("Local extraction merge is invalid.")
                for value in values:
                    if value in seen_fields[field_name]:
                        continue
                    if len(target) >= MAX_FIELD_LIST_ITEMS:
                        continue
                    seen_fields[field_name].add(value)
                    if field_name == "rejected_fields":
                        StrictLocalPipeline._append_rejection(target, value)
                    else:
                        target.append(value)

        if patient_fields:
            merged["patient"] = {
                "name": (
                    patient_fields["name"][0] if "name" in patient_fields else None
                ),
                "date_of_birth": (
                    patient_fields["date_of_birth"][0]
                    if "date_of_birth" in patient_fields
                    else None
                ),
            }
        return merged

    @staticmethod
    def _needs_image_escalation(markdown: str) -> bool:
        """Select only pages with deterministic OCR structure warnings."""
        lowered = markdown.casefold()
        return (
            not markdown.strip()
            or _TABLE_SEPARATOR_RE.search(markdown) is not None
            or any(marker in lowered for marker in _IMAGE_ESCALATION_MARKERS)
        )

    @staticmethod
    def _retain_selected_image(page: RasterizedPage, scratch: ScratchJob) -> Path:
        """Copy one selected PNG to a bounded scratch file for NuExtract."""
        source = scratch.verify_file(page.path.name)
        filename = f"extraction-page-{page.page_number:04d}.png"
        scratch.reserve_file(filename)
        total = 0
        try:
            with (
                scratch.open_file(source.name) as input_handle,
                scratch.open_file(filename) as output_handle,
            ):
                input_handle.seek(0)
                output_handle.seek(0)
                output_handle.truncate(0)
                while chunk := input_handle.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_SELECTED_IMAGE_BYTES:
                        raise LocalValidationError(
                            "Selected local page image exceeds the content limit."
                        )
                    output_handle.write(chunk)
            return scratch.verify_file(filename)
        except BaseException:
            scratch.remove_file(filename)
            raise

    def _manifest_identity(self, role: ModelRole) -> dict[str, object]:
        artifact = self._artifact(role)
        return {
            "schema_version": self.manifest.schema_version,
            "pack_revision": self.manifest.pack_revision,
            "platform": self.manifest.platform,
            "runtime": deepcopy(self.manifest.runtime),
            "validation_suite_version": self.manifest.validation_suite_version,
            "roles": sorted(item.value for item in ModelRole),
            "role": role.value,
            "repository": artifact.repository,
            "revision": artifact.revision,
            "quantization": artifact.quantization,
            "license": artifact.license,
            "attribution": artifact.attribution,
            "manifest_sha256": self._manifest_digest,
        }

    def _artifact(self, role: ModelRole) -> ManifestArtifact:
        try:
            return self._artifacts[role]
        except KeyError:
            raise LocalPolicyError("Strict-local model role is unavailable.") from None

    @staticmethod
    def _parse_ocr_result(
        raw: object,
        page: RasterizedPage,
        checkpoint_key: str,
    ) -> OCRCheckpoint:
        if not isinstance(raw, dict) or set(raw) != {"markdown", "page_number"}:
            raise LocalValidationError("Local OCR result is invalid.")
        markdown = raw.get("markdown")
        page_number = raw.get("page_number")
        if (
            not isinstance(markdown, str)
            or len(markdown.encode("utf-8")) > _MAX_OCR_MARKDOWN_BYTES
            or isinstance(page_number, bool)
            or page_number != page.page_number
        ):
            raise LocalValidationError("Local OCR result is invalid.")
        return OCRCheckpoint(
            page_number=page.page_number,
            checkpoint_key=checkpoint_key,
            image_sha256=page.sha256,
            markdown=markdown,
            width=page.width,
            height=page.height,
            warnings=(),
        )

    async def _progress(
        self,
        page_number: int | None,
        role: ModelRole,
        *,
        page_total: int | None = None,
    ) -> None:
        if self.on_progress is None:
            return
        payload: dict[str, object] = {
            "stage": role.value,
            "model_role": role.value,
        }
        if page_number is not None:
            payload["page_index"] = page_number
        if page_total is not None:
            payload["page_total"] = page_total
        value = self.on_progress(payload)
        if inspect.isawaitable(value):
            await value

    async def _persisting_extraction_checkpoint_progress(self) -> None:
        """Publish a content-free stage before the durable extraction commit."""

        await self._extraction_stage_progress("persisting_extraction_checkpoint")

    async def _extraction_stage_progress(self, stage: str) -> None:
        """Publish one fixed, content-free extraction phase."""

        if self.on_progress is None:
            return
        value = self.on_progress(
            {
                "stage": stage,
                "model_role": ModelRole.EXTRACTION.value,
            }
        )
        if inspect.isawaitable(value):
            await value


def combined_markdown_hash(pages: Mapping[int, str]) -> str:
    """Hash ordered OCR content for downstream extraction checkpoint identity."""
    digest = hashlib.sha256()
    for page_number, markdown in sorted(pages.items()):
        digest.update(str(page_number).encode("ascii"))
        digest.update(b"\0")
        digest.update(markdown.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
