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

from app.services.local_ai.adapters import to_extracted_entities
from app.services.local_ai.checkpoint_store import OCRCheckpoint
from app.services.local_ai.checkpoints import ocr_checkpoint_key
from app.services.local_ai.errors import LocalPolicyError, LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
    NUEXTRACT_TEMPLATE_V1,
    ClinicalDocumentExtraction,
)
from app.services.local_ai.extraction_validator import validate_clinical_extraction
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
_MAX_PAGES = 500
_MAX_OCR_MARKDOWN_BYTES = 4 * 1024 * 1024
_MAX_DOCUMENT_MARKDOWN_BYTES = 16 * 1024 * 1024
_MAX_RTF_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_EXTRACTION_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_SELECTED_IMAGES = 8
_MAX_SELECTED_IMAGE_BYTES = 64 * 1024 * 1024
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

    async def run(
        self,
        role: ModelRole,
        payload: dict[str, Any],
        on_progress: Callable[[dict[str, object]], object] | None = None,
    ) -> Any: ...


class OCRCheckpointStore(Protocol):
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


Rasterizer = Callable[..., Iterator[RasterizedPage]]
ProgressCallback = Callable[[dict[str, object]], object]
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
        checkpoints: OCRCheckpointStore,
        manifest: LocalAIManifest,
        scratch_root: Path,
        model_dir: Path,
        rasterize: Rasterizer = iter_rasterized_pages,
        source_digest: SourceDigest = _plaintext_source_digest,
        max_page_pixels: int = 40_000_000,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.manager = manager
        self.checkpoints = checkpoints
        self.manifest = manifest
        self.scratch_root = Path(scratch_root)
        self.model_dir = Path(model_dir)
        self.rasterize = rasterize
        self.source_digest = source_digest
        self.max_page_pixels = max_page_pixels
        self.on_progress = on_progress
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
        selected_images: dict[int, Path] = {}
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
                    if len(
                        selected_images
                    ) < _MAX_SELECTED_IMAGES and self._needs_image_escalation(
                        checkpoint.markdown
                    ):
                        selected_images[checkpoint.page_number] = (
                            self._retain_selected_image(page, scratch)
                        )
            if not pages:
                raise LocalValidationError("Local OCR returned no pages.")
            await self._progress(
                None,
                ModelRole.EXTRACTION,
                page_total=len(pages),
            )
            raw_extraction = await self.manager.run(
                ModelRole.EXTRACTION,
                self._extraction_payload(
                    job_id,
                    pages,
                    selected_images,
                    locked_manifest_path,
                ),
            )

        validated = validate_clinical_extraction(
            raw_extraction,
            pages,
            upload_id=str(upload.id),
            strict_local=True,
        )
        entities = to_extracted_entities(validated)
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
        raw_result = await self.manager.run(
            ModelRole.OCR,
            self._ocr_payload(job_id, page, locked_manifest_path),
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


def combined_markdown_hash(pages: Mapping[int, str]) -> str:
    """Hash ordered OCR content for downstream extraction checkpoint identity."""
    digest = hashlib.sha256()
    for page_number, markdown in sorted(pages.items()):
        digest.update(str(page_number).encode("ascii"))
        digest.update(b"\0")
        digest.update(markdown.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
