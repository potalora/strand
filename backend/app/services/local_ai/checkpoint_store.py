"""Encrypted durable storage for strict-local OCR page checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.local_ai import (
    LocalAIExtractionCheckpoint,
    LocalAIJob,
    LocalAIPage,
)
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.checkpoints import extraction_checkpoint_key
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.protocol import MAX_MESSAGE_BYTES

_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_OCR_MARKDOWN_BYTES = 4 * 1024 * 1024
_MAX_WARNING_CODES = 64
_MAX_EXTRACTION_RESULT_BYTES = 1024 * 1024
_MAX_VERSION_LENGTH = 128


@dataclass(frozen=True)
class OCRCheckpoint:
    """One validated page result recovered from encrypted persistence."""

    page_number: int
    checkpoint_key: str
    image_sha256: str
    markdown: str
    width: int
    height: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionCheckpoint:
    """One validated clinical extraction bound to its exact OCR inputs."""

    upload_id: str
    checkpoint_key: str
    source_sha256: str
    ocr_text_sha256: str
    page_bindings_sha256: str
    manifest_sha256: str
    schema_version: str
    prompt_version: str
    page_count: int
    extraction_result: dict[str, object]


@dataclass(frozen=True)
class RawExtractionCheckpoint:
    """One bounded raw worker response bound to its exact OCR inputs."""

    upload_id: str
    checkpoint_key: str
    source_sha256: str
    ocr_text_sha256: str
    page_bindings_sha256: str
    manifest_sha256: str
    schema_version: str
    prompt_version: str
    page_count: int
    raw_result_sha256: str
    raw_extraction_result: object


def extraction_checkpoint_identity(
    checkpoint: ExtractionCheckpoint | RawExtractionCheckpoint,
) -> tuple[object, ...]:
    """Return the non-content identity that must match before checkpoint reuse."""

    return (
        checkpoint.upload_id,
        checkpoint.checkpoint_key,
        checkpoint.source_sha256,
        checkpoint.ocr_text_sha256,
        checkpoint.page_bindings_sha256,
        checkpoint.manifest_sha256,
        checkpoint.schema_version,
        checkpoint.prompt_version,
        checkpoint.page_count,
    )


def _row_extraction_identity(
    row: LocalAIExtractionCheckpoint,
    upload_id: uuid.UUID,
) -> tuple[object, ...]:
    return (
        str(upload_id),
        row.checkpoint_key,
        row.source_sha256,
        row.ocr_text_sha256,
        row.page_bindings_sha256,
        row.manifest_sha256,
        row.schema_version,
        row.prompt_version,
        row.page_count,
    )


def raw_extraction_result_sha256(value: object) -> str:
    """Hash one strict JSON worker result after enforcing the protocol ceiling."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise LocalValidationError(
            "Raw extraction checkpoint result is invalid."
        ) from None
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise LocalValidationError("Raw extraction checkpoint result is invalid.")
    return hashlib.sha256(encoded).hexdigest()


def _raw_extraction_checkpoint_from_values(
    *,
    upload_id: object,
    checkpoint_key: object,
    source_sha256: object,
    ocr_text_sha256: object,
    page_bindings_sha256: object,
    manifest_sha256: object,
    schema_version: object,
    prompt_version: object,
    page_count: object,
    raw_result_sha256: object,
    raw_extraction_result: object,
    result_required: bool,
) -> RawExtractionCheckpoint:
    common = _extraction_checkpoint_common(
        upload_id=upload_id,
        checkpoint_key=checkpoint_key,
        source_sha256=source_sha256,
        ocr_text_sha256=ocr_text_sha256,
        page_bindings_sha256=page_bindings_sha256,
        manifest_sha256=manifest_sha256,
        schema_version=schema_version,
        prompt_version=prompt_version,
        page_count=page_count,
    )
    if not result_required and raw_extraction_result is None:
        result_hash = ""
    else:
        result_hash = _digest(
            raw_result_sha256,
            "Raw extraction checkpoint result digest",
        )
        computed_hash = raw_extraction_result_sha256(raw_extraction_result)
        if not hmac.compare_digest(result_hash, computed_hash):
            raise LocalValidationError("Raw extraction checkpoint result is invalid.")
    return RawExtractionCheckpoint(
        **common,
        raw_result_sha256=result_hash,
        raw_extraction_result=raw_extraction_result,
    )


def _version(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_VERSION_LENGTH
    ):
        raise LocalValidationError(f"{label} is invalid.")
    return value


def _extraction_result(value: object, *, required: bool) -> dict[str, object]:
    if not isinstance(value, dict) or (required and not value):
        raise LocalValidationError("Extraction checkpoint result is invalid.")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise LocalValidationError("Extraction checkpoint result is invalid.") from None
    if len(encoded) > _MAX_EXTRACTION_RESULT_BYTES:
        raise LocalValidationError("Extraction checkpoint result is invalid.")
    return value


def _extraction_checkpoint_from_values(
    *,
    upload_id: object,
    checkpoint_key: object,
    source_sha256: object,
    ocr_text_sha256: object,
    page_bindings_sha256: object,
    manifest_sha256: object,
    schema_version: object,
    prompt_version: object,
    page_count: object,
    extraction_result: object,
    result_required: bool,
) -> ExtractionCheckpoint:
    common = _extraction_checkpoint_common(
        upload_id=upload_id,
        checkpoint_key=checkpoint_key,
        source_sha256=source_sha256,
        ocr_text_sha256=ocr_text_sha256,
        page_bindings_sha256=page_bindings_sha256,
        manifest_sha256=manifest_sha256,
        schema_version=schema_version,
        prompt_version=prompt_version,
        page_count=page_count,
    )
    return ExtractionCheckpoint(
        **common,
        extraction_result=_extraction_result(
            extraction_result,
            required=result_required,
        ),
    )


def _extraction_checkpoint_common(
    *,
    upload_id: object,
    checkpoint_key: object,
    source_sha256: object,
    ocr_text_sha256: object,
    page_bindings_sha256: object,
    manifest_sha256: object,
    schema_version: object,
    prompt_version: object,
    page_count: object,
) -> dict[str, object]:
    upload = _uuid(upload_id, "Extraction checkpoint upload identifier")
    key = _digest(checkpoint_key, "Extraction checkpoint key")
    source_hash = _digest(source_sha256, "Extraction checkpoint source digest")
    ocr_hash = _digest(ocr_text_sha256, "Extraction checkpoint OCR digest")
    page_hash = _digest(
        page_bindings_sha256,
        "Extraction checkpoint page binding digest",
    )
    manifest_hash = _digest(
        manifest_sha256,
        "Extraction checkpoint manifest digest",
    )
    schema = _version(schema_version, "Extraction checkpoint schema version")
    prompt = _version(prompt_version, "Extraction checkpoint prompt version")
    expected_key = extraction_checkpoint_key(
        ocr_hash,
        schema,
        prompt,
        manifest_hash,
    )
    if not hmac.compare_digest(key, expected_key):
        raise LocalValidationError("Extraction checkpoint key is invalid.")
    return {
        "upload_id": str(upload),
        "checkpoint_key": key,
        "source_sha256": source_hash,
        "ocr_text_sha256": ocr_hash,
        "page_bindings_sha256": page_hash,
        "manifest_sha256": manifest_hash,
        "schema_version": schema,
        "prompt_version": prompt,
        "page_count": _positive_integer(
            page_count,
            "Extraction checkpoint page count",
        ),
    }


def _uuid(value: object, label: str) -> uuid.UUID:
    try:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        raise LocalValidationError(f"{label} is invalid.") from None
    return parsed


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise LocalValidationError(f"{label} is invalid.")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalValidationError(f"{label} is invalid.")
    return value


def _warnings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_WARNING_CODES:
        raise LocalValidationError("OCR warning codes are invalid.")
    output: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 128
            or any(
                not (
                    character.isascii() and (character.isalnum() or character in "._-")
                )
                for character in item
            )
        ):
            raise LocalValidationError("OCR warning codes are invalid.")
        output.append(item)
    return tuple(output)


def _checkpoint_from_values(
    *,
    page_number: object,
    checkpoint_key: object,
    image_sha256: object,
    result: object,
    warning_codes: object,
) -> OCRCheckpoint:
    page = _positive_integer(page_number, "OCR page number")
    key = _digest(checkpoint_key, "OCR checkpoint key")
    image_hash = _digest(image_sha256, "OCR image digest")
    if not isinstance(result, dict) or set(result) != {
        "markdown",
        "width",
        "height",
    }:
        raise LocalValidationError("OCR checkpoint result is invalid.")
    markdown = result.get("markdown")
    if (
        not isinstance(markdown, str)
        or len(markdown.encode("utf-8")) > _MAX_OCR_MARKDOWN_BYTES
    ):
        raise LocalValidationError("OCR checkpoint result is invalid.")
    return OCRCheckpoint(
        page_number=page,
        checkpoint_key=key,
        image_sha256=image_hash,
        markdown=markdown,
        width=_positive_integer(result.get("width"), "OCR page width"),
        height=_positive_integer(result.get("height"), "OCR page height"),
        warnings=_warnings(warning_codes),
    )


class CheckpointStore:
    """Read and durably commit encrypted local-AI checkpoints for one DB session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_ocr_page(
        self,
        job_id: str | uuid.UUID,
        page_number: int,
        checkpoint_key: str,
        image_sha256: str,
    ) -> OCRCheckpoint | None:
        """Return only an exact, still-valid page checkpoint."""
        job = _uuid(job_id, "Local AI job identifier")
        page = _positive_integer(page_number, "OCR page number")
        key = _digest(checkpoint_key, "OCR checkpoint key")
        image_hash = _digest(image_sha256, "OCR image digest")
        row = (
            await self._session.execute(
                select(LocalAIPage).where(
                    LocalAIPage.job_id == job,
                    LocalAIPage.page_number == page,
                )
            )
        ).scalar_one_or_none()
        if row is None or row.checkpoint_key != key or row.image_sha256 != image_hash:
            return None
        return _checkpoint_from_values(
            page_number=row.page_number,
            checkpoint_key=row.checkpoint_key,
            image_sha256=row.image_sha256,
            result=row.ocr_result,
            warning_codes=row.warnings,
        )

    async def put_ocr_page(
        self,
        job_id: str | uuid.UUID,
        checkpoint: OCRCheckpoint,
    ) -> OCRCheckpoint:
        """Replace a stale page slot and commit the exact encrypted result."""
        job = _uuid(job_id, "Local AI job identifier")
        validated = _checkpoint_from_values(
            page_number=checkpoint.page_number,
            checkpoint_key=checkpoint.checkpoint_key,
            image_sha256=checkpoint.image_sha256,
            result={
                "markdown": checkpoint.markdown,
                "width": checkpoint.width,
                "height": checkpoint.height,
            },
            warning_codes=checkpoint.warnings,
        )
        row = (
            await self._session.execute(
                select(LocalAIPage)
                .where(
                    LocalAIPage.job_id == job,
                    LocalAIPage.page_number == validated.page_number,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        values: dict[str, Any] = {
            "checkpoint_key": validated.checkpoint_key,
            "image_sha256": validated.image_sha256,
            "ocr_result": {
                "markdown": validated.markdown,
                "width": validated.width,
                "height": validated.height,
            },
            "warnings": list(validated.warnings),
        }
        if row is None:
            self._session.add(
                LocalAIPage(
                    job_id=job,
                    page_number=validated.page_number,
                    **values,
                )
            )
        else:
            for name, value in values.items():
                setattr(row, name, value)
        await self._session.commit()
        return validated

    async def count_pages(self, job_id: str | uuid.UUID) -> int:
        """Return the bounded number of checkpointed pages for a job."""
        job = _uuid(job_id, "Local AI job identifier")
        rows = await self._session.execute(
            select(LocalAIPage.id).where(LocalAIPage.job_id == job)
        )
        return len(rows.scalars().all())

    async def get_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint | None:
        """Return an exact raw worker result for the immutable job inputs."""

        job = _uuid(job_id, "Local AI job identifier")
        upload = _uuid(upload_id, "Extraction checkpoint upload identifier")
        validated_expected = _raw_extraction_checkpoint_from_values(
            **expected.__dict__,
            result_required=False,
        )
        if validated_expected.upload_id != str(upload):
            return None
        row = (
            await self._session.execute(
                select(LocalAIExtractionCheckpoint)
                .join(LocalAIJob, LocalAIJob.id == LocalAIExtractionCheckpoint.job_id)
                .join(
                    UploadedFile,
                    UploadedFile.id == LocalAIJob.upload_id,
                )
                .where(
                    LocalAIExtractionCheckpoint.job_id == job,
                    LocalAIJob.upload_id == upload,
                    LocalAIJob.manifest_sha256 == validated_expected.manifest_sha256,
                    UploadedFile.user_id == LocalAIJob.user_id,
                    UploadedFile.file_hash == validated_expected.source_sha256,
                    UploadedFile.processing_schema_version
                    == validated_expected.schema_version,
                )
            )
        ).scalar_one_or_none()
        if (
            row is None
            or row.raw_result_sha256 is None
            or row.raw_extraction_result is None
        ):
            return None
        raw_envelope = row.raw_extraction_result
        if not isinstance(raw_envelope, dict) or set(raw_envelope) != {"data"}:
            raise LocalValidationError("Raw extraction checkpoint result is invalid.")
        stored = _raw_extraction_checkpoint_from_values(
            upload_id=upload,
            checkpoint_key=row.checkpoint_key,
            source_sha256=row.source_sha256,
            ocr_text_sha256=row.ocr_text_sha256,
            page_bindings_sha256=row.page_bindings_sha256,
            manifest_sha256=row.manifest_sha256,
            schema_version=row.schema_version,
            prompt_version=row.prompt_version,
            page_count=row.page_count,
            raw_result_sha256=row.raw_result_sha256,
            raw_extraction_result=raw_envelope["data"],
            result_required=True,
        )
        if extraction_checkpoint_identity(stored) != extraction_checkpoint_identity(
            validated_expected
        ):
            return None
        return stored

    async def put_raw_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: RawExtractionCheckpoint,
    ) -> RawExtractionCheckpoint:
        """Replace a stale raw slot and commit before server validation."""

        job = _uuid(job_id, "Local AI job identifier")
        validated = _raw_extraction_checkpoint_from_values(
            **checkpoint.__dict__,
            result_required=True,
        )
        upload = _uuid(
            validated.upload_id,
            "Extraction checkpoint upload identifier",
        )
        valid_job = (
            await self._session.execute(
                select(LocalAIJob.id)
                .join(UploadedFile, UploadedFile.id == LocalAIJob.upload_id)
                .where(
                    LocalAIJob.id == job,
                    LocalAIJob.upload_id == upload,
                    LocalAIJob.manifest_sha256 == validated.manifest_sha256,
                    UploadedFile.user_id == LocalAIJob.user_id,
                    UploadedFile.file_hash == validated.source_sha256,
                    UploadedFile.processing_schema_version == validated.schema_version,
                )
            )
        ).scalar_one_or_none()
        if valid_job is None:
            raise LocalValidationError(
                "Raw extraction checkpoint job identity is invalid."
            )
        row = (
            await self._session.execute(
                select(LocalAIExtractionCheckpoint)
                .where(LocalAIExtractionCheckpoint.job_id == job)
                .with_for_update()
            )
        ).scalar_one_or_none()
        values: dict[str, Any] = {
            "checkpoint_key": validated.checkpoint_key,
            "source_sha256": validated.source_sha256,
            "ocr_text_sha256": validated.ocr_text_sha256,
            "page_bindings_sha256": validated.page_bindings_sha256,
            "manifest_sha256": validated.manifest_sha256,
            "schema_version": validated.schema_version,
            "prompt_version": validated.prompt_version,
            "page_count": validated.page_count,
            "raw_result_sha256": validated.raw_result_sha256,
            "raw_extraction_result": {
                "data": validated.raw_extraction_result,
            },
        }
        if row is None:
            self._session.add(LocalAIExtractionCheckpoint(job_id=job, **values))
        else:
            prior = _row_extraction_identity(row, upload)
            for name, value in values.items():
                setattr(row, name, value)
            if prior != extraction_checkpoint_identity(validated):
                row.extraction_result = None
        await self._session.commit()
        return validated

    async def get_extraction(
        self,
        job_id: str | uuid.UUID,
        upload_id: str | uuid.UUID,
        expected: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint | None:
        """Return an exact extraction checkpoint for the immutable job inputs."""

        job = _uuid(job_id, "Local AI job identifier")
        upload = _uuid(upload_id, "Extraction checkpoint upload identifier")
        validated_expected = _extraction_checkpoint_from_values(
            **expected.__dict__,
            result_required=False,
        )
        if validated_expected.upload_id != str(upload):
            return None
        row = (
            await self._session.execute(
                select(LocalAIExtractionCheckpoint)
                .join(LocalAIJob, LocalAIJob.id == LocalAIExtractionCheckpoint.job_id)
                .join(
                    UploadedFile,
                    UploadedFile.id == LocalAIJob.upload_id,
                )
                .where(
                    LocalAIExtractionCheckpoint.job_id == job,
                    LocalAIJob.upload_id == upload,
                    LocalAIJob.manifest_sha256 == validated_expected.manifest_sha256,
                    UploadedFile.user_id == LocalAIJob.user_id,
                    UploadedFile.file_hash == validated_expected.source_sha256,
                    UploadedFile.processing_schema_version
                    == validated_expected.schema_version,
                )
            )
        ).scalar_one_or_none()
        if row is None or row.extraction_result is None:
            return None
        stored = _extraction_checkpoint_from_values(
            upload_id=upload,
            checkpoint_key=row.checkpoint_key,
            source_sha256=row.source_sha256,
            ocr_text_sha256=row.ocr_text_sha256,
            page_bindings_sha256=row.page_bindings_sha256,
            manifest_sha256=row.manifest_sha256,
            schema_version=row.schema_version,
            prompt_version=row.prompt_version,
            page_count=row.page_count,
            extraction_result=row.extraction_result,
            result_required=True,
        )
        if extraction_checkpoint_identity(stored) != extraction_checkpoint_identity(
            validated_expected
        ):
            return None
        return stored

    async def put_extraction(
        self,
        job_id: str | uuid.UUID,
        checkpoint: ExtractionCheckpoint,
    ) -> ExtractionCheckpoint:
        """Replace a stale extraction slot and commit its encrypted result."""

        job = _uuid(job_id, "Local AI job identifier")
        validated = _extraction_checkpoint_from_values(
            **checkpoint.__dict__,
            result_required=True,
        )
        upload = _uuid(
            validated.upload_id,
            "Extraction checkpoint upload identifier",
        )
        valid_job = (
            await self._session.execute(
                select(LocalAIJob.id)
                .join(UploadedFile, UploadedFile.id == LocalAIJob.upload_id)
                .where(
                    LocalAIJob.id == job,
                    LocalAIJob.upload_id == upload,
                    LocalAIJob.manifest_sha256 == validated.manifest_sha256,
                    UploadedFile.user_id == LocalAIJob.user_id,
                    UploadedFile.file_hash == validated.source_sha256,
                    UploadedFile.processing_schema_version == validated.schema_version,
                )
            )
        ).scalar_one_or_none()
        if valid_job is None:
            raise LocalValidationError("Extraction checkpoint job identity is invalid.")
        row = (
            await self._session.execute(
                select(LocalAIExtractionCheckpoint)
                .where(LocalAIExtractionCheckpoint.job_id == job)
                .with_for_update()
            )
        ).scalar_one_or_none()
        values: dict[str, Any] = {
            "checkpoint_key": validated.checkpoint_key,
            "source_sha256": validated.source_sha256,
            "ocr_text_sha256": validated.ocr_text_sha256,
            "page_bindings_sha256": validated.page_bindings_sha256,
            "manifest_sha256": validated.manifest_sha256,
            "schema_version": validated.schema_version,
            "prompt_version": validated.prompt_version,
            "page_count": validated.page_count,
            "extraction_result": validated.extraction_result,
        }
        if row is None:
            self._session.add(LocalAIExtractionCheckpoint(job_id=job, **values))
        else:
            prior = _row_extraction_identity(row, upload)
            for name, value in values.items():
                setattr(row, name, value)
            if prior != extraction_checkpoint_identity(validated):
                row.raw_result_sha256 = None
                row.raw_extraction_result = None
        await self._session.commit()
        return validated
