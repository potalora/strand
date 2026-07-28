"""Encrypted durable storage for strict-local OCR page checkpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.local_ai import LocalAIPage
from app.services.local_ai.errors import LocalValidationError

_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_OCR_MARKDOWN_BYTES = 4 * 1024 * 1024
_MAX_WARNING_CODES = 64


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
    """Read and durably commit encrypted page checkpoints for one DB session."""

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
