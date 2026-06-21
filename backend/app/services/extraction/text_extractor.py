from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber
from striprtf.striprtf import rtf_to_text

from app.config import settings
from app.services.ai.llm import get_provider

if TYPE_CHECKING:
    from app.services.ai.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    PDF = "pdf"
    RTF = "rtf"
    TIFF = "tiff"
    UNKNOWN = "unknown"


SUPPORTED_EXTENSIONS = {
    ".pdf": FileType.PDF,
    ".rtf": FileType.RTF,
    ".tif": FileType.TIFF,
    ".tiff": FileType.TIFF,
}

LOCAL_TEXT_MIN_CHARS_PER_PAGE = 50


def _render_tables(tables: list | None) -> str:
    """Render pdfplumber extract_tables() output as pipe-delimited rows."""
    if not tables:
        return ""
    lines: list[str] = []
    for table in tables:
        for row in table:
            lines.append(" | ".join((cell or "") for cell in row))
    return "\n".join(lines)


def extract_text_from_pdf_local(file_path: Path) -> tuple[str, float]:
    """Extract text + tables from a PDF's embedded text layer via pdfplumber.

    Returns (text, confidence) where confidence is average characters per page.
    A scanned/image-only PDF yields ~0 confidence because it has no text layer.
    """
    page_texts: list[str] = []
    total_chars = 0
    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages) or 1
        for page in pdf.pages:
            parts = [page.extract_text() or ""]
            table_text = _render_tables(page.extract_tables())
            if table_text:
                parts.append(table_text)
            page_text = "\n".join(p for p in parts if p)
            total_chars += len(page_text)
            page_texts.append(page_text)
    text = "\n\n".join(page_texts)
    confidence = total_chars / page_count
    return text, confidence


def detect_file_type(file_path: Path) -> FileType:
    """Detect unstructured file type by extension."""
    ext = file_path.suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, FileType.UNKNOWN)


def extract_text_from_rtf(file_path: Path) -> str:
    """Extract plaintext from RTF using striprtf (local, no API)."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    return rtf_to_text(raw)


async def _ocr_via_provider(
    parts: list, config: LLMConfig | None, api_key: str, instruction: str
) -> str:
    """Run OCR over ``parts`` via the user's configured ``vision`` provider.

    Builds a multimodal request (the image/document parts plus a trailing text
    instruction) and routes it through ``get_provider("vision", config)``. On any
    provider failure (capability/auth/transport), falls back to a Gemini vision
    provider built from ``api_key`` when one is available; otherwise re-raises so
    the caller surfaces a clear "no vision provider" error.
    """
    from app.services.ai.llm import LLMMessage, LLMRequest
    from app.services.ai.llm.config import LLMConfig, ProviderCreds
    from app.services.ai.llm.registry import _build
    from app.services.ai.llm.types import TextPart

    cfg = config or LLMConfig.from_settings()
    req = LLMRequest(
        messages=[LLMMessage("user", [*parts, TextPart(instruction)])],
        model="",
        max_output_tokens=8192,
        temperature=0.0,
    )
    try:
        return (await get_provider("vision", cfg).complete(req)).text or ""
    except Exception:
        if api_key:  # fall back to Gemini vision when a Gemini key is available
            logger.warning("vision provider OCR failed — falling back to Gemini", exc_info=True)
            gem = _build(
                "gemini",
                LLMConfig(
                    routing={"default": "gemini"},
                    providers={
                        "gemini": ProviderCreds(api_key=api_key, model=settings.gemini_model)
                    },
                ),
            )
            return (await gem.complete(req)).text or ""
        raise


async def _extract_text_from_pdf_gemini(
    file_path: Path, api_key: str, config: LLMConfig | None = None
) -> str:
    """OCR a (scanned) PDF by sending its bytes to the configured vision provider.

    Named for back-compat (callers/tests patch this symbol); the vision provider
    is now config-driven with a Gemini fallback rather than a hard-pinned client.
    """
    from app.services.ai.llm.types import DocumentPart

    pdf_bytes = file_path.read_bytes()
    return await _ocr_via_provider(
        [DocumentPart(pdf_bytes, "application/pdf")],
        config,
        api_key,
        "Extract all text from this document faithfully. Preserve structure, tables, "
        "and formatting. Return only the extracted text, no commentary.",
    )


async def extract_text_from_pdf(
    file_path: Path, api_key: str, config: LLMConfig | None = None
) -> str:
    """Local-first PDF text extraction; fall back to vision OCR when untrustworthy."""
    try:
        text, confidence = extract_text_from_pdf_local(file_path)
        if confidence >= LOCAL_TEXT_MIN_CHARS_PER_PAGE and text.strip():
            logger.info("PDF %s: used local text layer (%.0f chars/page)", file_path.name, confidence)
            return text
        logger.info(
            "PDF %s: low local confidence (%.0f chars/page) — using vision OCR",
            file_path.name,
            confidence,
        )
    except Exception:
        logger.exception("Local PDF extraction failed for %s — using vision OCR", file_path.name)
    return await _extract_text_from_pdf_gemini(file_path, api_key, config)


async def extract_text_from_tiff(
    file_path: Path, api_key: str, config: LLMConfig | None = None
) -> str:
    """Extract text from a TIFF image via the configured vision provider (OCR)."""
    from app.services.ai.llm.types import ImagePart

    tiff_bytes = file_path.read_bytes()
    return await _ocr_via_provider(
        [ImagePart(tiff_bytes, "image/tiff")],
        config,
        api_key,
        "Extract all text from this scanned document. Preserve the original layout "
        "and structure. Return only the extracted text, no commentary.",
    )


async def extract_text(
    file_path: Path, api_key: str, config: LLMConfig | None = None
) -> tuple[str, FileType]:
    """Dispatch to the appropriate text extractor based on file type.

    Returns:
        tuple: (extracted_text, file_type)

    Raises:
        ValueError: If the file type is unsupported.
    """
    file_type = detect_file_type(file_path)
    if file_type == FileType.UNKNOWN:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    logger.info("Extracting text from %s (type=%s)", file_path.name, file_type.value)

    if file_type == FileType.RTF:
        text = extract_text_from_rtf(file_path)
    elif file_type == FileType.PDF:
        text = await extract_text_from_pdf(file_path, api_key, config)
    elif file_type == FileType.TIFF:
        text = await extract_text_from_tiff(file_path, api_key, config)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info("Extracted %d characters from %s", len(text), file_path.name)
    return text, file_type
