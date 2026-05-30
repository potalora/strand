from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types
from PIL import Image
from striprtf.striprtf import rtf_to_text

from app.config import settings

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


async def _extract_text_from_pdf_gemini(file_path: Path, api_key: str) -> str:
    """Extract text from PDF by sending bytes to Gemini 3 Flash."""
    client = genai.Client(api_key=api_key)
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "Extract all text from this document faithfully. Preserve structure, tables, and formatting. Return only the extracted text, no commentary.",
        ],
    )
    return response.text or ""


async def extract_text_from_pdf(file_path: Path, api_key: str) -> str:
    """Local-first PDF text extraction; fall back to Gemini vision when untrustworthy."""
    try:
        text, confidence = extract_text_from_pdf_local(file_path)
        if confidence >= LOCAL_TEXT_MIN_CHARS_PER_PAGE and text.strip():
            logger.info("PDF %s: used local text layer (%.0f chars/page)", file_path.name, confidence)
            return text
        logger.info(
            "PDF %s: low local confidence (%.0f chars/page) — using Gemini vision",
            file_path.name,
            confidence,
        )
    except Exception:
        logger.exception("Local PDF extraction failed for %s — using Gemini vision", file_path.name)
    return await _extract_text_from_pdf_gemini(file_path, api_key)


async def extract_text_from_tiff(file_path: Path, api_key: str) -> str:
    """Extract text from TIFF image via Gemini 3 Flash OCR."""
    client = genai.Client(api_key=api_key)
    img = Image.open(file_path)

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=[
            img,
            "Extract all text from this scanned document. Preserve the original layout and structure. Return only the extracted text, no commentary.",
        ],
    )
    return response.text or ""


async def extract_text(file_path: Path, api_key: str) -> tuple[str, FileType]:
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
        text = await extract_text_from_pdf(file_path, api_key)
    elif file_type == FileType.TIFF:
        text = await extract_text_from_tiff(file_path, api_key)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info("Extracted %d characters from %s", len(text), file_path.name)
    return text, file_type
