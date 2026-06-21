from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pdfplumber
from striprtf.striprtf import rtf_to_text

from app.config import settings

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


# Order to try vision providers when the chosen one returns empty/blocked.
# Anthropic + OpenAI read scanned PDFs that Gemini sometimes RECITATION-blocks,
# so they lead the fallback order.
_VISION_FALLBACK_ORDER = (
    "anthropic", "openai", "gemini", "openrouter", "vertex", "ollama", "lmstudio",
)


def _vision_candidates(config: LLMConfig, api_key: str) -> list:
    """Ordered ``(name, provider)`` vision candidates to try for OCR.

    The user's chosen ``vision`` provider goes first, then the other
    configured + enabled providers in a works-most-often order. Only providers
    that are configured (cloud: has a key; local: always) and enabled are
    included. ``api_key`` supplies a Gemini key when the resolved config lacks
    one (legacy callers pass the ``.env`` key).
    """
    from app.services.ai.llm.config import LLMConfig, ProviderCreds
    from app.services.ai.llm.registry import _build

    chosen = config.routing.get("vision") or config.routing.get("default") or "gemini"
    order = [chosen] + [n for n in _VISION_FALLBACK_ORDER if n != chosen]
    candidates: list = []
    seen: set[str] = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        creds = config.providers.get(name)
        # Honor the caller's explicit Gemini key when config carries none.
        if name == "gemini" and (creds is None or not creds.api_key) and api_key:
            gem_cfg = LLMConfig(
                routing={"default": "gemini"},
                providers={"gemini": ProviderCreds(api_key=api_key, model=settings.gemini_model)},
            )
            candidates.append((name, _build("gemini", gem_cfg)))
            continue
        if creds is None:
            continue
        is_local = name in ("ollama", "lmstudio")
        if not (bool(creds.api_key) or is_local) or not creds.enabled:
            continue
        candidates.append((name, _build(name, config)))
    return candidates


async def _ocr_via_provider(
    parts: list,
    config: LLMConfig | None,
    api_key: str,
    instruction: str,
    *,
    trace: list | None = None,
) -> str:
    """Run OCR over ``parts``, falling back across configured vision providers.

    Tries the user's chosen ``vision`` provider first; if it returns no text
    (e.g. a Gemini RECITATION block) or fails, the next configured + enabled
    vision provider is tried, in order. Returns the first non-empty OCR text;
    ``""`` only when every candidate yields no text; raises only when no
    candidate could be tried at all.

    When ``trace`` is provided, each attempt appends ``{"provider", "status"}``
    (``ok`` / ``refused`` / ``error``) so callers can surface a user-facing notice.
    """
    from app.services.ai.llm import LLMMessage, LLMRequest
    from app.services.ai.llm.config import LLMConfig
    from app.services.ai.llm.types import TextPart

    cfg = config or LLMConfig.from_settings()
    req = LLMRequest(
        messages=[LLMMessage("user", [*parts, TextPart(instruction)])],
        model="",
        max_output_tokens=8192,
        temperature=0.0,
    )
    candidates = _vision_candidates(cfg, api_key)
    if not candidates:
        raise RuntimeError("No vision-capable provider is configured for OCR")
    last_err: Exception | None = None
    for name, provider in candidates:
        try:
            text = (await provider.complete(req)).text or ""
        except Exception as exc:  # noqa: BLE001 - try the next provider
            last_err = exc
            if trace is not None:
                trace.append({"provider": name, "status": "error"})
            logger.warning("OCR via %s failed (%s); trying next vision provider", name, exc)
            continue
        if text.strip():
            if trace is not None:
                trace.append({"provider": name, "status": "ok"})
            return text
        if trace is not None:
            trace.append({"provider": name, "status": "refused"})
        logger.info("OCR via %s returned no text (blocked/empty); trying next", name)
    if last_err is not None:
        raise last_err
    return ""


_PROVIDER_LABELS = {
    "gemini": "Gemini", "vertex": "Vertex", "openai": "OpenAI",
    "anthropic": "Anthropic", "openrouter": "OpenRouter",
    "ollama": "Ollama", "lmstudio": "LM Studio",
}


def build_ocr_notice(trace: list) -> dict | None:
    """Map an OCR attempt ``trace`` to a user-facing notice, or ``None``.

    ``trace`` is a list of ``{"provider", "status"}`` (``ok``/``refused``/``error``).
    Returns an ``ocr_fallback`` info notice when an earlier provider refused/failed
    but a later one produced text; an ``ocr_unreadable`` warning when attempts were
    made but none produced text; ``None`` when no OCR happened or the first provider
    simply worked (nothing worth telling the user).
    """
    if not trace:
        return None

    def label(name: str) -> str:
        return _PROVIDER_LABELS.get(name, name)

    used = next((a["provider"] for a in trace if a["status"] == "ok"), None)
    failed = [a["provider"] for a in trace if a["status"] in ("refused", "error")]
    if used is None:
        tried = ", ".join(label(a["provider"]) for a in trace)
        return {
            "type": "ocr_unreadable",
            "level": "warning",
            "message": f"No AI provider could read this document (tried {tried}).",
            "detail": {"used": None, "refused": failed, "attempts": trace},
        }
    if not failed:
        return None  # the chosen provider worked first try — nothing to surface
    refused = ", ".join(label(p) for p in failed)
    return {
        "type": "ocr_fallback",
        "level": "info",
        "message": f"Read by {label(used)} — {refused} declined this document.",
        "detail": {"used": used, "refused": failed, "attempts": trace},
    }


async def _extract_text_from_pdf_gemini(
    file_path: Path, api_key: str, config: LLMConfig | None = None,
    *, trace: list | None = None,
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
        trace=trace,
    )


async def extract_text_from_pdf(
    file_path: Path, api_key: str, config: LLMConfig | None = None,
    *, trace: list | None = None,
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
    return await _extract_text_from_pdf_gemini(file_path, api_key, config, trace=trace)


async def extract_text_from_tiff(
    file_path: Path, api_key: str, config: LLMConfig | None = None,
    *, trace: list | None = None,
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
        trace=trace,
    )


async def extract_text(
    file_path: Path, api_key: str, config: LLMConfig | None = None,
    *, trace: list | None = None,
) -> tuple[str, FileType]:
    """Dispatch to the appropriate text extractor based on file type.

    Returns:
        tuple: (extracted_text, file_type)

    When ``trace`` is provided, vision OCR (scanned PDF / TIFF) records its
    per-provider attempts into it; RTF and text-layer PDFs leave it empty.

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
        text = await extract_text_from_pdf(file_path, api_key, config, trace=trace)
    elif file_type == FileType.TIFF:
        text = await extract_text_from_tiff(file_path, api_key, config, trace=trace)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info("Extracted %d characters from %s", len(text), file_path.name)
    return text, file_type
