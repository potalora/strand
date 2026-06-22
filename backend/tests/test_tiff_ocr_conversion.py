"""A3 — TIFF OCR must convert to a vision-supported format before the LLM call.

Gemini (and the other vision providers) accept PNG/JPEG/WEBP/PDF but NOT
image/tiff, so sending raw TIFF bytes to the vision API fails every scanned-TIFF
upload (observed: 5/5 TIFs → LLMResponseError, no text). Convert each TIFF page
to PNG with Pillow first. Multi-page TIFFs (common for scanned faxes) yield one
PNG image part per page.
"""
from __future__ import annotations

import io

from PIL import Image

from app.services.extraction.text_extractor import _tiff_to_png_parts

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_tiff(pages: int = 1) -> bytes:
    imgs = [Image.new("RGB", (8, 8), (30 * i, 0, 0)) for i in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="TIFF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def test_converts_single_page_tiff_to_png():
    parts = _tiff_to_png_parts(_make_tiff(1))
    assert len(parts) == 1
    assert parts[0].mime == "image/png"
    assert parts[0].data[:8] == PNG_MAGIC


def test_converts_each_page_of_multipage_tiff():
    parts = _tiff_to_png_parts(_make_tiff(3))
    assert len(parts) == 3
    assert all(p.mime == "image/png" for p in parts)
    assert all(p.data[:8] == PNG_MAGIC for p in parts)
