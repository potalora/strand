from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.ai.llm.config import LLMConfig, ProviderCreds
from app.services.ai.llm.types import DocumentPart, ImagePart, LLMResponse, LLMUsage
from app.services.extraction import text_extractor


@pytest.mark.asyncio
async def test_pdf_ocr_routes_to_vision_provider(tmp_path):
    """A scanned PDF's bytes reach the configured vision provider as a DocumentPart."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cfg = LLMConfig(
        routing={"default": "anthropic", "vision": "anthropic"},
        providers={"anthropic": ProviderCreds(api_key="k", model="claude-x")},
    )
    fake = AsyncMock()
    fake.complete.return_value = LLMResponse(
        text="OCR TEXT", finish_reason="stop", model="claude-x",
        usage=LLMUsage(1, 1, 2), raw=None,
    )
    # Force the local pdfplumber path to "low confidence" so the vision provider runs.
    with patch.object(text_extractor, "extract_text_from_pdf_local", return_value=("", 0.0)), \
         patch.object(text_extractor, "get_provider", return_value=fake):
        out = await text_extractor.extract_text_from_pdf(pdf, api_key="gem", config=cfg)

    assert out == "OCR TEXT"
    req = fake.complete.call_args.args[0]
    parts = req.messages[0].content
    assert any(isinstance(p, DocumentPart) for p in parts)


@pytest.mark.asyncio
async def test_tiff_ocr_routes_image_part_to_vision_provider(tmp_path):
    """A TIFF's bytes reach the configured vision provider as an ImagePart."""
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake-tiff")
    cfg = LLMConfig(
        routing={"default": "openai", "vision": "openai"},
        providers={"openai": ProviderCreds(api_key="k", model="gpt-4o-mini")},
    )
    fake = AsyncMock()
    fake.complete.return_value = LLMResponse(
        text="TIFF OCR", finish_reason="stop", model="gpt-4o-mini",
        usage=LLMUsage(1, 1, 2), raw=None,
    )
    with patch.object(text_extractor, "get_provider", return_value=fake):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="gem", config=cfg)

    assert out == "TIFF OCR"
    req = fake.complete.call_args.args[0]
    parts = req.messages[0].content
    image_part = next(p for p in parts if isinstance(p, ImagePart))
    assert image_part.mime == "image/tiff"
    assert image_part.data == b"II*\x00fake-tiff"


@pytest.mark.asyncio
async def test_pdf_ocr_falls_back_to_gemini_on_vision_failure(tmp_path):
    """When the configured vision provider fails and a Gemini key exists, fall back."""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cfg = LLMConfig(
        routing={"default": "anthropic", "vision": "anthropic"},
        providers={"anthropic": ProviderCreds(api_key="k", model="claude-x")},
    )
    failing = AsyncMock()
    failing.complete.side_effect = RuntimeError("provider can't do vision")
    gemini = AsyncMock()
    gemini.complete.return_value = LLMResponse(
        text="GEMINI FALLBACK", finish_reason="stop", model="gemini-3.5-flash",
        usage=LLMUsage(1, 1, 2), raw=None,
    )
    with patch.object(text_extractor, "extract_text_from_pdf_local", return_value=("", 0.0)), \
         patch.object(text_extractor, "get_provider", return_value=failing), \
         patch("app.services.ai.llm.registry._build", return_value=gemini) as build:
        out = await text_extractor.extract_text_from_pdf(pdf, api_key="gem", config=cfg)

    assert out == "GEMINI FALLBACK"
    assert build.call_args.args[0] == "gemini"
