from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.ai.llm.config import LLMConfig, ProviderCreds
from app.services.ai.llm.types import (
    DocumentPart,
    ImagePart,
    LLMResponse,
    LLMUsage,
)
from app.services.extraction import text_extractor


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop", model="m", usage=LLMUsage(1, 1, 2), raw=None)


@pytest.mark.asyncio
async def test_pdf_ocr_routes_a_document_part_to_vision_provider(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cfg = LLMConfig(
        routing={"default": "anthropic", "vision": "anthropic"},
        providers={"anthropic": ProviderCreds(api_key="k", model="claude-x")},
    )
    fake = AsyncMock()
    fake.complete.return_value = _resp("OCR TEXT")
    with patch.object(text_extractor, "extract_text_from_pdf_local", return_value=("", 0.0)), \
         patch.object(text_extractor, "_vision_candidates", return_value=[("anthropic", fake)]):
        out = await text_extractor.extract_text_from_pdf(pdf, api_key="gem", config=cfg)
    assert out == "OCR TEXT"
    req = fake.complete.call_args.args[0]
    assert any(isinstance(p, DocumentPart) for p in req.messages[0].content)


@pytest.mark.asyncio
async def test_ocr_capability_gap_falls_back_once_to_gemini(tmp_path):
    """A LOCAL capability gap (an on-machine model that can't do vision) is the
    ONE case that falls back: the attempt never left the host, so OCR may try the
    single documented Gemini cloud fallback. (Minimum-necessary egress, W12 —
    a cloud refusal is NOT re-sent to another vendor; see test_ocr_egress.py.)"""
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    blocked = AsyncMock()
    blocked.complete.return_value = _resp("")  # local model can't do vision
    good = AsyncMock()
    good.complete.return_value = _resp("RECOVERED TEXT")
    with patch.object(text_extractor, "extract_text_from_pdf_local", return_value=("", 0.0)), \
         patch.object(
             text_extractor, "_vision_candidates",
             return_value=[("ollama", blocked), ("gemini", good)]):
        out = await text_extractor.extract_text_from_pdf(
            pdf, api_key="gem", config=LLMConfig.from_settings())
    assert out == "RECOVERED TEXT"
    blocked.complete.assert_awaited_once()  # tried the chosen (local) provider first
    good.complete.assert_awaited_once()     # then the single Gemini fallback


@pytest.mark.asyncio
async def test_tiff_ocr_routes_an_image_part(tmp_path):
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00fake")
    fake = AsyncMock()
    fake.complete.return_value = _resp("TIFF TEXT")
    with patch.object(text_extractor, "_vision_candidates", return_value=[("openai", fake)]):
        out = await text_extractor.extract_text_from_tiff(tiff, api_key="gem")
    assert out == "TIFF TEXT"
    req = fake.complete.call_args.args[0]
    assert any(isinstance(p, ImagePart) for p in req.messages[0].content)


def test_vision_candidates_orders_chosen_first_and_filters_unconfigured():
    cfg = LLMConfig(
        routing={"default": "gemini", "vision": "openai"},
        providers={
            "openai": ProviderCreds(api_key="o", model="gpt"),
            "anthropic": ProviderCreds(api_key="a", model="claude"),
            "gemini": ProviderCreds(api_key="", model="gem"),  # no key, no fallback key
            "ollama": ProviderCreds(api_key="", base_url="http://x/v1", model="llava"),
        },
    )
    names = [n for n, _ in text_extractor._vision_candidates(cfg, api_key="")]
    assert names[0] == "openai"          # the chosen vision provider leads
    assert "anthropic" in names          # configured (has key)
    assert "gemini" not in names         # no key and no api_key fallback
    assert "ollama" in names             # local servers are always candidates


def test_vision_candidates_includes_gemini_when_api_key_supplied():
    cfg = LLMConfig(
        routing={"default": "gemini", "vision": "gemini"},
        providers={"gemini": ProviderCreds(api_key="", model="gem")},
    )
    names = [n for n, _ in text_extractor._vision_candidates(cfg, api_key="GEMKEY")]
    assert names[0] == "gemini"
