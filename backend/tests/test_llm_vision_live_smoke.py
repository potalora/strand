"""Live multimodal (vision) smoke tests.

Generates a small PNG containing known text and OCRs it through each vision-capable
provider, asserting the recognized text comes back. Gated on key/server presence;
a 429/quota response SKIPS (auth proven). Marked ``slow`` so the fast suite never
calls the network.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from app.config import settings
from app.services.ai.llm.config import LLMConfig
from app.services.ai.llm.registry import _build
from app.services.ai.llm.types import (
    ImagePart,
    LLMMessage,
    LLMRateLimitError,
    LLMRequest,
    TextPart,
)

pytestmark = pytest.mark.slow

_MARKER = "OCRTEST 42"


def _text_png(text: str = _MARKER) -> bytes:
    """Render ``text`` onto a small white PNG and return its bytes."""
    img = Image.new("RGB", (260, 80), "white")
    draw = ImageDraw.Draw(img)
    # Default PIL font is small but legible to modern vision models.
    draw.text((12, 28), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _ocr(provider_name: str) -> str:
    provider = _build(provider_name, LLMConfig.from_settings())
    req = LLMRequest(
        messages=[LLMMessage("user", [
            ImagePart(_text_png(), "image/png"),
            TextPart("Transcribe the exact text in this image. Return only the text."),
        ])],
        model="", max_output_tokens=64, temperature=0.0,
    )
    try:
        return (await provider.complete(req)).text or ""
    except LLMRateLimitError as exc:
        pytest.skip(f"{provider_name}: reached but rate-limited/quota-exhausted ({exc})")


def _has_marker(text: str) -> bool:
    return "42" in text or "OCRTEST" in text.upper()


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.gemini_api_key, reason="no GEMINI_API_KEY")
async def test_gemini_image_ocr() -> None:
    text = await _ocr("gemini")
    assert _has_marker(text), f"gemini OCR missed the marker: {text!r}"


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.anthropic_api_key, reason="no ANTHROPIC_API_KEY")
async def test_anthropic_image_ocr() -> None:
    text = await _ocr("anthropic")
    assert _has_marker(text), f"anthropic OCR missed the marker: {text!r}"


@pytest.mark.asyncio
@pytest.mark.skipif(not settings.openai_api_key, reason="no OPENAI_API_KEY")
async def test_openai_image_ocr() -> None:
    text = await _ocr("openai")
    assert _has_marker(text), f"openai OCR missed the marker: {text!r}"
