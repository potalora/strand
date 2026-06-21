from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.ai.llm.config import LLMConfig
from app.services.ai.llm.types import ImagePart, LLMResponse, LLMUsage
from app.services.extraction import text_extractor
from app.services.extraction.text_extractor import build_ocr_notice


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop", model="m", usage=LLMUsage(1, 1, 2), raw=None)


# --- build_ocr_notice (pure logic) ---

def test_notice_fallback_is_info_with_used_and_refused():
    notice = build_ocr_notice(
        [{"provider": "gemini", "status": "refused"}, {"provider": "anthropic", "status": "ok"}]
    )
    assert notice["type"] == "ocr_fallback" and notice["level"] == "info"
    assert notice["detail"]["used"] == "anthropic"
    assert notice["detail"]["refused"] == ["gemini"]
    assert "Anthropic" in notice["message"] and "Gemini" in notice["message"]


def test_notice_all_refused_is_warning_with_no_used():
    notice = build_ocr_notice(
        [{"provider": "gemini", "status": "refused"}, {"provider": "openai", "status": "error"}]
    )
    assert notice["type"] == "ocr_unreadable" and notice["level"] == "warning"
    assert notice["detail"]["used"] is None
    assert "No AI provider could read" in notice["message"]


def test_notice_first_try_success_is_none():
    # The chosen provider worked first try -> nothing worth surfacing.
    assert build_ocr_notice([{"provider": "gemini", "status": "ok"}]) is None


def test_notice_empty_trace_is_none():
    assert build_ocr_notice([]) is None


# --- _ocr_via_provider populates the trace ---

@pytest.mark.asyncio
async def test_ocr_trace_records_refusal_then_success():
    # A LOCAL capability gap (ollama can't do vision) is the one case that falls
    # back — to the single documented Gemini cloud fallback (W12 minimum-necessary
    # egress). A cloud refusal does NOT fan out; see test_ocr_egress.py.
    blocked = AsyncMock()
    blocked.complete.return_value = _resp("")  # refused / empty (local capability gap)
    good = AsyncMock()
    good.complete.return_value = _resp("OCR TEXT")
    trace: list = []
    with patch.object(
        text_extractor, "_vision_candidates",
        return_value=[("ollama", blocked), ("gemini", good)],
    ):
        out = await text_extractor._ocr_via_provider(
            [ImagePart(b"x", "image/png")], LLMConfig.from_settings(), "k", "instr", trace=trace)
    assert out == "OCR TEXT"
    assert trace == [
        {"provider": "ollama", "status": "refused"},
        {"provider": "gemini", "status": "ok"},
    ]
    # And the trace maps to a fallback notice.
    assert build_ocr_notice(trace)["type"] == "ocr_fallback"
