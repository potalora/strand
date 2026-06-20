from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config import settings
from app.services.extraction import text_extractor
from tests.conftest import private_fixture_root

# Real-data fixtures resolve via REAL_MEDICAL_FIXTURES_DIR (gitignored, off-repo);
# originals live under <root>/raw/. No in-repo fallback — skip cleanly if absent.
_FIXROOT = private_fixture_root()
_RAW = (_FIXROOT / "raw") if _FIXROOT else None
_NOTE = next(iter(_RAW.glob("note_*.pdf")), None) if _RAW else None
_SCANNED = (_RAW / "ibs_smart.pdf") if _RAW else None
_HAS_KEY = bool(settings.gemini_api_key)


@pytest.mark.slow
@pytest.mark.skipif(_NOTE is None, reason="real note PDF required")
def test_textlayer_note_uses_local_no_gemini_vision():
    text, confidence = text_extractor.extract_text_from_pdf_local(_NOTE)
    print(f"\nnote PDF confidence: {confidence:.1f} chars/page (threshold={text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE})")
    assert confidence >= text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE, (
        f"note confidence {confidence} below threshold — is it really a text-layer PDF?"
    )
    assert len(text.strip()) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_KEY or _NOTE is None, reason="GEMINI_API_KEY + note required")
@pytest.mark.asyncio
async def test_router_textlayer_note_skips_gemini_vision():
    with patch.object(text_extractor, "_extract_text_from_pdf_gemini") as gem:
        out = await text_extractor.extract_text_from_pdf(_NOTE, settings.gemini_api_key)
    assert len(out.strip()) > 0
    gem.assert_not_called()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_KEY or not (_SCANNED and _SCANNED.exists()),
                    reason="GEMINI_API_KEY + scanned ibs_smart.pdf required")
@pytest.mark.asyncio
async def test_router_scanned_pdf_falls_back_to_gemini_vision():
    text, confidence = text_extractor.extract_text_from_pdf_local(_SCANNED)
    print(f"\nibs_smart.pdf local confidence: {confidence:.1f} chars/page (threshold={text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE})")
    if confidence >= text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE:
        pytest.skip(f"ibs_smart.pdf has a text layer (conf={confidence}); not a scanned fixture")
    out = await text_extractor.extract_text_from_pdf(_SCANNED, settings.gemini_api_key)
    assert len(out.strip()) > 0
