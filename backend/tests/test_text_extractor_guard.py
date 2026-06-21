from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.extraction import text_extractor


@pytest.mark.asyncio
async def test_tiff_ocr_raises_when_no_vision_provider_and_no_gemini_key(monkeypatch, tmp_path):
    """With no reachable vision provider AND api_key="" there is no Gemini fallback.

    The old hard guard ("Vision OCR requires GEMINI_API_KEY") is gone; OCR no longer
    *requires* a Gemini key up front, but it must still error clearly when nothing can
    perform vision (the vision provider fails and the Gemini fallback is unavailable).
    """
    monkeypatch.setattr(text_extractor.settings, "gemini_api_key", "")
    f = tmp_path / "x.tiff"
    f.write_bytes(b"II*\x00")  # minimal tiff magic

    # The configured vision provider is unreachable; with api_key="" there is no
    # Gemini fallback, so the underlying failure must propagate.
    failing = AsyncMock()
    failing.complete.side_effect = RuntimeError("no vision provider reachable")
    with patch.object(text_extractor, "get_provider", return_value=failing):
        with pytest.raises(Exception):
            await text_extractor.extract_text_from_tiff(f, api_key="")
