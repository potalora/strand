from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.config import settings
from app.services.extraction.entity_extractor import extract_entities


def test_extract_entities_uses_configured_max_workers():
    fake_result = MagicMock()
    fake_result.extractions = []
    with patch("app.services.extraction.entity_extractor.lx.extract",
               return_value=fake_result) as mock_extract:
        extract_entities("some clinical text", "note.pdf", "key")
    _, kwargs = mock_extract.call_args
    assert kwargs["max_workers"] == settings.gemini_concurrency_limit
    assert kwargs["max_workers"] > 1
