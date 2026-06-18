"""The entity extractor exposes an optional progress callback so the worker can
surface section-level progress. Existing callers (no callback) are unaffected.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.extraction.entity_extractor import extract_entities


def _fake_lx_result():
    extraction = SimpleNamespace(
        extraction_class="condition",
        extraction_text="Hypertension",
        attributes={"status": "active"},
        char_interval=None,
    )
    return SimpleNamespace(extractions=[extraction])


def test_progress_callback_invoked_when_provided():
    cb = MagicMock()
    with patch(
        "app.services.extraction.entity_extractor.lx.extract",
        return_value=_fake_lx_result(),
    ):
        result = extract_entities(
            "Patient has hypertension.",
            "note.rtf",
            "fake-key",
            progress_callback=cb,
        )
    assert result.error is None
    assert cb.called


def test_extract_entities_works_without_callback():
    """Backward compatibility: the param is optional."""
    with patch(
        "app.services.extraction.entity_extractor.lx.extract",
        return_value=_fake_lx_result(),
    ):
        result = extract_entities("Patient has hypertension.", "note.rtf", "fake-key")
    assert result.error is None
    assert len(result.entities) == 1
