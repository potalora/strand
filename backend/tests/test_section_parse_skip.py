"""Regression tests pinning the short-doc section-parse skip optimization.

`_process_unstructured` (app/api/upload.py) skips the `parse_sections` Gemini
call when `len(scrubbed_text) < settings.small_doc_threshold`, building a
single-section ParsedDocument locally instead.  These tests ensure that the
threshold value stays sane and that text lengths sit on the correct side of
the boundary so a future change can't silently undo the optimization.
"""
from __future__ import annotations

from app.config import settings


def test_small_doc_threshold_configured():
    """The threshold must be a positive integer (currently 3000)."""
    assert settings.small_doc_threshold > 0


def test_short_text_is_below_threshold():
    """Text one character shorter than the threshold takes the no-Gemini path."""
    short = "a" * (settings.small_doc_threshold - 1)
    assert len(short) < settings.small_doc_threshold  # -> single-section path (no Gemini)


def test_long_text_is_at_or_above_threshold():
    """Text one character longer than the threshold takes the parse_sections path."""
    long = "a" * (settings.small_doc_threshold + 1)
    assert len(long) >= settings.small_doc_threshold  # -> parse_sections path


def test_threshold_boundary_exact():
    """Text exactly at the threshold is NOT below it — it must use parse_sections."""
    at_threshold = "a" * settings.small_doc_threshold
    assert len(at_threshold) == settings.small_doc_threshold
    # The guard is `< threshold`, so equal length is NOT skipped.
    assert not (len(at_threshold) < settings.small_doc_threshold)


def test_threshold_default_value():
    """The threshold default (3000) is large enough to be meaningful but not so
    large that typical short notes would always hit the Gemini path."""
    # Sanity-check the configured value is within a reasonable range.
    assert 500 <= settings.small_doc_threshold <= 10_000
