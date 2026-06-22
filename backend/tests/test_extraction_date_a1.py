"""A1 — AI-extracted record dates must come from the ORIGINAL (pre-de-id) text.

The de-id pass generalizes dates to year-only before entity extraction, so the
LLM only ever sees e.g. "2020" and returns year-only dates → records land on
2020-01-01. The effective_date is internal (never sent to an LLM), so we recover
the real document date from the original text and prefer it over the
de-identified entity dates, except for dateless types (family_history).
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.api.upload import _prefer_document_date


class _E:
    def __init__(self, entity_class: str):
        self.entity_class = entity_class


DOC = datetime(2020, 10, 21, tzinfo=timezone.utc)
YEAR_ONLY = datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_overrides_year_only_date_for_eligible_records():
    built = [
        (_E("observation"), {"effective_date": YEAR_ONLY}),
        (_E("condition"), {"effective_date": None}),
    ]
    _prefer_document_date(built, DOC)
    assert built[0][1]["effective_date"] == DOC  # year-only artifact replaced
    assert built[1][1]["effective_date"] == DOC  # dateless eligible record filled


def test_leaves_dateless_family_history_untouched():
    built = [(_E("family_history"), {"effective_date": None})]
    _prefer_document_date(built, DOC)
    assert built[0][1]["effective_date"] is None


def test_skips_none_record_dicts():
    built = [(_E("medication"), None)]
    _prefer_document_date(built, DOC)  # must not raise
    assert built[0][1] is None


def test_noop_when_no_document_date():
    built = [(_E("observation"), {"effective_date": YEAR_ONLY})]
    _prefer_document_date(built, None)
    assert built[0][1]["effective_date"] == YEAR_ONLY  # unchanged
