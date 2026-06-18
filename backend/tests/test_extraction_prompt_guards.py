"""Regression guard: the extraction prompt must document the precision rules
(A1-A4) and provider capture (B2) so the LangExtract few-shot keeps teaching
them. The pure post-extraction backstop lives in ``test_entity_validator.py``;
this pins the upstream prompt half.
"""
from __future__ import annotations

import re

from app.services.extraction.clinical_examples import (
    CLINICAL_EXAMPLES,
    CLINICAL_EXTRACTION_PROMPT,
)


def _prompt_lower() -> str:
    return CLINICAL_EXTRACTION_PROMPT.lower()


def test_prompt_requires_performed_procedures():
    p = _prompt_lower()
    # A1 — mentioned-not-performed
    assert "status post" in p or "s/p" in p
    assert "recommend" in p and "due for" in p


def test_prompt_rejects_value_only_fragments():
    # A2 — fragments
    p = _prompt_lower()
    assert "fragment" in p or "without a" in p or "value-only" in p


def test_prompt_routes_lifestyle_to_social_history():
    # A3 — lifestyle/counseling
    p = _prompt_lower()
    assert "social_history" in p
    assert "counsel" in p or "directive" in p or "recommend" in p


def test_prompt_rejects_drug_class_abbreviations():
    # A4 — drug classes / abbreviations
    p = _prompt_lower()
    assert "abbreviation" in p or "drug class" in p or "drug-class" in p


def test_prompt_captures_provider():
    # B2 — provider/attending capture
    p = _prompt_lower()
    assert "provider" in p


def test_examples_include_a_negative_rejection_example():
    """At least one few-shot example must demonstrate rejecting a recommended
    procedure / fragment / class abbreviation (a SKIP comment or absence)."""
    joined = " ".join(
        getattr(ex, "text", "") for ex in CLINICAL_EXAMPLES
    ).lower()
    # The negative example introduces recommendation/abbreviation language.
    assert re.search(r"recommend|due for|ppi|colonoscopy", joined)


def _extractions(entity_class: str):
    return [
        ex_x
        for ex in CLINICAL_EXAMPLES
        for ex_x in ex.extractions
        if ex_x.extraction_class == entity_class
    ]


def test_prompt_requests_structured_dosage_attributes():
    # B4 — dose/route/frequency as attributes on the medication entity
    p = _prompt_lower()
    assert "route" in p and "frequency" in p
    assert "as attributes on the medication" in p or "dosageinstruction" in p


def test_prompt_requests_condition_onset():
    # B5 — onset/diagnosed date on conditions
    assert "onset" in _prompt_lower()


def test_examples_emit_medication_dosage_attributes():
    """B4: medication few-shot entities carry dose/route/frequency inline as
    ATTRIBUTES (so the extractor emits structured dosage rather than separate
    entity classes that map to None and get dropped)."""
    med_attrs = [x.attributes for x in _extractions("medication")]
    with_dosage = [
        a for a in med_attrs
        if a.get("route") or a.get("frequency") or a.get("dose") or a.get("dosage")
    ]
    assert len(with_dosage) >= 2


def test_examples_emit_condition_onset_attribute():
    """B5: at least one condition few-shot entity demonstrates an onset attr."""
    onset_keys = {"onset_date", "onset", "since", "diagnosed", "diagnosis_date"}
    found = any(set(x.attributes) & onset_keys for x in _extractions("condition"))
    assert found
