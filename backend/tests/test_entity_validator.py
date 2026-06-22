"""Tests for the LangExtract over-extraction precision guards (remediation A1-A6).

The validator is a pure, defensive post-extraction layer that drops/repairs the
false entities LangExtract invents.  See ``docs/extraction-remediation.md``
section A and ``entity_validator.py``.
"""
from __future__ import annotations

import pytest

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_validator import (
    normalize_entity_text,
    validate_entities,
)


def _e(entity_class: str, text: str, **attrs) -> ExtractedEntity:
    return ExtractedEntity(entity_class=entity_class, text=text, attributes=dict(attrs))


def _classes(entities: list[ExtractedEntity]) -> list[str]:
    return [e.entity_class for e in entities]


def _texts(entities: list[ExtractedEntity]) -> list[str]:
    return [e.text for e in entities]


# ---------------------------------------------------------------------------
# A5 — PHI-scrubber placeholders extracted as content (drop, all classes)
# ---------------------------------------------------------------------------


def test_a5_drops_entities_that_are_phi_placeholders():
    entities = [
        _e("procedure", "[NAME] (12+ YO)"),
        _e("condition", "[DATE]"),
        _e("observation", "[REDACTED]"),
        _e("medication", "[PATIENT]"),
        _e("condition", "Hypertension"),
    ]
    out = validate_entities(entities)
    assert _texts(out) == ["Hypertension"]


def test_a5_keeps_text_with_parenthetical_icd_code():
    """ICD codes use parentheses, not the bracketed all-caps placeholder form."""
    out = validate_entities([_e("condition", "Gastroparesis (K31.84)", status="active")])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# A1 — Mentioned-not-performed procedures
# ---------------------------------------------------------------------------


def test_a1_drops_bare_procedure_with_no_evidence_of_performance():
    """A bare procedure name with no date / performed signal is a mention."""
    out = validate_entities([_e("procedure", "Colonoscopy")])
    assert out == []


def test_a1_drops_recommended_procedure():
    out = validate_entities([_e("procedure", "recommend colonoscopy")])
    assert out == []


def test_a1_drops_due_for_procedure():
    out = validate_entities([_e("procedure", "Colonoscopy", note="due for screening")])
    assert out == []


def test_a1_keeps_procedure_with_date_attribute():
    out = validate_entities([_e("procedure", "Colonoscopy", date="01/2024")])
    assert len(out) == 1
    assert out[0].entity_class == "procedure"


def test_a1_keeps_status_post_procedure():
    out = validate_entities([_e("procedure", "s/p Cystectomy")])
    assert len(out) == 1


def test_a1_keeps_surgical_resection_by_suffix():
    """A surgical -ectomy implies the surgery was performed."""
    out = validate_entities([_e("procedure", "Appendectomy")])
    assert len(out) == 1


def test_a1_drops_scheduled_procedure_even_with_date():
    """A planned/scheduled signal overrides a (future) date."""
    out = validate_entities([_e("procedure", "Colonoscopy scheduled", date="01/2030")])
    assert out == []


# ---------------------------------------------------------------------------
# A2 — Fragment entities (value-only, no named analyte/drug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["2mg", '5\' 9"', "120/80", "98.6", "  ", "140"])
def test_a2_drops_value_only_fragments(text):
    out = validate_entities([_e("observation", text)])
    assert out == []


@pytest.mark.parametrize("text", ["CBC", "CMP", "A1c", "FIT"])
def test_a2_keeps_named_valueless_panel_tokens(text):
    """Named-but-valueless lab/panel tokens are handled downstream (Agent B B8),
    so the fragment guard must NOT drop them."""
    out = validate_entities([_e("lab_result", text)])
    assert _texts(out) == [text]


@pytest.mark.parametrize("text", ["Glucose 95 mg/dL", "HbA1c 6.8%", "BP 120/80 mmHg", "WBC 6.2"])
def test_a2_keeps_named_value_pairs(text):
    out = validate_entities([_e("lab_result", text)])
    assert len(out) == 1


def test_a2_does_not_touch_conditions_or_allergies():
    """A2 only applies to observation-family entities, never conditions/meds."""
    entities = [_e("condition", "anemia"), _e("allergy", "Penicillin")]
    assert len(validate_entities(entities)) == 2


# ---------------------------------------------------------------------------
# A3 — Recommendations / lifestyle as clinical observations
# ---------------------------------------------------------------------------


def test_a3_reclassifies_lifestyle_observation_to_social_history():
    out = validate_entities([_e("observation", "Exercise: Tennis player")])
    assert len(out) == 1
    assert out[0].entity_class == "social_history"
    assert out[0].attributes.get("category") == "exercise"


def test_a3_reclassifies_diet_lab_to_social_history():
    out = validate_entities([_e("lab_result", "Diet: Low-fat, small frequent meals")])
    assert len(out) == 1
    assert out[0].entity_class == "social_history"
    assert out[0].attributes.get("category") == "diet"


def test_a3_drops_directive_counseling():
    """'Alcohol: avoid alcohol' is counseling, not the patient's social history."""
    out = validate_entities([_e("observation", "Alcohol: avoid alcohol")])
    assert out == []


def test_a3_drops_directive_social_history():
    out = validate_entities([_e("social_history", "Stop smoking", category="tobacco")])
    assert out == []


# ---------------------------------------------------------------------------
# A6 — occupation / hobby / diet free-text leaking into observations
# (real-data regression: LangExtract filed these as `observation` records)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # occupation / job titles
        "business analyst",
        "Software Engineer",
        "Occupation: accountant",
        # hobbies / sports
        "avid tennis player",
        "weight-lifting",
        "weight lifting",
        "plays golf on weekends",
        # free-text diet / food-preference commentary
        "low fodmap diet",
        "spicy and acidic foods",
        "Coffee is fine",
        "vegetarian",
    ],
)
def test_a6_drops_occupation_hobby_diet_observations(text):
    """Occupation, hobbies/sports, and free-text diet/food commentary are not
    clinical observations and must be dropped (they have no measured value)."""
    out = validate_entities([_e("observation", text)])
    assert out == [], f"expected {text!r} dropped, got {_texts(out)!r}"


def test_a6_drops_lifestyle_noise_for_lab_and_vital_classes():
    """The guard applies to every observation-family class, not just `observation`."""
    out = validate_entities(
        [_e("lab_result", "avid tennis player"), _e("vital", "business analyst")]
    )
    assert out == []


def test_a6_drops_verbose_substance_commentary():
    """Free-text substance commentary (no status qualifier, multi-clause) is noise."""
    out = validate_entities([_e("observation", "Only vape no cigarette smoke")])
    assert out == []


# --- what MUST survive: real vitals/labs + concise social-history status ----


@pytest.mark.parametrize(
    "text",
    [
        "BP 120/80 mmHg",
        "Weight 180 lbs",
        "Heart rate 72 bpm",
        "Glucose 95 mg/dL",
        "HbA1c 6.8%",
        "WBC 6.2",
        # named-but-valueless panel tokens stay (downstream B8)
        "CBC",
        "A1c",
        # a real lab whose name overlaps a diet keyword but carries a value
        "Fasting glucose 95 mg/dL",
    ],
)
def test_a6_keeps_real_vitals_and_labs(text):
    out = validate_entities([_e("observation", text)])
    assert _texts(out) == [text], f"expected {text!r} kept, got {_texts(out)!r}"


@pytest.mark.parametrize(
    "text",
    [
        "former smoker",
        "never smoker",
        "non-smoker",
        "current smoker",
        "tobacco use",
        "denies tobacco use",
    ],
)
def test_a6_keeps_concise_smoking_status(text):
    """Concise smoking/tobacco STATUS is legitimate social history — keep it."""
    out = validate_entities([_e("observation", text)])
    assert len(out) == 1, f"expected {text!r} kept, got {_texts(out)!r}"


def test_a6_keeps_alcohol_use_status():
    out = validate_entities([_e("observation", "alcohol use: social")])
    assert len(out) == 1


# ---------------------------------------------------------------------------
# A4 — Drug-class / abbreviation / garbage as medication
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["Go", "PPI", "SSRI", "NSAID", "LDN", "  ", "x7"])
def test_a4_drops_non_drug_medication_tokens(text):
    """Drug-class abbreviations, blanks, and letter+digit garbage ('x7') are
    never specific medication records."""
    out = validate_entities([_e("medication", text)])
    assert out == []


@pytest.mark.parametrize("text", ["D", "K", "A", "C", "E"])
def test_a4_drops_bare_single_letter_supplements(text):
    """Tightened A4 allowlist: a bare single letter is too ambiguous to keep as
    a standalone medication (it causes false positives when a stray single-letter
    token is misclassified as a med). A supplement is recognized ONLY when it is
    vitamin-prefixed ('vitamin D'), a letter+digit form ('D3'), or a named
    multi-character supplement ('folate'). Bare 'D'/'K'/'A'/'C'/'E' are dropped."""
    out = validate_entities([_e("medication", text, medication_group=text)])
    assert out == []


@pytest.mark.parametrize("text", ["Metformin", "Omeprazole", "Lisinopril", "Cefazolin"])
def test_a4_keeps_real_drug_names(text):
    out = validate_entities([_e("medication", text, medication_group=text)])
    assert _texts(out) == [text]


@pytest.mark.parametrize(
    "text",
    [
        # letter+digit vitamin forms the old garbage heuristic wrongly dropped
        "B12", "b12", "D3", "K2", "B6", "B1",
        # vitamin-prefixed forms (the 'vitamin' prefix is decisive)
        "vitamin D", "vitamin B12", "vitamin C",
        # word / compound supplements (multi-character named supplements)
        "folate", "B-complex", "omega-3", "CoQ10", "iron", "magnesium", "multivitamin",
    ],
)
def test_a4_keeps_vitamins_and_supplements(text):
    """Vitamins/supplements (vitamin B12, D3, etc.) are genuine medications and
    must NOT be dropped by the letter+digit garbage rule (recall loss)."""
    out = validate_entities([_e("medication", text, medication_group=text)])
    assert _texts(out) == [text]


# ---------------------------------------------------------------------------
# Provider entities must survive (Agent B attaches them downstream, B2)
# ---------------------------------------------------------------------------


def test_provider_entity_is_kept():
    out = validate_entities([_e("provider", "Dr. Smith", specialty="Cardiology")])
    assert _classes(out) == ["provider"]


# ---------------------------------------------------------------------------
# A6 — within-document duplicate normalization
# ---------------------------------------------------------------------------


def test_a6_normalize_collapses_parenthetical_date_variants():
    a = normalize_entity_text("Cystectomy (2020)")
    b = normalize_entity_text("Cystectomy")
    c = normalize_entity_text("Cystectomy  (Jan 2020)")
    assert a == b == c == "cystectomy"


def test_a6_normalize_collapses_whitespace_and_case():
    assert normalize_entity_text("  Partial   Cystectomy ") == "partial cystectomy"


def test_a6_normalize_strips_trailing_date_token():
    assert normalize_entity_text("Cystectomy 01/2024") == "cystectomy"


def test_a6_normalize_distinguishes_different_names():
    assert normalize_entity_text("Cystectomy") != normalize_entity_text("Colonoscopy")


# ---------------------------------------------------------------------------
# validate_entities is order-preserving and non-mutating for kept entities
# ---------------------------------------------------------------------------


def test_validate_preserves_order_of_survivors():
    entities = [
        _e("condition", "Diabetes", status="active"),
        _e("procedure", "Colonoscopy"),  # dropped (A1)
        _e("medication", "Metformin", medication_group="Metformin"),
    ]
    out = validate_entities(entities)
    assert _texts(out) == ["Diabetes", "Metformin"]
