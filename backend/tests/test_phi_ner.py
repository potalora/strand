"""Eval + tests for NER-based person-name redaction (phi_ner).

The regex PHI scrubber only removes *known* identifiers; arbitrary free-text
names (ordering providers, family members, anyone not in the patient record)
require NER. The central tension is recall vs. over-scrubbing: a general PERSON
NER model will tag eponymous diseases/devices ("Crohn", "Parkinson", "Hodgkin",
"Bell", "Foley") as people and destroy clinical meaning. These cases encode
both requirements — names MUST be redacted, clinical eponyms MUST survive.
"""

from __future__ import annotations

import pytest

# Skip the whole module gracefully if the spaCy model isn't installed.
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_md")
    _MODEL_AVAILABLE = True
except OSError:
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE, reason="en_core_web_md not installed"
)


# (case_id, text, names_that_must_be_redacted, clinical_terms_that_must_survive)
EVAL_CASES = [
    (
        "patient_provider_and_drug",
        "Pedro Otalora was seen today. Prescribed Rifaximin 550mg. "
        "Follow up with Dr. Waseem Ahmad in Gastroenterology.",
        ["Pedro", "Otalora", "Waseem", "Ahmad"],
        ["Rifaximin", "550mg", "Gastroenterology"],
    ),
    (
        "eponym_crohns_provider_redacted",
        "History of Crohn's disease, managed by Dr. Smith.",
        ["Smith"],
        ["Crohn's disease"],
    ),
    (
        "eponym_parkinsons_patient_redacted",
        "Maria Gonzalez presents with Parkinson's disease.",
        ["Maria", "Gonzalez"],
        ["Parkinson's disease"],
    ),
    (
        "multiple_eponyms_survive",
        "Workup for Alzheimer's disease, Hodgkin lymphoma, and Bell's palsy "
        "ordered by Dr. Robert Chen.",
        ["Robert", "Chen"],
        ["Alzheimer's disease", "Hodgkin lymphoma", "Bell's palsy"],
    ),
    (
        "no_names_pure_clinical",
        "Blood pressure 120/80. Hemoglobin A1c 6.8%. Metformin 500mg twice daily.",
        [],
        ["Metformin", "Hemoglobin A1c", "120/80"],
    ),
    (
        "family_member_name",
        "Patient's mother, Susan Miller, has a history of breast cancer.",
        ["Susan", "Miller"],
        ["breast cancer", "mother"],
    ),
    (
        "eponymous_device_foley",
        "Foley catheter placed. Patient John Doe tolerated the procedure well.",
        ["John", "Doe"],
        ["Foley catheter"],
    ),
]


@pytest.mark.parametrize(
    "case_id,text,redact,survive",
    EVAL_CASES,
    ids=[c[0] for c in EVAL_CASES],
)
def test_person_name_redaction(case_id, text, redact, survive):
    from app.services.ai.phi_ner import redact_named_entities

    redacted, report = redact_named_entities(text)

    for name in redact:
        assert name not in redacted, f"[{case_id}] name {name!r} not redacted: {redacted!r}"
    for term in survive:
        assert term in redacted, f"[{case_id}] clinical term {term!r} was destroyed: {redacted!r}"
    if redact:
        assert report.get("names", 0) >= 1


def test_redaction_token_is_generic():
    from app.services.ai.phi_ner import redact_named_entities

    redacted, _ = redact_named_entities("Seen by Dr. Jennifer Lee today.")
    assert "Jennifer" not in redacted
    assert "[NAME]" in redacted


def test_empty_and_nameless_text_unchanged():
    from app.services.ai.phi_ner import redact_named_entities

    assert redact_named_entities("")[0] == ""
    out, report = redact_named_entities("Metformin 500mg twice daily.")
    assert out == "Metformin 500mg twice daily."
    assert report == {}


def test_drug_names_are_not_redacted_as_locations():
    """Regression: the general NER model mislabels drugs as GPE (e.g. 'Rifaximin'
    -> GPE). Location redaction is therefore disabled here — drug names must
    survive the NER pass."""
    from app.services.ai.phi_ner import redact_named_entities

    redacted, report = redact_named_entities("Please try Rifaximin and Metformin.")
    assert "Rifaximin" in redacted
    assert "Metformin" in redacted
    assert report.get("locations", 0) == 0
