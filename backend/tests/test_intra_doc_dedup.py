"""Tests for within-document (intra-doc) deduplication of extracted records (A5).

These are pure-logic tests: they build ``ExtractedEntity`` objects, map them to
HealthRecord dicts via the real ``entity_to_health_record_dict`` builder, and
feed the resulting ``built_records`` list through ``dedup_within_document``.

The PRIME DIRECTIVE under test: collapse intra-document over-extraction WITHOUT
regressing recall. Encounters on clearly different dates and clearly different
drugs must NEVER be merged.
"""

from __future__ import annotations

from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.extraction.intra_doc_dedup import dedup_within_document

RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _build(entities, document_date=None):
    """Map a list of ExtractedEntity to (entity, record_dict) built_records."""
    user_id = uuid4()
    patient_id = uuid4()
    source_file_id = uuid4()
    built: list[tuple[ExtractedEntity, dict | None]] = []
    for ent in entities:
        rec = entity_to_health_record_dict(
            ent, user_id, patient_id, source_file_id, document_date=document_date
        )
        built.append((ent, rec))
    return built


def _records_of_type(built, record_type):
    return [rec for _, rec in built if rec is not None and rec["record_type"] == record_type]


# --- ENCOUNTERS ---------------------------------------------------------------


def test_encounter_fragments_collapse_to_one():
    """A real dated visit + 'UCSF MyChart' + 'After Visit Summary' → 1 encounter."""
    real = ExtractedEntity(
        "encounter",
        "Office Visit",
        {"date": "2024-09-24", "provider": "Dr. Jane Smith", "reason": "abdominal pain"},
    )
    mychart = ExtractedEntity("encounter", "UCSF MyChart", {})
    avs = ExtractedEntity("encounter", "After Visit Summary", {})

    built = _build([real, mychart, avs], document_date=None)
    out = dedup_within_document(built)

    encounters = _records_of_type(out, "encounter")
    assert len(encounters) == 1
    # The survivor is the informative real visit (has a provider + date), not chrome.
    survivor = encounters[0]
    assert survivor["effective_date"] is not None
    participants = survivor["fhir_resource"].get("participant") or []
    assert any((p.get("individual") or {}).get("display") for p in participants)


def test_bare_facility_name_encounter_collapses():
    """A bare practice name ('Peninsula Gastroenterology Group') is chrome, not a visit."""
    real = ExtractedEntity(
        "encounter",
        "Follow-up Visit",
        {"date": "2024-03-15", "provider": "Dr. Lee"},
    )
    facility = ExtractedEntity("encounter", "Peninsula Gastroenterology Group", {})
    dov = ExtractedEntity("encounter", "Date of Visit", {})

    built = _build([real, facility, dov], document_date=None)
    out = dedup_within_document(built)

    encounters = _records_of_type(out, "encounter")
    assert len(encounters) == 1


def test_encounters_on_different_dates_are_kept():
    """NEGATIVE: two encounters on clearly different dates are different visits."""
    jan = ExtractedEntity(
        "encounter", "Office Visit", {"date": "2024-01-10", "provider": "Dr. A"}
    )
    jun = ExtractedEntity(
        "encounter", "Office Visit", {"date": "2024-06-20", "provider": "Dr. B"}
    )

    built = _build([jan, jun], document_date=None)
    out = dedup_within_document(built)

    encounters = _records_of_type(out, "encounter")
    assert len(encounters) == 2


def test_all_dateless_encounters_collapse_to_one():
    """Multiple dateless encounters in one doc are the same (missing-date) visit."""
    a = ExtractedEntity("encounter", "Office Visit", {"provider": "Dr. C", "reason": "cough"})
    b = ExtractedEntity("encounter", "UCSF MyChart", {})

    built = _build([a, b], document_date=None)
    out = dedup_within_document(built)

    assert len(_records_of_type(out, "encounter")) == 1


# --- MEDICATIONS --------------------------------------------------------------


def _force_code(rec, code, system=RXNORM):
    rec["code_value"] = code
    rec["code_system"] = system
    return rec


def test_brand_and_generic_same_rxnorm_collapse():
    """Lexapro + escitalopram resolving to the SAME RxNorm code → 1 medication."""
    lexapro = ExtractedEntity("medication", "Lexapro", {})
    escit = ExtractedEntity("medication", "escitalopram", {})

    built = _build([lexapro, escit])
    # Force both to the same RxNorm code (brand + generic of the same ingredient).
    for _, rec in built:
        _force_code(rec, "321988")
    out = dedup_within_document(built)

    meds = _records_of_type(out, "medication")
    assert len(meds) == 1
    assert meds[0]["code_value"] == "321988"


def test_different_rxnorm_meds_are_kept():
    """NEGATIVE: two different drugs (different RxNorm codes) stay 2 records."""
    a = ExtractedEntity("medication", "Lisinopril", {})
    b = ExtractedEntity("medication", "Metformin", {})

    built = _build([a, b])
    _force_code(built[0][1], "29046")  # lisinopril
    _force_code(built[1][1], "6809")  # metformin
    out = dedup_within_document(built)

    assert len(_records_of_type(out, "medication")) == 2


def test_uncoded_med_casing_variants_collapse():
    """Both uncoded: near-identical names (casing/spacing) collapse via fuzzy match."""
    a = ExtractedEntity("medication", "CoQ10", {})
    b = ExtractedEntity("medication", "coq10", {})

    built = _build([a, b])
    for _, rec in built:
        rec["code_value"] = None
        rec["code_system"] = None
    out = dedup_within_document(built)

    assert len(_records_of_type(out, "medication")) == 1


def test_uncoded_different_drugs_are_kept():
    """NEGATIVE: two clearly different uncoded drugs must not fuzzy-merge."""
    a = ExtractedEntity("medication", "insulin glargine", {})
    b = ExtractedEntity("medication", "insulin aspart", {})

    built = _build([a, b])
    for _, rec in built:
        rec["code_value"] = None
        rec["code_system"] = None
    out = dedup_within_document(built)

    assert len(_records_of_type(out, "medication")) == 2


# --- CROSS-TYPE SAFETY --------------------------------------------------------


def test_non_encounter_non_medication_records_untouched():
    """Conditions/labs pass through unchanged; dedup only touches enc + meds."""
    cond = ExtractedEntity("condition", "Hypertension", {"status": "active"})
    lab = ExtractedEntity("lab_result", "Glucose 95 mg/dL", {})
    other_lab = ExtractedEntity("lab_result", "Sodium 140 mmol/L", {})

    built = _build([cond, lab, other_lab])
    out = dedup_within_document(built)

    assert len([r for _, r in out if r is not None]) == 3


def test_empty_input_is_noop():
    assert dedup_within_document([]) == []
