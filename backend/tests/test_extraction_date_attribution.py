"""Fix #7 — deterministic date-attribution logic test.

Evidence: a real note extracted 165 records but 130 (79%) were dateless even
though dates were present in the note. This test feeds known extracted-entity
dicts (mirroring what LangExtract produced — dates in standard keys, broadened
keys, embedded in attribute values, embedded in entity text, plus entities with
NO date that should fall back to the encounter/document date) into the
entity→FHIR date path and asserts:

  (1) the date-attribution rate clears a defined threshold, and
  (2) NO wrong date is attached — every recovered date equals its known source,
      and an inherently-dateless entity (family history) stays None.

Deterministic: no Gemini, fixed inputs, fixed expected outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

UTC = timezone.utc
# Encounter/document fallback date (post-scrub real notes carry a visit date).
DOC_DATE = datetime(2024, 9, 24, tzinfo=UTC)


def _e(entity_class: str, text: str, attrs: dict) -> ExtractedEntity:
    return ExtractedEntity(entity_class=entity_class, text=text, attributes=attrs)


# (entity, expected_effective_date) — expected is the KNOWN-correct date or None.
CASES: list[tuple[ExtractedEntity, datetime | None]] = [
    # 1. standard key 'onset_date'
    (_e("condition", "Hypertension", {"onset_date": "2024-03-15"}),
     datetime(2024, 3, 15, tzinfo=UTC)),
    # 2. standard key 'date'
    (_e("medication", "Lisinopril 10mg", {"date": "2024-09-24"}),
     datetime(2024, 9, 24, tzinfo=UTC)),
    # 3. broadened key 'collection_date'
    (_e("lab_result", "HbA1c", {"test": "HbA1c", "value": "6.8", "collection_date": "2024-09-20"}),
     datetime(2024, 9, 20, tzinfo=UTC)),
    # 4. broadened key 'result_date', post-scrub MM/YYYY (day dropped) → day=1
    (_e("lab_result", "Glucose", {"result_date": "9/2024"}),
     datetime(2024, 9, 1, tzinfo=UTC)),
    # 5. existing key 'performed_date'
    (_e("procedure", "Colonoscopy", {"performed_date": "2024-08-01"}),
     datetime(2024, 8, 1, tzinfo=UTC)),
    # 6. date embedded in entity TEXT
    (_e("imaging_result", "Chest X-ray performed 2024-07-15", {"procedure_name": "Chest X-ray"}),
     datetime(2024, 7, 15, tzinfo=UTC)),
    # 7. date embedded in a free-text ATTRIBUTE value (month name + year, post-scrub)
    (_e("condition", "Diabetes", {"note": "Diagnosed September 2023"}),
     datetime(2023, 9, 1, tzinfo=UTC)),
    # 8. NO own date → falls back to the encounter/document date
    (_e("vital", "BP 120/80", {"type": "blood pressure"}),
     DOC_DATE),
    # 9. NO own date, social history → fallback
    (_e("social_history", "Former smoker", {"category": "smoking"}),
     DOC_DATE),
    # 10. encounter with its own date
    (_e("encounter", "Office visit", {"date": "2024-09-24", "visit_type": "office"}),
     datetime(2024, 9, 24, tzinfo=UTC)),
    # 11. allergy with NO date → fallback
    (_e("allergy", "Penicillin", {"reaction": "rash"}),
     DOC_DATE),
    # 12. family history with NO date → MUST stay None (visit date would misrepresent it)
    (_e("family_history", "Mother breast cancer", {"relationship": "mother"}),
     None),
]


def _date_of(entity: ExtractedEntity) -> datetime | None:
    rec = entity_to_health_record_dict(
        entity, uuid4(), uuid4(), uuid4(), document_date=DOC_DATE
    )
    assert rec is not None, f"{entity.entity_class} should map to a record"
    return rec["effective_date"]


def test_no_wrong_dates_every_recovered_date_matches_source():
    """Every produced effective_date equals its known-correct value (or None)."""
    for entity, expected in CASES:
        got = _date_of(entity)
        assert got == expected, (
            f"{entity.entity_class} '{entity.text}': expected {expected!r}, got {got!r}"
        )


def test_attribution_rate_clears_threshold():
    """Dated share of dateable entities is high; family history stays dateless."""
    dateable = [(e, exp) for e, exp in CASES if exp is not None]
    dated = [e for e, _ in dateable if _date_of(e) is not None]

    # Every entity that SHOULD have a date gets one.
    assert len(dated) == len(dateable), (
        f"only {len(dated)}/{len(dateable)} dateable entities got a date"
    )

    overall_rate = sum(1 for e, _ in CASES if _date_of(e) is not None) / len(CASES)
    assert overall_rate >= 0.80, f"attribution rate {overall_rate:.2%} below 80%"


def test_family_history_never_gets_fallback_date():
    """A dateless family-history entity must not inherit the visit date."""
    fam = _e("family_history", "Father heart disease", {"relationship": "father"})
    assert _date_of(fam) is None


def test_blood_pressure_text_not_misread_as_date():
    """'120/80' must NOT be parsed as a date; entity falls back to doc date."""
    vital = _e("vital", "BP 120/80", {})
    assert _date_of(vital) == DOC_DATE
