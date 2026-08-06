from __future__ import annotations

import json
import logging

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    AssertionState,
    ClinicalDocumentExtraction,
)
from app.services.local_ai.extraction_validator import validate_clinical_extraction


def _medication(**overrides: object) -> dict[str, object]:
    fact: dict[str, object] = {
        "fact_id": "med-1",
        "name": "Metformin",
        "dose_value": "500",
        "dose_unit": "mg",
        "route": "oral",
        "frequency": "twice daily",
        "status": "active",
        "verbatim": "Metformin 500 mg oral twice daily",
        "page_number": 1,
        "evidence_excerpt": "Metformin 500 mg oral twice daily",
    }
    fact.update(overrides)
    return fact


def _validate(raw: object, page: str = "Metformin 500 mg oral twice daily"):
    return validate_clinical_extraction(raw, pages={1: page}, upload_id="upload-1")


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "conditions",
            {
                "name": "pneumonia",
                "assertion": "negated",
                "verbatim": "No evidence of pneumonia.",
                "page_number": 1,
                "evidence_excerpt": "No evidence of pneumonia.",
            },
            "No evidence of pneumonia.",
        ),
        (
            "conditions",
            {
                "name": "pulmonary embolism",
                "assertion": "uncertain",
                "verbatim": "Possible pulmonary embolism.",
                "page_number": 1,
                "evidence_excerpt": "Possible pulmonary embolism.",
            },
            "Possible pulmonary embolism.",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "mentioned_not_performed",
                "verbatim": "Colonoscopy cancelled.",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy cancelled.",
            },
            "Colonoscopy cancelled.",
        ),
    ],
)
def test_strict_local_rejects_non_promotable_facts(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    """Strict-local validation must not rely only on worker-side filtering."""

    # The shared validator retains supported qualifiers for non-strict callers.
    _validate({category: [fact]}, page=page)

    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*not promotable"):
        validate_clinical_extraction(
            {category: [fact]},
            pages={1: page},
            upload_id="upload-1",
            strict_local=True,
        )


def test_strict_local_rejects_weak_diagnostic_report() -> None:
    raw = {
        "diagnostic_reports": [
            {
                "name": "Laboratory results",
                "assertion": "present",
                "status": "unknown",
                "verbatim": "Laboratory results: potassium 4.1 mmol/L.",
                "page_number": 1,
                "evidence_excerpt": "Laboratory results: potassium 4.1 mmol/L.",
            }
        ]
    }
    page = "Laboratory results: potassium 4.1 mmol/L."

    _validate(raw, page=page)

    with pytest.raises(
        LocalValidationError, match=r"diagnostic_reports\[0\].*not promotable"
    ):
        validate_clinical_extraction(
            raw,
            pages={1: page},
            upload_id="upload-1",
            strict_local=True,
        )


def test_strict_local_keeps_grounded_diagnostic_report() -> None:
    page = "Final CT chest report: no acute findings."
    validated = validate_clinical_extraction(
        {
            "diagnostic_reports": [
                {
                    "name": "CT chest",
                    "findings": "no acute findings",
                    "assertion": "present",
                    "status": "final",
                    "verbatim": page,
                    "page_number": 1,
                    "evidence_excerpt": page,
                }
            ]
        },
        pages={1: page},
        upload_id="upload-1",
        strict_local=True,
    )

    assert validated.diagnostic_reports[0].name == "CT chest"


def test_critical_fact_requires_verbatim_evidence_and_valid_page() -> None:
    raw = {"medications": [_medication(page_number=3)]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*page 3"):
        validate_clinical_extraction(
            raw,
            pages={1: "Metformin 500 mg oral twice daily"},
            upload_id="upload-1",
        )


def test_validator_rejects_numeric_token_boundary_mismatch() -> None:
    raw = {
        "medications": [
            _medication(
                dose_value="50",
                verbatim="Metformin 500 mg oral twice daily",
                evidence_excerpt="Metformin 500 mg oral twice daily",
            )
        ]
    }

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*dose_value"):
        _validate(raw)


def test_validator_rejects_unit_substring_mismatch() -> None:
    raw = {
        "medications": [
            _medication(
                dose_unit="g",
                verbatim="Metformin 500 mg oral twice daily",
                evidence_excerpt="Metformin 500 mg oral twice daily",
            )
        ]
    }

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*dose_unit"):
        _validate(raw)


def test_missing_fact_is_explicitly_unresolved_not_inferred() -> None:
    validated = _validate(
        {
            "medications": [],
            "unresolved_fields": ["medications.dose"],
        },
        page="Dose not stated",
    )

    assert validated.unresolved_fields == ["medications.dose"]
    assert validated.medications == []


def test_whitespace_and_case_are_normalized_only_for_evidence_location() -> None:
    raw = {
        "medications": [
            _medication(
                name="METFORMIN",
                verbatim="METFORMIN 500 MG ORAL TWICE DAILY",
                evidence_excerpt="METFORMIN   500 MG\nORAL TWICE DAILY",
            )
        ]
    }
    page = "Header\nmetformin  \t500 mg oral\n twice daily\nFooter"

    validated = _validate(raw, page=page)

    evidence = validated.evidence[0]
    assert evidence.offset_representation == "whitespace-collapsed-casefold-v1"
    assert evidence.start_offset == len("header ")
    assert evidence.end_offset > evidence.start_offset


def test_cross_page_excerpt_reuse_is_rejected() -> None:
    raw = {"medications": [_medication(page_number=2)]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*page 2"):
        validate_clinical_extraction(
            raw,
            pages={
                1: "Metformin 500 mg oral twice daily",
                2: "Medication list continued",
            },
            upload_id="upload-1",
        )


def test_upload_id_is_required_and_nonempty() -> None:
    with pytest.raises(LocalValidationError, match="upload_id"):
        validate_clinical_extraction(
            {"medications": [_medication()]},
            pages={1: "Metformin 500 mg oral twice daily"},
            upload_id="",
        )


def test_unicode_normalization_is_not_used_to_invent_a_match() -> None:
    composed = "Caf\u00e9ine 10 mg"
    decomposed = "Cafe\u0301ine 10 mg"
    raw = {
        "medications": [
            _medication(
                name="Caf\u00e9ine",
                dose_value="10",
                verbatim=composed,
                evidence_excerpt=composed,
                route=None,
                frequency=None,
            )
        ]
    }

    with pytest.raises(
        LocalValidationError, match=r"medications\[0\].*evidence_excerpt"
    ):
        _validate(raw, page=decomposed)


def test_repeated_identical_fact_is_rejected_before_evidence_assignment() -> None:
    first = _medication(fact_id="med-1")
    second = _medication(fact_id="med-2")
    page = "Metformin 500 mg oral twice daily; Metformin 500 mg oral twice daily"

    with pytest.raises(LocalValidationError, match=r"medications\[1\].*duplicate"):
        _validate({"medications": [first, second]}, page=page)


@pytest.mark.parametrize("field", ["verbatim", "evidence_excerpt"])
def test_critical_fact_rejects_missing_or_blank_evidence(field: str) -> None:
    raw = {"medications": [_medication(**{field: "   "})]}

    with pytest.raises(LocalValidationError, match=rf"medications\[0\].*{field}"):
        _validate(raw)


def test_schema_rejects_extra_fields() -> None:
    raw = {"medications": [_medication(untrusted_payload="do not persist")]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\]"):
        _validate(raw)


def test_schema_rejects_bool_page_number() -> None:
    raw = {"medications": [_medication(page_number=True)]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*page_number"):
        _validate(raw)


def test_schema_rejects_non_string_safety_sensitive_values() -> None:
    raw = {"medications": [_medication(dose_value=500)]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*dose_value"):
        _validate(raw)


def test_schema_rejects_nonfinite_numbers() -> None:
    raw = {"medications": [_medication(confidence=float("nan"))]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*confidence"):
        _validate(raw)


def test_schema_rejects_control_characters() -> None:
    raw = {"medications": [_medication(name="Metformin\u007f")]}

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*name"):
        _validate(raw)


def test_schema_rejects_oversized_strings_before_evidence_search() -> None:
    raw = {"medications": [_medication(evidence_excerpt="x" * 5000)]}

    with pytest.raises(
        LocalValidationError, match=r"medications\[0\].*evidence_excerpt"
    ):
        _validate(raw)


def test_raw_json_size_is_bounded_before_json_parsing() -> None:
    raw = '{"medications":[],"padding":"' + ("x" * 1_048_576) + '"}'

    with pytest.raises(LocalValidationError, match="JSON exceeds size limit"):
        _validate(raw)


def test_schema_rejects_oversized_category_arrays() -> None:
    raw = {"medications": [_medication(fact_id=f"med-{index}") for index in range(257)]}

    with pytest.raises(LocalValidationError, match="medications"):
        _validate(raw)


def test_schema_rejects_coercive_tuple_lists() -> None:
    with pytest.raises(LocalValidationError, match="medications"):
        _validate({"medications": (_medication(),)})

    care_plan = {
        "title": "Care plan",
        "plan_items": ("monitor blood pressure",),
        "verbatim": "Care plan monitor blood pressure",
        "page_number": 1,
        "evidence_excerpt": "Care plan monitor blood pressure",
    }
    with pytest.raises(LocalValidationError, match=r"care_plans\[0\].*plan_items"):
        _validate(
            {"care_plans": [care_plan]},
            page="Care plan monitor blood pressure",
        )


def test_strict_json_rejects_duplicate_keys() -> None:
    raw = (
        '{"medications":[{"name":"Metformin","name":"Lisinopril",'
        '"verbatim":"Metformin","page_number":1,"evidence_excerpt":"Metformin"}]}'
    )

    with pytest.raises(LocalValidationError, match="duplicate JSON key"):
        _validate(raw, page="Metformin")


def test_strict_json_accepts_at_most_one_complete_code_fence() -> None:
    payload = json.dumps({"medications": [_medication()]})

    validated = _validate(f"```json\n{payload}\n```")
    assert validated.schema_version == CLINICAL_EXTRACTION_SCHEMA_VERSION

    with pytest.raises(LocalValidationError, match="JSON wrapper"):
        _validate(f"Result:\n```json\n{payload}\n```")
    with pytest.raises(LocalValidationError, match="JSON wrapper"):
        _validate(f"```json\n{payload}\n```\n```json\n{payload}\n```")
    with pytest.raises(LocalValidationError, match="JSON wrapper"):
        _validate(f"```json\n{payload}")


def test_strict_json_rejects_comments_trailing_content_and_constants() -> None:
    payload = json.dumps({"medications": [_medication()]})

    for raw in (
        f"{payload} trailing",
        f"{payload}\n{payload}",
        '{"medications": [], // comment\n"conditions": []}',
        '{"medications": [], "score": NaN}',
        '{"medications": [], "score": Infinity}',
    ):
        with pytest.raises(LocalValidationError):
            _validate(raw)


def test_one_malformed_fact_fails_the_whole_category() -> None:
    raw = {
        "medications": [
            _medication(fact_id="med-1"),
            _medication(
                fact_id="med-2",
                dose_value="50",
                verbatim="Metformin 500 mg oral twice daily",
            ),
        ]
    }

    with pytest.raises(LocalValidationError, match=r"medications\[1\].*dose_value"):
        _validate(raw)


def test_validator_rejects_date_tokens_absent_from_source() -> None:
    raw = {
        "conditions": [
            {
                "name": "Hypertension",
                "assertion": "present",
                "date": "2025-01-02",
                "verbatim": "Hypertension documented 2024-01-02",
                "page_number": 1,
                "evidence_excerpt": "Hypertension documented 2024-01-02",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*date"):
        _validate(raw, page="Hypertension documented 2024-01-02")


def test_validator_grounds_normalized_value_without_numeric_substrings() -> None:
    raw = {
        "labs": [
            {
                "name": "Glucose",
                "value": "500",
                "normalized_value": "50",
                "normalization_method": "decimal",
                "normalization_version": "1",
                "verbatim": "Glucose 500 mg/dL",
                "page_number": 1,
                "evidence_excerpt": "Glucose 500 mg/dL",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*normalization"):
        _validate(raw, page="Glucose 500 mg/dL")


def test_duplicate_fact_identifiers_are_rejected() -> None:
    raw = {
        "medications": [
            _medication(fact_id="duplicate"),
            _medication(fact_id="duplicate"),
        ]
    }

    with pytest.raises(LocalValidationError, match="fact_id"):
        _validate(raw)


def test_assertion_state_is_preserved_for_negation_and_family_history() -> None:
    page = "No diabetes. Family history: mother had colon cancer."
    raw = {
        "conditions": [
            {
                "fact_id": "condition-1",
                "name": "diabetes",
                "assertion": "negated",
                "verbatim": "No diabetes",
                "page_number": 1,
                "evidence_excerpt": "No diabetes",
            },
            {
                "fact_id": "condition-2",
                "name": "colon cancer",
                "assertion": "family_history",
                "relationship": "mother",
                "verbatim": "Family history: mother had colon cancer",
                "page_number": 1,
                "evidence_excerpt": "Family history: mother had colon cancer",
            },
        ]
    }

    validated = _validate(raw, page=page)

    assert [fact.assertion.value for fact in validated.conditions] == [
        "negated",
        "family_history",
    ]


def test_no_recurrence_is_an_explicit_subject_negation() -> None:
    text = "No recurrence of palpitations."
    validated = _validate(
        {
            "conditions": [
                {
                    "name": "palpitations",
                    "assertion": "negated",
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert validated.conditions[0].assertion.value == "negated"


def test_omitted_assertion_cannot_promote_negated_allergy() -> None:
    text = "No penicillin allergy"
    raw = {
        "allergies": [
            {
                "substance": "penicillin",
                "verbatim": text,
                "page_number": 1,
                "evidence_excerpt": text,
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"allergies\[0\].*assertion"):
        _validate(raw, page=text)


def test_educational_fact_is_rejected_instead_of_promoted() -> None:
    text = "Patient education: diabetes may cause neuropathy."
    raw = {
        "conditions": [
            {
                "name": "neuropathy",
                "assertion": "present",
                "verbatim": text,
                "page_number": 1,
                "evidence_excerpt": text,
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*assertion"):
        _validate(raw, page=text)


def test_performed_procedure_requires_source_performance_evidence() -> None:
    raw = {
        "procedures": [
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"procedures\[0\].*assertion"):
        _validate(raw, page="Colonoscopy")


def test_billed_procedure_line_item_supports_mentioned_not_performed() -> None:
    raw = {
        "procedures": [
            {
                "name": "Colonoscopy",
                "assertion": "mentioned_not_performed",
                "verbatim": "Colonoscopy",
                "page_number": 1,
                "evidence_excerpt": "Authorization for Colonoscopy",
            }
        ]
    }

    result = _validate(raw, page="Authorization for Colonoscopy")

    assert result.procedures[0].assertion == AssertionState.MENTIONED_NOT_PERFORMED


def test_billed_context_does_not_forbid_dated_present_procedure() -> None:
    raw = {
        "procedures": [
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "date": "2024-03-01",
                "verbatim": "Colonoscopy on 2024-03-01",
                "page_number": 1,
                "evidence_excerpt": "Billed Colonoscopy on 2024-03-01",
            }
        ]
    }

    result = _validate(raw, page="Billed Colonoscopy on 2024-03-01")

    assert result.procedures[0].assertion == AssertionState.PRESENT


def test_billing_form_page_supports_mentioned_not_performed_without_excerpt_wording() -> None:
    raw = {
        "procedures": [
            {
                "name": "Colonoscopy",
                "assertion": "mentioned_not_performed",
                "verbatim": "Colonoscopy",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy",
            }
        ]
    }

    result = _validate(raw, page="Place of Service: 11. Payer: Aetna. Colonoscopy")

    assert result.procedures[0].assertion == AssertionState.MENTIONED_NOT_PERFORMED


def test_billed_present_procedure_without_support_still_fails_closed() -> None:
    raw = {
        "procedures": [
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy",
                "page_number": 1,
                "evidence_excerpt": "Authorization for Colonoscopy",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"procedures\[0\].*assertion"):
        _validate(raw, page="Authorization for Colonoscopy")


def test_evidence_ids_are_stable_and_change_with_relevant_inputs() -> None:
    raw = {"medications": [_medication()]}

    one = _validate(raw)
    two = _validate(raw)
    other_upload = validate_clinical_extraction(
        raw,
        pages={1: "Metformin 500 mg oral twice daily"},
        upload_id="upload-2",
    )

    assert one.evidence[0].id == two.evidence[0].id
    assert one.medications[0].evidence_id == one.evidence[0].id
    assert one.evidence[0].id != other_upload.evidence[0].id


def test_evidence_id_changes_with_page_location_and_excerpt_hash() -> None:
    raw = {"medications": [_medication()]}
    base = _validate(raw)
    later_page = validate_clinical_extraction(
        {"medications": [_medication(page_number=2)]},
        pages={2: "Metformin 500 mg oral twice daily"},
        upload_id="upload-1",
    )
    expanded_excerpt = validate_clinical_extraction(
        {
            "medications": [
                _medication(
                    evidence_excerpt="Medication: Metformin 500 mg oral twice daily"
                )
            ]
        },
        pages={1: "Medication: Metformin 500 mg oral twice daily"},
        upload_id="upload-1",
    )

    assert base.evidence[0].id != later_page.evidence[0].id
    assert base.evidence[0].id != expanded_excerpt.evidence[0].id


def test_errors_and_logs_do_not_contain_phi_or_clinical_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canaries = [
        "Pedro Secret",
        "01/02/1970",
        "Metformin",
        "500",
        "model prompt",
    ]
    raw = {
        "medications": [
            _medication(
                name=canaries[0],
                dose_value=canaries[3],
                verbatim="Pedro Secret 500 mg model prompt",
                evidence_excerpt="Pedro Secret 500 mg model prompt",
                page_number=9,
            )
        ]
    }

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LocalValidationError) as exc_info:
            validate_clinical_extraction(
                raw,
                pages={1: f"{canaries[1]} {canaries[2]}"},
                upload_id="upload-phi",
            )

    combined = f"{exc_info.value} {caplog.text}"
    assert not any(canary in combined for canary in canaries)


def test_schema_covers_every_approved_clinical_category() -> None:
    fields = ClinicalDocumentExtraction.model_fields

    assert {
        "medications",
        "conditions",
        "procedures",
        "labs",
        "allergies",
        "encounters",
        "immunizations",
        "vital_signs",
        "diagnostic_reports",
        "care_plans",
        "unresolved_fields",
        "rejected_fields",
    } <= fields.keys()


@pytest.mark.parametrize(
    ("extracted", "source"),
    [
        ("-5", "5"),
        ("5", "<5"),
        ("mg", "mg/dL"),
        ("1", "1e3"),
        ("120", "120/80"),
        ("5", "< 5"),
        ("5", "\u22125"),
        ("5", "~5"),
        ("1", "1 \u00d7 10^3"),
        ("120", "120 / 80"),
        ("mg", "mg / dL"),
        ("mg", "mg\u00b7dL\u207b\u00b9"),
        ("-5", "< -5"),
        ("-5", "\u2264 \u22125"),
        ("-80", "120 / -80"),
        ("70", "70\u201399"),
        ("1", "1\u00d710\u00b3/uL"),
        ("mg", "mg dL\u207b\u00b9"),
    ],
)
def test_grounding_preserves_clinical_operators_and_compound_units(
    extracted: str,
    source: str,
) -> None:
    raw = {
        "labs": [
            {
                "name": "Glucose",
                "value": extracted if extracted != "mg" else "5",
                "unit": extracted if extracted == "mg" else None,
                "verbatim": f"Glucose {source}",
                "page_number": 1,
                "evidence_excerpt": f"Glucose {source}",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\]"):
        _validate(raw, page=f"Glucose {source}")


@pytest.mark.parametrize("unit", ["mL/min", "mmol/L"])
def test_unit_cannot_truncate_a_normalized_denominator(unit: str) -> None:
    source_unit = f"{unit}/1.73 m2"
    raw = {
        "labs": [
            {
                "name": "eGFR",
                "value": "50",
                "unit": unit,
                "verbatim": f"eGFR 50 {source_unit}",
                "page_number": 1,
                "evidence_excerpt": f"eGFR 50 {source_unit}",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*unit"):
        _validate(raw, page=f"eGFR 50 {source_unit}")


@pytest.mark.parametrize(
    ("normalized", "source"),
    [
        ("-5", "5"),
        ("5", "<5"),
        ("mg", "mg/dL"),
        ("1", "1e3"),
        ("120", "120/80"),
        ("5", "< 5"),
        ("5", "\u22125"),
        ("5", "~5"),
        ("1", "1 \u00d7 10^3"),
        ("120", "120 / 80"),
        ("mg", "mg / dL"),
        ("mg", "mg\u00b7dL\u207b\u00b9"),
        ("-5", "< -5"),
        ("-5", "\u2264 \u22125"),
        ("-80", "120 / -80"),
        ("70", "70\u201399"),
        ("1", "1\u00d710\u00b3/uL"),
        ("mg", "mg dL\u207b\u00b9"),
    ],
)
def test_normalized_grounding_preserves_clinical_lexemes(
    normalized: str,
    source: str,
) -> None:
    raw = {
        "labs": [
            {
                "name": "Glucose",
                "normalized_value": normalized,
                "normalization_method": "identity",
                "normalization_version": "1",
                "verbatim": f"Glucose {source}",
                "page_number": 1,
                "evidence_excerpt": f"Glucose {source}",
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*normalization"):
        _validate(raw, page=f"Glucose {source}")


@pytest.mark.parametrize(
    "raw",
    [
        {"Pedro Secret Name": []},
        {"patient": {"Pedro Secret Name": "value"}},
        {"medications": [_medication(**{"Pedro Secret Name": "value"})]},
    ],
)
def test_unknown_schema_keys_are_not_echoed_in_errors(raw: dict[str, object]) -> None:
    canary = "Pedro Secret Name"

    with pytest.raises(LocalValidationError) as exc_info:
        _validate(raw)

    assert canary not in str(exc_info.value)


@pytest.mark.parametrize(
    "patient",
    [
        {"name": "Fabricated Person"},
        {"date_of_birth": "1901-02-03"},
    ],
)
def test_patient_identity_must_be_grounded_on_a_page(
    patient: dict[str, str],
) -> None:
    with pytest.raises(LocalValidationError, match="patient"):
        _validate({"patient": patient}, page="No patient identity appears here")


def test_grounded_patient_identity_is_retained() -> None:
    validated = _validate(
        {"patient": {"name": "Pedro Patient", "date_of_birth": "1970-01-02"}},
        page="Pedro Patient was born 1970-01-02.",
    )

    assert validated.patient is not None
    assert validated.patient.name == "Pedro Patient"
    assert validated.patient.date_of_birth == "1970-01-02"


def test_evidence_id_changes_when_grounded_field_paths_change() -> None:
    base = {
        "medications": [
            _medication(
                dose_value=None,
                dose_unit=None,
                verbatim="Metformin 500 mg oral twice daily",
                evidence_excerpt="Metformin 500 mg oral twice daily",
            )
        ]
    }
    expanded = {
        "medications": [
            _medication(
                dose_value="500",
                dose_unit="mg",
                verbatim="Metformin 500 mg oral twice daily",
                evidence_excerpt="Metformin 500 mg oral twice daily",
            )
        ]
    }

    base_validated = _validate(base, page="Metformin 500 mg oral twice daily")
    expanded_validated = _validate(
        expanded,
        page="Metformin 500 mg oral twice daily",
    )

    assert base_validated.evidence[0].field_paths == [
        "medications[0].frequency",
        "medications[0].name",
        "medications[0].route",
        "medications[0].status",
    ]
    assert expanded_validated.evidence[0].field_paths == [
        "medications[0].dose_unit",
        "medications[0].dose_value",
        "medications[0].frequency",
        "medications[0].name",
        "medications[0].route",
        "medications[0].status",
    ]
    assert base_validated.evidence[0].id != expanded_validated.evidence[0].id


def test_family_history_and_negated_procedure_mislabels_fail_closed() -> None:
    cases = [
        (
            {
                "allergies": [
                    {
                        "substance": "penicillin",
                        "assertion": "present",
                        "status": "active",
                        "verbatim": "Mother allergic to penicillin",
                        "page_number": 1,
                        "evidence_excerpt": "Mother allergic to penicillin",
                    }
                ]
            },
            "Mother allergic to penicillin",
        ),
        (
            {
                "conditions": [
                    {
                        "name": "colon cancer",
                        "assertion": "uncertain",
                        "verbatim": "Mother had colon cancer",
                        "page_number": 1,
                        "evidence_excerpt": "Mother had colon cancer",
                    }
                ]
            },
            "Mother had colon cancer",
        ),
        (
            {
                "procedures": [
                    {
                        "name": "colonoscopy",
                        "assertion": "present",
                        "date": "2024-01-02",
                        "verbatim": "No colonoscopy 2024-01-02",
                        "page_number": 1,
                        "evidence_excerpt": "No colonoscopy 2024-01-02",
                    }
                ]
            },
            "No colonoscopy 2024-01-02",
        ),
        (
            {
                "conditions": [
                    {
                        "name": "colon cancer",
                        "assertion": "present",
                        "verbatim": "FHx: colon cancer",
                        "page_number": 1,
                        "evidence_excerpt": "FHx: colon cancer",
                    }
                ]
            },
            "FHx: colon cancer",
        ),
        (
            {
                "conditions": [
                    {
                        "name": "Diabetes",
                        "assertion": "present",
                        "verbatim": "Diabetes absent",
                        "page_number": 1,
                        "evidence_excerpt": "Diabetes absent",
                    }
                ]
            },
            "Diabetes absent",
        ),
        (
            {
                "procedures": [
                    {
                        "name": "Colonoscopy",
                        "assertion": "present",
                        "date": "2024-01-02",
                        "verbatim": "Colonoscopy cancelled 2024-01-02",
                        "page_number": 1,
                        "evidence_excerpt": "Colonoscopy cancelled 2024-01-02",
                    }
                ]
            },
            "Colonoscopy cancelled 2024-01-02",
        ),
    ]

    for raw, page in cases:
        with pytest.raises(LocalValidationError, match="assertion"):
            _validate(raw, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "medications",
            _medication(
                status="active",
                verbatim="Metformin stopped",
                evidence_excerpt="Metformin stopped",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Metformin stopped",
        ),
        (
            "allergies",
            {
                "substance": "Penicillin",
                "assertion": "present",
                "status": "active",
                "verbatim": "Penicillin allergy resolved",
                "page_number": 1,
                "evidence_excerpt": "Penicillin allergy resolved",
            },
            "Penicillin allergy resolved",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Office visit planned",
                "page_number": 1,
                "evidence_excerpt": "Office visit planned",
            },
            "Office visit planned",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine entered in error",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine entered in error",
            },
            "Influenza vaccine entered in error",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest preliminary",
                "page_number": 1,
                "evidence_excerpt": "CT chest preliminary",
            },
            "CT chest preliminary",
        ),
        (
            "care_plans",
            {
                "title": "Care plan",
                "status": "active",
                "verbatim": "Care plan inactive",
                "page_number": 1,
                "evidence_excerpt": "Care plan inactive",
            },
            "Care plan inactive",
        ),
    ],
)
def test_lifecycle_claims_cannot_contradict_source(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*status"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "conditions",
            {
                "name": "Diabetes",
                "assertion": "present",
                "verbatim": "Diabetes",
                "page_number": 1,
                "evidence_excerpt": "Diabetes absent",
            },
            "Diabetes absent",
        ),
        (
            "conditions",
            {
                "name": "colon cancer",
                "assertion": "present",
                "verbatim": "colon cancer",
                "page_number": 1,
                "evidence_excerpt": "FHx: colon cancer",
            },
            "FHx: colon cancer",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "date": "2024-01-02",
                "verbatim": "Colonoscopy 2024-01-02",
                "page_number": 1,
                "evidence_excerpt": "Cancelled: Colonoscopy 2024-01-02",
            },
            "Cancelled: Colonoscopy 2024-01-02",
        ),
    ],
)
def test_assertion_guards_cover_the_complete_grounded_excerpt(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*assertion"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "medications",
            _medication(
                status="active",
                verbatim="Patient is not taking Metformin",
                evidence_excerpt="Patient is not taking Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Patient is not taking Metformin",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Office visit cancelled",
                "page_number": 1,
                "evidence_excerpt": "Office visit cancelled",
            },
            "Office visit cancelled",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine not administered",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine not administered",
            },
            "Influenza vaccine not administered",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest not final",
                "page_number": 1,
                "evidence_excerpt": "CT chest not final",
            },
            "CT chest not final",
        ),
        (
            "care_plans",
            {
                "title": "Care plan",
                "status": "active",
                "verbatim": "Care plan not active",
                "page_number": 1,
                "evidence_excerpt": "Care plan not active",
            },
            "Care plan not active",
        ),
    ],
)
def test_negated_lifecycle_phrases_cannot_be_promoted(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*status"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "medications",
            _medication(
                status="active",
                name="Metformin",
                verbatim="Mother takes Metformin",
                evidence_excerpt="Mother takes Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Mother takes Metformin",
        ),
        (
            "labs",
            {
                "name": "glucose",
                "value": "95",
                "unit": "mg/dL",
                "verbatim": "Mother's glucose 95 mg/dL",
                "page_number": 1,
                "evidence_excerpt": "Mother's glucose 95 mg/dL",
            },
            "Mother's glucose 95 mg/dL",
        ),
        (
            "vital_signs",
            {
                "name": "Blood pressure",
                "value": "120/80",
                "unit": "mmHg",
                "verbatim": "Blood pressure 120/80 mmHg was not recorded",
                "page_number": 1,
                "evidence_excerpt": "Blood pressure 120/80 mmHg was not recorded",
            },
            "Blood pressure 120/80 mmHg was not recorded",
        ),
    ],
)
def test_family_and_negation_guards_cover_every_clinical_category(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*assertion"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "medications",
            _medication(
                status="active",
                verbatim="Patient never takes Metformin",
                evidence_excerpt="Patient never takes Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Patient never takes Metformin",
        ),
        (
            "medications",
            _medication(
                status="active",
                verbatim="Patient is not currently taking Metformin",
                evidence_excerpt="Patient is not currently taking Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Patient is not currently taking Metformin",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Office visit was not completed",
                "page_number": 1,
                "evidence_excerpt": "Office visit was not completed",
            },
            "Office visit was not completed",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine not yet administered",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine not yet administered",
            },
            "Influenza vaccine not yet administered",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest is not yet final",
                "page_number": 1,
                "evidence_excerpt": "CT chest is not yet final",
            },
            "CT chest is not yet final",
        ),
    ],
)
def test_modifier_based_lifecycle_negations_cannot_be_promoted(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*status"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Office visit was never completed",
                "page_number": 1,
                "evidence_excerpt": "Office visit was never completed",
            },
            "Office visit was never completed",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine was never received",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine was never received",
            },
            "Influenza vaccine was never received",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest report was never finalized",
                "page_number": 1,
                "evidence_excerpt": "CT chest report was never finalized",
            },
            "CT chest report was never finalized",
        ),
        (
            "care_plans",
            {
                "title": "Care plan",
                "status": "active",
                "verbatim": "Care plan is closed",
                "page_number": 1,
                "evidence_excerpt": "Care plan is closed",
            },
            "Care plan is closed",
        ),
    ],
)
def test_never_and_closed_lifecycle_states_cannot_be_promoted(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*status"):
        _validate({category: [fact]}, page=page)


@pytest.mark.parametrize(
    "relative_phrase",
    [
        "daughter",
        "daughters",
        "son",
        "son's",
        "child",
        "children",
        "husband",
        "spouse",
        "wife",
        "partner",
    ],
)
def test_close_family_terms_fail_closed_without_family_assertion(
    relative_phrase: str,
) -> None:
    text = f"Patient's {relative_phrase} takes Metformin"
    fact = _medication(
        status="active",
        verbatim=text,
        evidence_excerpt=text,
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*assertion"):
        _validate({"medications": [fact]}, page=text)


@pytest.mark.parametrize("relative", ["mom", "dad", "grandma", "grandpa"])
def test_informal_family_terms_fail_closed_across_assertionless_facts(
    relative: str,
) -> None:
    text = f"{relative} takes Metformin"
    fact = _medication(
        status="active",
        verbatim=text,
        evidence_excerpt=text,
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*assertion"):
        _validate({"medications": [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                verbatim="No Metformin",
                evidence_excerpt="No Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "No Metformin",
        ),
        (
            "encounters",
            {
                "name": "office visit",
                "status": "finished",
                "verbatim": "No office visit",
                "page_number": 1,
                "evidence_excerpt": "No office visit",
            },
            "No office visit",
        ),
        (
            "immunizations",
            {
                "name": "influenza vaccine",
                "status": "completed",
                "verbatim": "No influenza vaccine",
                "page_number": 1,
                "evidence_excerpt": "No influenza vaccine",
            },
            "No influenza vaccine",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "No CT chest",
                "page_number": 1,
                "evidence_excerpt": "No CT chest",
            },
            "No CT chest",
        ),
        (
            "care_plans",
            {
                "title": "diabetes care plan",
                "status": "active",
                "verbatim": "No diabetes care plan",
                "page_number": 1,
                "evidence_excerpt": "No diabetes care plan",
            },
            "No diabetes care plan",
        ),
    ],
)
def test_generic_subject_negation_fails_closed_for_promoted_categories(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*assertion"):
        _validate({category: [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                verbatim="Metformin listed",
                evidence_excerpt="Metformin listed",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
            ),
            "Metformin listed",
        ),
        (
            "immunizations",
            {
                "name": "influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine ordered",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine ordered",
            },
            "Influenza vaccine ordered",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest proposed",
                "page_number": 1,
                "evidence_excerpt": "CT chest proposed",
            },
            "CT chest proposed",
        ),
        (
            "care_plans",
            {
                "title": "diabetes care plan",
                "status": "active",
                "verbatim": "Diabetes care plan planned",
                "page_number": 1,
                "evidence_excerpt": "Diabetes care plan planned",
            },
            "Diabetes care plan planned",
        ),
    ],
)
def test_promoted_lifecycle_state_requires_positive_source_support(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*status"):
        _validate({category: [fact]}, page=text)


def test_final_report_can_contain_a_negated_finding() -> None:
    text = "Final CT chest showed no acute findings"
    validated = _validate(
        {
            "diagnostic_reports": [
                {
                    "name": "CT chest",
                    "findings": "no acute findings",
                    "assertion": "present",
                    "status": "final",
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert validated.diagnostic_reports[0].status.value == "final"


@pytest.mark.parametrize(
    "valid_date",
    [
        "2024",
        "2024-02",
        "2024-02-29",
        "02/29/2024",
        "February 2024",
        "February 29, 2024",
        "2024-02-29T14:30:00Z",
    ],
)
def test_semantically_valid_supported_dates_are_accepted(valid_date: str) -> None:
    text = f"Hypertension documented {valid_date}"
    validated = _validate(
        {
            "conditions": [
                {
                    "name": "Hypertension",
                    "assertion": "present",
                    "date": valid_date,
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert validated.conditions[0].date == valid_date


@pytest.mark.parametrize(
    "invalid_date",
    [
        "99/99/9999",
        "2023-02-29",
        "2024-13",
        "February 30, 2024",
        "2024-02-29T25:00:00Z",
        "tomorrow",
    ],
)
def test_impossible_or_unsupported_dates_fail_before_fhir(invalid_date: str) -> None:
    text = f"Hypertension documented {invalid_date}"
    raw = {
        "conditions": [
            {
                "name": "Hypertension",
                "assertion": "present",
                "date": invalid_date,
                "verbatim": text,
                "page_number": 1,
                "evidence_excerpt": text,
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*date"):
        _validate(raw, page=text)


def test_deep_json_inputs_fail_closed_without_leaking_recursion_errors() -> None:
    nested_mapping: dict[str, object] = {}
    cursor = nested_mapping
    for _ in range(1500):
        child: dict[str, object] = {}
        cursor["nested"] = child
        cursor = child

    nested_json = '{"nested":' * 1500 + "{}" + "}" * 1500

    for raw in (nested_mapping, nested_json):
        with pytest.raises(LocalValidationError, match="extraction"):
            _validate(raw, page="No clinical content")


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "conditions",
            {
                "name": "diabetes",
                "assertion": "present",
                "verbatim": "Possible diabetes",
                "page_number": 1,
                "evidence_excerpt": "Possible diabetes",
            },
            "Possible diabetes",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                dose_value="500",
                dose_unit="mg",
                frequency="daily",
                status="active",
                verbatim="Consider Metformin 500 mg daily",
                evidence_excerpt="Consider Metformin 500 mg daily",
                route=None,
            ),
            "Consider Metformin 500 mg daily",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "date": "2024-01-02",
                "verbatim": "Colonoscopy considered 2024-01-02",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy considered 2024-01-02",
            },
            "Colonoscopy considered 2024-01-02",
        ),
        (
            "immunizations",
            {
                "name": "Influenza",
                "status": "completed",
                "verbatim": "Influenza: No vaccine was given",
                "page_number": 1,
                "evidence_excerpt": "Influenza: No vaccine was given",
            },
            "Influenza: No vaccine was given",
        ),
        (
            "encounters",
            {
                "name": "Annual physical",
                "status": "finished",
                "verbatim": "Annual physical: no office visit was completed",
                "page_number": 1,
                "evidence_excerpt": "Annual physical: no office visit was completed",
            },
            "Annual physical: no office visit was completed",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "verbatim": "CT chest: no report was finalized",
                "page_number": 1,
                "evidence_excerpt": "CT chest: no report was finalized",
            },
            "CT chest: no report was finalized",
        ),
        (
            "care_plans",
            {
                "title": "Diabetes management",
                "status": "active",
                "verbatim": "Diabetes management: no care plan was active",
                "page_number": 1,
                "evidence_excerpt": "Diabetes management: no care plan was active",
            },
            "Diabetes management: no care plan was active",
        ),
    ],
)
def test_promoted_fact_requires_subject_scoped_positive_evidence(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


@pytest.mark.parametrize(
    ("assertion", "text"),
    [
        ("negated", "Diabetes documented"),
        ("family_history", "Diabetes documented"),
        ("uncertain", "Diabetes documented"),
    ],
)
def test_non_present_assertion_requires_reciprocal_source_support(
    assertion: str,
    text: str,
) -> None:
    fact = {
        "name": "Diabetes",
        "assertion": assertion,
        "verbatim": text,
        "page_number": 1,
        "evidence_excerpt": text,
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*assertion"):
        _validate({"conditions": [fact]}, page=text)


def test_uncertain_assertion_is_accepted_when_source_supports_it() -> None:
    text = "Possible diabetes"
    validated = _validate(
        {
            "conditions": [
                {
                    "name": "diabetes",
                    "assertion": "uncertain",
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert validated.conditions[0].assertion.value == "uncertain"


def test_decimal_dose_does_not_break_subject_scoped_lifecycle_evidence() -> None:
    text = "Metformin 0.5 mg daily"
    validated = _validate(
        {
            "medications": [
                _medication(
                    dose_value="0.5",
                    dose_unit="mg",
                    route=None,
                    frequency="daily",
                    status="active",
                    verbatim=text,
                    evidence_excerpt=text,
                )
            ]
        },
        page=text,
    )

    assert validated.medications[0].status.value == "active"


@pytest.mark.parametrize(
    ("category", "field_name", "value", "text"),
    [
        ("medications", "dose_value", "500 mg", "Metformin active 500 mg"),
        ("immunizations", "dose", "0.5 mL", "Influenza vaccine given 0.5 mL"),
        ("labs", "value", "95 mg/dL", "Glucose 95 mg/dL"),
        ("vital_signs", "value", "72 bpm", "Heart rate 72 bpm"),
    ],
)
def test_numeric_value_fields_reject_embedded_units(
    category: str,
    field_name: str,
    value: str,
    text: str,
) -> None:
    facts: dict[str, dict[str, object]] = {
        "medications": _medication(
            dose_value=value,
            dose_unit=None,
            route=None,
            frequency=None,
            status="active",
            verbatim=text,
            evidence_excerpt=text,
        ),
        "immunizations": {
            "name": "Influenza vaccine",
            "status": "completed",
            "dose": value,
            "verbatim": text,
            "page_number": 1,
            "evidence_excerpt": text,
        },
        "labs": {
            "name": "Glucose",
            "value": value,
            "verbatim": text,
            "page_number": 1,
            "evidence_excerpt": text,
        },
        "vital_signs": {
            "name": "Heart rate",
            "value": value,
            "verbatim": text,
            "page_number": 1,
            "evidence_excerpt": text,
        },
    }

    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*{field_name}"):
        _validate({category: [facts[category]]}, page=text)


def test_unit_cannot_truncate_a_per_denominator() -> None:
    text = "eGFR 50 mL/min per 1.73 m2"
    raw = {
        "labs": [
            {
                "name": "eGFR",
                "value": "50",
                "unit": "mL/min",
                "verbatim": text,
                "page_number": 1,
                "evidence_excerpt": text,
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*unit"):
        _validate(raw, page=text)


def test_corroborating_clinical_facts_on_different_pages_keep_both_evidence() -> None:
    first = _medication(
        fact_id="med-1",
        verbatim="Metformin active",
        evidence_excerpt="Metformin active",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
        status="active",
    )
    second = {
        **first,
        "fact_id": "med-2",
        "page_number": 2,
        "verbatim": "Current medication: Metformin",
        "evidence_excerpt": "Current medication: Metformin",
    }

    result = validate_clinical_extraction(
        {"medications": [first, second]},
        pages={
            1: "Metformin active",
            2: "Current medication: Metformin",
        },
        upload_id="upload-1",
    )

    assert [fact.page_number for fact in result.medications] == [1, 2]
    assert [evidence.page_number for evidence in result.evidence] == [1, 2]


def test_duplicate_clinical_facts_on_the_same_evidence_span_are_rejected() -> None:
    first = _medication(
        fact_id="med-1",
        verbatim="Metformin active",
        evidence_excerpt="Metformin active",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
        status="active",
    )
    second = {**first, "fact_id": "med-2"}

    with pytest.raises(LocalValidationError, match=r"medications\[1\].*duplicate"):
        validate_clinical_extraction(
            {"medications": [first, second]},
            pages={1: "Metformin active"},
            upload_id="upload-1",
        )


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "conditions",
            {
                "name": "diabetes",
                "assertion": "present",
                "verbatim": "Diabetes confirmed",
                "page_number": 1,
                "evidence_excerpt": "Possible diabetes. Diabetes confirmed",
            },
            "Possible diabetes. Diabetes confirmed",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                status="active",
                verbatim="Metformin active",
                evidence_excerpt="Consider Metformin. Metformin active",
            ),
            "Consider Metformin. Metformin active",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Office visit completed",
                "page_number": 1,
                "evidence_excerpt": "Office visit planned. Office visit completed",
            },
            "Office visit planned. Office visit completed",
        ),
    ],
)
def test_semantic_qualifiers_bind_to_the_exact_verbatim_occurrence(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    validated = _validate({category: [fact]}, page=page)

    assert len(getattr(validated, category)) == 1


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "conditions",
            {
                "name": "diabetes",
                "assertion": "present",
                "verbatim": "Possible diabetes",
                "page_number": 1,
                "evidence_excerpt": "Diabetes confirmed. Possible diabetes",
            },
            "Diabetes confirmed. Possible diabetes",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency="daily",
                status="active",
                verbatim="Consider Metformin daily",
                evidence_excerpt="Metformin active daily. Consider Metformin daily",
            ),
            "Metformin active daily. Consider Metformin daily",
        ),
    ],
)
def test_qualifier_from_another_occurrence_cannot_authorize_promotion(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=page)


def test_ambiguous_repeated_verbatim_cannot_select_another_qualified_occurrence() -> (
    None
):
    text = "Possible diabetes. Diabetes confirmed"
    fact = {
        "name": "Diabetes",
        "assertion": "present",
        "verbatim": "Diabetes",
        "page_number": 1,
        "evidence_excerpt": text,
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*verbatim"):
        _validate({"conditions": [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed and Lisinopril active",
                evidence_excerpt="Metformin listed and Lisinopril active",
            ),
            "Metformin listed and Lisinopril active",
        ),
        (
            "conditions",
            {
                "name": "Diabetes",
                "assertion": "uncertain",
                "verbatim": "Diabetes listed and possible asthma",
                "page_number": 1,
                "evidence_excerpt": "Diabetes listed and possible asthma",
            },
            "Diabetes listed and possible asthma",
        ),
        (
            "conditions",
            {
                "name": "Diabetes",
                "assertion": "family_history",
                "relationship": "mother",
                "verbatim": "Diabetes listed and mother has asthma",
                "page_number": 1,
                "evidence_excerpt": "Diabetes listed and mother has asthma",
            },
            "Diabetes listed and mother has asthma",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy listed and appendectomy performed",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy listed and appendectomy performed",
            },
            "Colonoscopy listed and appendectomy performed",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine listed and COVID vaccine given",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine listed and COVID vaccine given",
            },
            "Influenza vaccine listed and COVID vaccine given",
        ),
    ],
)
def test_qualifier_for_another_subject_cannot_authorize_fact_promotion(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


def test_ambiguous_repeated_medication_verbatim_is_rejected() -> None:
    text = "Metformin listed. Metformin active"
    fact = _medication(
        name="Metformin",
        status="active",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
        verbatim="Metformin",
        evidence_excerpt=text,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*verbatim"):
        _validate({"medications": [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed Lisinopril active",
                evidence_excerpt="Metformin listed\nLisinopril active",
            ),
            "Metformin listed\nLisinopril active",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed • Lisinopril active",
                evidence_excerpt="Metformin listed • Lisinopril active",
            ),
            "Metformin listed • Lisinopril active",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed / Lisinopril active",
                evidence_excerpt="Metformin listed / Lisinopril active",
            ),
            "Metformin listed / Lisinopril active",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed/Lisinopril active",
                evidence_excerpt="Metformin listed/Lisinopril active",
            ),
            "Metformin listed/Lisinopril active",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy listed\nAppendectomy performed",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy listed\nAppendectomy performed",
            },
            "Colonoscopy listed\nAppendectomy performed",
        ),
        (
            "conditions",
            {
                "name": "Diabetes",
                "assertion": "uncertain",
                "verbatim": "Diabetes listed\nPossible asthma",
                "page_number": 1,
                "evidence_excerpt": "Diabetes listed\nPossible asthma",
            },
            "Diabetes listed\nPossible asthma",
        ),
    ],
)
def test_ocr_list_boundary_cannot_leak_another_subject_qualifier(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Medication",
                status="active",
                dose_value="5",
                dose_unit="mg/dL",
                route=None,
                frequency=None,
                verbatim="Medication 5 mg/dL active",
                evidence_excerpt="Medication 5 mg/dL active",
            ),
            "Medication 5 mg/dL active",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "assertion": "present",
                "status": "final",
                "date": "01/02/2024",
                "verbatim": "CT chest 01/02/2024 finalized",
                "page_number": 1,
                "evidence_excerpt": "CT chest 01/02/2024 finalized",
            },
            "CT chest 01/02/2024 finalized",
        ),
    ],
)
def test_clinical_unit_and_date_slashes_remain_inside_fact_context(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    validated = _validate({category: [fact]}, page=text)

    assert len(getattr(validated, category)) == 1


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed Active Lisinopril",
                evidence_excerpt="Metformin listed\nActive Lisinopril",
            ),
            "Metformin listed\nActive Lisinopril",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed Current Lisinopril",
                evidence_excerpt="Metformin listed\nCurrent Lisinopril",
            ),
            "Metformin listed\nCurrent Lisinopril",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy listed Performed appendectomy",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy listed\nPerformed appendectomy",
            },
            "Colonoscopy listed\nPerformed appendectomy",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine listed Given COVID vaccine",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine listed\nGiven COVID vaccine",
            },
            "Influenza vaccine listed\nGiven COVID vaccine",
        ),
    ],
)
def test_status_led_newline_cannot_qualify_another_subject(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route=None,
                frequency=None,
                verbatim="Metformin listed Active",
                evidence_excerpt="Metformin listed\nActive",
            ),
            "Metformin listed\nActive",
        ),
        (
            "procedures",
            {
                "name": "Colonoscopy",
                "assertion": "present",
                "verbatim": "Colonoscopy listed Performed",
                "page_number": 1,
                "evidence_excerpt": "Colonoscopy listed\nPerformed",
            },
            "Colonoscopy listed\nPerformed",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "verbatim": "Influenza vaccine listed Given",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine listed\nGiven",
            },
            "Influenza vaccine listed\nGiven",
        ),
    ],
)
def test_status_only_wrapped_continuation_can_qualify_selected_subject(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    validated = _validate({category: [fact]}, page=text)

    assert len(getattr(validated, category)) == 1


@pytest.mark.parametrize(
    "frequency",
    ["Daily", "Weekly", "Monthly", "Nightly", "BID", "TID", "QID"],
)
def test_frequency_led_newline_cannot_qualify_another_medication(
    frequency: str,
) -> None:
    text = f"Metformin listed\n{frequency} Lisinopril"
    fact = _medication(
        name="Metformin",
        status="active",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=frequency,
        verbatim=f"Metformin listed {frequency} Lisinopril",
        evidence_excerpt=text,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\]"):
        _validate({"medications": [fact]}, page=text)


@pytest.mark.parametrize(
    "frequency",
    ["Daily", "Weekly", "Monthly", "Nightly", "BID", "TID", "QID"],
)
def test_frequency_only_wrapped_continuation_qualifies_selected_medication(
    frequency: str,
) -> None:
    text = f"Metformin\n{frequency}"
    fact = _medication(
        name="Metformin",
        status="active",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=frequency,
        verbatim=f"Metformin {frequency}",
        evidence_excerpt=text,
    )

    validated = _validate({"medications": [fact]}, page=text)

    assert len(validated.medications) == 1


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value=None,
                dose_unit=None,
                route="Oral",
                frequency="daily",
                verbatim="Metformin listed Oral Lisinopril daily",
                evidence_excerpt="Metformin listed\nOral Lisinopril daily",
            ),
            "Metformin listed\nOral Lisinopril daily",
        ),
        (
            "immunizations",
            {
                "name": "Influenza vaccine",
                "status": "completed",
                "route": "IM",
                "verbatim": "Influenza vaccine listed IM COVID vaccine given",
                "page_number": 1,
                "evidence_excerpt": "Influenza vaccine listed\nIM COVID vaccine given",
            },
            "Influenza vaccine listed\nIM COVID vaccine given",
        ),
        (
            "medications",
            _medication(
                name="Metformin",
                status="active",
                dose_value="500",
                dose_unit="mg",
                route=None,
                frequency="daily",
                verbatim="Metformin listed 500 mg Lisinopril daily",
                evidence_excerpt="Metformin listed\n500 mg Lisinopril daily",
            ),
            "Metformin listed\n500 mg Lisinopril daily",
        ),
    ],
)
def test_route_led_newline_cannot_qualify_another_named_subject(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


@pytest.mark.parametrize(
    ("text", "dose_value", "dose_unit", "route", "frequency"),
    [
        ("Metformin\nOral twice daily", None, None, "Oral", "twice daily"),
        ("Metformin\n500 mg oral twice daily", "500", "mg", "oral", "twice daily"),
        ("Metformin active\n0.5 mg", "0.5", "mg", None, None),
        ("Metformin active\n500", "500", None, None, None),
    ],
)
def test_qualifier_only_route_or_dose_continuation_stays_with_selected_subject(
    text: str,
    dose_value: str | None,
    dose_unit: str | None,
    route: str | None,
    frequency: str | None,
) -> None:
    fact = _medication(
        name="Metformin",
        status="active",
        dose_value=dose_value,
        dose_unit=dose_unit,
        route=route,
        frequency=frequency,
        verbatim=" ".join(text.splitlines()),
        evidence_excerpt=text,
    )

    validated = _validate({"medications": [fact]}, page=text)

    assert len(validated.medications) == 1


def test_route_dose_and_status_only_continuation_stays_with_selected_subject() -> None:
    text = "Influenza vaccine listed\nIM 0.5 mL given"
    fact = {
        "name": "Influenza vaccine",
        "status": "completed",
        "route": "IM",
        "dose": "0.5",
        "dose_unit": "mL",
        "verbatim": "Influenza vaccine listed IM 0.5 mL given",
        "page_number": 1,
        "evidence_excerpt": text,
    }

    validated = _validate({"immunizations": [fact]}, page=text)

    assert len(validated.immunizations) == 1


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "conditions",
            {
                "name": "diabetes",
                "assertion": "present",
                "verbatim": "Monitor for diabetes",
                "page_number": 1,
                "evidence_excerpt": "Monitor for diabetes",
            },
            "Monitor for diabetes",
        ),
        (
            "labs",
            {
                "name": "Glucose",
                "value": "95",
                "unit": "mg/dL",
                "assertion": "present",
                "verbatim": "Target glucose 95 mg/dL",
                "page_number": 1,
                "evidence_excerpt": "Target glucose 95 mg/dL",
            },
            "Target glucose 95 mg/dL",
        ),
        (
            "vital_signs",
            {
                "name": "Blood pressure",
                "value": "120/80",
                "unit": "mmHg",
                "assertion": "present",
                "verbatim": "Blood pressure should be 120/80 mmHg",
                "page_number": 1,
                "evidence_excerpt": "Blood pressure should be 120/80 mmHg",
            },
            "Blood pressure should be 120/80 mmHg",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "Advised office visit",
                "page_number": 1,
                "evidence_excerpt": "Advised office visit",
            },
            "Advised office visit",
        ),
        (
            "diagnostic_reports",
            {
                "name": "CT chest",
                "status": "final",
                "assertion": "present",
                "verbatim": "CT chest not performed",
                "page_number": 1,
                "evidence_excerpt": "CT chest not performed",
            },
            "CT chest not performed",
        ),
        (
            "labs",
            {
                "name": "Glucose",
                "assertion": "present",
                "verbatim": "Glucose not performed",
                "page_number": 1,
                "evidence_excerpt": "Glucose not performed",
            },
            "Glucose not performed",
        ),
        (
            "encounters",
            {
                "name": "Office visit",
                "status": "finished",
                "verbatim": "No office visit was completed",
                "page_number": 1,
                "evidence_excerpt": "No office visit was completed",
            },
            "No office visit was completed",
        ),
        (
            "care_plans",
            {
                "title": "Blood pressure plan",
                "status": "active",
                "plan_items": ["monitor blood pressure"],
                "verbatim": "Blood pressure plan: monitor blood pressure",
                "page_number": 1,
                "evidence_excerpt": "Blood pressure plan: monitor blood pressure",
            },
            "Blood pressure plan: monitor blood pressure",
        ),
    ],
)
def test_recommendation_or_target_language_cannot_become_observed_fact(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\]"):
        _validate({category: [fact]}, page=text)


def test_negative_family_history_is_not_promoted_as_family_condition() -> None:
    text = "Mother without diabetes"
    fact = {
        "name": "diabetes",
        "assertion": "family_history",
        "relationship": "Mother",
        "verbatim": text,
        "page_number": 1,
        "evidence_excerpt": text,
    }

    with pytest.raises(LocalValidationError, match=r"conditions\[0\].*assertion"):
        _validate({"conditions": [fact]}, page=text)


@pytest.mark.parametrize(
    ("category", "fact", "text"),
    [
        (
            "labs",
            {
                "name": "Glucose",
                "value": "95",
                "unit": "mg/dL",
                "verbatim": "Glucose 95; units mg/dL",
                "page_number": 1,
                "evidence_excerpt": "Glucose 95; units mg/dL",
            },
            "Glucose 95; units mg/dL",
        ),
        (
            "labs",
            {
                "name": "Glucose",
                "value": "95",
                "verbatim": "Glucose 95 mg/dL",
                "page_number": 1,
                "evidence_excerpt": "Glucose 95 mg/dL",
            },
            "Glucose 95 mg/dL",
        ),
        (
            "labs",
            {
                "name": "Level",
                "value": "10",
                "unit": "mmol/L",
                "verbatim": "Level 10^-3 mmol/L",
                "page_number": 1,
                "evidence_excerpt": "Level 10^-3 mmol/L",
            },
            "Level 10^-3 mmol/L",
        ),
    ],
)
def test_numeric_value_and_unit_must_be_exact_and_adjacent(
    category: str,
    fact: dict[str, object],
    text: str,
) -> None:
    with pytest.raises(LocalValidationError, match=rf"{category}\[0\].*(?:value|unit)"):
        _validate({category: [fact]}, page=text)


def test_numeric_value_and_unit_accept_markdown_table_cell_boundaries() -> None:
    text = "Potassium | 4.1 | mmol/L"
    result = _validate(
        {
            "labs": [
                {
                    "name": "Potassium",
                    "value": "4.1",
                    "unit": "mmol/L",
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert result.labs[0].value == "4.1"
    assert result.labs[0].unit == "mmol/L"


def test_html_table_entities_are_decoded_only_for_semantic_grounding() -> None:
    text = "<tr><td>TSH</td><td>&lt; 0.05</td><td>mIU/L</td></tr>"
    result = _validate(
        {
            "labs": [
                {
                    "name": "TSH",
                    "value": "< 0.05",
                    "unit": "mIU/L",
                    "verbatim": text,
                    "page_number": 1,
                    "evidence_excerpt": text,
                }
            ]
        },
        page=text,
    )

    assert result.labs[0].value == "< 0.05"
    assert result.labs[0].verbatim == text
    assert result.evidence[0].start_offset == 0


def test_duplicate_fact_comparison_preserves_case_varied_corroboration_across_pages() -> (
    None
):
    first = _medication(
        fact_id="med-upper",
        name="METFORMIN",
        verbatim="METFORMIN ACTIVE",
        evidence_excerpt="METFORMIN ACTIVE",
        dose_value=None,
        dose_unit=None,
        route=None,
        frequency=None,
        status="active",
    )
    second = {
        **first,
        "fact_id": "med-lower",
        "name": "metformin",
        "page_number": 2,
        "verbatim": "metformin active",
        "evidence_excerpt": "metformin active",
    }

    result = validate_clinical_extraction(
        {"medications": [first, second]},
        pages={1: "METFORMIN ACTIVE", 2: "metformin active"},
        upload_id="upload-1",
    )

    assert len(result.medications) == 2
    assert len(result.evidence) == 2


def test_model_authored_normalization_metadata_is_rejected() -> None:
    text = "Glucose 95 mg/dL"
    raw = {
        "labs": [
            {
                "name": "Glucose",
                "value": "95",
                "unit": "mg/dL",
                "normalized_value": "95",
                "normalization_method": "identity",
                "normalization_version": "1",
                "verbatim": text,
                "page_number": 1,
                "evidence_excerpt": text,
            }
        ]
    }

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*normalization"):
        _validate(raw, page=text)
