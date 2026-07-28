from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.grounded_summary import build_grounded_summary_input
from app.services.local_ai.summary_projection import project_summary_records


def _record(
    record_type: str,
    *,
    fhir: dict | None = None,
    display: str = "Recorded item",
    status: str | None = "unknown",
    source_format: str = "fhir",
    ai_extracted: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        record_type=record_type,
        fhir_resource=fhir or {"resourceType": "Basic"},
        display_text=display,
        status=status,
        effective_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        effective_date_end=None,
        source_format=source_format,
        ai_extracted=ai_extracted,
        source_section=None,
    )


def _evidence(record_id: UUID, paths: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        health_record_id=record_id,
        excerpt="Source excerpt",
        page_number=2,
        section="Medication list",
        field_paths=paths,
        source_metadata={"evidence_id": f"source_{record_id.hex}"},
    )


def _ground(projection):
    return build_grounded_summary_input(
        facts=projection.facts,
        evidence=projection.evidence,
        requested_scope={"summary_type": "full_health"},
        uncertainty_labels=projection.uncertainty_labels,
    )


def test_projects_rich_medication_and_exact_extraction_paths() -> None:
    record = _record(
        "medication",
        source_format="local_ai",
        ai_extracted=True,
        display="Metformin",
        status="active",
        fhir={
            "resourceType": "MedicationRequest",
            "status": "active",
            "authoredOn": "2024-01-02",
            "medicationCodeableConcept": {"text": "Metformin"},
            "dosageInstruction": [
                {
                    "doseAndRate": [
                        {"doseQuantity": {"value": 500, "unit": "mg"}},
                    ],
                    "route": {"text": "oral"},
                    "timing": {"code": {"text": "twice daily"}},
                }
            ],
            "dispenseRequest": {"validityPeriod": {"end": "2024-06-30"}},
            "_extraction_metadata": {
                "entity_class": "medication",
                "original_text": "Metformin 500 mg oral twice daily",
                "attributes": {
                    "dose": "500 mg",
                    "route": "oral",
                    "frequency": "twice daily",
                    "end_date": "2024-06-30",
                },
            },
        },
    )
    source = _evidence(
        record.id,
        [
            "medications[0].name",
            "medications[0].dose",
            "medications[0].route",
            "medications[0].frequency",
            "medications[0].end_date",
        ],
    )

    projection = project_summary_records([record], {record.id: [source]})

    assert projection.facts[0]["content"] == {
        "record_type": "medication",
        "name": "Metformin",
        "dose": "500 mg",
        "dose_value": 500,
        "dose_unit": "mg",
        "route": "oral",
        "frequency": "twice daily",
        "effective_date": "2024-01-02",
        "end_date": "2024-06-30",
        "status": "active",
    }
    assert projection.evidence[0]["field_paths"] == [
        "/name",
        "/dose",
        "/route",
        "/frequency",
        "/end_date",
    ]
    _ground(projection)


@pytest.mark.parametrize("record_type", ["condition", "allergy"])
def test_refuted_fhir_verification_status_is_never_projected_as_present(
    record_type: str,
) -> None:
    code_key = "code"
    record = _record(
        record_type,
        status="active",
        fhir={
            "resourceType": (
                "Condition" if record_type == "condition" else "AllergyIntolerance"
            ),
            code_key: {"text": "Refuted clinical item"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            + (
                                "condition-ver-status"
                                if record_type == "condition"
                                else "allergyintolerance-verification"
                            )
                        ),
                        "code": "refuted",
                    }
                ]
            },
            "_extraction_metadata": {
                "attributes": {"assertion": "present"},
            },
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["assertion"] == "negated"
    _ground(projection)


def test_fhir_verification_status_uses_canonical_code_over_display() -> None:
    record = _record(
        "condition",
        status="active",
        fhir={
            "resourceType": "Condition",
            "code": {"text": "Refuted clinical item"},
            "verificationStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                        ),
                        "code": "refuted",
                        "display": "Refuted condition",
                    }
                ]
            },
            "_extraction_metadata": {"attributes": {"assertion": "present"}},
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["assertion"] == "negated"
    assert "status" not in projection.facts[0]["content"]
    _ground(projection)


@pytest.mark.parametrize(
    ("record_type", "verification_code"),
    [
        ("condition", "entered-in-error"),
        ("allergy", "entered-in-error"),
        ("condition", "unsupported-verification-code"),
        ("allergy", "unsupported-verification-code"),
    ],
)
def test_ineligible_fhir_verification_status_fails_closed(
    record_type: str,
    verification_code: str,
) -> None:
    record = _record(
        record_type,
        status="active",
        fhir={
            "resourceType": (
                "Condition" if record_type == "condition" else "AllergyIntolerance"
            ),
            "code": {"text": "Ineligible clinical item"},
            "verificationStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/"
                            + (
                                "condition-ver-status"
                                if record_type == "condition"
                                else "allergyintolerance-verification"
                            )
                        ),
                        "code": verification_code,
                    }
                ]
            },
            "_extraction_metadata": {"attributes": {"assertion": "present"}},
        },
    )

    with pytest.raises(LocalValidationError, match="verification status"):
        project_summary_records([record], {})


@pytest.mark.parametrize(
    "verification_status",
    [
        {"text": "confirmed"},
        {"coding": [{"display": "refuted"}], "text": "confirmed"},
        {"coding": [{}], "text": "confirmed"},
        {"coding": "confirmed", "text": "confirmed"},
        {"coding": [{"code": "confirmed"}, {"code": "refuted"}]},
        {"coding": [{"code": "refuted"}]},
        {"coding": [{"system": None, "code": "refuted"}]},
    ],
)
def test_malformed_or_contradictory_fhir_verification_status_fails_closed(
    verification_status: object,
) -> None:
    record = _record(
        "condition",
        status="active",
        fhir={
            "resourceType": "Condition",
            "code": {"text": "Clinical item"},
            "verificationStatus": verification_status,
        },
    )

    with pytest.raises(LocalValidationError, match="verification status"):
        project_summary_records([record], {})


@pytest.mark.parametrize(
    "verification_code",
    [" Refuted ", "REFUTED", "refuted ", "un_confirmed"],
)
def test_noncanonical_fhir_verification_code_fails_closed(
    verification_code: str,
) -> None:
    record = _record(
        "condition",
        status="active",
        fhir={
            "resourceType": "Condition",
            "code": {"text": "Clinical item"},
            "verificationStatus": {
                "coding": [
                    {
                        "system": (
                            "http://terminology.hl7.org/CodeSystem/condition-ver-status"
                        ),
                        "code": verification_code,
                    }
                ]
            },
        },
    )

    with pytest.raises(LocalValidationError, match="verification status"):
        project_summary_records([record], {})


@pytest.mark.parametrize(
    ("record_type", "system"),
    [
        ("condition", "https://example.invalid/condition-verification"),
        ("allergy", "https://example.invalid/allergy-verification"),
        ("condition", ""),
        ("allergy", 123),
    ],
)
def test_unsupported_fhir_verification_coding_system_fails_closed(
    record_type: str,
    system: object,
) -> None:
    record = _record(
        record_type,
        status="active",
        fhir={
            "resourceType": (
                "Condition" if record_type == "condition" else "AllergyIntolerance"
            ),
            "code": {"text": "Clinical item"},
            "verificationStatus": {
                "coding": [{"system": system, "code": "confirmed"}],
                "text": "confirmed",
            },
        },
    )

    with pytest.raises(LocalValidationError, match="verification status"):
        project_summary_records([record], {})


def test_medication_statement_uses_dosage_and_effective_period() -> None:
    record = _record(
        "medication",
        status="active",
        fhir={
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"text": "Metformin"},
            "effectivePeriod": {
                "start": "2024-02-01",
                "end": "2024-08-31",
            },
            "dosage": [
                {
                    "doseAndRate": [
                        {"doseQuantity": {"value": 500, "unit": "mg"}},
                    ],
                    "route": {"text": "oral"},
                    "timing": {"code": {"text": "twice daily"}},
                }
            ],
        },
    )
    record.effective_date = None

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"] == {
        "record_type": "medication",
        "name": "Metformin",
        "dose_value": 500,
        "dose_unit": "mg",
        "route": "oral",
        "frequency": "twice daily",
        "effective_date": "2024-02-01",
        "end_date": "2024-08-31",
        "status": "active",
    }
    _ground(projection)


def test_medication_statement_uses_dosage_text_as_grounded_dose() -> None:
    record = _record(
        "medication",
        status="active",
        fhir={
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": {"text": "Metformin"},
            "dosage": [{"text": "Take one tablet by mouth twice daily"}],
        },
    )

    projection = project_summary_records([record], {})

    assert (
        projection.facts[0]["content"]["dose"] == "Take one tablet by mouth twice daily"
    )
    assert all(
        item["template_id"] != "medication_dose_missing"
        for item in projection.uncertainty_labels
    )
    _ground(projection)


def test_medication_request_does_not_treat_dispense_validity_as_therapy_dates() -> None:
    record = _record(
        "medication",
        status="active",
        fhir={
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Metformin"},
            "dispenseRequest": {
                "validityPeriod": {
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                }
            },
        },
    )
    record.effective_date = None

    projection = project_summary_records([record], {})
    content = projection.facts[0]["content"]

    assert "effective_date" not in content
    assert "end_date" not in content
    _ground(projection)


def test_unitless_immunization_dose_never_renders_none_text() -> None:
    record = _record(
        "immunization",
        fhir={
            "resourceType": "Immunization",
            "status": "completed",
            "vaccineCode": {"text": "Influenza"},
            "doseQuantity": {"value": 0.5},
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["dose"] == "0.5"
    _ground(projection)


@pytest.mark.parametrize(
    ("record_type", "fhir", "expected"),
    [
        (
            "observation",
            {
                "resourceType": "Observation",
                "code": {"text": "HbA1c"},
                "valueQuantity": {"value": 6.8, "unit": "%"},
                "referenceRange": [{"text": "4.0-5.6 %"}],
                "interpretation": [{"text": "high"}],
            },
            {"name": "HbA1c", "value": 6.8, "unit": "%", "interpretation": "high"},
        ),
        (
            "allergy",
            {
                "resourceType": "AllergyIntolerance",
                "code": {"text": "Penicillin"},
                "reaction": [
                    {
                        "manifestation": [{"text": "rash"}],
                        "severity": "mild",
                    }
                ],
            },
            {"substance": "Penicillin", "reaction": "rash", "severity": "mild"},
        ),
        (
            "family_history",
            {
                "resourceType": "FamilyMemberHistory",
                "relationship": {"text": "father"},
                "condition": [{"code": {"text": "Diabetes"}}],
            },
            {
                "record_type": "condition",
                "diagnosis": "Diabetes",
                "assertion": "family_history",
                "relationship": "father",
            },
        ),
        (
            "procedure",
            {
                "resourceType": "Procedure",
                "code": {"text": "Colonoscopy"},
                "performer": [{"actor": {"display": "Dr Smith"}}],
                "bodySite": [{"text": "colon"}],
            },
            {"name": "Colonoscopy", "provider": "Dr Smith", "body_site": "colon"},
        ),
    ],
)
def test_projects_rich_clinical_types(
    record_type: str,
    fhir: dict,
    expected: dict,
) -> None:
    record = _record(record_type, fhir=fhir, status=None)

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"].items() >= expected.items()
    _ground(projection)


@pytest.mark.parametrize(
    ("record_type", "fhir", "source_paths", "expected_paths"),
    [
        (
            "observation",
            {
                "resourceType": "Observation",
                "code": {"text": "HbA1c"},
                "valueQuantity": {"value": 6.8, "unit": "%"},
            },
            ["labs[0].test", "labs[0].value", "labs[0].unit"],
            ["/name", "/value", "/unit"],
        ),
        (
            "allergy",
            {
                "resourceType": "AllergyIntolerance",
                "code": {"text": "Penicillin"},
                "reaction": [{"manifestation": [{"text": "rash"}]}],
            },
            ["allergies[0].name", "allergies[0].reaction"],
            ["/substance", "/reaction"],
        ),
        (
            "family_history",
            {
                "resourceType": "FamilyMemberHistory",
                "relationship": {"text": "father"},
                "condition": [{"code": {"text": "Diabetes"}}],
            },
            [
                "family_history[0].condition",
                "family_history[0].relationship",
            ],
            ["/diagnosis", "/relationship"],
        ),
        (
            "procedure",
            {
                "resourceType": "Procedure",
                "code": {"text": "Colonoscopy"},
                "performer": [{"actor": {"display": "Dr Smith"}}],
            },
            ["procedures[0].name", "procedures[0].provider"],
            ["/name", "/provider"],
        ),
    ],
)
def test_extraction_paths_map_to_exact_projected_json_pointers(
    record_type: str,
    fhir: dict,
    source_paths: list[str],
    expected_paths: list[str],
) -> None:
    record = _record(
        record_type,
        fhir=fhir,
        source_format="local_ai",
        ai_extracted=True,
    )
    source = _evidence(record.id, source_paths)

    projection = project_summary_records([record], {record.id: [source]})

    assert projection.evidence[0]["field_paths"] == expected_paths
    _ground(projection)


def test_structured_full_health_projection_never_silently_drops_supported_records() -> (
    None
):
    supported = [
        "allergy",
        "appointment",
        "care_plan",
        "care_team",
        "communication",
        "condition",
        "diagnostic_report",
        "document",
        "encounter",
        "imaging",
        "immunization",
        "medication",
        "observation",
        "procedure",
        "questionnaire_response",
        "service_request",
    ]
    records = [
        _record(
            record_type,
            display=f"Item {index}",
            fhir=(
                {
                    "resourceType": "FamilyMemberHistory",
                    "relationship": {"text": "parent"},
                    "condition": [{"code": {"text": f"Item {index}"}}],
                }
                if record_type == "family_history"
                else None
            ),
        )
        for index, record_type in enumerate(supported)
    ]

    projection = project_summary_records(records, {})
    grounded = _ground(projection)

    assert len(projection.facts) == len(records)
    assert len(grounded.facts) == len(records)
    assert {fact.record_id for fact in grounded.facts} == {
        str(record.id) for record in records
    }
    assert {fact.content_json for fact in grounded.facts}


def test_extracted_record_without_source_evidence_fails_instead_of_disappearing() -> (
    None
):
    record = _record(
        "condition",
        source_format="local_ai",
        ai_extracted=True,
        display="Hypertension",
    )

    with pytest.raises(LocalValidationError, match="source evidence"):
        project_summary_records([record], {})


@pytest.mark.parametrize("value", [0, -1.5])
def test_structured_observation_preserves_zero_and_negative_results(
    value: int | float,
) -> None:
    record = _record(
        "observation",
        fhir={
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Net fluid balance"},
            "valueQuantity": {"value": value, "unit": "mL"},
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["value"] == value
    assert all(
        item["template_id"] != "result_missing"
        for item in projection.uncertainty_labels
    )
    _ground(projection)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_structured_observation_rejects_nonfinite_results(value: float) -> None:
    record = _record(
        "observation",
        fhir={
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Invalid result"},
            "valueQuantity": {"value": value, "unit": "mg"},
        },
    )

    with pytest.raises(LocalValidationError, match="finite"):
        project_summary_records([record], {})


def test_structured_observation_projects_comparator_quantity_as_typed_value() -> None:
    record = _record(
        "observation",
        fhir={
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "TSH"},
            "valueQuantity": {
                "value": 0.05,
                "comparator": "<",
                "unit": "mIU/L",
            },
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["value"] == {
        "kind": "quantity",
        "comparator": "<",
        "number": 0.05,
        "unit": "mIU/L",
    }
    assert "unit" not in projection.facts[0]["content"]
    assert {
        "/value/comparator",
        "/value/kind",
        "/value/number",
        "/value/unit",
    }.issubset(projection.evidence[0]["field_paths"])
    _ground(projection)


def test_extracted_observation_projects_ratio_as_typed_evidence_linked_value() -> None:
    record = _record(
        "observation",
        source_format="local_ai",
        ai_extracted=True,
        display="Blood pressure",
        fhir={
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Blood pressure"},
            "valueString": "Blood pressure 120/80 mmHg",
            "_extraction_metadata": {
                "entity_class": "vital",
                "original_text": "Blood pressure 120/80 mmHg",
                "attributes": {
                    "value": "120/80",
                    "unit": "mmHg",
                },
            },
        },
    )
    source = _evidence(
        record.id,
        [
            "vital_signs[0].name",
            "vital_signs[0].value",
            "vital_signs[0].unit",
        ],
    )

    projection = project_summary_records([record], {record.id: [source]})

    assert projection.facts[0]["content"]["value"] == {
        "kind": "ratio",
        "numerator": 120,
        "denominator": 80,
        "numerator_unit": "mmHg",
        "denominator_unit": "mmHg",
    }
    assert "unit" not in projection.facts[0]["content"]
    mapped_paths = set(projection.evidence[0]["field_paths"])
    assert {
        "/name",
        "/value/kind",
        "/value/numerator",
        "/value/denominator",
        "/value/numerator_unit",
        "/value/denominator_unit",
    } == mapped_paths
    _ground(projection)


def test_structured_observation_projects_fhir_ratio_without_losing_units() -> None:
    record = _record(
        "observation",
        fhir={
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Albumin creatinine ratio"},
            "valueRatio": {
                "numerator": {"value": 30, "unit": "mg"},
                "denominator": {"value": 1, "unit": "g"},
            },
        },
    )

    projection = project_summary_records([record], {})

    assert projection.facts[0]["content"]["value"] == {
        "kind": "ratio",
        "numerator": 30,
        "denominator": 1,
        "numerator_unit": "mg",
        "denominator_unit": "g",
    }
    _ground(projection)


def test_mapped_paths_accepts_an_exact_typed_json_pointer_first() -> None:
    from app.services.local_ai.summary_projection import _mapped_paths

    known_paths = {
        "/name",
        "/value/kind",
        "/value/numerator",
        "/value/denominator",
        "/value/numerator_unit",
        "/value/denominator_unit",
    }

    assert _mapped_paths(
        "observation",
        ["/value/numerator"],
        known_paths,
    ) == ["/value/numerator"]


def test_mapped_paths_does_not_reuse_a_dotted_predecessor_for_exact_pointer() -> None:
    from app.services.local_ai.summary_projection import _mapped_paths

    known_paths = {
        "/name",
        "/value/kind",
        "/value/numerator",
        "/value/denominator",
        "/value/numerator_unit",
        "/value/denominator_unit",
    }

    assert _mapped_paths(
        "observation",
        ["vital_signs[0].unit", "/value/numerator"],
        known_paths,
    ) == [
        "/value/denominator_unit",
        "/value/numerator_unit",
        "/value/numerator",
    ]
