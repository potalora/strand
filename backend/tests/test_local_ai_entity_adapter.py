from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from uuid import uuid4

import pytest

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.ingestion.content_hash import content_hash
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai import adapters
from app.services.local_ai.adapters import (
    to_extracted_entities,
    validated_extraction_to_health_record_dicts,
)
from app.services.local_ai.extraction_schema import ClinicalDocumentExtraction
from app.services.local_ai.extraction_validator import validate_clinical_extraction


def _fact(
    name_key: str, name: str, verbatim: str, **attrs: object
) -> dict[str, object]:
    return {
        name_key: name,
        "verbatim": verbatim,
        "page_number": 1,
        "evidence_excerpt": verbatim,
        **attrs,
    }


def _validated(raw: dict[str, object], page: str):
    return validate_clinical_extraction(
        raw,
        pages={1: page},
        upload_id="upload-adapter",
    )


def test_adapter_maps_every_category_to_existing_canonical_entity_contract() -> None:
    facts = {
        "medications": [
            _fact(
                "name",
                "Metformin",
                "Metformin 500 mg oral daily",
                dose_value="500",
                dose_unit="mg",
                route="oral",
                frequency="daily",
                status="active",
            )
        ],
        "conditions": [
            _fact("name", "Hypertension", "Hypertension active", assertion="present")
        ],
        "procedures": [
            _fact(
                "name",
                "Colonoscopy",
                "Colonoscopy performed 2024-01-02",
                assertion="present",
                date="2024-01-02",
            )
        ],
        "labs": [
            _fact(
                "name",
                "Glucose",
                "Glucose 95 mg/dL (70-99) 2024-01-02",
                value="95",
                unit="mg/dL",
                reference_range="70-99",
                date="2024-01-02",
            )
        ],
        "allergies": [
            _fact(
                "substance",
                "Penicillin",
                "Penicillin allergy rash",
                reaction="rash",
                status="active",
            )
        ],
        "encounters": [
            _fact(
                "name",
                "Office visit",
                "Office visit 2024-01-02",
                visit_type="office",
                date="2024-01-02",
                status="finished",
            )
        ],
        "immunizations": [
            _fact(
                "name",
                "Influenza vaccine",
                "Influenza vaccine administered 2024-01-02",
                date="2024-01-02",
                status="completed",
            )
        ],
        "vital_signs": [
            _fact(
                "name",
                "Heart rate",
                "Heart rate 72 bpm",
                value="72",
                unit="bpm",
            )
        ],
        "diagnostic_reports": [
            _fact(
                "name",
                "CT chest",
                "CT chest showed no acute findings",
                findings="no acute findings",
                status="final",
            )
        ],
        "care_plans": [
            _fact(
                "title",
                "Care plan",
                "Active care plan: monitor blood pressure",
                plan_items=["monitor blood pressure"],
                status="active",
            )
        ],
    }
    page = " | ".join(
        item["verbatim"] for category in facts.values() for item in category
    )

    entities = to_extracted_entities(_validated(facts, page))

    assert [entity.entity_class for entity in entities] == [
        "medication",
        "condition",
        "procedure",
        "lab_result",
        "allergy",
        "encounter",
        "immunization",
        "vital",
        "imaging_result",
        "assessment_plan",
    ]
    assert entities[0].attributes["value"] == "500"
    assert entities[3].attributes["test"] == "Glucose"
    assert entities[3].attributes["ref_low"] == "70"
    assert entities[3].attributes["ref_high"] == "99"
    assert entities[8].attributes["procedure_name"] == "CT chest"
    assert entities[9].attributes["plan_items"] == ["monitor blood pressure"]


def test_adapter_uses_server_owned_normalization_provenance() -> None:
    fact = _fact(
        "name",
        "Glucose",
        "Glucose 95 mg/dL",
        value="95",
        unit="mg/dL",
    )
    validated = _validated({"labs": [fact]}, "Glucose 95 mg/dL")

    entity = to_extracted_entities(validated)[0]

    assert entity.text == "Glucose"
    assert entity.attributes["_evidence_ids"] == [validated.evidence[0].id]
    assert entity.attributes["_source_page"] == 1
    assert entity.attributes["_verbatim"] == "Glucose 95 mg/dL"
    assert entity.attributes["_normalization_method"] == "verbatim-preserved"
    assert (
        entity.attributes["_normalization_version"] == "clinical-document-extraction.v1"
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
        uuid4(),
    )[0]
    metadata = record["fhir_resource"]["_extraction_metadata"]
    assert metadata["_evidence_ids"] == [validated.evidence[0].id]
    assert metadata["_source_page"] == 1
    assert metadata["_verbatim"] == "Glucose 95 mg/dL"
    assert metadata["_normalization_method"] == "verbatim-preserved"
    assert metadata["_normalization_version"] == "clinical-document-extraction.v1"
    assert "_strict_local_provenance_signature" not in metadata["attributes"]


def test_only_closed_validated_to_fhir_mapping_promotes_provenance() -> None:
    validated = _validated(
        {
            "medications": [
                _fact("name", "Metformin", "Metformin active", status="active")
            ]
        },
        "Metformin active",
    )
    entity = to_extracted_entities(validated)[0]

    dict.__setitem__(entity.attributes, "_source_page", 2)
    object.__setattr__(entity, "text", "Warfarin")
    copied = copy.copy(entity)
    cloned = type(entity)(
        entity_class=entity.entity_class,
        text=entity.text,
        attributes=dict(entity.attributes),
        confidence=entity.confidence,
        start_pos=entity.start_pos,
        end_pos=entity.end_pos,
    )

    for untrusted in (
        entity,
        copied,
        cloned,
        replace(entity, confidence=0.01),
        replace(entity, start_pos=999),
        replace(entity, end_pos=1000),
    ):
        record = entity_to_health_record_dict(untrusted, uuid4(), uuid4(), uuid4())
        metadata = record["fhir_resource"]["_extraction_metadata"]
        assert "_evidence_ids" not in metadata
        assert "_source_page" not in metadata
        assert "_verbatim" not in metadata

    trusted_record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
        uuid4(),
    )[0]
    trusted_metadata = trusted_record["fhir_resource"]["_extraction_metadata"]
    assert trusted_metadata["_evidence_ids"] == [validated.evidence[0].id]
    assert trusted_metadata["_source_page"] == 1
    assert trusted_metadata["_verbatim"] == "Metformin active"


def test_no_arbitrary_entity_provenance_verifier_or_signer_is_exposed() -> None:
    from app.services.local_ai import extraction_validator

    assert not hasattr(extraction_validator, "attach_validated_provenance")
    assert "_get_key" not in vars(adapters)
    assert "_canonical_provenance" not in vars(adapters)
    assert "_PROVENANCE_SIGNATURE_KEY" not in vars(adapters)
    assert not hasattr(adapters, "verify_strict_local_provenance")
    assert to_extracted_entities.__closure__ is None
    assert validated_extraction_to_health_record_dicts.__closure__ is None


def test_provenance_uses_opaque_entity_identity_not_attachable_attributes() -> None:
    validated = _validated(
        {
            "medications": [
                _fact("name", "Metformin", "Metformin active", status="active")
            ]
        },
        "Metformin active",
    )
    entity = to_extracted_entities(validated)[0]

    assert not any(key.startswith("_strict_local") for key in entity.attributes)
    entity.attributes["_source_page"] = 2

    forged = ExtractedEntity(
        entity_class=entity.entity_class,
        text=entity.text,
        attributes=dict(entity.attributes),
        confidence=entity.confidence,
        start_pos=entity.start_pos,
        end_pos=entity.end_pos,
    )
    record = entity_to_health_record_dict(forged, uuid4(), uuid4(), uuid4())
    metadata = record["fhir_resource"]["_extraction_metadata"]
    assert "_evidence_ids" not in metadata
    assert "_source_page" not in metadata


def test_copying_a_trusted_entity_does_not_copy_its_object_identity() -> None:
    validated = _validated(
        {
            "medications": [
                _fact("name", "Metformin", "Metformin active", status="active")
            ]
        },
        "Metformin active",
    )
    copied = copy.copy(to_extracted_entities(validated)[0])

    record = entity_to_health_record_dict(copied, uuid4(), uuid4(), uuid4())
    metadata = record["fhir_resource"]["_extraction_metadata"]
    assert "_evidence_ids" not in metadata
    assert "_source_page" not in metadata


def test_adapter_is_order_preserving_and_deterministic() -> None:
    page = "Lisinopril 10 mg daily. Metformin 500 mg daily."
    raw = {
        "medications": [
            _fact(
                "name",
                "Lisinopril",
                "Lisinopril 10 mg daily",
                dose_value="10",
                dose_unit="mg",
                frequency="daily",
                status="active",
            ),
            _fact(
                "name",
                "Metformin",
                "Metformin 500 mg daily",
                dose_value="500",
                dose_unit="mg",
                frequency="daily",
                status="active",
            ),
        ]
    }
    validated = _validated(raw, page)

    first = to_extracted_entities(validated)
    second = to_extracted_entities(validated)

    assert [entity.text for entity in first] == ["Lisinopril", "Metformin"]
    assert [entity.attributes["_evidence_ids"] for entity in first] == [
        entity.attributes["_evidence_ids"] for entity in second
    ]


def test_adapter_preserves_negated_and_family_history_without_active_promotion() -> (
    None
):
    page = "No diabetes. Family history: mother had colon cancer."
    raw = {
        "conditions": [
            _fact("name", "diabetes", "No diabetes", assertion="negated"),
            _fact(
                "name",
                "colon cancer",
                "Family history: mother had colon cancer",
                assertion="family_history",
                relationship="mother",
            ),
        ]
    }

    entities = to_extracted_entities(_validated(raw, page))

    assert entities[0].entity_class == "condition"
    assert entities[0].attributes["status"] == "negated"
    assert entities[1].entity_class == "family_history"
    assert entities[1].attributes["relationship"] == "mother"


def test_mentioned_not_performed_procedure_does_not_become_an_entity() -> None:
    text = "Colonoscopy recommended for next year"
    raw = {
        "procedures": [
            _fact(
                "name",
                "Colonoscopy",
                text,
                assertion="mentioned_not_performed",
            )
        ]
    }

    entities = to_extracted_entities(_validated(raw, text))

    assert entities == []


def test_negated_allergy_is_not_promoted_to_active_fhir_input() -> None:
    text = "No penicillin allergy"
    raw = {
        "allergies": [
            _fact(
                "substance",
                "penicillin",
                text,
                assertion="negated",
                status="inactive",
            )
        ]
    }

    assert to_extracted_entities(_validated(raw, text)) == []


def test_planned_encounter_is_not_promoted_to_finished_fhir_input() -> None:
    text = "Office visit planned 2027-01-02"
    raw = {
        "encounters": [
            _fact(
                "name",
                "Office visit",
                text,
                visit_type="office",
                date="2027-01-02",
                status="planned",
            )
        ]
    }

    assert to_extracted_entities(_validated(raw, text)) == []


def test_name_only_medication_does_not_gain_none_dosage_attributes() -> None:
    raw = {"medications": [_fact("name", "Aspirin", "Aspirin active", status="active")]}

    entity = to_extracted_entities(_validated(raw, "Aspirin active"))[0]

    assert "value" not in entity.attributes
    assert "unit" not in entity.attributes
    record = entity_to_health_record_dict(entity, uuid4(), uuid4(), uuid4())
    assert record["display_text"] == "Aspirin"


def test_unresolved_and_rejected_fields_do_not_become_entities() -> None:
    validated = _validated(
        {
            "unresolved_fields": ["medications.dose"],
            "rejected_fields": ["labs[0].value"],
        },
        "No usable clinical facts",
    )

    assert to_extracted_entities(validated) == []


def test_adapter_rejects_schema_objects_that_did_not_pass_evidence_validation() -> None:
    direct = ClinicalDocumentExtraction.model_validate(
        {"conditions": [_fact("name", "Asthma", "Asthma active", assertion="present")]}
    )

    with pytest.raises(
        LocalValidationError, match=r"conditions\[0\].*evidence validation"
    ):
        to_extracted_entities(direct)


def test_adapter_does_not_bypass_existing_entity_validator() -> None:
    raw = {
        "medications": [_fact("name", "PPI", "PPI active medication", status="active")]
    }

    with pytest.raises(
        LocalValidationError, match=r"medications\[0\].*downstream guard"
    ):
        to_extracted_entities(_validated(raw, "PPI active medication"))


def test_downstream_rejection_does_not_log_clinical_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "private business analyst canary"
    raw = {"labs": [_fact("name", canary, canary)]}
    validated = _validated(raw, canary)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LocalValidationError, match="downstream guard"):
            to_extracted_entities(validated)

    assert canary not in caplog.text


@pytest.mark.parametrize(
    ("category", "fact", "page"),
    [
        (
            "medications",
            _fact(
                "name",
                "Metformin",
                "Metformin stopped",
                status="stopped",
            ),
            "Metformin stopped",
        ),
        (
            "medications",
            _fact(
                "name",
                "Metformin",
                "Metformin historical",
                status="historical",
            ),
            "Metformin historical",
        ),
        (
            "medications",
            _fact("name", "Metformin", "Metformin listed", status="unknown"),
            "Metformin listed",
        ),
        (
            "allergies",
            _fact(
                "substance",
                "Penicillin",
                "Penicillin allergy inactive",
                assertion="present",
                status="inactive",
            ),
            "Penicillin allergy inactive",
        ),
        (
            "allergies",
            _fact(
                "substance",
                "Penicillin",
                "Penicillin allergy resolved",
                assertion="present",
                status="resolved",
            ),
            "Penicillin allergy resolved",
        ),
        (
            "allergies",
            _fact(
                "substance",
                "Penicillin",
                "Penicillin allergy historical",
                assertion="present",
                status="historical",
            ),
            "Penicillin allergy historical",
        ),
        (
            "allergies",
            _fact(
                "substance",
                "Penicillin",
                "Penicillin allergy listed",
                assertion="present",
                status="unknown",
            ),
            "Penicillin allergy listed",
        ),
        (
            "immunizations",
            _fact(
                "name",
                "Influenza vaccine",
                "Influenza vaccine entered in error",
                status="entered_in_error",
            ),
            "Influenza vaccine entered in error",
        ),
        (
            "immunizations",
            _fact(
                "name",
                "Influenza vaccine",
                "Influenza vaccine listed",
                status="unknown",
            ),
            "Influenza vaccine listed",
        ),
        (
            "diagnostic_reports",
            _fact(
                "name",
                "CT chest",
                "CT chest preliminary",
                assertion="present",
                status="preliminary",
            ),
            "CT chest preliminary",
        ),
        (
            "diagnostic_reports",
            _fact(
                "name",
                "CT chest",
                "CT chest amended",
                assertion="present",
                status="amended",
            ),
            "CT chest amended",
        ),
        (
            "diagnostic_reports",
            _fact(
                "name",
                "CT chest",
                "CT chest listed",
                assertion="present",
                status="unknown",
            ),
            "CT chest listed",
        ),
        (
            "care_plans",
            _fact(
                "title",
                "Care plan",
                "Care plan inactive",
                status="inactive",
            ),
            "Care plan inactive",
        ),
        (
            "care_plans",
            _fact(
                "title",
                "Care plan",
                "Care plan resolved",
                status="resolved",
            ),
            "Care plan resolved",
        ),
        (
            "care_plans",
            _fact(
                "title",
                "Care plan",
                "Care plan historical",
                status="historical",
            ),
            "Care plan historical",
        ),
        (
            "care_plans",
            _fact("title", "Care plan", "Care plan listed", status="unknown"),
            "Care plan listed",
        ),
    ],
)
def test_incompatible_lifecycle_states_are_not_promoted_to_fhir(
    category: str,
    fact: dict[str, object],
    page: str,
) -> None:
    entities = to_extracted_entities(_validated({category: [fact]}, page))

    assert entities == []


@pytest.mark.parametrize("status", ["in_progress", "planned", "unknown"])
def test_non_finished_encounters_never_become_finished(status: str) -> None:
    text = f"Office visit {status}"
    raw = {
        "encounters": [
            _fact(
                "name",
                "Office visit",
                text,
                visit_type="office",
                status=status,
            )
        ]
    }

    assert to_extracted_entities(_validated(raw, text)) == []


def test_validated_extraction_is_immutable_and_cannot_reuse_stale_evidence() -> None:
    validated = _validated(
        {"medications": [_fact("name", "Metformin", "Metformin active")]},
        "Metformin active",
    )

    with pytest.raises(Exception):
        validated.medications[0].name = "Warfarin"
    with pytest.raises(Exception):
        validated.medications = []

    care_plan = _validated(
        {
            "care_plans": [
                _fact(
                    "title",
                    "Care plan",
                    "Care plan active monitor blood pressure",
                    plan_items=["monitor blood pressure"],
                    status="active",
                )
            ]
        },
        "Care plan active monitor blood pressure",
    )
    care_plan.care_plans[0].plan_items.append("invented treatment")
    with pytest.raises(LocalValidationError, match="evidence validation required"):
        to_extracted_entities(care_plan)


def test_private_evidence_id_forgery_does_not_bypass_validation() -> None:
    direct = ClinicalDocumentExtraction.model_validate(
        {"medications": [_fact("name", "Warfarin", "Warfarin active")]}
    )
    direct.medications[0]._evidence_id = "ev1_" + "a" * 40

    with pytest.raises(LocalValidationError, match="evidence validation required"):
        to_extracted_entities(direct)


def test_strict_local_entity_attributes_remain_json_serializable() -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )

    entity = to_extracted_entities(validated)[0]

    json.dumps(entity.attributes)


def test_immunization_dose_value_and_unit_map_to_valid_fhir() -> None:
    text = "Influenza vaccine administered 0.5 mL"
    validated = _validated(
        {
            "immunizations": [
                _fact(
                    "name",
                    "Influenza vaccine",
                    text,
                    status="completed",
                    dose="0.5",
                    dose_unit="mL",
                )
            ]
        },
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["doseQuantity"] == {
        "value": 0.5,
        "unit": "mL",
    }


def test_immunization_scientific_dose_round_trips_to_valid_fhir() -> None:
    text = "Influenza vaccine 4.5 x 10^9 units administered"
    validated = _validated(
        {
            "immunizations": [
                _fact(
                    "name",
                    "Influenza vaccine",
                    text,
                    status="completed",
                    dose="4.5 x 10^9",
                    dose_unit="units",
                )
            ]
        },
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["doseQuantity"] == {
        "value": 4_500_000_000.0,
        "unit": "units",
    }


def test_closed_mapper_fails_when_fhir_structure_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )
    monkeypatch.setattr(
        adapters,
        "_strict_fhir_resource_is_valid",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*FHIR"):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


@pytest.mark.parametrize(
    ("source_value", "expected_value"),
    [
        ("1,5", 1.5),
        ("0,001", 0.001),
        ("0,500", 0.5),
        ("-0,125", -0.125),
        ("1,000", 1000.0),
        ("12,345", 12345.0),
        ("1.2e3", 1200.0),
        ("4.5 x 10^9", 4_500_000_000.0),
        ("4.5 \u00d7 10\u2079", 4_500_000_000.0),
    ],
)
def test_closed_mapper_preserves_grounded_numeric_forms_without_value_loss(
    source_value: str,
    expected_value: float,
) -> None:
    text = f"Glucose {source_value} mmol/L"
    validated = _validated(
        {
            "labs": [
                _fact(
                    "name",
                    "Glucose",
                    text,
                    value=source_value,
                    unit="mmol/L",
                )
            ]
        },
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["valueQuantity"] == {
        "value": expected_value,
        "unit": "mmol/L",
    }
    assert record["fhir_resource"]["_extraction_metadata"]["attributes"]["value"] == (
        source_value
    )


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("<5", "mg/dL"),
        ("~5", "mg/dL"),
        ("4-6", "mmol/L"),
        ("120/80", "mmHg"),
    ],
)
def test_closed_mapper_round_trips_non_quantity_lab_values_as_exact_value_string(
    value: str,
    unit: str,
) -> None:
    text = f"Measurement {value} {unit}"
    validated = _validated(
        {
            "labs": [
                _fact(
                    "name",
                    "Measurement",
                    text,
                    value=value,
                    unit=unit,
                )
            ]
        },
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    resource = record["fhir_resource"]
    assert "valueQuantity" not in resource
    assert resource["valueString"] == f"{value} {unit}"


@pytest.mark.parametrize("value", ["positive", "negative", "detected", "not detected"])
def test_closed_mapper_round_trips_qualitative_lab_values_exactly(value: str) -> None:
    text = f"COVID test {value}"
    validated = _validated(
        {"labs": [_fact("name", "COVID test", text, value=value)]},
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["valueString"] == value


def test_qualitative_lab_result_does_not_override_prefix_subject_negation() -> None:
    text = "No COVID test negative"

    with pytest.raises(LocalValidationError, match=r"labs\[0\].*assertion"):
        _validated(
            {"labs": [_fact("name", "COVID test", text, value="negative")]},
            text,
        )


def test_closed_mapper_rejects_qualitative_lab_value_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "COVID test positive"
    validated = _validated(
        {"labs": [_fact("name", "COVID test", text, value="positive")]},
        text,
    )
    real_mapper = entity_to_health_record_dict

    def replace_qualitative_result(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record["fhir_resource"]["valueString"] = "negative"
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        replace_qualitative_result,
    )

    with pytest.raises(
        LocalValidationError,
        match=r"labs\[0\].*lost validated value",
    ):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


def test_closed_mapper_rejects_service_request_for_a_qualitative_lab_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "COVID test positive"
    validated = _validated(
        {"labs": [_fact("name", "COVID test", text, value="positive")]},
        text,
    )
    real_mapper = entity_to_health_record_dict

    def replace_result_with_order(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record["record_type"] = "service_request"
        record["fhir_resource_type"] = "ServiceRequest"
        record["fhir_resource"] = {
            "resourceType": "ServiceRequest",
            "status": "active",
            "intent": "order",
            "code": {"text": "COVID test"},
            "_extraction_metadata": record["fhir_resource"]["_extraction_metadata"],
        }
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        replace_result_with_order,
    )

    with pytest.raises(
        LocalValidationError,
        match=r"labs\[0\].*incompatible record mapping",
    ):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


def test_closed_mapper_rejects_vital_numeric_substring_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Heart rate 50 bpm"
    validated = _validated(
        {
            "vital_signs": [
                _fact(
                    "name",
                    "Heart rate",
                    text,
                    value="50",
                    unit="bpm",
                )
            ]
        },
        text,
    )
    real_mapper = entity_to_health_record_dict

    def mapped_to_different_value(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record["fhir_resource"]["valueString"] = "Heart rate 150 bpm"
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        mapped_to_different_value,
    )

    with pytest.raises(
        LocalValidationError,
        match=r"vital_signs\[0\].*lost validated value",
    ):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


@pytest.mark.parametrize(
    ("value", "unit"),
    [
        ("50", "bpm"),
        ("50.5", "bpm"),
        ("<5", "breaths/min"),
        ("120/80", "mmHg"),
    ],
)
def test_closed_mapper_preserves_exact_vital_value_and_unit(
    value: str,
    unit: str,
) -> None:
    text = f"Measurement {value} {unit}"
    validated = _validated(
        {
            "vital_signs": [
                _fact(
                    "name",
                    "Measurement",
                    text,
                    value=value,
                    unit=unit,
                )
            ]
        },
        text,
    )

    record = validated_extraction_to_health_record_dicts(
        validated,
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["valueString"] == text


def test_closed_mapper_rejects_a_missing_downstream_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )
    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*FHIR"):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


def test_closed_mapper_rejects_medication_remapped_to_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )
    real_mapper = entity_to_health_record_dict

    def remap_to_observation(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record["record_type"] = "observation"
        record["fhir_resource_type"] = "Observation"
        record["fhir_resource"] = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Metformin"},
            "valueString": "Metformin",
            "_extraction_metadata": record["fhir_resource"]["_extraction_metadata"],
        }
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        remap_to_observation,
    )

    with pytest.raises(
        LocalValidationError,
        match=r"medications\[0\].*incompatible record mapping",
    ):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


@pytest.mark.parametrize(
    ("mutated_field", "mutated_value"),
    [
        ("record_type", "observation"),
        ("fhir_resource_type", "Observation"),
    ],
)
def test_closed_mapper_requires_record_and_declared_fhir_types_to_match_category(
    monkeypatch: pytest.MonkeyPatch,
    mutated_field: str,
    mutated_value: str,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )
    real_mapper = entity_to_health_record_dict

    def mutate_declared_type(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record[mutated_field] = mutated_value
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        mutate_declared_type,
    )

    with pytest.raises(LocalValidationError, match="incompatible record mapping"):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


def test_closed_mapper_requires_nested_fhir_resource_type_to_match_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )
    real_mapper = entity_to_health_record_dict

    def mutate_nested_resource_type(*args, **kwargs):
        record = real_mapper(*args, **kwargs)
        assert record is not None
        record["fhir_resource"] = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "Metformin"},
            "valueString": "Metformin",
            "_extraction_metadata": record["fhir_resource"]["_extraction_metadata"],
        }
        return record

    monkeypatch.setattr(
        "app.services.extraction.entity_to_fhir.entity_to_health_record_dict",
        mutate_nested_resource_type,
    )

    with pytest.raises(LocalValidationError, match="incompatible record mapping"):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )


@pytest.mark.parametrize(
    ("raw", "page", "entity_class", "mapping"),
    [
        (
            {
                "medications": [
                    _fact(
                        "name",
                        "Metformin",
                        "Metformin active",
                        status="active",
                    )
                ]
            },
            "Metformin active",
            "medication",
            ("medication", "MedicationRequest", "MedicationRequest"),
        ),
        (
            {
                "conditions": [
                    _fact(
                        "name",
                        "Hypertension",
                        "Hypertension confirmed",
                        assertion="present",
                    )
                ]
            },
            "Hypertension confirmed",
            "condition",
            ("condition", "Condition", "Condition"),
        ),
        (
            {
                "conditions": [
                    _fact(
                        "name",
                        "colon cancer",
                        "Family history: mother had colon cancer",
                        assertion="family_history",
                        relationship="mother",
                    )
                ]
            },
            "Family history: mother had colon cancer",
            "family_history",
            ("family_history", "FamilyMemberHistory", "FamilyMemberHistory"),
        ),
        (
            {
                "procedures": [
                    _fact(
                        "name",
                        "Colonoscopy",
                        "Colonoscopy performed",
                        assertion="present",
                    )
                ]
            },
            "Colonoscopy performed",
            "procedure",
            ("procedure", "Procedure", "Procedure"),
        ),
        (
            {
                "labs": [
                    _fact(
                        "name",
                        "Glucose",
                        "Glucose 95 mg/dL",
                        value="95",
                        unit="mg/dL",
                    )
                ]
            },
            "Glucose 95 mg/dL",
            "lab_result",
            ("observation", "Observation", "Observation"),
        ),
        (
            {"labs": [_fact("name", "CBC", "CBC reported")]},
            "CBC reported",
            "lab_result",
            ("service_request", "ServiceRequest", "ServiceRequest"),
        ),
        (
            {
                "allergies": [
                    _fact(
                        "substance",
                        "Penicillin",
                        "Penicillin allergy rash",
                        reaction="rash",
                        status="active",
                    )
                ]
            },
            "Penicillin allergy rash",
            "allergy",
            ("allergy", "AllergyIntolerance", "AllergyIntolerance"),
        ),
        (
            {
                "encounters": [
                    _fact(
                        "name",
                        "Office visit",
                        "Office visit completed",
                        status="finished",
                    )
                ]
            },
            "Office visit completed",
            "encounter",
            ("encounter", "Encounter", "Encounter"),
        ),
        (
            {
                "immunizations": [
                    _fact(
                        "name",
                        "Influenza vaccine",
                        "Influenza vaccine administered",
                        status="completed",
                    )
                ]
            },
            "Influenza vaccine administered",
            "immunization",
            ("immunization", "Immunization", "Immunization"),
        ),
        (
            {
                "vital_signs": [
                    _fact(
                        "name",
                        "Heart rate",
                        "Heart rate 72 bpm",
                        value="72",
                        unit="bpm",
                    )
                ]
            },
            "Heart rate 72 bpm",
            "vital",
            ("observation", "Observation", "Observation"),
        ),
        (
            {
                "diagnostic_reports": [
                    _fact(
                        "name",
                        "CT chest",
                        "CT chest final showed no acute findings",
                        findings="no acute findings",
                        status="final",
                    )
                ]
            },
            "CT chest final showed no acute findings",
            "imaging_result",
            ("diagnostic_report", "DiagnosticReport", "DiagnosticReport"),
        ),
        (
            {
                "care_plans": [
                    _fact(
                        "title",
                        "Care plan",
                        "Care plan active monitor blood pressure",
                        plan_items=["monitor blood pressure"],
                        status="active",
                    )
                ]
            },
            "Care plan active monitor blood pressure",
            "assessment_plan",
            ("document", "DocumentReference", "DocumentReference"),
        ),
    ],
)
def test_closed_mapper_allows_only_supported_task7_category_mappings(
    raw: dict[str, object],
    page: str,
    entity_class: str,
    mapping: tuple[str, str, str],
) -> None:
    record = validated_extraction_to_health_record_dicts(
        _validated(raw, page),
        uuid4(),
        uuid4(),
    )[0]

    assert record["fhir_resource"]["_extraction_metadata"]["entity_class"] == (
        entity_class
    )
    assert (
        record["record_type"],
        record["fhir_resource_type"],
        record["fhir_resource"]["resourceType"],
    ) == mapping
    assert record["content_hash"] == content_hash(record["fhir_resource"])


def test_closed_mapper_rejects_strict_fhir_validator_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated(
        {
            "medications": [
                _fact(
                    "name",
                    "Metformin",
                    "Metformin active",
                    status="active",
                )
            ]
        },
        "Metformin active",
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr(
        adapters,
        "_strict_fhir_resource_is_valid",
        unavailable,
        raising=False,
    )

    with pytest.raises(LocalValidationError, match=r"medications\[0\].*FHIR"):
        validated_extraction_to_health_record_dicts(
            validated,
            uuid4(),
            uuid4(),
        )
