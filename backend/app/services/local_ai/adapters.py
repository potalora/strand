"""Adapters from validated strict-local extraction to existing entity contracts."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from app.services.extraction.clinical_numbers import parse_clinical_decimal
from app.services.extraction.entity_types import ExtractedEntity
from app.services.extraction.entity_validator import validate_entities
from app.services.ingestion.content_hash import content_hash
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    CLINICAL_EXTRACTION_SCHEMA_VERSION,
    FACT_CATEGORY_NAMES,
    AssertionState,
    ClinicalDocumentExtraction,
    EvidenceFact,
)
from app.services.local_ai.extraction_validator import require_validated_extraction

__all__ = [
    "to_extracted_entities",
    "validated_extraction_to_health_record_dicts",
]

_REFERENCE_RANGE_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*[-\u2013\u2014]\s*([+-]?\d+(?:[.,]\d+)?)\s*$"
)


def _value(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _attributes(fact: EvidenceFact, names: tuple[str, ...]) -> dict[str, object]:
    attributes = {
        name: _value(value)
        for name in names
        if (value := getattr(fact, name, None)) is not None
    }
    attributes["_evidence_ids"] = [fact.evidence_id]
    attributes["_source_page"] = fact.page_number
    attributes["_verbatim"] = fact.verbatim
    attributes["_normalization_method"] = "verbatim-preserved"
    attributes["_normalization_version"] = CLINICAL_EXTRACTION_SCHEMA_VERSION
    return attributes


def _entity(
    fact: EvidenceFact,
    entity_class: str,
    text: str,
    attribute_names: tuple[str, ...],
    *,
    extras: dict[str, object] | None = None,
) -> ExtractedEntity:
    attributes = _attributes(fact, attribute_names)
    if extras:
        attributes.update(
            {key: value for key, value in extras.items() if value is not None}
        )
    return ExtractedEntity(
        entity_class=entity_class,
        text=text,
        attributes=attributes,
        confidence=fact.confidence if fact.confidence is not None else 0.8,
    )


def _condition(fact: EvidenceFact) -> ExtractedEntity | None:
    assertion = fact.assertion
    if assertion == AssertionState.FAMILY_HISTORY:
        return _entity(
            fact,
            "family_history",
            fact.name,
            ("relationship", "date"),
            extras={"condition": fact.name, "status": "historical"},
        )
    status = {
        AssertionState.PRESENT: "active",
        AssertionState.NEGATED: "negated",
    }.get(assertion)
    if status is None:
        return None
    return _entity(fact, "condition", fact.name, ("date",), extras={"status": status})


def _medication(fact: EvidenceFact) -> ExtractedEntity | None:
    if _value(fact.status) != "active":
        return None
    return _entity(
        fact,
        "medication",
        fact.name,
        ("route", "frequency", "status", "date"),
        extras={"value": fact.dose_value, "unit": fact.dose_unit},
    )


def _procedure(fact: EvidenceFact) -> ExtractedEntity | None:
    if fact.assertion in {
        AssertionState.MENTIONED_NOT_PERFORMED,
        AssertionState.NEGATED,
        AssertionState.UNCERTAIN,
        AssertionState.FAMILY_HISTORY,
    }:
        return None
    return _entity(
        fact,
        "procedure",
        fact.name,
        ("date", "provider"),
        extras={"status": "completed"},
    )


def _lab(fact: EvidenceFact) -> ExtractedEntity | None:
    if fact.assertion != AssertionState.PRESENT:
        return None
    extras: dict[str, object] = {"test": fact.name, "status": _value(fact.assertion)}
    if fact.reference_range:
        range_match = _REFERENCE_RANGE_RE.fullmatch(fact.reference_range)
        if range_match:
            extras["ref_low"], extras["ref_high"] = range_match.groups()
    return _entity(
        fact,
        "lab_result",
        fact.name,
        ("value", "unit", "reference_range", "interpretation", "date"),
        extras=extras,
    )


def _allergy(fact: EvidenceFact) -> ExtractedEntity | None:
    if fact.assertion != AssertionState.PRESENT or _value(fact.status) != "active":
        return None
    return _entity(
        fact,
        "allergy",
        fact.substance,
        ("reaction", "severity", "status", "date"),
        extras={"assertion": _value(fact.assertion)},
    )


def _encounter(fact: EvidenceFact) -> ExtractedEntity | None:
    if _value(fact.status) != "finished":
        return None
    return _entity(
        fact,
        "encounter",
        fact.name,
        ("visit_type", "date", "provider", "facility", "status"),
    )


def _immunization(fact: EvidenceFact) -> ExtractedEntity | None:
    if _value(fact.status) not in {"completed", "not_done"}:
        return None
    return _entity(
        fact,
        "immunization",
        fact.name,
        (
            "date",
            "status",
            "route",
            "site",
            "dose",
            "dose_unit",
            "manufacturer",
            "lot",
        ),
        extras={"vaccine": fact.name},
    )


def _vital(fact: EvidenceFact) -> ExtractedEntity | None:
    if fact.assertion != AssertionState.PRESENT:
        return None
    return _entity(
        fact,
        "vital",
        f"{fact.name} {fact.value}{f' {fact.unit}' if fact.unit else ''}",
        ("value", "unit", "date"),
        extras={"type": fact.name, "assertion": _value(fact.assertion)},
    )


def _diagnostic_report(fact: EvidenceFact) -> ExtractedEntity | None:
    if fact.assertion != AssertionState.PRESENT or _value(fact.status) != "final":
        return None
    return _entity(
        fact,
        "imaging_result",
        fact.name,
        (
            "findings",
            "interpretation",
            "date",
            "category",
            "performer",
            "status",
        ),
        extras={"procedure_name": fact.name, "assertion": _value(fact.assertion)},
    )


def _care_plan(fact: EvidenceFact) -> ExtractedEntity | None:
    if _value(fact.status) != "active":
        return None
    return _entity(
        fact,
        "assessment_plan",
        fact.title,
        ("plan_items", "status", "date"),
    )


_ADAPTERS: dict[str, Callable[[EvidenceFact], ExtractedEntity | None]] = {
    "medications": _medication,
    "conditions": _condition,
    "procedures": _procedure,
    "labs": _lab,
    "allergies": _allergy,
    "encounters": _encounter,
    "immunizations": _immunization,
    "vital_signs": _vital,
    "diagnostic_reports": _diagnostic_report,
    "care_plans": _care_plan,
}

_ALLOWED_RECORD_MAPPINGS: dict[
    tuple[str, str],
    frozenset[tuple[str, str, str]],
] = {
    ("medications", "medication"): frozenset(
        {("medication", "MedicationRequest", "MedicationRequest")}
    ),
    ("conditions", "condition"): frozenset({("condition", "Condition", "Condition")}),
    ("conditions", "family_history"): frozenset(
        {("family_history", "FamilyMemberHistory", "FamilyMemberHistory")}
    ),
    ("procedures", "procedure"): frozenset({("procedure", "Procedure", "Procedure")}),
    ("labs", "lab_result"): frozenset(
        {
            ("observation", "Observation", "Observation"),
            ("service_request", "ServiceRequest", "ServiceRequest"),
        }
    ),
    ("allergies", "allergy"): frozenset(
        {("allergy", "AllergyIntolerance", "AllergyIntolerance")}
    ),
    ("encounters", "encounter"): frozenset({("encounter", "Encounter", "Encounter")}),
    ("immunizations", "immunization"): frozenset(
        {("immunization", "Immunization", "Immunization")}
    ),
    ("vital_signs", "vital"): frozenset(
        {("observation", "Observation", "Observation")}
    ),
    ("diagnostic_reports", "imaging_result"): frozenset(
        {("diagnostic_report", "DiagnosticReport", "DiagnosticReport")}
    ),
    ("care_plans", "assessment_plan"): frozenset(
        {("document", "DocumentReference", "DocumentReference")}
    ),
}


def _adapt_fact(
    fact: EvidenceFact,
    category: str,
    index: int,
) -> ExtractedEntity | None:
    if fact.evidence_id is None:
        raise LocalValidationError(f"{category}[{index}]: evidence validation required")
    entity = _ADAPTERS[category](fact)
    if entity is None:
        return None
    guarded = validate_entities([entity], log_rejections=False)
    if len(guarded) != 1:
        raise LocalValidationError(
            f"{category}[{index}]: downstream guard rejected fact"
        )
    return guarded[0]


def _iter_adapted_facts(
    validated: ClinicalDocumentExtraction,
) -> Iterator[tuple[str, int, EvidenceFact, ExtractedEntity]]:
    require_validated_extraction(validated)
    for category in FACT_CATEGORY_NAMES:
        for index, fact in enumerate(getattr(validated, category)):
            entity = _adapt_fact(fact, category, index)
            if entity is not None:
                yield category, index, fact, entity


def to_extracted_entities(
    validated: ClinicalDocumentExtraction,
) -> list[ExtractedEntity]:
    """Adapt validated facts to the existing mutable entity contract.

    Entity objects are intentionally not trust capabilities. The generic FHIR
    mapper ignores their local-evidence attributes even when a caller copies,
    subclasses, or mutates them.
    """
    return [entity for _, _, _, entity in _iter_adapted_facts(validated)]


def _approved_provenance(fact: EvidenceFact) -> dict[str, object]:
    return {
        "_evidence_ids": [fact.evidence_id],
        "_source_page": fact.page_number,
        "_verbatim": fact.verbatim,
        "_normalization_method": "verbatim-preserved",
        "_normalization_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
    }


def _strict_fhir_resource_is_valid(resource: dict[str, Any]) -> bool:
    """Validate the closed-mapper output without the legacy fail-open wrapper."""
    if not isinstance(resource, dict):
        return False
    resource_type = resource.get("resourceType")
    if not isinstance(resource_type, str) or not resource_type:
        return False
    module = importlib.import_module(f"fhir.resources.R4B.{resource_type.lower()}")
    model = getattr(module, resource_type)
    cleaned = {
        key: value for key, value in resource.items() if key != "_extraction_metadata"
    }
    try:
        model.model_validate(cleaned)
    except ValidationError as error:
        problems = [
            item
            for item in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
            if not (
                item.get("type") == "missing"
                and item.get("loc")
                and str(item["loc"][0]) in {"patient", "subject"}
            )
        ]
        return not problems
    return True


def _record_mapping_is_compatible(
    category: str,
    fact: EvidenceFact,
    entity: ExtractedEntity,
    record: object,
) -> bool:
    """Require the closed mapper's category, entity, and three type fields to agree."""
    if not isinstance(record, dict):
        return False
    resource = record.get("fhir_resource")
    if not isinstance(resource, dict):
        return False
    actual = (
        record.get("record_type"),
        record.get("fhir_resource_type"),
        resource.get("resourceType"),
    )
    allowed = _ALLOWED_RECORD_MAPPINGS.get((category, entity.entity_class))
    if allowed is None or actual not in allowed:
        return False
    if actual == ("service_request", "ServiceRequest", "ServiceRequest"):
        from app.services.extraction.entity_to_fhir import _is_panel_order

        return category == "labs" and fact.value is None and _is_panel_order(entity)
    return True


def _numeric_equivalent(source: str, mapped: object) -> bool:
    source_number = parse_clinical_decimal(source)
    mapped_number = parse_clinical_decimal(mapped)
    if source_number is None or mapped_number is None:
        return False
    return source_number == mapped_number


def _quantity_matches(
    quantity: object,
    *,
    value: str,
    unit: str | None,
) -> bool:
    if not isinstance(quantity, dict) or not _numeric_equivalent(
        value, quantity.get("value")
    ):
        return False
    return unit is None or quantity.get("unit") == unit


def _mapped_fact_matches_fhir(
    fact: EvidenceFact,
    category: str,
    resource: dict[str, Any],
) -> bool:
    if category == "medications" and fact.dose_value is not None:
        instructions = resource.get("dosageInstruction")
        if not isinstance(instructions, list) or not instructions:
            return False
        rates = instructions[0].get("doseAndRate")
        return (
            isinstance(rates, list)
            and bool(rates)
            and _quantity_matches(
                rates[0].get("doseQuantity"),
                value=fact.dose_value,
                unit=fact.dose_unit,
            )
        )
    if category == "labs" and fact.value is not None:
        if parse_clinical_decimal(fact.value) is not None:
            return _quantity_matches(
                resource.get("valueQuantity"),
                value=fact.value,
                unit=fact.unit,
            )
        expected = f"{fact.value}{f' {fact.unit}' if fact.unit is not None else ''}"
        return resource.get("valueString") == expected
    if category == "immunizations" and fact.dose is not None:
        return _quantity_matches(
            resource.get("doseQuantity"),
            value=fact.dose,
            unit=fact.dose_unit,
        )
    if category == "vital_signs":
        mapped = resource.get("valueString")
        expected = (
            f"{fact.name} {fact.value}"
            f"{f' {fact.unit}' if fact.unit is not None else ''}"
        )
        return isinstance(mapped, str) and mapped == expected
    return True


def validated_extraction_to_health_record_dicts(
    validated: ClinicalDocumentExtraction,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None = None,
    document_date: datetime | None = None,
    document_provider: str | None = None,
) -> list[dict]:
    """Validate, adapt, and map to FHIR in one provenance trust boundary.

    No entity verifier or attachable capability is exposed. Provenance is
    copied from the still-registered validated fact only after the generic
    entity mapper has produced its record.
    """
    from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

    records: list[dict] = []
    for category, index, fact, entity in _iter_adapted_facts(validated):
        record = entity_to_health_record_dict(
            entity,
            user_id,
            patient_id,
            source_file_id,
            document_date,
            document_provider,
        )
        if record is None:
            raise LocalValidationError(
                f"{category}[{index}]: FHIR mapping returned no record"
            )
        if not _record_mapping_is_compatible(category, fact, entity, record):
            raise LocalValidationError(
                f"{category}[{index}]: incompatible record mapping"
            )
        resource = record["fhir_resource"]
        try:
            structurally_valid = _strict_fhir_resource_is_valid(resource)
        except Exception:
            structurally_valid = False
        if not structurally_valid:
            raise LocalValidationError(
                f"{category}[{index}]: FHIR mapping failed validation"
            )
        if not _mapped_fact_matches_fhir(fact, category, resource):
            raise LocalValidationError(
                f"{category}[{index}]: FHIR mapping lost validated value or unit"
            )
        resource["_extraction_metadata"].update(_approved_provenance(fact))
        record["content_hash"] = content_hash(resource)
        records.append(record)
    return records
