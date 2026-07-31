"""Truthful, allowlisted HealthRecord projections for grounded local summaries."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence
from uuid import UUID

from app.models.local_ai import ExtractionEvidence
from app.models.record import HealthRecord
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.grounded_summary import (
    _STATUS_VALUES,
    normalize_observation_summary_value,
    typed_observation_quantity,
    typed_observation_ratio,
)

_TYPE_ALIASES = {
    "family_history": "condition",
    "imaging": "imaging_study",
}
_SUPPORTED_TYPES = frozenset(_STATUS_VALUES)
_ASSERTIONS = frozenset(
    {"present", "negated", "family_history", "uncertain", "mentioned_not_performed"}
)
_SEVERITIES = frozenset({"mild", "moderate", "severe", "unknown"})
_PATH_TOKEN = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)(?:\[\d+\])?$")
_VERIFICATION_SYSTEMS = {
    "allergy": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
    "condition": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
}
_VERIFICATION_CODES = {
    "allergy": frozenset(
        {"confirmed", "entered-in-error", "presumed", "refuted", "unconfirmed"}
    ),
    "condition": frozenset(
        {
            "confirmed",
            "differential",
            "entered-in-error",
            "provisional",
            "refuted",
            "unconfirmed",
        }
    ),
}

_PATH_ALIASES: dict[str, dict[str, str]] = {
    "allergy": {
        "name": "/substance",
        "substance": "/substance",
        "allergen": "/substance",
        "reaction": "/reaction",
        "severity": "/severity",
        "status": "/status",
        "assertion": "/assertion",
        "date": "/date",
    },
    "condition": {
        "name": "/diagnosis",
        "condition": "/diagnosis",
        "diagnosis": "/diagnosis",
        "relationship": "/relationship",
        "status": "/status",
        "assertion": "/assertion",
        "date": "/date",
        "onset_date": "/date",
    },
    "medication": {
        "name": "/name",
        "medication": "/name",
        "drug": "/name",
        "dose": "/dose",
        "dosage": "/dose",
        "dose_value": "/dose_value",
        "value": "/dose_value",
        "dose_unit": "/dose_unit",
        "unit": "/dose_unit",
        "route": "/route",
        "frequency": "/frequency",
        "status": "/status",
        "date": "/effective_date",
        "effective_date": "/effective_date",
        "start_date": "/effective_date",
        "end_date": "/end_date",
    },
    "observation": {
        "name": "/name",
        "test": "/name",
        "type": "/name",
        "value": "/value",
        "unit": "/unit",
        "reference_range": "/reference_range",
        "interpretation": "/interpretation",
        "category": "/category",
        "status": "/status",
        "date": "/date",
    },
    "procedure": {
        "name": "/name",
        "procedure": "/name",
        "provider": "/provider",
        "body_site": "/body_site",
        "status": "/status",
        "assertion": "/assertion",
        "date": "/date",
    },
    "immunization": {
        "name": "/name",
        "vaccine": "/name",
        "vaccine_name": "/name",
        "immunization": "/name",
        "dose": "/dose",
        "lot": "/lot",
        "manufacturer": "/manufacturer",
        "route": "/route",
        "site": "/site",
        "status": "/status",
        "date": "/date",
    },
    "encounter": {
        "name": "/name",
        "type": "/name",
        "visit_type": "/visit_type",
        "provider": "/provider",
        "facility": "/facility",
        "status": "/status",
        "date": "/date",
    },
    "diagnostic_report": {
        "name": "/name",
        "procedure_name": "/name",
        "findings": "/findings",
        "interpretation": "/interpretation",
        "performer": "/performer",
        "provider": "/performer",
        "category": "/category",
        "status": "/status",
        "assertion": "/assertion",
        "date": "/date",
    },
    "imaging_study": {
        "name": "/name",
        "procedure_name": "/name",
        "findings": "/description",
        "description": "/description",
        "modality": "/modality",
        "body_site": "/body_site",
        "status": "/status",
        "date": "/date",
    },
    "care_plan": {
        "name": "/title",
        "title": "/title",
        "plan_items": "/plan_items",
        "status": "/status",
        "date": "/date",
    },
}


@dataclass(frozen=True)
class SummaryProjection:
    """Raw candidates consumed by ``build_grounded_summary_input``."""

    facts: list[dict[str, object]]
    evidence: list[dict[str, object]]
    uncertainty_labels: list[dict[str, object]]


def normalize_summary_record_type(record_type: str) -> str:
    """Return the grounded-summary type for a persisted record type."""
    normalized = _TYPE_ALIASES.get(record_type, record_type)
    if normalized not in _SUPPORTED_TYPES:
        raise LocalValidationError(
            "Record type is not supported for strict-local summary."
        )
    return normalized


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        candidate = value.get("text") or value.get("display")
        if candidate is None:
            coding = value.get("coding")
            if isinstance(coding, list) and coding and isinstance(coding[0], Mapping):
                candidate = coding[0].get("display") or coding[0].get("code")
        value = candidate
    if not isinstance(value, (str, int, float)):
        return None
    result = " ".join(str(value).split())
    return result or None


def _concept(value: object) -> str | None:
    return _text(value)


def _first(items: object) -> Mapping[str, Any]:
    if isinstance(items, list) and items and isinstance(items[0], Mapping):
        return items[0]
    return {}


def _date_value(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    raw = value.strip()
    try:
        if len(raw) == 10:
            return date.fromisoformat(raw).isoformat()
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _put(content: dict[str, object], key: str, value: object) -> None:
    if isinstance(value, str):
        value = _text(value)
    if value is not None and value != "" and value != [] and value != {}:
        content[key] = value


def _status(record_type: str, *values: object) -> str | None:
    allowed = _STATUS_VALUES[record_type]
    for value in values:
        candidate = _text(value)
        if candidate is None:
            continue
        normalized = candidate.casefold().replace(" ", "-")
        if normalized in allowed:
            return normalized
        underscored = normalized.replace("-", "_")
        if underscored in allowed:
            return underscored
    return None


def _verification_assertion(
    resource: Mapping[str, Any],
    record_type: str,
) -> str | None:
    """Map structured FHIR verification state to grounded assertion semantics."""
    if record_type not in {"allergy", "condition"}:
        return None
    value = resource.get("verificationStatus")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise LocalValidationError(
            "Structured FHIR verification status is not eligible for summary."
        )
    verification_codes: list[str] = []
    codings = value.get("coding")
    if not isinstance(codings, list) or not codings:
        raise LocalValidationError(
            "Structured FHIR verification status is not eligible for summary."
        )
    for coding in codings:
        if not isinstance(coding, Mapping):
            raise LocalValidationError(
                "Structured FHIR verification status is not eligible for summary."
            )
        system = coding.get("system")
        if system != _VERIFICATION_SYSTEMS[record_type]:
            raise LocalValidationError(
                "Structured FHIR verification status is not eligible for summary."
            )
        code = coding.get("code")
        if not isinstance(code, str) or code not in _VERIFICATION_CODES[record_type]:
            raise LocalValidationError(
                "Structured FHIR verification status is not eligible for summary."
            )
        verification_codes.append(code)
    normalized = set(verification_codes)
    if not normalized or len(normalized) != 1 or "entered-in-error" in normalized:
        raise LocalValidationError(
            "Structured FHIR verification status is not eligible for summary."
        )
    if "refuted" in normalized:
        return "negated"
    if normalized.intersection(
        {"differential", "presumed", "provisional", "unconfirmed"}
    ):
        return "uncertain"
    if "confirmed" in normalized:
        return "present"
    return None


def _metadata(
    resource: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any], str | None]:
    raw = resource.get("_extraction_metadata")
    if not isinstance(raw, Mapping):
        return None, {}, None
    attributes = raw.get("attributes")
    return (
        _text(raw.get("entity_class")),
        dict(attributes) if isinstance(attributes, Mapping) else {},
        _text(raw.get("original_text")),
    )


def _quantity(
    value: object,
    *,
    positive: bool = True,
) -> tuple[object | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    amount = value.get("value")
    if type(amount) in {int, float}:
        try:
            finite = math.isfinite(amount)
        except OverflowError:
            finite = False
        if not finite:
            raise LocalValidationError("Structured clinical quantity must be finite.")
    if type(amount) not in {int, float} or (positive and amount <= 0):
        amount = None
    return amount, _text(value.get("unit") or value.get("code"))


def _base_projection(record: HealthRecord, record_type: str) -> dict[str, object]:
    resource = record.fhir_resource if isinstance(record.fhir_resource, Mapping) else {}
    _, attrs, original_text = _metadata(resource)
    content: dict[str, object] = {"record_type": record_type}
    display = _text(record.display_text) or original_text
    effective = (
        _date_value(record.effective_date)
        or _date_value(resource.get("effectiveDateTime"))
        or _date_value(resource.get("authoredOn"))
        or _date_value(resource.get("issued"))
        or _date_value(resource.get("performedDateTime"))
        or _date_value(resource.get("occurrenceDateTime"))
        or _date_value(resource.get("date"))
    )
    status = _status(
        record_type, resource.get("status"), record.status, attrs.get("status")
    )
    verification_assertion = _verification_assertion(resource, record_type)
    if verification_assertion == "negated":
        status = None

    if record_type == "medication":
        name = (
            _concept(resource.get("medicationCodeableConcept"))
            or _concept(resource.get("medication"))
            or _text(attrs.get("medication") or attrs.get("name"))
            or display
        )
        _put(content, "name", name)
        is_statement = resource.get("resourceType") == "MedicationStatement"
        dosage = _first(
            resource.get("dosage")
            if is_statement
            else resource.get("dosageInstruction")
        )
        dose_and_rate = _first(dosage.get("doseAndRate"))
        amount, unit = _quantity(dose_and_rate.get("doseQuantity"))
        dose_text = (_text(dosage.get("text")) if is_statement else None) or _text(
            attrs.get("dose") or attrs.get("dosage")
        )
        _put(content, "dose", dose_text)
        _put(content, "dose_value", amount)
        _put(content, "dose_unit", unit)
        _put(content, "route", _concept(dosage.get("route")) or attrs.get("route"))
        timing = dosage.get("timing")
        timing_code = timing.get("code") if isinstance(timing, Mapping) else None
        _put(
            content,
            "frequency",
            _concept(timing_code) or attrs.get("frequency"),
        )
        effective_period = resource.get("effectivePeriod") if is_statement else None
        period_start = (
            _date_value(effective_period.get("start"))
            if isinstance(effective_period, Mapping)
            else None
        )
        period_end = (
            _date_value(effective_period.get("end"))
            if isinstance(effective_period, Mapping)
            else None
        )
        _put(content, "effective_date", period_start or effective)
        end_date = period_end if is_statement else _date_value(attrs.get("end_date"))
        _put(content, "end_date", end_date)
    elif record_type == "observation":
        _put(
            content,
            "name",
            _concept(resource.get("code")) or attrs.get("test") or display,
        )
        raw_quantity = resource.get("valueQuantity")
        quantity_value, quantity_unit = _quantity(raw_quantity, positive=False)
        comparator = (
            raw_quantity.get("comparator")
            if isinstance(raw_quantity, Mapping)
            else None
        )
        value: object | None
        unit: object | None
        try:
            if quantity_value is not None and comparator is not None:
                value = typed_observation_quantity(
                    quantity_value,
                    comparator,
                    quantity_unit or attrs.get("unit"),
                )
                unit = None
            elif isinstance(resource.get("valueRatio"), Mapping):
                ratio = resource["valueRatio"]
                numerator, numerator_unit = _quantity(
                    ratio.get("numerator"),
                    positive=False,
                )
                denominator, denominator_unit = _quantity(
                    ratio.get("denominator"),
                    positive=False,
                )
                if numerator is None or denominator is None:
                    raise ValueError("summary ratio is incomplete")
                value = typed_observation_ratio(
                    numerator,
                    denominator,
                    numerator_unit,
                    denominator_unit,
                )
                unit = None
            else:
                value = quantity_value
                if value is None and attrs.get("value") is not None:
                    value = attrs.get("value")
                if value is None:
                    for key in ("valueString", "valueInteger", "valueBoolean"):
                        if key in resource:
                            value = resource[key]
                            break
                value, unit = normalize_observation_summary_value(
                    value,
                    quantity_unit or attrs.get("unit"),
                )
        except ValueError:
            raise LocalValidationError(
                "Observation result is not eligible for strict-local summary."
            ) from None
        _put(content, "value", value)
        _put(content, "unit", unit)
        reference = _first(resource.get("referenceRange"))
        reference_text = _text(reference.get("text"))
        if reference_text is None and reference:
            low, low_unit = _quantity(reference.get("low"), positive=False)
            high, high_unit = _quantity(reference.get("high"), positive=False)
            if low is not None or high is not None:
                reference_text = (
                    f"{low if low is not None else ''}-"
                    f"{high if high is not None else ''} {low_unit or high_unit or ''}"
                ).strip()
        _put(content, "reference_range", reference_text or attrs.get("reference_range"))
        _put(
            content,
            "interpretation",
            _concept(_first(resource.get("interpretation")))
            or attrs.get("interpretation"),
        )
        _put(content, "category", _concept(_first(resource.get("category"))))
        _put(content, "date", effective)
    elif record_type == "allergy":
        _put(content, "substance", _concept(resource.get("code")) or display)
        reaction = _first(resource.get("reaction"))
        _put(
            content,
            "reaction",
            _concept(_first(reaction.get("manifestation"))) or attrs.get("reaction"),
        )
        severity = _text(reaction.get("severity") or attrs.get("severity"))
        if severity and severity.casefold() in _SEVERITIES:
            content["severity"] = severity.casefold()
        _put(content, "date", effective)
    elif record_type == "condition":
        family = record.record_type == "family_history"
        condition = _first(resource.get("condition")) if family else {}
        _put(
            content,
            "diagnosis",
            _concept(condition.get("code"))
            or _concept(resource.get("code"))
            or attrs.get("condition")
            or display,
        )
        relationship = _concept(resource.get("relationship")) or attrs.get(
            "relationship"
        )
        if family:
            if not _text(relationship):
                raise LocalValidationError(
                    "Family-history record is missing its source relationship."
                )
            content["assertion"] = "family_history"
            _put(content, "relationship", relationship)
            status = None
        _put(content, "date", effective)
    elif record_type == "procedure":
        _put(content, "name", _concept(resource.get("code")) or display)
        performer = _first(resource.get("performer"))
        actor = performer.get("actor") if isinstance(performer, Mapping) else None
        _put(content, "provider", _concept(actor) or attrs.get("provider"))
        _put(
            content,
            "body_site",
            _concept(_first(resource.get("bodySite"))) or attrs.get("body_site"),
        )
        _put(content, "date", effective)
    elif record_type == "appointment":
        _put(
            content,
            "title",
            _text(resource.get("description"))
            or _concept(resource.get("appointmentType"))
            or display,
        )
        _put(content, "start", _date_value(resource.get("start")) or effective)
        _put(
            content,
            "end",
            _date_value(resource.get("end")) or _date_value(record.effective_date_end),
        )
        participant = _first(resource.get("participant"))
        _put(content, "provider", _concept(participant.get("actor")))
        _put(
            content, "location", _concept(_first(resource.get("supportingInformation")))
        )
    elif record_type == "care_plan":
        _put(content, "title", _text(resource.get("title")) or display)
        activities = resource.get("activity")
        if isinstance(activities, list):
            items = [
                value
                for item in activities
                if isinstance(item, Mapping)
                for value in [
                    _text((item.get("detail") or {}).get("description"))
                    if isinstance(item.get("detail"), Mapping)
                    else None
                ]
                if value
            ]
            _put(content, "plan_items", items)
        _put(content, "date", effective)
    elif record_type == "care_team":
        _put(content, "name", _text(resource.get("name")) or display)
        participants = []
        for item in (
            resource.get("participant", [])
            if isinstance(resource.get("participant"), list)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            member = _concept(item.get("member"))
            role = _concept(_first(item.get("role")))
            projected = {
                key: value for key, value in (("name", member), ("role", role)) if value
            }
            if projected:
                participants.append(projected)
        _put(content, "participants", participants)
        _put(content, "date", effective)
    elif record_type == "communication":
        _put(
            content,
            "topic",
            _concept(resource.get("topic"))
            or _concept(_first(resource.get("category")))
            or display,
        )
        _put(content, "category", _concept(_first(resource.get("category"))))
        _put(content, "sender", _concept(resource.get("sender")))
        _put(content, "recipient", _concept(_first(resource.get("recipient"))))
        _put(content, "date", effective)
    elif record_type == "diagnostic_report":
        _put(content, "name", _concept(resource.get("code")) or display)
        _put(
            content,
            "findings",
            _text(resource.get("conclusion")) or attrs.get("findings"),
        )
        _put(content, "interpretation", attrs.get("interpretation"))
        _put(content, "performer", _concept(_first(resource.get("performer"))))
        category_text = (_concept(_first(resource.get("category"))) or "").casefold()
        if category_text:
            category = next(
                (
                    candidate
                    for candidate in (
                        "imaging",
                        "laboratory",
                        "pathology",
                        "endoscopy",
                        "nuclear_medicine",
                        "pulmonary",
                        "laboratory_panel",
                    )
                    if candidate.replace("_", " ") in category_text
                ),
                "other",
            )
            content["category"] = category
        _put(content, "date", effective)
    elif record_type == "document":
        _put(content, "title", _text(resource.get("description")) or display)
        _put(content, "document_type", _concept(resource.get("type")))
        _put(content, "author", _concept(_first(resource.get("author"))))
        _put(content, "date", effective)
    elif record_type == "encounter":
        _put(content, "name", _concept(resource.get("type")) or display)
        class_value = _text(resource.get("class")) or ""
        visit_type = "other"
        folded = class_value.casefold()
        if "virtual" in folded or "tele" in folded:
            visit_type = "telehealth"
        elif "emergency" in folded:
            visit_type = "emergency"
        elif "inpatient" in folded:
            visit_type = "inpatient"
        elif "outpatient" in folded or "ambulatory" in folded or "office" in folded:
            visit_type = "office"
        if class_value:
            _put(content, "visit_type", visit_type)
        participant = _first(resource.get("participant"))
        individual = (
            participant.get("individual") if isinstance(participant, Mapping) else None
        )
        _put(content, "provider", _concept(individual))
        _put(content, "facility", _concept(resource.get("serviceProvider")))
        _put(content, "date", effective)
    elif record_type == "imaging_study":
        _put(content, "name", _text(resource.get("description")) or display)
        series = _first(resource.get("series"))
        _put(
            content,
            "modality",
            _concept(series.get("modality"))
            or _concept(_first(resource.get("modality"))),
        )
        _put(content, "body_site", _concept(series.get("bodySite")))
        _put(
            content,
            "description",
            _text(series.get("description")) or attrs.get("findings"),
        )
        _put(content, "date", effective)
    elif record_type == "immunization":
        _put(content, "name", _concept(resource.get("vaccineCode")) or display)
        amount, unit = _quantity(resource.get("doseQuantity"))
        _put(
            content,
            "dose",
            " ".join(str(value) for value in (amount, unit) if value is not None)
            if amount is not None
            else None,
        )
        _put(content, "lot", resource.get("lotNumber"))
        _put(content, "manufacturer", _concept(resource.get("manufacturer")))
        _put(content, "route", _concept(resource.get("route")))
        _put(content, "site", _concept(resource.get("site")))
        _put(content, "date", effective)
    elif record_type == "questionnaire_response":
        _put(content, "title", _concept(resource.get("questionnaire")) or display)
        _put(content, "questionnaire", _concept(resource.get("questionnaire")))
        answers = []
        for item in (
            resource.get("item", []) if isinstance(resource.get("item"), list) else []
        ):
            if not isinstance(item, Mapping):
                continue
            answer = _first(item.get("answer"))
            answer_value = next(
                (value for key, value in answer.items() if key.startswith("value")),
                None,
            )
            projected = {
                key: value
                for key, value in (
                    ("question", _text(item.get("text"))),
                    ("answer", _concept(answer_value) or answer_value),
                )
                if value is not None
            }
            if projected:
                answers.append(projected)
        _put(content, "answers", answers)
        _put(content, "date", effective)
    elif record_type == "service_request":
        _put(content, "name", _concept(resource.get("code")) or display)
        _put(content, "intent", _text(resource.get("intent")))
        _put(content, "priority", _text(resource.get("priority")))
        _put(content, "requester", _concept(resource.get("requester")))
        _put(content, "performer", _concept(_first(resource.get("performer"))))
        _put(content, "date", effective)

    assertion = verification_assertion or _text(attrs.get("assertion"))
    if (
        assertion
        and assertion in _ASSERTIONS
        and record_type
        in {
            "allergy",
            "condition",
            "diagnostic_report",
            "procedure",
        }
    ):
        content["assertion"] = assertion
    if status is not None:
        content["status"] = status
    return content


def _leaf_paths(value: object, path: str = "") -> list[str]:
    if isinstance(value, Mapping) and value:
        result: list[str] = []
        for key in sorted(value):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            result.extend(_leaf_paths(value[key], f"{path}/{escaped}"))
        return result
    if isinstance(value, list) and value:
        result = []
        for index, item in enumerate(value):
            result.extend(_leaf_paths(item, f"{path}/{index}"))
        return result
    return [path]


def _mapped_paths(
    record_type: str,
    raw_paths: object,
    known_paths: set[str],
) -> list[str]:
    result: list[str] = []
    aliases = _PATH_ALIASES.get(record_type, {})
    if isinstance(raw_paths, list):
        for raw in raw_paths:
            if not isinstance(raw, str):
                continue
            exact_pointer = raw.startswith("/") and raw in known_paths
            source_field: str | None = None
            if exact_pointer:
                candidate = raw
            else:
                match = _PATH_TOKEN.search(raw)
                source_field = match.group(1) if match else None
                candidate = aliases.get(source_field) if source_field else None
            typed_observation_value = record_type == "observation" and any(
                path.startswith("/value/") for path in known_paths
            )
            if exact_pointer:
                candidates = [candidate]
            elif typed_observation_value and source_field == "value":
                candidates = sorted(
                    path
                    for path in known_paths
                    if path.startswith("/value/")
                    and not path.endswith("_unit")
                    and path != "/value/unit"
                )
            elif typed_observation_value and source_field == "unit":
                candidates = sorted(
                    path
                    for path in known_paths
                    if path == "/value/unit" or path.endswith("_unit")
                )
            else:
                candidates = (
                    [candidate]
                    if candidate in known_paths
                    else sorted(
                        path
                        for path in known_paths
                        if candidate is not None and path.startswith(f"{candidate}/")
                    )
                )
            for mapped in candidates:
                if mapped not in result:
                    result.append(mapped)
    return result


def _structured_evidence(
    record: HealthRecord,
    content: Mapping[str, object],
) -> dict[str, object]:
    excerpt = "Record-derived projection: " + json.dumps(
        content,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "id": f"structured_{record.id.hex}",
        "excerpt": excerpt[:2000],
        "page_number": None,
        "section": "Structured health record",
        "field_paths": _leaf_paths(content)[:32],
    }


def _uncertainties(
    record: HealthRecord,
    content: Mapping[str, object],
    evidence_ids: list[str],
    evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    templates: list[str] = []
    if not any(
        content.get(field)
        for field in ("date", "effective_date", "start", "end", "end_date")
    ):
        templates.append("record_date_missing")
    if not any(content.get(field) for field in ("status", "statuses")):
        templates.append("record_status_missing")
    if content["record_type"] == "medication":
        dosage = content.get("dosage")
        dosage_amount = dosage.get("amount") if isinstance(dosage, Mapping) else None
        if not any((content.get("dose"), content.get("dose_value"), dosage_amount)):
            templates.append("medication_dose_missing")
        if not content.get("end_date"):
            templates.append("medication_end_date_missing")
    if (content["record_type"] == "observation" and content.get("value") is None) or (
        content["record_type"] == "diagnostic_report" and not content.get("findings")
    ):
        templates.append("result_missing")
    if content.get("assertion") == "uncertain":
        templates.append("assertion_uncertain")
    if not any(
        item.get("page_number") is not None or item.get("section") for item in evidence
    ):
        templates.append("source_detail_missing")
    return [
        {
            "template_id": template,
            "record_ids": [str(record.id)],
            "evidence_ids": evidence_ids,
        }
        for template in templates
    ]


def project_summary_records(
    records: Sequence[HealthRecord],
    evidence_by_record: Mapping[UUID, Sequence[ExtractionEvidence]],
) -> SummaryProjection:
    """Project every selected record or fail visibly when it cannot be grounded."""
    facts: list[dict[str, object]] = []
    evidence_payload: list[dict[str, object]] = []
    uncertainty_payload: list[dict[str, object]] = []
    for record in records:
        record_type = normalize_summary_record_type(record.record_type)
        content = _base_projection(record, record_type)
        known_paths = set(_leaf_paths(content))
        linked = list(evidence_by_record.get(record.id, ()))
        if len(linked) > 32:
            raise LocalValidationError(
                "Record has too much source evidence for summary."
            )

        record_evidence: list[dict[str, object]] = []
        if linked:
            for item in linked:
                paths = _mapped_paths(record_type, item.field_paths, known_paths)
                if not paths:
                    raise LocalValidationError(
                        "Extracted record source evidence does not map to projected fields."
                    )
                excerpt = item.excerpt
                if not isinstance(excerpt, str) or not excerpt.strip():
                    raise LocalValidationError(
                        "Extracted record has invalid source evidence."
                    )
                source = {
                    "id": str(item.id),
                    "excerpt": excerpt.strip(),
                    "page_number": item.page_number,
                    "section": _text(item.section),
                    "field_paths": paths,
                }
                record_evidence.append(source)
        elif record.ai_extracted or record.source_format == "local_ai":
            raise LocalValidationError(
                "Extracted record has no source evidence for strict-local summary."
            )
        else:
            record_evidence.append(_structured_evidence(record, content))

        supported_paths = {
            str(path)
            for item in record_evidence
            for path in item["field_paths"]
            if isinstance(path, str)
        }
        qualifier_paths = {
            path
            for path in _leaf_paths(content)
            if path in {"/assertion", "/relationship", "/status"}
            or path.startswith("/statuses/")
        }
        if qualifier_paths - supported_paths:
            raise LocalValidationError(
                "Summary fact safety qualifier lacks evidence support."
            )

        evidence_ids = [str(item["id"]) for item in record_evidence]
        facts.append(
            {
                "record_id": str(record.id),
                "content": content,
                "evidence_ids": evidence_ids,
            }
        )
        evidence_payload.extend(record_evidence)
        uncertainty_payload.extend(
            _uncertainties(record, content, evidence_ids, record_evidence)
        )
    return SummaryProjection(
        facts=facts,
        evidence=evidence_payload,
        uncertainty_labels=uncertainty_payload,
    )
