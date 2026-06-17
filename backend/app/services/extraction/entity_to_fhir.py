from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from uuid import UUID, uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.ingestion.content_hash import content_hash
from app.utils.date_utils import parse_datetime

logger = logging.getLogger(__name__)

# Map LangExtract entity classes to FHIR record types
ENTITY_TO_RECORD_TYPE: dict[str, tuple[str, str] | None] = {
    # Existing
    "medication": ("medication", "MedicationRequest"),
    "condition": ("condition", "Condition"),
    "lab_result": ("observation", "Observation"),
    "vital": ("observation", "Observation"),
    "procedure": ("procedure", "Procedure"),
    "allergy": ("allergy", "AllergyIntolerance"),
    # New
    "encounter": ("encounter", "Encounter"),
    "imaging_result": ("diagnostic_report", "DiagnosticReport"),
    "family_history": ("family_history", "FamilyMemberHistory"),
    "assessment_plan": ("document", "DocumentReference"),
    "social_history": ("observation", "Observation"),
    # Non-storable types (return None)
    "provider": None,
    "dosage": None,
    "route": None,
    "frequency": None,
    "duration": None,
    "date": None,
}


def entity_to_health_record_dict(
    entity: ExtractedEntity,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None = None,
    document_date: datetime | None = None,
) -> dict | None:
    """Convert an extracted entity to a dict suitable for creating a HealthRecord.

    Returns None for entity types that should not be stored as individual records
    (e.g., dosage, route, frequency — these are attributes of medications).

    ``document_date`` is the encounter/visit date for the source document; it is
    used as a last-resort effective_date when the entity carries no date of its
    own (see :func:`_extract_effective_date`).
    """
    mapping = ENTITY_TO_RECORD_TYPE.get(entity.entity_class)
    if mapping is None:
        return None

    record_type, fhir_resource_type = mapping
    fhir_resource = _build_fhir_resource(entity, fhir_resource_type)
    display_text = _build_display_text(entity)

    effective_date = _extract_effective_date(entity, document_date)

    return {
        "id": uuid4(),
        "patient_id": patient_id,
        "user_id": user_id,
        "record_type": record_type,
        "fhir_resource_type": fhir_resource_type,
        "fhir_resource": fhir_resource,
        "content_hash": content_hash(fhir_resource),
        "source_format": "ai_extracted",
        "source_file_id": source_file_id,
        "effective_date": effective_date,
        "status": entity.attributes.get("status", "unknown"),
        "category": [record_type],
        "code_display": entity.text,
        "display_text": display_text,
        "is_duplicate": False,
        "confidence_score": entity.confidence,
        "ai_extracted": True,
    }


# Explicit date-bearing attribute keys, broadest-first. LangExtract is not
# consistent about which key it uses, so we accept the common variants.
_DATE_ATTRIBUTE_KEYS = (
    "date",
    "effective_date",
    "onset_date",
    "performed_date",
    "recorded_date",
    "result_date",
    "collection_date",
    "collected_date",
    "specimen_date",
    "report_date",
    "reported_date",
    "visit_date",
    "encounter_date",
    "service_date",
    "diagnosis_date",
    "diagnosed_date",
    "administered_date",
    "administration_date",
    "start_date",
    "observation_date",
    "observed_date",
    "measured_date",
    "noted_date",
    "datetime",
    "timestamp",
)

# Fixed default so generalized month/year values resolve their missing day to
# the 1st deterministically (instead of dateutil's implicit "today").
_DAY_DEFAULT = datetime(2000, 1, 1)

# Recognize dates embedded in free text. Each alternative REQUIRES a 4-digit
# year so numeric noise (e.g. a blood pressure "120/80", a ratio, a dose) is
# never misread as a date.
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_TEXT_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"  # ISO 2024-09-24
    r"|\d{4}-\d{1,2}"  # ISO year-month 2024-09
    r"|\d{1,2}/\d{1,2}/\d{4}"  # 09/24/2024
    r"|\d{1,2}/\d{4}"  # generalized 9/2024 (day dropped)
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}}"  # September 24, 2024
    rf"|(?:{_MONTHS})\.?\s+\d{{4}}"  # generalized September 2024
    r")\b",
    re.IGNORECASE,
)

# Entity classes whose effective date is sensibly the encounter/visit date when
# the entity has none of its own. Family history is excluded: it is relationship-
# based and inherently dateless, so stamping the visit date would misrepresent it.
_DOCUMENT_DATE_INELIGIBLE = frozenset({"family_history"})


def _find_date_in_text(text: str | None) -> datetime | None:
    """Return the first parseable date embedded in free text, or None.

    Only matches tokens carrying a 4-digit year, so numeric clinical values are
    not misread as dates.
    """
    if not text:
        return None
    match = _DATE_TEXT_PATTERN.search(str(text))
    if not match:
        return None
    return parse_datetime(match.group(0), default=_DAY_DEFAULT)


def _extract_effective_date(
    entity: ExtractedEntity, document_date: datetime | None = None
) -> datetime | None:
    """Extract a clinical date for an entity, never fabricating a wrong one.

    Resolution order (most reliable first):
      1. explicit date-bearing attribute keys (parsed directly, then scanned for
         an embedded date if the raw value is free text);
      2. a date embedded in any other attribute value;
      3. a date embedded in the entity text;
      4. the document/encounter ``document_date`` fallback (for eligible types).

    Returns None if no trustworthy date is available.
    """
    attrs = entity.attributes or {}

    # 1. Explicit date keys.
    for key in _DATE_ATTRIBUTE_KEYS:
        raw = attrs.get(key)
        if not raw:
            continue
        parsed = parse_datetime(str(raw), default=_DAY_DEFAULT)
        if parsed:
            return parsed
        embedded = _find_date_in_text(str(raw))
        if embedded:
            return embedded

    # 2. Dates embedded in other (non-date-key) attribute values.
    for key, value in attrs.items():
        if key in _DATE_ATTRIBUTE_KEYS or not isinstance(value, str):
            continue
        embedded = _find_date_in_text(value)
        if embedded:
            return embedded

    # 3. Date embedded in the entity text.
    embedded = _find_date_in_text(entity.text)
    if embedded:
        return embedded

    # 4. Document/encounter fallback for eligible entity types.
    if document_date is not None and entity.entity_class not in _DOCUMENT_DATE_INELIGIBLE:
        return document_date

    return None


def resolve_document_date(
    entities: list[ExtractedEntity], primary_visit_date: str | None = None
) -> datetime | None:
    """Derive a single document/encounter date to use as an entity fallback.

    Prefers an encounter entity's own date; otherwise the parsed document's
    primary visit date. Returns None when neither is available.
    """
    for entity in entities:
        if entity.entity_class == "encounter":
            enc_date = _extract_effective_date(entity)
            if enc_date:
                return enc_date
    if primary_visit_date:
        return parse_datetime(str(primary_visit_date), default=_DAY_DEFAULT)
    return None


def _build_fhir_resource(entity: ExtractedEntity, resource_type: str) -> dict:
    """Build a minimal FHIR resource JSON from an extracted entity."""
    resource: dict = {"resourceType": resource_type}
    attrs = entity.attributes

    if resource_type == "MedicationRequest":
        resource["status"] = "active"
        resource["intent"] = "order"
        resource["medicationCodeableConcept"] = {"text": entity.text}
        # Attach grouped dosage info if available
        dosage_parts = []
        if attrs.get("medication_group"):
            dose_text = entity.text
            if "value" in attrs and "unit" in attrs:
                dose_text = f"{entity.text} {attrs['value']}{attrs['unit']}"
            dosage_parts.append(dose_text)
        if dosage_parts:
            resource["dosageInstruction"] = [{"text": " ".join(dosage_parts)}]

    elif resource_type == "Condition":
        status = attrs.get("status", "active")
        if status in ("negated", "ruled_out", "absent"):
            status = "inactive"  # FHIR-valid status for negated conditions
        resource["clinicalStatus"] = {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": status}]
        }
        resource["code"] = {"text": entity.text}

    elif resource_type == "Observation":
        resource["status"] = "final"
        if entity.entity_class == "lab_result":
            resource["category"] = [{"coding": [{"code": "laboratory"}]}]
            resource["code"] = {"text": attrs.get("test", entity.text)}
            if "value" in attrs:
                try:
                    resource["valueQuantity"] = {
                        "value": float(attrs["value"]),
                        "unit": attrs.get("unit", ""),
                    }
                except (ValueError, TypeError):
                    resource["valueString"] = attrs.get("value", entity.text)
            if "ref_low" in attrs or "ref_high" in attrs:
                ref_range: dict = {}
                if "ref_low" in attrs:
                    try:
                        ref_range["low"] = {"value": float(attrs["ref_low"])}
                    except (ValueError, TypeError):
                        pass
                if "ref_high" in attrs:
                    try:
                        ref_range["high"] = {"value": float(attrs["ref_high"])}
                    except (ValueError, TypeError):
                        pass
                if ref_range:
                    resource["referenceRange"] = [ref_range]
        elif entity.entity_class == "social_history":
            category_label = attrs.get("category", "social-history")
            resource["category"] = [{"coding": [{"code": "social-history"}]}]
            resource["code"] = {"text": category_label.replace("_", " ").title()}
            resource["valueString"] = attrs.get("value", entity.text)
        else:
            # vital
            resource["category"] = [{"coding": [{"code": "vital-signs"}]}]
            resource["code"] = {"text": attrs.get("type", entity.text)}
            resource["valueString"] = entity.text

    elif resource_type == "Procedure":
        resource["status"] = "completed"
        resource["code"] = {"text": entity.text}

    elif resource_type == "AllergyIntolerance":
        resource["clinicalStatus"] = {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]
        }
        resource["code"] = {"text": entity.text}
        if "reaction" in attrs:
            resource["reaction"] = [{"manifestation": [{"text": attrs["reaction"]}]}]

    elif resource_type == "Encounter":
        visit_type = attrs.get("visit_type", "office")
        class_map = {
            "office": ("AMB", "ambulatory"),
            "telehealth": ("VR", "virtual"),
            "emergency": ("EMER", "emergency"),
            "inpatient": ("IMP", "inpatient encounter"),
        }
        class_code, class_display = class_map.get(visit_type, ("AMB", "ambulatory"))
        resource["status"] = "finished"
        resource["class"] = {"code": class_code, "display": class_display}
        cpt_code = attrs.get("cpt_code")
        if cpt_code:
            resource["type"] = [{"coding": [{"system": "http://www.ama-assn.org/go/cpt", "code": cpt_code}]}]
        reason = attrs.get("reason")
        if reason:
            resource["reasonCode"] = [{"text": reason}]
        date_val = attrs.get("date")
        if date_val:
            resource["period"] = {"start": date_val}

    elif resource_type == "DiagnosticReport":
        category = attrs.get("category", "imaging")
        resource["status"] = "final"
        resource["category"] = [{"coding": [{"code": category, "display": category.replace("_", " ").title()}]}]
        resource["code"] = {"text": attrs.get("procedure_name", entity.text)}
        findings = attrs.get("findings")
        if findings:
            resource["conclusion"] = findings
        interpretation = attrs.get("interpretation")
        if interpretation:
            resource["conclusionCode"] = [{"text": interpretation}]

    elif resource_type == "FamilyMemberHistory":
        relationship = attrs.get("relationship", "unknown")
        rel_map = {
            "mother": ("MTH", "Mother"),
            "father": ("FTH", "Father"),
            "sibling": ("SIB", "Sibling"),
            "sister": ("SIS", "Sister"),
            "brother": ("BRO", "Brother"),
            "grandmother": ("GRMTH", "Grandmother"),
            "grandfather": ("GRFTH", "Grandfather"),
            "grandparent": ("GRPRN", "Grandparent"),
            "aunt": ("AUNT", "Aunt"),
            "uncle": ("UNCLE", "Uncle"),
            "child": ("CHILD", "Child"),
        }
        rel_code, rel_display = rel_map.get(relationship.lower(), ("FAMMEMB", relationship.title()))
        resource["status"] = "completed"
        resource["relationship"] = {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode", "code": rel_code, "display": rel_display}],
        }
        condition_text = attrs.get("condition", entity.text)
        condition_entry: dict = {"code": {"text": condition_text}}
        notes = attrs.get("notes")
        if notes:
            condition_entry["note"] = [{"text": notes}]
        resource["condition"] = [condition_entry]

    elif resource_type == "DocumentReference":
        resource["status"] = "current"
        resource["type"] = {"coding": [{"system": "http://loinc.org", "code": "51847-2", "display": "Assessment and Plan"}]}
        resource["content"] = [{
            "attachment": {
                "contentType": "text/plain",
                "data": base64.b64encode(entity.text.encode()).decode(),
            },
        }]
        plan_items = attrs.get("plan_items")
        if plan_items and isinstance(plan_items, list):
            resource["description"] = "; ".join(plan_items)

    # Store extraction metadata
    resource["_extraction_metadata"] = {
        "entity_class": entity.entity_class,
        "original_text": entity.text,
        "attributes": attrs,
        "start_pos": entity.start_pos,
        "end_pos": entity.end_pos,
        "confidence": entity.confidence,
    }

    return resource


def _build_display_text(entity: ExtractedEntity) -> str:
    """Build a human-readable display text for a given entity."""
    attrs = entity.attributes
    cls = entity.entity_class

    if cls == "medication":
        parts = [entity.text]
        if "value" in attrs and "unit" in attrs:
            parts.append(f"{attrs['value']}{attrs['unit']}")
        return " ".join(parts)

    if cls == "condition":
        status = attrs.get("status", "")
        if status:
            return f"{entity.text} ({status})"
        return entity.text

    if cls == "lab_result":
        parts = [attrs.get("test", entity.text)]
        if "value" in attrs:
            parts.append(f": {attrs['value']}{attrs.get('unit', '')}")
        return "".join(parts)

    if cls == "vital":
        return entity.text

    if cls == "procedure":
        date = attrs.get("date", "")
        if date:
            return f"{entity.text} ({date})"
        return entity.text

    if cls == "allergy":
        reaction = attrs.get("reaction", "")
        if reaction:
            return f"{entity.text} — {reaction}"
        return entity.text

    if cls == "encounter":
        visit_type = attrs.get("visit_type", "visit")
        date = attrs.get("date", "")
        return f"{visit_type.title()} encounter{' — ' + date if date else ''}"

    if cls == "imaging_result":
        name = attrs.get("procedure_name", entity.text)
        findings = attrs.get("findings", "")
        return f"{name}: {findings}" if findings else name

    if cls == "family_history":
        rel = attrs.get("relationship", "Family member")
        condition = attrs.get("condition", entity.text)
        return f"{rel.title()}: {condition}"

    if cls == "assessment_plan":
        plan_items = attrs.get("plan_items", [])
        count = len(plan_items) if isinstance(plan_items, list) else 0
        return f"Assessment & Plan ({count} items)" if count else "Assessment & Plan"

    if cls == "social_history":
        category = attrs.get("category", "Social")
        value = attrs.get("value", entity.text)
        return f"{category.replace('_', ' ').title()}: {value}"

    return entity.text
