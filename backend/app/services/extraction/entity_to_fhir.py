from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from uuid import UUID, uuid4
from xml.sax.saxutils import escape as _xml_escape

from app.services.extraction import terminology
from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.terminology import parse_dosage  # re-exported for callers/tests
from app.services.ingestion.content_hash import content_hash
from app.services.ingestion.fhir_validation import validate_and_log_fhir
from app.utils.date_utils import parse_datetime

__all__ = [
    "entity_to_health_record_dict",
    "resolve_document_date",
    "resolve_document_provider",
    "parse_lab_measurement",
    "parse_dosage",
]

logger = logging.getLogger(__name__)

# v3 ObservationInterpretation code system (H/L/HH/LL/A/N/POS/NEG).
_INTERP_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"

# Attribute keys that may carry the provider/performer of a record. Checked in
# order; the first non-empty value wins (B2).
_PROVIDER_ATTR_KEYS = (
    "provider",
    "performer",
    "performed_by",
    "ordering_provider",
    "ordered_by",
    "rendering_provider",
    "attending",
    "attending_provider",
    "physician",
    "doctor",
    "provider_name",
)

# Lab *panel/order* names that, when they appear with no measured value, denote
# an order rather than a result — emitted as a ServiceRequest, not an empty
# Observation (B8).
_LAB_PANEL_NAMES = frozenset(
    {
        "cbc",
        "cbc with differential",
        "complete blood count",
        "cmp",
        "comprehensive metabolic panel",
        "bmp",
        "basic metabolic panel",
        "metabolic panel",
        "lipid panel",
        "lipid profile",
        "hepatic panel",
        "liver panel",
        "liver function tests",
        "lft",
        "lfts",
        "thyroid panel",
        "urinalysis",
        "ua",
        "fit",
        "fecal immunochemical test",
        "a1c",
        "hemoglobin a1c",
    }
)

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
    document_provider: str | None = None,
) -> dict | None:
    """Convert an extracted entity to a dict suitable for creating a HealthRecord.

    Returns None for entity types that should not be stored as individual records
    (e.g., dosage, route, frequency — these are attributes of medications).

    ``document_date`` is the encounter/visit date for the source document; it is
    used as a last-resort effective_date when the entity carries no date of its
    own (see :func:`_extract_effective_date`).

    ``document_provider`` is the document-level attending/ordering provider; it
    is attached to the resource (participant/performer/requester) when the entity
    carries no provider of its own (B2). Both new params are optional with safe
    defaults so existing call sites keep working unchanged.
    """
    mapping = ENTITY_TO_RECORD_TYPE.get(entity.entity_class)
    if mapping is None:
        return None

    record_type, fhir_resource_type = mapping

    # B1 — standard terminology coding (None for unknown/uncodable terms).
    coding = _resolve_coding(entity)
    # B2 — provider/performer (entity-level first, document-level fallback).
    provider = _resolve_provider(entity, document_provider)

    # B8 — a named lab panel with no value is an order, not an empty result.
    if _is_panel_order(entity):
        record_type, fhir_resource_type = "service_request", "ServiceRequest"

    fhir_resource = _build_fhir_resource(
        entity, fhir_resource_type, coding=coding, provider=provider
    )
    # WS-D: log-only structural validation of the AI-built resource. ai_built=True
    # so a partial resource is never failed/strict-rejected — it always ingests;
    # fail-open/non-latching, never blocks.
    validate_and_log_fhir(fhir_resource, record_type, ai_built=True)
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
        "code_system": coding.system if coding else None,
        "code_value": coding.code if coding else None,
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


def resolve_document_provider(entities: list[ExtractedEntity]) -> str | None:
    """Derive a single document-level provider to attach to records (B2).

    Returns the first ``provider`` entity's name — preferring a ``name``/
    ``provider_name`` attribute, falling back to the entity text. Callers pass
    the result as ``document_provider`` to :func:`entity_to_health_record_dict`
    so encounters/observations/procedures without their own provider still get
    a participant/performer. Returns ``None`` when no provider was extracted.
    """
    for entity in entities:
        if entity.entity_class != "provider":
            continue
        attrs = entity.attributes or {}
        name = _first_attr(attrs, ("name", "provider_name", "full_name", "display"))
        if name:
            return name
        if entity.text and entity.text.strip():
            return entity.text.strip()
    return None


# --- Richness helpers (B1-B8) ----------------------------------------------


def _first_attr(attrs: dict, keys: tuple[str, ...]) -> str | None:
    """Return the first non-blank string value among ``keys`` in ``attrs``."""
    for key in keys:
        val = attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _narrative_div(text: str) -> str:
    """Wrap a plain-text visit summary in a FHIR Narrative XHTML ``div``.

    FHIR R4 ``Encounter`` has no dedicated note/summary field, so an AI-extracted
    visit summary is stored as the resource's human-readable ``text.div``
    (the canonical FHIR place for it). The text is XML-escaped to keep the div
    well-formed.
    """
    return f'<div xmlns="http://www.w3.org/1999/xhtml">{_xml_escape(text)}</div>'


# Attribute keys that may carry a visit summary / chief complaint / narrative
# on an AI-extracted encounter, checked in order.
_ENCOUNTER_SUMMARY_KEYS = (
    "summary",
    "visit_summary",
    "chief_complaint",
    "narrative",
    "hpi",
    "history_of_present_illness",
)


def _resolve_coding(entity: ExtractedEntity) -> terminology.Coding | None:
    """Map an extracted entity to a standard terminology coding (B1).

    Returns ``None`` for entity types without a curated map or unknown terms.
    """
    attrs = entity.attributes or {}
    cls = entity.entity_class
    if cls == "condition":
        return terminology.lookup_condition(entity.text)
    if cls == "medication":
        return terminology.lookup_medication(entity.text)
    if cls == "lab_result":
        return terminology.lookup_lab(attrs.get("test") or entity.text)
    if cls == "procedure":
        return terminology.lookup_procedure(attrs.get("procedure_name") or entity.text)
    return None


def _resolve_provider(
    entity: ExtractedEntity, document_provider: str | None = None
) -> str | None:
    """Resolve the provider/performer for an entity (B2).

    Prefers a provider named in the entity's own attributes; otherwise falls
    back to the document-level provider passed by the caller.
    """
    attrs = entity.attributes or {}
    own = _first_attr(attrs, _PROVIDER_ATTR_KEYS)
    if own:
        return own
    if document_provider and document_provider.strip():
        return document_provider.strip()
    return None


# Reference range like "(70-99)", "135–145", "3.5 to 5.0".
_LAB_RANGE_RE = re.compile(
    r"\(?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*\)?"
)
# A measured value plus optional unit. The value must not be glued to a letter
# (so "A1c"/"B12"/"T4" analyte digits are not mistaken for the value).
_LAB_VALUE_RE = re.compile(
    r"(?<![A-Za-z\d.])(\d+(?:\.\d+)?)\s*"
    r"(%|[A-Za-zµ][A-Za-z0-9µ%/.\-]*)?"
)
# Word interpretation flags (longest first).
_INTERP_WORDS = (
    ("critical high", "HH"),
    ("critical low", "LL"),
    ("abnormal", "A"),
    ("normal", "N"),
    ("positive", "POS"),
    ("negative", "NEG"),
    ("high", "H"),
    ("low", "L"),
    ("wnl", "N"),
)
_INTERP_WORD_RE = re.compile(
    r"(?:^|\s)(critical high|critical low|abnormal|normal|positive|negative|high|low|wnl)(?:\s|$)",
    re.IGNORECASE,
)
# Trailing single/double-letter flag, requiring whitespace before it so the
# trailing "L" of "mIU/L" is not read as a "low" flag.
_INTERP_LETTER_RE = re.compile(r"(?:^|\s)(HH|LL|H|L)\s*$", re.IGNORECASE)


def parse_lab_measurement(text: str | None) -> dict:
    """Parse a lab string into value / unit / reference range / interpretation.

    Handles strings like ``"Glucose 95 mg/dL (70-99) H"`` →
    ``{value: 95.0, unit: "mg/dL", ref_low: 70.0, ref_high: 99.0,
    interpretation: "H", analyte: "Glucose"}``. Missing pieces are ``None``.
    """
    result: dict = {
        "value": None,
        "unit": None,
        "ref_low": None,
        "ref_high": None,
        "interpretation": None,
        "analyte": None,
    }
    if not text:
        return result
    s = str(text)

    # 1. Reference range (removed before value parsing so its numbers don't leak).
    m = _LAB_RANGE_RE.search(s)
    if m:
        try:
            result["ref_low"] = float(m.group(1))
            result["ref_high"] = float(m.group(2))
        except (ValueError, TypeError):
            pass
        s = s[: m.start()] + " " + s[m.end():]

    # 2. Interpretation flag (word form anywhere, else a trailing letter flag).
    interp = None
    wm = _INTERP_WORD_RE.search(s)
    if wm:
        token = wm.group(1).lower()
        for word, code in _INTERP_WORDS:
            if token == word:
                interp = code
                break
        s = s[: wm.start()] + " " + s[wm.end():]
    if interp is None:
        lm = _INTERP_LETTER_RE.search(s)
        if lm:
            interp = lm.group(1).upper()
            s = s[: lm.start()]
    result["interpretation"] = interp

    # 3. Value + unit.
    vm = _LAB_VALUE_RE.search(s)
    if vm:
        try:
            result["value"] = float(vm.group(1))
        except (ValueError, TypeError):
            result["value"] = None
        unit = (vm.group(2) or "").strip().rstrip(".,;:")
        result["unit"] = unit or None
        analyte = str(text)[: _value_offset_in_original(str(text), vm.group(1))].strip()
        result["analyte"] = analyte.rstrip(",;:- ").strip() or None
    else:
        result["analyte"] = str(text).strip() or None

    return result


def _value_offset_in_original(original: str, value_str: str) -> int:
    """Offset of the measured value in the original text (for analyte slicing)."""
    idx = original.find(value_str)
    return idx if idx >= 0 else len(original)


def _lab_measurement(entity: ExtractedEntity) -> dict:
    """Merge explicit lab attributes with values parsed from the entity text.

    Explicit attributes win; text parsing fills the gaps.
    """
    attrs = entity.attributes or {}
    parsed = parse_lab_measurement(entity.text)

    def coalesce(attr_keys: tuple[str, ...], parsed_key: str):
        val = _first_attr(attrs, attr_keys)
        return val if val is not None else parsed[parsed_key]

    return {
        "value": coalesce(("value",), "value"),
        "unit": coalesce(("unit", "units"), "unit"),
        "ref_low": coalesce(("ref_low", "reference_low", "low"), "ref_low"),
        "ref_high": coalesce(("ref_high", "reference_high", "high"), "ref_high"),
        "interpretation": coalesce(("interpretation", "flag", "abnormal_flag"), "interpretation"),
        "analyte": attrs.get("test") or parsed["analyte"] or entity.text,
    }


def _is_panel_order(entity: ExtractedEntity) -> bool:
    """True when a lab entity is a known panel name carrying no measured value (B8)."""
    if entity.entity_class != "lab_result":
        return False
    if _lab_measurement(entity)["value"] is not None:
        return False
    attrs = entity.attributes or {}
    name = terminology.normalize_term(attrs.get("test") or entity.text)
    return name in _LAB_PANEL_NAMES


def _build_dosage_instruction(entity: ExtractedEntity) -> dict | None:
    """Build a FHIR ``dosageInstruction`` element from a medication entity (B4).

    Parses a sig string (``dosage``/``sig``/``dose`` attributes) into
    ``doseAndRate`` + ``route`` + ``timing``, falling back to explicit
    value/unit/frequency attributes. Returns ``None`` when nothing is known.
    """
    attrs = entity.attributes or {}
    sig = _first_attr(attrs, ("dosage", "sig", "dose", "instructions", "directions"))
    parsed = parse_dosage(sig) if sig else None
    di: dict = {}
    if sig:
        di["text"] = sig

    dose_value = dose_unit = None
    if parsed and parsed["dose_value"] is not None:
        dose_value, dose_unit = parsed["dose_value"], parsed["dose_unit"]
    elif attrs.get("value"):
        try:
            dose_value = float(attrs["value"])
            dose_unit = attrs.get("unit") or None
        except (ValueError, TypeError):
            pass
    if dose_value is not None:
        dq: dict = {"value": dose_value}
        if dose_unit:
            dq["unit"] = dose_unit
        di["doseAndRate"] = [{"doseQuantity": dq}]
        di.setdefault("text", f"{entity.text} {('%g' % dose_value)}{dose_unit or ''}".strip())

    route = attrs.get("route") or (parsed["route"] if parsed else None)
    if route:
        di["route"] = {"text": route}

    if parsed and parsed["frequency"]:
        di["timing"] = {
            "repeat": {
                "frequency": parsed["frequency"],
                "period": parsed["period"],
                "periodUnit": parsed["period_unit"],
            }
        }
    elif attrs.get("frequency"):
        pf = parse_dosage(str(attrs["frequency"]))
        if pf["frequency"]:
            di["timing"] = {
                "repeat": {
                    "frequency": pf["frequency"],
                    "period": pf["period"],
                    "periodUnit": pf["period_unit"],
                }
            }
        else:
            di["text"] = (di.get("text", "") + " " + str(attrs["frequency"])).strip()

    if parsed and parsed["as_needed"]:
        di["asNeeded"] = True

    return di or None


def _build_fhir_resource(
    entity: ExtractedEntity,
    resource_type: str,
    coding: terminology.Coding | None = None,
    provider: str | None = None,
) -> dict:
    """Build a minimal FHIR resource JSON from an extracted entity."""
    resource: dict = {"resourceType": resource_type}
    attrs = entity.attributes

    if resource_type == "MedicationRequest":
        resource["status"] = "active"
        resource["intent"] = "order"
        med_cc: dict = {"text": entity.text}
        if coding:
            med_cc["coding"] = [coding.as_coding()]
        resource["medicationCodeableConcept"] = med_cc
        dosage_instruction = _build_dosage_instruction(entity)
        if dosage_instruction:
            resource["dosageInstruction"] = [dosage_instruction]
        if provider:
            resource["requester"] = {"display": provider}

    elif resource_type == "Condition":
        status = attrs.get("status", "active")
        if status in ("negated", "ruled_out", "absent"):
            status = "inactive"  # FHIR-valid status for negated conditions
        resource["clinicalStatus"] = {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": status}]
        }
        code_obj: dict = {"text": entity.text}
        if coding:
            code_obj["coding"] = [coding.as_coding()]
        resource["code"] = code_obj
        onset = _first_attr(
            attrs,
            ("onset_date", "onset", "onset_datetime", "since", "since_date",
             "diagnosed", "diagnosis_date", "diagnosed_date"),
        )
        if onset:
            resource["onsetDateTime"] = onset

    elif resource_type == "Observation":
        resource["status"] = "final"
        if entity.entity_class == "lab_result":
            meas = _lab_measurement(entity)
            resource["category"] = [{"coding": [{"code": "laboratory"}]}]
            code_obj = {"text": meas["analyte"] or entity.text}
            if coding:
                code_obj["coding"] = [coding.as_coding()]
            resource["code"] = code_obj
            if meas["value"] is not None:
                try:
                    vq: dict = {"value": float(meas["value"])}
                    if meas["unit"]:
                        vq["unit"] = meas["unit"]
                    resource["valueQuantity"] = vq
                except (ValueError, TypeError):
                    resource["valueString"] = str(meas["value"])
            ref_range = {}
            if meas["ref_low"] is not None:
                try:
                    ref_range["low"] = {"value": float(meas["ref_low"])}
                except (ValueError, TypeError):
                    pass
            if meas["ref_high"] is not None:
                try:
                    ref_range["high"] = {"value": float(meas["ref_high"])}
                except (ValueError, TypeError):
                    pass
            if ref_range:
                resource["referenceRange"] = [ref_range]
            if meas["interpretation"]:
                resource["interpretation"] = [
                    {"coding": [{"system": _INTERP_SYSTEM, "code": str(meas["interpretation"])}]}
                ]
            if provider:
                resource["performer"] = [{"display": provider}]
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
        proc_code: dict = {"text": entity.text}
        if coding:
            proc_code["coding"] = [coding.as_coding()]
        resource["code"] = proc_code
        if provider:
            resource["performer"] = [{"actor": {"display": provider}}]

    elif resource_type == "ServiceRequest":
        # A lab panel/order named without a measured value (B8).
        resource["status"] = "active"
        resource["intent"] = "order"
        sr_code: dict = {"text": (attrs.get("test") or entity.text)}
        if coding:
            sr_code["coding"] = [coding.as_coding()]
        resource["code"] = sr_code
        if provider:
            resource["requester"] = {"display": provider}

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

        # Visit type / title — a readable label for the visit. Prefer an explicit
        # description, else the extracted visit phrase (entity.text). Merge a CPT
        # code into the same CodeableConcept so the title and code travel together.
        type_text = _first_attr(attrs, ("visit_description", "encounter_type", "visit_type_text"))
        if not type_text and entity.text and entity.text.strip():
            type_text = entity.text.strip()
        cpt_code = attrs.get("cpt_code")
        type_concept: dict = {}
        if type_text:
            type_concept["text"] = type_text
        if cpt_code:
            type_concept["coding"] = [
                {"system": "http://www.ama-assn.org/go/cpt", "code": cpt_code}
            ]
        if type_concept:
            resource["type"] = [type_concept]

        reason = attrs.get("reason")
        if reason:
            resource["reasonCode"] = [{"text": reason}]
        date_val = attrs.get("date")
        if date_val:
            resource["period"] = {"start": date_val}
        if provider:
            resource["participant"] = [{"individual": {"display": provider}}]
        facility = _first_attr(
            attrs, ("facility", "medical_center", "location", "service_provider", "organization")
        )
        if facility:
            resource["serviceProvider"] = {"display": facility}

        # Visit summary / chief complaint → FHIR human-readable narrative.
        summary = _first_attr(attrs, _ENCOUNTER_SUMMARY_KEYS)
        if summary:
            resource["text"] = {"status": "additional", "div": _narrative_div(summary)}

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
        # B7 — performing lab / radiologist.
        performer = provider or _first_attr(
            attrs, ("lab", "performing_lab", "radiologist", "facility")
        )
        if performer:
            resource["performer"] = [{"display": performer}]

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
