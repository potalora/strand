from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.fhir_validation import validate_and_log_fhir
from app.services.ingestion.idempotent_inserter import idempotent_insert_records

logger = logging.getLogger(__name__)

SUPPORTED_RESOURCE_TYPES = {
    "Condition": "condition",
    "Observation": "observation",
    "MedicationRequest": "medication",
    "MedicationStatement": "medication",
    "AllergyIntolerance": "allergy",
    "Procedure": "procedure",
    "Encounter": "encounter",
    "Immunization": "immunization",
    "DiagnosticReport": "diagnostic_report",
    "DocumentReference": "document",
    "ImagingStudy": "imaging",
    "ServiceRequest": "service_request",
    "CarePlan": "care_plan",
    "Communication": "communication",
    "Appointment": "appointment",
    "CareTeam": "care_team",
    "ImmunizationRecommendation": "immunization",
    "QuestionnaireResponse": "questionnaire_response",
}


def extract_effective_date(resource: dict) -> datetime | None:
    """Extract the primary date from a FHIR resource for timeline ordering."""
    date_fields = [
        "effectiveDateTime",
        "issued",
        "date",
        "authoredOn",
        "occurrenceDateTime",
        "recordedDate",
        "onsetDateTime",
        "created",
        "sent",
        "start",
    ]
    for field in date_fields:
        val = resource.get(field)
        if val:
            return _parse_fhir_date(val)

    period = resource.get("effectivePeriod") or resource.get("period")
    if period and period.get("start"):
        return _parse_fhir_date(period["start"])

    if resource.get("meta", {}).get("lastUpdated"):
        return _parse_fhir_date(resource["meta"]["lastUpdated"])

    return None


def extract_effective_date_end(resource: dict) -> datetime | None:
    """Extract end date from a FHIR resource period."""
    period = resource.get("effectivePeriod") or resource.get("period")
    if period and period.get("end"):
        return _parse_fhir_date(period["end"])
    return None


def _parse_fhir_date(value: str) -> datetime | None:
    """Parse various FHIR date/datetime formats."""
    if not value:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # Handle timezone offsets like -07:00
    if "T" in value:
        try:
            clean = value
            if clean.endswith("Z"):
                clean = clean[:-1] + "+00:00"
            return datetime.fromisoformat(clean)
        except ValueError:
            pass
    logger.debug("Could not parse FHIR date: %s", value)
    return None


def extract_coding(resource: dict) -> tuple[str | None, str | None, str | None]:
    """Extract primary coding (system, code, display) from a FHIR resource."""
    code_obj = resource.get("code")
    if not code_obj:
        # Try type for DocumentReference (but type can be a list for Encounter)
        type_val = resource.get("type")
        if isinstance(type_val, dict):
            code_obj = type_val
        elif isinstance(type_val, list) and type_val and isinstance(type_val[0], dict):
            code_obj = type_val[0]
    if not code_obj or not isinstance(code_obj, dict):
        return None, None, None

    codings = code_obj.get("coding", [])
    if codings:
        c = codings[0]
        return c.get("system"), c.get("code"), c.get("display")

    text = code_obj.get("text")
    if text:
        return None, None, text
    return None, None, None


def extract_categories(resource: dict) -> list[str]:
    """Extract category codes from a FHIR resource."""
    categories = resource.get("category", [])
    result = []
    for cat in categories:
        if isinstance(cat, dict):
            codings = cat.get("coding", [])
            for c in codings:
                if c.get("code"):
                    result.append(c["code"])
            if not codings and cat.get("text"):
                result.append(cat["text"])
    return result or None


def extract_status(resource: dict) -> str | None:
    """Extract status from a FHIR resource."""
    status = resource.get("status")
    if status:
        return status
    clinical_status = resource.get("clinicalStatus")
    if clinical_status and isinstance(clinical_status, dict):
        codings = clinical_status.get("coding", [])
        if codings:
            return codings[0].get("code")
    return None


def build_display_text(resource: dict, resource_type: str) -> str:
    """Build a human-readable display text from a FHIR resource."""
    code_obj = resource.get("code")
    if not code_obj:
        type_val = resource.get("type")
        if isinstance(type_val, dict):
            code_obj = type_val
        elif isinstance(type_val, list) and type_val and isinstance(type_val[0], dict):
            code_obj = type_val[0]
    if code_obj and isinstance(code_obj, dict):
        text = code_obj.get("text")
        if text:
            return text
        codings = code_obj.get("coding", [])
        for c in codings:
            if c.get("display"):
                return c["display"]

    if resource_type == "Encounter":
        enc_type = resource.get("type", [])
        if enc_type:
            text = enc_type[0].get("text") if isinstance(enc_type[0], dict) else None
            if text:
                return text
        cls = resource.get("class", {})
        code = cls.get("code", "") if isinstance(cls, dict) else ""
        return f"Encounter ({code})" if code else "Encounter"

    if resource_type == "Immunization":
        vaccine = resource.get("vaccineCode", {})
        text = vaccine.get("text")
        if text:
            return text
        codings = vaccine.get("coding", [])
        for c in codings:
            if c.get("display"):
                return c["display"]
        return "Immunization"

    if resource_type == "MedicationRequest":
        med_ref = resource.get("medicationReference", {})
        display = med_ref.get("display")
        if display:
            return display
        med_cc = resource.get("medicationCodeableConcept", {})
        text = med_cc.get("text")
        if text:
            return text
        dosage = resource.get("dosageInstruction", [])
        if dosage:
            return dosage[0].get("text", "Medication")
        return "Medication Request"

    if resource_type == "DiagnosticReport":
        conclusion = resource.get("conclusion")
        if conclusion:
            code_text = code_obj.get("text", "Diagnostic Report") if code_obj else "Diagnostic Report"
            return f"{code_text}: {conclusion[:100]}"
        return "Diagnostic Report"

    if resource_type == "DocumentReference":
        desc = resource.get("description")
        if desc:
            return desc
        return "Document"

    if resource_type == "Communication":
        payload = resource.get("payload", [])
        if payload:
            content = payload[0].get("contentString", "")
            if content:
                return content[:100]
        return "Communication"

    if resource_type == "Appointment":
        desc = resource.get("description")
        if desc:
            return desc
        return "Appointment"

    if resource_type == "ServiceRequest":
        code_obj = resource.get("code", {})
        text = code_obj.get("text")
        if text:
            return text
        return "Service Request"

    if resource_type == "CarePlan":
        title = resource.get("title")
        if title:
            return title
        return "Care Plan"

    if resource_type == "FamilyMemberHistory":
        conditions = resource.get("condition", [])
        if conditions and isinstance(conditions[0], dict):
            code = conditions[0].get("code", {})
            text = code.get("text") if isinstance(code, dict) else None
            if text:
                rel = resource.get("relationship", {})
                rel_text = rel.get("text") if isinstance(rel, dict) else None
                return f"{text} ({rel_text})" if rel_text else text
        return "Family History"

    if resource_type == "CareTeam":
        name = resource.get("name")
        if name:
            return name
        return "Care Team"

    if resource_type == "ImmunizationRecommendation":
        recs = resource.get("recommendation", [])
        if recs:
            vaccine = recs[0].get("vaccineCode", [])
            if vaccine and isinstance(vaccine, list) and vaccine[0].get("text"):
                return vaccine[0]["text"]
            if vaccine and isinstance(vaccine, list):
                codings = vaccine[0].get("coding", [])
                for c in codings:
                    if c.get("display"):
                        return c["display"]
        return "Immunization Recommendation"

    if resource_type == "QuestionnaireResponse":
        questionnaire = resource.get("questionnaire")
        if questionnaire:
            return f"Questionnaire: {questionnaire}"
        return "Questionnaire Response"

    return resource_type


# Resource types that NAME the references an encounter points at (provider,
# medical center, location). The biggest structured-encounter gap was
# reference-only participants/serviceProviders — ~48% carried
# ``individual.reference = "Practitioner/<id>"`` with no ``display``, so the UI
# dropped them. FHIR bundles (and CDA→FHIR output) usually ship these resources
# alongside the encounter; we index them and backfill the missing ``display``.
_REFERENCE_RESOURCE_TYPES = frozenset({"Practitioner", "Organization", "Location"})


def _human_name_to_display(name: Any) -> str | None:
    """Render a FHIR ``HumanName`` (or a list of them) as a single display string.

    Prefers an explicit ``text``; otherwise joins prefix + given + family +
    suffix. Accepts a bare string for resilience. Returns ``None`` when nothing
    usable is present.
    """
    if isinstance(name, list):
        name = next((n for n in name if isinstance(n, dict) and n), None)
    if isinstance(name, str):
        stripped = name.strip()
        return stripped or None
    if not isinstance(name, dict):
        return None

    text = name.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Ordered name parts: prefix → given → family → suffix.
    parts: list[str] = []
    for key in ("prefix", "given", "family", "suffix"):
        val = name.get(key)
        if isinstance(val, list):
            parts.extend(str(p).strip() for p in val if p and str(p).strip())
        elif isinstance(val, str) and val.strip():
            parts.append(val.strip())

    full = " ".join(parts).strip()
    return full or None


def _resource_reference_name(resource: dict) -> str | None:
    """Display name for a referenceable resource, or ``None`` if it has none."""
    rt = resource.get("resourceType")
    if rt == "Practitioner":
        return _human_name_to_display(resource.get("name"))
    if rt in ("Organization", "Location"):
        name = resource.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def build_reference_name_map(entries: Iterable[dict]) -> dict[str, str]:
    """Map ``"<ResourceType>/<id>"`` and ``fullUrl`` → display name.

    Indexes every Practitioner/Organization/Location in the bundle so an
    encounter's ``reference`` strings can be resolved to readable names. Keyed by
    both the relative reference (``Practitioner/p1``) and the entry ``fullUrl``
    (``urn:uuid:...``) to cover both reference styles. First name wins.
    """
    ref_map: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resource = entry.get("resource")
        if not isinstance(resource, dict):
            continue
        if resource.get("resourceType") not in _REFERENCE_RESOURCE_TYPES:
            continue
        name = _resource_reference_name(resource)
        if not name:
            continue
        rt = resource.get("resourceType")
        rid = resource.get("id")
        if rt and rid:
            ref_map.setdefault(f"{rt}/{rid}", name)
        full_url = entry.get("fullUrl")
        if isinstance(full_url, str) and full_url:
            ref_map.setdefault(full_url, name)
    return ref_map


def _fill_reference_display(ref_obj: Any, ref_map: dict[str, str]) -> None:
    """Populate ``display`` on a FHIR Reference from ``ref_map`` (in place).

    Only fills when the reference is missing a display; never overwrites a name
    the source already provided.
    """
    if not isinstance(ref_obj, dict) or ref_obj.get("display"):
        return
    reference = ref_obj.get("reference")
    if not isinstance(reference, str):
        return
    name = ref_map.get(reference)
    if name:
        ref_obj["display"] = name


def resolve_encounter_references(resource: dict, ref_map: dict[str, str] | None) -> dict:
    """Backfill provider/facility/location ``display`` on a FHIR Encounter.

    Resolves ``participant[].individual`` (provider), ``serviceProvider`` (the
    medical center) and ``location[].location`` from the bundle's contained
    Practitioner/Organization/Location resources. Assumes those referenced
    resources travel in the same bundle (the common case for FHIR exports and
    CDA→FHIR output); references to resources not present are left untouched and
    the UI drops opaque refs. Mutates and returns ``resource``.
    """
    if not ref_map or resource.get("resourceType") != "Encounter":
        return resource
    for participant in resource.get("participant", []) or []:
        if isinstance(participant, dict):
            _fill_reference_display(participant.get("individual"), ref_map)
    _fill_reference_display(resource.get("serviceProvider"), ref_map)
    for loc in resource.get("location", []) or []:
        if isinstance(loc, dict):
            _fill_reference_display(loc.get("location"), ref_map)
    return resource


# Tokens that mark a location display as a *facility / organization* (B6).
_FACILITY_KEYWORDS = (
    "hospital", "clinic", "medical", "health", "center", "centre", "associates",
    "group", "practice", "pharmacy", "laborator", "imaging", "radiology",
    "institute", "department", "dept", "medicine", "care", "physicians",
    "specialists", "oncology", "cardiology", "orthopedic", "surgery", "urgent",
)
# Tokens that mark a location display as a *room / bed* — never a serviceProvider.
_ROOM_KEYWORDS = (
    "room", "rm ", " rm", "bed", "exam", "suite", "floor", "wing", "bay",
    "chair", "cubicle", "pod ", "ward",
)


def _looks_like_facility(display: str) -> bool:
    """True when a location display clearly names a facility/org, not a room."""
    d = display.lower()
    if any(tok in d for tok in _ROOM_KEYWORDS):
        return False
    return any(tok in d for tok in _FACILITY_KEYWORDS)


def enrich_encounter_service_provider(resource: dict) -> dict:
    """Surface a facility as ``serviceProvider`` on a FHIR Encounter (B6).

    Many encounter exports carry the facility only under ``location`` and leave
    ``serviceProvider`` empty (0% filled in the audit). When the encounter has
    no serviceProvider but a location whose display clearly names a facility,
    promote that display to ``serviceProvider``. Conservative: room/bed-style
    location names are skipped so an exam room is never labelled the provider.
    Mutates and returns ``resource``.
    """
    if resource.get("resourceType") != "Encounter":
        return resource
    if resource.get("serviceProvider"):
        return resource
    for loc in resource.get("location", []) or []:
        if not isinstance(loc, dict):
            continue
        display = (loc.get("location") or {}).get("display")
        if display and _looks_like_facility(display):
            resource["serviceProvider"] = {"display": display}
            break
    return resource


def map_fhir_resource(resource: dict, ref_map: dict[str, str] | None = None) -> dict | None:
    """Map a single FHIR resource to a health_records insert dict.

    ``ref_map`` (from :func:`build_reference_name_map`) lets encounters resolve
    reference-only providers/facilities/locations to readable names. Optional so
    existing callers keep working; resolution is simply skipped when absent.
    """
    resource_type = resource.get("resourceType")
    if not resource_type or resource_type not in SUPPORTED_RESOURCE_TYPES:
        return None

    if resource_type == "Encounter":
        # Resolve reference-only names first, then fall back to promoting a
        # facility-like location to serviceProvider if still unset.
        resolve_encounter_references(resource, ref_map)
        enrich_encounter_service_provider(resource)

    record_type = SUPPORTED_RESOURCE_TYPES[resource_type]
    code_system, code_value, code_display = extract_coding(resource)
    categories = extract_categories(resource)
    status = extract_status(resource)
    effective_date = extract_effective_date(resource)
    effective_date_end = extract_effective_date_end(resource)
    display_text = build_display_text(resource, resource_type)

    # WS-D: log-only structural validation of the mapped bundle resource.
    # Fail-open/non-latching — never blocks ingestion (the resource is returned
    # regardless of any drift signal).
    validate_and_log_fhir(resource, record_type, ai_built=False)

    return {
        "record_type": record_type,
        "fhir_resource_type": resource_type,
        "fhir_resource": resource,
        "source_format": "fhir_r4",
        "effective_date": effective_date,
        "effective_date_end": effective_date_end,
        "status": status,
        "category": categories,
        "code_system": code_system,
        "code_value": code_value,
        "code_display": code_display,
        "display_text": display_text,
    }


async def parse_fhir_bundle(
    file_path: Path,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None,
    db: AsyncSession,
    batch_size: int = 100,
    progress_callback: Any = None,
) -> dict:
    """Parse a FHIR R4 JSON bundle and insert records into the database.

    Returns a summary dict with counts.
    """
    file_size = file_path.stat().st_size
    stats = {"total_entries": 0, "records_inserted": 0, "records_skipped": 0, "errors": []}

    if file_size > 10 * 1024 * 1024:
        # Use streaming parser for large files
        stats = await _parse_large_bundle(
            file_path, user_id, patient_id, source_file_id, db, batch_size, progress_callback
        )
    else:
        stats = await _parse_small_bundle(
            file_path, user_id, patient_id, source_file_id, db, batch_size, progress_callback
        )

    logger.info(
        "FHIR parsing complete: %d entries, %d inserted, %d skipped, %d errors",
        stats["total_entries"],
        stats["records_inserted"],
        stats["records_skipped"],
        len(stats["errors"]),
    )
    return stats


async def _parse_small_bundle(
    file_path: Path,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None,
    db: AsyncSession,
    batch_size: int,
    progress_callback: Any,
) -> dict:
    """Parse a FHIR bundle that fits in memory."""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    stats = {"total_entries": 0, "records_inserted": 0, "records_skipped": 0, "errors": []}

    resource_type = data.get("resourceType")
    if resource_type == "Bundle":
        entries = data.get("entry", [])
    else:
        entries = [{"resource": data}]

    stats["total_entries"] = len(entries)
    # Index Practitioner/Organization/Location names so encounters can resolve
    # reference-only providers/facilities to readable names.
    ref_map = build_reference_name_map(entries)
    batch = []

    for i, entry in enumerate(entries):
        resource = entry.get("resource")
        if not resource:
            stats["records_skipped"] += 1
            continue

        rt = resource.get("resourceType")
        if rt == "Patient":
            stats["records_skipped"] += 1
            continue

        try:
            mapped = map_fhir_resource(resource, ref_map)
            if mapped:
                mapped["user_id"] = user_id
                mapped["patient_id"] = patient_id
                mapped["source_file_id"] = source_file_id
                batch.append(mapped)
            else:
                stats["records_skipped"] += 1
        except Exception as e:
            stats["errors"].append({"entry_index": i, "error": str(e)})
            continue

        if len(batch) >= batch_size:
            result = await idempotent_insert_records(db, batch)
            stats["records_inserted"] += result["inserted"]
            stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
            stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
            batch.clear()
            await db.commit()
            if progress_callback:
                await progress_callback(i + 1, stats["total_entries"], stats["records_inserted"])

    if batch:
        result = await idempotent_insert_records(db, batch)
        stats["records_inserted"] += result["inserted"]
        stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
        stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
        batch.clear()
        await db.commit()

    return stats


async def _parse_large_bundle(
    file_path: Path,
    user_id: UUID,
    patient_id: UUID,
    source_file_id: UUID | None,
    db: AsyncSession,
    batch_size: int,
    progress_callback: Any,
) -> dict:
    """Parse a large FHIR bundle using streaming JSON parser."""
    import ijson

    stats = {"total_entries": 0, "records_inserted": 0, "records_skipped": 0, "errors": []}
    batch = []

    # First streaming pass: index Practitioner/Organization/Location names only
    # (small even in large bundles — just id→name strings) so the main pass can
    # resolve encounter reference-only providers/facilities/locations.
    with open(file_path, "rb") as f:
        ref_map = build_reference_name_map(ijson.items(f, "entry.item"))

    with open(file_path, "rb") as f:
        entries = ijson.items(f, "entry.item")
        for i, entry in enumerate(entries):
            stats["total_entries"] += 1
            resource = entry.get("resource")
            if not resource:
                stats["records_skipped"] += 1
                continue

            rt = resource.get("resourceType")
            if rt == "Patient":
                stats["records_skipped"] += 1
                continue

            try:
                mapped = map_fhir_resource(resource, ref_map)
                if mapped:
                    mapped["user_id"] = user_id
                    mapped["patient_id"] = patient_id
                    mapped["source_file_id"] = source_file_id
                    batch.append(mapped)
                else:
                    stats["records_skipped"] += 1
            except Exception as e:
                stats["errors"].append({"entry_index": i, "error": str(e)})
                continue

            if len(batch) >= batch_size:
                result = await idempotent_insert_records(db, batch)
                stats["records_inserted"] += result["inserted"]
                stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
                stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
                batch.clear()
                await db.commit()
                if progress_callback:
                    await progress_callback(
                        stats["total_entries"],
                        stats["total_entries"],
                        stats["records_inserted"],
                    )

    if batch:
        result = await idempotent_insert_records(db, batch)
        stats["records_inserted"] += result["inserted"]
        stats["records_updated"] = stats.get("records_updated", 0) + result["updated"]
        stats["records_unchanged"] = stats.get("records_unchanged", 0) + result["unchanged"]
        batch.clear()
        await db.commit()

    return stats
