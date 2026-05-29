from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

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


def map_fhir_resource(resource: dict) -> dict | None:
    """Map a single FHIR resource to a health_records insert dict."""
    resource_type = resource.get("resourceType")
    if not resource_type or resource_type not in SUPPORTED_RESOURCE_TYPES:
        return None

    record_type = SUPPORTED_RESOURCE_TYPES[resource_type]
    code_system, code_value, code_display = extract_coding(resource)
    categories = extract_categories(resource)
    status = extract_status(resource)
    effective_date = extract_effective_date(resource)
    effective_date_end = extract_effective_date_end(resource)
    display_text = build_display_text(resource, resource_type)

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
            mapped = map_fhir_resource(resource)
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
                mapped = map_fhir_resource(resource)
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
