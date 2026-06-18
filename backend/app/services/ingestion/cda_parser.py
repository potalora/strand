"""CDA-to-FHIR parser.

Converts CDA R2 XML documents (C-CDA) to FHIR R4 resources using
python-fhir-converter, then maps them to health_records insert dicts
using the same extraction functions as fhir_parser.py.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from app.services.ingestion.fhir_parser import (
    SUPPORTED_RESOURCE_TYPES,
    build_display_text,
    build_reference_name_map,
    enrich_encounter_service_provider,
    extract_categories,
    extract_coding,
    extract_effective_date,
    extract_effective_date_end,
    extract_status,
    resolve_encounter_references,
)

logger = logging.getLogger(__name__)

# CDA template names to try, in order of likelihood.
CDA_TEMPLATES = [
    "CCD",
    "ConsultationNote",
    "DischargeSummary",
    "ProgressNote",
    "ReferralNote",
    "TransferSummary",
    "HistoryandPhysical",
    "OperativeNote",
    "ProcedureNote",
]

# Resource types produced by the converter that are not clinical data.
_SKIP_RESOURCE_TYPES = {"Patient", "Practitioner", "Organization", "Composition"}


def parse_cda_document(
    file_path: Path,
    manifest_doc: Any | None = None,
) -> list[dict]:
    """Parse a CDA XML document and return mapped FHIR record dicts.

    Args:
        file_path: Path to the CDA XML file.
        manifest_doc: Optional XDMDocument with hash for integrity check.

    Returns:
        List of record dicts suitable for bulk_insert_records, or [] on error.
    """
    if not file_path.exists():
        logger.warning("CDA file not found: %s", file_path)
        return []

    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        logger.error("Failed to read CDA file %s: %s", file_path, exc)
        return []

    # --- Hash validation ---
    if manifest_doc is not None:
        # IHE XDM standard mandates SHA-1 for document integrity hashes
        actual_hash = hashlib.sha1(raw_bytes).hexdigest()
        if actual_hash != manifest_doc.hash:
            logger.warning(
                "Hash mismatch for %s: expected %s, got %s",
                file_path.name,
                manifest_doc.hash,
                actual_hash,
            )
            return []

    xml_content = raw_bytes.decode("utf-8", errors="replace")

    # Derive metadata from manifest or defaults.
    doc_uri = manifest_doc.uri if manifest_doc else file_path.name
    institution = manifest_doc.author_institution if manifest_doc else ""

    # --- CDA -> FHIR conversion ---
    bundle = _render_cda_to_fhir(xml_content, filename=file_path.name)
    if bundle is None:
        return []

    # --- Map each entry ---
    records: list[dict] = []
    entries = bundle.get("entry", [])
    # Index Practitioner/Organization/Location names (built from ALL entries,
    # including the Practitioner/Organization the loop below skips) so encounters
    # can resolve reference-only providers/facilities to readable names.
    ref_map = build_reference_name_map(entries)
    for entry in entries:
        resource = entry.get("resource")
        if not resource:
            continue

        resource_type = resource.get("resourceType", "")

        # Skip non-clinical resources.
        if resource_type in _SKIP_RESOURCE_TYPES:
            continue

        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            logger.debug("Skipping unsupported resource type from CDA: %s", resource_type)
            continue

        if resource_type == "Encounter":
            resolve_encounter_references(resource, ref_map)
            enrich_encounter_service_provider(resource)

        record_type = SUPPORTED_RESOURCE_TYPES[resource_type]
        code_system, code_value, code_display = extract_coding(resource)
        categories = extract_categories(resource)
        status = extract_status(resource)
        effective_date = extract_effective_date(resource)
        effective_date_end = extract_effective_date_end(resource)
        display_text = build_display_text(resource, resource_type)

        # Attach extraction metadata to the FHIR resource.
        resource["_extraction_metadata"] = {
            "source_format": "cda_r2",
            "source_document": doc_uri,
            "source_institution": institution,
        }

        records.append(
            {
                "record_type": record_type,
                "fhir_resource_type": resource_type,
                "fhir_resource": resource,
                "source_format": "cda_r2",
                "effective_date": effective_date,
                "effective_date_end": effective_date_end,
                "status": status,
                "category": categories,
                "code_system": code_system,
                "code_value": code_value,
                "code_display": code_display,
                "display_text": display_text,
            }
        )

    logger.info(
        "CDA parsing of %s produced %d records (from %d entries)",
        file_path.name,
        len(records),
        len(entries),
    )
    return records


def _render_cda_to_fhir(xml_content: str, filename: str = "<unknown>") -> dict | None:
    """Try CDA templates in order and return the first successful FHIR Bundle."""
    from fhir_converter.renderers import CcdaRenderer

    renderer = CcdaRenderer()

    for template in CDA_TEMPLATES:
        try:
            result = renderer.render_to_fhir(template, xml_content)
            if result and result.get("entry"):
                return result
        except Exception as exc:
            logger.debug("Template %s did not match %s: %s", template, filename, exc)
            continue

    logger.warning("No CDA template produced a valid FHIR Bundle")
    return None
