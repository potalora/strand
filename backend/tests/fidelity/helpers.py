"""Utility functions for fidelity tests.

Provides FHIR path resolution and comparison helpers that work
at the mapper level (no DB required).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def resolve_fhir_path(resource: dict, path: str) -> Any:
    """Resolve a dot-separated path within a FHIR resource dict.

    Supports numeric indexes for arrays, e.g.:
        "coding.0.code" → resource["coding"][0]["code"]
        "category.1.text" → resource["category"][1]["text"]

    Returns None if the path cannot be resolved.
    """
    current: Any = resource
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx] if idx < len(current) else None
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            try:
                idx = int(part)
                # numeric key in dict — try as list index if value is a list
                current = None
            except ValueError:
                current = current.get(part)
        else:
            return None
    return current


def load_tsv_rows(tsv_path: Path) -> list[dict[str, str]]:
    """Load all data rows from a TSV file as list of dicts."""
    if not tsv_path.exists():
        return []
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def enrich_companion_rows(
    export_dir: Path,
    table_name: str,
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Apply the production companion-table enrichment used by Epic parsing."""

    from app.services.ingestion.epic_parser import (
        _VitalsValueIndex,
        _enrich_epic_row,
    )

    with _VitalsValueIndex(export_dir) as vitals_values:
        return [_enrich_epic_row(table_name, row, vitals_values) for row in rows]


def get_mapper_for_table(table_name: str):
    """Get the mapper instance for a given Epic table name."""
    from app.services.ingestion.epic_parser import EPIC_TABLE_MAPPERS

    return EPIC_TABLE_MAPPERS.get(table_name)


def map_all_rows(table_name: str, rows: list[dict[str, str]]) -> list[dict | None]:
    """Map all TSV rows through the appropriate mapper, returning results."""
    mapper = get_mapper_for_table(table_name)
    if not mapper:
        return []
    return [mapper.to_fhir(row) for row in rows]


def map_with_metadata(table_name: str, rows: list[dict[str, str]]) -> list[dict]:
    """Map rows and extract metadata using fhir_parser functions.

    Returns list of dicts with both fhir_resource and extracted metadata
    (effective_date, status, code_system, code_value, display_text, etc.)
    """
    from app.services.ingestion.fhir_parser import (
        build_display_text,
        extract_categories,
        extract_coding,
        extract_effective_date,
        extract_effective_date_end,
        extract_status,
    )
    from app.services.ingestion.epic_parser import RECORD_TYPE_MAP

    mapper = get_mapper_for_table(table_name)
    if not mapper:
        return []

    results = []
    for row in rows:
        fhir_resource = mapper.to_fhir(row)
        if fhir_resource is None:
            results.append({"fhir_resource": None, "skipped": True, "row": row})
            continue

        resource_type = fhir_resource.get("resourceType", "Unknown")
        record_type = RECORD_TYPE_MAP.get(resource_type, resource_type.lower())
        code_system, code_value, code_display = extract_coding(fhir_resource)

        results.append(
            {
                "fhir_resource": fhir_resource,
                "skipped": False,
                "row": row,
                "resource_type": resource_type,
                "record_type": record_type,
                "effective_date": extract_effective_date(fhir_resource),
                "effective_date_end": extract_effective_date_end(fhir_resource),
                "status": extract_status(fhir_resource),
                "category": extract_categories(fhir_resource),
                "code_system": code_system,
                "code_value": code_value,
                "code_display": code_display,
                "display_text": build_display_text(fhir_resource, resource_type),
            }
        )

    return results
