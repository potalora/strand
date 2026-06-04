"""Human-readable labels for a record's provenance.

``source_format`` is stored as a machine token (``fhir_r4``, ``epic_ehi``,
``cda_r2``, ``ai_extracted``, …). The Overview/Recent UI shows a friendly label
("Per {source}"). This keeps that mapping in one place so every endpoint that
exposes a source string is consistent.
"""

from __future__ import annotations

_SOURCE_LABELS: dict[str, str] = {
    "fhir_r4": "FHIR R4",
    "fhir": "FHIR R4",
    "epic_ehi": "Epic EHI",
    "epic_ehi_single": "Epic EHI",
    "cda_r2": "CDA",
    "cda": "CDA",
    "ai_extracted": "AI extraction",
}


def _is_machine_identifier(value: str) -> bool:
    """True if ``value`` is a system identifier, not a human-readable name.

    CDA/FHIR records often carry a ``source_system`` like ``urn:oid:2.16.840…``
    or a bare OID — useless as a UI label. We detect those so we can fall back to
    the clean ``source_format`` token instead.
    """
    s = value.strip().lower()
    if not s:
        return True
    if s.startswith(("urn:", "http://", "https://", "oid:")):
        return True
    # bare OID, e.g. "2.16.840.1.113883.19"
    if "." in s and all(c.isdigit() or c == "." for c in s):
        return True
    return False


def source_label(source_format: str | None, source_system: str | None = None) -> str:
    """Return a human-readable source label for a record.

    Prefers an explicit ``source_system`` when it's a human-readable name (e.g. an
    institution), but ignores machine identifiers (``urn:oid:…``, bare OIDs). Falls
    back to mapping the machine ``source_format`` token to a friendly label, then to
    a title-cased version of the raw token.
    """
    if source_system and not _is_machine_identifier(source_system):
        # A real institution name wins; but a bare format token that leaked into
        # source_system (e.g. "fhir") should still map to its friendly label.
        return _SOURCE_LABELS.get(source_system.strip().lower(), source_system.strip())
    if not source_format:
        return "Unknown"
    fmt = source_format.strip().lower()
    if fmt in _SOURCE_LABELS:
        return _SOURCE_LABELS[fmt]
    return source_format.replace("_", " ").title()
