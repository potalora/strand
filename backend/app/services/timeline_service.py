from __future__ import annotations

import re
from typing import Any

# 8-4-4-4-12 hex UUID. A reference whose id is a bare UUID (e.g.
# "Practitioner/3f2504e0-...") is opaque — not human-meaningful — and is
# dropped, matching the frontend `performerName` behaviour in
# components/retro/renderers/shared.tsx.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _reference_name(ref: Any) -> str | None:
    """Resolve a single FHIR Reference to a human-readable provider name.

    Prefers ``display``; falls back to a non-opaque ``reference`` string; drops
    references whose id is a bare UUID.
    """
    if not isinstance(ref, dict):
        return None
    display = ref.get("display")
    if isinstance(display, str) and display.strip():
        return display.strip()
    reference = ref.get("reference")
    if isinstance(reference, str) and reference.strip():
        tail = reference.split("/", 1)[-1]
        if _UUID_RE.match(tail):
            return None
        return reference.strip()
    return None


def extract_provider_display(fhir_resource: Any, record_type: str | None = None) -> str | None:
    """Best-effort flat provider/performer name for a compact timeline event.

    The Timeline carries a single human-readable provider string rather than the
    full participant/performer structure that RecordDetail renders. Reads the
    R4B paths the RecordDetail renderers use:

      - Encounter   → ``participant[].individual``
      - Observation → ``performer[]`` (flat ``Reference[]``)
      - Procedure   → ``performer[].actor``

    Returns the first human-readable name, or ``None`` when no non-opaque
    provider is present. ``record_type`` is accepted for caller symmetry but the
    extraction keys off the resource structure.
    """
    if not isinstance(fhir_resource, dict):
        return None

    candidates: list[Any] = []

    participants = fhir_resource.get("participant")
    if isinstance(participants, list):
        candidates.extend(
            p.get("individual") if isinstance(p, dict) else None for p in participants
        )

    performers = fhir_resource.get("performer")
    if isinstance(performers, list):
        for p in performers:
            if isinstance(p, dict) and isinstance(p.get("actor"), dict):
                candidates.append(p["actor"])
            else:
                candidates.append(p)

    for candidate in candidates:
        name = _reference_name(candidate)
        if name:
            return name
    return None
