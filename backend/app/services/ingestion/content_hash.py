"""Canonical content hashing for health records.

Produces a stable sha256 over the clinically meaningful content of a FHIR
resource, ignoring volatile fields (server timestamps, ingestion metadata,
rendered narrative) so that re-ingesting the same record yields the same hash
while a genuine clinical change yields a different one.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Top-level keys removed entirely before hashing.
_NOISE_KEYS = frozenset({"_extraction_metadata", "text"})
# Keys removed from the `meta` object before hashing.
_META_NOISE_KEYS = frozenset({"lastUpdated", "versionId", "source"})


def canonicalize(resource: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `resource` with volatile/noise fields removed."""
    out: dict[str, Any] = {}
    for key, value in resource.items():
        if key in _NOISE_KEYS:
            continue
        if key == "meta" and isinstance(value, dict):
            cleaned = {k: v for k, v in value.items() if k not in _META_NOISE_KEYS}
            if cleaned:
                out[key] = cleaned
            continue
        out[key] = value
    return out


def content_hash(resource: dict[str, Any]) -> str:
    """Stable hex sha256 of a resource's canonical clinical content."""
    canonical = canonicalize(resource)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
