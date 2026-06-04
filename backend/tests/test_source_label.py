"""Tests for the human-readable source-label helper.

Real CDA/FHIR records carry a ``source_system`` that is often a machine
identifier (``urn:oid:…``, bare OID) rather than an institution name. The label
helper must ignore those and fall back to the clean ``source_format`` token.
"""

from __future__ import annotations

import pytest

from app.services.utils.source_label import source_label


@pytest.mark.parametrize(
    "source_format,source_system,expected",
    [
        # format-token mapping
        ("fhir_r4", None, "FHIR R4"),
        ("fhir", None, "FHIR R4"),
        ("cda_r2", None, "CDA"),
        ("cda", None, "CDA"),
        ("epic_ehi", None, "Epic EHI"),
        ("epic_ehi_single", None, "Epic EHI"),
        ("ai_extracted", None, "AI extraction"),
        # unknown token → title-cased fallback
        ("some_new_source", None, "Some New Source"),
        # human-readable source_system wins
        ("fhir_r4", "Quest Diagnostics", "Quest Diagnostics"),
        ("cda_r2", "Memorial Hospital", "Memorial Hospital"),
        # a format token that leaked into source_system still maps to a label
        ("cda_r2", "fhir", "FHIR R4"),
        # machine-identifier source_system is ignored → fall back to format
        ("cda_r2", "urn:oid:2.16.840.1.113883.19", "CDA"),
        ("fhir", "urn:ietf:rfc:3986", "FHIR R4"),
        ("cda_r2", "2.16.840.1.113883.3.6399.6.2.1", "CDA"),
        ("fhir_r4", "https://example.org/fhir", "FHIR R4"),
        # nothing usable
        (None, None, "Unknown"),
        (None, "urn:oid:1.2.3", "Unknown"),
    ],
)
def test_source_label(source_format, source_system, expected):
    assert source_label(source_format, source_system) == expected
