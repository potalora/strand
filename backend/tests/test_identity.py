from __future__ import annotations

from pathlib import Path

import pytest

from app.services.ingestion.identity import Identity, extract_identity

# Real extract location (gitignored). Skip when absent.
_XDM_DOC = Path(__file__).resolve().parents[2] / (
    "HealthSummary_May_29_2026/IHE_XDM/Pedro1/DOC0001.XML"
)


@pytest.mark.fidelity
@pytest.mark.skipif(not _XDM_DOC.exists(), reason="real XDM extract not present")
def test_cda_renderer_preserves_source_id():
    """Probe: does CcdaRenderer carry the CDA <id> into resource.id/identifier?

    This is a discovery test. We assert that AT LEAST ONE clinical resource
    produced from the real CDA carries either a non-UUID `id` or a populated
    `identifier`. If this fails, identity.py CDA branch must parse <id> directly.
    """
    from fhir_converter.renderers import CcdaRenderer

    renderer = CcdaRenderer()
    bundle = renderer.render_to_fhir("CCD", _XDM_DOC.read_text(encoding="utf-8"))

    has_identifier = False
    has_meaningful_id = False
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") in {"Bundle", "Composition", "Patient"}:
            continue
        if res.get("identifier"):
            has_identifier = True
        rid = res.get("id", "")
        # A bare UUID id is renderer-generated, not source-stable.
        if rid and "-" not in rid:
            has_meaningful_id = True

    assert has_identifier or has_meaningful_id, (
        "CcdaRenderer dropped source <id>; identity.py CDA branch needs a "
        "direct-XML fallback (parse act <id root extension>)."
    )


def test_explicit_fields_take_precedence():
    rec = {
        "source_format": "epic_ehi",
        "external_id": "ORDER_MED_123",
        "source_system": "epic:ORDER_MED",
        "fhir_resource": {"resourceType": "MedicationRequest", "id": "ignored"},
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="epic:ORDER_MED", external_id="ORDER_MED_123")


def test_fhir_resource_id():
    rec = {
        "source_format": "fhir_r4",
        "fhir_resource": {"resourceType": "Condition", "id": "cond-1"},
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="fhir", external_id="Condition/cond-1")


def test_fhir_identifier_preferred_over_id():
    rec = {
        "source_format": "fhir_r4",
        "fhir_resource": {
            "resourceType": "Condition",
            "id": "gen-uuid",
            "identifier": [{"system": "urn:epic", "value": "PROB-9"}],
        },
    }
    ident = extract_identity(rec)
    assert ident == Identity(source_system="urn:epic", external_id="Condition/PROB-9")


def test_fhir_no_id_returns_none():
    rec = {"source_format": "fhir_r4", "fhir_resource": {"resourceType": "Condition"}}
    assert extract_identity(rec) is None


def test_unknown_format_returns_none():
    rec = {"source_format": "mystery", "fhir_resource": {"resourceType": "X", "id": "1"}}
    assert extract_identity(rec) is None


def test_extraction_never_raises_on_bad_input():
    assert extract_identity({"source_format": "fhir_r4"}) is None
    assert extract_identity({"source_format": "fhir_r4", "fhir_resource": None}) is None


def test_cda_identifier_with_urn_oid():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Condition",
            "id": "881e0b55-1111-2222-3333-444455556666",  # renderer UUID, ignored
            "identifier": [{"system": "urn:oid:1.2.840.114350.1.13.516.2.7.2.768076", "value": "26510156"}],
        },
    }
    ident = extract_identity(rec)
    assert ident == Identity(
        source_system="urn:oid:1.2.840.114350.1.13.516.2.7.2.768076",
        external_id="Condition/26510156",
    )


def test_cda_npi_root_is_not_used_as_identity():
    """Provider NPI (urn:oid:2.16.840.1.113883.4.6) must not become the record identity."""
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Practitioner",
            "identifier": [{"system": "urn:oid:2.16.840.1.113883.4.6", "value": "1234567890"}],
        },
    }
    assert extract_identity(rec) is None


def test_cda_person_root_is_not_used_as_identity():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {
            "resourceType": "Condition",
            "identifier": [{"system": "urn:oid:2.16.840.1.113883.4.2", "value": "999"}],
        },
    }
    # Only non-act identifier present, and id absent -> None
    assert extract_identity(rec) is None


def test_cda_nullflavor_id_does_not_crash():
    rec = {
        "source_format": "cda_r2",
        "fhir_resource": {"resourceType": "Practitioner", "id": {"nullFlavor": "UNK"}},
    }
    assert extract_identity(rec) is None


def test_epic_identity_from_row():
    from app.services.ingestion.identity import epic_identity

    row = {"ORDER_MED_ID": "555", "ORDER_ID": "9", "LINE": "2"}
    ident = epic_identity("ORDER_MED", ["ORDER_MED_ID", "LINE"], row)
    assert ident == Identity(source_system="epic:ORDER_MED", external_id="555|2")


def test_epic_identity_missing_pk_returns_none():
    from app.services.ingestion.identity import epic_identity

    assert epic_identity("ORDER_MED", ["ORDER_MED_ID"], {"OTHER": "x"}) is None
