"""Encounter enrichment — surface visit/provider/facility/summary detail.

Covers the four legs of the encounter-detail work:

* FHIR reference resolution (``fhir_parser``): a bundle's contained
  Practitioner/Organization/Location resources are looked up so an encounter
  that carried only ``reference`` strings ends up with provider/facility/
  location ``display`` names — the #1 gap (≈48% of structured encounters were
  reference-only).
* CDA reference resolution (``cda_parser``): the same resolution applied to
  encounters produced by the CDA→FHIR converter.
* Epic encounters (``epic_mappers.encounters``): visit type + provider fallback
  from PAT_ENC columns.
* AI encounters (``entity_to_fhir``): facility/medical-center, a readable visit
  type/title, and a visit summary captured on AI-extracted encounters.

Pure-function tests unless noted; one DB-backed integration test exercises the
full ``parse_fhir_bundle`` wiring.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient
from app.models.record import HealthRecord
from app.models.user import User
from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.ingestion.epic_mappers.encounters import PatEncMapper
from app.services.ingestion.fhir_parser import (
    _human_name_to_display,
    build_reference_name_map,
    map_fhir_resource,
    parse_fhir_bundle,
    resolve_encounter_references,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encounter(
    *,
    participant_ref: str | None = None,
    participant_display: str | None = None,
    service_provider_ref: str | None = None,
    location_ref: str | None = None,
) -> dict:
    res: dict = {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"code": "AMB"},
        "period": {"start": "2024-03-01"},
    }
    if participant_ref or participant_display:
        individual: dict = {}
        if participant_ref:
            individual["reference"] = participant_ref
        if participant_display:
            individual["display"] = participant_display
        res["participant"] = [{"individual": individual}]
    if service_provider_ref:
        res["serviceProvider"] = {"reference": service_provider_ref}
    if location_ref:
        res["location"] = [{"location": {"reference": location_ref}}]
    return res


def _entity(entity_class: str, text: str, **attrs) -> ExtractedEntity:
    return ExtractedEntity(
        entity_class=entity_class,
        text=text,
        attributes=dict(attrs),
        start_pos=0,
        end_pos=len(text),
        confidence=0.85,
    )


def _ai_encounter(**attrs) -> dict:
    text = attrs.pop("_text", "Office visit")
    rec = entity_to_health_record_dict(_entity("encounter", text, **attrs), uuid4(), uuid4(), uuid4())
    return rec["fhir_resource"]


# ---------------------------------------------------------------------------
# FHIR HumanName rendering
# ---------------------------------------------------------------------------


class TestHumanNameToDisplay:
    def test_given_family_list(self):
        name = [{"given": ["Jane"], "family": "Smith"}]
        assert _human_name_to_display(name) == "Jane Smith"

    def test_text_wins(self):
        name = [{"text": "Dr. Jane Smith", "given": ["Jane"], "family": "Smith"}]
        assert _human_name_to_display(name) == "Dr. Jane Smith"

    def test_prefix_and_suffix(self):
        name = [{"prefix": ["Dr."], "given": ["Jane"], "family": "Smith", "suffix": ["MD"]}]
        assert _human_name_to_display(name) == "Dr. Jane Smith MD"

    def test_plain_string(self):
        assert _human_name_to_display("Springfield Clinic") == "Springfield Clinic"

    def test_empty_returns_none(self):
        assert _human_name_to_display(None) is None
        assert _human_name_to_display([]) is None
        assert _human_name_to_display([{}]) is None


# ---------------------------------------------------------------------------
# Reference name map
# ---------------------------------------------------------------------------


class TestBuildReferenceNameMap:
    def test_practitioner_indexed_by_type_id_and_full_url(self):
        entries = [
            {
                "fullUrl": "urn:uuid:abc",
                "resource": {
                    "resourceType": "Practitioner",
                    "id": "p1",
                    "name": [{"given": ["Jane"], "family": "Smith"}],
                },
            }
        ]
        ref_map = build_reference_name_map(entries)
        assert ref_map["Practitioner/p1"] == "Jane Smith"
        assert ref_map["urn:uuid:abc"] == "Jane Smith"

    def test_organization_and_location_names(self):
        entries = [
            {"resource": {"resourceType": "Organization", "id": "o1", "name": "Mercy Hospital"}},
            {"resource": {"resourceType": "Location", "id": "l1", "name": "Cardiology Suite"}},
        ]
        ref_map = build_reference_name_map(entries)
        assert ref_map["Organization/o1"] == "Mercy Hospital"
        assert ref_map["Location/l1"] == "Cardiology Suite"

    def test_resource_without_name_skipped(self):
        entries = [{"resource": {"resourceType": "Practitioner", "id": "p2"}}]
        assert build_reference_name_map(entries) == {}

    def test_clinical_resources_ignored(self):
        entries = [{"resource": {"resourceType": "Condition", "id": "c1", "code": {"text": "x"}}}]
        assert build_reference_name_map(entries) == {}


# ---------------------------------------------------------------------------
# Resolve encounter references
# ---------------------------------------------------------------------------


class TestResolveEncounterReferences:
    def test_resolves_provider_facility_location(self):
        ref_map = {
            "Practitioner/p1": "Dr. Jane Smith",
            "Organization/o1": "Mercy Hospital",
            "Location/l1": "Cardiology Suite",
        }
        enc = _encounter(
            participant_ref="Practitioner/p1",
            service_provider_ref="Organization/o1",
            location_ref="Location/l1",
        )
        resolve_encounter_references(enc, ref_map)
        assert enc["participant"][0]["individual"]["display"] == "Dr. Jane Smith"
        assert enc["serviceProvider"]["display"] == "Mercy Hospital"
        assert enc["location"][0]["location"]["display"] == "Cardiology Suite"

    def test_existing_display_not_overwritten(self):
        ref_map = {"Practitioner/p1": "Resolved Name"}
        enc = _encounter(participant_ref="Practitioner/p1", participant_display="Original Name")
        resolve_encounter_references(enc, ref_map)
        assert enc["participant"][0]["individual"]["display"] == "Original Name"

    def test_unresolvable_reference_left_alone(self):
        enc = _encounter(participant_ref="Practitioner/missing")
        resolve_encounter_references(enc, {"Practitioner/other": "Someone"})
        assert "display" not in enc["participant"][0]["individual"]

    def test_empty_map_is_noop(self):
        enc = _encounter(participant_ref="Practitioner/p1")
        resolve_encounter_references(enc, {})
        assert "display" not in enc["participant"][0]["individual"]

    def test_non_encounter_untouched(self):
        obs = {"resourceType": "Observation", "performer": [{"reference": "Practitioner/p1"}]}
        before = json.loads(json.dumps(obs))
        resolve_encounter_references(obs, {"Practitioner/p1": "Dr. X"})
        assert obs == before


class TestMapFhirResourceWithRefMap:
    def test_provider_resolved_through_map(self):
        ref_map = {"Practitioner/p1": "Dr. Jane Smith"}
        mapped = map_fhir_resource(_encounter(participant_ref="Practitioner/p1"), ref_map)
        assert mapped is not None
        assert (
            mapped["fhir_resource"]["participant"][0]["individual"]["display"]
            == "Dr. Jane Smith"
        )

    def test_without_ref_map_still_maps(self):
        mapped = map_fhir_resource(_encounter(participant_ref="Practitioner/p1"))
        assert mapped is not None
        # No resolution possible, but the resource still maps cleanly.
        assert "display" not in mapped["fhir_resource"]["participant"][0]["individual"]


# ---------------------------------------------------------------------------
# Full parse_fhir_bundle wiring (DB-backed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_bundle_resolves_practitioner_name(
    db_session: AsyncSession, tmp_path: Path
):
    """A bundle whose Encounter references a Practitioner present as a separate
    resource ends up storing the provider name on the encounter."""
    user = User(
        id=uuid4(),
        email="enc_ref_resolution",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    patient = Patient(id=uuid4(), user_id=user.id, fhir_id="enc-ref-pat", gender="female")
    db_session.add(patient)
    await db_session.commit()

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "fullUrl": "urn:uuid:prac-1",
                "resource": {
                    "resourceType": "Practitioner",
                    "id": "prac-1",
                    "name": [{"given": ["Gregory"], "family": "House", "prefix": ["Dr."]}],
                },
            },
            {
                "fullUrl": "urn:uuid:org-1",
                "resource": {
                    "resourceType": "Organization",
                    "id": "org-1",
                    "name": "Princeton-Plainsboro Teaching Hospital",
                },
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "status": "finished",
                    "class": {"code": "AMB"},
                    "period": {"start": "2024-05-02"},
                    "participant": [{"individual": {"reference": "Practitioner/prac-1"}}],
                    "serviceProvider": {"reference": "Organization/org-1"},
                }
            },
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    await parse_fhir_bundle(bundle_path, user.id, patient.id, None, db_session)

    enc = (
        await db_session.execute(
            select(HealthRecord).where(
                HealthRecord.user_id == user.id,
                HealthRecord.record_type == "encounter",
            )
        )
    ).scalar_one()
    res = enc.fhir_resource
    assert res["participant"][0]["individual"]["display"] == "Dr. Gregory House"
    assert res["serviceProvider"]["display"] == "Princeton-Plainsboro Teaching Hospital"


# ---------------------------------------------------------------------------
# CDA reference resolution
# ---------------------------------------------------------------------------


def test_cda_encounter_resolves_provider_name(tmp_path: Path):
    """CDA-derived encounters get provider/facility names resolved from the
    bundle's Practitioner/Organization resources."""
    from app.services.ingestion import cda_parser

    cda_file = tmp_path / "doc.xml"
    cda_file.write_text("<ClinicalDocument/>")

    synthetic_bundle = {
        "entry": [
            {
                "resource": {
                    "resourceType": "Practitioner",
                    "id": "prac-9",
                    "name": [{"given": ["Lisa"], "family": "Cuddy"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Organization",
                    "id": "org-9",
                    "name": "General Hospital",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-9",
                    "status": "finished",
                    "class": {"code": "AMB"},
                    "period": {"start": "2024-04-01"},
                    "participant": [{"individual": {"reference": "Practitioner/prac-9"}}],
                    "serviceProvider": {"reference": "Organization/org-9"},
                }
            },
        ]
    }

    with patch.object(cda_parser, "_render_cda_to_fhir", return_value=synthetic_bundle):
        records = cda_parser.parse_cda_document(cda_file)

    enc = next(r for r in records if r["fhir_resource_type"] == "Encounter")
    res = enc["fhir_resource"]
    assert res["participant"][0]["individual"]["display"] == "Lisa Cuddy"
    assert res["serviceProvider"]["display"] == "General Hospital"


# ---------------------------------------------------------------------------
# Epic encounters — visit type + provider fallback
# ---------------------------------------------------------------------------


class TestEpicEncounterEnrichment:
    def test_visit_type_from_enc_type_column(self):
        row = {
            "CONTACT_DATE": "1/10/2024 12:00:00 AM",
            "ENC_TYPE_C_NAME": "Office Visit",
        }
        res = PatEncMapper().to_fhir(row)
        assert res["type"][0]["text"] == "Office Visit"

    def test_no_visit_type_no_type_field(self):
        res = PatEncMapper().to_fhir({"CONTACT_DATE": "1/10/2024 12:00:00 AM"})
        assert "type" not in res

    def test_provider_falls_back_to_pcp(self):
        row = {
            "CONTACT_DATE": "1/10/2024 12:00:00 AM",
            "PCP_PROV_ID_PROV_NAME": "Dr. Wilson",
        }
        res = PatEncMapper().to_fhir(row)
        assert res["participant"][0]["individual"]["display"] == "Dr. Wilson"

    def test_visit_provider_beats_pcp(self):
        row = {
            "CONTACT_DATE": "1/10/2024 12:00:00 AM",
            "VISIT_PROV_ID_PROV_NAME": "Dr. House",
            "PCP_PROV_ID_PROV_NAME": "Dr. Wilson",
        }
        res = PatEncMapper().to_fhir(row)
        assert res["participant"][0]["individual"]["display"] == "Dr. House"


# ---------------------------------------------------------------------------
# AI encounters — facility / visit type / summary
# ---------------------------------------------------------------------------


class TestAIEncounterEnrichment:
    def test_facility_from_medical_center_attr(self):
        res = _ai_encounter(visit_type="office", medical_center="Sunrise Medical Center")
        assert res["serviceProvider"]["display"] == "Sunrise Medical Center"

    def test_facility_from_facility_attr(self):
        res = _ai_encounter(visit_type="office", facility="Eastside Clinic")
        assert res["serviceProvider"]["display"] == "Eastside Clinic"

    def test_visit_type_title_from_entity_text(self):
        res = _ai_encounter(_text="GI Follow-Up Visit", visit_type="telehealth")
        assert res["type"][0]["text"] == "GI Follow-Up Visit"

    def test_explicit_visit_description_beats_text(self):
        res = _ai_encounter(_text="Office visit", visit_description="Annual Physical")
        assert res["type"][0]["text"] == "Annual Physical"

    def test_cpt_code_merged_into_type_concept(self):
        res = _ai_encounter(_text="Office visit", cpt_code="99213")
        concept = res["type"][0]
        assert concept["text"] == "Office visit"
        assert concept["coding"][0]["code"] == "99213"

    def test_summary_stored_as_narrative(self):
        res = _ai_encounter(
            visit_type="office",
            summary="Patient reports improved reflux; continue current regimen.",
        )
        div = res["text"]["div"]
        assert "improved reflux" in div
        assert res["text"]["status"] == "additional"

    def test_chief_complaint_used_as_summary(self):
        res = _ai_encounter(visit_type="office", chief_complaint="Abdominal pain x3 days")
        assert "Abdominal pain" in res["text"]["div"]

    def test_summary_escapes_xml(self):
        res = _ai_encounter(visit_type="office", summary="BP <120 & stable")
        div = res["text"]["div"]
        assert "&lt;120" in div and "&amp;" in div

    def test_no_summary_no_narrative(self):
        res = _ai_encounter(visit_type="office")
        assert "text" not in res

    def test_provider_still_attaches(self):
        res = _ai_encounter(visit_type="office", provider="Dr. Jane Smith")
        assert res["participant"][0]["individual"]["display"] == "Dr. Jane Smith"


# ---------------------------------------------------------------------------
# Extraction prompt / few-shot guards (the AI capture half)
# ---------------------------------------------------------------------------


class TestEncounterExtractionPrompt:
    def test_prompt_requests_facility_and_summary(self):
        from app.services.extraction.clinical_examples import CLINICAL_EXTRACTION_PROMPT

        p = CLINICAL_EXTRACTION_PROMPT.lower()
        assert "facility" in p or "medical_center" in p
        assert "summary" in p

    def test_example_encounter_carries_facility_and_summary(self):
        from app.services.extraction.clinical_examples import CLINICAL_EXAMPLES

        enc_attrs = [
            x.attributes
            for ex in CLINICAL_EXAMPLES
            for x in ex.extractions
            if x.extraction_class == "encounter"
        ]
        assert any(a.get("facility") or a.get("medical_center") for a in enc_attrs)
        assert any(a.get("summary") for a in enc_attrs)
