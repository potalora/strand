"""B6 richness for fhir_parser: surface a facility/serviceProvider on FHIR
Encounters that name a facility-like location but carry no serviceProvider.

Conservative by design — only promotes a location whose display clearly names a
facility (keyword-gated), never a room/bed/exam-room. Pure-function, no DB.
"""
from __future__ import annotations

from app.services.ingestion.fhir_parser import (
    enrich_encounter_service_provider,
    map_fhir_resource,
)


def _encounter(location_display: str | None = None, service_provider=None) -> dict:
    res: dict = {
        "resourceType": "Encounter",
        "status": "finished",
        "class": {"code": "AMB"},
        "period": {"start": "2024-03-01"},
    }
    if location_display is not None:
        res["location"] = [{"location": {"display": location_display}}]
    if service_provider is not None:
        res["serviceProvider"] = service_provider
    return res


class TestEnrichServiceProvider:
    def test_facility_location_promoted(self):
        res = enrich_encounter_service_provider(_encounter("Downtown Medical Center"))
        assert res["serviceProvider"]["display"] == "Downtown Medical Center"

    def test_department_name_promoted(self):
        res = enrich_encounter_service_provider(_encounter("Internal Medicine"))
        assert res["serviceProvider"]["display"] == "Internal Medicine"

    def test_room_location_not_promoted(self):
        res = enrich_encounter_service_provider(_encounter("Exam Room 3"))
        assert "serviceProvider" not in res

    def test_existing_service_provider_untouched(self):
        res = enrich_encounter_service_provider(
            _encounter("Downtown Medical Center", service_provider={"reference": "Organization/1"})
        )
        assert res["serviceProvider"] == {"reference": "Organization/1"}

    def test_no_location_no_change(self):
        res = enrich_encounter_service_provider(_encounter())
        assert "serviceProvider" not in res

    def test_non_encounter_untouched(self):
        obs = {"resourceType": "Observation", "status": "final"}
        assert enrich_encounter_service_provider(obs) == obs


class TestMapFhirResourceAppliesEnrichment:
    def test_encounter_through_map_gets_service_provider(self):
        mapped = map_fhir_resource(_encounter("Westside Health Clinic"))
        assert mapped is not None
        assert mapped["fhir_resource"]["serviceProvider"]["display"] == "Westside Health Clinic"
