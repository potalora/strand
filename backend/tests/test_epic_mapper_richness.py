"""Richness tests for Epic mappers: RxNorm coding + structured dosage on
OrderMedMapper (B1/B4), and serviceProvider/facility on PatEncMapper (B6).

Pure-function tests — no DB.
"""
from __future__ import annotations

from app.services.extraction import terminology as t
from app.services.ingestion.epic_mappers.encounters import PatEncMapper
from app.services.ingestion.epic_mappers.medications import OrderMedMapper


class TestOrderMedRxNorm:
    def test_known_med_gets_rxnorm_coding(self):
        row = {"DISPLAY_NAME": "Metformin 500mg", "ORDER_STATUS_C_NAME": "Active"}
        res = OrderMedMapper().to_fhir(row)
        cc = res["medicationCodeableConcept"]
        assert cc["text"] == "Metformin 500mg"
        assert cc["coding"][0]["system"] == t.RXNORM_SYSTEM
        assert cc["coding"][0]["code"] == "6809"

    def test_unknown_med_has_no_coding(self):
        row = {"DISPLAY_NAME": "Experimental Compound XZ", "ORDER_STATUS_C_NAME": "Active"}
        res = OrderMedMapper().to_fhir(row)
        assert "coding" not in res["medicationCodeableConcept"]

    def test_fallback_name_column_still_codes(self):
        row = {"MEDICATION_ID_MEDICATION_NAME": "Lisinopril", "ORDER_STATUS_C_NAME": "Active"}
        res = OrderMedMapper().to_fhir(row)
        assert res["medicationCodeableConcept"]["coding"][0]["code"] == "29046"


class TestOrderMedDosage:
    def test_structured_dose_and_timing(self):
        row = {
            "DISPLAY_NAME": "Metformin 500mg",
            "DOSAGE": "500 mg twice daily",
            "MED_ROUTE_C_NAME": "Oral",
            "ORDER_STATUS_C_NAME": "Active",
        }
        di = OrderMedMapper().to_fhir(row)["dosageInstruction"][0]
        # original text + route preserved (fidelity contract)
        assert di["text"] == "500 mg twice daily"
        assert di["route"]["text"] == "Oral"
        # newly structured
        assert di["doseAndRate"][0]["doseQuantity"]["value"] == 500.0
        assert di["doseAndRate"][0]["doseQuantity"]["unit"] == "mg"
        assert di["timing"]["repeat"]["frequency"] == 2
        assert di["timing"]["repeat"]["periodUnit"] == "d"

    def test_no_dosage_no_instruction(self):
        row = {"DISPLAY_NAME": "Metformin", "ORDER_STATUS_C_NAME": "Active"}
        res = OrderMedMapper().to_fhir(row)
        assert "dosageInstruction" not in res


class TestPatEncFacility:
    def test_service_provider_from_location_column(self):
        row = {
            "CONTACT_DATE": "1/10/2024 12:00:00 AM",
            "LOC_ID_LOC_NAME": "Downtown Medical Center",
            "DEPARTMENT_ID_EXTERNAL_NAME": "Internal Medicine",
        }
        res = PatEncMapper().to_fhir(row)
        assert res["serviceProvider"]["display"] == "Downtown Medical Center"
        # department still drives location
        assert res["location"][0]["location"]["display"] == "Internal Medicine"

    def test_service_provider_falls_back_to_department(self):
        row = {
            "CONTACT_DATE": "1/10/2024 12:00:00 AM",
            "DEPARTMENT_ID_EXTERNAL_NAME": "Cardiology",
        }
        res = PatEncMapper().to_fhir(row)
        assert res["serviceProvider"]["display"] == "Cardiology"

    def test_no_facility_no_service_provider(self):
        row = {"CONTACT_DATE": "1/10/2024 12:00:00 AM"}
        res = PatEncMapper().to_fhir(row)
        assert "serviceProvider" not in res
