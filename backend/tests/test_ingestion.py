from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.ingestion.fhir_parser import (
    extract_coding,
    extract_effective_date,
    extract_status,
    map_fhir_resource,
    build_display_text,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_sample_bundle() -> dict:
    return json.loads((FIXTURES_DIR / "sample_fhir_bundle.json").read_text())


class TestFHIRParser:
    """Unit tests for FHIR parser functions (no DB needed)."""

    def test_map_condition(self):
        """Map a Condition resource to health_records fields."""
        resource = {
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes mellitus"}
                ]
            },
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "onsetDateTime": "2020-03-15",
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["record_type"] == "condition"
        assert result["fhir_resource_type"] == "Condition"
        assert result["code_system"] == "http://snomed.info/sct"
        assert result["code_value"] == "44054006"
        assert result["code_display"] == "Type 2 diabetes mellitus"
        assert result["status"] == "active"
        assert result["effective_date"] is not None

    def test_map_observation(self):
        """Map an Observation resource."""
        resource = {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "coding": [
                    {"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}
                ]
            },
            "effectiveDateTime": "2024-01-10T10:30:00Z",
            "valueQuantity": {"value": 6.8, "unit": "%"},
        }
        result = map_fhir_resource(resource)
        assert result is not None
        assert result["record_type"] == "observation"
        assert result["code_system"] == "http://loinc.org"
        assert result["effective_date"] is not None

    def test_unsupported_resource_returns_none(self):
        """Unsupported resource types return None."""
        result = map_fhir_resource({"resourceType": "Organization"})
        assert result is None

    def test_patient_not_in_supported(self):
        """Patient resource type is not in SUPPORTED_RESOURCE_TYPES."""
        result = map_fhir_resource({"resourceType": "Patient"})
        assert result is None

    def test_extract_dates(self):
        """Extract dates from various FHIR date fields."""
        assert extract_effective_date({"effectiveDateTime": "2024-01-10T10:30:00Z"}) is not None
        assert extract_effective_date({"onsetDateTime": "2020-03-15"}) is not None
        assert extract_effective_date({"issued": "2024-01-10T10:30:00+00:00"}) is not None
        assert extract_effective_date({}) is None

    def test_extract_dates_period(self):
        """Extract dates from period fields."""
        resource = {"effectivePeriod": {"start": "2024-01-01", "end": "2024-01-05"}}
        assert extract_effective_date(resource) is not None

    def test_extract_coding(self):
        """Extract coding system, code, display."""
        resource = {
            "code": {
                "coding": [
                    {"system": "http://loinc.org", "code": "12345", "display": "Test"}
                ]
            }
        }
        system, code, display = extract_coding(resource)
        assert system == "http://loinc.org"
        assert code == "12345"
        assert display == "Test"

    def test_extract_coding_text_only(self):
        """Extract display from code.text when no coding array."""
        resource = {"code": {"text": "Some condition"}}
        system, code, display = extract_coding(resource)
        assert system is None
        assert code is None
        assert display == "Some condition"

    def test_extract_coding_no_code(self):
        """No code object returns all None."""
        system, code, display = extract_coding({})
        assert system is None and code is None and display is None

    def test_extract_status(self):
        """Extract status from different FHIR patterns."""
        assert extract_status({"status": "active"}) == "active"
        assert extract_status({"clinicalStatus": {"coding": [{"code": "active"}]}}) == "active"
        assert extract_status({}) is None

    def test_build_display_text_encounter(self):
        """Build display text for Encounter resource."""
        resource = {"type": [{"text": "Office visit"}], "class": {"code": "AMB"}}
        text = build_display_text(resource, "Encounter")
        assert "Office visit" in text

    def test_build_display_text_immunization(self):
        """Build display text for Immunization resource."""
        resource = {"vaccineCode": {"coding": [{"display": "Flu shot"}]}}
        text = build_display_text(resource, "Immunization")
        assert "Flu shot" in text

    def test_parse_bundle_entry_count(self):
        """Verify sample bundle has expected entries."""
        bundle = _load_sample_bundle()
        entries = bundle.get("entry", [])
        assert len(entries) == 18  # Patient + 17 clinical resource entries

    def test_map_all_sample_entries(self):
        """Map all non-Patient entries from sample bundle."""
        bundle = _load_sample_bundle()
        mapped = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if resource and resource.get("resourceType") != "Patient":
                result = map_fhir_resource(resource)
                if result:
                    mapped.append(result)
        assert len(mapped) >= 10  # Most of the 17 non-Patient entries should map
        types = {m["record_type"] for m in mapped}
        assert "condition" in types
        assert "observation" in types
        assert "medication" in types
        assert "encounter" in types


class TestEpicMappers:
    """Unit tests for all Epic table mappers (no DB needed)."""

    def test_allergy_mapper(self):
        """AllergyMapper produces AllergyIntolerance."""
        from app.services.ingestion.epic_mappers.allergies import AllergyMapper

        mapper = AllergyMapper()
        row = {
            "ALLERGEN_ID_ALLERGEN_NAME": "Penicillin",
            "REACTION": "Hives",
            "DATE_NOTED": "3/15/2021 12:00:00 AM",
            "SEVERITY_C_NAME": "Severe",
            "ALRGY_STATUS_C_NAME": "Active",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "AllergyIntolerance"
        assert result["code"]["text"] == "Penicillin"
        assert result["reaction"][0]["manifestation"][0]["text"] == "Hives"
        assert result["reaction"][0]["severity"] == "severe"
        assert result["recordedDate"] is not None

    def test_allergy_mapper_empty_allergen_returns_none(self):
        """AllergyMapper returns None when allergen is empty."""
        from app.services.ingestion.epic_mappers.allergies import AllergyMapper

        mapper = AllergyMapper()
        assert mapper.to_fhir({"ALLERGEN_ID_ALLERGEN_NAME": ""}) is None

    def test_allergy_mapper_inactive_status(self):
        """AllergyMapper maps inactive status correctly."""
        from app.services.ingestion.epic_mappers.allergies import AllergyMapper

        mapper = AllergyMapper()
        row = {
            "ALLERGEN_ID_ALLERGEN_NAME": "Aspirin",
            "REACTION": "",
            "DATE_NOTED": "",
            "SEVERITY_C_NAME": "",
            "ALRGY_STATUS_C_NAME": "Inactive",
        }
        result = mapper.to_fhir(row)
        assert result["clinicalStatus"]["coding"][0]["code"] == "inactive"

    def test_immune_mapper(self):
        """ImmuneMapper produces Immunization."""
        from app.services.ingestion.epic_mappers.immunizations import ImmuneMapper

        mapper = ImmuneMapper()
        row = {
            "IMMUNZATN_ID_NAME": "COVID-19 Vaccine",
            "IMMUNE_DATE": "1/15/2021 12:00:00 AM",
            "DOSE": "0.3 mL",
            "ROUTE_C_NAME": "Intramuscular",
            "SITE_C_NAME": "Left Deltoid",
            "MFG_C_NAME": "Pfizer",
            "LOT": "EL9261",
            "IMMNZTN_STATUS_C_NAME": "Administered",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Immunization"
        assert result["vaccineCode"]["text"] == "COVID-19 Vaccine"
        assert result["status"] == "completed"
        assert result["manufacturer"]["display"] == "Pfizer"
        assert result["lotNumber"] == "EL9261"
        assert result["route"]["text"] == "Intramuscular"

    def test_immune_mapper_empty_name_returns_none(self):
        """ImmuneMapper returns None when vaccine name is empty."""
        from app.services.ingestion.epic_mappers.immunizations import ImmuneMapper

        mapper = ImmuneMapper()
        assert mapper.to_fhir({"IMMUNZATN_ID_NAME": ""}) is None

    def test_order_proc_mapper(self):
        """OrderProcMapper produces Procedure."""
        from app.services.ingestion.epic_mappers.procedures import OrderProcMapper

        mapper = OrderProcMapper()
        row = {
            "DESCRIPTION": "CT Abdomen with Contrast",
            "ORDER_INST": "6/10/2023 12:00:00 AM",
            "ORDER_STATUS_C_NAME": "Completed",
            "AUTHRZING_PROV_ID_PROV_NAME": "Dr. Smith",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Procedure"
        assert result["code"]["text"] == "CT Abdomen with Contrast"
        assert result["status"] == "completed"
        assert result["performer"][0]["actor"]["display"] == "Dr. Smith"

    def test_order_proc_mapper_pending_status(self):
        """OrderProcMapper maps pending status correctly."""
        from app.services.ingestion.epic_mappers.procedures import OrderProcMapper

        mapper = OrderProcMapper()
        row = {
            "DESCRIPTION": "MRI Brain",
            "ORDER_STATUS_C_NAME": "Pending",
        }
        result = mapper.to_fhir(row)
        assert result["status"] == "preparation"

    def test_vitals_mapper(self):
        """VitalsMapper produces Observation with vital-signs category."""
        from app.services.ingestion.epic_mappers.vitals import VitalsMapper

        mapper = VitalsMapper()
        row = {
            "FLO_MEAS_NAME": "Blood Pressure Systolic",
            "MEAS_VALUE": "120",
            "UNITS": "mmHg",
            "RECORDED_TIME": "2/15/2024 10:30:00 AM",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Observation"
        assert result["category"][0]["coding"][0]["code"] == "vital-signs"
        assert result["valueQuantity"]["value"] == 120.0
        assert result["valueQuantity"]["unit"] == "mmHg"

    def test_vitals_mapper_string_value(self):
        """VitalsMapper handles non-numeric values."""
        from app.services.ingestion.epic_mappers.vitals import VitalsMapper

        mapper = VitalsMapper()
        row = {
            "FLO_MEAS_NAME": "Pain Scale",
            "MEAS_VALUE": "Moderate",
            "UNITS": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["valueString"] == "Moderate"

    def test_vitals_mapper_no_value_returns_none(self):
        """VitalsMapper returns None when value is missing."""
        from app.services.ingestion.epic_mappers.vitals import VitalsMapper

        mapper = VitalsMapper()
        assert mapper.to_fhir({"FLO_MEAS_NAME": "BP", "MEAS_VALUE": ""}) is None

    def test_referral_mapper(self):
        """ReferralMapper produces ServiceRequest."""
        from app.services.ingestion.epic_mappers.referrals import ReferralMapper

        mapper = ReferralMapper()
        row = {
            "REFERRING_PROV_ID_REFERRING_PROV_NAM": "Dr. Jones",
            "REFERRAL_PROV_ID_PROV_NAME": "Dr. Specialist",
            "RFL_STATUS_C_NAME": "Completed",
            "RSN_FOR_RFL_C_NAME": "Cardiology Consultation",
            "START_DATE": "3/1/2024 12:00:00 AM",
            "EXP_DATE": "6/1/2024 12:00:00 AM",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "ServiceRequest"
        assert result["code"]["text"] == "Cardiology Consultation"
        assert result["status"] == "completed"
        assert result["requester"]["display"] == "Dr. Jones"
        assert result["performer"][0]["display"] == "Dr. Specialist"

    def test_referral_mapper_no_reason_no_provider_returns_none(self):
        """ReferralMapper returns None when both reason and provider are empty."""
        from app.services.ingestion.epic_mappers.referrals import ReferralMapper

        mapper = ReferralMapper()
        assert mapper.to_fhir({
            "RSN_FOR_RFL_C_NAME": "",
            "REFERRAL_PROV_ID_PROV_NAME": "",
        }) is None

    def test_encounter_dx_mapper(self):
        """EncounterDxMapper produces Condition with encounter-diagnosis category."""
        from app.services.ingestion.epic_mappers.encounter_dx import EncounterDxMapper

        mapper = EncounterDxMapper()
        row = {
            "DX_ID_DX_NAME": "Acute Bronchitis",
            "CONTACT_DATE": "2/10/2024 12:00:00 AM",
            "PRIMARY_DX_YN": "Y",
            "ANNOTATION": "Follow-up in 2 weeks",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Condition"
        assert result["code"]["text"] == "Acute Bronchitis"
        assert result["category"][0]["coding"][0]["code"] == "encounter-diagnosis"
        assert result["_primaryDiagnosis"] is True
        assert result["note"][0]["text"] == "Follow-up in 2 weeks"

    def test_encounter_dx_mapper_empty_dx_returns_none(self):
        """EncounterDxMapper returns None when dx name is empty."""
        from app.services.ingestion.epic_mappers.encounter_dx import EncounterDxMapper

        mapper = EncounterDxMapper()
        assert mapper.to_fhir({"DX_ID_DX_NAME": ""}) is None

    def test_social_hx_mapper(self):
        """SocialHxMapper produces Observation with social-history category."""
        from app.services.ingestion.epic_mappers.social_hx import SocialHxMapper

        mapper = SocialHxMapper()
        row = {
            "SOCIAL_HX_TYPE_C_NAME": "Tobacco Use",
            "SOCIAL_HX_COMMENT": "Former smoker, quit 2015",
            "CONTACT_DATE": "1/5/2024 12:00:00 AM",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Observation"
        assert result["category"][0]["coding"][0]["code"] == "social-history"
        assert result["code"]["text"] == "Tobacco Use"
        assert result["valueString"] == "Former smoker, quit 2015"

    def test_social_hx_mapper_empty_returns_none(self):
        """SocialHxMapper returns None when all fields empty."""
        from app.services.ingestion.epic_mappers.social_hx import SocialHxMapper

        mapper = SocialHxMapper()
        assert mapper.to_fhir({}) is None

    def test_family_hx_mapper(self):
        """FamilyHxMapper produces FamilyMemberHistory."""
        from app.services.ingestion.epic_mappers.family_hx import FamilyHxMapper

        mapper = FamilyHxMapper()
        row = {
            "FAM_MEDICAL_DX_ID_DX_NAME": "Type 2 Diabetes",
            "RELATION_C_NAME": "Mother",
            "AGE_OF_ONSET": "55",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "FamilyMemberHistory"
        assert result["relationship"]["text"] == "Mother"
        assert result["condition"][0]["code"]["text"] == "Type 2 Diabetes"
        assert result["condition"][0]["onsetAge"]["value"] == 55

    def test_family_hx_mapper_empty_dx_returns_none(self):
        """FamilyHxMapper returns None when dx name is empty."""
        from app.services.ingestion.epic_mappers.family_hx import FamilyHxMapper

        mapper = FamilyHxMapper()
        assert mapper.to_fhir({"FAM_MEDICAL_DX_ID_DX_NAME": ""}) is None

    def test_family_hx_mapper_non_numeric_onset(self):
        """FamilyHxMapper handles non-numeric age of onset."""
        from app.services.ingestion.epic_mappers.family_hx import FamilyHxMapper

        mapper = FamilyHxMapper()
        row = {
            "FAM_MEDICAL_DX_ID_DX_NAME": "Heart Disease",
            "RELATION_C_NAME": "Father",
            "AGE_OF_ONSET": "childhood",
        }
        result = mapper.to_fhir(row)
        assert result["condition"][0]["onsetString"] == "childhood"

    # ── ProblemListMapper tests ──────────────────────────────────────

    def test_problem_list_mapper_happy_path(self):
        """ProblemListMapper produces Condition with all fields populated."""
        from app.services.ingestion.epic_mappers.problems import ProblemListMapper

        mapper = ProblemListMapper()
        row = {
            "DX_ID_DX_NAME": "Essential Hypertension",
            "DESCRIPTION": "High blood pressure",
            "NOTED_DATE": "3/15/2021 12:00:00 AM",
            "RESOLVED_DATE": "",
            "PROBLEM_STATUS_C_NAME": "Active",
            "CHRONIC_YN": "Y",
            "PROBLEM_CMT": "Controlled with medication",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Condition"
        assert result["code"]["text"] == "High blood pressure"
        assert result["clinicalStatus"]["coding"][0]["code"] == "active"
        assert result["category"][0]["coding"][0]["code"] == "problem-list-item"
        assert result["onsetDateTime"] is not None
        # chronic flag adds a second category entry
        assert any(cat.get("text") == "chronic" for cat in result["category"])
        assert result["note"][0]["text"] == "Controlled with medication"

    def test_problem_list_mapper_empty_gate_returns_none(self):
        """ProblemListMapper returns None when both DX_ID_DX_NAME and DESCRIPTION are empty."""
        from app.services.ingestion.epic_mappers.problems import ProblemListMapper

        mapper = ProblemListMapper()
        row = {
            "DX_ID_DX_NAME": "",
            "DESCRIPTION": "",
            "NOTED_DATE": "",
            "RESOLVED_DATE": "",
            "PROBLEM_STATUS_C_NAME": "",
            "CHRONIC_YN": "",
            "PROBLEM_CMT": "",
        }
        assert mapper.to_fhir(row) is None

    def test_problem_list_mapper_resolved_status(self):
        """ProblemListMapper maps resolved status and abatementDateTime."""
        from app.services.ingestion.epic_mappers.problems import ProblemListMapper

        mapper = ProblemListMapper()
        row = {
            "DX_ID_DX_NAME": "Ankle Sprain",
            "DESCRIPTION": "",
            "NOTED_DATE": "1/10/2023 12:00:00 AM",
            "RESOLVED_DATE": "3/10/2023 12:00:00 AM",
            "PROBLEM_STATUS_C_NAME": "Resolved",
            "CHRONIC_YN": "N",
            "PROBLEM_CMT": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["clinicalStatus"]["coding"][0]["code"] == "resolved"
        assert "abatementDateTime" in result
        assert result["code"]["text"] == "Ankle Sprain"

    def test_problem_list_mapper_falls_back_to_dx_name(self):
        """ProblemListMapper uses DX_ID_DX_NAME when DESCRIPTION is empty."""
        from app.services.ingestion.epic_mappers.problems import ProblemListMapper

        mapper = ProblemListMapper()
        row = {
            "DX_ID_DX_NAME": "Type 2 Diabetes",
            "DESCRIPTION": "",
            "PROBLEM_STATUS_C_NAME": "Active",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["code"]["text"] == "Type 2 Diabetes"

    # ── MedicalHxMapper tests ────────────────────────────────────────

    def test_medical_hx_mapper_happy_path(self):
        """MedicalHxMapper produces Condition with Medical History category."""
        from app.services.ingestion.epic_mappers.problems import MedicalHxMapper

        mapper = MedicalHxMapper()
        row = {
            "DX_ID_DX_NAME": "Appendectomy",
            "MEDICAL_HX_DATE": "6/20/2015 12:00:00 AM",
            "MED_HX_ANNOTATION": "Emergency surgery, no complications",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Condition"
        assert result["code"]["text"] == "Appendectomy"
        assert result["category"][0]["text"] == "Medical History"
        assert result["category"][0]["coding"][0]["code"] == "problem-list-item"
        assert result["onsetDateTime"] is not None
        assert result["note"][0]["text"] == "Emergency surgery, no complications"
        assert result["clinicalStatus"]["coding"][0]["code"] == "active"

    def test_medical_hx_mapper_empty_gate_returns_none(self):
        """MedicalHxMapper returns None when DX_ID_DX_NAME is empty."""
        from app.services.ingestion.epic_mappers.problems import MedicalHxMapper

        mapper = MedicalHxMapper()
        assert mapper.to_fhir({"DX_ID_DX_NAME": ""}) is None
        assert mapper.to_fhir({}) is None

    def test_medical_hx_mapper_no_annotation_no_date(self):
        """MedicalHxMapper works with only the dx name, no optional fields."""
        from app.services.ingestion.epic_mappers.problems import MedicalHxMapper

        mapper = MedicalHxMapper()
        row = {
            "DX_ID_DX_NAME": "Childhood Asthma",
            "MEDICAL_HX_DATE": "",
            "MED_HX_ANNOTATION": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["code"]["text"] == "Childhood Asthma"
        assert "onsetDateTime" not in result
        assert "note" not in result

    # ── OrderMedMapper tests ─────────────────────────────────────────

    def test_order_med_mapper_happy_path(self):
        """OrderMedMapper produces MedicationRequest with all fields."""
        from app.services.ingestion.epic_mappers.medications import OrderMedMapper

        mapper = OrderMedMapper()
        row = {
            "DISPLAY_NAME": "Lisinopril 10 MG Oral Tablet",
            "MEDICATION_ID_MEDICATION_NAME": "Lisinopril",
            "DOSAGE": "10 mg daily",
            "DESCRIPTION": "Take once daily",
            "MED_ROUTE_C_NAME": "Oral",
            "START_DATE": "1/5/2024 12:00:00 AM",
            "END_DATE": "7/5/2024 12:00:00 AM",
            "ORDERING_DATE": "1/3/2024 12:00:00 AM",
            "ORDER_STATUS_C_NAME": "Completed",
            "QUANTITY": "90",
            "REFILLS": "3",
            "MED_PRESC_PROV_ID_PROV_NAME": "Dr. Garcia",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "MedicationRequest"
        assert result["medicationCodeableConcept"]["text"] == "Lisinopril 10 MG Oral Tablet"
        assert result["status"] == "completed"
        assert result["intent"] == "order"
        assert result["authoredOn"] is not None
        assert result["dosageInstruction"][0]["text"] == "10 mg daily"
        assert result["dosageInstruction"][0]["route"]["text"] == "Oral"
        assert result["dispenseRequest"]["quantity"]["value"] == "90"
        assert result["dispenseRequest"]["numberOfRepeatsAllowed"] == "3"
        assert result["requester"]["display"] == "Dr. Garcia"
        # effectivePeriod with start and end
        assert "effectivePeriod" in result
        assert "start" in result["effectivePeriod"]
        assert "end" in result["effectivePeriod"]

    def test_order_med_mapper_empty_gate_returns_none(self):
        """OrderMedMapper returns None when both DISPLAY_NAME and MEDICATION_ID_MEDICATION_NAME are empty."""
        from app.services.ingestion.epic_mappers.medications import OrderMedMapper

        mapper = OrderMedMapper()
        row = {
            "DISPLAY_NAME": "",
            "MEDICATION_ID_MEDICATION_NAME": "",
        }
        assert mapper.to_fhir(row) is None

    def test_order_med_mapper_cancelled_status(self):
        """OrderMedMapper maps cancelled/discontinued status correctly."""
        from app.services.ingestion.epic_mappers.medications import OrderMedMapper

        mapper = OrderMedMapper()
        row = {
            "DISPLAY_NAME": "Atorvastatin 20 MG",
            "ORDER_STATUS_C_NAME": "Discontinued",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["status"] == "cancelled"

    def test_order_med_mapper_fallback_to_medication_name(self):
        """OrderMedMapper falls back to MEDICATION_ID_MEDICATION_NAME when DISPLAY_NAME is empty."""
        from app.services.ingestion.epic_mappers.medications import OrderMedMapper

        mapper = OrderMedMapper()
        row = {
            "DISPLAY_NAME": "",
            "MEDICATION_ID_MEDICATION_NAME": "Metformin",
            "ORDER_STATUS_C_NAME": "Active",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["medicationCodeableConcept"]["text"] == "Metformin"
        assert result["status"] == "active"

    # ── OrderResultsMapper tests ─────────────────────────────────────

    def test_order_results_mapper_happy_path(self):
        """OrderResultsMapper produces Observation with all lab fields."""
        from app.services.ingestion.epic_mappers.results import OrderResultsMapper

        mapper = OrderResultsMapper()
        row = {
            "COMPONENT_ID_NAME": "Glucose",
            "ORD_VALUE": "95",
            "ORD_NUM_VALUE": "95",
            "REFERENCE_UNIT": "mg/dL",
            "REFERENCE_LOW": "70",
            "REFERENCE_HIGH": "100",
            "RESULT_DATE": "2/10/2024 12:00:00 AM",
            "RESULT_STATUS_C_NAME": "Final",
            "RESULT_FLAG_C_NAME": "",
            "COMPON_LNC_ID_LNC_LONG_NAME": "Glucose [Mass/volume] in Blood",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Observation"
        assert result["status"] == "final"
        assert result["category"][0]["coding"][0]["code"] == "laboratory"
        assert result["code"]["text"] == "Glucose"
        assert result["code"]["coding"][0]["system"] == "http://loinc.org"
        assert result["code"]["coding"][0]["display"] == "Glucose [Mass/volume] in Blood"
        assert result["valueQuantity"]["value"] == 95.0
        assert result["valueQuantity"]["unit"] == "mg/dL"
        assert result["effectiveDateTime"] is not None
        assert result["referenceRange"][0]["low"]["value"] == 70.0
        assert result["referenceRange"][0]["high"]["value"] == 100.0

    def test_order_results_mapper_empty_gate_returns_none(self):
        """OrderResultsMapper returns None when COMPONENT_ID_NAME is empty."""
        from app.services.ingestion.epic_mappers.results import OrderResultsMapper

        mapper = OrderResultsMapper()
        assert mapper.to_fhir({"COMPONENT_ID_NAME": ""}) is None
        assert mapper.to_fhir({}) is None

    def test_order_results_mapper_high_flag(self):
        """OrderResultsMapper maps high result flag to H interpretation."""
        from app.services.ingestion.epic_mappers.results import OrderResultsMapper

        mapper = OrderResultsMapper()
        row = {
            "COMPONENT_ID_NAME": "Hemoglobin A1c",
            "ORD_VALUE": "7.2",
            "ORD_NUM_VALUE": "7.2",
            "REFERENCE_UNIT": "%",
            "REFERENCE_LOW": "",
            "REFERENCE_HIGH": "",
            "RESULT_DATE": "",
            "RESULT_STATUS_C_NAME": "Final",
            "RESULT_FLAG_C_NAME": "High",
            "COMPON_LNC_ID_LNC_LONG_NAME": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["interpretation"][0]["coding"][0]["code"] == "H"

    def test_order_results_mapper_preliminary_status(self):
        """OrderResultsMapper maps preliminary status correctly."""
        from app.services.ingestion.epic_mappers.results import OrderResultsMapper

        mapper = OrderResultsMapper()
        row = {
            "COMPONENT_ID_NAME": "WBC",
            "ORD_VALUE": "Pending",
            "ORD_NUM_VALUE": "",
            "REFERENCE_UNIT": "",
            "RESULT_STATUS_C_NAME": "Preliminary",
            "RESULT_FLAG_C_NAME": "",
            "COMPON_LNC_ID_LNC_LONG_NAME": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["status"] == "preliminary"
        assert result["valueString"] == "Pending"
        assert "valueQuantity" not in result

    # ── PatEncMapper tests ───────────────────────────────────────────

    def test_pat_enc_mapper_happy_path(self):
        """PatEncMapper produces Encounter with all fields populated."""
        from app.services.ingestion.epic_mappers.encounters import PatEncMapper

        mapper = PatEncMapper()
        row = {
            "CONTACT_DATE": "4/15/2024 12:00:00 AM",
            "APPT_STATUS_C_NAME": "Completed",
            "FIN_CLASS_C_NAME": "Outpatient",
            "DEPARTMENT_ID_EXTERNAL_NAME": "Family Medicine Clinic",
            "VISIT_PROV_ID_PROV_NAME": "Dr. Williams",
            "VISIT_PROV_TITLE_NAME": "MD",
            "HOSP_DISCHRG_TIME": "",
            "CONTACT_COMMENT": "Annual physical exam",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "Encounter"
        assert result["status"] == "finished"
        assert result["class"]["code"] == "AMB"
        assert result["period"]["start"] is not None
        assert result["location"][0]["location"]["display"] == "Family Medicine Clinic"
        assert result["participant"][0]["individual"]["display"] == "Dr. Williams, MD"
        assert result["reasonCode"][0]["text"] == "Annual physical exam"

    def test_pat_enc_mapper_empty_gate_returns_none(self):
        """PatEncMapper returns None when CONTACT_DATE is empty or unparseable."""
        from app.services.ingestion.epic_mappers.encounters import PatEncMapper

        mapper = PatEncMapper()
        assert mapper.to_fhir({"CONTACT_DATE": ""}) is None
        assert mapper.to_fhir({}) is None

    def test_pat_enc_mapper_cancelled_no_show_status(self):
        """PatEncMapper maps 'No Show' to cancelled status."""
        from app.services.ingestion.epic_mappers.encounters import PatEncMapper

        mapper = PatEncMapper()
        row = {
            "CONTACT_DATE": "5/1/2024 12:00:00 AM",
            "APPT_STATUS_C_NAME": "No Show",
            "FIN_CLASS_C_NAME": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["status"] == "cancelled"

    def test_pat_enc_mapper_inpatient_class(self):
        """PatEncMapper maps inpatient financial class to IMP encounter class."""
        from app.services.ingestion.epic_mappers.encounters import PatEncMapper

        mapper = PatEncMapper()
        row = {
            "CONTACT_DATE": "6/10/2024 12:00:00 AM",
            "APPT_STATUS_C_NAME": "Completed",
            "FIN_CLASS_C_NAME": "Inpatient",
            "DEPARTMENT_ID_EXTERNAL_NAME": "ICU",
            "VISIT_PROV_ID_PROV_NAME": "Dr. Lee",
            "VISIT_PROV_TITLE_NAME": "",
            "HOSP_DISCHRG_TIME": "6/15/2024 12:00:00 AM",
            "CONTACT_COMMENT": "",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["class"]["code"] == "IMP"
        assert "end" in result["period"]
        # No title → provider name only (no trailing comma)
        assert result["participant"][0]["individual"]["display"] == "Dr. Lee"

    # ── DocInformationMapper tests ───────────────────────────────────

    def test_doc_information_mapper_happy_path(self):
        """DocInformationMapper produces DocumentReference with all fields."""
        from app.services.ingestion.epic_mappers.documents import DocInformationMapper

        mapper = DocInformationMapper()
        row = {
            "DOC_INFO_TYPE_C_NAME": "Discharge Summary",
            "DOC_RECV_TIME": "7/20/2024 12:00:00 AM",
            "DOC_STAT_C_NAME": "Active",
            "DOC_DESCR": "Post-surgical discharge instructions",
            "RECV_BY_USER_ID_NAME": "Nurse Johnson",
            "IS_SCANNED_YN": "Y",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["resourceType"] == "DocumentReference"
        assert result["type"]["text"] == "Discharge Summary"
        assert result["status"] == "current"
        assert result["description"] == "Post-surgical discharge instructions"
        assert result["date"] is not None
        assert result["author"][0]["display"] == "Nurse Johnson"
        assert result["category"][0]["text"] == "scanned"

    def test_doc_information_mapper_empty_gate_returns_none(self):
        """DocInformationMapper returns None when DOC_INFO_TYPE_C_NAME is empty."""
        from app.services.ingestion.epic_mappers.documents import DocInformationMapper

        mapper = DocInformationMapper()
        assert mapper.to_fhir({"DOC_INFO_TYPE_C_NAME": ""}) is None
        assert mapper.to_fhir({}) is None

    def test_doc_information_mapper_superseded_status(self):
        """DocInformationMapper maps deleted/inactive status to superseded."""
        from app.services.ingestion.epic_mappers.documents import DocInformationMapper

        mapper = DocInformationMapper()
        row = {
            "DOC_INFO_TYPE_C_NAME": "Lab Report",
            "DOC_RECV_TIME": "",
            "DOC_STAT_C_NAME": "Inactive",
            "DOC_DESCR": "",
            "RECV_BY_USER_ID_NAME": "",
            "IS_SCANNED_YN": "N",
        }
        result = mapper.to_fhir(row)
        assert result is not None
        assert result["status"] == "superseded"
        # description falls back to doc_type when DOC_DESCR is empty
        assert result["description"] == "Lab Report"
        assert "date" not in result
        assert "author" not in result
        assert "category" not in result


class TestEpicMapperRegistration:
    """Verify all mappers are properly registered."""

    def test_all_mappers_registered(self):
        """All expected Epic table names have mappers."""
        from app.services.ingestion.epic_parser import EPIC_TABLE_MAPPERS

        expected_tables = {
            "PROBLEM_LIST", "PROBLEM_LIST_ALL", "MEDICAL_HX",
            "ORDER_MED", "ORDER_RESULTS", "PAT_ENC", "DOC_INFORMATION",
            "ALLERGY", "IMMUNE", "ORDER_PROC", "IP_FLWSHT_MEAS",
            "REFERRAL", "PAT_ENC_DX", "SOCIAL_HX", "FAMILY_HX",
        }
        assert expected_tables.issubset(set(EPIC_TABLE_MAPPERS.keys()))

    def test_record_type_map_includes_new_types(self):
        """RECORD_TYPE_MAP includes ServiceRequest and FamilyMemberHistory."""
        from app.services.ingestion.epic_parser import RECORD_TYPE_MAP

        assert "ServiceRequest" in RECORD_TYPE_MAP
        assert RECORD_TYPE_MAP["ServiceRequest"] == "service_request"
        assert "FamilyMemberHistory" in RECORD_TYPE_MAP
        assert RECORD_TYPE_MAP["FamilyMemberHistory"] == "condition"


class TestEpicFixtures:
    """Verify synthetic Epic TSV fixtures exist and have correct structure."""

    def test_epic_tsv_files_exist(self):
        """Sample Epic TSV directory has expected files."""
        tsv_dir = FIXTURES_DIR / "sample_epic_tsv"
        if not tsv_dir.exists():
            pytest.skip("No sample Epic TSV directory")
        expected = {"PATIENT.tsv", "PROBLEM_LIST.tsv", "ORDER_RESULTS.tsv",
                    "MEDICATIONS.tsv", "ENCOUNTERS.tsv", "ALLERGIES.tsv"}
        actual = {f.name for f in tsv_dir.glob("*.tsv")}
        assert expected.issubset(actual)

    def test_epic_tsv_has_headers(self):
        """Each Epic TSV file has a header row."""
        tsv_dir = FIXTURES_DIR / "sample_epic_tsv"
        if not tsv_dir.exists():
            pytest.skip("No sample Epic TSV directory")
        for tsv_path in tsv_dir.glob("*.tsv"):
            with open(tsv_path, "r", encoding="utf-8-sig") as f:
                header = f.readline().strip()
                assert len(header) > 0, f"Empty header in {tsv_path.name}"
                assert "\t" in header, f"No tab separator in {tsv_path.name} header"
