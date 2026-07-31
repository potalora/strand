from __future__ import annotations

import base64
from uuid import uuid4


from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import (
    _build_display_text,
    entity_to_health_record_dict,
)

USER_ID = uuid4()
PATIENT_ID = uuid4()
SOURCE_FILE_ID = uuid4()


def _make_entity(entity_class: str, text: str, **attrs) -> ExtractedEntity:
    return ExtractedEntity(
        entity_class=entity_class, text=text, attributes=attrs, confidence=0.85
    )


# ---------- TestEncounterBuilder ----------


class TestEncounterBuilder:
    def test_produces_record(self):
        entity = _make_entity("encounter", "Office visit", visit_type="office")
        result = entity_to_health_record_dict(
            entity, USER_ID, PATIENT_ID, SOURCE_FILE_ID
        )
        assert result is not None
        assert result["record_type"] == "encounter"
        assert result["fhir_resource_type"] == "Encounter"
        assert result["ai_extracted"] is True

    def test_fhir_structure(self):
        entity = _make_entity(
            "encounter",
            "Follow-up",
            visit_type="office",
            cpt_code="99213",
            reason="Diabetes follow-up",
            date="2024-03-15",
        )
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        fhir = result["fhir_resource"]
        assert fhir["resourceType"] == "Encounter"
        assert fhir["status"] == "finished"
        assert fhir["class"]["code"] == "AMB"
        assert fhir["class"]["display"] == "ambulatory"
        assert fhir["type"][0]["coding"][0]["code"] == "99213"
        assert fhir["reasonCode"][0]["text"] == "Diabetes follow-up"
        assert fhir["period"]["start"] == "2024-03-15"

    def test_class_mapping(self):
        mappings = {
            "office": ("AMB", "ambulatory"),
            "telehealth": ("VR", "virtual"),
            "emergency": ("EMER", "emergency"),
            "inpatient": ("IMP", "inpatient encounter"),
        }
        for visit_type, (expected_code, expected_display) in mappings.items():
            entity = _make_entity("encounter", "Visit", visit_type=visit_type)
            result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
            fhir = result["fhir_resource"]
            assert fhir["class"]["code"] == expected_code, f"Failed for {visit_type}"
            assert fhir["class"]["display"] == expected_display, (
                f"Failed for {visit_type}"
            )

    def test_display_text(self):
        entity = _make_entity(
            "encounter", "Visit", visit_type="office", date="2024-03-15"
        )
        assert _build_display_text(entity) == "Office encounter — 2024-03-15"


# ---------- TestDiagnosticReportBuilder ----------


class TestDiagnosticReportBuilder:
    def test_produces_record(self):
        entity = _make_entity("imaging_result", "Chest X-ray")
        result = entity_to_health_record_dict(
            entity, USER_ID, PATIENT_ID, SOURCE_FILE_ID
        )
        assert result is not None
        assert result["record_type"] == "diagnostic_report"
        assert result["fhir_resource_type"] == "DiagnosticReport"

    def test_fhir_structure(self):
        entity = _make_entity(
            "imaging_result",
            "Chest X-ray",
            procedure_name="Chest X-ray PA",
            findings="No acute findings",
            interpretation="Normal",
        )
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        fhir = result["fhir_resource"]
        assert fhir["resourceType"] == "DiagnosticReport"
        assert fhir["status"] == "final"
        assert fhir["code"]["text"] == "Chest X-ray PA"
        assert fhir["conclusion"] == "No acute findings"
        assert fhir["conclusionCode"][0]["text"] == "Normal"
        assert fhir["category"][0]["coding"][0]["code"] == "imaging"

    def test_display_text(self):
        entity = _make_entity(
            "imaging_result", "MRI Brain", procedure_name="MRI Brain", findings="Normal"
        )
        assert _build_display_text(entity) == "MRI Brain: Normal"

        entity_no_findings = _make_entity(
            "imaging_result", "CT Abdomen", procedure_name="CT Abdomen"
        )
        assert _build_display_text(entity_no_findings) == "CT Abdomen"


# ---------- TestFamilyHistoryBuilder ----------


class TestFamilyHistoryBuilder:
    def test_produces_record(self):
        entity = _make_entity("family_history", "Breast cancer", relationship="mother")
        result = entity_to_health_record_dict(
            entity, USER_ID, PATIENT_ID, SOURCE_FILE_ID
        )
        assert result is not None
        assert result["record_type"] == "family_history"
        assert result["fhir_resource_type"] == "FamilyMemberHistory"

    def test_fhir_structure(self):
        entity = _make_entity(
            "family_history",
            "Diabetes",
            relationship="father",
            condition="Type 2 Diabetes",
            notes="Diagnosed at age 50",
        )
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        fhir = result["fhir_resource"]
        assert fhir["resourceType"] == "FamilyMemberHistory"
        assert fhir["status"] == "completed"
        rel_coding = fhir["relationship"]["coding"][0]
        assert (
            rel_coding["system"] == "http://terminology.hl7.org/CodeSystem/v3-RoleCode"
        )
        assert rel_coding["code"] == "FTH"
        assert rel_coding["display"] == "Father"
        assert fhir["condition"][0]["code"]["text"] == "Type 2 Diabetes"
        assert fhir["condition"][0]["note"][0]["text"] == "Diagnosed at age 50"

    def test_relationship_mapping(self):
        mappings = {
            "mother": ("MTH", "Mother"),
            "father": ("FTH", "Father"),
            "sibling": ("SIB", "Sibling"),
            "grandmother": ("GRMTH", "Grandmother"),
            "grandfather": ("GRFTH", "Grandfather"),
            "grandparent": ("GRPRN", "Grandparent"),
        }
        for rel, (expected_code, expected_display) in mappings.items():
            entity = _make_entity("family_history", "Some condition", relationship=rel)
            result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
            fhir = result["fhir_resource"]
            coding = fhir["relationship"]["coding"][0]
            assert coding["code"] == expected_code, f"Failed for {rel}"
            assert coding["display"] == expected_display, f"Failed for {rel}"

    def test_display_text(self):
        entity = _make_entity(
            "family_history",
            "Heart disease",
            relationship="mother",
            condition="Heart disease",
        )
        assert _build_display_text(entity) == "Mother: Heart disease"


# ---------- TestAssessmentPlanBuilder ----------


class TestAssessmentPlanBuilder:
    def test_produces_record(self):
        entity = _make_entity("assessment_plan", "Continue current medications")
        result = entity_to_health_record_dict(
            entity, USER_ID, PATIENT_ID, SOURCE_FILE_ID
        )
        assert result is not None
        assert result["record_type"] == "document"
        assert result["fhir_resource_type"] == "DocumentReference"

    def test_fhir_structure(self):
        entity = _make_entity(
            "assessment_plan",
            "Patient improving",
            plan_items=["Continue metformin", "Follow up in 3 months"],
        )
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        fhir = result["fhir_resource"]
        assert fhir["resourceType"] == "DocumentReference"
        assert fhir["status"] == "current"
        assert fhir["type"]["coding"][0]["code"] == "51847-2"
        assert fhir["type"]["coding"][0]["system"] == "http://loinc.org"
        # Content should be base64-encoded
        decoded = base64.b64decode(fhir["content"][0]["attachment"]["data"]).decode()
        assert decoded == "Patient improving"
        assert fhir["description"] == "Continue metformin; Follow up in 3 months"

    def test_display_text(self):
        entity = _make_entity(
            "assessment_plan", "Plan notes", plan_items=["Item 1", "Item 2", "Item 3"]
        )
        assert _build_display_text(entity) == "Assessment & Plan (3 items)"

        entity_no_items = _make_entity("assessment_plan", "Plan notes")
        assert _build_display_text(entity_no_items) == "Assessment & Plan"


# ---------- TestSocialHistoryBuilder ----------


class TestSocialHistoryBuilder:
    def test_produces_record(self):
        entity = _make_entity("social_history", "Non-smoker", category="smoking_status")
        result = entity_to_health_record_dict(
            entity, USER_ID, PATIENT_ID, SOURCE_FILE_ID
        )
        assert result is not None
        assert result["record_type"] == "observation"
        assert result["fhir_resource_type"] == "Observation"

    def test_fhir_structure(self):
        entity = _make_entity(
            "social_history",
            "Non-smoker",
            category="smoking_status",
            value="Never smoker",
        )
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        fhir = result["fhir_resource"]
        assert fhir["resourceType"] == "Observation"
        assert fhir["status"] == "final"
        assert fhir["category"][0]["coding"][0]["code"] == "social-history"
        assert fhir["code"]["text"] == "Smoking Status"
        assert fhir["valueString"] == "Never smoker"

    def test_display_text(self):
        entity = _make_entity(
            "social_history",
            "Non-smoker",
            category="smoking_status",
            value="Never smoker",
        )
        assert _build_display_text(entity) == "Smoking Status: Never smoker"


# ---------- TestExistingEntityTypesUnchanged ----------


class TestExistingEntityTypesUnchanged:
    def test_medication_still_works(self):
        entity = _make_entity("medication", "Metformin", medication_group="Metformin")
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        assert result is not None
        assert result["record_type"] == "medication"
        assert result["fhir_resource"]["resourceType"] == "MedicationRequest"

    def test_condition_still_works(self):
        entity = _make_entity("condition", "hypertension", status="active")
        result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
        assert result is not None
        assert result["record_type"] == "condition"
        assert (
            result["fhir_resource"]["clinicalStatus"]["coding"][0]["code"] == "active"
        )

    def test_non_storable_returns_none(self):
        for cls in ("provider", "dosage", "route", "frequency", "duration", "date"):
            entity = _make_entity(cls, "some text")
            result = entity_to_health_record_dict(entity, USER_ID, PATIENT_ID)
            assert result is None, f"{cls} should return None"
