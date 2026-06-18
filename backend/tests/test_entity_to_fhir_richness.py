"""Richness tests for entity_to_fhir: coding, providers, lab values, dosage,
condition onset, diagnostic-report performer, and lab-panel orders.

These pin the recall improvements (B1-B8) for the AI-extracted (LangExtract)
path. All pure functions — no DB.
"""
from __future__ import annotations

from uuid import uuid4

from app.services.extraction import terminology as t
from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import (
    entity_to_health_record_dict,
    parse_dosage,
    parse_lab_measurement,
    resolve_document_provider,
)


def _entity(entity_class: str, text: str, **attrs) -> ExtractedEntity:
    return ExtractedEntity(
        entity_class=entity_class,
        text=text,
        attributes=dict(attrs),
        start_pos=0,
        end_pos=len(text),
        confidence=0.85,
    )


def _record(entity: ExtractedEntity, **kw):
    return entity_to_health_record_dict(entity, uuid4(), uuid4(), uuid4(), **kw)


# --------------------------------------------------------------------------
# B1 — terminology coding on AI records
# --------------------------------------------------------------------------


class TestConditionCoding:
    def test_condition_gets_icd10_code_in_dict(self):
        rec = _record(_entity("condition", "Type 2 Diabetes", status="active"))
        assert rec["code_system"] == t.ICD10_SYSTEM
        assert rec["code_value"] == "E11.9"

    def test_condition_resource_carries_coding(self):
        rec = _record(_entity("condition", "Hypertension", status="active"))
        coding = rec["fhir_resource"]["code"]["coding"][0]
        assert coding["system"] == t.ICD10_SYSTEM
        assert coding["code"] == "I10"
        # original text preserved alongside the coding
        assert rec["fhir_resource"]["code"]["text"] == "Hypertension"

    def test_unknown_condition_stays_uncoded(self):
        rec = _record(_entity("condition", "Some Rare Syndrome", status="active"))
        assert rec["code_system"] is None
        assert rec["code_value"] is None
        assert "coding" not in rec["fhir_resource"]["code"]


class TestMedicationCoding:
    def test_medication_gets_rxnorm(self):
        rec = _record(_entity("medication", "Metformin 500mg"))
        assert rec["code_system"] == t.RXNORM_SYSTEM
        assert rec["code_value"] == "6809"
        coding = rec["fhir_resource"]["medicationCodeableConcept"]["coding"][0]
        assert coding["code"] == "6809"

    def test_unknown_med_uncoded(self):
        rec = _record(_entity("medication", "Go"))
        assert rec["code_value"] is None


class TestLabCoding:
    def test_lab_gets_loinc(self):
        rec = _record(_entity("lab_result", "Glucose", value="95", unit="mg/dL"))
        assert rec["code_system"] == t.LOINC_SYSTEM
        assert rec["code_value"] == "2345-7"


class TestProcedureCoding:
    def test_procedure_gets_local_marker(self):
        # SNOMED CT and CPT are license-restricted, so procedures resolve to a
        # local, permissively-licensed PROCEDURE_SYSTEM marker (not SNOMED).
        rec = _record(_entity("procedure", "Colonoscopy", date="2024-03-01"))
        assert rec["code_system"] == t.PROCEDURE_SYSTEM
        assert rec["code_value"] == "colonoscopy"


# --------------------------------------------------------------------------
# B3 — lab value + unit + reference range + interpretation
# --------------------------------------------------------------------------


class TestParseLabMeasurement:
    def test_full_string(self):
        r = parse_lab_measurement("Glucose 95 mg/dL (70-99) H")
        assert r["value"] == 95.0
        assert r["unit"] == "mg/dL"
        assert r["ref_low"] == 70.0
        assert r["ref_high"] == 99.0
        assert r["interpretation"] == "H"

    def test_en_dash_range_no_flag(self):
        r = parse_lab_measurement("Sodium 140 mmol/L (135–145)")
        assert r["value"] == 140.0
        assert r["unit"] == "mmol/L"
        assert r["ref_low"] == 135.0
        assert r["ref_high"] == 145.0
        assert r["interpretation"] is None

    def test_percent_unit(self):
        r = parse_lab_measurement("Hemoglobin A1c 6.8%")
        assert r["value"] == 6.8
        assert r["unit"] == "%"

    def test_value_only(self):
        r = parse_lab_measurement("95")
        assert r["value"] == 95.0
        assert r["unit"] is None

    def test_low_flag_not_confused_with_liter_unit(self):
        r = parse_lab_measurement("TSH 2.5 mIU/L")
        assert r["value"] == 2.5
        assert r["unit"] == "mIU/L"
        assert r["interpretation"] is None

    def test_word_flag_low(self):
        r = parse_lab_measurement("Potassium 3.1 mmol/L (3.5-5.0) Low")
        assert r["interpretation"] == "L"
        assert r["ref_low"] == 3.5

    def test_no_numeric_value(self):
        r = parse_lab_measurement("Positive")
        assert r["value"] is None


class TestLabObservationBuild:
    def test_lab_from_text_gets_quantity_range_interp(self):
        rec = _record(_entity("lab_result", "Glucose 95 mg/dL (70-99) H"))
        res = rec["fhir_resource"]
        assert res["valueQuantity"]["value"] == 95.0
        assert res["valueQuantity"]["unit"] == "mg/dL"
        assert res["referenceRange"][0]["low"]["value"] == 70.0
        assert res["referenceRange"][0]["high"]["value"] == 99.0
        assert res["interpretation"][0]["coding"][0]["code"] == "H"

    def test_explicit_attrs_still_win(self):
        rec = _record(_entity("lab_result", "Glucose", test="Glucose", value="88", unit="mg/dL"))
        res = rec["fhir_resource"]
        assert res["valueQuantity"]["value"] == 88.0
        assert res["valueQuantity"]["unit"] == "mg/dL"


# --------------------------------------------------------------------------
# B2 — provider / performer attachment
# --------------------------------------------------------------------------


class TestProviderAttachment:
    def test_encounter_participant_from_attr(self):
        rec = _record(_entity("encounter", "Office visit", visit_type="office",
                              provider="Dr. Jane Smith", date="2024-03-01"))
        res = rec["fhir_resource"]
        assert res["participant"][0]["individual"]["display"] == "Dr. Jane Smith"

    def test_observation_performer_from_attr(self):
        rec = _record(_entity("lab_result", "Glucose", value="95", unit="mg/dL",
                              performer="Dr. Jane Smith"))
        res = rec["fhir_resource"]
        assert res["performer"][0]["display"] == "Dr. Jane Smith"

    def test_procedure_performer_from_attr(self):
        rec = _record(_entity("procedure", "Colonoscopy", date="2024-03-01",
                              provider="Dr. Gomez"))
        res = rec["fhir_resource"]
        assert res["performer"][0]["actor"]["display"] == "Dr. Gomez"

    def test_document_provider_fallback(self):
        rec = _record(
            _entity("procedure", "Colonoscopy", date="2024-03-01"),
            document_provider="Dr. Attending",
        )
        res = rec["fhir_resource"]
        assert res["performer"][0]["actor"]["display"] == "Dr. Attending"

    def test_entity_provider_beats_document_provider(self):
        rec = _record(
            _entity("procedure", "Colonoscopy", date="2024-03-01", provider="Dr. Own"),
            document_provider="Dr. Attending",
        )
        assert rec["fhir_resource"]["performer"][0]["actor"]["display"] == "Dr. Own"


# --------------------------------------------------------------------------
# B4 — medication dosage parsing
# --------------------------------------------------------------------------


class TestParseDosage:
    def test_dose_route_freq(self):
        r = parse_dosage("500 mg PO twice daily")
        assert r["dose_value"] == 500.0
        assert r["dose_unit"] == "mg"
        assert r["route"] == "oral"
        assert r["frequency"] == 2
        assert r["period_unit"] == "d"

    def test_abbreviated_sig(self):
        r = parse_dosage("10mg po qd")
        assert r["dose_value"] == 10.0
        assert r["route"] == "oral"
        assert r["frequency"] == 1

    def test_prn(self):
        r = parse_dosage("400 mg as needed")
        assert r["as_needed"] is True

    def test_weekly(self):
        r = parse_dosage("Methotrexate 15mg once weekly")
        assert r["period_unit"] == "wk"


class TestMedicationDosageBuild:
    def test_dosage_instruction_structured(self):
        rec = _record(_entity("medication", "Metformin", dosage="500 mg PO twice daily"))
        di = rec["fhir_resource"]["dosageInstruction"][0]
        assert di["doseAndRate"][0]["doseQuantity"]["value"] == 500.0
        assert di["doseAndRate"][0]["doseQuantity"]["unit"] == "mg"
        assert di["route"]["text"] == "oral"
        assert di["timing"]["repeat"]["frequency"] == 2
        assert di["timing"]["repeat"]["periodUnit"] == "d"
        assert di["text"]  # original sig preserved


# --------------------------------------------------------------------------
# B5 — condition onset
# --------------------------------------------------------------------------


class TestConditionOnset:
    def test_onset_from_attr(self):
        rec = _record(_entity("condition", "Crohn's Disease", status="active",
                              onset_date="2015-06-01"))
        assert rec["fhir_resource"]["onsetDateTime"] == "2015-06-01"

    def test_onset_since_attr(self):
        rec = _record(_entity("condition", "Asthma", status="active", since="2010"))
        assert rec["fhir_resource"]["onsetDateTime"] == "2010"


# --------------------------------------------------------------------------
# B7 — diagnostic report performer
# --------------------------------------------------------------------------


class TestDiagnosticReportPerformer:
    def test_performer_from_lab_attr(self):
        rec = _record(_entity("imaging_result", "CT Chest", findings="No acute findings",
                              lab="Radiology Associates"))
        res = rec["fhir_resource"]
        assert res["performer"][0]["display"] == "Radiology Associates"


# --------------------------------------------------------------------------
# B8 — lab panel orders (no value) -> ServiceRequest
# --------------------------------------------------------------------------


class TestLabPanelOrder:
    def test_panel_without_value_becomes_service_request(self):
        rec = _record(_entity("lab_result", "CBC"))
        assert rec["record_type"] == "service_request"
        assert rec["fhir_resource_type"] == "ServiceRequest"
        res = rec["fhir_resource"]
        assert res["resourceType"] == "ServiceRequest"
        assert res["intent"] == "order"
        assert res["code"]["text"] == "CBC"

    def test_panel_with_value_stays_observation(self):
        rec = _record(_entity("lab_result", "CBC", value="5.2", unit="x10E3/uL"))
        assert rec["record_type"] == "observation"
        assert rec["fhir_resource_type"] == "Observation"

    def test_non_panel_valueless_lab_stays_observation(self):
        rec = _record(_entity("lab_result", "Glucose"))
        assert rec["fhir_resource_type"] == "Observation"


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


class TestResolveDocumentProvider:
    def test_from_provider_entity_text(self):
        ents = [_entity("provider", "Dr. Jane Smith"), _entity("condition", "Asthma")]
        assert resolve_document_provider(ents) == "Dr. Jane Smith"

    def test_prefers_name_attribute(self):
        ents = [_entity("provider", "attending physician", name="Dr. Gomez")]
        assert resolve_document_provider(ents) == "Dr. Gomez"

    def test_none_when_no_provider_entity(self):
        assert resolve_document_provider([_entity("condition", "Asthma")]) is None

    def test_skips_blank_provider(self):
        ents = [_entity("provider", "   "), _entity("provider", "Dr. Real")]
        assert resolve_document_provider(ents) == "Dr. Real"


class TestBackwardCompat:
    def test_minimal_signature_unchanged(self):
        rec = entity_to_health_record_dict(
            _entity("condition", "Type 2 Diabetes", status="active"),
            uuid4(), uuid4(), uuid4(),
        )
        assert rec is not None
        assert rec["content_hash"]

    def test_non_storable_still_none(self):
        assert _record(_entity("dosage", "10mg")) is None
