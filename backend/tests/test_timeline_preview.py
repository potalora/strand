from __future__ import annotations

from app.schemas.timeline import TimelineGauge
from app.services.timeline_preview import build_timeline_preview


def _lab(value, unit, interp=None, low=None, high=None, value_string=None):
    r = {
        "resourceType": "Observation",
        "category": [{"coding": [{"code": "laboratory"}]}],
    }
    if value_string is not None:
        r["valueString"] = value_string
    else:
        r["valueQuantity"] = {"value": value, "unit": unit}
    if interp:
        r["interpretation"] = [{"coding": [{"code": interp}]}]
    if low is not None and high is not None:
        r["referenceRange"] = [{"low": {"value": low}, "high": {"value": high}}]
    return r


# ---- observation / lab ----
def test_lab_with_range_and_low_flag():
    p = build_timeline_preview(_lab(17, "ng/mL", "L", 30, 100), "observation")
    assert p is not None
    assert p.value == "17"
    assert p.unit == "ng/mL"
    assert p.flag == "LOW"
    assert p.emphasis == "notable"
    assert p.gauge == TimelineGauge(value=17, low=30, high=100)


def test_lab_normal_no_flag_no_gauge_when_range_absent():
    p = build_timeline_preview(_lab(5.0, "mIU/L", "N"), "observation")
    assert p.value == "5"
    assert p.unit == "mIU/L"
    assert p.flag is None
    assert p.emphasis == "normal"
    assert p.gauge is None


def test_lab_value_string_yields_no_gauge():
    p = build_timeline_preview(_lab(None, None, value_string="Detected"), "observation")
    assert p.value == "Detected"
    assert p.gauge is None


# ---- observation / vital ----
def test_vital_blood_pressure_composes_systolic_over_diastolic():
    r = {
        "resourceType": "Observation",
        "category": [{"coding": [{"code": "vital-signs"}]}],
        "component": [
            {"code": {"coding": [{"code": "8480-6"}]}, "valueQuantity": {"value": 120, "unit": "mmHg"}},
            {"code": {"coding": [{"code": "8462-4"}]}, "valueQuantity": {"value": 80, "unit": "mmHg"}},
        ],
    }
    p = build_timeline_preview(r, "observation")
    assert p.value == "120/80"
    assert p.unit == "mmHg"


def test_vital_simple_value():
    r = {
        "resourceType": "Observation",
        "category": [{"coding": [{"code": "vital-signs"}]}],
        "valueQuantity": {"value": 98.6, "unit": "degF"},
    }
    p = build_timeline_preview(r, "observation")
    assert p.value == "98.6"
    assert p.unit == "degF"


# ---- observation / social ----
def test_social_history_value_text():
    r = {
        "resourceType": "Observation",
        "category": [{"coding": [{"code": "social-history"}]}],
        "valueCodeableConcept": {"text": "Former smoker"},
    }
    p = build_timeline_preview(r, "observation")
    assert p.value == "Former smoker"
    assert p.gauge is None


# ---- condition ----
def test_condition_active_with_onset_facet():
    r = {
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "onsetDateTime": "2024-03-01",
    }
    p = build_timeline_preview(r, "condition")
    assert p.flag == "ACTIVE"
    assert p.emphasis == "normal"
    assert "onset 2024" in p.facets


def test_condition_resolved_is_muted():
    r = {"resourceType": "Condition", "clinicalStatus": {"coding": [{"code": "resolved"}]}}
    p = build_timeline_preview(r, "condition")
    assert p.flag == "RESOLVED"
    assert p.emphasis == "muted"


def test_condition_negated_via_verification_status():
    r = {
        "resourceType": "Condition",
        "clinicalStatus": {"coding": [{"code": "inactive"}]},
        "verificationStatus": {"coding": [{"code": "refuted"}]},
    }
    p = build_timeline_preview(r, "condition")
    assert p.flag == "NEGATED"
    assert p.emphasis == "muted"


# ---- medication ----
def test_medication_dose_route_frequency_status():
    r = {
        "resourceType": "MedicationRequest",
        "status": "active",
        "dosageInstruction": [
            {
                "route": {"text": "oral"},
                "timing": {"code": {"text": "1×/day"}},
                "doseAndRate": [{"doseQuantity": {"value": 20, "unit": "mg"}}],
            }
        ],
    }
    p = build_timeline_preview(r, "medication")
    assert p.value == "20 mg"
    assert p.flag == "ACTIVE"
    assert p.facets == ["oral", "1×/day"]


def test_medication_stopped_is_muted():
    r = {"resourceType": "MedicationRequest", "status": "stopped"}
    p = build_timeline_preview(r, "medication")
    assert p.flag == "STOPPED"
    assert p.emphasis == "muted"


# ---- allergy ----
def test_allergy_high_criticality_with_reaction():
    r = {
        "resourceType": "AllergyIntolerance",
        "criticality": "high",
        "reaction": [{"manifestation": [{"text": "hives"}]}],
    }
    p = build_timeline_preview(r, "allergy")
    assert p.flag == "HIGH"
    assert p.emphasis == "notable"
    assert "hives" in p.facets


# ---- procedure ----
def test_procedure_status_and_body_site():
    r = {
        "resourceType": "Procedure",
        "status": "completed",
        "bodySite": [{"text": "left knee"}],
    }
    p = build_timeline_preview(r, "procedure")
    assert p.flag == "COMPLETED"
    assert "left knee" in p.facets


# ---- immunization ----
def test_immunization_dose_number_facet():
    r = {
        "resourceType": "Immunization",
        "status": "completed",
        "protocolApplied": [{"doseNumberPositiveInt": 2}],
    }
    p = build_timeline_preview(r, "immunization")
    assert p.flag == "COMPLETED"
    assert "dose 2" in p.facets


# ---- encounter ----
def test_encounter_class_label_and_reason():
    r = {
        "resourceType": "Encounter",
        "class": {"code": "AMB"},
        "reasonCode": [{"text": "annual physical"}],
    }
    p = build_timeline_preview(r, "encounter")
    assert p.flag == "Ambulatory"
    assert "annual physical" in p.facets


# ---- fallbacks ----
def test_document_returns_none():
    assert build_timeline_preview({"resourceType": "DocumentReference"}, "document") is None


def test_non_dict_returns_none():
    assert build_timeline_preview(None, "observation") is None


def test_empty_observation_returns_none():
    assert build_timeline_preview({"resourceType": "Observation"}, "observation") is None
