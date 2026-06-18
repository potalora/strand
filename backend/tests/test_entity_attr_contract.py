"""Cross-agent contract guards: the attribute vocabulary the extraction prompt
(Agent A) instructs LangExtract to emit must match what entity_to_fhir
(Agent B) consumes — structured dosage (B4), condition onset (B5), and discrete
lab value/unit/range/interpretation (B3).
"""
from __future__ import annotations

from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict


def _resource(entity: ExtractedEntity) -> dict:
    rec = entity_to_health_record_dict(entity, uuid4(), uuid4())
    assert rec is not None
    return rec["fhir_resource"]


def test_med_dose_route_frequency_attrs_produce_dosage_instruction():
    e = ExtractedEntity(
        "medication", "Metformin",
        {"medication_group": "Metformin", "dose": "500mg", "route": "PO", "frequency": "BID"},
    )
    res = _resource(e)
    di = res.get("dosageInstruction")
    assert di and isinstance(di, list)
    assert di[0].get("route", {}).get("text") == "PO"


def test_med_sig_attr_produces_dosage_instruction():
    e = ExtractedEntity("medication", "Omeprazole", {"dosage": "40mg PO daily"})
    res = _resource(e)
    assert res.get("dosageInstruction")


def test_condition_onset_attr_produces_onset_datetime():
    e = ExtractedEntity("condition", "Type 2 diabetes", {"status": "active", "onset_date": "2015"})
    res = _resource(e)
    assert res.get("onsetDateTime") == "2015"


def test_lab_discrete_attrs_produce_value_unit_range_interpretation():
    e = ExtractedEntity(
        "lab_result", "Glucose",
        {"test": "Glucose", "value": "95", "unit": "mg/dL",
         "ref_low": "70", "ref_high": "99", "interpretation": "H"},
    )
    res = _resource(e)
    assert res["valueQuantity"]["value"] == 95.0
    assert res["valueQuantity"]["unit"] == "mg/dL"
    assert res["referenceRange"][0]["low"]["value"] == 70.0
    assert res["referenceRange"][0]["high"]["value"] == 99.0
    assert res["interpretation"][0]["coding"][0]["code"] == "H"
