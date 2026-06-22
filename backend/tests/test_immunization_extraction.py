"""A7 — vaccines extracted from unstructured notes must become Immunization
records (FHIR R4B ``Immunization``), NOT MedicationRequest, and each dose must
keep its own administration date.

Real-data defect: an After-Visit-Summary immunization summary listing several
COVID-19 vaccine doses (each with an administration date) was extracted as
``medication`` records (wrong record_type) with ALL dose dates dropped.

These are pure-function tests over the entity -> FHIR path (no DB, no Gemini):
the entity shapes mirror what LangExtract produces — both the correct
``immunization`` class and the observed ``medication`` mislabel — and assert the
record_type, the FHIR Immunization shape (vaccineCode / occurrenceDateTime /
status), and per-dose date preservation. A parallel set of conservative cases
pins that real oral/systemic medications are NOT reclassified as vaccines.
"""
from __future__ import annotations

from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict

CVX_SYSTEM = "http://hl7.org/fhir/sid/cvx"


def _entity(entity_class: str, text: str, **attrs) -> ExtractedEntity:
    return ExtractedEntity(
        entity_class=entity_class,
        text=text,
        attributes=dict(attrs),
        start_pos=0,
        end_pos=len(text),
        confidence=0.9,
    )


def _record(entity: ExtractedEntity, **kw):
    return entity_to_health_record_dict(entity, uuid4(), uuid4(), uuid4(), **kw)


# --------------------------------------------------------------------------
# Core defect — a vaccine becomes an Immunization with its date preserved
# --------------------------------------------------------------------------


class TestImmunizationClass:
    """When extraction emits the ``immunization`` class directly."""

    def test_record_type_and_resource(self):
        rec = _record(
            _entity("immunization", "COVID-19 Vaccine", vaccine="COVID-19", date="12/15/2020")
        )
        assert rec["record_type"] == "immunization"
        assert rec["fhir_resource_type"] == "Immunization"
        res = rec["fhir_resource"]
        assert res["resourceType"] == "Immunization"
        assert res["status"] == "completed"
        assert "COVID-19" in res["vaccineCode"]["text"]

    def test_occurrence_and_effective_date_preserved(self):
        rec = _record(_entity("immunization", "COVID-19 Vaccine", date="12/15/2020"))
        # effective_date (timeline) preserved
        assert rec["effective_date"] is not None
        assert rec["effective_date"].year == 2020
        assert rec["effective_date"].month == 12
        assert rec["effective_date"].day == 15
        # occurrenceDateTime carried on the resource
        assert rec["fhir_resource"]["occurrenceDateTime"].startswith("2020-12-15")

    def test_cvx_coding_when_present(self):
        rec = _record(
            _entity("immunization", "COVID-19 Vaccine", cvx="208", date="12/15/2020")
        )
        coding = rec["fhir_resource"]["vaccineCode"]["coding"][0]
        assert coding["system"] == CVX_SYSTEM
        assert coding["code"] == "208"

    def test_refused_status_maps_to_not_done(self):
        rec = _record(_entity("immunization", "Influenza Vaccine", status="refused"))
        assert rec["fhir_resource"]["status"] == "not-done"


class TestMislabeledMedicationVaccine:
    """The observed defect: vaccines come back labeled ``medication``."""

    def test_covid_medication_reclassified_to_immunization(self):
        rec = _record(_entity("medication", "COVID-19 Vaccine (Pfizer)", date="01/05/2021"))
        assert rec["record_type"] == "immunization"
        assert rec["fhir_resource_type"] == "Immunization"
        assert rec["fhir_resource"]["resourceType"] == "Immunization"

    def test_date_preserved_on_reclassified_vaccine(self):
        rec = _record(_entity("medication", "COVID-19 Vaccine (Pfizer)", date="01/05/2021"))
        assert rec["effective_date"] is not None
        assert (rec["effective_date"].year, rec["effective_date"].month,
                rec["effective_date"].day) == (2021, 1, 5)
        assert rec["fhir_resource"]["occurrenceDateTime"].startswith("2021-01-05")

    def test_procedure_class_vaccination_reclassified(self):
        rec = _record(_entity("procedure", "Influenza vaccination", date="10/15/2021"))
        assert rec["record_type"] == "immunization"


class TestMultiDoseImmunizationSummary:
    """An AVS immunization summary lists multiple dated doses — each dose must
    become a distinct Immunization record carrying its OWN date (the core A7
    regression: all dose dates were previously collapsed/dropped)."""

    DOSES = [
        ("COVID-19 Vaccine (Pfizer)", "12/15/2020", (2020, 12, 15)),
        ("COVID-19 Vaccine (Pfizer)", "01/05/2021", (2021, 1, 5)),
        ("COVID-19 Booster (Moderna)", "10/01/2021", (2021, 10, 1)),
        ("Influenza Vaccine", "10/15/2021", (2021, 10, 15)),
    ]

    def test_each_dose_is_a_dated_immunization(self):
        records = [
            _record(_entity("medication", name, date=date)) for name, date, _ in self.DOSES
        ]
        # all reclassified to immunization
        assert all(r["record_type"] == "immunization" for r in records)
        # every dose keeps its distinct administration date
        got = [
            (r["effective_date"].year, r["effective_date"].month, r["effective_date"].day)
            for r in records
        ]
        assert got == [expected for _, _, expected in self.DOSES]
        # no date was dropped
        assert all(r["fhir_resource"].get("occurrenceDateTime") for r in records)


class TestVaccineKeywordGuard:
    """Robust detection across common vaccine names/brands (medication-class)."""

    VACCINES = [
        "Tdap",
        "Td booster",
        "MMR vaccine",
        "Pneumococcal (Prevnar 13)",
        "Pneumovax 23",
        "Shingrix",
        "Zoster vaccine",
        "Hepatitis B vaccine",
        "Hepatitis A",
        "Varicella vaccine",
        "HPV (Gardasil)",
        "SARS-CoV-2 mRNA vaccine",
        "Comirnaty",
        "Spikevax",
    ]

    def test_known_vaccines_become_immunizations(self):
        for name in self.VACCINES:
            rec = _record(_entity("medication", name))
            assert rec["record_type"] == "immunization", f"{name!r} not detected as vaccine"
            assert rec["fhir_resource"]["resourceType"] == "Immunization"


class TestConservativeNoFalsePositives:
    """Real oral/systemic medications must NOT be reclassified as vaccines."""

    MEDICATIONS = [
        "Metformin 500mg",
        "Lisinopril 10mg",
        "Fluconazole 150mg",      # starts with 'flu' but is an antifungal
        "Fluoxetine 20mg",        # starts with 'flu'
        "Oseltamivir 75mg",       # treats influenza, but is an antiviral drug
        "Atorvastatin 40mg",
        "Amoxicillin 500mg",
        "Toradol",                # contains 'td' substring, not the Td vaccine
    ]

    def test_medications_stay_medications(self):
        for name in self.MEDICATIONS:
            rec = _record(_entity("medication", name))
            assert rec["record_type"] == "medication", f"{name!r} wrongly reclassified"
            assert rec["fhir_resource_type"] == "MedicationRequest"

    def test_oseltamivir_indication_influenza_not_reclassified(self):
        # An antiviral whose *indication* mentions influenza must stay a med;
        # the indication attribute is not scanned for vaccine keywords.
        rec = _record(_entity("medication", "Oseltamivir", indication="influenza prophylaxis"))
        assert rec["record_type"] == "medication"

    def test_condition_named_like_a_vaccine_disease_unaffected(self):
        # 'Hepatitis B' as a CONDITION stays a condition (guard only fires on
        # medication/procedure entities).
        rec = _record(_entity("condition", "Hepatitis B", status="active"))
        assert rec["record_type"] == "condition"
