"""Tests for the bundled, offline terminology lookup.

These pin the public API (unchanged) and the behavior of the new
bundled-index backend: ICD-10-CM conditions, RxNorm medications, a curated
LOINC lab subset, local procedure markers, and a local supplement overlay.
All lookups are offline (no network) — they read the committed
``terminology_data/*.json.gz`` indexes.
"""
from __future__ import annotations

import pytest

from app.services.extraction import terminology as t


class TestNormalizeTerm:
    def test_lowercases_and_strips(self):
        assert t.normalize_term("  Type 2 Diabetes  ") == "type 2 diabetes"

    def test_collapses_internal_whitespace(self):
        assert t.normalize_term("Type 2   Diabetes") == "type 2 diabetes"

    def test_removes_possessive_apostrophe(self):
        assert t.normalize_term("Crohn's Disease") == "crohns disease"

    def test_removes_parenthetical(self):
        assert t.normalize_term("Hemoglobin A1c (HbA1c)") == "hemoglobin a1c"

    def test_strips_dose_token_attached(self):
        assert t.normalize_term("Metformin 500mg") == "metformin"

    def test_strips_dose_token_spaced(self):
        assert t.normalize_term("Lisinopril 10 mg") == "lisinopril"

    def test_strips_punctuation_keeps_words(self):
        assert t.normalize_term("Glucose, Serum") == "glucose serum"

    def test_none_returns_empty(self):
        assert t.normalize_term(None) == ""

    def test_empty_returns_empty(self):
        assert t.normalize_term("   ") == ""


class TestCoding:
    def test_as_coding_shape(self):
        c = t.Coding(system=t.ICD10_SYSTEM, code="E11.9", display="Type 2 diabetes mellitus")
        assert c.as_coding() == {
            "system": t.ICD10_SYSTEM,
            "code": "E11.9",
            "display": "Type 2 diabetes mellitus",
        }


class TestSystemConstants:
    def test_public_system_uris_unchanged(self):
        assert t.ICD10_SYSTEM == "http://hl7.org/fhir/sid/icd-10-cm"
        assert t.RXNORM_SYSTEM == "http://www.nlm.nih.gov/research/umls/rxnorm"
        assert t.LOINC_SYSTEM == "http://loinc.org"
        # Kept for backward compatibility even though SNOMED is no longer emitted.
        assert t.SNOMED_SYSTEM == "http://snomed.info/sct"
        # New local marker systems.
        assert t.PROCEDURE_SYSTEM.endswith("/procedure")
        assert t.SUPPLEMENT_SYSTEM.endswith("/supplement")


class TestConditionLookup:
    @pytest.mark.parametrize(
        "text,code",
        [
            ("Type 2 Diabetes", "E11.9"),
            ("type 2 diabetes mellitus", "E11.9"),
            ("T2DM", "E11.9"),
            ("Diabetes", "E11.9"),
            ("Hypertension", "I10"),
            ("HTN", "I10"),
            ("High Blood Pressure", "I10"),
            ("Hyperlipidemia", "E78.5"),
            ("Hypothyroidism", "E03.9"),
            ("GERD", "K21.9"),
            ("Asthma", "J45.909"),
            ("Crohn's Disease", "K50.90"),
            ("crohns disease", "K50.90"),
            ("Ulcerative Colitis", "K51.90"),
            ("Hodgkin Lymphoma", "C81.90"),
            ("Atrial Fibrillation", "I48.91"),
            ("Vitamin D Deficiency", "E55.9"),
            ("Irritable Bowel Syndrome", "K58.9"),
            ("IBS", "K58.9"),
        ],
    )
    def test_known_conditions(self, text, code):
        c = t.lookup_condition(text)
        assert c is not None, f"expected a coding for {text!r}"
        assert c.system == t.ICD10_SYSTEM
        assert c.code == code
        assert c.display  # non-empty human label

    def test_formal_icd10_description_resolves_from_full_index(self):
        # A formal label that matches an official ICD-10-CM description (not in
        # the curated colloquial overlay) should still resolve via the bundled
        # full index — this is the breadth the old hand-curated map lacked.
        c = t.lookup_condition("Sepsis, unspecified organism")
        assert c is not None and c.system == t.ICD10_SYSTEM and c.code == "A41.9"

    def test_unknown_condition_returns_none(self):
        assert t.lookup_condition("zzz unknown condition") is None

    def test_blank_returns_none(self):
        assert t.lookup_condition("") is None
        assert t.lookup_condition(None) is None


class TestMedicationLookup:
    @pytest.mark.parametrize(
        "text,code",
        [
            ("Metformin", "6809"),
            ("Metformin 500mg", "6809"),
            ("metformin 500 mg tablet", "6809"),
            ("Lisinopril", "29046"),
            ("Levothyroxine", "10582"),
            ("Omeprazole", "7646"),
            ("Rifaximin", "35619"),
            ("Naltrexone", "7243"),
            ("Albuterol", "435"),
        ],
    )
    def test_known_medications(self, text, code):
        c = t.lookup_medication(text)
        assert c is not None, f"expected a coding for {text!r}"
        assert c.system == t.RXNORM_SYSTEM
        assert c.code == code

    @pytest.mark.parametrize(
        "text",
        [
            "metformin", "doxycycline", "low-dose naltrexone", "naltrexone",
            "escitalopram", "pantoprazole", "rifaximin", "montelukast",
            "rosuvastatin", "duloxetine", "hydrocortisone", "lactulose",
            "azelastine", "ketoconazole", "cyanocobalamin", "vitamin B12",
        ],
    )
    def test_user_drug_list_all_resolve_to_rxnorm(self, text):
        """Every medication on the user's real list resolves to an RxNorm code."""
        c = t.lookup_medication(text)
        assert c is not None, f"user drug {text!r} did not resolve"
        assert c.system == t.RXNORM_SYSTEM
        assert c.code.isdigit()

    def test_brand_name_resolves_via_rxnorm(self):
        # Brand names are covered natively by the RxNorm brand (BN) concepts.
        assert t.lookup_medication("Synthroid") is not None
        assert t.lookup_medication("Lasix") is not None

    def test_unknown_medication_returns_none(self):
        assert t.lookup_medication("Go") is None
        assert t.lookup_medication("nonexistent-drug-xyzzy") is None


class TestB12Synonyms:
    """Bare/abbreviated B12 forms resolve to RxNorm 11248 (vitamin B12).

    The user's records use bare forms ("b12", "b12 injection", "daily b12").
    All must map to the SAME RxNorm coding as "vitamin B12"/"cyanocobalamin".
    """

    @pytest.mark.parametrize("text", ["B12", "b12", "vitamin B12", "b-12", "b 12"])
    def test_b12_variants_resolve_to_rxnorm_vitamin_b12(self, text):
        c = t.lookup_medication(text)
        assert c is not None, f"{text!r} did not resolve"
        assert c.system == t.RXNORM_SYSTEM
        assert c.code == "11248"

    def test_b12_matches_existing_canonical_forms(self):
        canonical = t.lookup_medication("vitamin B12")
        assert canonical is not None and canonical.code == "11248"
        assert t.lookup_medication("B12") == canonical
        assert t.lookup_medication("cyanocobalamin") == canonical

    @pytest.mark.parametrize("text", ["b12 injection", "daily b12", "b12 1000 mcg"])
    def test_b12_in_phrases_resolves(self, text):
        c = t.lookup_medication(text)
        assert c is not None, f"{text!r} did not resolve"
        assert c.system == t.RXNORM_SYSTEM
        assert c.code == "11248"


class TestSupplementOverlay:
    @pytest.mark.parametrize(
        "text",
        ["Candibactin-AR", "Candibactin-BR", "Akkermansia", "Digest Gold",
         "FODZYME", "probiotics"],
    )
    def test_supplements_get_local_marker(self, text):
        c = t.lookup_medication(text)
        assert c is not None, f"supplement {text!r} did not resolve"
        assert c.system == t.SUPPLEMENT_SYSTEM
        assert c.display  # human-readable supplement label


class TestLabLookup:
    @pytest.mark.parametrize(
        "text,code",
        [
            ("Hemoglobin A1c", "4548-4"),
            ("HbA1c", "4548-4"),
            ("A1c", "4548-4"),
            ("Glucose", "2345-7"),
            ("Creatinine", "2160-0"),
            ("Potassium", "2823-3"),
            ("Sodium", "2951-2"),
            ("TSH", "3016-3"),
            ("Free T4", "3024-7"),
            ("LDL Cholesterol", "2089-1"),
            ("HDL", "2085-9"),
            ("Triglycerides", "2571-8"),
            ("Hemoglobin", "718-7"),
            ("Hematocrit", "4544-3"),
            ("Platelets", "777-3"),
            ("Vitamin B12", "2132-9"),
            ("Ferritin", "2276-4"),
            ("ALT", "1742-6"),
            ("eGFR", "33914-3"),
            ("CRP", "1988-5"),
            ("ESR", "4537-7"),
        ],
    )
    def test_known_labs(self, text, code):
        c = t.lookup_lab(text)
        assert c is not None, f"expected a coding for {text!r}"
        assert c.system == t.LOINC_SYSTEM
        assert c.code == code

    def test_unknown_lab_returns_none(self):
        assert t.lookup_lab("mystery analyte") is None


class TestProcedureLookup:
    """Procedures resolve to local category markers (SNOMED/CPT dropped)."""

    @pytest.mark.parametrize(
        "text,code",
        [
            ("Colonoscopy", "colonoscopy"),
            ("Echocardiogram", "echocardiogram"),
            ("EKG", "ecg"),
            ("ECG", "ecg"),
            ("Electrocardiogram", "ecg"),
            ("Mammogram", "mammogram"),
            ("Chest X-ray", "chest_xray"),
            ("Cholecystectomy", "cholecystectomy"),
            ("Appendectomy", "appendectomy"),
        ],
    )
    def test_known_procedures(self, text, code):
        c = t.lookup_procedure(text)
        assert c is not None, f"expected a coding for {text!r}"
        assert c.system == t.PROCEDURE_SYSTEM
        assert c.code == code
        assert c.display

    def test_procedures_no_longer_emit_snomed(self):
        c = t.lookup_procedure("Colonoscopy")
        assert c is not None and c.system != t.SNOMED_SYSTEM

    def test_unknown_procedure_returns_none(self):
        assert t.lookup_procedure("imaginary surgery") is None


class TestDispatch:
    def test_lookup_dispatches_by_category(self):
        assert t.lookup("condition", "Hypertension").code == "I10"
        assert t.lookup("medication", "Metformin").code == "6809"
        assert t.lookup("lab", "Glucose").code == "2345-7"
        assert t.lookup("observation", "Glucose").code == "2345-7"
        assert t.lookup("procedure", "Colonoscopy").code == "colonoscopy"

    def test_lookup_unknown_category_returns_none(self):
        assert t.lookup("widget", "Hypertension") is None


class TestLazyIndexAttributes:
    """The module exposes index dicts (lazily materialized) for back-compat."""

    def test_indexes_are_populated_and_cached(self):
        assert len(t.CONDITION_INDEX) > 1000   # full ICD-10-CM index
        assert len(t.MEDICATION_INDEX) > 1000  # full RxNorm index + overlays
        assert len(t.LAB_INDEX) > 50           # curated LOINC subset
        assert len(t.PROCEDURE_INDEX) >= 7     # curated procedures
        # Cached: a second access returns the very same object.
        assert t.CONDITION_INDEX is t.CONDITION_INDEX

    def test_index_values_are_coding_objects(self):
        coding = t.MEDICATION_INDEX[t.normalize_term("metformin")]
        assert isinstance(coding, t.Coding)
        assert coding.code == "6809"

    def test_unknown_module_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = t.NOPE_INDEX


class TestGracefulDegradation:
    def test_missing_index_file_returns_empty(self, monkeypatch, tmp_path):
        """A missing data file degrades to an empty index, not a crash."""
        monkeypatch.setattr(t, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(t, "_INDEX_CACHE", {})
        assert t.lookup_condition("Hypertension") is None
        assert t._load_index("condition") == {}
