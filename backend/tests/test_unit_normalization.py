"""Exhaustive TDD for the read-time unit normalizer (BACKEND-TODO #5).

``normalize_value(code_value, value, unit) -> (normalized_value, canonical_unit)``
converts a numeric observation reading to the curated canonical unit for its LOINC
code so cross-source series are consistent. It is a PURE function:

Rules under test (define expected output first, then assert):
  * known code + known source unit  -> converted, rounded to 2 dp, canonical unit
  * known code + already-canonical unit (case/space-insensitive) -> unchanged value
  * known code + unknown source unit -> value unchanged, canonical unit returned
  * unknown code -> value + unit returned verbatim (NEVER guess a conversion)
  * unit is None -> value returned, canonical unit returned for known codes else None
  * every conversion direction we curate is covered, both ways where reversible.
"""

from __future__ import annotations

import pytest

from app.services.utils.unit_normalization import normalize_value


# --------------------------------------------------------------------------- #
# HbA1c (4548-4): canonical "%". mmol/mol -> % via NGSP: % = (mmol/mol / 10.929) + 2.15
# --------------------------------------------------------------------------- #


def test_a1c_already_percent_is_noop():
    assert normalize_value("4548-4", 6.8, "%") == (6.8, "%")


def test_a1c_mmol_per_mol_converts_to_percent():
    # 53 mmol/mol -> (53 / 10.929) + 2.15 = 7.0001... -> 7.00
    value, unit = normalize_value("4548-4", 53.0, "mmol/mol")
    assert unit == "%"
    assert value == pytest.approx(7.0, abs=0.01)


def test_a1c_percent_case_and_space_insensitive_noop():
    assert normalize_value("4548-4", 6.8, " % ") == (6.8, "%")


def test_a1c_unknown_unit_keeps_value_returns_canonical():
    assert normalize_value("4548-4", 6.8, "weird-unit") == (6.8, "%")


# --------------------------------------------------------------------------- #
# Cholesterol family -> mg/dL (mmol/L * 38.67)
#   LDL 13457-7, HDL 2085-9, total chol 2093-3
# --------------------------------------------------------------------------- #


def test_ldl_mmol_per_l_to_mg_dl():
    # 4.24 mmol/L * 38.67 = 163.96 -> ~164 mg/dL
    value, unit = normalize_value("13457-7", 4.24, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(163.96, abs=0.5)


def test_ldl_already_mg_dl_noop():
    assert normalize_value("13457-7", 120.0, "mg/dL") == (120.0, "mg/dL")


def test_hdl_mmol_per_l_to_mg_dl():
    value, unit = normalize_value("2085-9", 1.55, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(1.55 * 38.67, abs=0.01)


def test_total_cholesterol_mmol_per_l_to_mg_dl():
    value, unit = normalize_value("2093-3", 5.2, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(5.2 * 38.67, abs=0.01)


# --------------------------------------------------------------------------- #
# Triglycerides 2571-8 -> mg/dL (mmol/L * 88.57)
# --------------------------------------------------------------------------- #


def test_triglycerides_mmol_per_l_to_mg_dl():
    value, unit = normalize_value("2571-8", 1.7, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(1.7 * 88.57, abs=0.01)


def test_triglycerides_already_mg_dl_noop():
    assert normalize_value("2571-8", 150.0, "mg/dL") == (150.0, "mg/dL")


# --------------------------------------------------------------------------- #
# Glucose 2339-0 / 2345-7 -> mg/dL (mmol/L * 18.0182)
# --------------------------------------------------------------------------- #


def test_glucose_2339_mmol_per_l_to_mg_dl():
    value, unit = normalize_value("2339-0", 5.5, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(5.5 * 18.0182, abs=0.01)


def test_glucose_2345_mmol_per_l_to_mg_dl():
    value, unit = normalize_value("2345-7", 7.0, "mmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(7.0 * 18.0182, abs=0.01)


def test_glucose_already_mg_dl_noop():
    assert normalize_value("2339-0", 99.0, "mg/dL") == (99.0, "mg/dL")


# --------------------------------------------------------------------------- #
# Creatinine 2160-0 -> mg/dL (µmol/L / 88.42)
# --------------------------------------------------------------------------- #


def test_creatinine_umol_per_l_to_mg_dl():
    # 88.42 µmol/L / 88.42 = 1.0 mg/dL
    value, unit = normalize_value("2160-0", 88.42, "umol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(1.0, abs=0.01)


def test_creatinine_micro_sign_variant():
    value, unit = normalize_value("2160-0", 88.42, "µmol/L")
    assert unit == "mg/dL"
    assert value == pytest.approx(1.0, abs=0.01)


def test_creatinine_already_mg_dl_noop():
    assert normalize_value("2160-0", 0.9, "mg/dL") == (0.9, "mg/dL")


# --------------------------------------------------------------------------- #
# Unknown code / None unit / guards
# --------------------------------------------------------------------------- #


def test_unknown_code_returns_value_and_unit_verbatim():
    assert normalize_value("not-a-loinc", 128.0, "mmHg") == (128.0, "mmHg")


def test_unknown_code_none_unit_returns_none_unit():
    assert normalize_value("not-a-loinc", 128.0, None) == (128.0, None)


def test_known_code_none_unit_returns_canonical_unit_value_unchanged():
    # No source unit to convert from -> trust the value, label it canonical.
    assert normalize_value("4548-4", 6.8, None) == (6.8, "%")


def test_none_code_is_noop():
    assert normalize_value(None, 5.0, "mg/dL") == (5.0, "mg/dL")


def test_rounding_to_two_decimals():
    # LDL 3.3333 mmol/L * 38.67 = 128.90... -> rounded 2dp
    value, _ = normalize_value("13457-7", 3.3333, "mmol/L")
    assert value == round(3.3333 * 38.67, 2)
