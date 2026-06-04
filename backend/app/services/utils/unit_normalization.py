"""Read-time unit normalization for observation values (BACKEND-TODO #5).

A curated, per-LOINC table that converts a numeric reading to a single canonical
unit so cross-source time-series are consistent (e.g. an HbA1c reported in
mmol/mol by one lab and in % by another both render on one sparkline).

Design constraints (descriptive, not normative — CLAUDE.md Absolute Rule #1):
  * Conversions are factual unit math only — no clinical interpretation.
  * NEVER guess a conversion we do not explicitly curate. If the code is unknown,
    or the source unit is not one we know how to convert from, the value is
    returned UNCHANGED.
  * Original units are preserved in ``fhir_resource``; this runs at read time.

``normalize_value`` is pure and exhaustively unit-tested in
``tests/test_unit_normalization.py``.
"""

from __future__ import annotations

from typing import Callable

# Decimal places for the normalized value. Enough to preserve mg/dL-scale
# precision without exposing float noise.
_ROUND_DP = 2


def _canon(unit: str | None) -> str:
    """Canonicalize a unit string for case/space/sign-insensitive comparison."""
    if not unit:
        return ""
    return unit.strip().lower().replace("µ", "u").replace("μ", "u")


# Conversion factors / functions keyed by (canonical source unit). Each maps a
# value in the source unit to the canonical unit. Only directions we are sure of
# are listed; anything else falls through to a no-op.
#
# Each entry is a dict mapping a canonicalized SOURCE unit to a callable
# (value -> converted value). The canonical unit itself always maps to identity.

# Cholesterol family: mmol/L -> mg/dL via the molar mass of cholesterol.
_CHOL_MMOL_TO_MGDL = 38.67
# Triglycerides: mmol/L -> mg/dL.
_TRIG_MMOL_TO_MGDL = 88.57
# Glucose: mmol/L -> mg/dL.
_GLUCOSE_MMOL_TO_MGDL = 18.0182
# Creatinine: µmol/L -> mg/dL.
_CREAT_UMOL_DIVISOR = 88.42
# HbA1c: mmol/mol (IFCC) -> % (NGSP).
_A1C_IFCC_DIVISOR = 10.929
_A1C_NGSP_OFFSET = 2.15


def _mgdl_table(mmol_factor: float) -> dict[str, Callable[[float], float]]:
    """Build a mg/dL canonical table converting from mmol/L by ``mmol_factor``."""
    return {
        "mg/dl": lambda v: v,
        "mmol/l": lambda v: v * mmol_factor,
    }


# code_value -> (canonical_unit, {source_unit_canon: converter})
_CODE_TABLE: dict[str, tuple[str, dict[str, Callable[[float], float]]]] = {
    # HbA1c
    "4548-4": (
        "%",
        {
            "%": lambda v: v,
            "mmol/mol": lambda v: (v / _A1C_IFCC_DIVISOR) + _A1C_NGSP_OFFSET,
        },
    ),
    # LDL cholesterol
    "13457-7": ("mg/dL", _mgdl_table(_CHOL_MMOL_TO_MGDL)),
    # HDL cholesterol
    "2085-9": ("mg/dL", _mgdl_table(_CHOL_MMOL_TO_MGDL)),
    # Total cholesterol
    "2093-3": ("mg/dL", _mgdl_table(_CHOL_MMOL_TO_MGDL)),
    # Triglycerides
    "2571-8": ("mg/dL", _mgdl_table(_TRIG_MMOL_TO_MGDL)),
    # Glucose (two common LOINC codes)
    "2339-0": ("mg/dL", _mgdl_table(_GLUCOSE_MMOL_TO_MGDL)),
    "2345-7": ("mg/dL", _mgdl_table(_GLUCOSE_MMOL_TO_MGDL)),
    # Creatinine
    "2160-0": (
        "mg/dL",
        {
            "mg/dl": lambda v: v,
            "umol/l": lambda v: v / _CREAT_UMOL_DIVISOR,
        },
    ),
}


def normalize_value(
    code_value: str | None, value: float, unit: str | None
) -> tuple[float, str | None]:
    """Normalize a numeric observation reading to its canonical unit.

    Args:
        code_value: The observation's LOINC ``code_value`` (or None).
        value: The numeric reading.
        unit: The unit as recorded in the source (or None).

    Returns:
        ``(normalized_value, canonical_unit)``. If the code is unknown the value
        and unit are returned verbatim. If the code is known but the source unit
        is unknown (or None), the value is returned unchanged labeled with the
        canonical unit. A conversion is applied only when both the code and the
        source unit are curated.
    """
    if not code_value or code_value not in _CODE_TABLE:
        return value, unit

    canonical_unit, converters = _CODE_TABLE[code_value]
    source = _canon(unit)

    # Already canonical (case/space/sign-insensitive) -> no-op, value untouched.
    if source == _canon(canonical_unit):
        return value, canonical_unit

    converter = converters.get(source)
    if converter is None:
        # Known code, unknown/None source unit: never guess. Trust the value,
        # label it canonical.
        return value, canonical_unit

    return round(converter(value), _ROUND_DP), canonical_unit
