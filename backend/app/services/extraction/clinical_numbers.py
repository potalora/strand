"""Deterministic parsing for source-grounded clinical numeric forms."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_GROUPED_DECIMAL_RE = re.compile(
    r"^[+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?(?:0|[1-9]\d*),\d+(?:[eE][+-]?\d+)?$")
_STANDARD_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_X10_RE = re.compile(
    r"^(?P<mantissa>[+-]?\d+(?:[.,]\d+)?)\s*[x\u00d7]\s*10"
    r"(?:\s*\^\s*(?P<ascii_exp>[+-]?\d+)|"
    r"(?P<super_exp>[\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))$",
    re.IGNORECASE,
)
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "\u2070": "0",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u207a": "+",
        "\u207b": "-",
    }
)
_MAX_ABS_EXPONENT = 1000


def _plain_decimal(value: str) -> Decimal | None:
    if _GROUPED_DECIMAL_RE.fullmatch(value):
        normalized = value.replace(",", "")
    elif _DECIMAL_COMMA_RE.fullmatch(value):
        normalized = value.replace(",", ".")
    elif _STANDARD_DECIMAL_RE.fullmatch(value):
        normalized = value
    else:
        return None
    try:
        parsed = Decimal(normalized)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def parse_clinical_decimal(value: object) -> Decimal | None:
    """Parse decimal, grouped-comma, E-notation, and clinical ``x10`` forms."""
    if isinstance(value, bool):
        return None
    text = str(value).strip().replace("\u2212", "-")
    scientific = _X10_RE.fullmatch(text)
    if scientific is None:
        return _plain_decimal(text)
    mantissa = _plain_decimal(scientific.group("mantissa"))
    if mantissa is None:
        return None
    exponent_text = scientific.group("ascii_exp")
    if exponent_text is None:
        exponent_text = scientific.group("super_exp").translate(
            _SUPERSCRIPT_TRANSLATION
        )
    try:
        exponent = int(exponent_text)
    except ValueError:
        return None
    if abs(exponent) > _MAX_ABS_EXPONENT:
        return None
    parsed = mantissa.scaleb(exponent)
    return parsed if parsed.is_finite() else None
