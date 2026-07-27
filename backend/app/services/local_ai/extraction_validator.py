"""Deterministic, fail-closed validation for strict-local clinical extraction."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import weakref
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from pydantic import ValidationError

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.extraction_schema import (
    EVIDENCE_LOCATION_NORMALIZATION,
    FACT_CATEGORY_NAMES,
    AssertionState,
    ClinicalDocumentExtraction,
    EvidenceFact,
    ValidatedEvidence,
)

MAX_EXTRACTION_JSON_BYTES = 1_048_576
_FENCE_RE = re.compile(
    r"\A[ \t\r\n]*```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t\r\n]*\Z",
    re.DOTALL | re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"""
    \d{1,4}(?:\s*[-/.]\s*\d{1,2}){1,2}
    |
    (?:
        (?:(?:[<>]=?|\u2264|\u2265)\s*[+\-\u2212]?\s*)
        |
        (?:[~+\-\u00b1\u2212\u2248]\s*)
    )?
    \d+(?:[.,]\d+)?\s*[x\u00d7]\s*10\s*\^\s*
    (?:[+\-\u2212])?\s*\d+
    (?:\s*/\s*[^\W\d_][^\W_]*(?:[-'][^\W_]+)*)?
    |
    (?:
        (?:(?:[<>]=?|\u2264|\u2265)\s*[+\-\u2212]?\s*)
        |
        (?:[~+\-\u00b1\u2212\u2248]\s*)
    )?
    \d+(?:[.,]\d+)?\s*[x\u00d7]\s*10\s*
    [\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+
    (?:\s*/\s*[^\W\d_][^\W_]*(?:[-'][^\W_]+)*)?
    |
    (?:
        (?:(?:[<>]=?|\u2264|\u2265)\s*[+\-\u2212]?\s*)
        |
        (?:[~+\-\u00b1\u2212\u2248]\s*)
    )?
    \d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?
    \s*/\s*(?:[+\-\u2212]\s*)?
    \d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?
    |
    (?:
        (?:(?:[<>]=?|\u2264|\u2265)\s*[+\-\u2212]?\s*)
        |
        (?:[~+\-\u00b1\u2212\u2248]\s*)
    )?
    \d+(?:[.,]\d+)?(?:[eE][+\-\u2212]?\d+)?
    \s*[-\u2013\u2014]\s*
    (?:[+\-\u2212]\s*)?
    \d+(?:[.,]\d+)?(?:[eE][+\-\u2212]?\d+)?
    |
    (?:
        (?:(?:[<>]=?|\u2264|\u2265)\s*[+\-\u2212]?\s*)
        |
        (?:[~+\-\u00b1\u2212\u2248]\s*)
    )?
    \d+(?:[.,]\d+)?(?:[eE][+\-\u2212]?\d+)?
    |
    [^\W\d_][^\W_]*(?:[-'][^\W_]+)*
    (?:\s*[/\u00b7]\s*[^\W\d_][^\W_]*(?:[-'][^\W_]+)*
    (?:[\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)*)+
    (?:\s*/\s*\d+(?:[.,]\d+)?\s*
    [^\W\d_][^\W_]*(?:[\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)*)?
    |
    [^\W\d_][^\W_]*(?:[-'][^\W_]+)*
    \s+[^\W\d_][^\W_]*(?:[-'][^\W_]+)*
    [\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+
    |
    [^\W\d_][^\W_]*(?:[-'][^\W_]+)*
    |
    %
    """,
    re.UNICODE | re.VERBOSE,
)
_FAMILY_RE = re.compile(
    r"\b(?:family history|family hx|fhx|mother|father|sister|brother|"
    r"daughters?|sons?|child|children|husbands?|spouses?|wife|wives|"
    r"partners?|parents?|siblings?|grandmothers?|grandfathers?|"
    r"grandchildren|grandsons?|granddaughters?|maternal|paternal|"
    r"aunts?|uncles?|cousins?|nieces?|nephews?|moms?|dads?|"
    r"grandmas?|grandpas?)\b",
    re.IGNORECASE,
)
_EDUCATIONAL_RE = re.compile(
    r"\b(?:patient education|educational|for example|may cause|can cause|"
    r"risks include|general information)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspected?|question of|"
    r"concern for|cannot rule out|can(?:not|'t) exclude|may have|might have|"
    r"uncertain|unclear|equivocal|unlikely)\b",
    re.IGNORECASE,
)
_NOT_PERFORMED_RE = re.compile(
    r"(?:\b(?:recommend(?:ed|ation)?|consider(?:ed|ing)?|due for|"
    r"plan(?:ned)? to|scheduled?|proposed|to be done|will need|discussed|"
    r"offered|cancelled|canceled|deferred|aborted|not performed|never performed|"
    r"not done|never done|not completed|never completed)\b|"
    r"\bno\b(?:(?![.;|\n]).){0,80}\b(?:performed|done|completed)\b)",
    re.IGNORECASE,
)
_PERFORMED_RE = re.compile(
    r"(?:\b(?:s/p|status post|underwent|performed|post-?op|history of|removed|"
    r"resection|excision|completed|done on|biopsy)\b|"
    r"\b\w+(?:ectomy|otomy|ostomy|oplasty|plasty)\b)",
    re.IGNORECASE,
)
_NON_FACTUAL_CLAIM_RE = re.compile(
    r"\b(?:recommend(?:ed|ation)?|consider(?:ed|ing)?|proposed|planned?|"
    r"scheduled?|target(?:ed)?|goal|monitor(?:ing)?(?:\s+for)?|should|"
    r"advis(?:e|ed|ing)|due for|to be done|will need|offered|discussed)\b",
    re.IGNORECASE,
)
_PHI_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]+\]")
_DISALLOWED_MEDICATION_NAMES = frozenset({"ppi", "ssri", "nsaid", "ldn", "go"})
_NUMERIC_VALUE_RE = re.compile(
    r"""
    (?:
        (?:[<>]=?|\u2264|\u2265|[~+\-\u00b1\u2212\u2248])?\s*
        \d+(?:[.,]\d+)?(?:[eE][+\-\u2212]?\d+)?
        (?:
            \s*(?:/|[-\u2013\u2014])\s*
            [+\-\u2212]?\d+(?:[.,]\d+)?(?:[eE][+\-\u2212]?\d+)?
        )?
        |
        (?:[<>]=?|\u2264|\u2265|[~+\-\u00b1\u2212\u2248])?\s*
        \d+(?:[.,]\d+)?\s*[x\u00d7]\s*10
        (?:\s*\^\s*[+\-\u2212]?\d+|[\u207a\u207b]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)
    )
    """,
    re.VERBOSE,
)
_NUMERIC_VALUE_FIELDS = {
    "medications": frozenset({"dose_value"}),
    "immunizations": frozenset({"dose"}),
    "vital_signs": frozenset({"value"}),
}
_NUMERIC_UNIT_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "medications": (("dose_value", "dose_unit"),),
    "labs": (("value", "unit"),),
    "immunizations": (("dose", "dose_unit"),),
    "vital_signs": (("value", "unit"),),
}
_UNIT_SUFFIX_RE = re.compile(
    r"\s*(?P<unit>%|[^\W\d_][\w%./\u00b5\u03bc\u00b7^+-]*)",
    re.UNICODE,
)
_NON_UNIT_WORDS = frozenset(
    {
        "abnormal",
        "and",
        "at",
        "high",
        "is",
        "low",
        "normal",
        "on",
        "was",
        "were",
    }
)
_LIFECYCLE_BLOCKERS = {
    "medications": re.compile(
        r"\b(?:consider(?:ed|ing)?|recommend(?:ed|ation)?|proposed|planned?|"
        r"discussed|offered|may start|might start|not prescribed|never prescribed)\b",
        re.IGNORECASE,
    ),
    "allergies": re.compile(
        r"\b(?:possible|suspected?|question of|history of|uncertain|equivocal)\b",
        re.IGNORECASE,
    ),
    "encounters": re.compile(
        r"\bno\b(?:(?![.;|\n]).){0,80}\b"
        r"(?:visit|encounter|completed|attended|seen)\b",
        re.IGNORECASE,
    ),
    "immunizations": re.compile(
        r"\bno\b(?:(?![.;|\n]).){0,80}\b"
        r"(?:vaccine|vaccination|dose|shot|immunization|administered|"
        r"received|given|completed|vaccinated)\b",
        re.IGNORECASE,
    ),
    "diagnostic_reports": re.compile(
        r"\bno\b(?:(?![.;|\n]).){0,80}\b"
        r"(?:report|study|test|performed|finalized|reported|completed)\b",
        re.IGNORECASE,
    ),
    "care_plans": re.compile(
        r"\bno\b(?:(?![.;|\n]).){0,80}\b(?:care plan|plan|active)\b",
        re.IGNORECASE,
    ),
}
_DUPLICATE_SIGNATURE_EXCLUDES = {
    "fact_id",
    "verbatim",
    "page_number",
    "evidence_excerpt",
    "confidence",
    "normalized_value",
    "normalization_method",
    "normalization_version",
}

_STATUS_SIGNALS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "medications": (
        (
            "stopped",
            re.compile(
                r"\b(?:stopped|discontinued|no longer taking|not taking|"
                r"not currently taking|never takes?|never taking|does not take|"
                r"doesn't take|held|ceased)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "historical",
            re.compile(
                r"\b(?:historical|previously took|prior medication|past medication)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "active",
            re.compile(
                r"\b(?:active|current|taking|takes|continue(?:d)?|"
                r"daily|weekly|monthly|nightly|bid|tid|qid)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "allergies": (
        ("resolved", re.compile(r"\bresolved\b", re.IGNORECASE)),
        (
            "inactive",
            re.compile(r"\b(?:inactive|not active|no longer active)\b", re.IGNORECASE),
        ),
        (
            "historical",
            re.compile(r"\b(?:historical|past allergy)\b", re.IGNORECASE),
        ),
        ("active", re.compile(r"\b(?:active|allergy|allergic)\b", re.IGNORECASE)),
    ),
    "encounters": (
        (
            "cancelled",
            re.compile(
                r"\b(?:cancelled|canceled|aborted|"
                r"(?:not|never)(?:\s+\w+){0,3}\s+(?:finished|completed))\b",
                re.IGNORECASE,
            ),
        ),
        (
            "planned",
            re.compile(r"\b(?:planned|scheduled|upcoming)\b", re.IGNORECASE),
        ),
        (
            "in_progress",
            re.compile(r"\b(?:in[_ ]progress|ongoing)\b", re.IGNORECASE),
        ),
        (
            "finished",
            re.compile(
                r"\b(?:finished|completed|visited|seen|encounter|visit)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "immunizations": (
        (
            "planned",
            re.compile(
                r"\b(?:ordered|proposed|planned|scheduled|recommended|due)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "entered_in_error",
            re.compile(r"\b(?:entered in error|recorded in error)\b", re.IGNORECASE),
        ),
        (
            "not_done",
            re.compile(
                r"\b(?:not done|not administered|not received|not given|"
                r"not yet administered|not yet received|not yet given|"
                r"(?:not|never)(?:\s+\w+){0,3}\s+"
                r"(?:administered|received|given|completed|vaccinated)|"
                r"refused|declined)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "completed",
            re.compile(
                r"\b(?:administered|received|given|completed|vaccinated)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    "diagnostic_reports": (
        (
            "planned",
            re.compile(
                r"\b(?:ordered|proposed|planned|scheduled|recommended|pending)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "not_performed",
            re.compile(
                r"\b(?:not performed|not done|cancelled|canceled|aborted)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "not_final",
            re.compile(
                r"\b(?:(?:not|never)(?:\s+\w+){0,3}\s+"
                r"(?:final(?:ized)?|reported))\b",
                re.IGNORECASE,
            ),
        ),
        ("preliminary", re.compile(r"\bpreliminary\b", re.IGNORECASE)),
        ("amended", re.compile(r"\bamended\b", re.IGNORECASE)),
        (
            "final",
            re.compile(r"\b(?:final|finalized|reported|showed)\b", re.IGNORECASE),
        ),
    ),
    "care_plans": (
        (
            "planned",
            re.compile(
                r"\b(?:ordered|proposed|planned|draft|recommended)\b",
                re.IGNORECASE,
            ),
        ),
        ("resolved", re.compile(r"\bresolved\b", re.IGNORECASE)),
        (
            "inactive",
            re.compile(
                r"\b(?:inactive|not active|no longer active|closed|ended|"
                r"discontinued)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "historical",
            re.compile(r"\b(?:historical|past plan)\b", re.IGNORECASE),
        ),
        (
            "active",
            re.compile(
                r"\b(?:active|current|continue(?:d)?)\b",
                re.IGNORECASE,
            ),
        ),
    ),
}

_PROMOTED_LIFECYCLE_STATES = {
    "medications": "active",
    "allergies": "active",
    "encounters": "finished",
    "immunizations": "completed",
    "diagnostic_reports": "final",
    "care_plans": "active",
}

_CRITICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "medications": (
        "name",
        "dose_value",
        "dose_unit",
        "route",
        "frequency",
        "date",
    ),
    "conditions": ("name", "relationship", "date"),
    "procedures": ("name", "date", "provider"),
    "labs": (
        "name",
        "value",
        "unit",
        "reference_range",
        "interpretation",
        "date",
    ),
    "allergies": ("substance", "reaction", "severity", "date"),
    "encounters": ("name", "date", "provider", "facility"),
    "immunizations": (
        "name",
        "date",
        "route",
        "site",
        "dose",
        "dose_unit",
        "manufacturer",
        "lot",
    ),
    "vital_signs": ("name", "value", "unit", "date"),
    "diagnostic_reports": (
        "name",
        "findings",
        "interpretation",
        "date",
        "performer",
    ),
    "care_plans": ("title", "plan_items", "date"),
}


class _DuplicateJSONKey(ValueError):
    pass


def _fail(path: str, reason: str, *, page: int | None = None) -> None:
    location = path or "extraction"
    page_suffix = f" on page {page}" if page is not None else ""
    raise LocalValidationError(f"{location}{page_suffix}: {reason}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError


def _strip_json_wrapper(raw: str) -> str:
    if "```" not in raw:
        return raw
    match = _FENCE_RE.fullmatch(raw)
    if match is None or "```" in match.group("body"):
        _fail("extraction", "invalid JSON wrapper")
    return match.group("body")


def _parse_raw(raw: Mapping[str, object] | str) -> dict[str, object]:
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_EXTRACTION_JSON_BYTES:
            _fail("extraction", "JSON exceeds size limit")
        payload = _strip_json_wrapper(raw)
        try:
            decoded = json.loads(
                payload,
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except _DuplicateJSONKey:
            _fail("extraction", "duplicate JSON key")
        except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
            _fail("extraction", "invalid strict JSON")
        if not isinstance(decoded, dict):
            _fail("extraction", "top-level JSON value must be an object")
        return decoded

    if not isinstance(raw, Mapping):
        _fail("extraction", "input must be a mapping or JSON string")
    try:
        encoded = json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("extraction", "mapping is not strict JSON data")
    if len(encoded) > MAX_EXTRACTION_JSON_BYTES:
        _fail("extraction", "JSON exceeds size limit")
    return dict(raw)


def _format_location(location: tuple[str | int, ...]) -> str:
    if not location:
        return "extraction"
    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    return ".".join(parts)


def _parse_schema(payload: dict[str, object]) -> ClinicalDocumentExtraction:
    try:
        return ClinicalDocumentExtraction.model_validate(payload)
    except RecursionError:
        _fail("extraction", "schema nesting exceeds limit")
    except ValidationError as error:
        first = error.errors(
            include_url=False, include_context=False, include_input=False
        )[0]
        location = first.get("loc", ())
        if first.get("type") == "extra_forbidden":
            location = location[:-1]
        path = _format_location(location)
        if path == "extraction" and "duplicate fact_id" in first.get("msg", ""):
            _fail("fact_id", "duplicate fact identifier")
        _fail(path, "schema validation failed")


def _validate_pages(pages: Mapping[int, str]) -> dict[int, str]:
    if not isinstance(pages, Mapping):
        _fail("pages", "must be a page mapping")
    validated: dict[int, str] = {}
    for number, text in pages.items():
        if type(number) is not int or number <= 0:
            _fail("pages", "page numbers must be positive integers")
        if not isinstance(text, str):
            _fail(f"pages[{number}]", "page content must be text", page=number)
        validated[number] = text
    return validated


def _locator_text(value: str) -> str:
    """Return the representation used for all stored evidence offsets.

    Only Unicode case folding and whitespace collapsing are applied. Unicode
    normalization is intentionally not applied, so the validator never equates
    distinct source code-point sequences. Offsets are indices into this
    normalized representation, not byte or original-text offsets.
    """
    return " ".join(value.casefold().split())


def _tokens(value: str) -> list[str]:
    replacements = str.maketrans(
        {
            "\u2212": "-",
            "\u2264": "<=",
            "\u2265": ">=",
            "\u2248": "~",
            "\u00d7": "x",
        }
    )
    return [
        re.sub(r"\s+", "", token.translate(replacements)).casefold()
        for token in _TOKEN_RE.findall(value)
    ]


def _contains_token_sequence(haystack: str, needle: str) -> bool:
    expected = _tokens(needle)
    available = _tokens(haystack)
    if not expected:
        return False
    width = len(expected)
    return any(
        available[index : index + width] == expected for index in range(len(available))
    )


def _numeric_literal_pattern(value: str) -> re.Pattern[str]:
    """Match one complete source numeric form without rebinding an exponent."""
    pieces = re.split(r"(\s+|[/\u2013\u2014])", value.strip())
    body = "".join(
        r"\s+"
        if piece.isspace()
        else rf"\s*{re.escape(piece)}\s*"
        if piece in {"/", "\u2013", "\u2014"}
        else re.escape(piece)
        for piece in pieces
        if piece
    )
    return re.compile(rf"(?<![\w.,])(?P<value>{body})(?![\w.,])", re.IGNORECASE)


def _iter_exact_numeric_matches(source: str, value: str):
    for match in _numeric_literal_pattern(value).finditer(source):
        prefix = source[: match.start()]
        suffix = source[match.end() :]
        if re.search(r"(?:\^|\d[eE])\s*[+\-\u2212]?\s*$", prefix):
            continue
        if re.match(r"\s*(?:\^|[eE][+\-\u2212]?\d)", suffix):
            continue
        yield match


def _unit_pattern(unit: str) -> re.Pattern[str]:
    pieces = re.split(r"(\s+|[/\u00b7])", unit.strip())
    body = "".join(
        r"\s+"
        if piece.isspace()
        else rf"\s*{re.escape(piece)}\s*"
        if piece in {"/", "\u00b7"}
        else re.escape(piece)
        for piece in pieces
        if piece
    )
    return re.compile(rf"\s*{body}(?![\w%])", re.IGNORECASE)


def _validate_numeric_unit_pair(
    fact: EvidenceFact,
    *,
    value_field: str,
    unit_field: str,
    path: str,
) -> None:
    value = getattr(fact, value_field, None)
    if value is None or _NUMERIC_VALUE_RE.fullmatch(value) is None:
        return
    matches = list(_iter_exact_numeric_matches(fact.verbatim, value))
    if not matches:
        _fail(f"{path}.{value_field}", "numeric form is not exactly grounded")
    unit = getattr(fact, unit_field, None)
    if unit is not None:
        unit_pattern = _unit_pattern(unit)
        if not any(unit_pattern.match(fact.verbatim, match.end()) for match in matches):
            _fail(
                f"{path}.{unit_field}",
                "unit must be adjacent to its numeric source value",
            )
        return
    for match in matches:
        suffix = _UNIT_SUFFIX_RE.match(fact.verbatim, match.end())
        if suffix is None:
            continue
        candidate = suffix.group("unit")
        if candidate.casefold() not in _NON_UNIT_WORDS:
            _fail(f"{path}.{unit_field}", "source unit cannot be omitted")


def _iter_grounded_values(fact: EvidenceFact, category: str):
    for field_name in _CRITICAL_FIELDS[category]:
        value = getattr(fact, field_name, None)
        if value is None:
            continue
        if isinstance(value, list):
            for index, item in enumerate(value):
                yield f"{field_name}[{index}]", item
        else:
            yield field_name, value


def _subject_text(fact: EvidenceFact, category: str) -> str:
    field_name = {
        "allergies": "substance",
        "care_plans": "title",
    }.get(category, "name")
    value = getattr(fact, field_name, "")
    return value if isinstance(value, str) else ""


def _subject_pattern(subject: str) -> str | None:
    tokens = _tokens(subject)
    if not tokens:
        return None
    return r"(?<!\w)" + r"\W+".join(re.escape(token) for token in tokens) + r"(?!\w)"


_QUALIFIER_CONTINUATION_WORDS = frozenset(
    {
        "active",
        "administered",
        "as",
        "at",
        "bedtime",
        "bid",
        "bpm",
        "cm",
        "completed",
        "continued",
        "current",
        "daily",
        "day",
        "days",
        "dl",
        "every",
        "g",
        "given",
        "hour",
        "hours",
        "im",
        "in",
        "inhaled",
        "intramuscular",
        "intravenous",
        "iu",
        "iv",
        "kg",
        "l",
        "mcg",
        "meq",
        "mg",
        "min",
        "ml",
        "mm",
        "mmhg",
        "mmol",
        "mol",
        "month",
        "monthly",
        "months",
        "ng",
        "needed",
        "nightly",
        "once",
        "oral",
        "pg",
        "po",
        "performed",
        "prn",
        "qid",
        "received",
        "subcutaneous",
        "tid",
        "times",
        "topical",
        "twice",
        "ug",
        "unit",
        "units",
        "week",
        "weekly",
        "weeks",
        "\u00b5g",
        "\u03bcg",
    }
)


def _is_qualifier_only_continuation(line: str) -> bool:
    """Return whether a wrapped line contains qualifiers but no named subject."""
    without_numbers = _NUMERIC_VALUE_RE.sub(" ", line)
    words = [
        word.casefold()
        for word in re.findall(r"[^\W\d_]+", without_numbers, re.UNICODE)
    ]
    return bool(words) and all(word in _QUALIFIER_CONTINUATION_WORDS for word in words)


def _semantic_boundaries(source: str) -> list[tuple[int, int]]:
    boundaries = [
        (match.start(), match.end())
        for match in re.finditer(
            r"(?<!\d)[.,](?!\d)|[;|\u2022\u25cf\u25aa\u25e6]|"
            r"\b(?:and|but|while|whereas)\b",
            source,
            re.IGNORECASE,
        )
    ]
    status_led_continuations = {
        "active",
        "continued",
        "current",
        "given",
        "performed",
    }
    frequency_led_continuations = {
        "bid",
        "daily",
        "monthly",
        "nightly",
        "once",
        "qid",
        "tid",
        "twice",
        "weekly",
    }
    continuation_words = {
        "as",
        "at",
        "im",
        "inhaled",
        "intramuscular",
        "intravenous",
        "iv",
        "oral",
        "subcutaneous",
        "topical",
    }
    for match in re.finditer(r"\r?\n+", source):
        after = source[match.end() :]
        next_token = re.match(r"[ \t]*(?P<token>[^\W\d_]+|\d+)", after)
        if next_token is not None:
            token = next_token.group("token").casefold()
            line = re.split(r"\r?\n", after, maxsplit=1)[0].strip()
            if token in status_led_continuations and re.fullmatch(
                rf"{re.escape(token)}[\s.:;-]*",
                line,
                re.IGNORECASE,
            ):
                continue
            line_words = [
                word.casefold() for word in re.findall(r"[^\W\d_]+", line, re.UNICODE)
            ]
            if (
                token in frequency_led_continuations
                and line_words
                and all(word in frequency_led_continuations for word in line_words)
            ):
                continue
            if (
                token.isdigit() or token in continuation_words
            ) and _is_qualifier_only_continuation(line):
                continue
        boundaries.append((match.start(), match.end()))
    for match in re.finditer(r"/", source):
        before = source[: match.start()]
        after = source[match.end() :]
        left_digit = re.search(r"\d\s*$", before) is not None
        right_digit = re.match(r"\s*\d", after) is not None
        if left_digit and right_digit:
            continue
        left_unit = re.search(
            r"(?<![A-Za-z\u00b5\u03bc])([A-Za-z\u00b5\u03bc%]{1,4})\s*$",
            before,
        )
        right_unit = re.match(r"\s*([A-Za-z\u00b5\u03bc%]{1,4})(?![A-Za-z])", after)
        if left_unit is not None and right_unit is not None:
            continue
        boundaries.append((match.start(), match.end()))
    return sorted(set(boundaries))


def _subject_context(source: str, subject: str, path: str) -> str:
    """Return the bounded source clause containing the clinical subject."""
    pattern = _subject_pattern(subject)
    if pattern is None:
        _fail(f"{path}.verbatim", "fact subject is not groundable")
    matches = list(re.finditer(pattern, source, re.IGNORECASE))
    if len(matches) != 1:
        _fail(f"{path}.verbatim", "fact subject occurrence is ambiguous")
    match = matches[0]
    boundaries = _semantic_boundaries(source)
    preceding = [end for _, end in boundaries if end <= match.start()]
    start = max(preceding) if preceding else 0
    following = [start for start, _ in boundaries if start >= match.end()]
    end = min(following) if following else len(source)
    return source[start:end]


def _verbatim_pattern(verbatim: str) -> re.Pattern[str]:
    chunks = re.split(r"(\s+)", verbatim.strip())
    body = "".join(r"\s+" if chunk.isspace() else re.escape(chunk) for chunk in chunks)
    return re.compile(body, re.IGNORECASE)


def _verbatim_context(fact: EvidenceFact, path: str) -> str:
    """Return the clause around the exact evidence occurrence selected by the model."""
    source = fact.evidence_excerpt
    matches = list(_verbatim_pattern(fact.verbatim).finditer(source))
    if not matches:
        _fail(f"{path}.verbatim", "verbatim occurrence is not groundable")
    if len(matches) != 1:
        _fail(f"{path}.verbatim", "verbatim occurrence is ambiguous")
    match = matches[0]
    boundaries = _semantic_boundaries(source)
    preceding = [end for _, end in boundaries if end <= match.start()]
    following = [start for start, _ in boundaries if start >= match.end()]
    clause_start = max(preceding) if preceding else 0
    clause_end = min(following) if following else len(source)
    return source[clause_start:clause_end]


def _fact_context(fact: EvidenceFact, category: str, path: str) -> str:
    return _subject_context(
        _verbatim_context(fact, path),
        _subject_text(fact, category),
        path,
    )


def _subject_is_negated(source: str, subject: str) -> bool:
    subject_pattern = _subject_pattern(subject)
    if subject_pattern is None:
        return False
    before = re.compile(
        rf"\b(?:no(?:\s+evidence\s+of)?|without|denies?|denied|"
        rf"negative\s+for|ruled?\s+out|absence\s+of)\s+(?:the\s+)?"
        rf"{subject_pattern}",
        re.IGNORECASE,
    )
    after = re.compile(
        rf"{subject_pattern}(?:(?![;|\n]).){{0,80}}\b"
        r"(?:absent|not present|ruled out|negative|not documented|not confirmed|"
        r"not recorded)\b",
        re.IGNORECASE,
    )
    return before.search(source) is not None or after.search(source) is not None


_QUALITATIVE_LAB_RESULTS = frozenset(
    {
        "absent",
        "abnormal",
        "detected",
        "equivocal",
        "indeterminate",
        "negative",
        "nonreactive",
        "normal",
        "not detected",
        "positive",
        "present",
        "reactive",
        "trace",
    }
)


def _subject_has_prefix_negation(source: str, subject: str) -> bool:
    subject_pattern = _subject_pattern(subject)
    if subject_pattern is None:
        return False
    return (
        re.search(
            rf"\b(?:no(?:\s+evidence\s+of)?|without|denies?|denied|"
            rf"negative\s+for|ruled?\s+out|absence\s+of)\s+(?:the\s+)?"
            rf"{subject_pattern}",
            source,
            re.IGNORECASE,
        )
        is not None
    )


def _is_grounded_qualitative_lab_result(
    fact: EvidenceFact,
    context: str,
) -> bool:
    value = getattr(fact, "value", None)
    if not isinstance(value, str) or value.casefold() not in _QUALITATIVE_LAB_RESULTS:
        return False
    subject_pattern = _subject_pattern(_subject_text(fact, "labs"))
    value_pattern = _subject_pattern(value)
    if subject_pattern is None or value_pattern is None:
        return False
    return (
        re.search(
            rf"{subject_pattern}\s*(?:(?:result(?:ed)?|was|is)\s*)?"
            rf"[:= -]*{value_pattern}",
            context,
            re.IGNORECASE,
        )
        is not None
    )


def _is_supported_date(value: str) -> bool:
    """Return whether a grounded date has a deterministic valid calendar form."""
    if re.fullmatch(r"\d{4}", value):
        return 1 <= int(value) <= 9999
    partial = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{2})", value)
    if partial is not None:
        return 1 <= int(partial["year"]) <= 9999 and 1 <= int(partial["month"]) <= 12
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T"
        r"\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
        r"(?:Z|[+-]\d{2}:\d{2})?",
        value,
    ):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return True
    for date_format in (
        "%m/%d/%Y",
        "%B %Y",
        "%b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            datetime.strptime(value, date_format)
        except ValueError:
            continue
        return True
    return False


def _validate_assertion_guards(fact: EvidenceFact, category: str, path: str) -> None:
    assertion = getattr(fact, "assertion", None)
    context = _fact_context(fact, category, path)
    subject_negated = _subject_is_negated(context, _subject_text(fact, category))
    family_history = _FAMILY_RE.search(context) is not None
    uncertain = _UNCERTAIN_RE.search(context) is not None
    if (
        category == "labs"
        and _is_grounded_qualitative_lab_result(fact, context)
        and not _subject_has_prefix_negation(
            context,
            _subject_text(fact, category),
        )
    ):
        subject_negated = False
        uncertain = False
    non_factual = _NON_FACTUAL_CLAIM_RE.search(context) is not None or (
        category != "procedures" and _NOT_PERFORMED_RE.search(context) is not None
    )
    explicit_active_plan = (
        category == "care_plans"
        and re.search(r"\b(?:active|current|continued?)\b", context, re.IGNORECASE)
        is not None
    )
    if _EDUCATIONAL_RE.search(context):
        _fail(f"{path}.assertion", "educational text cannot become a clinical fact")
    claimed_value = getattr(fact, "status", None)
    claimed = claimed_value.value if hasattr(claimed_value, "value") else claimed_value
    promoted = _PROMOTED_LIFECYCLE_STATES.get(category)
    if non_factual and not explicit_active_plan:
        if promoted is not None and claimed == promoted:
            _fail(f"{path}.status", "non-factual language cannot be promoted")
        if (
            not (
                category == "procedures"
                and assertion == AssertionState.MENTIONED_NOT_PERFORMED
            )
            and promoted is None
        ):
            _fail(
                f"{path}.assertion",
                "non-factual language cannot become a clinical fact",
            )
    if assertion is not None:
        if assertion == AssertionState.PRESENT and (
            subject_negated or family_history or uncertain
        ):
            _fail(f"{path}.assertion", "present assertion contradicts source evidence")
        if assertion == AssertionState.NEGATED and not subject_negated:
            _fail(f"{path}.assertion", "negated assertion lacks source support")
        if assertion == AssertionState.FAMILY_HISTORY and (
            not family_history or subject_negated or uncertain
        ):
            _fail(
                f"{path}.assertion",
                "family-history assertion contradicts source evidence",
            )
        if assertion == AssertionState.UNCERTAIN and not uncertain:
            _fail(f"{path}.assertion", "uncertain assertion lacks source support")
        if (
            assertion == AssertionState.MENTIONED_NOT_PERFORMED
            and category != "procedures"
        ):
            _fail(f"{path}.assertion", "assertion is unsupported for this category")
    elif subject_negated or family_history or uncertain:
        _fail(
            f"{path}.assertion",
            "source qualifier cannot become a promoted clinical fact",
        )
    if category == "procedures":
        mentioned = _NOT_PERFORMED_RE.search(context) is not None
        if mentioned and assertion != AssertionState.MENTIONED_NOT_PERFORMED:
            _fail(f"{path}.assertion", "mentioned procedure cannot be marked performed")
        if not mentioned and assertion == AssertionState.MENTIONED_NOT_PERFORMED:
            _fail(
                f"{path}.assertion",
                "mentioned-not-performed assertion lacks source support",
            )
        if (
            assertion == AssertionState.PRESENT
            and getattr(fact, "date", None) is None
            and _PERFORMED_RE.search(context) is None
        ):
            _fail(f"{path}.assertion", "performed assertion lacks source support")


def _validate_local_precision_guards(
    fact: EvidenceFact, category: str, path: str
) -> None:
    values = [fact.verbatim]
    values.extend(str(value) for _, value in _iter_grounded_values(fact, category))
    if any(_PHI_PLACEHOLDER_RE.search(value) for value in values):
        _fail(path, "placeholder content cannot become a clinical fact")
    if category == "medications":
        name = str(getattr(fact, "name", "")).strip().casefold()
        if name in _DISALLOWED_MEDICATION_NAMES or len(re.sub(r"[^a-z]", "", name)) < 3:
            _fail(f"{path}.name", "downstream guard rejected fact")
    for field_name in _NUMERIC_VALUE_FIELDS.get(category, ()):
        value = getattr(fact, field_name, None)
        if value is not None and _NUMERIC_VALUE_RE.fullmatch(value) is None:
            _fail(f"{path}.{field_name}", "numeric value must not contain a unit")
    if category == "labs":
        value = getattr(fact, "value", None)
        if (
            value is not None
            and any(character.isdigit() for character in value)
            and _NUMERIC_VALUE_RE.fullmatch(value) is None
        ):
            _fail(f"{path}.value", "numeric value must not contain a unit")
    for field_name in ("unit", "dose_unit"):
        unit = getattr(fact, field_name, None)
        if unit is not None and _unit_is_truncated(fact.verbatim, unit):
            _fail(f"{path}.{field_name}", "unit omits a source denominator")
    for value_field, unit_field in _NUMERIC_UNIT_PAIRS.get(category, ()):
        _validate_numeric_unit_pair(
            fact,
            value_field=value_field,
            unit_field=unit_field,
            path=path,
        )


def _unit_is_truncated(source: str, unit: str) -> bool:
    normalized_source = _locator_text(source)
    normalized_unit = _locator_text(unit)
    pattern = re.compile(
        rf"(?<!\w){re.escape(normalized_unit)}(?!\w)",
        re.IGNORECASE,
    )
    found = False
    for match in pattern.finditer(normalized_source):
        found = True
        suffix = normalized_source[match.end() :]
        if re.match(r"\s*(?:/|\bper\b)\s*\S", suffix, re.IGNORECASE) is None:
            return False
    return found


def _validate_lifecycle_status(
    fact: EvidenceFact,
    category: str,
    path: str,
) -> None:
    """Reject a lifecycle claim that contradicts an explicit source signal."""
    signals = _STATUS_SIGNALS.get(category)
    if signals is None:
        return
    if (
        category == "allergies"
        and getattr(fact, "assertion", None) != AssertionState.PRESENT
    ):
        return
    source = _fact_context(fact, category, path)
    detected = next(
        (status for status, pattern in signals if pattern.search(source)),
        None,
    )
    claimed_value = getattr(fact, "status", None)
    claimed = claimed_value.value if hasattr(claimed_value, "value") else claimed_value
    promoted = _PROMOTED_LIFECYCLE_STATES[category]
    blocker = _LIFECYCLE_BLOCKERS.get(category)
    if claimed == promoted and (
        _UNCERTAIN_RE.search(source) is not None
        or (blocker is not None and blocker.search(source) is not None)
    ):
        _fail(f"{path}.status", "lifecycle state contradicts source evidence")
    if detected is not None and claimed not in {detected, "unknown"}:
        _fail(f"{path}.status", "lifecycle state contradicts source evidence")
    if claimed == promoted and detected != promoted:
        _fail(f"{path}.status", "lifecycle state lacks source support")


def _validate_duplicate_facts(extraction: ClinicalDocumentExtraction) -> None:
    for category in FACT_CATEGORY_NAMES:
        seen: set[str] = set()
        for index, fact in enumerate(getattr(extraction, category)):
            signature = json.dumps(
                _canonical_duplicate_value(
                    fact.model_dump(
                        mode="json",
                        exclude=_DUPLICATE_SIGNATURE_EXCLUDES,
                    )
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in seen:
                _fail(f"{category}[{index}]", "duplicate clinical fact")
            seen.add(signature)


def _canonical_duplicate_value(value: Any) -> Any:
    if isinstance(value, str):
        return _locator_text(value)
    if isinstance(value, list):
        return [_canonical_duplicate_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_duplicate_value(item) for key, item in value.items()}
    return value


def _make_evidence(
    fact: EvidenceFact,
    *,
    upload_id: str,
    path: str,
    page_text: str,
    field_paths: list[str],
) -> ValidatedEvidence:
    normalized_page = _locator_text(page_text)
    normalized_excerpt = _locator_text(fact.evidence_excerpt)
    normalized_verbatim = _locator_text(fact.verbatim)
    start = normalized_page.find(normalized_excerpt)
    if start < 0:
        _fail(f"{path}.evidence_excerpt", "excerpt not found", page=fact.page_number)
    if normalized_excerpt.find(normalized_verbatim) < 0:
        _fail(f"{path}.verbatim", "verbatim text not contained in evidence excerpt")
    end = start + len(normalized_excerpt)
    excerpt_hash = hashlib.sha256(normalized_excerpt.encode("utf-8")).hexdigest()
    canonical = {
        "version": "evidence-id.v1",
        "upload_id": upload_id,
        "page_number": fact.page_number,
        "start_offset": start,
        "end_offset": end,
        "offset_representation": EVIDENCE_LOCATION_NORMALIZATION,
        "excerpt_sha256": excerpt_hash,
        "field_paths": field_paths,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ValidatedEvidence(
        id=f"ev1_{digest[:40]}",
        upload_id=upload_id,
        page_number=fact.page_number,
        excerpt_sha256=excerpt_hash,
        start_offset=start,
        end_offset=end,
        offset_representation=EVIDENCE_LOCATION_NORMALIZATION,
        field_paths=field_paths,
    )


def _validate_patient_identity(
    extraction: ClinicalDocumentExtraction,
    pages: Mapping[int, str],
) -> None:
    if extraction.patient is None:
        return
    for field_name in ("name", "date_of_birth"):
        value = getattr(extraction.patient, field_name)
        if (
            field_name == "date_of_birth"
            and value is not None
            and not _is_supported_date(value)
        ):
            _fail("patient.date_of_birth", "date is not a supported calendar value")
        if value is not None and not any(
            _contains_token_sequence(page_text, value) for page_text in pages.values()
        ):
            _fail(f"patient.{field_name}", "value is not grounded on a document page")


def _validation_digest(extraction: ClinicalDocumentExtraction) -> str:
    payload = {
        "document": extraction.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in extraction.evidence],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_clinical_extraction(
    raw: Mapping[str, object] | str,
    pages: Mapping[int, str],
    *,
    upload_id: str,
) -> ClinicalDocumentExtraction:
    """Strictly parse and ground a complete NuExtract clinical document result."""
    if (
        type(upload_id) is not str
        or not upload_id
        or len(upload_id) > 256
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159
            for character in upload_id
        )
    ):
        _fail("upload_id", "must be bounded text")
    payload = _parse_raw(raw)
    extraction = _parse_schema(payload)
    _validate_duplicate_facts(extraction)
    page_map = _validate_pages(pages)
    _validate_patient_identity(extraction, page_map)
    evidence: list[ValidatedEvidence] = []

    for category in FACT_CATEGORY_NAMES:
        for index, fact in enumerate(getattr(extraction, category)):
            path = f"{category}[{index}]"
            page_text = page_map.get(fact.page_number)
            if page_text is None:
                _fail(path, "referenced page does not exist", page=fact.page_number)
            if any(
                value is not None
                for value in (
                    fact.normalized_value,
                    fact.normalization_method,
                    fact.normalization_version,
                )
            ):
                _fail(
                    f"{path}.normalization",
                    "model-authored normalization metadata is unsupported",
                )
            grounded_values = list(_iter_grounded_values(fact, category))
            field_paths = sorted(
                f"{path}.{field_name}" for field_name, _ in grounded_values
            )
            for semantic_field in ("assertion", "status"):
                if hasattr(fact, semantic_field):
                    field_paths.append(f"{path}.{semantic_field}")
            field_paths.sort()
            record = _make_evidence(
                fact,
                upload_id=upload_id,
                path=path,
                page_text=page_text,
                field_paths=field_paths,
            )
            for field_name, value in grounded_values:
                if field_name == "date" and not _is_supported_date(value):
                    _fail(f"{path}.date", "date is not a supported calendar value")
                if not _contains_token_sequence(fact.verbatim, value):
                    _fail(f"{path}.{field_name}", "token not grounded in verbatim")
                if not _contains_token_sequence(page_text, value):
                    _fail(
                        f"{path}.{field_name}",
                        "token not grounded on referenced page",
                        page=fact.page_number,
                    )
            _validate_assertion_guards(fact, category, path)
            _validate_local_precision_guards(fact, category, path)
            _validate_lifecycle_status(fact, category, path)
            fact._evidence_id = record.id
            fact._evidence_start_offset = record.start_offset
            fact._evidence_end_offset = record.end_offset
            evidence.append(record)

    evidence_ids = [item.id for item in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        _fail("evidence", "duplicate evidence identifier")
    extraction._evidence = tuple(evidence)
    return extraction


def _make_validation_api():
    records: dict[
        int,
        tuple[weakref.ReferenceType[ClinicalDocumentExtraction], str],
    ] = {}

    def validate(
        raw: Mapping[str, object] | str,
        pages: Mapping[int, str],
        *,
        upload_id: str,
    ) -> ClinicalDocumentExtraction:
        extraction = _validate_clinical_extraction(
            raw,
            pages,
            upload_id=upload_id,
        )
        object_id = id(extraction)

        def discard(
            reference: weakref.ReferenceType[ClinicalDocumentExtraction],
        ) -> None:
            current = records.get(object_id)
            if current is not None and current[0] is reference:
                records.pop(object_id, None)

        records[object_id] = (
            weakref.ref(extraction, discard),
            _validation_digest(extraction),
        )
        return extraction

    def require(extraction: ClinicalDocumentExtraction) -> None:
        """Reject schema objects not registered by deterministic validation."""
        facts = [
            (category, index, fact)
            for category in FACT_CATEGORY_NAMES
            for index, fact in enumerate(getattr(extraction, category))
        ]
        first_path = f"{facts[0][0]}[{facts[0][1]}]" if facts else "extraction"
        record = records.get(id(extraction))
        registered = (
            record is not None
            and record[0]() is extraction
            and hmac.compare_digest(record[1], _validation_digest(extraction))
        )
        if not registered:
            _fail(first_path, "evidence validation required")
        if len(facts) != len(extraction.evidence):
            _fail("extraction", "evidence validation required")
        for (category, index, fact), evidence in zip(
            facts,
            extraction.evidence,
            strict=True,
        ):
            prefix = f"{category}[{index}]."
            if (
                fact.evidence_id != evidence.id
                or not evidence.field_paths
                or any(
                    not field_path.startswith(prefix)
                    for field_path in evidence.field_paths
                )
            ):
                _fail(f"{category}[{index}]", "evidence validation required")

    return validate, require


validate_clinical_extraction, require_validated_extraction = _make_validation_api()
