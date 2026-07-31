"""Qwen3.5 reference-only inference over the server-owned grounded contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import date

from .common import (
    Generate,
    GenerationError,
    LoadedRole,
    WorkerInputError,
    WorkerInputLimitError,
    _token_count,
    bounded_json,
    generate_content,
    load_role_from_payload,
    load_summary_processor_from_payload,
    parse_json_object,
    requested_output_tokens,
    validate_token_budget,
    validated_requested_output_tokens,
)

SUMMARY_OUTPUT_CAP = 4096
MAX_SUMMARY_INPUT_BYTES = 4 * 1024 * 1024
MAX_SUMMARY_SCHEMA_BYTES = 8 * 1024 * 1024
MAX_SUMMARY_SCHEMA_COMPLEXITY_UNITS = 16_384
MAX_FACTS = 512
MAX_EVIDENCE = 512
MAX_JSON_DEPTH = 12
MAX_JSON_NODES = 10_000
MAX_JSON_STRING = 4096
MAX_JSON_CHARACTERS = 50_000
MAX_EVIDENCE_EXCERPT_CHARS = 2000
MAX_OBSERVATION_UNIT_CHARACTERS = 512
MAX_SECTIONS = 20
MAX_CLAIMS_PER_SECTION = 100
MAX_UNCERTAINTIES = 100

SERVER_SAFETY_RULES = (
    "Select only server-supplied fact fields and their linked evidence.",
    "Do not emit free-text clinical claims, headings, or uncertainties.",
    "Do not create, correct, infer, or repair clinical facts.",
    "Do not provide diagnoses, treatment recommendations, medical advice, "
    "or clinical decision support.",
)
SUMMARY_TYPES = frozenset({"full_health", "category", "date_range", "single_record"})
SUMMARY_CATEGORIES = frozenset(
    {
        "allergy",
        "appointment",
        "care_plan",
        "care_team",
        "communication",
        "condition",
        "diagnostic_report",
        "document",
        "encounter",
        "imaging_study",
        "immunization",
        "medication",
        "observation",
        "procedure",
        "questionnaire_response",
        "service_request",
    }
)
SUMMARY_HEADINGS = frozenset(
    {
        "Overview",
        "Allergies",
        "Appointments",
        "Care plans",
        "Care teams",
        "Communications",
        "Conditions",
        "Diagnostic reports",
        "Documents",
        "Encounters",
        "Imaging studies",
        "Immunizations",
        "Medications",
        "Observations",
        "Procedures",
        "Questionnaires",
        "Service requests",
        "Other records",
    }
)
UNCERTAINTY_LABELS = {
    "assertion_uncertain": "The source record marks this information as uncertain.",
    "medication_dose_missing": "Medication dose is not available.",
    "medication_end_date_missing": "Medication end date is not available.",
    "record_date_missing": "Record date is not available.",
    "record_status_missing": "Record status is not available.",
    "result_missing": "Result value is not available.",
    "source_detail_missing": "Source detail is not available.",
}
RECORD_TYPE_HEADINGS = {
    "allergy": "Allergies",
    "appointment": "Appointments",
    "care_plan": "Care plans",
    "care_team": "Care teams",
    "communication": "Communications",
    "condition": "Conditions",
    "diagnostic_report": "Diagnostic reports",
    "document": "Documents",
    "encounter": "Encounters",
    "imaging_study": "Imaging studies",
    "immunization": "Immunizations",
    "medication": "Medications",
    "observation": "Observations",
    "procedure": "Procedures",
    "questionnaire_response": "Questionnaires",
    "service_request": "Service requests",
}
GENERIC_SUMMARY_SCALAR_FIELDS = frozenset({"date", "name", "status"})
SUMMARY_SCALAR_FIELDS = {
    "allergy": frozenset(
        {
            "assertion",
            "date",
            "name",
            "reaction",
            "record_type",
            "severity",
            "status",
            "substance",
        }
    ),
    "appointment": frozenset(
        {
            "date",
            "end",
            "location",
            "name",
            "provider",
            "record_type",
            "start",
            "status",
            "title",
        }
    ),
    "care_plan": frozenset({"date", "name", "record_type", "status", "title"}),
    "care_team": frozenset({"date", "name", "record_type", "status"}),
    "communication": frozenset(
        {
            "category",
            "date",
            "recipient",
            "record_type",
            "sender",
            "status",
            "topic",
        }
    ),
    "condition": frozenset(
        {
            "assertion",
            "date",
            "diagnosis",
            "name",
            "record_type",
            "relationship",
            "status",
        }
    ),
    "diagnostic_report": frozenset(
        {
            "assertion",
            "category",
            "date",
            "findings",
            "interpretation",
            "name",
            "performer",
            "record_type",
            "status",
        }
    ),
    "document": frozenset({"author", "date", "document_type", "record_type", "status", "title"}),
    "encounter": frozenset(
        {
            "date",
            "facility",
            "name",
            "provider",
            "record_type",
            "status",
            "visit_type",
        }
    ),
    "imaging_study": frozenset(
        {
            "body_site",
            "date",
            "description",
            "modality",
            "name",
            "record_type",
            "status",
        }
    ),
    "immunization": frozenset(
        {
            "date",
            "dose",
            "lot",
            "manufacturer",
            "name",
            "record_type",
            "route",
            "site",
            "status",
        }
    ),
    "medication": frozenset(
        {
            "date",
            "dose",
            "dose_unit",
            "dose_value",
            "effective_date",
            "end_date",
            "frequency",
            "name",
            "record_type",
            "route",
            "status",
        }
    ),
    "observation": frozenset(
        {
            "category",
            "date",
            "interpretation",
            "name",
            "record_type",
            "reference_range",
            "status",
            "unit",
            "value",
        }
    ),
    "procedure": frozenset(
        {
            "assertion",
            "body_site",
            "date",
            "name",
            "provider",
            "record_type",
            "status",
        }
    ),
    "questionnaire_response": frozenset(
        {"date", "questionnaire", "record_type", "status", "title"}
    ),
    "service_request": frozenset(
        {
            "date",
            "intent",
            "name",
            "performer",
            "priority",
            "record_type",
            "requester",
            "status",
        }
    ),
}
SUMMARY_OBJECT_FIELDS = {
    "medication": {"dosage": frozenset({"amount", "unit"})},
}
SUMMARY_SCALAR_LIST_FIELDS = {
    "care_plan": frozenset({"plan_items"}),
    "medication": frozenset({"statuses"}),
}
SUMMARY_OBJECT_LIST_FIELDS = {
    "care_team": {
        "participants": frozenset({"name", "role"}),
    },
    "questionnaire_response": {
        "answers": frozenset({"answer", "question"}),
    },
}
SUMMARY_DATE_FIELDS = ("date", "effective_date", "start", "end", "end_date")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RECORD_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_FORBIDDEN_FACT_FIELDS = frozenset(
    {
        "raw",
        "raw_bytes",
        "raw_document",
        "raw_upload",
        "source_document",
        "uploaded_file",
        "uploaded_files",
        "unresolved",
        "unresolved_fields",
        "rejected",
        "rejected_fields",
    }
)
_SUMMARY_INPUT_KEYS = frozenset(
    {
        "requested_scope",
        "facts",
        "evidence",
        "uncertainty_labels",
        "safety_rules",
    }
)
_SUMMARY_TRANSPORT_KEYS = frozenset(
    {
        "job_id",
        "manifest_path",
        "model_dir",
        "manifest_identity",
        "max_output_tokens",
    }
)
_SCOPE_KEYS = frozenset({"summary_type", "category", "date_from", "date_to", "record_ids"})


def _contains_render_control(value: str, *, allow_line_breaks: bool = False) -> bool:
    for character in value:
        category = unicodedata.category(character)
        if category in {"Cf", "Cs", "Zl", "Zp"}:
            return True
        if category == "Cc" and not (allow_line_breaks and character in {"\t", "\n", "\r"}):
            return True
    return False


def _safe_text(
    value: object,
    *,
    max_length: int,
    one_line: bool,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > max_length
        or (one_line and ("\n" in value or "\r" in value))
        or _contains_render_control(value, allow_line_breaks=not one_line)
    ):
        raise WorkerInputError("Summary request contains invalid text.")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise WorkerInputError("Summary request has an invalid identifier.")
    return value


def _record_identifier(value: object) -> str:
    if not isinstance(value, str) or _RECORD_IDENTIFIER.fullmatch(value) is None:
        raise WorkerInputError("Summary request has an invalid record identifier.")
    return value


def _identifier_list(
    value: object,
    *,
    minimum: int = 1,
    maximum: int = 32,
    records: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise WorkerInputError("Summary request has invalid identifier bindings.")
    parser = _record_identifier if records else _identifier
    result = [parser(item) for item in value]
    if len(result) != len(set(result)):
        raise WorkerInputError("Summary request has duplicate identifier bindings.")
    return result


def _field_path_list(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise WorkerInputError("Summary request has invalid fact field bindings.")
    result = [_safe_text(item, max_length=1024, one_line=True) for item in value]
    if any(not item.startswith("/") for item in result) or len(result) != len(set(result)):
        raise WorkerInputError("Summary request has invalid fact field bindings.")
    return result


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        raise WorkerInputError("Summary request JSON is invalid.") from None


def _validate_plain_json(
    value: object,
    *,
    forbidden_fields: frozenset[str] = frozenset(),
) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            raise WorkerInputError("Summary request JSON exceeds structural limits.")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float:
            if item != item or item in {float("inf"), float("-inf")}:
                raise WorkerInputError("Summary request JSON is invalid.")
            return
        if type(item) is str:
            if len(item) > MAX_JSON_STRING or _contains_render_control(item):
                raise WorkerInputError("Summary request JSON string is invalid.")
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if (
                    type(key) is not str
                    or not key
                    or len(key) > 128
                    or _contains_render_control(key)
                    or key.casefold() in forbidden_fields
                ):
                    raise WorkerInputError("Summary request JSON key is invalid.")
                visit(child, depth + 1)
            return
        raise WorkerInputError("Summary request is not plain JSON.")

    visit(value, 0)
    if len(_canonical_json(value)) > MAX_JSON_CHARACTERS:
        raise WorkerInputError("Summary request JSON exceeds its limit.")


def _strict_json_value(value: str) -> object:
    def reject_constant(_value: str) -> None:
        raise ValueError

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
        raise WorkerInputError("Summary request contains invalid canonical JSON.") from None
    _validate_plain_json(parsed, forbidden_fields=_FORBIDDEN_FACT_FIELDS)
    if _canonical_json(parsed) != value:
        raise WorkerInputError("Summary request contains noncanonical JSON.")
    return parsed


def _is_plain_json_scalar(value: object) -> bool:
    return value is None or type(value) in {bool, int, float, str}


def _finite_json_number(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _typed_observation_value(value: object) -> bool:
    if type(value) is not dict:
        return False
    kind = value.get("kind")
    if kind == "quantity":
        if set(value) != {"kind", "comparator", "number", "unit"}:
            return False
        number = value.get("number")
        if value.get("comparator") not in {"<", "<=", ">", ">=", "~"} or not _finite_json_number(
            number
        ):
            return False
        try:
            _safe_text(
                value.get("unit"),
                max_length=MAX_OBSERVATION_UNIT_CHARACTERS,
                one_line=True,
            )
        except WorkerInputError:
            return False
        return True
    if kind == "ratio":
        if set(value) != {
            "kind",
            "numerator",
            "denominator",
            "numerator_unit",
            "denominator_unit",
        }:
            return False
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if (
            not _finite_json_number(numerator)
            or not _finite_json_number(denominator)
            or denominator == 0
        ):
            return False
        try:
            for unit_name in ("numerator_unit", "denominator_unit"):
                _safe_text(
                    value.get(unit_name),
                    max_length=MAX_OBSERVATION_UNIT_CHARACTERS,
                    one_line=True,
                )
        except WorkerInputError:
            return False
        return True
    return False


def _validate_safe_summary_content(value: Mapping[str, object]) -> None:
    record_type = value.get("record_type")
    if record_type is None:
        scalar_fields = GENERIC_SUMMARY_SCALAR_FIELDS
        object_fields: Mapping[str, frozenset[str]] = {}
        scalar_list_fields = frozenset()
        object_list_fields: Mapping[str, frozenset[str]] = {}
    else:
        if type(record_type) is not str or record_type not in SUMMARY_SCALAR_FIELDS:
            raise WorkerInputError("Summary fact has an unsupported record type.")
        scalar_fields = SUMMARY_SCALAR_FIELDS[record_type]
        object_fields = SUMMARY_OBJECT_FIELDS.get(record_type, {})
        scalar_list_fields = SUMMARY_SCALAR_LIST_FIELDS.get(record_type, frozenset())
        object_list_fields = SUMMARY_OBJECT_LIST_FIELDS.get(record_type, {})

    for key, item in value.items():
        if record_type == "observation" and key == "value" and type(item) is dict:
            if not _typed_observation_value(item):
                raise WorkerInputError("Summary fact has an invalid observation value.")
            continue
        if key in scalar_fields:
            if not _is_plain_json_scalar(item):
                raise WorkerInputError("Summary fact has an invalid scalar field.")
            continue
        nested_fields = object_fields.get(key)
        if nested_fields is not None:
            if (
                type(item) is not dict
                or set(item) - nested_fields
                or not all(_is_plain_json_scalar(child) for child in item.values())
            ):
                raise WorkerInputError("Summary fact has an invalid object field.")
            continue
        if key in scalar_list_fields:
            if type(item) is not list or not all(_is_plain_json_scalar(child) for child in item):
                raise WorkerInputError("Summary fact has an invalid list field.")
            continue
        item_fields = object_list_fields.get(key)
        if item_fields is not None:
            if type(item) is not list:
                raise WorkerInputError("Summary fact has an invalid object-list field.")
            for child in item:
                if (
                    type(child) is not dict
                    or set(child) - item_fields
                    or not all(_is_plain_json_scalar(nested) for nested in child.values())
                ):
                    raise WorkerInputError("Summary fact has an invalid object-list field.")
            continue
        raise WorkerInputError("Summary fact contains a non-allowlisted field.")


def _scope(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _SCOPE_KEYS:
        raise WorkerInputError("Summary request has invalid requested scope.")
    summary_type = value.get("summary_type")
    category = value.get("category")
    date_from = value.get("date_from")
    date_to = value.get("date_to")
    record_ids = _identifier_list(
        value.get("record_ids"),
        minimum=0,
        maximum=100,
        records=True,
    )
    if summary_type not in SUMMARY_TYPES:
        raise WorkerInputError("Summary request has invalid requested scope.")
    if category is not None and category not in SUMMARY_CATEGORIES:
        raise WorkerInputError("Summary request has invalid requested scope.")
    if summary_type == "category":
        if category is None:
            raise WorkerInputError("Summary request has invalid requested scope.")
    elif category is not None:
        raise WorkerInputError("Summary request has invalid requested scope.")
    parsed_dates: list[str | None] = []
    for date_value in (date_from, date_to):
        if date_value is None:
            parsed_dates.append(None)
            continue
        if not isinstance(date_value, str):
            raise WorkerInputError("Summary request has invalid requested scope.")
        try:
            parsed = date.fromisoformat(date_value)
        except ValueError:
            raise WorkerInputError("Summary request has invalid requested scope.") from None
        if parsed.isoformat() != date_value:
            raise WorkerInputError("Summary request has invalid requested scope.")
        parsed_dates.append(date_value)
    if summary_type == "date_range":
        if parsed_dates[0] is None or parsed_dates[1] is None or parsed_dates[0] > parsed_dates[1]:
            raise WorkerInputError("Summary request has invalid requested scope.")
    elif any(item is not None for item in parsed_dates):
        raise WorkerInputError("Summary request has invalid requested scope.")
    if summary_type == "single_record":
        if len(record_ids) != 1:
            raise WorkerInputError("Summary request has invalid requested scope.")
    elif record_ids:
        raise WorkerInputError("Summary request has invalid requested scope.")
    return {
        "summary_type": summary_type,
        "category": category,
        "date_from": parsed_dates[0],
        "date_to": parsed_dates[1],
        "record_ids": record_ids,
    }


def _fact_content(fact: Mapping[str, object]) -> dict[str, object]:
    content = _strict_json_value(str(fact["content_json"]))
    if type(content) is not dict:
        raise WorkerInputError("Summary fact is invalid.")
    return content


def _fact_date(content: Mapping[str, object]) -> date | None:
    for field_name in SUMMARY_DATE_FIELDS:
        raw = content.get(field_name)
        if type(raw) is not str or len(raw) < 10:
            continue
        if len(raw) > 10 and raw[10] not in {"T", " "}:
            continue
        try:
            parsed = date.fromisoformat(raw[:10])
        except ValueError:
            continue
        if parsed.isoformat() == raw[:10]:
            return parsed
    return None


def _validate_fact_scope(
    scope: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
) -> None:
    summary_type = scope["summary_type"]
    if summary_type == "single_record":
        expected_record_id = scope["record_ids"][0]  # type: ignore[index]
        if any(fact["record_id"] != expected_record_id for fact in facts):
            raise WorkerInputError("Summary facts do not match the requested scope.")
    elif summary_type == "category":
        if any(_fact_content(fact).get("record_type") != scope["category"] for fact in facts):
            raise WorkerInputError("Summary facts do not match the requested scope.")
    elif summary_type == "date_range":
        date_from = date.fromisoformat(str(scope["date_from"]))
        date_to = date.fromisoformat(str(scope["date_to"]))
        for fact in facts:
            fact_date = _fact_date(_fact_content(fact))
            if fact_date is None or not date_from <= fact_date <= date_to:
                raise WorkerInputError("Summary facts do not match the requested scope.")


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _flatten_content(content: Mapping[str, object]) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []

    def visit(value: object, path: str) -> None:
        if type(value) is dict and value:
            for key in sorted(value):
                visit(value[key], f"{path}/{_escape_pointer_token(key)}")
            return
        if type(value) is list and value:
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
            return
        fields.append({"path": path, "value_json": _canonical_json(value)})

    for key in sorted(content):
        visit(content[key], f"/{_escape_pointer_token(key)}")
    if not fields:
        raise WorkerInputError("Summary fact has no fields.")
    return fields


def _fact(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "fact_id",
        "record_id",
        "content_json",
        "fields",
        "evidence_ids",
    }:
        raise WorkerInputError("Summary fact is invalid.")
    fact_id = _identifier(value.get("fact_id"))
    record_id = _record_identifier(value.get("record_id"))
    content_json = _safe_text(
        value.get("content_json"),
        max_length=MAX_JSON_CHARACTERS,
        one_line=True,
    )
    content = _strict_json_value(content_json)
    if type(content) is not dict or not content:
        raise WorkerInputError("Summary fact is invalid.")
    _validate_safe_summary_content(content)
    fields_value = value.get("fields")
    if not isinstance(fields_value, list) or not 1 <= len(fields_value) <= MAX_JSON_NODES:
        raise WorkerInputError("Summary fact is invalid.")
    fields: list[dict[str, str]] = []
    for item in fields_value:
        if type(item) is not dict or set(item) != {"path", "value_json"}:
            raise WorkerInputError("Summary fact field is invalid.")
        path = _safe_text(item.get("path"), max_length=1024, one_line=True)
        if not path.startswith("/"):
            raise WorkerInputError("Summary fact field is invalid.")
        value_json = _safe_text(
            item.get("value_json"),
            max_length=MAX_JSON_STRING + 32,
            one_line=True,
        )
        parsed = _strict_json_value(value_json)
        if type(parsed) in {list, dict} and parsed:
            raise WorkerInputError("Summary fact field is invalid.")
        fields.append({"path": path, "value_json": value_json})
    if fields != _flatten_content(content):
        raise WorkerInputError("Summary fact fields do not match its snapshot.")
    evidence_ids = _identifier_list(value.get("evidence_ids"))
    snapshot = _canonical_json(
        {
            "content_json": content_json,
            "evidence_ids": sorted(evidence_ids),
            "record_id": record_id,
        }
    )
    expected_fact_id = "fact1_" + hashlib.sha256(snapshot.encode()).hexdigest()[:40]
    if fact_id != expected_fact_id:
        raise WorkerInputError("Summary fact identifier does not match its snapshot.")
    record_type = content.get("record_type")
    allowed_heading = (
        RECORD_TYPE_HEADINGS.get(record_type, "Other records")
        if isinstance(record_type, str)
        else "Other records"
    )
    return {
        "fact_id": fact_id,
        "record_id": record_id,
        "content_json": content_json,
        "fields": fields,
        "evidence_ids": evidence_ids,
        "allowed_heading": allowed_heading,
    }


def _evidence(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "evidence_id",
        "source_id",
        "excerpt",
        "page_number",
        "section",
        "fact_ids",
        "field_paths",
    }:
        raise WorkerInputError("Summary evidence is invalid.")
    evidence_id = _identifier(value.get("evidence_id"))
    source_id = _identifier(value.get("source_id"))
    excerpt = _safe_text(
        value.get("excerpt"),
        max_length=MAX_EVIDENCE_EXCERPT_CHARS,
        one_line=False,
    )
    page_number = value.get("page_number")
    section = value.get("section")
    if page_number is not None and (
        not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or not 1 <= page_number <= 100_000
    ):
        raise WorkerInputError("Summary evidence is invalid.")
    safe_section = (
        _safe_text(section, max_length=2000, one_line=True) if section is not None else None
    )
    field_paths = _field_path_list(value.get("field_paths"))
    snapshot = _canonical_json(
        {
            "excerpt": excerpt,
            "field_paths": sorted(field_paths),
            "page_number": page_number,
            "section": safe_section,
            "source_id": source_id,
        }
    )
    expected_id = "evidence1_" + hashlib.sha256(snapshot.encode()).hexdigest()[:40]
    if evidence_id != expected_id:
        raise WorkerInputError("Summary evidence identifier does not match its snapshot.")
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "excerpt": excerpt,
        "page_number": page_number,
        "section": safe_section,
        "fact_ids": _identifier_list(value.get("fact_ids")),
        "field_paths": field_paths,
    }


def _uncertainty(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "uncertainty_id",
        "template_id",
        "label",
        "fact_ids",
        "evidence_ids",
    }:
        raise WorkerInputError("Summary uncertainty is invalid.")
    uncertainty_id = _identifier(value.get("uncertainty_id"))
    template_id = value.get("template_id")
    if (
        type(template_id) is not str
        or template_id not in UNCERTAINTY_LABELS
        or value.get("label") != UNCERTAINTY_LABELS[template_id]
    ):
        raise WorkerInputError("Summary uncertainty label is invalid.")
    label = _safe_text(value.get("label"), max_length=2000, one_line=True)
    fact_ids = _identifier_list(value.get("fact_ids"))
    evidence_ids = _identifier_list(value.get("evidence_ids"))
    snapshot = _canonical_json(
        {
            "evidence_ids": sorted(evidence_ids),
            "fact_ids": sorted(fact_ids),
            "template_id": template_id,
        }
    )
    expected_id = "uncertainty1_" + hashlib.sha256(snapshot.encode()).hexdigest()[:40]
    if uncertainty_id != expected_id:
        raise WorkerInputError("Summary uncertainty identifier does not match its snapshot.")
    return {
        "uncertainty_id": uncertainty_id,
        "template_id": template_id,
        "label": label,
        "fact_ids": fact_ids,
        "evidence_ids": evidence_ids,
    }


def _validated_input(payload: Mapping[str, object]) -> dict[str, object]:
    if (
        not _SUMMARY_INPUT_KEYS.issubset(payload)
        or set(payload) - _SUMMARY_INPUT_KEYS - _SUMMARY_TRANSPORT_KEYS
    ):
        raise WorkerInputError("Summary request is invalid.")
    facts_value = payload.get("facts")
    evidence_value = payload.get("evidence")
    uncertainty_value = payload.get("uncertainty_labels")
    if (
        not isinstance(facts_value, list)
        or len(facts_value) > MAX_FACTS
        or not isinstance(evidence_value, list)
        or len(evidence_value) > MAX_EVIDENCE
        or not isinstance(uncertainty_value, list)
        or len(uncertainty_value) > MAX_UNCERTAINTIES
        or payload.get("safety_rules") != list(SERVER_SAFETY_RULES)
    ):
        raise WorkerInputError("Summary request is invalid.")
    scope = _scope(payload.get("requested_scope"))
    facts = [_fact(value) for value in facts_value]
    evidence = [_evidence(value) for value in evidence_value]
    uncertainties = [_uncertainty(value) for value in uncertainty_value]
    fact_registry = {str(item["fact_id"]): item for item in facts}
    evidence_registry = {str(item["evidence_id"]): item for item in evidence}
    uncertainty_registry = {str(item["uncertainty_id"]): item for item in uncertainties}
    if (
        len(fact_registry) != len(facts)
        or len(evidence_registry) != len(evidence)
        or len(uncertainty_registry) != len(uncertainties)
    ):
        raise WorkerInputError("Summary request contains duplicate identifiers.")
    _validate_fact_scope(scope, facts)
    for fact_id, fact in fact_registry.items():
        for evidence_id in fact["evidence_ids"]:
            linked = evidence_registry.get(evidence_id)
            if linked is None or fact_id not in linked["fact_ids"]:
                raise WorkerInputError("Summary fact and evidence links disagree.")
    for evidence_id, evidence_item in evidence_registry.items():
        for fact_id in evidence_item["fact_ids"]:
            linked = fact_registry.get(fact_id)
            if linked is None or evidence_id not in linked["evidence_ids"]:
                raise WorkerInputError("Summary fact and evidence links disagree.")
            known_paths = {
                str(field["path"]) for field in linked["fields"] if isinstance(field, dict)
            }
            if set(evidence_item["field_paths"]) - known_paths:
                raise WorkerInputError("Summary evidence has invalid fact field support.")
    for item in uncertainty_registry.values():
        if set(item["fact_ids"]) - set(fact_registry) or set(item["evidence_ids"]) - set(
            evidence_registry
        ):
            raise WorkerInputError("Summary uncertainty support is invalid.")
        if any(
            not set(item["evidence_ids"]).intersection(fact_registry[fact_id]["evidence_ids"])
            for fact_id in item["fact_ids"]
        ):
            raise WorkerInputError("Summary uncertainty support is invalid.")
    return {
        "requested_scope": scope,
        "facts": facts,
        "evidence": evidence,
        "uncertainty_labels": uncertainties,
        "safety_rules": list(SERVER_SAFETY_RULES),
    }


def _record_heading(fact: Mapping[str, object]) -> str:
    fields = fact.get("fields")
    if not isinstance(fields, list):
        return "Other records"
    for field in fields:
        if isinstance(field, dict) and field.get("path") == "/record_type":
            try:
                record_type = json.loads(str(field.get("value_json")))
            except (TypeError, ValueError, json.JSONDecodeError):
                return "Other records"
            if isinstance(record_type, str):
                return RECORD_TYPE_HEADINGS.get(record_type, "Other records")
    return "Other records"


def _validated_output(
    raw: str,
    safe_input: Mapping[str, object],
) -> dict[str, object]:
    try:
        value = parse_json_object(raw)
        _validate_plain_json(value)
        if set(value) != {"sections", "uncertainties"}:
            raise ValueError
        sections = value.get("sections")
        uncertainty_refs = value.get("uncertainties")
        if (
            not isinstance(sections, list)
            or len(sections) > MAX_SECTIONS
            or not isinstance(uncertainty_refs, list)
            or len(uncertainty_refs) > MAX_UNCERTAINTIES
        ):
            raise ValueError
        facts_value = safe_input["facts"]
        evidence_value = safe_input["evidence"]
        uncertainties_value = safe_input["uncertainty_labels"]
        if (
            not isinstance(facts_value, list)
            or not isinstance(evidence_value, list)
            or not isinstance(uncertainties_value, list)
        ):
            raise ValueError
        facts = {str(item["fact_id"]): item for item in facts_value}
        evidence = {str(item["evidence_id"]): item for item in evidence_value}
        uncertainties = {str(item["uncertainty_id"]): item for item in uncertainties_value}
        for section in sections:
            if type(section) is not dict or set(section) != {"heading", "claims"}:
                raise ValueError
            heading = section.get("heading")
            claims = section.get("claims")
            if (
                heading not in SUMMARY_HEADINGS
                or not isinstance(claims, list)
                or len(claims) > MAX_CLAIMS_PER_SECTION
            ):
                raise ValueError
            for claim in claims:
                if type(claim) is not dict or set(claim) != {
                    "fact_id",
                    "field_paths",
                    "evidence_ids",
                }:
                    raise ValueError
                fact_id = _identifier(claim.get("fact_id"))
                field_paths = _field_path_list(claim.get("field_paths"))
                evidence_ids = _identifier_list(claim.get("evidence_ids"))
                fact = facts.get(fact_id)
                if fact is None:
                    raise ValueError
                known_paths = {
                    str(field["path"]) for field in fact["fields"] if isinstance(field, dict)
                }
                if (
                    set(field_paths) - known_paths
                    or not set(evidence_ids).issubset(fact["evidence_ids"])
                    or heading not in {"Overview", fact["allowed_heading"]}
                ):
                    raise ValueError
                supported_paths = {
                    str(path)
                    for evidence_id in evidence_ids
                    for path in evidence[evidence_id]["field_paths"]
                }
                if set(field_paths) - supported_paths:
                    raise ValueError
        seen_uncertainties: set[str] = set()
        for reference in uncertainty_refs:
            if type(reference) is not dict or set(reference) != {
                "uncertainty_id",
                "fact_ids",
                "evidence_ids",
            }:
                raise ValueError
            uncertainty_id = _identifier(reference.get("uncertainty_id"))
            fact_ids = _identifier_list(reference.get("fact_ids"))
            evidence_ids = _identifier_list(reference.get("evidence_ids"))
            uncertainty = uncertainties.get(uncertainty_id)
            if (
                uncertainty is None
                or uncertainty_id in seen_uncertainties
                or set(fact_ids) != set(uncertainty["fact_ids"])
                or set(evidence_ids) != set(uncertainty["evidence_ids"])
            ):
                raise ValueError
            seen_uncertainties.add(uncertainty_id)
    except (
        GenerationError,
        KeyError,
        TypeError,
        ValueError,
        WorkerInputError,
    ):
        raise GenerationError("Local summary returned invalid JSON.") from None
    return value


def _validated_reference_document(value: object) -> dict[str, object]:
    """Validate a maximal reference-only document before loading its tokenizer."""

    try:
        _validate_plain_json(value)
        if type(value) is not dict or set(value) != {"sections", "uncertainties"}:
            raise ValueError
        sections = value["sections"]
        uncertainty_refs = value["uncertainties"]
        if (
            not isinstance(sections, list)
            or len(sections) > MAX_SECTIONS
            or not isinstance(uncertainty_refs, list)
            or len(uncertainty_refs) > MAX_UNCERTAINTIES
        ):
            raise ValueError
        headings: set[str] = set()
        facts: set[str] = set()
        for section in sections:
            if type(section) is not dict or set(section) != {"heading", "claims"}:
                raise ValueError
            heading = section["heading"]
            claims = section["claims"]
            if (
                heading not in SUMMARY_HEADINGS
                or heading in headings
                or not isinstance(claims, list)
                or len(claims) > MAX_CLAIMS_PER_SECTION
            ):
                raise ValueError
            headings.add(str(heading))
            for claim in claims:
                if type(claim) is not dict or set(claim) != {
                    "fact_id",
                    "field_paths",
                    "evidence_ids",
                }:
                    raise ValueError
                fact_id = _identifier(claim["fact_id"])
                if fact_id in facts:
                    raise ValueError
                facts.add(fact_id)
                _field_path_list(claim["field_paths"])
                _identifier_list(claim["evidence_ids"])
        uncertainties: set[str] = set()
        for reference in uncertainty_refs:
            if type(reference) is not dict or set(reference) != {
                "uncertainty_id",
                "fact_ids",
                "evidence_ids",
            }:
                raise ValueError
            uncertainty_id = _identifier(reference["uncertainty_id"])
            if uncertainty_id in uncertainties:
                raise ValueError
            uncertainties.add(uncertainty_id)
            _identifier_list(reference["fact_ids"])
            _identifier_list(reference["evidence_ids"])
    except (KeyError, TypeError, ValueError, WorkerInputError):
        raise WorkerInputError("Summary reference document is invalid.") from None
    return value


def count_summary_reference_tokens(
    payload: Mapping[str, object],
    *,
    processor_loader: Callable[[Mapping[str, object]], object] = (
        load_summary_processor_from_payload
    ),
) -> dict[str, int]:
    """Count compact maximal-reference JSON without loading summary model weights."""

    if set(payload) - {
        "job_id",
        "reference_document",
        "manifest_path",
        "model_dir",
        "manifest_identity",
    }:
        raise WorkerInputError("Summary token-count request is invalid.")
    reference = _validated_reference_document(payload.get("reference_document"))
    compact = bounded_json(reference, max_bytes=MAX_SUMMARY_INPUT_BYTES)
    processor = processor_loader(payload)
    return {"token_count": _token_count(processor, compact)}


def _contains_once(item_schema: Mapping[str, object]) -> dict[str, object]:
    return {
        "contains": dict(item_schema),
        "minContains": 0,
        "maxContains": 1,
    }


def _claim_schema(
    fact: Mapping[str, object],
    evidence: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    fact_id = str(fact["fact_id"])
    linked_evidence_ids = sorted(str(item) for item in fact["evidence_ids"])
    fact_paths = {str(field["path"]) for field in fact["fields"] if isinstance(field, Mapping)}
    evidence_paths = {
        evidence_id: sorted(
            fact_paths.intersection(str(path) for path in evidence[evidence_id]["field_paths"])
        )
        for evidence_id in linked_evidence_ids
    }
    supported_paths = sorted({path for paths in evidence_paths.values() for path in paths})
    if not supported_paths or any(not paths for paths in evidence_paths.values()):
        raise WorkerInputError("Summary schema has invalid evidence bindings.")

    link_conditions: list[dict[str, object]] = []
    for path in supported_paths:
        supporting_evidence = sorted(
            evidence_id for evidence_id, paths in evidence_paths.items() if path in paths
        )
        link_conditions.append(
            {
                "if": {
                    "properties": {
                        "field_paths": {"contains": {"const": path}},
                    },
                    "required": ["field_paths"],
                },
                "then": {
                    "properties": {
                        "evidence_ids": {
                            "contains": {"enum": supporting_evidence},
                        }
                    }
                },
            }
        )
    for evidence_id, paths in evidence_paths.items():
        link_conditions.append(
            {
                "if": {
                    "properties": {
                        "evidence_ids": {"contains": {"const": evidence_id}},
                    },
                    "required": ["evidence_ids"],
                },
                "then": {
                    "properties": {
                        "field_paths": {"contains": {"enum": paths}},
                    }
                },
            }
        )

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["fact_id", "field_paths", "evidence_ids"],
        "properties": {
            "fact_id": {"const": fact_id},
            "field_paths": {
                "type": "array",
                "items": {"enum": supported_paths},
                "minItems": 1,
                "maxItems": min(32, len(supported_paths)),
                "uniqueItems": True,
            },
            "evidence_ids": {
                "type": "array",
                "items": {"enum": linked_evidence_ids},
                "minItems": 1,
                "maxItems": min(32, len(linked_evidence_ids)),
                "uniqueItems": True,
            },
        },
        "allOf": link_conditions,
    }


def _summary_output_schema(
    safe_input: Mapping[str, object],
) -> dict[str, object]:
    """Build a bounded schema containing only validated reference bindings."""

    facts_value = safe_input["facts"]
    evidence_value = safe_input["evidence"]
    uncertainties_value = safe_input["uncertainty_labels"]
    if (
        not isinstance(facts_value, list)
        or not isinstance(evidence_value, list)
        or not isinstance(uncertainties_value, list)
    ):
        raise WorkerInputError("Summary schema input is invalid.")
    complexity_units = len(uncertainties_value)
    for item in facts_value:
        if not isinstance(item, Mapping):
            raise WorkerInputError("Summary schema input is invalid.")
        fields = item.get("fields")
        evidence_ids = item.get("evidence_ids")
        if not isinstance(fields, list) or not isinstance(evidence_ids, list):
            raise WorkerInputError("Summary schema input is invalid.")
        complexity_units += 1 + len(fields) + len(evidence_ids) + (len(fields) * len(evidence_ids))
        if complexity_units > MAX_SUMMARY_SCHEMA_COMPLEXITY_UNITS:
            raise WorkerInputLimitError(
                "Summary output schema complexity exceeds its validated limit."
            )
    facts = {str(item["fact_id"]): item for item in facts_value if isinstance(item, Mapping)}
    evidence = {
        str(item["evidence_id"]): item for item in evidence_value if isinstance(item, Mapping)
    }
    claim_schemas = {fact_id: _claim_schema(fact, evidence) for fact_id, fact in facts.items()}
    facts_by_heading: dict[str, list[str]] = {"Overview": sorted(facts)}
    for fact_id, fact in facts.items():
        facts_by_heading.setdefault(str(fact["allowed_heading"]), []).append(fact_id)
    for fact_ids in facts_by_heading.values():
        fact_ids.sort()

    section_variants: list[dict[str, object]] = []
    for heading in sorted(facts_by_heading):
        fact_ids = facts_by_heading[heading]
        if not fact_ids:
            continue
        claims: dict[str, object] = {
            "type": "array",
            "items": {
                "oneOf": [claim_schemas[fact_id] for fact_id in fact_ids],
            },
            "maxItems": min(MAX_CLAIMS_PER_SECTION, len(fact_ids)),
            "uniqueItems": True,
            "allOf": [
                _contains_once(
                    {
                        "properties": {"fact_id": {"const": fact_id}},
                        "required": ["fact_id"],
                    }
                )
                for fact_id in fact_ids
            ],
        }
        section_variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["heading", "claims"],
                "properties": {
                    "heading": {"const": heading},
                    "claims": claims,
                },
            }
        )

    sections: dict[str, object] = {
        "type": "array",
        "maxItems": min(MAX_SECTIONS, len(section_variants)),
        "uniqueItems": True,
    }
    if section_variants:
        sections["items"] = {"oneOf": section_variants}
        sections["allOf"] = [
            _contains_once(
                {
                    "properties": {"heading": {"const": heading}},
                    "required": ["heading"],
                }
            )
            for heading in sorted(facts_by_heading)
            if facts_by_heading[heading]
        ]
        sections["allOf"].extend(  # type: ignore[union-attr]
            _contains_once(
                {
                    "properties": {
                        "claims": {
                            "contains": {
                                "properties": {
                                    "fact_id": {"const": fact_id},
                                },
                                "required": ["fact_id"],
                            }
                        }
                    },
                    "required": ["claims"],
                }
            )
            for fact_id in sorted(facts)
        )

    uncertainty_schemas: list[dict[str, object]] = []
    for item in sorted(
        uncertainties_value,
        key=lambda value: str(value["uncertainty_id"]) if isinstance(value, Mapping) else "",
    ):
        if not isinstance(item, Mapping):
            raise WorkerInputError("Summary schema input is invalid.")
        uncertainty_schemas.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["uncertainty_id", "fact_ids", "evidence_ids"],
                "properties": {
                    "uncertainty_id": {"const": item["uncertainty_id"]},
                    "fact_ids": {"const": item["fact_ids"]},
                    "evidence_ids": {"const": item["evidence_ids"]},
                },
            }
        )
    uncertainties: dict[str, object] = {
        "type": "array",
        "maxItems": min(MAX_UNCERTAINTIES, len(uncertainty_schemas)),
        "uniqueItems": True,
    }
    if uncertainty_schemas:
        uncertainties["items"] = {"oneOf": uncertainty_schemas}
        uncertainties["allOf"] = [
            _contains_once(
                {
                    "properties": {
                        "uncertainty_id": {"const": item["properties"]["uncertainty_id"]["const"]}
                    },
                    "required": ["uncertainty_id"],
                }
            )
            for item in uncertainty_schemas
        ]

    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sections", "uncertainties"],
        "properties": {
            "sections": sections,
            "uncertainties": uncertainties,
        },
    }
    try:
        bounded_json(schema, max_bytes=MAX_SUMMARY_SCHEMA_BYTES)
    except WorkerInputError:
        raise WorkerInputLimitError("Summary output schema exceeds its validated limit.") from None
    return schema


def run_summary(
    payload: Mapping[str, object],
    *,
    loaded: LoadedRole | None = None,
    generate_fn: Generate = generate_content,
) -> dict[str, object]:
    """Select typed references without allowing model-authored clinical prose."""

    safe_input = _validated_input(payload)
    validated_requested_output_tokens(payload, role_cap=SUMMARY_OUTPUT_CAP)
    json_schema = _summary_output_schema(safe_input)
    selected = loaded or load_role_from_payload("summary", payload)
    serialized = bounded_json(safe_input, max_bytes=MAX_SUMMARY_INPUT_BYTES)
    max_tokens = requested_output_tokens(payload, selected, role_cap=SUMMARY_OUTPUT_CAP)
    base_prompt = (
        "Return exactly one JSON object with keys sections and uncertainties. "
        "sections MUST be a JSON array of objects with exactly heading and claims; "
        "uncertainties MUST be a JSON array. Never emit clinical free text. "
        "A claim contains exactly fact_id, field_paths, and evidence_ids copied from "
        "mutually linked INPUT_JSON values. Each claim's field_paths MUST be a subset "
        "of BOTH the selected fact's fields.path values and every selected evidence's "
        "field_paths; omit /record_type unless evidence explicitly supports it. "
        "Treat the leaves of a typed observation value as one measurement and select "
        "only its supplied leaves. Use the selected fact's allowed_heading exactly as "
        "its section heading; use Overview only when grouping claims across record "
        "types. An uncertainty contains exactly uncertainty_id and its exact fact_ids "
        "and evidence_ids. Follow safety_rules exactly. "
    )
    prompt = f"{base_prompt}INPUT_JSON={serialized}"
    validate_token_budget(selected, [prompt], max_output_tokens=max_tokens)
    raw = generate_fn(
        model=selected.model,
        processor=selected.processor,
        prompt=prompt,
        images=[],
        max_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        input_token_limit=selected.decode_limits["max_input_tokens"],
        enable_thinking=False,
        json_schema=json_schema,
    )
    return _validated_output(raw, safe_input)
