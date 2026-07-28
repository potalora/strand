"""Fail-closed input and output contracts for grounded local summaries."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, NoReturn

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from app.services.local_ai.errors import LocalValidationError

MAX_FACTS = 512
MAX_EVIDENCE = 512
MAX_JSON_DEPTH = 12
MAX_JSON_NODES = 10_000
MAX_JSON_STRING = 4096
MAX_JSON_CHARACTERS = 50_000
MAX_SUMMARY_INPUT_BYTES = 4 * 1024 * 1024
MAX_RAW_OUTPUT_CHARACTERS = 100_000
MAX_RENDERED_CONTENT_CHARACTERS = 50_000
MAX_OBSERVATION_UNIT_CHARACTERS = 512

SERVER_MEDICAL_DISCLAIMER = (
    "> This summary organizes information from the supplied records. "
    "It is not medical advice, a diagnosis, or a treatment recommendation. "
    "Review it with a qualified healthcare professional."
)

SERVER_SAFETY_RULES = (
    "Select only server-supplied fact fields and their linked evidence.",
    "Do not emit free-text clinical claims, headings, or uncertainties.",
    "Do not create, correct, infer, or repair clinical facts.",
    "Do not provide diagnoses, treatment recommendations, medical advice, "
    "or clinical decision support.",
)

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

SummaryType = Literal["full_health", "category", "date_range", "single_record"]
SummaryCategory = Literal[
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
]
SummaryHeading = Literal[
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
]

_RECORD_TYPE_HEADINGS: dict[str, SummaryHeading] = {
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

UncertaintyTemplateId = Literal[
    "assertion_uncertain",
    "medication_dose_missing",
    "medication_end_date_missing",
    "record_date_missing",
    "record_status_missing",
    "result_missing",
    "source_detail_missing",
]

_UNCERTAINTY_LABELS: dict[str, str] = {
    "assertion_uncertain": "The source record marks this information as uncertain.",
    "medication_dose_missing": "Medication dose is not available.",
    "medication_end_date_missing": "Medication end date is not available.",
    "record_date_missing": "Record date is not available.",
    "record_status_missing": "Record status is not available.",
    "result_missing": "Result value is not available.",
    "source_detail_missing": "Source detail is not available.",
}

_GENERIC_SUMMARY_SCALAR_FIELDS = frozenset({"date", "name", "status"})
_SUMMARY_SCALAR_FIELDS: dict[str, frozenset[str]] = {
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
    "document": frozenset(
        {"author", "date", "document_type", "record_type", "status", "title"}
    ),
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
_SUMMARY_OBJECT_FIELDS: dict[str, dict[str, frozenset[str]]] = {
    "medication": {"dosage": frozenset({"amount", "unit"})},
}
_SUMMARY_SCALAR_LIST_FIELDS: dict[str, frozenset[str]] = {
    "care_plan": frozenset({"plan_items"}),
    "medication": frozenset({"statuses"}),
}
_SUMMARY_OBJECT_LIST_FIELDS: dict[str, dict[str, frozenset[str]]] = {
    "care_team": {
        "participants": frozenset({"name", "role"}),
    },
    "questionnaire_response": {
        "answers": frozenset({"answer", "question"}),
    },
}
_SUMMARY_DATE_FIELDS = ("date", "effective_date", "start", "end", "end_date")
_SUMMARY_STATUS_FIELDS = ("status", "statuses")
_ASSERTION_VALUES = frozenset(
    {
        "present",
        "negated",
        "family_history",
        "uncertain",
        "mentioned_not_performed",
    }
)
_STATUS_VALUES: dict[str, frozenset[str]] = {
    "allergy": frozenset({"active", "inactive", "resolved", "historical", "unknown"}),
    "appointment": frozenset(
        {
            "proposed",
            "pending",
            "booked",
            "arrived",
            "fulfilled",
            "cancelled",
            "noshow",
            "entered-in-error",
            "entered_in_error",
            "checked-in",
            "waitlist",
            "unknown",
        }
    ),
    "care_plan": frozenset(
        {
            "draft",
            "active",
            "on-hold",
            "on_hold",
            "revoked",
            "completed",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "care_team": frozenset(
        {
            "proposed",
            "active",
            "suspended",
            "inactive",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "communication": frozenset(
        {
            "preparation",
            "in-progress",
            "in_progress",
            "not-done",
            "not_done",
            "on-hold",
            "on_hold",
            "stopped",
            "completed",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "condition": frozenset(
        {
            "active",
            "recurrence",
            "relapse",
            "inactive",
            "remission",
            "resolved",
            "historical",
            "unknown",
        }
    ),
    "diagnostic_report": frozenset(
        {
            "registered",
            "partial",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "appended",
            "cancelled",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "document": frozenset(
        {"current", "superseded", "entered-in-error", "entered_in_error", "unknown"}
    ),
    "encounter": frozenset(
        {
            "planned",
            "arrived",
            "triaged",
            "in-progress",
            "in_progress",
            "onleave",
            "finished",
            "cancelled",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "imaging_study": frozenset(
        {
            "registered",
            "available",
            "cancelled",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "immunization": frozenset(
        {
            "completed",
            "entered-in-error",
            "entered_in_error",
            "not-done",
            "not_done",
            "unknown",
        }
    ),
    "medication": frozenset(
        {
            "active",
            "on-hold",
            "on_hold",
            "ended",
            "stopped",
            "completed",
            "cancelled",
            "entered-in-error",
            "entered_in_error",
            "draft",
            "historical",
            "unknown",
        }
    ),
    "observation": frozenset(
        {
            "registered",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "cancelled",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "procedure": frozenset(
        {
            "preparation",
            "in-progress",
            "in_progress",
            "not-done",
            "not_done",
            "on-hold",
            "on_hold",
            "stopped",
            "completed",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
    "questionnaire_response": frozenset(
        {
            "in-progress",
            "in_progress",
            "completed",
            "amended",
            "entered-in-error",
            "entered_in_error",
            "stopped",
            "unknown",
        }
    ),
    "service_request": frozenset(
        {
            "draft",
            "active",
            "on-hold",
            "on_hold",
            "revoked",
            "completed",
            "entered-in-error",
            "entered_in_error",
            "unknown",
        }
    ),
}
_ENUM_FIELD_VALUES: dict[tuple[str, str], frozenset[str]] = {
    **{
        (record_type, "assertion"): _ASSERTION_VALUES
        for record_type in ("allergy", "condition", "diagnostic_report", "procedure")
    },
    **{
        (record_type, "status"): values
        for record_type, values in _STATUS_VALUES.items()
    },
    ("allergy", "severity"): frozenset({"mild", "moderate", "severe", "unknown"}),
    ("diagnostic_report", "category"): frozenset(
        {
            "imaging",
            "laboratory",
            "pathology",
            "endoscopy",
            "nuclear_medicine",
            "pulmonary",
            "laboratory_panel",
            "other",
        }
    ),
    ("encounter", "visit_type"): frozenset(
        {"office", "telehealth", "emergency", "inpatient", "other"}
    ),
    ("service_request", "intent"): frozenset(
        {
            "proposal",
            "plan",
            "directive",
            "order",
            "original-order",
            "reflex-order",
            "filler-order",
            "instance-order",
            "option",
        }
    ),
    ("service_request", "priority"): frozenset({"routine", "urgent", "asap", "stat"}),
}
_CANONICAL_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_DATETIME_OFFSET_PATTERN = re.compile(r"[+-](\d{2}):(\d{2})$")
_POSITIVE_QUANTITY_PATTERN = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?: (?P<unit>[A-Za-zµμ%][A-Za-z0-9µμ%./^*-]{0,31}))?$"
)
_FINITE_MEASUREMENT_PATTERN = re.compile(
    r"^(?:<=|>=|[<>≤≥~])?"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?:(?: )?(?P<unit>%|[A-Za-zµμ][A-Za-z0-9µμ%./^*-]{0,31}))?$"
)
_COMPARATOR_MEASUREMENT_INPUT_PATTERN = re.compile(
    r"^\s*(?P<comparator><=|>=|[<>≤≥~])\s*"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+(?P<unit>%|[A-Za-zµμ][A-Za-z0-9µμ%./^*-]{0,31}))?\s*$"
)
_RATIO_MEASUREMENT_INPUT_PATTERN = re.compile(
    r"^\s*(?P<numerator>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*/\s*"
    r"(?P<denominator>[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+(?P<unit>%|[A-Za-zµμ][A-Za-z0-9µμ%./^*-]{0,31}))?\s*$"
)
_COMPARATOR_CANONICAL = {
    "<": "<",
    "<=": "<=",
    "≤": "<=",
    ">": ">",
    ">=": ">=",
    "≥": ">=",
    "~": "~",
}
_CLINICAL_SI_UNIT_ATOM = r"(?:(?:da|mc|[fpnuµμmcdhkMGT])?(?:g|L|m|s|A|K|mol|Eq|Osm))"
_CLINICAL_CONVENTIONAL_UNIT_ATOM = (
    r"(?:%|IU|U|mmHg|cmH2O|bpm|rpm|units?|tablets?|capsules?|puffs?|"
    r"drops?|min|h|hr|day|wk)"
)
_CLINICAL_UNIT_ATOM = (
    rf"(?:{_CLINICAL_SI_UNIT_ATOM}|{_CLINICAL_CONVENTIONAL_UNIT_ATOM})"
)
_CLINICAL_UNIT_SUFFIX_PATTERN = re.compile(
    rf"^{_CLINICAL_UNIT_ATOM}(?:\^?-?\d+)?"
    rf"(?:[/.*]{_CLINICAL_UNIT_ATOM}(?:\^?-?\d+)?)*$",
    re.IGNORECASE,
)
_ATTACHED_COMPOUND_UNIT_PATTERN = re.compile(
    r"^[A-Za-zµμ%][A-Za-z0-9µμ%^-]{0,15}"
    r"(?:[/.*][A-Za-zµμ%][A-Za-z0-9µμ%^-]{0,15})+$"
)
_ATTACHED_SINGLE_CLINICAL_LABEL_PATTERN = re.compile(
    rf"^(?:{_CLINICAL_UNIT_ATOM}(?:\^?-?\d+)?|[A-Z][A-Z0-9]{{1,7}})$"
)
_ATTACHED_CLINICAL_COUNT_LABELS = frozenset(
    {
        "beat",
        "beats",
        "breath",
        "breaths",
        "cell",
        "cells",
        "colony",
        "colonies",
        "copy",
        "copies",
        "count",
        "counts",
    }
)
_NONFINITE_NUMBER_WORDS = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)
_NUMERIC_UNIT_FIELD_PAIRS = (
    ("/value", "/unit"),
    ("/dose_value", "/dose_unit"),
    ("/dosage/amount", "/dosage/unit"),
)
_POSITIVE_STATUS_VALUES_FOR_ASSERTION: dict[str, frozenset[str]] = {
    "allergy": frozenset({"active", "inactive", "resolved", "historical"}),
    "condition": frozenset(
        {
            "active",
            "recurrence",
            "relapse",
            "inactive",
            "remission",
            "resolved",
            "historical",
        }
    ),
    "diagnostic_report": frozenset(
        {
            "registered",
            "partial",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "appended",
        }
    ),
    "procedure": frozenset(
        {
            "preparation",
            "in-progress",
            "in_progress",
            "on-hold",
            "on_hold",
            "stopped",
            "completed",
        }
    ),
}


def _contains_render_control(value: str, *, allow_line_breaks: bool = False) -> bool:
    """Return whether text contains invisible or layout-controlling Unicode."""
    for character in value:
        category = unicodedata.category(character)
        if category in {"Cf", "Cs", "Zl", "Zp"}:
            return True
        if category == "Cc" and not (
            allow_line_breaks and character in {"\t", "\n", "\r"}
        ):
            return True
    return False


def _safe_single_line(value: str) -> str:
    """Reject blank, padded, multiline, or control-containing text."""
    if not value.strip() or value != value.strip():
        raise ValueError("must be nonblank text without surrounding whitespace")
    if "\n" in value or "\r" in value:
        raise ValueError("must be one line")
    if _contains_render_control(value):
        raise ValueError("contains an unsafe control character")
    return value


def _safe_excerpt(value: str) -> str:
    """Reject blank or invisible-control-containing evidence."""
    if not value.strip() or value != value.strip():
        raise ValueError("must be nonblank text without surrounding whitespace")
    if _contains_render_control(value, allow_line_breaks=True):
        raise ValueError("contains an unsafe control character")
    return value


def _safe_iso_date(value: str) -> str:
    """Require a canonical ISO calendar date without accepting datetime text."""
    _safe_single_line(value)
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("must be an ISO calendar date") from None
    if parsed.isoformat() != value:
        raise ValueError("must be a canonical ISO calendar date")
    return value


Identifier = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
]
RecordIdentifier = Annotated[
    StrictStr,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
]
FactPath = Annotated[
    StrictStr,
    StringConstraints(min_length=2, max_length=1024, pattern=r"^/"),
    AfterValidator(_safe_single_line),
]
SafeLine = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=2000),
    AfterValidator(_safe_single_line),
]
EvidenceExcerpt = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=2000),
    AfterValidator(_safe_excerpt),
]
ISODate = Annotated[
    StrictStr,
    StringConstraints(min_length=10, max_length=10),
    AfterValidator(_safe_iso_date),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class GroundedSummaryScope(_StrictModel):
    """Allowlisted summary request metadata safe to send to a local worker."""

    summary_type: SummaryType
    category: SummaryCategory | None = None
    date_from: ISODate | None = None
    date_to: ISODate | None = None
    record_ids: Annotated[
        tuple[RecordIdentifier, ...],
        Field(max_length=100),
    ] = ()

    @model_validator(mode="after")
    def _filters_match_summary_type(self) -> GroundedSummaryScope:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("duplicate record identifier")
        if self.summary_type == "category":
            if self.category is None:
                raise ValueError("category summaries require a category")
        elif self.category is not None:
            raise ValueError("category is only valid for category summaries")
        if self.summary_type == "date_range":
            if self.date_from is None or self.date_to is None:
                raise ValueError("date range summaries require both dates")
            if self.date_from > self.date_to:
                raise ValueError("date range is reversed")
        elif self.date_from is not None or self.date_to is not None:
            raise ValueError("date filters are only valid for date range summaries")
        if self.summary_type == "single_record":
            if len(self.record_ids) != 1:
                raise ValueError("single record summaries require one record")
        elif self.record_ids:
            raise ValueError("record ids are only valid for single record summaries")
        return self


class GroundedFactField(_StrictModel):
    """One immutable, server-validated leaf in a clinical fact."""

    path: FactPath
    value_json: Annotated[
        StrictStr,
        StringConstraints(min_length=1, max_length=MAX_JSON_STRING + 32),
        AfterValidator(_safe_single_line),
    ]

    @field_validator("value_json")
    @classmethod
    def _value_is_canonical_plain_json(cls, value: str) -> str:
        try:
            parsed = json.loads(
                value,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("fact field value is not strict JSON") from None
        if (
            type(parsed) not in {str, int, float, bool, list, dict}
            and parsed is not None
        ):
            raise ValueError("fact field value is not plain JSON")
        if type(parsed) in {list, dict} and parsed:
            raise ValueError("fact fields can contain only scalar or empty values")
        _validate_bounded_json(parsed)
        if _canonical_json_value(parsed) != value:
            raise ValueError("fact field value is not canonical JSON")
        return value


class GroundedSummaryFact(_StrictModel):
    """Immutable canonical clinical fact sent to and returned from the worker."""

    fact_id: Identifier
    record_id: RecordIdentifier
    content_json: Annotated[
        StrictStr,
        StringConstraints(min_length=2, max_length=MAX_JSON_CHARACTERS),
        AfterValidator(_safe_single_line),
    ]
    fields: Annotated[
        tuple[GroundedFactField, ...],
        Field(min_length=1, max_length=MAX_JSON_NODES),
    ]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _snapshot_is_canonical_and_bound(self) -> GroundedSummaryFact:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        if len(self.fields) != len({field.path for field in self.fields}):
            raise ValueError("duplicate fact field path")
        try:
            content = _parse_canonical_content(self.content_json)
            _validate_safe_summary_content(content)
        except ValueError:
            raise ValueError("fact content snapshot is invalid") from None
        expected_fields = _flatten_content(content)
        if self.fields != expected_fields:
            raise ValueError("fact fields do not match the canonical snapshot")
        if self.fact_id != _stable_fact_id(
            self.record_id,
            self.content_json,
            self.evidence_ids,
        ):
            raise ValueError("fact identifier does not match its content")
        return self


class GroundedSummaryEvidence(_StrictModel):
    """Immutable evidence excerpt and server-derived reverse fact links."""

    evidence_id: Identifier
    source_id: Identifier
    excerpt: EvidenceExcerpt
    page_number: Annotated[StrictInt, Field(ge=1, le=100_000)] | None = None
    section: SafeLine | None = None
    fact_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]
    field_paths: Annotated[
        tuple[FactPath, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _snapshot_is_content_bound(self) -> GroundedSummaryEvidence:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("duplicate fact identifier")
        if len(self.field_paths) != len(set(self.field_paths)):
            raise ValueError("duplicate fact field path")
        expected = _stable_evidence_id(
            source_id=self.source_id,
            excerpt=self.excerpt,
            page_number=self.page_number,
            section=self.section,
            field_paths=self.field_paths,
        )
        if self.evidence_id != expected:
            raise ValueError("evidence identifier does not match its content")
        return self


class GroundedSummaryUncertainty(_StrictModel):
    """Server-owned uncertainty label with immutable clinical support."""

    uncertainty_id: Identifier
    template_id: UncertaintyTemplateId
    label: SafeLine
    fact_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _snapshot_is_content_bound(self) -> GroundedSummaryUncertainty:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("duplicate fact identifier")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        if self.label != _UNCERTAINTY_LABELS[self.template_id]:
            raise ValueError("uncertainty label is not server-owned")
        expected = _stable_uncertainty_id(
            self.template_id,
            self.fact_ids,
            self.evidence_ids,
        )
        if self.uncertainty_id != expected:
            raise ValueError("uncertainty identifier does not match its content")
        return self


class GroundedClaim(_StrictModel):
    """A model-selected fact and exact validated fields to render."""

    fact_id: Identifier
    field_paths: Annotated[
        tuple[FactPath, ...],
        Field(min_length=1, max_length=32),
    ]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _support_is_unique(self) -> GroundedClaim:
        if len(self.field_paths) != len(set(self.field_paths)):
            raise ValueError("duplicate fact field path")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        return self


class GroundedSection(_StrictModel):
    """A server-headed group of grounded fact selections."""

    heading: SummaryHeading
    claims: Annotated[
        tuple[GroundedClaim, ...],
        Field(max_length=100),
    ]


class GroundedUncertaintyReference(_StrictModel):
    """A model-selected server-owned uncertainty and its exact support."""

    uncertainty_id: Identifier
    fact_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _support_is_unique(self) -> GroundedUncertaintyReference:
        if len(self.fact_ids) != len(set(self.fact_ids)):
            raise ValueError("duplicate fact identifier")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        return self


class GroundedSummaryDocument(_StrictModel):
    """Typed model output containing references but no clinical free text."""

    sections: Annotated[
        tuple[GroundedSection, ...],
        Field(max_length=20),
    ]
    uncertainties: Annotated[
        tuple[GroundedUncertaintyReference, ...],
        Field(max_length=100),
    ]

    @model_validator(mode="after")
    def _references_are_unique(self) -> GroundedSummaryDocument:
        headings = [section.heading for section in self.sections]
        if len(headings) != len(set(headings)):
            raise ValueError("duplicate section heading")
        fact_ids = [
            claim.fact_id for section in self.sections for claim in section.claims
        ]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("duplicate fact claim")
        uncertainty_ids = [item.uncertainty_id for item in self.uncertainties]
        if len(uncertainty_ids) != len(set(uncertainty_ids)):
            raise ValueError("duplicate uncertainty identifier")
        return self


class RenderedGroundedSummary(_StrictModel):
    """Validated typed output paired with deterministic server Markdown."""

    selection_document: GroundedSummaryDocument
    document: GroundedSummaryDocument
    markdown: StrictStr


class GroundedSummaryInput(_StrictModel):
    """Server-owned bounded payload sent to the local summary worker."""

    requested_scope: GroundedSummaryScope
    facts: Annotated[
        tuple[GroundedSummaryFact, ...],
        Field(max_length=MAX_FACTS),
    ]
    evidence: Annotated[
        tuple[GroundedSummaryEvidence, ...],
        Field(max_length=MAX_EVIDENCE),
    ]
    uncertainty_labels: Annotated[
        tuple[GroundedSummaryUncertainty, ...],
        Field(max_length=100),
    ] = ()
    safety_rules: Annotated[
        tuple[StrictStr, ...],
        Field(
            min_length=len(SERVER_SAFETY_RULES),
            max_length=len(SERVER_SAFETY_RULES),
        ),
    ] = SERVER_SAFETY_RULES

    @model_validator(mode="after")
    def _registry_is_consistent(self) -> GroundedSummaryInput:
        if self.safety_rules != SERVER_SAFETY_RULES:
            raise ValueError("safety rules are server-owned")
        fact_registry = {fact.fact_id: fact for fact in self.facts}
        evidence_registry = {item.evidence_id: item for item in self.evidence}
        uncertainty_registry = {
            item.uncertainty_id: item for item in self.uncertainty_labels
        }
        if len(fact_registry) != len(self.facts):
            raise ValueError("duplicate fact identifier")
        if len(evidence_registry) != len(self.evidence):
            raise ValueError("duplicate evidence identifier")
        if len(uncertainty_registry) != len(self.uncertainty_labels):
            raise ValueError("duplicate uncertainty identifier")
        _validate_support_registry(
            fact_registry,
            evidence_registry,
            uncertainty_registry,
        )
        try:
            _validate_grounded_fact_scope(self.requested_scope, self.facts)
        except (RecursionError, TypeError, ValueError):
            raise ValueError("summary facts do not match requested scope") from None
        try:
            serialized = json.dumps(
                self.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (OverflowError, RecursionError, TypeError, ValueError):
            raise ValueError("summary input has invalid structure") from None
        if len(serialized) > MAX_SUMMARY_INPUT_BYTES:
            raise ValueError("summary input exceeds the size limit")
        return self


def _validate_summary_text(value: object) -> None:
    if type(value) is not str:
        raise ValueError("summary text field has an invalid type")
    _safe_single_line(value)


def _validate_positive_summary_quantity(value: object) -> None:
    if type(value) is str:
        _safe_single_line(value)
        match = _POSITIVE_QUANTITY_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                "summary quantity does not use canonical numeric-unit text"
            )
        parsed = Decimal(match.group("number"))
    elif type(value) in {int, float}:
        parsed = Decimal(str(value))
    else:
        raise ValueError("summary quantity has an invalid type")
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError("summary quantity must be positive")


def _looks_numeric_summary_text(value: str) -> bool:
    folded = value.casefold()
    return (
        folded in _NONFINITE_NUMBER_WORDS
        or _has_nonfinite_clinical_quantity(value)
        or value.startswith(("<", ">", "≤", "≥", "~"))
        or bool(value)
        and value[0] in "+-.0123456789"
    )


def _has_nonfinite_clinical_quantity(value: str) -> bool:
    unsigned = value[1:] if value.startswith(("+", "-")) else value
    folded = unsigned.casefold()
    for token in ("infinity", "nan", "inf"):
        if folded.startswith(token):
            raw_suffix = unsigned[len(token) :]
            if raw_suffix[:1].isspace():
                return bool(raw_suffix.strip())
            suffix = raw_suffix.strip()
            return (
                not suffix
                or _CLINICAL_UNIT_SUFFIX_PATTERN.fullmatch(suffix) is not None
                or _ATTACHED_COMPOUND_UNIT_PATTERN.fullmatch(suffix) is not None
                or _ATTACHED_SINGLE_CLINICAL_LABEL_PATTERN.fullmatch(suffix) is not None
                or suffix.casefold() in _ATTACHED_CLINICAL_COUNT_LABELS
            )
    return False


def _validate_finite_summary_measurement(value: str) -> None:
    _safe_single_line(value)
    if value.casefold() in _NONFINITE_NUMBER_WORDS:
        raise ValueError("summary measurement must be finite")
    match = _FINITE_MEASUREMENT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("summary measurement does not use canonical numeric-unit text")
    try:
        parsed = Decimal(match.group("number"))
    except InvalidOperation:
        raise ValueError("summary measurement is not numeric") from None
    if not parsed.is_finite():
        raise ValueError("summary measurement must be finite")


def _json_number(value: object) -> int | float:
    """Return a finite plain-JSON number without accepting booleans."""
    if type(value) in {int, float}:
        parsed = Decimal(str(value))
    elif type(value) is str:
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            raise ValueError("summary measurement is not numeric") from None
    else:
        raise ValueError("summary measurement is not numeric")
    if not parsed.is_finite():
        raise ValueError("summary measurement must be finite")
    if parsed == parsed.to_integral_value():
        return int(parsed)
    result = float(parsed)
    if not math.isfinite(result):
        raise ValueError("summary measurement must be finite")
    return result


def typed_observation_quantity(
    number: object,
    comparator: object,
    unit: object,
) -> dict[str, object]:
    """Build the canonical typed value for a comparator observation."""
    if type(comparator) is not str or comparator not in _COMPARATOR_CANONICAL:
        raise ValueError("summary comparator is invalid")
    if type(unit) is not str:
        raise ValueError("summary measurement unit is required")
    canonical_unit = _safe_single_line(unit.strip())
    if len(canonical_unit) > MAX_OBSERVATION_UNIT_CHARACTERS:
        raise ValueError("summary measurement unit is too long")
    return {
        "kind": "quantity",
        "comparator": _COMPARATOR_CANONICAL[comparator],
        "number": _json_number(number),
        "unit": canonical_unit,
    }


def typed_observation_ratio(
    numerator: object,
    denominator: object,
    numerator_unit: object,
    denominator_unit: object,
) -> dict[str, object]:
    """Build the canonical typed value for a ratio observation."""
    if type(numerator_unit) is not str or type(denominator_unit) is not str:
        raise ValueError("summary ratio units are required")
    canonical_numerator = _json_number(numerator)
    canonical_denominator = _json_number(denominator)
    if canonical_denominator == 0:
        raise ValueError("summary ratio denominator cannot be zero")
    canonical_numerator_unit = _safe_single_line(numerator_unit.strip())
    canonical_denominator_unit = _safe_single_line(denominator_unit.strip())
    if (
        len(canonical_numerator_unit) > MAX_OBSERVATION_UNIT_CHARACTERS
        or len(canonical_denominator_unit) > MAX_OBSERVATION_UNIT_CHARACTERS
    ):
        raise ValueError("summary ratio unit is too long")
    return {
        "kind": "ratio",
        "numerator": canonical_numerator,
        "denominator": canonical_denominator,
        "numerator_unit": canonical_numerator_unit,
        "denominator_unit": canonical_denominator_unit,
    }


def normalize_observation_summary_value(
    value: object,
    unit: object,
) -> tuple[object, object | None]:
    """Convert comparator and ratio strings to immutable typed measurements."""
    if type(value) is not str:
        return value, unit
    comparator_match = _COMPARATOR_MEASUREMENT_INPUT_PATTERN.fullmatch(value)
    if comparator_match is not None:
        inline_unit = comparator_match.group("unit")
        selected_unit = unit or inline_unit
        if unit and inline_unit and _safe_single_line(str(unit).strip()) != inline_unit:
            raise ValueError("summary measurement units disagree")
        return (
            typed_observation_quantity(
                comparator_match.group("number"),
                comparator_match.group("comparator"),
                selected_unit,
            ),
            None,
        )
    ratio_match = _RATIO_MEASUREMENT_INPUT_PATTERN.fullmatch(value)
    if ratio_match is not None:
        inline_unit = ratio_match.group("unit")
        selected_unit = unit or inline_unit
        if unit and inline_unit and _safe_single_line(str(unit).strip()) != inline_unit:
            raise ValueError("summary measurement units disagree")
        return (
            typed_observation_ratio(
                ratio_match.group("numerator"),
                ratio_match.group("denominator"),
                selected_unit,
                selected_unit,
            ),
            None,
        )
    return value, unit


def _validate_typed_observation_value(value: Mapping[str, object]) -> None:
    kind = value.get("kind")
    if kind == "quantity":
        if set(value) != {"kind", "comparator", "number", "unit"}:
            raise ValueError("summary comparator quantity has an invalid shape")
        canonical = typed_observation_quantity(
            value["number"],
            value["comparator"],
            value["unit"],
        )
    elif kind == "ratio":
        if set(value) != {
            "kind",
            "numerator",
            "denominator",
            "numerator_unit",
            "denominator_unit",
        }:
            raise ValueError("summary ratio has an invalid shape")
        canonical = typed_observation_ratio(
            value["numerator"],
            value["denominator"],
            value["numerator_unit"],
            value["denominator_unit"],
        )
    else:
        raise ValueError("summary typed observation value has an invalid kind")
    if dict(value) != canonical:
        raise ValueError("summary typed observation value is not canonical")


def _validate_summary_scalar(
    record_type: str | None,
    field_name: str,
    value: object,
) -> None:
    if field_name == "record_type":
        if value != record_type:
            raise ValueError("summary record type is inconsistent")
        return
    if field_name in _SUMMARY_DATE_FIELDS:
        if type(value) is not str:
            raise ValueError("summary date field has an invalid type")
        _parse_canonical_summary_fact_date(value)
        return
    enum_values = (
        _ENUM_FIELD_VALUES.get((record_type, field_name))
        if record_type is not None
        else None
    )
    if enum_values is not None:
        if type(value) is not str or value not in enum_values:
            raise ValueError("summary enum field has an invalid value")
        return
    if record_type == "medication" and field_name == "dose_value":
        _validate_positive_summary_quantity(value)
        return
    if field_name == "dose" and record_type in {"immunization", "medication"}:
        if type(value) is str:
            if _looks_numeric_summary_text(value):
                _validate_positive_summary_quantity(value)
            else:
                _validate_summary_text(value)
        else:
            _validate_positive_summary_quantity(value)
        return
    if record_type == "observation" and field_name == "value":
        if isinstance(value, Mapping):
            _validate_typed_observation_value(value)
        elif type(value) is str:
            if _looks_numeric_summary_text(value):
                _validate_finite_summary_measurement(value)
            else:
                _validate_summary_text(value)
        elif type(value) not in {bool, int, float}:
            raise ValueError("summary observation value has an invalid type")
        return
    _validate_summary_text(value)


def _validate_safe_summary_content(value: dict[str, object]) -> None:
    """Require a typed, record-specific projection rather than arbitrary JSON."""
    record_type = value.get("record_type")
    if record_type is None:
        scalar_fields = _GENERIC_SUMMARY_SCALAR_FIELDS
        object_fields: dict[str, frozenset[str]] = {}
        scalar_list_fields = frozenset()
        object_list_fields: dict[str, frozenset[str]] = {}
    else:
        if type(record_type) is not str or record_type not in _SUMMARY_SCALAR_FIELDS:
            raise ValueError("record type is not supported for summaries")
        scalar_fields = _SUMMARY_SCALAR_FIELDS[record_type]
        object_fields = _SUMMARY_OBJECT_FIELDS.get(record_type, {})
        scalar_list_fields = _SUMMARY_SCALAR_LIST_FIELDS.get(
            record_type,
            frozenset(),
        )
        object_list_fields = _SUMMARY_OBJECT_LIST_FIELDS.get(record_type, {})

    for key, item in value.items():
        if key in scalar_fields:
            _validate_summary_scalar(record_type, key, item)
            continue
        nested_fields = object_fields.get(key)
        if nested_fields is not None:
            if type(item) is not dict or set(item) - nested_fields:
                raise ValueError("summary object field has an invalid shape")
            for nested_key, child in item.items():
                if (
                    record_type == "medication"
                    and key == "dosage"
                    and nested_key == "amount"
                ):
                    _validate_positive_summary_quantity(child)
                else:
                    _validate_summary_text(child)
            continue
        if key in scalar_list_fields:
            if type(item) is not list:
                raise ValueError("summary list field has an invalid shape")
            for child in item:
                if record_type == "medication" and key == "statuses":
                    _validate_summary_scalar(record_type, "status", child)
                else:
                    _validate_summary_text(child)
            continue
        item_fields = object_list_fields.get(key)
        if item_fields is not None:
            if type(item) is not list:
                raise ValueError("summary object-list field has an invalid shape")
            for child in item:
                if type(child) is not dict or not child or set(child) - item_fields:
                    raise ValueError("summary object-list field has an invalid value")
                for nested_key, nested in child.items():
                    if (
                        record_type == "questionnaire_response"
                        and key == "answers"
                        and nested_key == "answer"
                    ):
                        if type(nested) is str:
                            _validate_summary_text(nested)
                        elif type(nested) not in {bool, int, float}:
                            raise ValueError(
                                "summary questionnaire answer has an invalid type"
                            )
                    else:
                        _validate_summary_text(nested)
            continue
        raise ValueError("summary content contains a non-allowlisted field")
    _validate_cross_field_summary_content(value, record_type)


def _parse_canonical_summary_fact_date(raw: str) -> date:
    if len(raw) == 10:
        try:
            parsed_date = date.fromisoformat(raw)
        except ValueError:
            raise ValueError("summary fact date is malformed") from None
        if parsed_date.isoformat() != raw:
            raise ValueError("summary fact date is not canonical")
        return parsed_date
    if _CANONICAL_DATETIME_PATTERN.fullmatch(raw) is None:
        raise ValueError("summary fact date-time is malformed")
    offset = _DATETIME_OFFSET_PATTERN.search(raw)
    if offset is not None and (
        int(offset.group(1)) >= 24 or int(offset.group(2)) >= 60
    ):
        raise ValueError("summary fact date-time offset is not canonical")
    try:
        parsed_datetime = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("summary fact date-time is malformed") from None
    return parsed_datetime.date()


def _parse_canonical_summary_temporal(raw: str) -> datetime:
    """Return a canonical instant for deterministic cross-field chronology."""
    if len(raw) == 10:
        parsed_date = _parse_canonical_summary_fact_date(raw)
        return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    _parse_canonical_summary_fact_date(raw)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_cross_field_summary_content(
    value: Mapping[str, object],
    record_type: str | None,
) -> None:
    if record_type == "appointment" and "start" in value and "end" in value:
        start = _parse_canonical_summary_temporal(value["start"])
        end = _parse_canonical_summary_temporal(value["end"])
        if start > end:
            raise ValueError("appointment start is after end")

    if record_type == "medication" and "end_date" in value:
        end = _parse_canonical_summary_temporal(value["end_date"])
        for start_field in ("effective_date", "date"):
            if start_field in value:
                start = _parse_canonical_summary_temporal(value[start_field])
                if start > end:
                    raise ValueError("medication start is after end")

    assertion = value.get("assertion")
    status = value.get("status")
    if (
        type(assertion) is str
        and type(status) is str
        and assertion in {"negated", "mentioned_not_performed"}
        and status
        in _POSITIVE_STATUS_VALUES_FOR_ASSERTION.get(
            record_type or "",
            frozenset(),
        )
    ):
        raise ValueError("assertion contradicts lifecycle status")
    if assertion == "family_history":
        if record_type != "condition":
            raise ValueError("family-history assertion is invalid for this record type")
        if not _has_recorded_value(value.get("relationship")):
            raise ValueError("family-history condition requires a relationship")
        if status in {"active", "recurrence", "relapse"}:
            raise ValueError("family-history assertion contradicts patient status")
    if (
        record_type == "procedure"
        and assertion == "present"
        and status in {"not-done", "not_done", "entered-in-error", "entered_in_error"}
    ):
        raise ValueError("present procedure contradicts lifecycle status")


def _parse_summary_fact_dates(value: Mapping[str, JsonValue]) -> tuple[date, ...]:
    """Parse all dates in the explicit ``_SUMMARY_DATE_FIELDS`` precedence."""
    parsed_dates: list[date] = []
    for field_name in _SUMMARY_DATE_FIELDS:
        if field_name not in value or value[field_name] is None:
            continue
        raw = value.get(field_name)
        if type(raw) is not str:
            raise ValueError("summary fact date has an invalid type")
        parsed_dates.append(_parse_canonical_summary_fact_date(raw))
    return tuple(parsed_dates)


def _validate_fact_scope(
    scope: GroundedSummaryScope,
    facts: Sequence[_FactCandidate],
) -> None:
    _validate_scope_records(
        scope,
        tuple((fact.record_id, fact.content) for fact in facts),
    )


def _validate_grounded_fact_scope(
    scope: GroundedSummaryScope,
    facts: Sequence[GroundedSummaryFact],
) -> None:
    _validate_scope_records(
        scope,
        tuple(
            (fact.record_id, _parse_canonical_content(fact.content_json))
            for fact in facts
        ),
    )


def _validate_scope_records(
    scope: GroundedSummaryScope,
    facts: Sequence[tuple[str, Mapping[str, JsonValue]]],
) -> None:
    if scope.summary_type == "single_record":
        expected_record_id = scope.record_ids[0]
        if len(facts) != 1 or facts[0][0] != expected_record_id:
            raise ValueError("single-record scope contains another record")
    elif scope.summary_type == "category":
        if any(content.get("record_type") != scope.category for _, content in facts):
            raise ValueError("category scope contains another record type")
    elif scope.summary_type == "date_range":
        date_from = date.fromisoformat(scope.date_from)
        date_to = date.fromisoformat(scope.date_to)
        for _, content in facts:
            fact_dates = _parse_summary_fact_dates(content)
            if not fact_dates or any(
                not date_from <= fact_date <= date_to for fact_date in fact_dates
            ):
                raise ValueError("date scope contains an out-of-range fact")


class _FactCandidate(_StrictModel):
    """Caller fact before the server creates its immutable snapshot."""

    record_id: RecordIdentifier
    content: dict[str, JsonValue]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @field_validator("content", mode="before")
    @classmethod
    def _content_is_a_plain_object(cls, value: object) -> object:
        if type(value) is not dict or not value:
            raise ValueError("must be a nonempty JSON object")
        _validate_bounded_json(value, forbidden_fields=_FORBIDDEN_FACT_FIELDS)
        _validate_safe_summary_content(value)
        return value

    @model_validator(mode="after")
    def _evidence_ids_are_unique(self) -> _FactCandidate:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        return self


class _EvidenceCandidate(_StrictModel):
    """Caller evidence before stable ids and reverse links are derived."""

    id: Identifier
    excerpt: EvidenceExcerpt
    page_number: Annotated[StrictInt, Field(ge=1, le=100_000)] | None = None
    section: SafeLine | None = None
    field_paths: Annotated[
        tuple[FactPath, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _field_paths_are_unique(self) -> _EvidenceCandidate:
        if len(self.field_paths) != len(set(self.field_paths)):
            raise ValueError("duplicate fact field path")
        return self


class _UncertaintyCandidate(_StrictModel):
    """Caller uncertainty before record and source evidence ids are resolved."""

    template_id: UncertaintyTemplateId
    record_ids: Annotated[
        tuple[RecordIdentifier, ...],
        Field(min_length=1, max_length=32),
    ]
    evidence_ids: Annotated[
        tuple[Identifier, ...],
        Field(min_length=1, max_length=32),
    ]

    @model_validator(mode="after")
    def _support_is_unique(self) -> _UncertaintyCandidate:
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("duplicate record identifier")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence identifier")
        return self


def _has_recorded_value(value: object) -> bool:
    """Return whether a validated fact field contains a nonempty value."""
    if value is None:
        return False
    if type(value) is str:
        return bool(value.strip())
    if type(value) in {list, dict}:
        return bool(value)
    return True


def _uncertainty_template_is_applicable(
    template_id: UncertaintyTemplateId,
    *,
    facts: Sequence[Mapping[str, JsonValue]],
    evidence_has_source_detail: Sequence[bool],
) -> bool:
    """Check that a server-owned missing/uncertain label matches its support."""
    if template_id == "assertion_uncertain":
        return all(fact.get("assertion") == "uncertain" for fact in facts)
    if template_id == "medication_dose_missing":
        return all(
            fact.get("record_type") == "medication"
            and not any(
                _has_recorded_value(value)
                for value in (
                    fact.get("dose"),
                    fact.get("dose_value"),
                    (
                        fact.get("dosage", {}).get("amount")
                        if type(fact.get("dosage")) is dict
                        else None
                    ),
                )
            )
            for fact in facts
        )
    if template_id == "medication_end_date_missing":
        return all(
            fact.get("record_type") == "medication"
            and not _has_recorded_value(fact.get("end_date"))
            for fact in facts
        )
    if template_id == "record_date_missing":
        return all(
            not any(
                _has_recorded_value(fact.get(field)) for field in _SUMMARY_DATE_FIELDS
            )
            for fact in facts
        )
    if template_id == "record_status_missing":
        return all(
            not any(
                _has_recorded_value(fact.get(field)) for field in _SUMMARY_STATUS_FIELDS
            )
            for fact in facts
        )
    if template_id == "result_missing":
        return all(
            (
                fact.get("record_type") == "observation"
                and not _has_recorded_value(fact.get("value"))
            )
            or (
                fact.get("record_type") == "diagnostic_report"
                and not _has_recorded_value(fact.get("findings"))
            )
            for fact in facts
        )
    if template_id == "source_detail_missing":
        return not any(evidence_has_source_detail)
    return False


def _uncertainty_is_applicable(
    candidate: _UncertaintyCandidate,
    *,
    facts_by_record_id: Mapping[str, _FactCandidate],
    evidence_by_source_id: Mapping[str, _EvidenceCandidate],
) -> bool:
    return _uncertainty_template_is_applicable(
        candidate.template_id,
        facts=tuple(
            facts_by_record_id[record_id].content for record_id in candidate.record_ids
        ),
        evidence_has_source_detail=tuple(
            evidence_by_source_id[evidence_id].page_number is not None
            or evidence_by_source_id[evidence_id].section is not None
            for evidence_id in candidate.evidence_ids
        ),
    )


def _validate_bounded_json(
    value: object,
    *,
    forbidden_fields: frozenset[str] = frozenset(),
) -> None:
    """Require bounded, finite, acyclic plain JSON without render controls."""
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            raise ValueError("JSON value exceeds structural limits")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float:
            if item != item or item in {float("inf"), float("-inf")}:
                raise ValueError("JSON number must be finite")
            return
        if type(item) is str:
            if len(item) > MAX_JSON_STRING:
                raise ValueError("JSON string exceeds the limit")
            if _contains_render_control(item):
                raise ValueError("JSON string contains an unsafe control character")
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
                ):
                    raise ValueError("JSON object key is invalid")
                if key.casefold() in forbidden_fields:
                    raise ValueError("JSON object contains a forbidden field")
                visit(child, depth + 1)
            return
        raise ValueError("value is not plain JSON")

    try:
        visit(value, 0)
        encoded = _canonical_json_value(value)
    except RecursionError:
        raise ValueError("JSON value exceeds structural limits") from None
    if len(encoded) > MAX_JSON_CHARACTERS:
        raise ValueError("JSON value exceeds the encoded size limit")


def _canonical_json_value(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        raise ValueError("JSON value cannot be encoded safely") from None


def _parse_canonical_content(value: str) -> dict[str, JsonValue]:
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("fact content is not strict JSON") from None
    if type(parsed) is not dict or not parsed:
        raise ValueError("fact content must be a nonempty object")
    _validate_bounded_json(parsed, forbidden_fields=_FORBIDDEN_FACT_FIELDS)
    if _canonical_json_value(parsed) != value:
        raise ValueError("fact content is not canonical JSON")
    return parsed


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _flatten_content(content: Mapping[str, JsonValue]) -> tuple[GroundedFactField, ...]:
    fields: list[GroundedFactField] = []

    def visit(value: JsonValue, path: str) -> None:
        if type(value) is dict and value:
            for key in sorted(value):
                visit(value[key], f"{path}/{_escape_pointer_token(key)}")
            return
        if type(value) is list and value:
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")
            return
        fields.append(
            GroundedFactField(
                path=path,
                value_json=_canonical_json_value(value),
            )
        )

    for key in sorted(content):
        visit(content[key], f"/{_escape_pointer_token(key)}")
    if not fields:
        raise ValueError("fact content has no fields")
    return tuple(fields)


def _stable_fact_id(
    record_id: str,
    content_json: str,
    evidence_ids: Sequence[str],
) -> str:
    payload = _canonical_json_value(
        {
            "content_json": content_json,
            "evidence_ids": sorted(evidence_ids),
            "record_id": record_id,
        }
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"fact1_{digest[:40]}"


def _stable_evidence_id(
    *,
    source_id: str,
    excerpt: str,
    page_number: int | None,
    section: str | None,
    field_paths: Sequence[str],
) -> str:
    payload = _canonical_json_value(
        {
            "excerpt": excerpt,
            "field_paths": sorted(field_paths),
            "page_number": page_number,
            "section": section,
            "source_id": source_id,
        }
    )
    return f"evidence1_{hashlib.sha256(payload.encode()).hexdigest()[:40]}"


def _stable_uncertainty_id(
    template_id: str,
    fact_ids: Sequence[str],
    evidence_ids: Sequence[str],
) -> str:
    payload = _canonical_json_value(
        {
            "evidence_ids": sorted(evidence_ids),
            "fact_ids": sorted(fact_ids),
            "template_id": template_id,
        }
    )
    return f"uncertainty1_{hashlib.sha256(payload.encode()).hexdigest()[:40]}"


def build_grounded_summary_input(
    *,
    facts: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
    requested_scope: Mapping[str, object],
    uncertainty_labels: Sequence[Mapping[str, object]] = (),
) -> GroundedSummaryInput:
    """Create a canonical immutable worker payload from validated clinical data."""
    if isinstance(facts, (str, bytes)) or not isinstance(facts, Sequence):
        raise LocalValidationError("Summary has invalid fact input")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise LocalValidationError("Summary has invalid evidence input")
    if isinstance(uncertainty_labels, (str, bytes)) or not isinstance(
        uncertainty_labels,
        Sequence,
    ):
        raise LocalValidationError("Summary has invalid uncertainty labels")
    if type(requested_scope) is not dict:
        raise LocalValidationError("Summary has invalid requested scope")

    try:
        scope = GroundedSummaryScope.model_validate(requested_scope)
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary has invalid requested scope") from None
    try:
        fact_candidates = tuple(_FactCandidate.model_validate(item) for item in facts)
    except (RecursionError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError) and any(
            error["loc"] and error["loc"][-1] == "evidence_ids"
            for error in exc.errors(include_input=False)
        ):
            raise LocalValidationError("Summary has invalid evidence binding") from None
        raise LocalValidationError("Summary has invalid fact input") from None
    try:
        _validate_fact_scope(scope, fact_candidates)
    except (RecursionError, TypeError, ValueError):
        raise LocalValidationError(
            "Summary facts do not match the requested scope"
        ) from None
    try:
        evidence_candidates = tuple(
            _EvidenceCandidate.model_validate(item) for item in evidence
        )
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary has invalid evidence input") from None
    try:
        uncertainty_candidates = tuple(
            _UncertaintyCandidate.model_validate(item) for item in uncertainty_labels
        )
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary has invalid uncertainty labels") from None

    source_evidence_ids = [item.id for item in evidence_candidates]
    if len(source_evidence_ids) != len(set(source_evidence_ids)):
        raise LocalValidationError("Summary has duplicate evidence identifiers")
    evidence_aliases = {
        item.id: _stable_evidence_id(
            source_id=item.id,
            excerpt=item.excerpt,
            page_number=item.page_number,
            section=item.section,
            field_paths=item.field_paths,
        )
        for item in evidence_candidates
    }
    evidence_by_source_id = {item.id: item for item in evidence_candidates}

    record_ids = [candidate.record_id for candidate in fact_candidates]
    if len(record_ids) != len(set(record_ids)):
        raise LocalValidationError("Summary has duplicate record identifiers")

    grounded_facts: list[GroundedSummaryFact] = []
    fact_aliases: dict[str, str] = {}
    reverse_links: dict[str, list[str]] = {
        stable_id: [] for stable_id in evidence_aliases.values()
    }
    for candidate in fact_candidates:
        unknown = set(candidate.evidence_ids) - set(evidence_aliases)
        if unknown:
            raise LocalValidationError("Summary fact references unknown evidence")
        content_json = _canonical_json_value(candidate.content)
        fields = _flatten_content(candidate.content)
        known_paths = {field.path for field in fields}
        if any(
            set(evidence_by_source_id[source_id].field_paths) - known_paths
            for source_id in candidate.evidence_ids
        ):
            raise LocalValidationError(
                "Summary evidence references an unknown fact field"
            )
        stable_evidence_ids = tuple(
            evidence_aliases[source_id] for source_id in candidate.evidence_ids
        )
        fact_id = _stable_fact_id(
            candidate.record_id,
            content_json,
            stable_evidence_ids,
        )
        try:
            grounded_fact = GroundedSummaryFact(
                fact_id=fact_id,
                record_id=candidate.record_id,
                content_json=content_json,
                fields=fields,
                evidence_ids=stable_evidence_ids,
            )
        except (RecursionError, TypeError, ValueError, ValidationError):
            raise LocalValidationError("Summary has invalid fact input") from None
        grounded_facts.append(grounded_fact)
        fact_aliases[candidate.record_id] = fact_id
        for evidence_id in stable_evidence_ids:
            reverse_links[evidence_id].append(fact_id)

    if len({fact.fact_id for fact in grounded_facts}) != len(grounded_facts):
        raise LocalValidationError("Summary has duplicate stable fact identifiers")
    if any(not links for links in reverse_links.values()):
        raise LocalValidationError("Summary evidence is not linked to a fact")

    grounded_evidence = tuple(
        GroundedSummaryEvidence(
            evidence_id=evidence_aliases[item.id],
            source_id=item.id,
            excerpt=item.excerpt,
            page_number=item.page_number,
            section=item.section,
            fact_ids=tuple(reverse_links[evidence_aliases[item.id]]),
            field_paths=item.field_paths,
        )
        for item in evidence_candidates
    )

    grounded_uncertainties: list[GroundedSummaryUncertainty] = []
    fact_candidates_by_record_id = {
        candidate.record_id: candidate for candidate in fact_candidates
    }
    for candidate in uncertainty_candidates:
        if set(candidate.record_ids) - set(fact_aliases):
            raise LocalValidationError("Summary uncertainty references unknown fact")
        if set(candidate.evidence_ids) - set(evidence_aliases):
            raise LocalValidationError(
                "Summary uncertainty references unknown evidence"
            )
        stable_fact_ids = tuple(
            fact_aliases[record_id] for record_id in candidate.record_ids
        )
        stable_evidence_ids = tuple(
            evidence_aliases[evidence_id] for evidence_id in candidate.evidence_ids
        )
        if any(
            not set(stable_evidence_ids).intersection(
                next(
                    fact.evidence_ids
                    for fact in grounded_facts
                    if fact.fact_id == fact_id
                )
            )
            for fact_id in stable_fact_ids
        ):
            raise LocalValidationError("Summary uncertainty support is not linked")
        if any(
            not set(reverse_links[evidence_id]).intersection(stable_fact_ids)
            for evidence_id in stable_evidence_ids
        ):
            raise LocalValidationError("Summary uncertainty support is not linked")
        if not _uncertainty_is_applicable(
            candidate,
            facts_by_record_id=fact_candidates_by_record_id,
            evidence_by_source_id=evidence_by_source_id,
        ):
            raise LocalValidationError("Summary uncertainty is not applicable")
        grounded_uncertainties.append(
            GroundedSummaryUncertainty(
                uncertainty_id=_stable_uncertainty_id(
                    candidate.template_id,
                    stable_fact_ids,
                    stable_evidence_ids,
                ),
                template_id=candidate.template_id,
                label=_UNCERTAINTY_LABELS[candidate.template_id],
                fact_ids=stable_fact_ids,
                evidence_ids=stable_evidence_ids,
            )
        )

    try:
        result = GroundedSummaryInput(
            requested_scope=scope,
            facts=tuple(grounded_facts),
            evidence=grounded_evidence,
            uncertainty_labels=tuple(grounded_uncertainties),
        )
    except (RecursionError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError) and any(
            "size limit" in error["msg"] for error in exc.errors(include_input=False)
        ):
            raise LocalValidationError("Summary input exceeds the size limit") from None
        raise LocalValidationError("Summary input has invalid structure") from None
    try:
        serialized = json.dumps(
            result.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError):
        raise LocalValidationError("Summary input has invalid structure") from None
    if len(serialized) > MAX_SUMMARY_INPUT_BYTES:
        raise LocalValidationError("Summary input exceeds the size limit")
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise ValueError("nonstandard JSON constant")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_summary_document(
    raw: Mapping[str, object] | str,
) -> GroundedSummaryDocument:
    if type(raw) is str:
        if len(raw) > MAX_RAW_OUTPUT_CHARACTERS:
            raise LocalValidationError("Summary output exceeds the size limit")
        try:
            payload = json.loads(
                raw,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
            raise LocalValidationError("Summary output is not strict JSON") from None
    elif type(raw) is dict:
        payload = raw
    else:
        raise LocalValidationError("Summary output must be a JSON object")

    if type(payload) is not dict:
        raise LocalValidationError("Summary output must be a JSON object")
    try:
        _validate_bounded_json(payload)
        return GroundedSummaryDocument.model_validate(payload)
    except (RecursionError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError) and any(
            error["loc"]
            and error["loc"][-1] in {"field_paths", "evidence_ids"}
            and error["type"] == "too_short"
            for error in exc.errors(include_input=False)
        ):
            raise LocalValidationError(
                "Every factual claim requires fact field and evidence support"
            ) from None
        raise LocalValidationError("Summary output has invalid structure") from None


def parse_grounded_summary_document(
    raw: Mapping[str, object] | str,
) -> GroundedSummaryDocument:
    """Parse bounded reference-only output before registry-specific validation."""
    return _parse_summary_document(raw)


def _extract_fact_registry(
    facts: Mapping[str, Mapping[str, object] | GroundedSummaryFact],
) -> dict[str, GroundedSummaryFact]:
    registry: dict[str, GroundedSummaryFact] = {}
    try:
        for fact_id, value in facts.items():
            if type(fact_id) is not str or not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,128}",
                fact_id,
            ):
                raise ValueError
            fact = (
                value
                if isinstance(value, GroundedSummaryFact)
                else GroundedSummaryFact.model_validate(value)
            )
            if fact.fact_id != fact_id:
                raise ValueError
            verified = GroundedSummaryFact.model_validate(fact.model_dump())
            if verified.fact_id != _stable_fact_id(
                verified.record_id,
                verified.content_json,
                verified.evidence_ids,
            ):
                raise ValueError
            registry[fact_id] = verified
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary support registry is invalid") from None
    return registry


def _extract_evidence_registry(
    evidence: Mapping[
        str,
        Mapping[str, object] | GroundedSummaryEvidence,
    ],
) -> dict[str, GroundedSummaryEvidence]:
    registry: dict[str, GroundedSummaryEvidence] = {}
    try:
        for evidence_id, value in evidence.items():
            if type(evidence_id) is not str or not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,128}",
                evidence_id,
            ):
                raise ValueError
            item = (
                value
                if isinstance(value, GroundedSummaryEvidence)
                else GroundedSummaryEvidence.model_validate(value)
            )
            if item.evidence_id != evidence_id:
                raise ValueError
            verified = GroundedSummaryEvidence.model_validate(item.model_dump())
            registry[evidence_id] = verified
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary support registry is invalid") from None
    return registry


def _extract_uncertainty_registry(
    uncertainties: Mapping[
        str,
        Mapping[str, object] | GroundedSummaryUncertainty,
    ],
) -> dict[str, GroundedSummaryUncertainty]:
    registry: dict[str, GroundedSummaryUncertainty] = {}
    try:
        for uncertainty_id, value in uncertainties.items():
            if type(uncertainty_id) is not str or not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,128}",
                uncertainty_id,
            ):
                raise ValueError
            item = (
                value
                if isinstance(value, GroundedSummaryUncertainty)
                else GroundedSummaryUncertainty.model_validate(value)
            )
            if item.uncertainty_id != uncertainty_id:
                raise ValueError
            verified = GroundedSummaryUncertainty.model_validate(item.model_dump())
            registry[uncertainty_id] = verified
    except (RecursionError, TypeError, ValueError, ValidationError):
        raise LocalValidationError("Summary uncertainty support is invalid") from None
    return registry


def _validate_support_registry(
    facts: Mapping[str, GroundedSummaryFact],
    evidence: Mapping[str, GroundedSummaryEvidence],
    uncertainties: Mapping[str, GroundedSummaryUncertainty],
) -> None:
    for fact_id, fact in facts.items():
        for evidence_id in fact.evidence_ids:
            linked_evidence = evidence.get(evidence_id)
            if linked_evidence is None or fact_id not in linked_evidence.fact_ids:
                raise ValueError("fact and evidence links disagree")
    for evidence_id, item in evidence.items():
        for fact_id in item.fact_ids:
            linked_fact = facts.get(fact_id)
            if linked_fact is None or evidence_id not in linked_fact.evidence_ids:
                raise ValueError("fact and evidence links disagree")
            if set(item.field_paths) - {field.path for field in linked_fact.fields}:
                raise ValueError("evidence references an unknown fact field")
    for item in uncertainties.values():
        if set(item.fact_ids) - set(facts):
            raise ValueError("uncertainty references unknown fact")
        if set(item.evidence_ids) - set(evidence):
            raise ValueError("uncertainty references unknown evidence")
        for fact_id in item.fact_ids:
            if not set(item.evidence_ids).intersection(facts[fact_id].evidence_ids):
                raise ValueError("uncertainty has unlinked support")
        for evidence_id in item.evidence_ids:
            if not set(evidence[evidence_id].fact_ids).intersection(item.fact_ids):
                raise ValueError("uncertainty has unrelated evidence")
        fact_contents = tuple(
            _parse_canonical_content(facts[fact_id].content_json)
            for fact_id in item.fact_ids
        )
        evidence_has_source_detail = tuple(
            evidence[evidence_id].page_number is not None
            or evidence[evidence_id].section is not None
            for evidence_id in item.evidence_ids
        )
        if not _uncertainty_template_is_applicable(
            item.template_id,
            facts=fact_contents,
            evidence_has_source_detail=evidence_has_source_detail,
        ):
            raise ValueError("uncertainty is not applicable")


def _is_numeric_observation_measurement(value: object) -> bool:
    if type(value) in {int, float}:
        return True
    return (
        type(value) is str and _FINITE_MEASUREMENT_PATTERN.fullmatch(value) is not None
    )


def _fact_safety_companion_paths(
    fact: GroundedSummaryFact,
    selected_paths: Sequence[str],
) -> tuple[str, ...]:
    companions = [
        field.path
        for field in fact.fields
        if field.path in {"/assertion", "/relationship", "/status"}
        or field.path.startswith("/statuses/")
    ]
    known_fields = {field.path: field for field in fact.fields}
    selected = set(selected_paths)
    typed_value_paths = tuple(
        path for path in known_fields if path.startswith("/value/")
    )
    if selected.intersection(typed_value_paths):
        companions.extend(typed_value_paths)
    for numeric_path, unit_path in _NUMERIC_UNIT_FIELD_PAIRS:
        if not selected.intersection({numeric_path, unit_path}):
            continue
        numeric_field = known_fields.get(numeric_path)
        unit_field = known_fields.get(unit_path)
        if numeric_field is None:
            raise LocalValidationError(
                "Summary unit cannot render without its numeric value"
            )
        value = json.loads(numeric_field.value_json)
        if numeric_path == "/value" and not _is_numeric_observation_measurement(value):
            continue
        if unit_field is None:
            inline_unit = False
            if type(value) is str:
                pattern = (
                    _FINITE_MEASUREMENT_PATTERN
                    if numeric_path == "/value"
                    else _POSITIVE_QUANTITY_PATTERN
                )
                match = pattern.fullmatch(value)
                inline_unit = match is not None and match.group("unit") is not None
            if not inline_unit:
                raise LocalValidationError(
                    "Summary numeric value cannot render without its unit"
                )
            continue
        if numeric_path in selected and unit_path not in selected:
            companions.append(unit_path)
        if unit_path in selected and numeric_path not in selected:
            companions.append(numeric_path)
    return tuple(dict.fromkeys(companions))


def _expand_summary_safety_companions(
    document: GroundedSummaryDocument,
    *,
    facts: Mapping[str, GroundedSummaryFact],
    evidence: Mapping[str, GroundedSummaryEvidence],
    uncertainties: Mapping[str, GroundedSummaryUncertainty],
) -> GroundedSummaryDocument:
    expanded_sections: list[GroundedSection] = []
    selected_fact_ids: set[str] = set()
    for section in document.sections:
        expanded_claims: list[GroundedClaim] = []
        for claim in section.claims:
            selected_fact_ids.add(claim.fact_id)
            fact = facts[claim.fact_id]
            field_paths = list(claim.field_paths)
            evidence_ids = list(claim.evidence_ids)
            for companion_path in _fact_safety_companion_paths(
                fact,
                claim.field_paths,
            ):
                if companion_path in field_paths:
                    continue
                supporting_evidence = sorted(
                    evidence_id
                    for evidence_id in fact.evidence_ids
                    if companion_path in evidence[evidence_id].field_paths
                )
                if not supporting_evidence:
                    raise LocalValidationError(
                        "Summary fact qualifier lacks evidence support"
                    )
                field_paths.append(companion_path)
                if supporting_evidence[0] not in evidence_ids:
                    evidence_ids.append(supporting_evidence[0])
            rendered_paths = set(field_paths)
            evidence_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if rendered_paths.intersection(evidence[evidence_id].field_paths)
            ]
            try:
                expanded_claims.append(
                    GroundedClaim(
                        fact_id=claim.fact_id,
                        field_paths=tuple(field_paths),
                        evidence_ids=tuple(evidence_ids),
                    )
                )
            except (TypeError, ValueError, ValidationError):
                raise LocalValidationError(
                    "Summary safety companions exceed output limits"
                ) from None
        expanded_sections.append(
            GroundedSection(
                heading=section.heading,
                claims=tuple(expanded_claims),
            )
        )

    uncertainty_references = list(document.uncertainties)
    referenced_uncertainty_ids = {
        reference.uncertainty_id for reference in uncertainty_references
    }
    for uncertainty_id in sorted(uncertainties):
        uncertainty = uncertainties[uncertainty_id]
        if (
            uncertainty_id not in referenced_uncertainty_ids
            and selected_fact_ids.intersection(uncertainty.fact_ids)
        ):
            uncertainty_references.append(
                GroundedUncertaintyReference(
                    uncertainty_id=uncertainty_id,
                    fact_ids=uncertainty.fact_ids,
                    evidence_ids=uncertainty.evidence_ids,
                )
            )

    try:
        return GroundedSummaryDocument(
            sections=tuple(expanded_sections),
            uncertainties=tuple(uncertainty_references),
        )
    except (TypeError, ValueError, ValidationError):
        raise LocalValidationError(
            "Summary safety companions exceed output limits"
        ) from None


def _unescape_pointer_token(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _field_label(path: str) -> str:
    parts = [_unescape_pointer_token(part) for part in path.split("/")[1:]]
    keys = [part for part in parts if not part.isdigit()]
    label = keys[-1] if keys else "Value"
    return label.replace("_", " ").strip().capitalize()


def _expected_fact_heading(fact: GroundedSummaryFact) -> SummaryHeading:
    record_type = next(
        (
            json.loads(field.value_json)
            for field in fact.fields
            if field.path == "/record_type"
        ),
        None,
    )
    if type(record_type) is not str:
        return "Other records"
    return _RECORD_TYPE_HEADINGS.get(record_type, "Other records")


def _render_field_value(value_json: str) -> str:
    value = json.loads(value_json)
    if type(value) is str:
        return value
    if value is None:
        return "not recorded"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is list:
        return "empty list"
    if type(value) is dict:
        return "empty object"
    return str(value)


def _render_typed_observation_value(value: Mapping[str, object]) -> str:
    """Render one already-validated typed measurement without adding facts."""
    if value["kind"] == "quantity":
        return f"{value['comparator']}{value['number']} {value['unit']}"
    numerator_unit = str(value["numerator_unit"])
    denominator_unit = str(value["denominator_unit"])
    if numerator_unit == denominator_unit:
        return f"{value['numerator']}/{value['denominator']} {numerator_unit}"
    return (
        f"{value['numerator']} {numerator_unit}/"
        f"{value['denominator']} {denominator_unit}"
    )


def _render_claim_fields(
    fact: GroundedSummaryFact,
    field_paths: Sequence[str],
) -> str:
    fields = {field.path: field for field in fact.fields}
    content = _parse_canonical_content(fact.content_json)
    typed_value = content.get("value")
    rendered: list[str] = []
    rendered_typed_value = False
    for path in field_paths:
        if path.startswith("/value/") and isinstance(typed_value, Mapping):
            if not rendered_typed_value:
                rendered.append(
                    f"Value: {_render_typed_observation_value(typed_value)}"
                )
                rendered_typed_value = True
            continue
        rendered.append(
            f"{_field_label(path)}: {_render_field_value(fields[path].value_json)}"
        )
    return "; ".join(rendered)


def _escape_markdown_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for marker in ("[", "]", "*", "_", "`", "<", ">", "#", "|"):
        escaped = escaped.replace(marker, f"\\{marker}")
    return escaped


def _render_markdown(
    document: GroundedSummaryDocument,
    *,
    facts: Mapping[str, GroundedSummaryFact],
    uncertainties: Mapping[str, GroundedSummaryUncertainty],
) -> str:
    blocks: list[str] = []
    for section in document.sections:
        lines = [f"## {section.heading}", ""]
        for claim in section.claims:
            fact = facts[claim.fact_id]
            statement = _render_claim_fields(fact, claim.field_paths)
            lines.append(
                f"- {_escape_markdown_text(statement)}. "
                f"[Fact: {claim.fact_id}; "
                f"Evidence: {', '.join(sorted(claim.evidence_ids))}]"
            )
        blocks.append("\n".join(lines))
    if document.uncertainties:
        lines = ["## Uncertainties", ""]
        for reference in document.uncertainties:
            uncertainty = uncertainties[reference.uncertainty_id]
            lines.append(
                f"- {_escape_markdown_text(uncertainty.label)} "
                f"[Facts: {', '.join(sorted(reference.fact_ids))}; "
                f"Evidence: {', '.join(sorted(reference.evidence_ids))}]"
            )
        blocks.append("\n".join(lines))
    blocks.append(SERVER_MEDICAL_DISCLAIMER)
    return "\n\n".join(blocks)


def validate_and_render_summary(
    raw: Mapping[str, object] | str,
    *,
    facts: Mapping[str, Mapping[str, object] | GroundedSummaryFact],
    evidence: Mapping[
        str,
        Mapping[str, object] | GroundedSummaryEvidence,
    ],
    uncertainties: Mapping[
        str,
        Mapping[str, object] | GroundedSummaryUncertainty,
    ]
    | None = None,
) -> RenderedGroundedSummary:
    """Validate worker references and server-render only canonical fact fields."""
    if (
        not isinstance(facts, Mapping)
        or not isinstance(evidence, Mapping)
        or (uncertainties is not None and not isinstance(uncertainties, Mapping))
    ):
        raise LocalValidationError("Summary support registry is invalid")
    document = _parse_summary_document(raw)
    fact_registry = _extract_fact_registry(facts)
    evidence_registry = _extract_evidence_registry(evidence)
    uncertainty_registry = _extract_uncertainty_registry(uncertainties or {})
    try:
        _validate_support_registry(
            fact_registry,
            evidence_registry,
            uncertainty_registry,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "uncertainty" in str(exc):
            raise LocalValidationError(
                "Summary uncertainty support is invalid"
            ) from None
        raise LocalValidationError("Summary support registry is invalid") from None

    for section in document.sections:
        for claim in section.claims:
            fact = fact_registry.get(claim.fact_id)
            if fact is None:
                raise LocalValidationError("Summary references an unknown fact")
            if set(claim.evidence_ids) - set(evidence_registry):
                raise LocalValidationError("Summary references unknown evidence")
            known_paths = {field.path for field in fact.fields}
            if set(claim.field_paths) - known_paths:
                raise LocalValidationError("Summary references an unknown fact field")
            if not set(claim.evidence_ids).issubset(fact.evidence_ids):
                raise LocalValidationError("Summary claim support is not linked")
            supported_paths = {
                path
                for evidence_id in claim.evidence_ids
                for path in evidence_registry[evidence_id].field_paths
            }
            if set(claim.field_paths) - supported_paths:
                raise LocalValidationError("Summary claim lacks evidence field support")
            expected_heading = _expected_fact_heading(fact)
            if section.heading not in {"Overview", expected_heading}:
                raise LocalValidationError(
                    "Summary places a fact in an invalid section"
                )

    for reference in document.uncertainties:
        uncertainty = uncertainty_registry.get(reference.uncertainty_id)
        if uncertainty is None:
            raise LocalValidationError("Summary references an unknown uncertainty")
        if set(reference.fact_ids) != set(uncertainty.fact_ids) or set(
            reference.evidence_ids
        ) != set(uncertainty.evidence_ids):
            raise LocalValidationError("Summary uncertainty support does not match")

    if fact_registry and not any(section.claims for section in document.sections):
        raise LocalValidationError("Summary contains no grounded claims")

    selection_document = document
    document = _expand_summary_safety_companions(
        document,
        facts=fact_registry,
        evidence=evidence_registry,
        uncertainties=uncertainty_registry,
    )

    try:
        markdown = _render_markdown(
            document,
            facts=fact_registry,
            uncertainties=uncertainty_registry,
        )
    except (KeyError, RecursionError, TypeError, ValueError):
        raise LocalValidationError("Summary output cannot be rendered safely") from None
    if len(markdown) > MAX_RENDERED_CONTENT_CHARACTERS:
        raise LocalValidationError("Summary output exceeds the size limit")
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        and character not in {"\n", "\r"}
        for character in markdown
    ):
        raise LocalValidationError("Summary output contains an unsafe control")
    return RenderedGroundedSummary(
        selection_document=selection_document,
        document=document,
        markdown=markdown,
    )
