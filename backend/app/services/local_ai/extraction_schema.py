"""Strict, versioned NuExtract clinical-document output schema."""

from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

CLINICAL_EXTRACTION_SCHEMA_VERSION = "clinical-document-extraction.v1"
EVIDENCE_LOCATION_NORMALIZATION = "whitespace-collapsed-casefold-v1"

MAX_FACTS_PER_CATEGORY = 256
MAX_FIELD_LIST_ITEMS = 128
MAX_PLAN_ITEMS = 64
MAX_SHORT_TEXT = 512
MAX_LONG_TEXT = 4096


def _safe_text(value: str) -> str:
    """Reject blank text and unsafe control characters without rewriting it."""
    if not value.strip():
        raise ValueError("must not be blank")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in value
    ):
        raise ValueError("contains a control character")
    return value


ShortText = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=MAX_SHORT_TEXT),
    AfterValidator(_safe_text),
]
LongText = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=MAX_LONG_TEXT),
    AfterValidator(_safe_text),
]
FactId = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
]
PageNumber = Annotated[StrictInt, Field(ge=1, le=100_000)]
Confidence = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class _StrictModel(BaseModel):
    # Scalars are strict at their field declarations. Keeping enum coercion
    # enabled is intentional because JSON can represent enum members only as
    # their string values.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class AssertionState(StrEnum):
    PRESENT = "present"
    NEGATED = "negated"
    FAMILY_HISTORY = "family_history"
    UNCERTAIN = "uncertain"
    MENTIONED_NOT_PERFORMED = "mentioned_not_performed"


class MedicationStatus(StrEnum):
    ACTIVE = "active"
    STOPPED = "stopped"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class ClinicalStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESOLVED = "resolved"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class EncounterStatus(StrEnum):
    FINISHED = "finished"
    IN_PROGRESS = "in_progress"
    PLANNED = "planned"
    UNKNOWN = "unknown"


class ImmunizationStatus(StrEnum):
    COMPLETED = "completed"
    NOT_DONE = "not_done"
    ENTERED_IN_ERROR = "entered_in_error"
    UNKNOWN = "unknown"


class ReportStatus(StrEnum):
    FINAL = "final"
    PRELIMINARY = "preliminary"
    AMENDED = "amended"
    UNKNOWN = "unknown"


class NormalizationMethod(StrEnum):
    IDENTITY = "identity"
    DECIMAL = "decimal"
    ISO_DATE = "iso-date"
    UCUM_UNIT = "ucum-unit"


class EvidenceFact(_StrictModel):
    """Fields common to each source-grounded clinical fact."""

    fact_id: FactId | None = None
    verbatim: LongText
    page_number: PageNumber
    evidence_excerpt: LongText
    confidence: Confidence | None = None
    normalized_value: ShortText | None = None
    normalization_method: NormalizationMethod | None = None
    normalization_version: ShortText | None = None

    _evidence_id: str | None = PrivateAttr(default=None)
    _evidence_start_offset: int | None = PrivateAttr(default=None)
    _evidence_end_offset: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _normalization_is_versioned(self) -> EvidenceFact:
        supplied = (
            self.normalized_value,
            self.normalization_method,
            self.normalization_version,
        )
        if any(item is not None for item in supplied) and any(
            item is None for item in supplied
        ):
            raise ValueError("normalized values require a method and version")
        return self

    @property
    def evidence_id(self) -> str | None:
        return self._evidence_id


class PatientExtraction(_StrictModel):
    """Optional patient identifiers; never adapted into clinical FHIR records."""

    name: ShortText | None = None
    date_of_birth: ShortText | None = None


class MedicationFact(EvidenceFact):
    name: ShortText
    dose_value: ShortText | None = None
    dose_unit: ShortText | None = None
    route: ShortText | None = None
    frequency: ShortText | None = None
    status: MedicationStatus = MedicationStatus.UNKNOWN
    date: ShortText | None = None


class ConditionFact(EvidenceFact):
    name: ShortText
    assertion: AssertionState = AssertionState.UNCERTAIN
    relationship: ShortText | None = None
    date: ShortText | None = None


class ProcedureFact(EvidenceFact):
    name: ShortText
    assertion: AssertionState = AssertionState.UNCERTAIN
    date: ShortText | None = None
    provider: ShortText | None = None


class LabFact(EvidenceFact):
    name: ShortText
    value: ShortText | None = None
    unit: ShortText | None = None
    reference_range: ShortText | None = None
    interpretation: ShortText | None = None
    date: ShortText | None = None
    assertion: AssertionState = AssertionState.PRESENT


class AllergyFact(EvidenceFact):
    substance: ShortText
    reaction: ShortText | None = None
    severity: ShortText | None = None
    status: ClinicalStatus = ClinicalStatus.UNKNOWN
    date: ShortText | None = None
    assertion: AssertionState = AssertionState.PRESENT


class EncounterFact(EvidenceFact):
    name: ShortText
    visit_type: Literal["office", "telehealth", "emergency", "inpatient", "other"] = (
        "other"
    )
    date: ShortText | None = None
    provider: ShortText | None = None
    facility: ShortText | None = None
    status: EncounterStatus = EncounterStatus.UNKNOWN


class ImmunizationFact(EvidenceFact):
    name: ShortText
    date: ShortText | None = None
    status: ImmunizationStatus = ImmunizationStatus.UNKNOWN
    route: ShortText | None = None
    site: ShortText | None = None
    dose: ShortText | None = None
    dose_unit: ShortText | None = None
    manufacturer: ShortText | None = None
    lot: ShortText | None = None


class VitalSignFact(EvidenceFact):
    name: ShortText
    value: ShortText
    unit: ShortText | None = None
    date: ShortText | None = None
    assertion: AssertionState = AssertionState.PRESENT


class DiagnosticReportFact(EvidenceFact):
    name: ShortText
    findings: LongText | None = None
    interpretation: LongText | None = None
    date: ShortText | None = None
    category: Literal[
        "imaging",
        "laboratory",
        "pathology",
        "endoscopy",
        "nuclear_medicine",
        "pulmonary",
        "laboratory_panel",
        "other",
    ] = "other"
    performer: ShortText | None = None
    status: ReportStatus = ReportStatus.UNKNOWN
    assertion: AssertionState = AssertionState.PRESENT


class CarePlanFact(EvidenceFact):
    title: ShortText
    plan_items: Annotated[
        list[ShortText], Field(strict=True, max_length=MAX_PLAN_ITEMS)
    ] = Field(default_factory=list)
    status: ClinicalStatus = ClinicalStatus.UNKNOWN
    date: ShortText | None = None


class ValidatedEvidence(_StrictModel):
    """Server-generated evidence location in the normalized locator representation."""

    id: Annotated[
        StrictStr,
        StringConstraints(
            min_length=44,
            max_length=44,
            pattern=r"^ev1_[0-9a-f]{40}$",
        ),
    ]
    upload_id: Annotated[StrictStr, StringConstraints(min_length=1, max_length=256)]
    page_number: PageNumber
    excerpt_sha256: Annotated[
        StrictStr,
        StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
    ]
    start_offset: Annotated[StrictInt, Field(ge=0)]
    end_offset: Annotated[StrictInt, Field(ge=0)]
    offset_representation: Literal["whitespace-collapsed-casefold-v1"]
    field_paths: Annotated[
        list[ShortText], Field(strict=True, min_length=1, max_length=16)
    ]

    @model_validator(mode="after")
    def _ordered_offsets(self) -> ValidatedEvidence:
        if self.end_offset <= self.start_offset:
            raise ValueError("end offset must follow start offset")
        return self


_FieldList = Annotated[
    list[ShortText], Field(strict=True, max_length=MAX_FIELD_LIST_ITEMS)
]


class ClinicalDocumentExtraction(_StrictModel):
    """Complete strict NuExtract output before entity/FHIR adaptation."""

    schema_version: Literal["clinical-document-extraction.v1"] = (
        CLINICAL_EXTRACTION_SCHEMA_VERSION
    )
    patient: PatientExtraction | None = None
    medications: Annotated[
        list[MedicationFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    conditions: Annotated[
        list[ConditionFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    procedures: Annotated[
        list[ProcedureFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    labs: Annotated[
        list[LabFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    allergies: Annotated[
        list[AllergyFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    encounters: Annotated[
        list[EncounterFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    immunizations: Annotated[
        list[ImmunizationFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    vital_signs: Annotated[
        list[VitalSignFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    diagnostic_reports: Annotated[
        list[DiagnosticReportFact],
        Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY),
    ] = Field(default_factory=list)
    care_plans: Annotated[
        list[CarePlanFact], Field(strict=True, max_length=MAX_FACTS_PER_CATEGORY)
    ] = Field(default_factory=list)
    unresolved_fields: _FieldList = Field(default_factory=list)
    rejected_fields: _FieldList = Field(default_factory=list)

    _evidence: tuple[ValidatedEvidence, ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def _identifiers_and_field_lists_are_unique(self) -> ClinicalDocumentExtraction:
        fact_ids = [
            fact.fact_id
            for category in FACT_CATEGORY_NAMES
            for fact in getattr(self, category)
            if fact.fact_id is not None
        ]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("duplicate fact_id")
        if len(self.unresolved_fields) != len(set(self.unresolved_fields)):
            raise ValueError("duplicate unresolved field")
        if len(self.rejected_fields) != len(set(self.rejected_fields)):
            raise ValueError("duplicate rejected field")
        return self

    @property
    def evidence(self) -> tuple[ValidatedEvidence, ...]:
        return self._evidence


FACT_CATEGORY_NAMES = (
    "medications",
    "conditions",
    "procedures",
    "labs",
    "allergies",
    "encounters",
    "immunizations",
    "vital_signs",
    "diagnostic_reports",
    "care_plans",
)


def _verbatim() -> dict[str, str]:
    return {"type": "verbatim-string"}


def _fact_template(**fields: object) -> dict[str, object]:
    return {
        **fields,
        "verbatim": _verbatim(),
        "page_number": {"type": "integer", "minimum": 1},
        "evidence_excerpt": _verbatim(),
    }


NUEXTRACT_TEMPLATE_V1: dict[str, object] = {
    "schema_version": CLINICAL_EXTRACTION_SCHEMA_VERSION,
    "patient": {"name": _verbatim(), "date_of_birth": _verbatim()},
    "medications": [
        _fact_template(
            name=_verbatim(),
            dose_value=_verbatim(),
            dose_unit=_verbatim(),
            route=_verbatim(),
            frequency=_verbatim(),
            status={
                "type": "string",
                "enum": [item.value for item in MedicationStatus],
            },
            date=_verbatim(),
        )
    ],
    "conditions": [
        _fact_template(
            name=_verbatim(),
            assertion={
                "type": "string",
                "enum": [item.value for item in AssertionState],
            },
            relationship=_verbatim(),
            date=_verbatim(),
        )
    ],
    "procedures": [
        _fact_template(
            name=_verbatim(),
            assertion={
                "type": "string",
                "enum": [item.value for item in AssertionState],
            },
            date=_verbatim(),
            provider=_verbatim(),
        )
    ],
    "labs": [
        _fact_template(
            name=_verbatim(),
            value=_verbatim(),
            unit=_verbatim(),
            reference_range=_verbatim(),
            interpretation=_verbatim(),
            date=_verbatim(),
        )
    ],
    "allergies": [
        _fact_template(
            substance=_verbatim(),
            reaction=_verbatim(),
            severity=_verbatim(),
            status={"type": "string", "enum": [item.value for item in ClinicalStatus]},
            date=_verbatim(),
            assertion={
                "type": "string",
                "enum": [item.value for item in AssertionState],
            },
        )
    ],
    "encounters": [
        _fact_template(
            name=_verbatim(),
            visit_type={
                "type": "string",
                "enum": ["office", "telehealth", "emergency", "inpatient", "other"],
            },
            date=_verbatim(),
            provider=_verbatim(),
            facility=_verbatim(),
            status={
                "type": "string",
                "enum": [item.value for item in EncounterStatus],
            },
        )
    ],
    "immunizations": [
        _fact_template(
            name=_verbatim(),
            date=_verbatim(),
            status={
                "type": "string",
                "enum": [item.value for item in ImmunizationStatus],
            },
            route=_verbatim(),
            site=_verbatim(),
            dose=_verbatim(),
            dose_unit=_verbatim(),
            manufacturer=_verbatim(),
            lot=_verbatim(),
        )
    ],
    "vital_signs": [
        _fact_template(
            name=_verbatim(), value=_verbatim(), unit=_verbatim(), date=_verbatim()
        )
    ],
    "diagnostic_reports": [
        _fact_template(
            name=_verbatim(),
            findings=_verbatim(),
            interpretation=_verbatim(),
            date=_verbatim(),
            category={
                "type": "string",
                "enum": [
                    "imaging",
                    "laboratory",
                    "pathology",
                    "endoscopy",
                    "nuclear_medicine",
                    "pulmonary",
                    "laboratory_panel",
                    "other",
                ],
            },
            status={
                "type": "string",
                "enum": [item.value for item in ReportStatus],
            },
            assertion={
                "type": "string",
                "enum": [item.value for item in AssertionState],
            },
        )
    ],
    "care_plans": [
        _fact_template(
            title=_verbatim(),
            plan_items=[_verbatim()],
            status={"type": "string", "enum": [item.value for item in ClinicalStatus]},
            date=_verbatim(),
        )
    ],
    "unresolved_fields": [],
    "rejected_fields": [],
}
