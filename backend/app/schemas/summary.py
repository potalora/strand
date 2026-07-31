from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.services.local_ai.grounded_summary import (
    MAX_FACTS,
    MAX_RAW_OUTPUT_CHARACTERS,
    GroundedSummaryDocument,
    SummaryCategory,
)
from app.services.local_ai.types import ProcessingMode

SafeProvenanceText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=2048),
]
SummaryRecordType = SummaryCategory | Literal["family_history", "imaging"]
BoundedRecordIds = Annotated[list[UUID], Field(max_length=MAX_FACTS)]
BoundedRecordTypes = Annotated[list[SummaryRecordType], Field(max_length=MAX_FACTS)]
BoundedPastedResponse = Annotated[
    str,
    StringConstraints(max_length=MAX_RAW_OUTPUT_CHARACTERS),
]


class SummaryRuntimeIdentity(BaseModel):
    """Bounded runtime identity from a validated strict-local manifest."""

    model_config = ConfigDict(extra="forbid")

    name: SafeProvenanceText
    version: SafeProvenanceText


class StrictLocalSummaryModelIdentity(BaseModel):
    """Safe strict-local summary model fields exposed to API clients."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["summary"]
    repository: SafeProvenanceText
    revision: SafeProvenanceText
    quantization: SafeProvenanceText
    runtime: SummaryRuntimeIdentity


class StrictLocalSummaryProvenance(BaseModel):
    """Exact validated-pack identity without local paths or secrets."""

    model_config = ConfigDict(extra="forbid")

    processing_mode: Literal["validated_strict_local"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pack_revision: SafeProvenanceText
    model: StrictLocalSummaryModelIdentity


class RoutedSummaryProvenanceBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: SafeProvenanceText
    model: SafeProvenanceText


class CustomLocalSummaryProvenance(RoutedSummaryProvenanceBase):
    """Safe provider/model identity for custom-local generation."""

    processing_mode: Literal["custom_local"]


class CloudAssistedSummaryProvenance(RoutedSummaryProvenanceBase):
    """Safe provider/model identity for cloud-assisted generation."""

    processing_mode: Literal["cloud_assisted"]


RoutedSummaryProvenance = Annotated[
    CustomLocalSummaryProvenance | CloudAssistedSummaryProvenance,
    Field(discriminator="processing_mode"),
]
SummaryModelProvenance = StrictLocalSummaryProvenance | RoutedSummaryProvenance


class BuildPromptRequest(BaseModel):
    patient_id: UUID
    summary_type: str = "full"
    category: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    output_format: Literal["natural_language", "json", "both"] = "natural_language"
    record_ids: BoundedRecordIds | None = None
    record_types: BoundedRecordTypes | None = None


class PromptResponse(BaseModel):
    id: UUID
    summary_type: str
    system_prompt: str
    user_prompt: str
    target_model: str
    suggested_config: dict
    record_count: int
    de_identification_report: dict | None
    copyable_payload: str
    generated_at: datetime
    processing_mode: ProcessingMode | None = None
    model_provenance: SummaryModelProvenance | None = None


class PromptDetailResponse(PromptResponse):
    response_text: str | None = None
    response_format: Literal["natural_language", "json", "both"] | None = None
    typed_response: GroundedSummaryDocument | None = None


class PromptListResponse(BaseModel):
    items: list[PromptResponse]


class PasteResponseRequest(BaseModel):
    prompt_id: UUID
    response_text: BoundedPastedResponse


class PasteResponseResponse(BaseModel):
    id: UUID
    prompt_id: UUID
    response_pasted_at: datetime
    typed_response: GroundedSummaryDocument
    natural_language: str | None = None
    json_data: dict | None = None


class GenerateSummaryRequest(BaseModel):
    patient_id: UUID
    summary_type: str = "full"
    category: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    output_format: Literal["natural_language", "json", "both"] = "natural_language"
    custom_system_prompt: Annotated[str, StringConstraints(max_length=4096)] | None = (
        None
    )
    custom_user_prompt: Annotated[str, StringConstraints(max_length=4096)] | None = None
    provider: str | None = None
    model: str | None = None
    processing_mode: ProcessingMode = ProcessingMode.CLOUD_ASSISTED
    record_ids: BoundedRecordIds | None = None


class DuplicateWarning(BaseModel):
    total_records: int
    deduped_records: int
    duplicates_excluded: int
    message: str | None = None


class GenerateSummaryResponseBase(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: UUID
    natural_language: str | None = None
    json_data: dict | None = None
    record_count: int
    duplicate_warning: DuplicateWarning | None = None
    de_identification_report: dict | None = None
    model_used: str
    generated_at: datetime


class StrictLocalGenerateSummaryResponse(GenerateSummaryResponseBase):
    processing_mode: Literal["validated_strict_local"]
    model_provenance: StrictLocalSummaryProvenance | None = None
    typed_response: GroundedSummaryDocument


class StrictLocalSummaryAccepted(BaseModel):
    """Content-free acknowledgement for a durable strict-local summary job."""

    id: UUID
    job_id: UUID
    processing_mode: Literal["validated_strict_local"]
    kind: Literal["summary"] = "summary"
    status: Literal["queued"] = "queued"
    stage: Literal["queued"] = "queued"
    created_at: datetime


class CustomLocalGenerateSummaryResponse(GenerateSummaryResponseBase):
    processing_mode: Literal["custom_local"]
    model_provenance: CustomLocalSummaryProvenance | None = None
    typed_response: GroundedSummaryDocument


class CloudAssistedGenerateSummaryResponse(GenerateSummaryResponseBase):
    processing_mode: Literal["cloud_assisted"]
    model_provenance: CloudAssistedSummaryProvenance | None = None
    typed_response: GroundedSummaryDocument


GenerateSummaryResponse = Annotated[
    StrictLocalGenerateSummaryResponse
    | CustomLocalGenerateSummaryResponse
    | CloudAssistedGenerateSummaryResponse,
    Field(discriminator="processing_mode"),
]


class SummaryItemCreate(BaseModel):
    record_id: UUID


class SummaryItemResponse(BaseModel):
    id: UUID
    record_id: UUID
    created_at: datetime
