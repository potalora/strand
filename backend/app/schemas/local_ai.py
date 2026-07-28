"""Bounded, content-safe API contracts for the validated local model pack."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app.services.local_ai.types import ModelRole, ProcessingMode

PackPlatform = Literal[
    "apple_silicon",
    "linux_cpu",
    "linux_cuda",
    "linux_rocm",
    "unsupported",
]
PackState = Literal[
    "not_installed",
    "downloading",
    "verifying",
    "ready",
    "update_available",
    "failed",
    "preview",
]
OperationAction = Literal["install", "update", "verify", "rollback"]
OperationState = Literal["queued", "running", "paused", "failed", "completed"]
LocalJobKind = Literal["ingestion", "summary"]
LocalJobStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalPackOperationCreated(_StrictResponse):
    """Minimal response returned when a lifecycle operation is queued."""

    operation_id: UUID
    state: Literal["queued"]


class LocalPackOperationResponse(_StrictResponse):
    """Document-free progress for one persisted model-pack operation."""

    id: UUID
    action: OperationAction
    state: OperationState
    current_role: ModelRole | None = None
    bytes_done: StrictInt = Field(ge=0)
    bytes_total: StrictInt = Field(ge=0)
    message: StrictStr | None = Field(default=None, max_length=160)
    retryable: bool


class LocalModelArtifactResponse(_StrictResponse):
    """Immutable artifact identity and honest installation/validation state."""

    role: ModelRole
    repository: StrictStr = Field(min_length=1, max_length=256)
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    quantization: StrictStr = Field(min_length=1, max_length=64)
    runtime: StrictStr = Field(min_length=1, max_length=128)
    license: StrictStr = Field(min_length=1, max_length=64)
    download_bytes: StrictInt = Field(ge=1)
    # Resident memory is intentionally nullable until measured on the release
    # fixture rig. Artifact byte size is not a defensible substitute.
    expected_memory_bytes: StrictInt | None = Field(default=None, ge=1)
    installed: bool
    validated: bool


class LocalPackStatusResponse(_StrictResponse):
    """Current platform, manifest, artifacts, and lifecycle operation."""

    platform: PackPlatform
    compatible: bool
    enabled: bool
    state: PackState
    status_reason: (
        Literal[
            "feature_disabled",
            "release_evidence_missing",
        ]
        | None
    ) = None
    active_revision: StrictStr | None = Field(default=None, max_length=128)
    available_revision: StrictStr | None = Field(default=None, max_length=128)
    models: list[LocalModelArtifactResponse] = Field(max_length=3)
    operation: LocalPackOperationResponse | None = None


class LocalAIJobResponse(_StrictResponse):
    """Content-free status for one owner-scoped local processing job."""

    id: UUID
    kind: LocalJobKind
    status: LocalJobStatus
    stage: StrictStr = Field(min_length=1, max_length=32)
    cancel_requested: bool
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExtractionEvidenceResponse(_StrictResponse):
    """One bounded, decrypted evidence span owned by the requesting user."""

    id: StrictStr = Field(min_length=1, max_length=128)
    page_number: StrictInt | None = Field(default=None, ge=1, le=100_000)
    section: StrictStr | None = Field(default=None, max_length=512)
    excerpt: StrictStr = Field(min_length=1, max_length=4096)
    start_offset: StrictInt | None = Field(default=None, ge=0)
    end_offset: StrictInt | None = Field(default=None, ge=0)
    field_paths: list[StrictStr] = Field(max_length=128)


class ExtractionModelIdentityResponse(_StrictResponse):
    """Exact ingestion-model identity, without summary output or content."""

    role: Literal["ocr", "extraction"]
    repository: StrictStr = Field(min_length=1, max_length=256)
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    quantization: StrictStr = Field(min_length=1, max_length=64)
    runtime: StrictStr = Field(min_length=1, max_length=128)


class RecordExtractionEvidenceResponse(_StrictResponse):
    """Grounding and provenance for one user-owned extracted record."""

    record_id: UUID
    processing_mode: ProcessingMode
    schema_version: StrictStr = Field(min_length=1, max_length=128)
    evidence: list[ExtractionEvidenceResponse] = Field(max_length=256)
    unresolved_fields: list[StrictStr] = Field(max_length=128)
    rejected_fields: list[StrictStr] = Field(max_length=128)
    models: list[ExtractionModelIdentityResponse] = Field(max_length=2)
