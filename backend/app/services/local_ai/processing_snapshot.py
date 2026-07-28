"""Resolve immutable processing policy snapshots for newly queued work."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.llm_settings import UserLLMPreferences
from app.models.local_ai import LocalAIJob
from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalPolicyError, LocalValidationError
from app.services.local_ai.manifest import (
    canonicalize_manifest_snapshot,
    load_manifest,
)
from app.services.local_ai.lifecycle_lock import acquire_local_ai_lifecycle_lock
from app.services.local_ai.pack_operations import PackOperationStore
from app.services.local_ai.release_evidence import load_release_evidence
from app.services.local_ai.types import ProcessingMode

EXTRACTION_SCHEMA_VERSION = "clinical-document-extraction.v1"


@dataclass(frozen=True)
class ProcessingSnapshot:
    """A detached policy and optional exact model-pack snapshot."""

    mode: ProcessingMode
    manifest_snapshot: dict | None
    manifest_sha256: str | None
    schema_version: str | None


def build_ingestion_job(
    *,
    upload_id: UUID,
    user_id: UUID,
    snapshot: ProcessingSnapshot,
) -> LocalAIJob | None:
    """Build the strict-local job that must commit with a queued upload."""

    if snapshot.mode is not ProcessingMode.VALIDATED_STRICT_LOCAL:
        return None
    if (
        snapshot.manifest_snapshot is None
        or snapshot.manifest_sha256 is None
        or snapshot.schema_version != EXTRACTION_SCHEMA_VERSION
    ):
        raise LocalPolicyError("Validated strict-local job snapshot is invalid.")
    return LocalAIJob(
        user_id=user_id,
        upload_id=upload_id,
        kind="ingestion",
        processing_mode=snapshot.mode.value,
        manifest_snapshot=snapshot.manifest_snapshot,
        manifest_sha256=snapshot.manifest_sha256,
        status="queued",
        stage="queued",
        progress={},
        audit_metadata={},
    )


def _coerce_mode(value: ProcessingMode | str | None) -> ProcessingMode:
    if value is None:
        return ProcessingMode.CLOUD_ASSISTED
    try:
        return ProcessingMode(value)
    except (TypeError, ValueError) as exc:
        raise LocalPolicyError("The requested processing mode is invalid.") from exc


def _current_strict_snapshot() -> ProcessingSnapshot:
    """Load the one exact pack currently eligible for strict-local admission."""

    unavailable = "Validated strict-local model pack is unavailable."
    if not settings.local_ai_enabled:
        raise LocalPolicyError(unavailable)
    try:
        locked = load_manifest(Path(settings.local_ai_manifest_path))
        store = ArtifactStore(Path(settings.local_ai_model_dir))
        operations = PackOperationStore(store)
        active = store.active_manifest()
        if (
            active != locked
            or not store.has_validation_receipt(locked)
            or operations.has_nonterminal()
        ):
            raise LocalPolicyError(unavailable)
        load_release_evidence(
            Path(settings.local_ai_release_evidence_path),
            manifest=locked,
            benchmark_path=Path(settings.local_ai_benchmark_path),
            fidelity_path=Path(settings.local_ai_fidelity_path),
        )
        detached_manifest = json.loads(json.dumps(asdict(locked), ensure_ascii=True))
        snapshot, digest = canonicalize_manifest_snapshot(detached_manifest)
    except LocalPolicyError:
        raise
    except (LocalValidationError, OSError) as exc:
        raise LocalPolicyError(unavailable) from exc
    return ProcessingSnapshot(
        mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        schema_version=EXTRACTION_SCHEMA_VERSION,
    )


async def revalidate_strict_snapshot_admission(
    db: AsyncSession,
    snapshot: ProcessingSnapshot,
) -> None:
    """Recheck a strict snapshot while holding the global enqueue/mutation lock."""

    if snapshot.mode is not ProcessingMode.VALIDATED_STRICT_LOCAL:
        return
    await acquire_local_ai_lifecycle_lock(db)
    if _current_strict_snapshot() != snapshot:
        raise LocalPolicyError("Validated strict-local model pack is unavailable.")


async def resolve_new_job_snapshot(
    db: AsyncSession,
    user_id: UUID,
    explicit_mode: ProcessingMode | str | None = None,
) -> ProcessingSnapshot:
    """Resolve explicit mode, then stored preference, then the cloud default.

    Validated strict-local work is accepted only when the feature is enabled and
    the exact locked manifest is the fully verified active artifact pack.
    """

    selected = explicit_mode
    if selected is None:
        preference = (
            await db.execute(
                select(UserLLMPreferences).where(UserLLMPreferences.user_id == user_id)
            )
        ).scalar_one_or_none()
        selected = preference.processing_mode if preference is not None else None
    mode = _coerce_mode(selected)

    if mode is not ProcessingMode.VALIDATED_STRICT_LOCAL:
        return ProcessingSnapshot(
            mode=mode,
            manifest_snapshot=None,
            manifest_sha256=None,
            schema_version=None,
        )

    await acquire_local_ai_lifecycle_lock(db)
    return _current_strict_snapshot()


async def resolve_new_ingestion_snapshot(
    db: AsyncSession,
    user_id: UUID,
    explicit_mode: ProcessingMode | str | None = None,
) -> ProcessingSnapshot:
    """Resolve a new ingestion snapshot without silently changing modes."""

    snapshot = await resolve_new_job_snapshot(db, user_id, explicit_mode)
    if snapshot.mode not in {
        ProcessingMode.CLOUD_ASSISTED,
        ProcessingMode.VALIDATED_STRICT_LOCAL,
    }:
        raise LocalPolicyError(
            f"{snapshot.mode.value} is not available for document ingestion."
        )
    return snapshot
