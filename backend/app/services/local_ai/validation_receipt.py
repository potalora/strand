"""In-memory and persisted receipts for the exact runtime validation gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import LocalAIManifest
from app.services.local_ai.runtime_identity import (
    WorkerRuntimeIdentity,
    require_manifest_runtime_identity,
)

VALIDATION_RECEIPT_KEYS = frozenset(
    {
        "pack_revision",
        "manifest_sha256",
        "platform",
        "runtime_name",
        "runtime_version",
        "worker_bundle_sha256",
        "validation_suite_version",
        "verifier_version",
    }
)
VERIFIER_VERSION = "strict-local-pack-verifier.v1"
_SEAL = object()


@dataclass(frozen=True)
class RuntimeValidationReceipt:
    """Process-local proof returned only after the fixture verifier succeeds."""

    payload: dict[str, str]
    _seal: object


def _manifest_digest(manifest: LocalAIManifest) -> str:
    from app.services.local_ai.artifact_store import manifest_sha256

    return manifest_sha256(manifest)


def expected_validation_payload(manifest: LocalAIManifest) -> dict[str, str]:
    """Return the only persisted receipt payload valid for ``manifest``."""

    return {
        "pack_revision": manifest.pack_revision,
        "manifest_sha256": _manifest_digest(manifest),
        "platform": manifest.platform,
        "runtime_name": manifest.runtime["name"],
        "runtime_version": manifest.runtime["version"],
        "worker_bundle_sha256": manifest.runtime["worker_bundle_sha256"],
        "validation_suite_version": manifest.validation_suite_version,
        "verifier_version": VERIFIER_VERSION,
    }


def _issue_runtime_validation_receipt(
    manifest: LocalAIManifest,
    observed_identity: WorkerRuntimeIdentity,
) -> RuntimeValidationReceipt:
    """Seal a receipt after the caller has completed the runtime fixture gate."""

    require_manifest_runtime_identity(manifest, observed_identity)
    return RuntimeValidationReceipt(
        payload=expected_validation_payload(manifest),
        _seal=_SEAL,
    )


def validated_receipt_payload(
    receipt: RuntimeValidationReceipt,
    manifest: LocalAIManifest,
) -> dict[str, str]:
    """Validate an in-memory receipt and return a detached exact payload."""

    if (
        not isinstance(receipt, RuntimeValidationReceipt)
        or receipt._seal is not _SEAL
        or receipt.payload != expected_validation_payload(manifest)
    ):
        raise LocalValidationError("Model runtime validation receipt is invalid")
    return dict(receipt.payload)


def validate_persisted_receipt(
    value: object,
    manifest: LocalAIManifest,
) -> dict[str, str]:
    """Validate a bounded persisted receipt against one exact manifest."""

    if (
        not isinstance(value, dict)
        or set(value) != VALIDATION_RECEIPT_KEYS
        or value != expected_validation_payload(manifest)
        or not all(isinstance(item, str) for item in value.values())
    ):
        raise LocalValidationError("Model runtime validation receipt is invalid")
    return dict(value)


def validation_receipt_sha256(payload: dict[str, Any]) -> str:
    """Return the canonical digest embedded in activation pointers."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalValidationError(
            "Model runtime validation receipt is invalid"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()
