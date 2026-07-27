"""Deterministic, versioned keys for resumable local-AI checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from app.services.local_ai.errors import LocalValidationError

_KEY_FORMAT_VERSION = 1
_MAX_VERSION_LENGTH = 128
_MAX_VERSION_UTF8_BYTES = 256
_SHA256_HEX = frozenset("0123456789abcdef")


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise LocalValidationError("Checkpoint digest must be lowercase SHA-256.")
    return value


def _version(value: object) -> str:
    try:
        encoded = value.encode("utf-8") if isinstance(value, str) else b""
    except UnicodeEncodeError:
        encoded = b""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_VERSION_LENGTH
        or not encoded
        or len(encoded) > _MAX_VERSION_UTF8_BYTES
    ):
        raise LocalValidationError("Checkpoint version is invalid.")
    return value


def _page_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LocalValidationError("Checkpoint page number must be a positive integer.")
    return value


def _validate_scalar(value: object) -> None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        raise LocalValidationError("Checkpoint dependency must be a finite scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise LocalValidationError("Checkpoint dependency must be a finite scalar.")


def _key(kind: str, payload: dict[str, Any]) -> str:
    for value in payload.values():
        _validate_scalar(value)
    encoded = json.dumps(
        {
            "key_format_version": _KEY_FORMAT_VERSION,
            "kind": kind,
            **payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ocr_checkpoint_key(
    upload_hash: str,
    page_number: int,
    raster_version: str,
    manifest_digest: str,
) -> str:
    """Return the checkpoint key for one rasterized OCR page."""
    return _key(
        "ocr",
        {
            "manifest_digest": _digest(manifest_digest),
            "page_number": _page_number(page_number),
            "raster_version": _version(raster_version),
            "upload_hash": _digest(upload_hash),
        },
    )


def extraction_checkpoint_key(
    ocr_text_hash: str,
    schema_version: str,
    prompt_version: str,
    manifest_digest: str,
) -> str:
    """Return the checkpoint key for grounded clinical extraction."""
    return _key(
        "extraction",
        {
            "manifest_digest": _digest(manifest_digest),
            "ocr_text_hash": _digest(ocr_text_hash),
            "prompt_version": _version(prompt_version),
            "schema_version": _version(schema_version),
        },
    )


def summary_checkpoint_key(
    validated_fact_hash: str,
    summary_schema_version: str,
    manifest_digest: str,
) -> str:
    """Return the checkpoint key for a grounded local summary."""
    return _key(
        "summary",
        {
            "manifest_digest": _digest(manifest_digest),
            "summary_schema_version": _version(summary_schema_version),
            "validated_fact_hash": _digest(validated_fact_hash),
        },
    )
