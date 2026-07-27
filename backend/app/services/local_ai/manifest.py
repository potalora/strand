"""Parse immutable, bounded local-AI model manifests."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.types import ModelRole

SCHEMA_VERSION = 1
MAX_MANIFEST_FILES = 64
MAX_MANIFEST_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_PACK_BYTES = 20 * 1024 * 1024 * 1024
MAX_DECODE_TOKENS = 1_000_000

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PACK_REVISION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
ALLOWED_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".json",
        ".txt",
        ".model",
        ".tiktoken",
        ".jinja",
        ".md",
        ".license",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".py", ".pyc", ".bin", ".pkl", ".pickle", ".so", ".dylib", ".exe"}
)
ALLOWED_LICENSES = frozenset({"apache-2.0", "bsd-2-clause", "bsd-3-clause", "mit"})
EXPECTED_ROLES = frozenset(ModelRole)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "pack_revision",
        "platform",
        "runtime",
        "validation_suite_version",
        "artifacts",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "role",
        "repository",
        "revision",
        "quantization",
        "license",
        "attribution",
        "decode_limits",
        "files",
    }
)
_FILE_KEYS = frozenset({"path", "sha256", "size"})
_RUNTIME_KEYS = frozenset({"name", "version"})
_DECODE_KEYS = frozenset({"max_input_tokens", "max_output_tokens"})


@dataclass(frozen=True)
class ManifestFile:
    """One hash-pinned, size-bounded file within a model artifact."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ManifestArtifact:
    """One immutable model selected for a logical local-AI role."""

    role: ModelRole
    repository: str
    revision: str
    quantization: str
    license: str
    attribution: str
    decode_limits: dict[str, int]
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class LocalAIManifest:
    """A complete three-role local-AI pack lock."""

    schema_version: int
    pack_revision: str
    platform: str
    runtime: dict[str, str]
    validation_suite_version: str
    artifacts: tuple[ManifestArtifact, ...]


def _require_object(value: Any, *, keys: frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LocalValidationError(f"Manifest {context} has an invalid structure")
    return value


def _require_string(raw: dict[str, Any], key: str, *, context: str) -> str:
    value = raw.get(key)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise LocalValidationError(f"Manifest {context} must be a nonempty string")
    return value


def safe_relative_manifest_path(raw_path: Any) -> str:
    """Return a canonical relative POSIX path or fail without echoing its value."""

    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or "\x00" in raw_path
    ):
        raise LocalValidationError("Manifest path is invalid")
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or raw_path in {".", ".."}
        or ".." in path.parts
        or "." in path.parts
        or str(path) != raw_path
        or not path.name
    ):
        raise LocalValidationError("Manifest path is invalid")
    return str(path)


def manifest_path_suffix(path: str) -> str:
    """Return the final suffix, rejecting non-lowercase manifest file types."""

    suffix = PurePosixPath(path).suffix
    if suffix != suffix.lower():
        raise LocalValidationError("Manifest file suffix must be lowercase")
    return suffix


def _safe_manifest_file(raw_value: Any) -> ManifestFile:
    raw = _require_object(raw_value, keys=_FILE_KEYS, context="file")
    path = safe_relative_manifest_path(raw.get("path"))
    suffix = manifest_path_suffix(path)
    if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        raise LocalValidationError("Manifest file type is forbidden")

    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
        raise LocalValidationError(
            "Manifest SHA-256 must be 64 lowercase hex characters"
        )

    size = raw.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise LocalValidationError("Manifest file size must be a positive integer")
    if size > MAX_MANIFEST_FILE_BYTES:
        raise LocalValidationError("Manifest file size exceeds the configured limit")
    return ManifestFile(path=path, sha256=sha256, size=size)


def _safe_decode_limits(raw_value: Any) -> dict[str, int]:
    raw = _require_object(
        raw_value,
        keys=_DECODE_KEYS,
        context="decode limits",
    )
    limits: dict[str, int] = {}
    for key in sorted(_DECODE_KEYS):
        value = raw.get(key)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > MAX_DECODE_TOKENS
        ):
            raise LocalValidationError("Manifest decode limits are invalid")
        limits[key] = value
    return limits


def _safe_artifact(raw_value: Any) -> ManifestArtifact:
    raw = _require_object(raw_value, keys=_ARTIFACT_KEYS, context="artifact")
    role_value = raw.get("role")
    if not isinstance(role_value, str):
        raise LocalValidationError("Manifest model role is invalid")
    try:
        role = ModelRole(role_value)
    except ValueError as exc:
        raise LocalValidationError("Manifest model role is invalid") from exc

    repository = _require_string(raw, "repository", context="repository")
    if REPOSITORY_RE.fullmatch(repository) is None:
        raise LocalValidationError("Manifest repository is invalid")

    revision = raw.get("revision")
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise LocalValidationError("Every model must use an immutable revision")

    quantization = _require_string(raw, "quantization", context="quantization")
    license_name = _require_string(raw, "license", context="license").lower()
    if license_name not in ALLOWED_LICENSES:
        raise LocalValidationError("Manifest license is not allowlisted")
    attribution = _require_string(raw, "attribution", context="attribution")
    decode_limits = _safe_decode_limits(raw.get("decode_limits"))

    files_value = raw.get("files")
    if not isinstance(files_value, list):
        raise LocalValidationError("Manifest files have an invalid structure")
    if not files_value:
        raise LocalValidationError("Manifest artifact files must be nonempty")
    files = tuple(_safe_manifest_file(file_value) for file_value in files_value)
    paths = [file.path for file in files]
    if len(paths) != len(set(paths)):
        raise LocalValidationError("Manifest artifact contains a duplicate path")
    return ManifestArtifact(
        role=role,
        repository=repository,
        revision=revision,
        quantization=quantization,
        license=license_name,
        attribution=attribution,
        decode_limits=decode_limits,
        files=files,
    )


def parse_manifest(raw_value: Any) -> LocalAIManifest:
    """Validate an in-memory version-1 immutable local-AI manifest payload."""

    raw = _require_object(raw_value, keys=_TOP_LEVEL_KEYS, context="root")
    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        raise LocalValidationError("Manifest schema version must be 1")

    pack_revision = _require_string(raw, "pack_revision", context="pack revision")
    if PACK_REVISION_RE.fullmatch(pack_revision) is None:
        raise LocalValidationError("Manifest pack revision is invalid")
    if raw.get("platform") != "apple_silicon":
        raise LocalValidationError("Manifest platform is invalid")

    runtime_raw = _require_object(
        raw.get("runtime"),
        keys=_RUNTIME_KEYS,
        context="runtime",
    )
    runtime = {
        "name": _require_string(runtime_raw, "name", context="runtime name"),
        "version": _require_string(runtime_raw, "version", context="runtime version"),
    }
    validation_suite_version = _require_string(
        raw,
        "validation_suite_version",
        context="validation suite version",
    )

    artifacts_value = raw.get("artifacts")
    if not isinstance(artifacts_value, list):
        raise LocalValidationError("Manifest artifacts have an invalid structure")
    artifacts = tuple(_safe_artifact(value) for value in artifacts_value)
    roles = [artifact.role for artifact in artifacts]
    if len(artifacts) != 3 or frozenset(roles) != EXPECTED_ROLES:
        raise LocalValidationError("Manifest must contain exactly one artifact per role")
    if len(roles) != len(set(roles)):
        raise LocalValidationError("Manifest must contain exactly one artifact per role")

    files = [file for artifact in artifacts for file in artifact.files]
    if len(files) > MAX_MANIFEST_FILES:
        raise LocalValidationError("Manifest file count exceeds the configured limit")
    if sum(file.size for file in files) > MAX_MANIFEST_PACK_BYTES:
        raise LocalValidationError("Manifest pack size exceeds the configured limit")

    return LocalAIManifest(
        schema_version=schema_version,
        pack_revision=pack_revision,
        platform="apple_silicon",
        runtime=runtime,
        validation_suite_version=validation_suite_version,
        artifacts=artifacts,
    )


def load_manifest(path: Path) -> LocalAIManifest:
    """Load and fully validate a version-1 immutable local-AI manifest."""

    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LocalValidationError("Manifest could not be read as JSON") from exc
    return parse_manifest(raw_value)


def canonicalize_manifest_snapshot(raw_value: Any) -> tuple[dict[str, Any], str]:
    """Return a detached canonical schema-v1 snapshot and lowercase digest."""

    manifest = parse_manifest(raw_value)
    canonical_bytes = json.dumps(
        asdict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    snapshot = json.loads(canonical_bytes)
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return deepcopy(snapshot), digest
