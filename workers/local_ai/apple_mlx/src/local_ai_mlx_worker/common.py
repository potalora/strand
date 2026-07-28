"""Shared offline artifact validation and MLX generation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

Role = Literal["ocr", "extraction", "summary"]
Loader = Callable[..., tuple[object, object]]
Generate = Callable[..., str]

EXPECTED_ROLES = frozenset({"ocr", "extraction", "summary"})
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
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
FORBIDDEN_SUFFIXES = frozenset({".py", ".pyc", ".bin", ".pkl", ".pickle", ".so", ".dylib", ".exe"})
ALLOWED_LICENSES = frozenset({"apache-2.0", "bsd-2-clause", "bsd-3-clause", "mit"})
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_FILES = 64
MAX_MANIFEST_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_MANIFEST_PACK_BYTES = 20 * 1024 * 1024 * 1024
MAX_DECODE_TOKENS = 1_000_000
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

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
_RUNTIME_KEYS = frozenset({"name", "version"})
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
_DECODE_KEYS = frozenset({"max_input_tokens", "max_output_tokens"})
_FILE_KEYS = frozenset({"path", "sha256", "size"})
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "pack_revision",
        "platform",
        "runtime",
        "validation_suite_version",
        "roles",
        "role",
        "repository",
        "revision",
        "quantization",
        "license",
        "attribution",
        "manifest_sha256",
    }
)
_PACK_REVISION = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkerInputError(ValueError):
    """A request is malformed without exposing its content."""


class ArtifactUnavailableError(ValueError):
    """A locked artifact cannot be safely loaded."""


class GenerationError(RuntimeError):
    """A model returned unusable content."""


@dataclass(frozen=True)
class LoadedRole:
    """One manifest-verified role loaded from a local filesystem path."""

    role: Role
    model: object
    processor: object
    model_path: str
    decode_limits: dict[str, int]
    repository_files_used: frozenset[str]


def require_offline_environment() -> None:
    """Require the process-level offline flags set by the model manager."""

    if any(os.environ.get(name) != value for name, value in OFFLINE_ENVIRONMENT.items()):
        raise ArtifactUnavailableError("Local model loading requires offline mode.")


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _strict_json_loads(value: str) -> object:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _read_manifest(value: Mapping[str, object] | str | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise ArtifactUnavailableError("Locked manifest is invalid.")
        return dict(value)
    path = Path(value)
    try:
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ArtifactUnavailableError("Locked manifest is unavailable.")
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > MAX_MANIFEST_BYTES
        ):
            raise ArtifactUnavailableError("Locked manifest is unavailable.")
        parsed = _strict_json_loads(path.read_text(encoding="utf-8"))
    except ArtifactUnavailableError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ArtifactUnavailableError("Locked manifest is unavailable.") from None
    if type(parsed) is not dict:
        raise ArtifactUnavailableError("Locked manifest is invalid.")
    return parsed


def _plain_object(
    value: object,
    *,
    keys: frozenset[str],
    context: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ArtifactUnavailableError(f"Locked manifest {context} has an invalid structure.")
    return value


def _manifest_string(
    raw: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str:
    value = raw.get(key)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ArtifactUnavailableError(f"Locked manifest {context} is invalid.")
    return value


def _decode_limits(value: object) -> dict[str, int]:
    raw = _plain_object(value, keys=_DECODE_KEYS, context="decode limits")
    if not all(
        isinstance(item, int) and not isinstance(item, bool) and 0 < item <= MAX_DECODE_TOKENS
        for item in raw.values()
    ):
        raise ArtifactUnavailableError("Locked manifest decode limits are invalid.")
    return {
        "max_input_tokens": int(raw["max_input_tokens"]),
        "max_output_tokens": int(raw["max_output_tokens"]),
    }


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ArtifactUnavailableError("Manifest model path is invalid.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ArtifactUnavailableError("Manifest model path is invalid.")
    suffix = path.suffix
    if suffix != suffix.lower() or suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        raise ArtifactUnavailableError("Manifest model file type is invalid.")
    return path


def _validated_manifest(
    manifest: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    raw = _plain_object(manifest, keys=_TOP_LEVEL_KEYS, context="root")
    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ArtifactUnavailableError("Locked manifest is incompatible.")
    pack_revision = _manifest_string(raw, "pack_revision", context="pack revision")
    if _PACK_REVISION.fullmatch(pack_revision) is None:
        raise ArtifactUnavailableError("Locked manifest pack revision is invalid.")
    if raw.get("platform") != "apple_silicon":
        raise ArtifactUnavailableError("Locked manifest is incompatible.")
    runtime = _plain_object(raw.get("runtime"), keys=_RUNTIME_KEYS, context="runtime")
    if runtime != {"name": "mlx-vlm", "version": "0.5.0"}:
        raise ArtifactUnavailableError("Locked manifest is incompatible.")
    validation_suite_version = _manifest_string(
        raw,
        "validation_suite_version",
        context="validation suite version",
    )

    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 3:
        raise ArtifactUnavailableError("Locked manifest role set is invalid.")
    artifacts: dict[str, dict[str, object]] = {}
    total_files = 0
    total_bytes = 0
    for value in raw_artifacts:
        artifact = _plain_object(value, keys=_ARTIFACT_KEYS, context="artifact")
        role = artifact.get("role")
        if not isinstance(role, str) or role not in EXPECTED_ROLES or role in artifacts:
            raise ArtifactUnavailableError("Locked manifest role set is invalid.")
        repository = _manifest_string(artifact, "repository", context="repository")
        revision = _manifest_string(artifact, "revision", context="revision")
        quantization = _manifest_string(
            artifact,
            "quantization",
            context="quantization",
        )
        license_name = _manifest_string(artifact, "license", context="license")
        attribution = _manifest_string(
            artifact,
            "attribution",
            context="attribution",
        )
        if _REPOSITORY.fullmatch(repository) is None:
            raise ArtifactUnavailableError("Locked manifest repository is invalid.")
        if _COMMIT.fullmatch(revision) is None:
            raise ArtifactUnavailableError("Locked manifest revision is invalid.")
        if license_name not in ALLOWED_LICENSES:
            raise ArtifactUnavailableError("Locked manifest license is invalid.")
        limits = _decode_limits(artifact.get("decode_limits"))
        raw_files = artifact.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ArtifactUnavailableError("Model artifact file list is invalid.")
        declared: set[str] = set()
        for raw_file_value in raw_files:
            raw_file = _plain_object(
                raw_file_value,
                keys=_FILE_KEYS,
                context="file",
            )
            relative = str(_safe_relative_path(raw_file.get("path")))
            digest = raw_file.get("sha256")
            size = raw_file.get("size")
            if relative in declared:
                raise ArtifactUnavailableError("Model artifact file list contains a duplicate.")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ArtifactUnavailableError("Model artifact digest is invalid.")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or size > MAX_MANIFEST_FILE_BYTES
            ):
                raise ArtifactUnavailableError("Model artifact size is invalid.")
            declared.add(relative)
            total_files += 1
            total_bytes += size
        if total_files > MAX_MANIFEST_FILES or total_bytes > MAX_MANIFEST_PACK_BYTES:
            raise ArtifactUnavailableError("Locked manifest pack limits are invalid.")
        artifacts[role] = {
            "role": role,
            "repository": repository,
            "revision": revision,
            "quantization": quantization,
            "license": license_name,
            "attribution": attribution,
            "decode_limits": limits,
            "files": raw_files,
        }
    if frozenset(artifacts) != EXPECTED_ROLES:
        raise ArtifactUnavailableError("Locked manifest role set is invalid.")
    header = {
        "schema_version": schema_version,
        "pack_revision": pack_revision,
        "platform": "apple_silicon",
        "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
        "validation_suite_version": validation_suite_version,
    }
    return artifacts, header


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    try:
        value = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ArtifactUnavailableError("Locked manifest is invalid.") from None
    return hashlib.sha256(value).hexdigest()


def build_manifest_identity(
    manifest: Mapping[str, object],
    role: Role | str,
) -> dict[str, object]:
    """Build the exact non-PHI manifest identity expected by a worker request."""

    if role not in EXPECTED_ROLES:
        raise ArtifactUnavailableError("Locked manifest role set is invalid.")
    artifacts, header = _validated_manifest(manifest)
    artifact = artifacts[str(role)]
    return {
        **header,
        "roles": sorted(EXPECTED_ROLES),
        "role": role,
        "repository": artifact["repository"],
        "revision": artifact["revision"],
        "quantization": artifact["quantization"],
        "license": artifact["license"],
        "attribution": artifact["attribution"],
        "manifest_sha256": _manifest_digest(manifest),
    }


def _assert_expected_identity(
    manifest: Mapping[str, object],
    role: Role,
    expected_identity: Mapping[str, object],
) -> None:
    if type(expected_identity) is not dict or set(expected_identity) != _IDENTITY_KEYS:
        raise ArtifactUnavailableError("Locked manifest identity is invalid.")
    if dict(expected_identity) != build_manifest_identity(manifest, role):
        raise ArtifactUnavailableError("Locked manifest identity does not match.")


def _assert_real_directory(path: Path) -> Path:
    try:
        if not path.is_absolute() or path.is_symlink():
            raise ArtifactUnavailableError("Model artifact root is unavailable.")
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except ArtifactUnavailableError:
        raise
    except OSError:
        raise ArtifactUnavailableError("Model artifact root is unavailable.") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactUnavailableError("Model artifact root is unavailable.")
    return resolved


def _hash_regular_file(role_root: Path, relative: PurePosixPath) -> tuple[int, str]:
    target = role_root.joinpath(*relative.parts)
    try:
        current = role_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ArtifactUnavailableError("Model artifact path contains a symlink.")
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(role_root):
            raise ArtifactUnavailableError("Model artifact path escapes its role root.")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ArtifactUnavailableError("Model artifact file is invalid.")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except ArtifactUnavailableError:
        raise
    except OSError:
        raise ArtifactUnavailableError("Model artifact file is unavailable.") from None
    return metadata.st_size, digest.hexdigest()


def _contains_key(value: object, key: str) -> bool:
    if type(value) is dict:
        return key in value or any(_contains_key(item, key) for item in value.values())
    if type(value) is list:
        return any(_contains_key(item, key) for item in value)
    return False


def _assert_exact_artifact_tree(role_root: Path, declared: frozenset[str]) -> None:
    observed: set[str] = set()
    try:
        for path in role_root.rglob("*"):
            relative = path.relative_to(role_root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactUnavailableError("Model artifact path contains a symlink.")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ArtifactUnavailableError("Model artifact contains a forbidden file.")
            suffix = path.suffix.lower()
            if suffix == ".py":
                raise ArtifactUnavailableError("Model artifact contains repository Python.")
            if suffix in FORBIDDEN_SUFFIXES:
                raise ArtifactUnavailableError("Model artifact contains a forbidden file.")
            observed.add(relative)
    except ArtifactUnavailableError:
        raise
    except OSError:
        raise ArtifactUnavailableError("Model artifact tree is unavailable.") from None
    if observed != set(declared):
        raise ArtifactUnavailableError("Model artifact tree contains an undeclared file.")


def _reject_auto_map(role_root: Path, files: frozenset[str]) -> None:
    for relative in files:
        if not relative.endswith(".json"):
            continue
        path = role_root.joinpath(*PurePosixPath(relative).parts)
        try:
            if path.stat().st_size > MAX_ARTIFACT_JSON_BYTES:
                raise ArtifactUnavailableError("Model JSON artifact is too large.")
            value = _strict_json_loads(path.read_text(encoding="utf-8"))
        except ArtifactUnavailableError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
            raise ArtifactUnavailableError("Model JSON artifact is invalid.") from None
        file_name = PurePosixPath(relative).name
        is_configuration = file_name == "config.json" or file_name.endswith("_config.json")
        if is_configuration and _contains_key(value, "auto_map"):
            raise ArtifactUnavailableError("Model configuration contains auto_map.")


def _default_loader(path: str, **kwargs: object) -> tuple[object, object]:
    from mlx_vlm import utils

    model_path = _assert_real_directory(Path(path))
    options = dict(kwargs)
    lazy = options.pop("lazy", False)
    if not isinstance(lazy, bool):
        raise ArtifactUnavailableError("Model loader configuration is invalid.")

    # The public mlx_vlm.load() entrypoint first resolves a missing path through
    # snapshot_download(). Calling the local component loaders with a Path keeps
    # the verified artifact boundary explicit and fail-closed.
    model = utils.load_model(model_path, lazy=lazy, **options)
    image_processor = utils.load_image_processor(model_path, **options)
    eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    processor = utils.load_processor(
        model_path,
        True,
        eos_token_ids=eos_token_id,
        **options,
    )
    if image_processor is not None:
        processor.image_processor = image_processor
    return model, processor


def load_role(
    role: Role | str,
    locked_manifest: Mapping[str, object] | str | Path,
    model_dir: str | Path,
    *,
    expected_identity: Mapping[str, object] | None = None,
    trust_remote_code: bool = False,
    loader: Loader | None = None,
) -> LoadedRole:
    """Revalidate and load one exact role using an absolute local path only."""

    if role not in EXPECTED_ROLES:
        raise WorkerInputError("Local model role is invalid.")
    if trust_remote_code:
        raise ArtifactUnavailableError("Model repository remote code is disabled.")
    require_offline_environment()

    typed_role: Role = role  # type: ignore[assignment]
    manifest = _read_manifest(locked_manifest)
    artifacts, _header = _validated_manifest(manifest)
    if expected_identity is not None:
        _assert_expected_identity(manifest, typed_role, expected_identity)
    artifact = artifacts[typed_role]
    pack_root = _assert_real_directory(Path(model_dir))
    role_root = _assert_real_directory(pack_root / typed_role)
    if not role_root.is_relative_to(pack_root):
        raise ArtifactUnavailableError("Model role path escapes the pack root.")

    raw_files = artifact["files"]
    if not isinstance(raw_files, list):
        raise ArtifactUnavailableError("Model artifact file list is invalid.")
    declared: set[str] = set()
    for raw_file_value in raw_files:
        if not isinstance(raw_file_value, dict):
            raise ArtifactUnavailableError("Model artifact file entry is invalid.")
        relative = _safe_relative_path(raw_file_value["path"])
        relative_text = str(relative)
        declared.add(relative_text)
        size, digest = _hash_regular_file(role_root, relative)
        if size != raw_file_value["size"]:
            raise ArtifactUnavailableError("Model artifact size does not match manifest.")
        if digest != raw_file_value["sha256"]:
            raise ArtifactUnavailableError("Model artifact hash does not match manifest.")

    files = frozenset(declared)
    _assert_exact_artifact_tree(role_root, files)
    _reject_auto_map(role_root, files)
    selected_loader = loader or _default_loader
    try:
        model, processor = selected_loader(
            str(role_root),
            lazy=False,
            local_files_only=True,
            trust_remote_code=False,
        )
    except ArtifactUnavailableError:
        raise
    except Exception:
        raise ArtifactUnavailableError("Model artifact could not be loaded.") from None
    limits = artifact["decode_limits"]
    if not isinstance(limits, dict):
        raise ArtifactUnavailableError("Model decode limits are invalid.")
    return LoadedRole(
        role=typed_role,
        model=model,
        processor=processor,
        model_path=str(role_root),
        decode_limits={
            "max_input_tokens": int(limits["max_input_tokens"]),
            "max_output_tokens": int(limits["max_output_tokens"]),
        },
        repository_files_used=files,
    )


def load_role_from_payload(role: Role, payload: Mapping[str, object]) -> LoadedRole:
    """Load a role from absolute manifest and active-pack paths in a request."""

    manifest_path = payload.get("manifest_path")
    model_dir = payload.get("model_dir")
    identity = payload.get("manifest_identity")
    if (
        not isinstance(manifest_path, str)
        or not Path(manifest_path).is_absolute()
        or not isinstance(model_dir, str)
        or not Path(model_dir).is_absolute()
        or type(identity) is not dict
    ):
        raise WorkerInputError("Local worker model paths or identity are invalid.")
    return load_role(
        role,
        manifest_path,
        model_dir,
        expected_identity=identity,
        trust_remote_code=False,
    )


def requested_output_tokens(
    payload: Mapping[str, object],
    loaded: LoadedRole,
    *,
    role_cap: int,
) -> int:
    """Return the smallest positive request, manifest, and role token bound."""

    requested = payload.get("max_output_tokens", role_cap)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        raise WorkerInputError("Local worker output token limit is invalid.")
    return min(requested, loaded.decode_limits["max_output_tokens"], role_cap)


def bounded_json(value: object, *, max_bytes: int) -> str:
    """Serialize JSON deterministically under a hard byte limit."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        raise WorkerInputError("Local worker JSON input is invalid.") from None
    if len(encoded.encode("utf-8")) > max_bytes:
        raise WorkerInputError("Local worker JSON input exceeds its limit.")
    return encoded


def parse_json_object(value: str) -> dict[str, object]:
    """Parse a strict JSON object, rejecting duplicate keys and non-finite values."""

    try:
        parsed = _strict_json_loads(value)
    except (TypeError, json.JSONDecodeError, ValueError, RecursionError):
        raise GenerationError("Local generation returned invalid JSON.") from None
    if type(parsed) is not dict:
        raise GenerationError("Local generation returned invalid JSON.")
    return parsed


def _token_count(processor: object, value: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    try:
        if callable(encode):
            encoded = encode(value, add_special_tokens=True)
        elif callable(tokenizer):
            encoded = tokenizer(
                value,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            if isinstance(encoded, Mapping):
                encoded = encoded.get("input_ids")
        else:
            raise TypeError
        if (
            isinstance(encoded, str | bytes)
            or not isinstance(encoded, Sequence)
            or len(encoded) <= 0
        ):
            raise TypeError
        first = encoded[0]
        if isinstance(first, Sequence) and not isinstance(first, str | bytes):
            if len(encoded) != 1:
                raise TypeError
            return len(first)
        return len(encoded)
    except Exception:
        raise WorkerInputError("Local worker tokenizer could not bound input.") from None


def _runtime_context_limit_for(model: object, processor: object) -> int | None:
    candidates: list[object] = []
    tokenizer = getattr(processor, "tokenizer", processor)
    candidates.append(getattr(tokenizer, "model_max_length", None))
    config = getattr(model, "config", None)
    for value in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
        getattr(config, "llm_config", None),
    ):
        candidates.extend(
            (
                getattr(value, "max_position_embeddings", None),
                getattr(value, "model_max_length", None),
                getattr(value, "max_sequence_length", None),
            )
        )
    limits = [
        int(value)
        for value in candidates
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_DECODE_TOKENS
    ]
    return min(limits) if limits else None


def _runtime_context_limit(loaded: LoadedRole) -> int | None:
    return _runtime_context_limit_for(loaded.model, loaded.processor)


def _strip_terminal_eos_suffix(value: str, processor: object) -> str:
    """Remove only the tokenizer-declared EOS token at the generated suffix."""

    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token = getattr(tokenizer, "eos_token", None)
    if (
        not isinstance(eos_token, str)
        or not eos_token
        or len(eos_token) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in eos_token)
    ):
        return value
    candidate = value.rstrip()
    if not candidate.endswith(eos_token):
        return value
    return candidate[: -len(eos_token)].rstrip()


def validate_token_budget(
    loaded: LoadedRole,
    text_parts: Sequence[str],
    *,
    max_output_tokens: int,
) -> None:
    """Enforce manifest input tokens and the tokenizer/model context window."""

    text = "\n".join(text_parts)
    count = _token_count(loaded.processor, text)
    if count > loaded.decode_limits["max_input_tokens"]:
        raise WorkerInputError("Local worker input exceeds its token limit.")
    runtime_limit = _runtime_context_limit(loaded)
    if runtime_limit is not None and count + max_output_tokens > runtime_limit:
        raise WorkerInputError("Local worker input exceeds its model context token limit.")


def validate_scratch_image(
    value: object,
    scratch_value: object,
    *,
    expected_sha256: object | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> str:
    """Return one regular bounded PNG contained by the per-job scratch directory."""

    if (
        not isinstance(value, str)
        or not isinstance(scratch_value, str)
        or not Path(value).is_absolute()
        or not Path(scratch_value).is_absolute()
    ):
        raise WorkerInputError("Local worker image scratch path is invalid.")
    try:
        scratch = Path(scratch_value)
        if scratch.is_symlink():
            raise WorkerInputError("Local worker image scratch path is invalid.")
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise WorkerInputError("Local worker image scratch path is invalid.")
        target = Path(value)
        if target.is_symlink():
            raise WorkerInputError("Local worker image scratch path is invalid.")
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(scratch):
            raise WorkerInputError("Local worker image is outside job scratch.")
        relative = resolved.relative_to(scratch)
        current = scratch
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise WorkerInputError("Local worker image scratch path is invalid.")
        metadata = resolved.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > max_bytes
            or resolved.suffix.lower() != ".png"
        ):
            raise WorkerInputError("Local worker image is invalid.")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
                raise WorkerInputError("Local worker image digest is invalid.")
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise WorkerInputError("Local worker image digest does not match.")

        from PIL import Image, UnidentifiedImageError

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(resolved) as image:
                width, height = image.size
                if (
                    image.format != "PNG"
                    or isinstance(width, bool)
                    or isinstance(height, bool)
                    or not isinstance(width, int)
                    or not isinstance(height, int)
                    or width <= 0
                    or height <= 0
                    or width * height > max_pixels
                ):
                    raise WorkerInputError("Local worker image dimensions are invalid.")
                image.verify()
    except WorkerInputError:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise WorkerInputError("Local worker image is invalid.") from None
    return str(resolved)


def generate_content(
    *,
    model: object,
    processor: object,
    prompt: str,
    images: list[str],
    max_tokens: int,
    temperature: float,
    do_sample: bool,
    input_token_limit: int,
    enable_thinking: bool = False,
    template: str | None = None,
    mode: str | None = None,
    instructions: str | None = None,
) -> str:
    """Apply the local chat template and run quiet, bounded MLX generation."""

    if temperature != 0.0 or do_sample:
        raise WorkerInputError("Local worker sampling configuration is forbidden.")
    from mlx_vlm import apply_chat_template, generate

    config = getattr(model, "config", None)
    if config is None:
        raise GenerationError("Local generation failed.")
    template_options: dict[str, object] = {
        "enable_thinking": enable_thinking,
    }
    if template is not None:
        template_options["template"] = template
    if mode is not None:
        template_options["mode"] = mode
    if instructions is not None:
        template_options["instructions"] = instructions
    formatted = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(images),
        **template_options,
    )
    if not isinstance(formatted, str):
        raise GenerationError("Local generation failed.")
    count = _token_count(processor, formatted)
    if count > input_token_limit:
        raise WorkerInputError("Local worker input exceeds its token limit.")
    runtime_limit = _runtime_context_limit_for(model, processor)
    if runtime_limit is not None and count + max_tokens > runtime_limit:
        raise WorkerInputError("Local worker input exceeds its model context token limit.")
    result = generate(
        model,
        processor,
        formatted,
        image=images or None,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    text = getattr(result, "text", result)
    if not isinstance(text, str):
        raise GenerationError("Local generation failed.")
    return _strip_terminal_eos_suffix(text, processor)
