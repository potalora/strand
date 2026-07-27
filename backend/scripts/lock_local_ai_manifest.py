"""Resolve and hash a candidate local-AI catalog without activating any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import httpx

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    ALLOWED_LICENSES,
    ALLOWED_SUFFIXES,
    COMMIT_RE,
    EXPECTED_ROLES,
    FORBIDDEN_SUFFIXES,
    MAX_MANIFEST_FILE_BYTES,
    MAX_MANIFEST_FILES,
    MAX_MANIFEST_PACK_BYTES,
    PACK_REVISION_RE,
    REPOSITORY_RE,
    SHA256_RE,
    load_manifest,
    manifest_path_suffix,
    safe_relative_manifest_path,
)
from app.services.local_ai.types import ModelRole

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co"
MAX_HF_METADATA_BYTES = 8 * 1024 * 1024
MAX_JSON_METADATA_BYTES = 128 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MIN_FREE_BYTES_AFTER_CACHE = 15 * 1024 * 1024 * 1024

_CATALOG_KEYS = frozenset(
    {
        "schema_version",
        "pack_revision",
        "platform",
        "runtime",
        "validation_suite_version",
        "candidates",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "role",
        "repository",
        "quantization",
        "license",
        "attribution",
        "decode_limits",
    }
)
_RUNTIME_KEYS = frozenset({"name", "version"})
_DECODE_KEYS = frozenset({"max_input_tokens", "max_output_tokens"})
_REMOTE_CODE_KEYS = frozenset(
    {"auto_map", "requires_remote_code", "trust_remote_code"}
)


@dataclass(frozen=True)
class Candidate:
    role: ModelRole
    repository: str
    quantization: str
    license: str
    attribution: str
    decode_limits: dict[str, int]


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    size: int
    expected_sha256: str | None


@dataclass(frozen=True)
class RepositoryPlan:
    candidate: Candidate
    revision: str
    files: tuple[RepositoryFile, ...]


def _object(value: Any, keys: frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LocalValidationError(f"Candidate catalog {context} is invalid")
    return value


def _string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise LocalValidationError(f"Candidate catalog {context} is invalid")
    return value


def _load_catalog(path: Path) -> tuple[dict[str, Any], tuple[Candidate, ...]]:
    try:
        raw_value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalValidationError("Candidate catalog could not be read as JSON") from exc

    raw = _object(raw_value, _CATALOG_KEYS, "root")
    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise LocalValidationError("Candidate catalog schema version is invalid")
    if raw.get("platform") != "apple_silicon":
        raise LocalValidationError("Candidate catalog platform is invalid")
    pack_revision = _string(raw, "pack_revision", "pack revision")
    if PACK_REVISION_RE.fullmatch(pack_revision) is None:
        raise LocalValidationError("Candidate catalog pack revision is invalid")
    _string(raw, "validation_suite_version", "validation suite version")

    runtime = _object(raw.get("runtime"), _RUNTIME_KEYS, "runtime")
    _string(runtime, "name", "runtime name")
    _string(runtime, "version", "runtime version")

    candidate_values = raw.get("candidates")
    if not isinstance(candidate_values, list):
        raise LocalValidationError("Candidate catalog candidates are invalid")
    candidates: list[Candidate] = []
    for candidate_value in candidate_values:
        item = _object(candidate_value, _CANDIDATE_KEYS, "candidate")
        role_value = item.get("role")
        if not isinstance(role_value, str):
            raise LocalValidationError("Candidate catalog model role is invalid")
        try:
            role = ModelRole(role_value)
        except ValueError as exc:
            raise LocalValidationError("Candidate catalog model role is invalid") from exc

        repository = _string(item, "repository", "repository")
        if REPOSITORY_RE.fullmatch(repository) is None:
            raise LocalValidationError("Candidate catalog repository is invalid")
        quantization = _string(item, "quantization", "quantization")
        license_name = _string(item, "license", "license").lower()
        if license_name not in ALLOWED_LICENSES:
            raise LocalValidationError("Candidate catalog license is not allowlisted")
        attribution = _string(item, "attribution", "attribution")

        decode_raw = _object(item.get("decode_limits"), _DECODE_KEYS, "decode limits")
        decode_limits: dict[str, int] = {}
        for key in sorted(_DECODE_KEYS):
            value = decode_raw.get(key)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > 1_000_000
            ):
                raise LocalValidationError("Candidate catalog decode limits are invalid")
            decode_limits[key] = value
        candidates.append(
            Candidate(
                role=role,
                repository=repository,
                quantization=quantization,
                license=license_name,
                attribution=attribution,
                decode_limits=decode_limits,
            )
        )

    roles = [candidate.role for candidate in candidates]
    if len(candidates) != 3 or frozenset(roles) != EXPECTED_ROLES:
        raise LocalValidationError(
            "Candidate catalog must contain exactly one candidate per role"
        )
    if len(roles) != len(set(roles)):
        raise LocalValidationError(
            "Candidate catalog must contain exactly one candidate per role"
        )
    return raw, tuple(candidates)


def _contains_remote_code_requirement(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _REMOTE_CODE_KEYS and child not in (None, False, {}, []):
                return True
            if _contains_remote_code_requirement(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_remote_code_requirement(child) for child in value)
    return False


def _metadata_json(
    client: httpx.Client,
    candidate: Candidate,
) -> dict[str, Any]:
    url = f"{HF_BASE_URL}/api/models/{candidate.repository}/revision/main"
    try:
        with client.stream("GET", url, params={"blobs": "true"}) as response:
            response.raise_for_status()
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None:
                try:
                    declared_bytes = int(declared_length)
                except ValueError as exc:
                    raise LocalValidationError(
                        "Hugging Face metadata Content-Length is invalid"
                    ) from exc
                if declared_bytes < 0:
                    raise LocalValidationError(
                        "Hugging Face metadata Content-Length is invalid"
                    )
                if declared_bytes > MAX_HF_METADATA_BYTES:
                    raise LocalValidationError(
                        "Hugging Face metadata response is too large"
                    )

            content = bytearray()
            for chunk in response.iter_bytes(CHUNK_BYTES):
                if len(content) + len(chunk) > MAX_HF_METADATA_BYTES:
                    raise LocalValidationError(
                        "Hugging Face metadata response is too large"
                    )
                content.extend(chunk)
    except LocalValidationError:
        raise
    except httpx.HTTPError as exc:
        raise LocalValidationError("Hugging Face metadata request failed") from exc
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LocalValidationError("Hugging Face repository metadata is invalid") from exc
    if not isinstance(value, dict):
        raise LocalValidationError("Hugging Face repository metadata is invalid")
    return value


def _metadata_path(value: Any) -> str:
    try:
        return safe_relative_manifest_path(value)
    except LocalValidationError as exc:
        raise LocalValidationError("Repository metadata contains an unsafe path") from exc


def _validate_repository_metadata(
    candidate: Candidate,
    metadata: dict[str, Any],
) -> RepositoryPlan:
    revision = metadata.get("sha")
    if not isinstance(revision, str) or COMMIT_RE.fullmatch(revision) is None:
        raise LocalValidationError("Repository did not resolve to an immutable revision")
    if _contains_remote_code_requirement(metadata):
        raise LocalValidationError("Repository metadata requires remote code")

    card_data = metadata.get("cardData")
    if not isinstance(card_data, dict):
        raise LocalValidationError("Repository license metadata is missing")
    metadata_license = card_data.get("license")
    if not isinstance(metadata_license, str):
        raise LocalValidationError("Repository license metadata is missing")
    if metadata_license.lower() != candidate.license:
        raise LocalValidationError("Repository license metadata does not match the catalog")

    siblings = metadata.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise LocalValidationError("Repository file metadata is missing")

    files: list[RepositoryFile] = []
    seen_paths: set[str] = set()
    for sibling_value in siblings:
        if not isinstance(sibling_value, dict):
            raise LocalValidationError("Repository file metadata is invalid")
        path = _metadata_path(sibling_value.get("rfilename"))
        if path in seen_paths:
            raise LocalValidationError("Repository metadata contains a duplicate path")
        seen_paths.add(path)
        suffix = manifest_path_suffix(path)
        if suffix in FORBIDDEN_SUFFIXES:
            raise LocalValidationError("Repository contains a forbidden file type")
        if (
            sibling_value.get("type") == "symlink"
            or sibling_value.get("symlink") not in (None, False)
            or sibling_value.get("target") is not None
        ):
            raise LocalValidationError("Repository contains a symlink")
        if suffix not in ALLOWED_SUFFIXES:
            continue

        size = sibling_value.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise LocalValidationError("Repository file size metadata is unresolved")
        if size > MAX_MANIFEST_FILE_BYTES:
            raise LocalValidationError("Repository file exceeds the configured size limit")

        expected_sha256: str | None = None
        lfs_value = sibling_value.get("lfs")
        if lfs_value is not None:
            if not isinstance(lfs_value, dict):
                raise LocalValidationError("Repository LFS metadata is unresolved")
            lfs_size = lfs_value.get("size")
            lfs_sha256 = lfs_value.get("sha256")
            if (
                not isinstance(lfs_size, int)
                or isinstance(lfs_size, bool)
                or lfs_size != size
                or not isinstance(lfs_sha256, str)
                or SHA256_RE.fullmatch(lfs_sha256) is None
            ):
                raise LocalValidationError("Repository LFS metadata is unresolved")
            expected_sha256 = lfs_sha256
        elif suffix == ".safetensors":
            raise LocalValidationError("Repository LFS metadata is unresolved")
        files.append(
            RepositoryFile(
                path=path,
                size=size,
                expected_sha256=expected_sha256,
            )
        )

    if not files or not any(
        manifest_path_suffix(file.path) == ".safetensors" for file in files
    ):
        raise LocalValidationError("Repository has no allowlisted model artifact files")
    return RepositoryPlan(
        candidate=candidate,
        revision=revision,
        files=tuple(sorted(files, key=lambda item: item.path)),
    )


def _resolve_all_metadata(
    client: httpx.Client,
    candidates: tuple[Candidate, ...],
) -> tuple[RepositoryPlan, ...]:
    plans = tuple(
        _validate_repository_metadata(candidate, _metadata_json(client, candidate))
        for candidate in candidates
    )
    files = [file for plan in plans for file in plan.files]
    if len(files) > MAX_MANIFEST_FILES:
        raise LocalValidationError("Candidate pack file count exceeds the configured limit")
    if sum(file.size for file in files) > MAX_MANIFEST_PACK_BYTES:
        raise LocalValidationError("Candidate pack size exceeds the configured limit")
    return plans


def _require_validation_cache_capacity(
    output_parent: Path,
    plans: tuple[RepositoryPlan, ...],
) -> None:
    planned_bytes = sum(file.size for plan in plans for file in plan.files)
    free_bytes = shutil.disk_usage(output_parent).free
    if free_bytes - planned_bytes < MIN_FREE_BYTES_AFTER_CACHE:
        raise LocalValidationError(
            "Candidate validation cache would leave less than 15 GiB free"
        )


def _download_url(plan: RepositoryPlan, file: RepositoryFile) -> str:
    path = quote(file.path, safe="/")
    return (
        f"{HF_BASE_URL}/{plan.candidate.repository}/resolve/"
        f"{plan.revision}/{path}"
    )


def _stream_file(
    client: httpx.Client,
    plan: RepositoryPlan,
    file: RepositoryFile,
    cache_root: Path,
    bytes_seen: list[int],
) -> dict[str, Any]:
    destination = cache_root / plan.candidate.role.value / file.path
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            try:
                with client.stream("GET", _download_url(plan, file)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(CHUNK_BYTES):
                        if not chunk:
                            continue
                        count += len(chunk)
                        bytes_seen[0] += len(chunk)
                        if (
                            count > file.size
                            or count > MAX_MANIFEST_FILE_BYTES
                            or bytes_seen[0] > MAX_MANIFEST_PACK_BYTES
                        ):
                            raise LocalValidationError(
                                "Downloaded bytes exceed repository metadata"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
            except httpx.HTTPError as exc:
                raise LocalValidationError("Model artifact download failed") from exc
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    sha256 = digest.hexdigest()
    if count != file.size:
        destination.unlink(missing_ok=True)
        raise LocalValidationError("Downloaded bytes do not match repository metadata")
    if file.expected_sha256 is not None and sha256 != file.expected_sha256:
        destination.unlink(missing_ok=True)
        raise LocalValidationError("Downloaded hash does not match repository metadata")
    return {"path": file.path, "sha256": sha256, "size": count}


def _json_metadata_files(plans: tuple[RepositoryPlan, ...]) -> Iterator[tuple[int, int]]:
    for plan_index, plan in enumerate(plans):
        for file_index, file in enumerate(plan.files):
            if manifest_path_suffix(file.path) == ".json":
                yield plan_index, file_index


def _inspect_downloaded_json(path: Path, expected_size: int) -> None:
    if expected_size > MAX_JSON_METADATA_BYTES:
        raise LocalValidationError("Repository JSON metadata exceeds the inspection limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalValidationError("Repository JSON metadata is invalid") from exc
    if _contains_remote_code_requirement(value):
        raise LocalValidationError("Repository configuration requires remote code")


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(output_path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def lock_catalog(catalog_path: Path, output_path: Path) -> int:
    """Resolve, inspect, hash, and atomically write all three candidates.

    This operation has no document inputs and never installs or activates a pack.
    """

    catalog, candidates = _load_catalog(catalog_path)
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    headers = {"User-Agent": "MedTimeline-local-ai-lock/1"}
    timeout = httpx.Timeout(60.0, connect=15.0)
    with httpx.Client(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        # Phase 1 inspects every repository listing before requesting any file.
        plans = _resolve_all_metadata(client, candidates)
        _require_validation_cache_capacity(output_path.parent, plans)
        bytes_seen = [0]
        with tempfile.TemporaryDirectory(
            prefix=".local-ai-lock-",
            dir=output_path.parent,
        ) as temporary_name:
            cache_root = Path(temporary_name)
            cache_root.chmod(0o700)
            locked_files: dict[tuple[int, int], dict[str, Any]] = {}

            # Phase 2 downloads and inspects all bounded JSON configuration before
            # any model weight bytes can be requested.
            for plan_index, file_index in _json_metadata_files(plans):
                plan = plans[plan_index]
                file = plan.files[file_index]
                locked_files[(plan_index, file_index)] = _stream_file(
                    client,
                    plan,
                    file,
                    cache_root,
                    bytes_seen,
                )
                _inspect_downloaded_json(
                    cache_root / plan.candidate.role.value / file.path,
                    file.size,
                )

            artifacts: list[dict[str, Any]] = []
            for plan_index, plan in enumerate(plans):
                files: list[dict[str, Any]] = []
                for file_index, file in enumerate(plan.files):
                    locked_file = locked_files.get((plan_index, file_index))
                    if locked_file is None:
                        locked_file = _stream_file(
                            client,
                            plan,
                            file,
                            cache_root,
                            bytes_seen,
                        )
                    files.append(locked_file)
                artifacts.append(
                    {
                        "role": plan.candidate.role.value,
                        "repository": plan.candidate.repository,
                        "revision": plan.revision,
                        "quantization": plan.candidate.quantization,
                        "license": plan.candidate.license,
                        "attribution": plan.candidate.attribution,
                        "decode_limits": plan.candidate.decode_limits,
                        "files": files,
                    }
                )

            lock_value = {
                "schema_version": catalog["schema_version"],
                "pack_revision": catalog["pack_revision"],
                "platform": catalog["platform"],
                "runtime": catalog["runtime"],
                "validation_suite_version": catalog["validation_suite_version"],
                "artifacts": artifacts,
            }
            temporary_lock = cache_root / "candidate.lock.json"
            temporary_lock.write_bytes(_canonical_bytes(lock_value))
            temporary_lock.chmod(0o600)
            load_manifest(temporary_lock)
            _atomic_write(output_path, temporary_lock.read_bytes())
    return len(candidates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and hash the strict-local model candidate catalog."
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        count = lock_catalog(args.catalog, args.output)
    except LocalValidationError as exc:
        logger.error("catalog lock failed: %s", exc)
        return 1
    logger.info("locked %d candidate artifacts", count)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
