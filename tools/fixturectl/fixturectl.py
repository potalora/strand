#!/usr/bin/env python3
"""Create and verify privacy-preserving private-fixture manifests.

The manifest contains only normalized relative paths and file metadata.  It never
contains a source-root path or file contents.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import sys
from typing import Any, NoReturn
import unicodedata


_DATASET_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CREATED_AT = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_FIELDS = frozenset({"schema_version", "datasets"})
_DATASET_FIELDS = frozenset(
    {"allowed_top_level", "max_files", "max_total_bytes", "destination"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "created_at",
        "hash_algorithm",
        "file_count",
        "total_bytes",
        "files",
    }
)
_FILE_FIELDS = frozenset({"path", "size", "mode", "sha256"})
_READ_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024
_MAX_KEY_BYTES = 1024 * 1024
_PUBLIC_KEY_MODES = frozenset({0o400, 0o440, 0o444, 0o600, 0o640, 0o644})


class FixturePolicyError(ValueError):
    """A local dataset or its policy violates the fixture safety contract."""


class FixtureManifestError(ValueError):
    """A manifest is malformed or does not match the local dataset."""


class FixtureSignatureError(ValueError):
    """A signature operation violated the fixture signing contract."""


@dataclass(frozen=True)
class DatasetPolicy:
    dataset_id: str
    allowed_top_level: tuple[str, ...]
    max_files: int
    max_total_bytes: int
    destination: str


@dataclass(frozen=True)
class Policy:
    schema_version: int
    datasets: dict[str, DatasetPolicy]


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    mode: str
    sha256: str


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    dataset: str
    created_at: str
    hash_algorithm: str
    file_count: int
    total_bytes: int
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class VerifiedManifest:
    manifest: Manifest
    canonical_bytes: bytes
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixturePolicyError("duplicate key in JSON")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except FixturePolicyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixturePolicyError("policy JSON is unreadable or invalid") from exc


def _exact_fields(
    value: object,
    expected: frozenset[str],
    error_type: type[FixturePolicyError] | type[FixtureManifestError],
    message: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise error_type(message)
    if not all(isinstance(key, str) for key in value):
        raise error_type(message)
    return value


def _positive_limit(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe_component(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
        and unicodedata.normalize("NFC", value) == value
    )


def load_policy(path: Path) -> Policy:
    """Load a strict, source-path-free fixture policy."""

    body = _exact_fields(
        _read_strict_json(path),
        _POLICY_FIELDS,
        FixturePolicyError,
        "policy fields are invalid",
    )
    if body["schema_version"] != 1:
        raise FixturePolicyError("policy schema version is invalid")
    datasets_body = body["datasets"]
    if not isinstance(datasets_body, dict):
        raise FixturePolicyError("policy datasets are invalid")

    datasets: dict[str, DatasetPolicy] = {}
    for dataset_id, raw_dataset in datasets_body.items():
        if not isinstance(dataset_id, str) or not _DATASET_ID.fullmatch(dataset_id):
            raise FixturePolicyError("dataset id is invalid")
        dataset = _exact_fields(
            raw_dataset,
            _DATASET_FIELDS,
            FixturePolicyError,
            "dataset policy fields are invalid",
        )
        allowed = dataset["allowed_top_level"]
        if (
            not isinstance(allowed, list)
            or not allowed
            or not all(_safe_component(item) for item in allowed)
            or len({item.casefold() for item in allowed}) != len(allowed)
        ):
            raise FixturePolicyError("allowed top-level paths are invalid")
        if not _positive_limit(dataset["max_files"]) or not _positive_limit(
            dataset["max_total_bytes"]
        ):
            raise FixturePolicyError("dataset limit is invalid")

        expected_destination = f"/srv/private-fixtures/{dataset_id}"
        if dataset["destination"] != expected_destination:
            raise FixturePolicyError("dataset destination is invalid")

        datasets[dataset_id] = DatasetPolicy(
            dataset_id=dataset_id,
            allowed_top_level=tuple(allowed),
            max_files=dataset["max_files"],
            max_total_bytes=dataset["max_total_bytes"],
            destination=dataset["destination"],
        )
    return Policy(schema_version=1, datasets=datasets)


def _mode_bits(metadata: os.stat_result) -> int:
    return stat.S_IMODE(metadata.st_mode)


def _require_owner(metadata: os.stat_result) -> None:
    if metadata.st_uid != os.getuid():
        raise FixturePolicyError("dataset entry owner is invalid")


def _canonical_relative(path: Path, root: Path) -> str:
    raw = path.relative_to(root).as_posix()
    normalized = unicodedata.normalize("NFC", raw)
    if normalized != raw:
        raise FixturePolicyError("path collision or normalization is unsafe")
    return normalized


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_flags(*, directory: bool = False, nonblocking: bool = False) -> int:
    flags = os.O_RDONLY
    if directory and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if nonblocking and hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _hash_regular_file(
    name: str,
    directory_fd: int,
    expected: os.stat_result,
) -> str:
    try:
        descriptor = os.open(
            name,
            _read_flags(nonblocking=True),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise FixturePolicyError("regular file could not be opened safely") from exc

    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(expected):
            raise FixturePolicyError("dataset changed during manifest creation")
        while chunk := os.read(descriptor, _READ_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise FixturePolicyError("dataset changed during manifest creation")
    except OSError as exc:
        raise FixturePolicyError("regular file could not be read safely") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _classify_forbidden(metadata: os.stat_result) -> str:
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        return "symlink forbidden"
    if stat.S_ISFIFO(mode):
        return "fifo forbidden"
    if stat.S_ISSOCK(mode):
        return "socket forbidden"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "device forbidden"
    return "unsupported dataset entry forbidden"


def _open_dataset_root(root: Path) -> tuple[int, os.stat_result]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise FixturePolicyError("dataset root is inaccessible") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FixturePolicyError("dataset root must be a real directory")
    _require_owner(root_metadata)
    if _mode_bits(root_metadata) != 0o700:
        raise FixturePolicyError("dataset root mode must be 0700")
    try:
        root_fd = os.open(root, _read_flags(directory=True))
    except OSError as exc:
        raise FixturePolicyError("dataset root could not be opened safely") from exc
    try:
        anchored_root = os.fstat(root_fd)
        if _metadata_identity(anchored_root) != _metadata_identity(root_metadata):
            raise FixturePolicyError("dataset changed during manifest creation")
    except OSError as exc:
        os.close(root_fd)
        raise FixturePolicyError("dataset changed during manifest creation") from exc
    except BaseException:
        os.close(root_fd)
        raise
    return root_fd, anchored_root


def _scan_dataset(
    policy: DatasetPolicy,
    root: Path,
    *,
    anchored_root_fd: int | None = None,
) -> tuple[ManifestFile, ...]:
    if anchored_root_fd is None:
        root_fd, anchored_root = _open_dataset_root(root)
    else:
        try:
            root_fd = os.dup(anchored_root_fd)
        except OSError as exc:
            raise FixturePolicyError(
                "dataset changed during manifest creation"
            ) from exc
        try:
            anchored_root = os.fstat(root_fd)
            try:
                current_root = root.lstat()
            except OSError as exc:
                raise FixturePolicyError(
                    "dataset changed during manifest creation"
                ) from exc
            if _metadata_identity(anchored_root) != _metadata_identity(current_root):
                raise FixturePolicyError("dataset changed during manifest creation")
        except OSError as exc:
            os.close(root_fd)
            raise FixturePolicyError(
                "dataset changed during manifest creation"
            ) from exc
        except BaseException:
            os.close(root_fd)
            raise

    files: list[ManifestFile] = []
    collision_keys: set[str] = set()
    total_bytes = 0

    def visit(
        directory_fd: int,
        relative_parent: str,
        expected_directory: os.stat_result,
    ) -> None:
        nonlocal total_bytes
        try:
            before = os.fstat(directory_fd)
            if not stat.S_ISDIR(before.st_mode) or _metadata_identity(
                before
            ) != _metadata_identity(expected_directory):
                raise FixturePolicyError("dataset changed during manifest creation")
            with os.scandir(directory_fd) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise FixturePolicyError("dataset directory is inaccessible") from exc
        for entry in entries:
            try:
                metadata = os.stat(
                    entry.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise FixturePolicyError("dataset entry is inaccessible") from exc
            raw_relative = (
                f"{relative_parent}/{entry.name}" if relative_parent else entry.name
            )
            relative = unicodedata.normalize("NFC", raw_relative)
            if relative != raw_relative:
                raise FixturePolicyError("path collision or normalization is unsafe")
            top_level = relative.split("/", 1)[0]
            if top_level not in policy.allowed_top_level:
                raise FixturePolicyError("top-level path forbidden")
            collision_key = unicodedata.normalize("NFC", relative).casefold()
            if collision_key in collision_keys:
                raise FixturePolicyError("path collision detected")
            collision_keys.add(collision_key)
            _require_owner(metadata)

            if stat.S_ISDIR(metadata.st_mode):
                if _mode_bits(metadata) != 0o700:
                    raise FixturePolicyError("directory mode must be 0700")
                try:
                    child_fd = os.open(
                        entry.name,
                        _read_flags(directory=True),
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise FixturePolicyError(
                        "dataset directory could not be opened safely"
                    ) from exc
                try:
                    visit(child_fd, relative, metadata)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise FixturePolicyError(_classify_forbidden(metadata))
            if _mode_bits(metadata) != 0o600:
                raise FixturePolicyError("file mode must be 0600")
            if metadata.st_nlink != 1:
                raise FixturePolicyError("hardlink forbidden")

            size = metadata.st_size
            if size < 0:
                raise FixturePolicyError("file size is invalid")
            total_bytes += size
            if len(files) + 1 > policy.max_files:
                raise FixturePolicyError("file limit exceeded")
            if total_bytes > policy.max_total_bytes:
                raise FixturePolicyError("byte limit exceeded")
            files.append(
                ManifestFile(
                    path=relative,
                    size=size,
                    mode="0600",
                    sha256=_hash_regular_file(entry.name, directory_fd, metadata),
                )
            )

        try:
            after = os.fstat(directory_fd)
        except OSError as exc:
            raise FixturePolicyError(
                "dataset changed during manifest creation"
            ) from exc
        if _metadata_identity(before) != _metadata_identity(after):
            raise FixturePolicyError("dataset changed during manifest creation")

    try:
        visit(root_fd, "", anchored_root)
        return tuple(sorted(files, key=lambda item: item.path))
    finally:
        os.close(root_fd)


def _format_created_at(created_at: datetime) -> str:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise FixtureManifestError("created_at must be timezone-aware")
    return (
        created_at.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_manifest(policy: DatasetPolicy, root: Path, created_at: datetime) -> Manifest:
    """Build a deterministic manifest for a quiescent synthetic or private dataset root."""

    return _build_manifest(policy, root, created_at)


def _build_manifest(
    policy: DatasetPolicy,
    root: Path,
    created_at: datetime,
    *,
    anchored_root_fd: int | None = None,
) -> Manifest:
    timestamp = _format_created_at(created_at)
    files = _scan_dataset(policy, root, anchored_root_fd=anchored_root_fd)
    return Manifest(
        schema_version=1,
        dataset=policy.dataset_id,
        created_at=timestamp,
        hash_algorithm="sha256",
        file_count=len(files),
        total_bytes=sum(item.size for item in files),
        files=files,
    )


def canonical_json(manifest: Manifest) -> bytes:
    """Serialize a manifest using its one canonical UTF-8 representation."""

    return (
        json.dumps(
            asdict(manifest), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _manifest_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise FixtureManifestError("manifest path is invalid")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FixtureManifestError("manifest path is invalid")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        return normalized
    return value


def manifest_from_mapping(value: object) -> Manifest:
    """Parse an already-decoded manifest and enforce the exact v1 schema."""

    body = _exact_fields(
        value,
        _MANIFEST_FIELDS,
        FixtureManifestError,
        "manifest fields are invalid",
    )
    if body["schema_version"] != 1:
        raise FixtureManifestError("manifest schema version is invalid")
    dataset = body["dataset"]
    if not isinstance(dataset, str) or not _DATASET_ID.fullmatch(dataset):
        raise FixtureManifestError("manifest dataset is invalid")
    created_at = body["created_at"]
    if not isinstance(created_at, str) or not _CREATED_AT.fullmatch(created_at):
        raise FixtureManifestError("manifest created_at is invalid")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise FixtureManifestError("manifest created_at is invalid") from exc
    if body["hash_algorithm"] != "sha256":
        raise FixtureManifestError("manifest hash algorithm is invalid")
    if not _nonnegative_integer(body["file_count"]):
        raise FixtureManifestError("manifest file count is invalid")
    if not _nonnegative_integer(body["total_bytes"]):
        raise FixtureManifestError("manifest byte count is invalid")
    raw_files = body["files"]
    if not isinstance(raw_files, list):
        raise FixtureManifestError("manifest files are invalid")

    files: list[ManifestFile] = []
    collision_keys: set[str] = set()
    non_normalized_path_seen = False
    for raw_file in raw_files:
        file_body = _exact_fields(
            raw_file,
            _FILE_FIELDS,
            FixtureManifestError,
            "manifest file fields are invalid",
        )
        raw_path = file_body["path"]
        path = _manifest_path(raw_path)
        non_normalized_path_seen = non_normalized_path_seen or raw_path != path
        collision_key = unicodedata.normalize("NFC", path).casefold()
        if collision_key in collision_keys:
            raise FixtureManifestError("manifest path collision detected")
        collision_keys.add(collision_key)
        if not _nonnegative_integer(file_body["size"]):
            raise FixtureManifestError("manifest file size is invalid")
        if file_body["mode"] != "0600":
            raise FixtureManifestError("manifest file mode is invalid")
        digest = file_body["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise FixtureManifestError("manifest file hash is invalid")
        files.append(
            ManifestFile(
                path=path,
                size=file_body["size"],
                mode="0600",
                sha256=digest,
            )
        )

    if non_normalized_path_seen:
        raise FixtureManifestError("manifest path is not normalized")

    if body["file_count"] != len(files):
        raise FixtureManifestError("manifest file count does not match files")
    if body["total_bytes"] != sum(item.size for item in files):
        raise FixtureManifestError("manifest byte count does not match files")
    if [item.path for item in files] != sorted(item.path for item in files):
        raise FixtureManifestError("manifest file order is invalid")
    return Manifest(
        schema_version=1,
        dataset=dataset,
        created_at=created_at,
        hash_algorithm="sha256",
        file_count=len(files),
        total_bytes=body["total_bytes"],
        files=tuple(files),
    )


def _open_checked_regular(
    path: Path,
    *,
    error_type: type[FixtureManifestError] | type[FixtureSignatureError],
    label: str,
    allowed_modes: frozenset[int] | None,
    max_bytes: int,
) -> tuple[int, os.stat_result]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise error_type(f"{label} must be a regular file")
        if metadata.st_uid != os.getuid():
            raise error_type(f"{label} owner is invalid")
        if allowed_modes is not None and _mode_bits(metadata) not in allowed_modes:
            raise error_type(f"{label} mode is invalid")
        if metadata.st_nlink != 1:
            raise error_type(f"{label} hardlink is forbidden")
        if metadata.st_size < 0 or metadata.st_size > max_bytes:
            raise error_type(f"{label} size limit exceeded")
        descriptor = os.open(path, _read_flags(nonblocking=True))
        try:
            before = os.fstat(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        if _metadata_identity(before) != _metadata_identity(metadata):
            os.close(descriptor)
            raise error_type(f"{label} changed during read")
        return descriptor, before
    except (FixtureManifestError, FixtureSignatureError):
        raise
    except OSError as exc:
        raise error_type(f"{label} file is unavailable or unsafe") from exc


def _read_checked_descriptor(
    descriptor: int,
    before: os.stat_result,
    *,
    error_type: type[FixtureManifestError] | type[FixtureSignatureError],
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, _READ_SIZE):
            total += len(chunk)
            if total > max_bytes:
                raise error_type(f"{label} size limit exceeded")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise error_type(f"{label} changed during read")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return b"".join(chunks)
    except (FixtureManifestError, FixtureSignatureError):
        raise
    except OSError as exc:
        raise error_type(f"{label} file is unavailable or unsafe") from exc


def _read_checked_regular(
    path: Path,
    *,
    error_type: type[FixtureManifestError] | type[FixtureSignatureError],
    label: str,
    allowed_modes: frozenset[int] | None,
    max_bytes: int,
) -> bytes:
    descriptor, before = _open_checked_regular(
        path,
        error_type=error_type,
        label=label,
        allowed_modes=allowed_modes,
        max_bytes=max_bytes,
    )
    try:
        return _read_checked_descriptor(
            descriptor,
            before,
            error_type=error_type,
            label=label,
            max_bytes=max_bytes,
        )
    finally:
        os.close(descriptor)


def _manifest_from_bytes(raw: bytes) -> Manifest:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_manifest_duplicate_keys,
        )
    except FixtureManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureManifestError("manifest JSON is unreadable or invalid") from exc
    manifest = manifest_from_mapping(value)
    if raw != canonical_json(manifest):
        raise FixtureManifestError("manifest JSON is not canonical")
    return manifest


def _open_manifest(path: Path) -> tuple[int, os.stat_result, Manifest, bytes]:
    descriptor, before = _open_checked_regular(
        path,
        error_type=FixtureManifestError,
        label="manifest",
        allowed_modes=frozenset({0o600}),
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    try:
        raw = _read_checked_descriptor(
            descriptor,
            before,
            error_type=FixtureManifestError,
            label="manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        return descriptor, before, _manifest_from_bytes(raw), raw
    except BaseException:
        os.close(descriptor)
        raise


def load_manifest(path: Path) -> Manifest:
    """Load a strict canonical manifest without disclosing its path on failure."""

    descriptor, _, manifest, _ = _open_manifest(path)
    os.close(descriptor)
    return manifest


def _reject_manifest_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureManifestError("duplicate key in manifest JSON")
        result[key] = value
    return result


def _open_signature_output(path: Path) -> tuple[int, int, str, str, os.stat_result]:
    parent = path.parent
    parent_fd: int | None = None
    try:
        parent_metadata = parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise FixtureSignatureError("signature output parent is invalid")
        if parent_metadata.st_uid != os.getuid():
            raise FixtureSignatureError("signature output parent owner is invalid")
        if _mode_bits(parent_metadata) != 0o700:
            raise FixtureSignatureError("signature output parent mode is invalid")
        parent_fd = os.open(parent, _read_flags(directory=True))
        anchored_parent = os.fstat(parent_fd)
        if _metadata_identity(anchored_parent) != _metadata_identity(parent_metadata):
            raise FixtureSignatureError("signature output parent changed")
        if not _safe_component(path.name):
            raise FixtureSignatureError("signature output filename is invalid")
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FixtureSignatureError("signature output already exists")

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        for _ in range(32):
            temporary_name = f".fixturectl-signature-{secrets.token_hex(16)}.tmp"
            try:
                signature_fd = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            try:
                os.fchmod(signature_fd, 0o600)
                temporary_metadata = os.fstat(signature_fd)
            except BaseException:
                os.close(signature_fd)
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except OSError:
                    pass
                raise
            return (
                parent_fd,
                signature_fd,
                temporary_name,
                path.name,
                temporary_metadata,
            )
        raise FixtureSignatureError("signature temporary name allocation failed")
    except FixtureSignatureError:
        if parent_fd is not None:
            os.close(parent_fd)
        raise
    except OSError:
        if parent_fd is not None:
            os.close(parent_fd)
        raise FixtureSignatureError(
            "signature output is unavailable or unsafe"
        ) from None


def _unlink_if_same(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _same_file(current, expected):
            os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    except OSError:
        return


def _install_signature(
    parent_fd: int,
    signature_fd: int,
    temporary_name: str,
    final_name: str,
    temporary_metadata: os.stat_result,
) -> None:
    final_link_created = False
    try:
        generated = os.fstat(signature_fd)
        if (
            not _same_file(generated, temporary_metadata)
            or not stat.S_ISREG(generated.st_mode)
            or generated.st_uid != os.getuid()
            or generated.st_nlink != 1
            or generated.st_size <= 0
            or generated.st_size > _MAX_SIGNATURE_BYTES
        ):
            raise FixtureSignatureError("signature file changed during creation")
        os.fchmod(signature_fd, 0o600)
        os.fsync(signature_fd)
        generated = os.fstat(signature_fd)
        if _mode_bits(generated) != 0o600:
            raise FixtureSignatureError("signature file mode is invalid")
        os.link(
            temporary_name,
            final_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        final_link_created = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        installed = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not _same_file(installed, generated)
            or installed.st_nlink != 1
            or installed.st_uid != os.getuid()
            or _mode_bits(installed) != 0o600
        ):
            raise FixtureSignatureError("signature installation failed safely")
        os.fsync(parent_fd)
    except FixtureSignatureError:
        if final_link_created:
            _unlink_if_same(parent_fd, final_name, temporary_metadata)
        raise
    except OSError:
        if final_link_created:
            _unlink_if_same(parent_fd, final_name, temporary_metadata)
        raise FixtureSignatureError("signature installation failed safely") from None


def _require_stable_descriptor(
    descriptor: int,
    before: os.stat_result,
    label: str,
) -> None:
    try:
        after = os.fstat(descriptor)
    except OSError:
        raise FixtureSignatureError(
            f"{label} changed during signature operation"
        ) from None
    if _metadata_identity(before) != _metadata_identity(after):
        raise FixtureSignatureError(f"{label} changed during signature operation")


def _discard_subprocess_output(exc: BaseException) -> None:
    for attribute in ("output", "stdout", "stderr"):
        if hasattr(exc, attribute):
            try:
                setattr(exc, attribute, None)
            except (AttributeError, TypeError):
                pass


def _descriptor_path(descriptor: int) -> str:
    return f"/dev/fd/{descriptor}"


def sign_manifest(manifest: Path, secret_key: Path, signature: Path) -> None:
    """Create a detached Minisign signature over held, validated descriptors."""

    manifest_fd, manifest_before, _, _ = _open_manifest(manifest)
    secret_key_fd: int | None = None
    parent_fd: int | None = None
    signature_fd: int | None = None
    temporary_name: str | None = None
    temporary_metadata: os.stat_result | None = None
    installed = False
    try:
        secret_key_fd, secret_key_before = _open_checked_regular(
            secret_key,
            error_type=FixtureSignatureError,
            label="secret key",
            allowed_modes=frozenset({0o600}),
            max_bytes=_MAX_KEY_BYTES,
        )
        (
            parent_fd,
            signature_fd,
            temporary_name,
            final_name,
            temporary_metadata,
        ) = _open_signature_output(signature)
        pass_fds = (manifest_fd, secret_key_fd, signature_fd)
        try:
            subprocess.run(
                [
                    "minisign",
                    "-S",
                    "-s",
                    _descriptor_path(secret_key_fd),
                    "-m",
                    _descriptor_path(manifest_fd),
                    "-x",
                    _descriptor_path(signature_fd),
                ],
                check=True,
                timeout=60,
                stdin=None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=pass_fds,
            )
        except subprocess.TimeoutExpired as exc:
            _discard_subprocess_output(exc)
            raise FixtureSignatureError("signing command timed out") from None
        except subprocess.CalledProcessError as exc:
            _discard_subprocess_output(exc)
            raise FixtureSignatureError("manifest signing failed") from None
        except OSError:
            raise FixtureSignatureError("Minisign is unavailable") from None

        _require_stable_descriptor(manifest_fd, manifest_before, "manifest")
        _require_stable_descriptor(secret_key_fd, secret_key_before, "secret key")
        _install_signature(
            parent_fd,
            signature_fd,
            temporary_name,
            final_name,
            temporary_metadata,
        )
        installed = True
    finally:
        if (
            not installed
            and parent_fd is not None
            and temporary_name is not None
            and temporary_metadata is not None
        ):
            _unlink_if_same(parent_fd, temporary_name, temporary_metadata)
        if signature_fd is not None:
            os.close(signature_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        if secret_key_fd is not None:
            os.close(secret_key_fd)
        os.close(manifest_fd)


def verify_signature(
    manifest: Path,
    signature: Path,
    public_key: Path,
) -> VerifiedManifest:
    """Verify held bytes and return the exact verified manifest snapshot."""

    manifest_fd, manifest_before, parsed_manifest, manifest_bytes = _open_manifest(
        manifest
    )
    signature_fd: int | None = None
    public_key_fd: int | None = None
    try:
        signature_fd, signature_before = _open_checked_regular(
            signature,
            error_type=FixtureSignatureError,
            label="signature file",
            allowed_modes=frozenset({0o600}),
            max_bytes=_MAX_SIGNATURE_BYTES,
        )
        signature_bytes = _read_checked_descriptor(
            signature_fd,
            signature_before,
            error_type=FixtureSignatureError,
            label="signature file",
            max_bytes=_MAX_SIGNATURE_BYTES,
        )
        if not signature_bytes:
            raise FixtureSignatureError("signature file is empty")
        public_key_fd, public_key_before = _open_checked_regular(
            public_key,
            error_type=FixtureSignatureError,
            label="public key",
            allowed_modes=_PUBLIC_KEY_MODES,
            max_bytes=_MAX_KEY_BYTES,
        )
        public_key_bytes = _read_checked_descriptor(
            public_key_fd,
            public_key_before,
            error_type=FixtureSignatureError,
            label="public key",
            max_bytes=_MAX_KEY_BYTES,
        )
        if not public_key_bytes:
            raise FixtureSignatureError("public key is empty")
        pass_fds = (manifest_fd, signature_fd, public_key_fd)
        try:
            subprocess.run(
                [
                    "minisign",
                    "-V",
                    "-q",
                    "-p",
                    _descriptor_path(public_key_fd),
                    "-m",
                    _descriptor_path(manifest_fd),
                    "-x",
                    _descriptor_path(signature_fd),
                ],
                check=True,
                timeout=60,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                pass_fds=pass_fds,
            )
        except subprocess.TimeoutExpired as exc:
            _discard_subprocess_output(exc)
            raise FixtureSignatureError("signature verification timed out") from None
        except subprocess.CalledProcessError as exc:
            _discard_subprocess_output(exc)
            raise FixtureSignatureError("signature verification failed") from None
        except OSError:
            raise FixtureSignatureError("Minisign is unavailable") from None

        _require_stable_descriptor(manifest_fd, manifest_before, "manifest")
        _require_stable_descriptor(signature_fd, signature_before, "signature file")
        _require_stable_descriptor(public_key_fd, public_key_before, "public key")
        return VerifiedManifest(
            manifest=parsed_manifest,
            canonical_bytes=manifest_bytes,
            sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
    finally:
        if public_key_fd is not None:
            os.close(public_key_fd)
        if signature_fd is not None:
            os.close(signature_fd)
        os.close(manifest_fd)


def verify_manifest(manifest: Manifest, policy: DatasetPolicy, root: Path) -> None:
    """Fail unless a fresh scan exactly matches the supplied manifest."""

    if manifest.dataset != policy.dataset_id:
        raise FixtureManifestError("manifest mismatch")
    current = build_manifest(policy, root, datetime.now(UTC))
    if (
        manifest.schema_version != current.schema_version
        or manifest.hash_algorithm != current.hash_algorithm
        or manifest.file_count != current.file_count
        or manifest.total_bytes != current.total_bytes
        or manifest.files != current.files
    ):
        raise FixtureManifestError("manifest mismatch")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _directory_is_within(candidate_fd: int, root_fd: int) -> bool:
    try:
        root_metadata = os.fstat(root_fd)
        current_fd = os.dup(candidate_fd)
    except OSError as exc:
        raise FixtureManifestError("output containment check failed") from exc
    try:
        try:
            for _ in range(256):
                current = os.fstat(current_fd)
                if _same_file(current, root_metadata):
                    return True
                parent_fd = os.open(
                    "..", _read_flags(directory=True), dir_fd=current_fd
                )
                try:
                    parent = os.fstat(parent_fd)
                except BaseException:
                    os.close(parent_fd)
                    raise
                if _same_file(current, parent):
                    os.close(parent_fd)
                    return False
                os.close(current_fd)
                current_fd = parent_fd
        except OSError as exc:
            raise FixtureManifestError("output containment check failed") from exc
    finally:
        os.close(current_fd)
    raise FixtureManifestError("output directory ancestry is too deep")


def _write_private(
    path: Path,
    body: bytes,
    *,
    forbidden_root_fd: int | None = None,
) -> None:
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise FixtureManifestError(
                "manifest output parent must be a real directory"
            )
        if parent_metadata.st_uid != os.getuid():
            raise FixtureManifestError("manifest output parent owner is invalid")
        if _mode_bits(parent_metadata) != 0o700:
            raise FixtureManifestError("manifest output parent mode must be 0700")
        parent_fd = os.open(parent, _read_flags(directory=True))
    except FixtureManifestError:
        raise
    except OSError as exc:
        raise FixtureManifestError("manifest output parent is unsafe") from exc

    try:
        anchored_parent = os.fstat(parent_fd)
        if _metadata_identity(anchored_parent) != _metadata_identity(parent_metadata):
            raise FixtureManifestError("manifest output parent changed")
        if forbidden_root_fd is not None:
            if _directory_is_within(parent_fd, forbidden_root_fd):
                raise FixtureManifestError(
                    "manifest output must be outside the dataset"
                )

        name = path.name
        if not _safe_component(name):
            raise FixtureManifestError("manifest output filename is invalid")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise FixtureManifestError(
                "manifest output could not be created safely"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise FixtureManifestError(
                "manifest output could not be written safely"
            ) from exc
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    except OSError as exc:
        raise FixtureManifestError("manifest output operation failed safely") from exc
    finally:
        os.close(parent_fd)


def _die(message: str) -> NoReturn:
    sys.stderr.write(f"fixturectl: {message}\n")
    raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fixturectl")
    parser.add_argument("--policy", type=Path, required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    manifest = subcommands.add_parser("manifest")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    create = manifest_commands.add_parser("create")
    create.add_argument("--dataset", required=True)
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = manifest_commands.add_parser("verify")
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        try:
            dataset_policy = policy.datasets[args.dataset]
        except KeyError as exc:
            raise FixturePolicyError("dataset id is not authorized") from exc
        if args.manifest_command == "create":
            root_fd, _ = _open_dataset_root(args.root)
            try:
                manifest = _build_manifest(
                    dataset_policy,
                    args.root,
                    datetime.now(UTC),
                    anchored_root_fd=root_fd,
                )
                _write_private(
                    args.output,
                    canonical_json(manifest),
                    forbidden_root_fd=root_fd,
                )
            finally:
                os.close(root_fd)
        else:
            verify_manifest(load_manifest(args.manifest), dataset_policy, args.root)
    except (FixturePolicyError, FixtureManifestError, FixtureSignatureError, OSError):
        _die("validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
