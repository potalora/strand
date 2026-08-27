#!/usr/bin/env python3
"""Create and verify privacy-preserving private-fixture manifests.

The manifest contains only normalized relative paths and file metadata.  It never
contains a source-root path or file contents.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import errno
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
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
_PROMOTION_THREAD_LOCKS_GUARD = threading.Lock()
_PROMOTION_THREAD_LOCKS: dict[str, threading.Lock] = {}
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_SSH_DESTINATION = re.compile(
    r"^(?:[a-z_][a-z0-9_-]{0,31}@)?[a-z0-9](?:[a-z0-9.-]{0,252}[a-z0-9])?$"
)
_TRANSFER_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "source_alias",
        "state_directory",
        "ssh_executable",
        "rsync_executable",
        "rsync_version",
        "targets",
    }
)
_TRANSFER_TARGET_FIELDS = frozenset(
    {
        "ssh_destination",
        "target_id",
        "remote_fixturectl",
    }
)
_RECEIVER_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "target_id",
        "policy",
        "public_key",
        "fixturectl_executable",
        "rsync_executable",
        "rsync_version",
    }
)
_RECEIVER_CONFIG_PATH = Path("/etc/fixturectl/receiver.json")
_RECEIVER_PROTOCOL_VERSION = 1
_PINNED_RSYNC_VERSION = "3.5.0"
_PINNED_RSYNC_PROTOCOL = 32
_RSYNC_DATA_SERVER_OPTIONS = "-logDtpRe.LsfxCIvu"
_RSYNC_METADATA_SERVER_OPTIONS = "-logDtpre.iLsfxCIvu"


class FixturePolicyError(ValueError):
    """A local dataset or its policy violates the fixture safety contract."""


class FixtureManifestError(ValueError):
    """A manifest is malformed or does not match the local dataset."""


class FixtureSignatureError(ValueError):
    """A signature operation violated the fixture signing contract."""


class FixtureTransferError(ValueError):
    """A transfer or release operation violated the fixture safety contract."""


class FixtureTransferApplyError(FixtureTransferError):
    """An applied transfer failed after remote staging was created."""


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
    signature_sha256: str | None = None


@dataclass(frozen=True)
class TransferPlan:
    dataset: str
    source: Path
    destination: str
    manifest_sha256: str
    files_from: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class TransferTarget:
    alias: str
    ssh_destination: str
    target_id: str
    remote_fixturectl: str


@dataclass(frozen=True)
class ReceiverConfig:
    schema_version: int
    protocol_version: int
    target_id: str
    policy: Path
    public_key: Path
    fixturectl_executable: str
    rsync_executable: str
    rsync_version: str


@dataclass(frozen=True)
class TransferConfig:
    schema_version: int
    source_alias: str
    state_directory: Path
    ssh_executable: str
    rsync_executable: str
    rsync_version: str
    targets: dict[str, TransferTarget]


@dataclass(frozen=True)
class TransferResult:
    plan: TransferPlan
    summary: str
    applied: bool
    transfer_id: str | None
    receipt_path: Path | None


@dataclass(frozen=True)
class Receipt:
    schema_version: int
    dataset: str
    manifest_sha256: str
    file_count: int
    total_bytes: int
    verified_at: str
    target_id: str
    status: str


@dataclass(frozen=True)
class StagingLayout:
    dataset_root: Path
    staging: Path
    transfer_id: str
    root_fd: int
    incoming_fd: int
    staging_fd: int
    data_fd: int

    def close(self) -> None:
        for descriptor in (
            self.data_fd,
            self.staging_fd,
            self.incoming_fd,
            self.root_fd,
        ):
            os.close(descriptor)


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


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return identity/security fields stable across child-directory changes."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        _mode_bits(metadata),
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
    error_type: (
        type[FixtureManifestError]
        | type[FixtureSignatureError]
        | type[FixtureTransferError]
    ),
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
    except (FixtureManifestError, FixtureSignatureError, FixtureTransferError):
        raise
    except OSError as exc:
        raise error_type(f"{label} file is unavailable or unsafe") from exc


def _read_checked_descriptor(
    descriptor: int,
    before: os.stat_result,
    *,
    error_type: (
        type[FixtureManifestError]
        | type[FixtureSignatureError]
        | type[FixtureTransferError]
    ),
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
    except (FixtureManifestError, FixtureSignatureError, FixtureTransferError):
        raise
    except OSError as exc:
        raise error_type(f"{label} file is unavailable or unsafe") from exc


def _read_checked_regular(
    path: Path,
    *,
    error_type: (
        type[FixtureManifestError]
        | type[FixtureSignatureError]
        | type[FixtureTransferError]
    ),
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
            signature_sha256=hashlib.sha256(signature_bytes).hexdigest(),
        )
    finally:
        if public_key_fd is not None:
            os.close(public_key_fd)
        if signature_fd is not None:
            os.close(signature_fd)
        os.close(manifest_fd)


def verify_manifest(
    manifest: Manifest,
    policy: DatasetPolicy,
    root: Path,
    *,
    _anchored_root_fd: int | None = None,
) -> None:
    """Fail unless a fresh scan exactly matches the supplied manifest."""

    if manifest.dataset != policy.dataset_id:
        raise FixtureManifestError("manifest mismatch")
    current = _build_manifest(
        policy,
        root,
        datetime.now(UTC),
        anchored_root_fd=_anchored_root_fd,
    )
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


def _validate_verified_manifest(verified: VerifiedManifest) -> None:
    canonical = canonical_json(verified.manifest)
    if canonical != verified.canonical_bytes:
        raise FixtureTransferError("verified manifest bytes are inconsistent")
    if hashlib.sha256(canonical).hexdigest() != verified.sha256:
        raise FixtureTransferError("verified manifest digest is inconsistent")


def _reject_transfer_config_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureTransferError("duplicate key in transfer config")
        result[key] = value
    return result


def _safe_remote_absolute_path(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("/"):
        return False
    path = PurePosixPath(value)
    return (
        str(path) == value
        and path.is_absolute()
        and len(path.parts) > 1
        and all(_safe_component(part) for part in path.parts[1:])
    )


def _tailnet_ssh_destination(value: object) -> bool:
    if not isinstance(value, str) or not _SSH_DESTINATION.fullmatch(value):
        return False
    host = value.rsplit("@", 1)[-1]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address in ipaddress.ip_network(
        "100.64.0.0/10"
    ) or address in ipaddress.ip_network("fd7a:115c:a1e0::/48")


def load_transfer_config(path: Path) -> TransferConfig:
    raw = _read_checked_regular(
        path,
        error_type=FixtureTransferError,
        label="transfer config",
        allowed_modes=frozenset({0o600}),
        max_bytes=64 * 1024,
    )
    try:
        body = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_transfer_config_duplicate_keys,
        )
    except FixtureTransferError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise FixtureTransferError("transfer config JSON is invalid") from None
    if not isinstance(body, dict) or set(body) != _TRANSFER_CONFIG_FIELDS:
        raise FixtureTransferError("transfer config fields are invalid")
    if (
        body["schema_version"] != 1
        or not isinstance(body["source_alias"], str)
        or not _DATASET_ID.fullmatch(body["source_alias"])
        or not isinstance(body["state_directory"], str)
        or not Path(body["state_directory"]).is_absolute()
        or not _safe_remote_absolute_path(body["ssh_executable"])
        or not _safe_remote_absolute_path(body["rsync_executable"])
        or body["rsync_version"] != _PINNED_RSYNC_VERSION
        or not isinstance(body["targets"], dict)
        or not body["targets"]
        or len(body["targets"]) > 32
    ):
        raise FixtureTransferError("transfer config values are invalid")

    targets: dict[str, TransferTarget] = {}
    for alias, value in body["targets"].items():
        if (
            not isinstance(alias, str)
            or not _DATASET_ID.fullmatch(alias)
            or not isinstance(value, dict)
            or set(value) != _TRANSFER_TARGET_FIELDS
        ):
            raise FixtureTransferError("transfer target fields are invalid")
        if (
            not _tailnet_ssh_destination(value["ssh_destination"])
            or not isinstance(value["target_id"], str)
            or not _DATASET_ID.fullmatch(value["target_id"])
            or not _safe_remote_absolute_path(value["remote_fixturectl"])
        ):
            raise FixtureTransferError("transfer target values are invalid")
        targets[alias] = TransferTarget(
            alias=alias,
            ssh_destination=value["ssh_destination"],
            target_id=value["target_id"],
            remote_fixturectl=value["remote_fixturectl"],
        )

    state_directory = Path(body["state_directory"])
    state_fd, _ = _open_private_directory(
        state_directory,
        "transfer state directory",
    )
    os.close(state_fd)
    return TransferConfig(
        schema_version=1,
        source_alias=body["source_alias"],
        state_directory=state_directory,
        ssh_executable=body["ssh_executable"],
        rsync_executable=body["rsync_executable"],
        rsync_version=body["rsync_version"],
        targets=targets,
    )


def load_receiver_config(path: Path = _RECEIVER_CONFIG_PATH) -> ReceiverConfig:
    raw = _read_checked_regular(
        path,
        error_type=FixtureTransferError,
        label="receiver config",
        allowed_modes=frozenset({0o600}),
        max_bytes=64 * 1024,
    )
    try:
        body = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_transfer_config_duplicate_keys,
        )
    except FixtureTransferError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise FixtureTransferError("receiver config JSON is invalid") from None
    if not isinstance(body, dict) or set(body) != _RECEIVER_CONFIG_FIELDS:
        raise FixtureTransferError("receiver config fields are invalid")
    if (
        body["schema_version"] != 1
        or body["protocol_version"] != _RECEIVER_PROTOCOL_VERSION
        or not isinstance(body["target_id"], str)
        or not _DATASET_ID.fullmatch(body["target_id"])
        or not _safe_remote_absolute_path(body["policy"])
        or not _safe_remote_absolute_path(body["public_key"])
        or not _safe_remote_absolute_path(body["fixturectl_executable"])
        or not _safe_remote_absolute_path(body["rsync_executable"])
        or body["rsync_version"] != _PINNED_RSYNC_VERSION
    ):
        raise FixtureTransferError("receiver config values are invalid")
    return ReceiverConfig(
        schema_version=1,
        protocol_version=_RECEIVER_PROTOCOL_VERSION,
        target_id=body["target_id"],
        policy=Path(body["policy"]),
        public_key=Path(body["public_key"]),
        fixturectl_executable=body["fixturectl_executable"],
        rsync_executable=body["rsync_executable"],
        rsync_version=body["rsync_version"],
    )


def _open_private_directory(path: Path, label: str) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FixtureTransferError(f"{label} must be a real directory")
        if metadata.st_uid != os.getuid():
            raise FixtureTransferError(f"{label} owner is invalid")
        if _mode_bits(metadata) != 0o700:
            raise FixtureTransferError(f"{label} mode is invalid")
        descriptor = os.open(path, _read_flags(directory=True))
        anchored = os.fstat(descriptor)
        if _directory_identity(anchored) != _directory_identity(metadata):
            os.close(descriptor)
            descriptor = None
            raise FixtureTransferError(f"{label} changed during validation")
        return descriptor, anchored
    except FixtureTransferError:
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise FixtureTransferError(f"{label} is unavailable or unsafe") from None


def _open_private_child_directory(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    if not _safe_component(name):
        raise FixtureTransferError(f"{label} name is invalid")
    descriptor: int | None = None
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FixtureTransferError(f"{label} must be a real directory")
        if metadata.st_uid != os.getuid():
            raise FixtureTransferError(f"{label} owner is invalid")
        if _mode_bits(metadata) != 0o700:
            raise FixtureTransferError(f"{label} mode is invalid")
        descriptor = os.open(
            name,
            _read_flags(directory=True),
            dir_fd=parent_fd,
        )
        anchored = os.fstat(descriptor)
        if _directory_identity(anchored) != _directory_identity(metadata):
            os.close(descriptor)
            descriptor = None
            raise FixtureTransferError(f"{label} changed during validation")
        return descriptor, anchored
    except FixtureTransferError:
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise FixtureTransferError(f"{label} is unavailable or unsafe") from None


def _require_linked_directory(
    descriptor: int,
    parent_fd: int,
    name: str,
    label: str,
) -> None:
    try:
        anchored = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise FixtureTransferError(f"{label} changed") from None
    if _directory_identity(anchored) != _directory_identity(linked):
        raise FixtureTransferError(f"{label} changed")


def _open_held_transfer_file_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[int, os.stat_result, bytes]:
    if not _safe_component(name):
        raise FixtureTransferError(f"{label} filename is invalid")
    descriptor: int | None = None
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise FixtureTransferError(f"{label} must be a regular file")
        if metadata.st_uid != os.getuid():
            raise FixtureTransferError(f"{label} owner is invalid")
        if _mode_bits(metadata) != 0o600:
            raise FixtureTransferError(f"{label} mode is invalid")
        if metadata.st_nlink != 1:
            raise FixtureTransferError(f"{label} hardlink is forbidden")
        if metadata.st_size < 0 or metadata.st_size > max_bytes:
            raise FixtureTransferError(f"{label} size limit exceeded")
        descriptor = os.open(
            name,
            _read_flags(nonblocking=True),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(metadata):
            raise FixtureTransferError(f"{label} changed during read")
        body = _read_checked_descriptor(
            descriptor,
            before,
            error_type=FixtureTransferError,
            label=label,
            max_bytes=max_bytes,
        )
        return descriptor, before, body
    except FixtureTransferError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise FixtureTransferError(f"{label} file is unavailable or unsafe") from None


def _require_linked_regular(
    descriptor: int,
    before: os.stat_result,
    parent_fd: int,
    name: str,
    label: str,
) -> None:
    try:
        held = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise FixtureTransferError(f"{label} changed") from None
    if _metadata_identity(held) != _metadata_identity(before) or _metadata_identity(
        linked
    ) != _metadata_identity(before):
        raise FixtureTransferError(f"{label} changed")


def _require_configured_directory(
    path: Path,
    descriptor: int,
    label: str,
) -> None:
    try:
        configured = path.lstat()
        anchored = os.fstat(descriptor)
    except OSError:
        raise FixtureTransferError(f"{label} changed") from None
    if _directory_identity(configured) != _directory_identity(anchored):
        raise FixtureTransferError(f"{label} changed")


def _write_private_at(parent_fd: int, name: str, body: bytes, label: str) -> None:
    if not _safe_component(name):
        raise FixtureTransferError(f"{label} filename is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    created: os.stat_result | None = None
    complete = False
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        created = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or _mode_bits(metadata) != 0o600
        ):
            raise FixtureTransferError(f"{label} creation failed safely")
        os.fsync(parent_fd)
        complete = True
    except FileExistsError:
        raise FixtureTransferError(f"{label} already exists") from None
    except FixtureTransferError:
        raise
    except OSError:
        raise FixtureTransferError(f"{label} could not be written safely") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created is not None and not complete:
            _unlink_if_same(parent_fd, name, created)


def _rename_noreplace(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename a directory without replacing an existing destination."""

    if not _safe_component(source_name) or not _safe_component(destination_name):
        raise FixtureTransferError("release rename component is invalid")
    libc = ctypes.CDLL(None, use_errno=True)
    function: Any
    flags: int
    if sys.platform.startswith("linux"):
        try:
            function = libc.renameat2
        except AttributeError:
            raise FixtureTransferError(
                "atomic no-clobber rename is unavailable"
            ) from None
        flags = _RENAME_NOREPLACE
    elif sys.platform == "darwin":
        try:
            function = libc.renameatx_np
        except AttributeError:
            raise FixtureTransferError(
                "atomic no-clobber rename is unavailable"
            ) from None
        flags = _RENAME_EXCL
    else:
        raise FixtureTransferError("atomic no-clobber rename is unavailable")

    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        source_fd,
        os.fsencode(source_name),
        destination_fd,
        os.fsencode(destination_name),
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FixtureTransferError("release already exists")
    if error_number in {errno.ENOSYS, errno.ENOTSUP}:
        raise FixtureTransferError("atomic no-clobber rename is unavailable")
    raise FixtureTransferError("release promotion rename failed")


def plan_transfer(
    verified: VerifiedManifest,
    policy: DatasetPolicy,
    source: Path,
    destination: str,
    state_directory: Path,
) -> TransferPlan:
    """Build a private NUL-delimited rsync list from verified exact bytes."""

    _validate_verified_manifest(verified)
    if verified.manifest.dataset != policy.dataset_id:
        raise FixtureTransferError("transfer dataset does not match policy")
    if destination != policy.destination:
        raise FixtureTransferError("transfer destination does not match policy")
    verify_manifest(verified.manifest, policy, source)
    state_fd, _ = _open_private_directory(state_directory, "transfer state directory")
    os.close(state_fd)
    files_from = state_directory / f"{verified.sha256}.files-from0"
    file_list = b"".join(
        item.path.encode("utf-8") + b"\0" for item in verified.manifest.files
    )
    try:
        _write_private(files_from, file_list)
    except FixtureManifestError:
        try:
            existing = _read_checked_regular(
                files_from,
                error_type=FixtureTransferError,
                label="transfer file list",
                allowed_modes=frozenset({0o600}),
                max_bytes=_MAX_MANIFEST_BYTES,
            )
        except FixtureTransferError:
            raise FixtureTransferError("transfer file list is unsafe") from None
        if existing != file_list:
            raise FixtureTransferError("transfer file list differs") from None
    return TransferPlan(
        dataset=policy.dataset_id,
        source=source,
        destination=destination,
        manifest_sha256=verified.sha256,
        files_from=files_from,
        file_count=verified.manifest.file_count,
        total_bytes=verified.manifest.total_bytes,
    )


def render_transfer_summary(
    plan: TransferPlan,
    *,
    source_alias: str,
    destination_alias: str,
    retained_releases: int,
) -> str:
    """Render aggregate-only dry-run output."""

    for alias in (source_alias, destination_alias):
        if not _DATASET_ID.fullmatch(alias):
            raise FixtureTransferError("transfer alias is invalid")
    if not _nonnegative_integer(retained_releases):
        raise FixtureTransferError("retained release count is invalid")
    return (
        f"dataset: {plan.dataset}\n"
        f"source: {source_alias}\n"
        f"destination: {destination_alias}\n"
        f"files: {plan.file_count}\n"
        f"bytes: {plan.total_bytes}\n"
        f"release: {plan.manifest_sha256}\n"
        f"retained releases: {retained_releases}\n"
        "deletions: 0\n"
    )


def new_transfer_id() -> str:
    return secrets.token_hex(16)


def _ensure_private_state_leaf(
    root: Path,
    dataset: str,
    manifest_sha: str,
) -> Path:
    if not _DATASET_ID.fullmatch(dataset) or not _SHA256.fullmatch(manifest_sha):
        raise FixtureTransferError("transfer state identity is invalid")
    root_fd, _ = _open_private_directory(root, "transfer state directory")
    dataset_fd: int | None = None
    release_fd: int | None = None
    try:
        try:
            os.mkdir(dataset, 0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        dataset_fd, _ = _open_private_child_directory(
            root_fd,
            dataset,
            "dataset state directory",
        )
        try:
            os.mkdir(manifest_sha, 0o700, dir_fd=dataset_fd)
            os.fsync(dataset_fd)
        except FileExistsError:
            pass
        release_fd, _ = _open_private_child_directory(
            dataset_fd,
            manifest_sha,
            "release state directory",
        )
    except FixtureTransferError:
        raise
    except OSError:
        raise FixtureTransferError("transfer state could not be prepared") from None
    finally:
        if release_fd is not None:
            os.close(release_fd)
        if dataset_fd is not None:
            os.close(dataset_fd)
        os.close(root_fd)
    return root / dataset / manifest_sha


def _verify_pinned_rsync(executable: str, expected_version: str) -> None:
    output = _run_transfer_process(
        [executable, "--version"],
        capture=True,
        timeout=10,
    )
    match = re.search(
        rb"^rsync\s+version\s+([0-9]+\.[0-9]+\.[0-9]+)\s+"
        rb"protocol version\s+([0-9]+)$",
        output,
        re.MULTILINE,
    )
    if (
        match is None
        or match.group(1).decode("ascii") != expected_version
        or int(match.group(2)) != _PINNED_RSYNC_PROTOCOL
    ):
        raise FixtureTransferError("rsync does not match pinned protocol")


def _ssh_command(
    ssh: str,
    target: TransferTarget,
    *remote_arguments: str,
) -> list[str]:
    for argument in remote_arguments:
        if not re.fullmatch(r"[A-Za-z0-9_./=-]+", argument):
            raise FixtureTransferError("remote command argument is invalid")
    return [
        ssh,
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "--",
        target.ssh_destination,
        target.remote_fixturectl,
        *remote_arguments,
    ]


def _run_transfer_process(
    arguments: list[str],
    *,
    capture: bool = False,
    timeout: int = 120,
) -> bytes:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        raise FixtureTransferError("transfer step failed safely") from None
    output = completed.stdout or b""
    if len(output) > 64 * 1024:
        raise FixtureTransferError("transfer response exceeded safe limit")
    return output


def _write_local_receipt(path: Path, receipt: Receipt) -> None:
    body = _receipt_json(receipt)
    try:
        _write_private(path, body)
    except FixtureManifestError:
        try:
            existing = _read_checked_regular(
                path,
                error_type=FixtureTransferError,
                label="local receipt",
                allowed_modes=frozenset({0o600}),
                max_bytes=64 * 1024,
            )
        except FixtureTransferError:
            raise FixtureTransferError("local receipt is unsafe") from None
        if existing != body:
            raise FixtureTransferError("local receipt differs") from None


def _verify_preflight_response(
    raw: bytes,
    *,
    dataset: str,
    target: TransferTarget,
) -> None:
    try:
        body = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_transfer_config_duplicate_keys,
        )
    except (FixtureTransferError, UnicodeError, json.JSONDecodeError):
        raise FixtureTransferError("receiver preflight response is invalid") from None
    if (
        not isinstance(body, dict)
        or set(body) != {"dataset", "protocol_version", "status", "target_id"}
        or body["dataset"] != dataset
        or body["protocol_version"] != _RECEIVER_PROTOCOL_VERSION
        or body["status"] != "ready"
        or body["target_id"] != target.target_id
    ):
        raise FixtureTransferError("receiver identity does not match approved target")


def _run_quarantined_step(
    arguments: list[str],
    *,
    dataset: str,
    transfer_id: str,
    remediation: str,
    capture: bool = False,
    timeout: int = 120,
) -> bytes:
    try:
        return _run_transfer_process(
            arguments,
            capture=capture,
            timeout=timeout,
        )
    except FixtureTransferError:
        raise FixtureTransferApplyError(
            f"transfer_id={transfer_id} status=quarantined; remediation: {remediation}"
        ) from None


def _recovery_command(
    *,
    config_path: Path,
    dataset: str,
    target_alias: str,
    transfer_id: str,
    manifest_sha: str,
) -> str:
    return shlex.join(
        [
            "fixturectl",
            "transfer-status",
            "--dataset",
            dataset,
            "--target",
            target_alias,
            "--config",
            str(config_path),
            "--transfer-id",
            transfer_id,
            "--manifest-sha",
            manifest_sha,
        ]
    )


def _confirmation_pending_error(transfer_id: str, remediation: str) -> NoReturn:
    raise FixtureTransferApplyError(
        f"transfer_id={transfer_id} status=confirmation-pending; "
        f"remediation: {remediation}"
    )


def _transfer_status_from_bytes(
    raw: bytes,
    *,
    dataset: str,
    transfer_id: str,
) -> str:
    try:
        body = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_transfer_config_duplicate_keys,
        )
    except (FixtureTransferError, UnicodeError, json.JSONDecodeError):
        raise FixtureTransferError("receiver status response is invalid") from None
    if (
        not isinstance(body, dict)
        or set(body) != {"dataset", "status", "transfer_id"}
        or body["dataset"] != dataset
        or body["transfer_id"] != transfer_id
        or body["status"] not in {"not-found", "quarantined", "unsafe"}
    ):
        raise FixtureTransferError("receiver status response is invalid")
    return body["status"]


def run_transfer_status(
    *,
    config_path: Path,
    dataset: str,
    target_alias: str,
    transfer_id: str,
    manifest_sha: str,
) -> tuple[str, Path | None]:
    """Reconcile one interrupted apply without reading or transferring source data."""

    if not _DATASET_ID.fullmatch(dataset):
        raise FixtureTransferError("status dataset is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", transfer_id):
        raise FixtureTransferError("status transfer id is invalid")
    if not _SHA256.fullmatch(manifest_sha):
        raise FixtureTransferError("status release id is invalid")
    config = load_transfer_config(config_path)
    try:
        target = config.targets[target_alias]
    except KeyError:
        raise FixtureTransferError("status target is not approved") from None
    status_output = _run_transfer_process(
        _ssh_command(
            config.ssh_executable,
            target,
            "receive",
            "status",
            "--dataset",
            dataset,
            "--transfer-id",
            transfer_id,
        ),
        capture=True,
    )
    status = _transfer_status_from_bytes(
        status_output,
        dataset=dataset,
        transfer_id=transfer_id,
    )
    if status != "not-found":
        return status, None

    receipt_output = _run_transfer_process(
        _ssh_command(
            config.ssh_executable,
            target,
            "receipt",
            "print",
            "--dataset",
            dataset,
            "--manifest-sha",
            manifest_sha,
        ),
        capture=True,
    )
    receipt = _receipt_from_bytes(receipt_output)
    if (
        receipt.dataset != dataset
        or receipt.manifest_sha256 != manifest_sha
        or receipt.target_id != target.target_id
        or receipt.status != "verified"
    ):
        raise FixtureTransferError("remote receipt does not match recovery request")
    state_leaf = _ensure_private_state_leaf(
        config.state_directory,
        dataset,
        manifest_sha,
    )
    receipt_path = state_leaf / "receipt.json"
    _write_local_receipt(receipt_path, receipt)
    return "verified", receipt_path


def _require_state_outside_source(state: Path, source: Path) -> None:
    source_fd, _ = _open_dataset_root(source)
    state_fd, _ = _open_private_directory(state, "transfer state directory")
    try:
        if _directory_is_within(state_fd, source_fd):
            raise FixtureTransferError("transfer state must be outside source")
    except FixtureManifestError:
        raise FixtureTransferError("transfer state containment is invalid") from None
    finally:
        os.close(state_fd)
        os.close(source_fd)


def run_transfer(
    *,
    config_path: Path,
    policy: DatasetPolicy,
    dataset: str,
    target_alias: str,
    source: Path,
    manifest_path: Path,
    signature_path: Path,
    public_key: Path,
    apply: bool,
    retained_releases: int,
    verified_at: datetime,
) -> TransferResult:
    """Plan by default; explicitly apply one-way transfer to an approved target."""

    if dataset != policy.dataset_id:
        raise FixtureTransferError("transfer dataset does not match policy")
    if manifest_path.name != "manifest.json":
        raise FixtureTransferError("manifest must use canonical name")
    if signature_path.name != "manifest.minisig":
        raise FixtureTransferError("signature must use canonical name")
    config = load_transfer_config(config_path)
    try:
        target = config.targets[target_alias]
    except KeyError:
        raise FixtureTransferError("transfer target is not approved") from None
    _require_state_outside_source(config.state_directory, source)
    verified = verify_signature(manifest_path, signature_path, public_key)
    _validate_verified_manifest(verified)
    if verified.signature_sha256 is None:
        raise FixtureTransferError("verified signature digest is missing")
    if verified.manifest.dataset != dataset:
        raise FixtureTransferError("signed manifest dataset is invalid")

    state_leaf = _ensure_private_state_leaf(
        config.state_directory,
        dataset,
        verified.sha256,
    )
    plan = plan_transfer(
        verified,
        policy,
        source,
        policy.destination,
        state_leaf,
    )
    summary = render_transfer_summary(
        plan,
        source_alias=config.source_alias,
        destination_alias=target.alias,
        retained_releases=retained_releases,
    )
    if not apply:
        return TransferResult(
            plan=plan,
            summary=summary,
            applied=False,
            transfer_id=None,
            receipt_path=None,
        )

    ssh = config.ssh_executable
    rsync = config.rsync_executable
    _verify_pinned_rsync(rsync, config.rsync_version)
    transfer_id = new_transfer_id()
    remote_staging = f"{policy.destination}/incoming/{transfer_id}"
    remote_data = f"{remote_staging}/data/"
    remote_root = f"{target.ssh_destination}:{remote_staging}/"
    rsync_rsh = shlex.join(
        [
            ssh,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
    )

    preflight_output = _run_transfer_process(
        _ssh_command(
            ssh,
            target,
            "receive",
            "preflight",
            "--dataset",
            dataset,
            "--required-bytes",
            str(plan.total_bytes),
        ),
        capture=True,
    )
    _verify_preflight_response(
        preflight_output,
        dataset=dataset,
        target=target,
    )
    remediation = _recovery_command(
        config_path=config_path,
        dataset=dataset,
        target_alias=target_alias,
        transfer_id=transfer_id,
        manifest_sha=verified.sha256,
    )
    _run_quarantined_step(
        _ssh_command(
            ssh,
            target,
            "receive",
            "create-staging",
            "--dataset",
            dataset,
            "--transfer-id",
            transfer_id,
        ),
        dataset=dataset,
        transfer_id=transfer_id,
        remediation=remediation,
    )
    data_rsync_path = shlex.join(
        [
            target.remote_fixturectl,
            "ssh-dispatch-rsync",
            "--dataset",
            dataset,
            "--transfer-id",
            transfer_id,
            "--kind",
            "data",
        ]
    )
    metadata_rsync_path = shlex.join(
        [
            target.remote_fixturectl,
            "ssh-dispatch-rsync",
            "--dataset",
            dataset,
            "--transfer-id",
            transfer_id,
            "--kind",
            "metadata",
        ]
    )
    _run_quarantined_step(
        [
            rsync,
            "--archive",
            "--from0",
            f"--files-from={plan.files_from}",
            "--partial",
            f"--partial-dir=.fixturectl-partial-{transfer_id}",
            "--chmod=D700,F600",
            f"--rsh={rsync_rsh}",
            f"--rsync-path={data_rsync_path}",
            "--",
            f"{source}{os.sep}",
            f"{target.ssh_destination}:{remote_data}",
        ],
        dataset=dataset,
        transfer_id=transfer_id,
        remediation=remediation,
        timeout=14_400,
    )
    _run_quarantined_step(
        [
            rsync,
            "--archive",
            "--chmod=F600",
            f"--rsh={rsync_rsh}",
            f"--rsync-path={metadata_rsync_path}",
            "--",
            str(manifest_path),
            str(signature_path),
            remote_root,
        ],
        dataset=dataset,
        transfer_id=transfer_id,
        remediation=remediation,
        timeout=600,
    )
    _run_quarantined_step(
        _ssh_command(
            ssh,
            target,
            "receive",
            "verify",
            "--dataset",
            dataset,
            "--transfer-id",
            transfer_id,
        ),
        dataset=dataset,
        transfer_id=transfer_id,
        remediation=remediation,
    )
    try:
        _run_transfer_process(
            _ssh_command(
                ssh,
                target,
                "receive",
                "promote",
                "--dataset",
                dataset,
                "--transfer-id",
                transfer_id,
                "--manifest-sha",
                verified.sha256,
            )
        )
        receipt_output = _run_transfer_process(
            _ssh_command(
                ssh,
                target,
                "receipt",
                "print",
                "--dataset",
                dataset,
                "--manifest-sha",
                verified.sha256,
            ),
            capture=True,
        )
        receipt = _receipt_from_bytes(receipt_output)
        if (
            receipt.dataset != dataset
            or receipt.manifest_sha256 != verified.sha256
            or receipt.file_count != verified.manifest.file_count
            or receipt.total_bytes != verified.manifest.total_bytes
            or receipt.target_id != target.target_id
            or receipt.status != "verified"
        ):
            raise FixtureTransferError("remote receipt does not match local manifest")
        receipt_path = state_leaf / "receipt.json"
        _write_local_receipt(receipt_path, receipt)
    except FixtureTransferError:
        _confirmation_pending_error(transfer_id, remediation)
    return TransferResult(
        plan=plan,
        summary=summary,
        applied=True,
        transfer_id=transfer_id,
        receipt_path=receipt_path,
    )


def create_staging(dataset_root: Path, transfer_id: str) -> Path:
    """Create one exclusive private incoming directory."""

    if not re.fullmatch(r"[0-9a-f]{32}", transfer_id):
        raise FixtureTransferError("transfer id is invalid")
    root_fd, _ = _open_private_directory(dataset_root, "dataset root")
    incoming_fd: int | None = None
    try:
        incoming_fd, _ = _open_private_child_directory(
            root_fd,
            "incoming",
            "incoming directory",
        )
        try:
            os.mkdir(transfer_id, 0o700, dir_fd=incoming_fd)
        except FileExistsError:
            raise FixtureTransferError("staging exists") from None
        os.fsync(incoming_fd)
    except FixtureTransferError:
        raise
    except OSError:
        raise FixtureTransferError("staging could not be created safely") from None
    finally:
        if incoming_fd is not None:
            os.close(incoming_fd)
        os.close(root_fd)
    return dataset_root / "incoming" / transfer_id


def _resolved_staging(
    staging: Path,
    policy: DatasetPolicy,
    *,
    receipt_expected: bool,
) -> StagingLayout:
    dataset_root = Path(policy.destination)
    transfer_id = staging.name
    expected_staging = dataset_root / "incoming" / transfer_id
    if staging != expected_staging:
        raise FixtureTransferError("staging target root is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", transfer_id):
        raise FixtureTransferError("staging transfer id is invalid")

    root_fd, _ = _open_private_directory(dataset_root, "dataset root")
    incoming_fd: int | None = None
    staging_fd: int | None = None
    data_fd: int | None = None
    try:
        incoming_fd, _ = _open_private_child_directory(
            root_fd,
            "incoming",
            "incoming directory",
        )
        staging_fd, _ = _open_private_child_directory(
            incoming_fd,
            transfer_id,
            "staging directory",
        )
        with os.scandir(staging_fd) as iterator:
            entries = {entry.name for entry in iterator}
        expected = {"data", "manifest.json", "manifest.minisig"}
        if receipt_expected:
            expected.add("receipt.json")
        if entries != expected:
            raise FixtureTransferError("staging layout is invalid")
        data_fd, _ = _open_private_child_directory(
            staging_fd,
            "data",
            "staging data directory",
        )
        return StagingLayout(
            dataset_root=dataset_root,
            staging=expected_staging,
            transfer_id=transfer_id,
            root_fd=root_fd,
            incoming_fd=incoming_fd,
            staging_fd=staging_fd,
            data_fd=data_fd,
        )
    except BaseException as exc:
        for descriptor in (data_fd, staging_fd, incoming_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)
        if isinstance(exc, OSError):
            raise FixtureTransferError("staging layout is unavailable") from None
        raise


def _receipt_json(receipt: Receipt) -> bytes:
    return (
        json.dumps(
            asdict(receipt),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def receive(
    staging: Path,
    manifest_path: Path,
    signature_path: Path,
    public_key: Path,
    policy: DatasetPolicy,
    *,
    target_id: str,
    verified_at: datetime,
) -> Receipt:
    """Verify a complete staging tree and write an aggregate-only receipt."""

    layout = _resolved_staging(
        staging,
        policy,
        receipt_expected=False,
    )
    manifest_fd: int | None = None
    signature_fd: int | None = None
    try:
        if manifest_path != layout.staging / "manifest.json":
            raise FixtureTransferError("staging manifest location is invalid")
        if signature_path != layout.staging / "manifest.minisig":
            raise FixtureTransferError("staging signature location is invalid")
        if not _DATASET_ID.fullmatch(target_id):
            raise FixtureTransferError("target id is invalid")

        manifest_fd, manifest_before, anchored_manifest = _open_held_transfer_file_at(
            layout.staging_fd,
            "manifest.json",
            label="staging manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        signature_fd, signature_before, anchored_signature = (
            _open_held_transfer_file_at(
                layout.staging_fd,
                "manifest.minisig",
                label="staging signature",
                max_bytes=_MAX_SIGNATURE_BYTES,
            )
        )
        verified = verify_signature(
            manifest_path,
            signature_path,
            public_key,
        )
        _validate_verified_manifest(verified)
        if verified.signature_sha256 is None:
            raise FixtureTransferError("verified signature digest is missing")
        if anchored_manifest != verified.canonical_bytes:
            raise FixtureTransferError("staging manifest changed after verification")
        if hashlib.sha256(anchored_signature).hexdigest() != verified.signature_sha256:
            raise FixtureTransferError("staging signature changed after verification")
        if verified.manifest.dataset != policy.dataset_id:
            raise FixtureTransferError("staging dataset does not match policy")
        verify_manifest(
            verified.manifest,
            policy,
            layout.staging / "data",
            _anchored_root_fd=layout.data_fd,
        )
        _require_linked_regular(
            manifest_fd,
            manifest_before,
            layout.staging_fd,
            "manifest.json",
            "staging manifest",
        )
        _require_linked_regular(
            signature_fd,
            signature_before,
            layout.staging_fd,
            "manifest.minisig",
            "staging signature",
        )

        receipt = Receipt(
            schema_version=1,
            dataset=policy.dataset_id,
            manifest_sha256=verified.sha256,
            file_count=verified.manifest.file_count,
            total_bytes=verified.manifest.total_bytes,
            verified_at=_format_created_at(verified_at),
            target_id=target_id,
            status="verified",
        )
        _require_linked_directory(
            layout.staging_fd,
            layout.incoming_fd,
            layout.transfer_id,
            "staging directory",
        )
        _write_private_at(
            layout.staging_fd,
            "receipt.json",
            _receipt_json(receipt),
            "receipt",
        )
        return receipt
    finally:
        if signature_fd is not None:
            os.close(signature_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        layout.close()


def _reject_receipt_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureTransferError("duplicate key in receipt JSON")
        result[key] = value
    return result


def _receipt_from_bytes(raw: bytes) -> Receipt:
    try:
        body = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_receipt_duplicate_keys,
        )
    except FixtureTransferError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise FixtureTransferError("receipt JSON is invalid") from None
    expected = {
        "schema_version",
        "dataset",
        "manifest_sha256",
        "file_count",
        "total_bytes",
        "verified_at",
        "target_id",
        "status",
    }
    if not isinstance(body, dict) or set(body) != expected:
        raise FixtureTransferError("receipt fields are invalid")
    if (
        body["schema_version"] != 1
        or not isinstance(body["dataset"], str)
        or not _DATASET_ID.fullmatch(body["dataset"])
        or not isinstance(body["manifest_sha256"], str)
        or not _SHA256.fullmatch(body["manifest_sha256"])
        or not _nonnegative_integer(body["file_count"])
        or not _nonnegative_integer(body["total_bytes"])
        or not isinstance(body["verified_at"], str)
        or not _CREATED_AT.fullmatch(body["verified_at"])
        or not isinstance(body["target_id"], str)
        or not _DATASET_ID.fullmatch(body["target_id"])
        or body["status"] != "verified"
    ):
        raise FixtureTransferError("receipt values are invalid")
    try:
        datetime.strptime(body["verified_at"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise FixtureTransferError("receipt values are invalid") from None
    receipt = Receipt(**body)
    if raw != _receipt_json(receipt):
        raise FixtureTransferError("receipt JSON is not canonical")
    return receipt


def _load_receipt(path: Path) -> Receipt:
    raw = _read_checked_regular(
        path,
        error_type=FixtureTransferError,
        label="receipt",
        allowed_modes=frozenset({0o600}),
        max_bytes=64 * 1024,
    )
    return _receipt_from_bytes(raw)


def _switch_current(root_fd: int, manifest_sha: str) -> None:
    temporary = f".current-{secrets.token_hex(16)}"
    try:
        try:
            current = os.stat("current", dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISLNK(current.st_mode):
            raise FixtureTransferError("current release pointer is unsafe")
        os.symlink(f"releases/{manifest_sha}", temporary, dir_fd=root_fd)
        os.replace(temporary, "current", src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
    except FixtureTransferError:
        raise
    except OSError:
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except OSError:
            pass
        raise FixtureTransferError("current release pointer update failed") from None


def _verify_release(
    release: Path,
    manifest_sha: str,
    policy: DatasetPolicy,
    public_key: Path,
    *,
    root_fd: int,
    releases_fd: int,
) -> None:
    owned_releases_fd = os.dup(releases_fd)
    release_fd: int | None = None
    data_fd: int | None = None
    manifest_fd: int | None = None
    signature_fd: int | None = None
    receipt_fd: int | None = None
    try:
        release_fd, _ = _open_private_child_directory(
            owned_releases_fd,
            release.name,
            "existing release",
        )
        with os.scandir(release_fd) as iterator:
            entries = {entry.name for entry in iterator}
        if entries != {
            "data",
            "manifest.json",
            "manifest.minisig",
            "receipt.json",
        }:
            raise FixtureTransferError("existing release differs")
        data_fd, _ = _open_private_child_directory(
            release_fd,
            "data",
            "existing release data",
        )
        manifest_path = release / "manifest.json"
        signature_path = release / "manifest.minisig"
        manifest_fd, manifest_before, actual_manifest = _open_held_transfer_file_at(
            release_fd,
            "manifest.json",
            label="existing release manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        signature_fd, signature_before, actual_signature = _open_held_transfer_file_at(
            release_fd,
            "manifest.minisig",
            label="existing release signature",
            max_bytes=_MAX_SIGNATURE_BYTES,
        )
        receipt_fd, receipt_before, receipt_bytes = _open_held_transfer_file_at(
            release_fd,
            "receipt.json",
            label="receipt",
            max_bytes=64 * 1024,
        )
        verified = verify_signature(manifest_path, signature_path, public_key)
        _validate_verified_manifest(verified)
        if verified.signature_sha256 is None:
            raise FixtureTransferError("existing release differs")
        receipt = _receipt_from_bytes(receipt_bytes)
        if (
            actual_manifest != verified.canonical_bytes
            or hashlib.sha256(actual_signature).hexdigest() != verified.signature_sha256
            or verified.sha256 != manifest_sha
            or verified.manifest.dataset != policy.dataset_id
            or receipt.manifest_sha256 != manifest_sha
            or receipt.dataset != policy.dataset_id
            or receipt.file_count != verified.manifest.file_count
            or receipt.total_bytes != verified.manifest.total_bytes
        ):
            raise FixtureTransferError("existing release differs")
        verify_manifest(
            verified.manifest,
            policy,
            release / "data",
            _anchored_root_fd=data_fd,
        )
        _require_linked_regular(
            manifest_fd,
            manifest_before,
            release_fd,
            "manifest.json",
            "existing release manifest",
        )
        _require_linked_regular(
            signature_fd,
            signature_before,
            release_fd,
            "manifest.minisig",
            "existing release signature",
        )
        _require_linked_regular(
            receipt_fd,
            receipt_before,
            release_fd,
            "receipt.json",
            "receipt",
        )
        _require_linked_directory(
            data_fd,
            release_fd,
            "data",
            "existing release data",
        )
        _require_linked_directory(
            release_fd,
            owned_releases_fd,
            release.name,
            "existing release",
        )
        _require_linked_directory(
            owned_releases_fd,
            root_fd,
            "releases",
            "releases directory",
        )
        _require_configured_directory(Path(policy.destination), root_fd, "dataset root")
    finally:
        if receipt_fd is not None:
            os.close(receipt_fd)
        if signature_fd is not None:
            os.close(signature_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        if data_fd is not None:
            os.close(data_fd)
        if release_fd is not None:
            os.close(release_fd)
        os.close(owned_releases_fd)


def _promote_with_process_lock(
    dataset_root: Path,
    staging: Path,
    manifest_sha: str,
    policy: DatasetPolicy,
    public_key: Path,
) -> Path:
    """Promote verified staging under a process-safe lock and atomically swap current."""

    if not _SHA256.fullmatch(manifest_sha):
        raise FixtureTransferError("release id is invalid")
    if str(dataset_root) != policy.destination:
        raise FixtureTransferError("promotion target root does not match policy")
    root_fd, _ = _open_private_directory(dataset_root, "dataset root")
    lock_fd: int | None = None
    layout: StagingLayout | None = None
    releases_fd: int | None = None
    manifest_fd: int | None = None
    signature_fd: int | None = None
    receipt_fd: int | None = None
    try:
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        for attempt in range(16):
            try:
                lock_fd = os.open(
                    ".promote.lock",
                    lock_flags,
                    0o600,
                    dir_fd=root_fd,
                )
                break
            except OSError as exc:
                if exc.errno in {errno.EINTR, errno.ENOENT} and attempt < 15:
                    continue
                raise FixtureTransferError("promotion lock open failed") from None
        if lock_fd is None:
            raise FixtureTransferError("promotion lock open failed")
        try:
            os.fchmod(lock_fd, 0o600)
            lock_metadata = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.getuid()
                or lock_metadata.st_nlink != 1
                or _mode_bits(lock_metadata) != 0o600
            ):
                raise FixtureTransferError("promotion lock is unsafe")
        except FixtureTransferError:
            raise
        except OSError as exc:
            raise FixtureTransferError(
                f"promotion lock validation failed (errno {exc.errno})"
            ) from None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise FixtureTransferError(
                f"promotion lock acquisition failed (errno {exc.errno})"
            ) from None

        layout = _resolved_staging(staging, policy, receipt_expected=True)
        if _directory_identity(os.fstat(root_fd)) != _directory_identity(
            os.fstat(layout.root_fd)
        ):
            raise FixtureTransferError("dataset root changed")
        manifest_path = layout.staging / "manifest.json"
        signature_path = layout.staging / "manifest.minisig"
        manifest_fd, manifest_before, actual_manifest = _open_held_transfer_file_at(
            layout.staging_fd,
            "manifest.json",
            label="staging manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        signature_fd, signature_before, actual_signature = _open_held_transfer_file_at(
            layout.staging_fd,
            "manifest.minisig",
            label="staging signature",
            max_bytes=_MAX_SIGNATURE_BYTES,
        )
        receipt_fd, receipt_before, receipt_bytes = _open_held_transfer_file_at(
            layout.staging_fd,
            "receipt.json",
            label="receipt",
            max_bytes=64 * 1024,
        )
        receipt = _receipt_from_bytes(receipt_bytes)
        verified = verify_signature(
            manifest_path,
            signature_path,
            public_key,
        )
        _validate_verified_manifest(verified)
        if verified.signature_sha256 is None:
            raise FixtureTransferError("verified signature digest is missing")
        if (
            actual_manifest != verified.canonical_bytes
            or hashlib.sha256(actual_signature).hexdigest() != verified.signature_sha256
            or verified.sha256 != manifest_sha
            or receipt.manifest_sha256 != manifest_sha
            or receipt.dataset != policy.dataset_id
            or receipt.file_count != verified.manifest.file_count
            or receipt.total_bytes != verified.manifest.total_bytes
        ):
            raise FixtureTransferError("release id does not match verified staging")
        verify_manifest(
            verified.manifest,
            policy,
            layout.staging / "data",
            _anchored_root_fd=layout.data_fd,
        )
        _require_linked_regular(
            manifest_fd,
            manifest_before,
            layout.staging_fd,
            "manifest.json",
            "staging manifest",
        )
        _require_linked_regular(
            signature_fd,
            signature_before,
            layout.staging_fd,
            "manifest.minisig",
            "staging signature",
        )
        _require_linked_regular(
            receipt_fd,
            receipt_before,
            layout.staging_fd,
            "receipt.json",
            "receipt",
        )
        _require_linked_directory(
            layout.staging_fd,
            layout.incoming_fd,
            layout.transfer_id,
            "staging directory",
        )

        releases_fd, _ = _open_private_child_directory(
            root_fd,
            "releases",
            "releases directory",
        )
        release = dataset_root / "releases" / manifest_sha
        try:
            os.stat(manifest_sha, dir_fd=releases_fd, follow_symlinks=False)
        except FileNotFoundError:
            try:
                _rename_noreplace(
                    layout.incoming_fd,
                    layout.transfer_id,
                    releases_fd,
                    manifest_sha,
                )
                promoted = os.stat(
                    manifest_sha,
                    dir_fd=releases_fd,
                    follow_symlinks=False,
                )
                if _directory_identity(promoted) != _directory_identity(
                    os.fstat(layout.staging_fd)
                ):
                    raise FixtureTransferError("staging directory changed")
                os.fsync(releases_fd)
                os.fsync(layout.incoming_fd)
            except FixtureTransferError:
                raise
            except OSError:
                raise FixtureTransferError("release promotion rename failed") from None
        except OSError:
            raise FixtureTransferError(
                "release path is unavailable or unsafe"
            ) from None
        else:
            try:
                _verify_release(
                    release,
                    manifest_sha,
                    policy,
                    public_key,
                    root_fd=root_fd,
                    releases_fd=releases_fd,
                )
            except (
                FixtureManifestError,
                FixturePolicyError,
                FixtureSignatureError,
                FixtureTransferError,
                OSError,
            ):
                raise FixtureTransferError("existing release differs") from None
        try:
            _verify_release(
                release,
                manifest_sha,
                policy,
                public_key,
                root_fd=root_fd,
                releases_fd=releases_fd,
            )
        except (
            FixtureManifestError,
            FixturePolicyError,
            FixtureSignatureError,
            FixtureTransferError,
            OSError,
        ):
            raise FixtureTransferError("promoted release failed verification") from None
        _require_configured_directory(dataset_root, root_fd, "dataset root")
        _require_linked_directory(
            releases_fd,
            root_fd,
            "releases",
            "releases directory",
        )
        _switch_current(root_fd, manifest_sha)
        return release
    except FixtureTransferError:
        raise
    except OSError:
        raise FixtureTransferError("release promotion failed safely") from None
    finally:
        if receipt_fd is not None:
            os.close(receipt_fd)
        if signature_fd is not None:
            os.close(signature_fd)
        if manifest_fd is not None:
            os.close(manifest_fd)
        if releases_fd is not None:
            os.close(releases_fd)
        if layout is not None:
            layout.close()
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(root_fd)


def promote(
    dataset_root: Path,
    staging: Path,
    manifest_sha: str,
    policy: DatasetPolicy,
    public_key: Path,
) -> Path:
    """Serialize threads, then use the filesystem lock to serialize processes."""

    lock_key = os.path.abspath(dataset_root)
    with _PROMOTION_THREAD_LOCKS_GUARD:
        thread_lock = _PROMOTION_THREAD_LOCKS.setdefault(
            lock_key,
            threading.Lock(),
        )
    with thread_lock:
        return _promote_with_process_lock(
            dataset_root,
            staging,
            manifest_sha,
            policy,
            public_key,
        )


def _create_staging_data(dataset_root: Path, transfer_id: str) -> Path:
    staging = create_staging(dataset_root, transfer_id)
    staging_fd, _ = _open_private_directory(staging, "staging directory")
    try:
        os.mkdir("data", 0o700, dir_fd=staging_fd)
        os.fsync(staging_fd)
    except OSError:
        raise FixtureTransferError("staging data could not be created safely") from None
    finally:
        os.close(staging_fd)
    return staging


def _receiver_preflight(
    policy: DatasetPolicy,
    *,
    required_bytes: int,
) -> None:
    if not _nonnegative_integer(required_bytes):
        raise FixtureTransferError("required byte count is invalid")
    root = Path(policy.destination)
    root_fd, _ = _open_private_directory(root, "dataset root")
    incoming_fd: int | None = None
    releases_fd: int | None = None
    try:
        incoming_fd, _ = _open_private_child_directory(
            root_fd,
            "incoming",
            "incoming directory",
        )
        releases_fd, _ = _open_private_child_directory(
            root_fd,
            "releases",
            "releases directory",
        )
        if shutil.disk_usage(root).free < required_bytes:
            raise FixtureTransferError("target capacity is insufficient")
    finally:
        if releases_fd is not None:
            os.close(releases_fd)
        if incoming_fd is not None:
            os.close(incoming_fd)
        os.close(root_fd)


def _receiver_status(policy: DatasetPolicy, transfer_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", transfer_id):
        raise FixtureTransferError("transfer id is invalid")
    root_fd, _ = _open_private_directory(Path(policy.destination), "dataset root")
    incoming_fd: int | None = None
    try:
        incoming_fd, _ = _open_private_child_directory(
            root_fd,
            "incoming",
            "incoming directory",
        )
        try:
            metadata = os.stat(
                transfer_id,
                dir_fd=incoming_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return "not-found"
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or _mode_bits(metadata) != 0o700
        ):
            return "unsafe"
        return "quarantined"
    finally:
        if incoming_fd is not None:
            os.close(incoming_fd)
        os.close(root_fd)


def _validate_rsync_upload_command(
    receiver: ReceiverConfig,
    policy: DatasetPolicy,
    *,
    dataset: str,
    transfer_id: str,
    kind: str,
    server_arguments: list[str],
) -> list[str]:
    """Validate the only rsync server commands the forced SSH identity may run."""

    if dataset != policy.dataset_id or not _DATASET_ID.fullmatch(dataset):
        raise FixtureTransferError("rsync dataset is invalid")
    if not re.fullmatch(r"[0-9a-f]{32}", transfer_id):
        raise FixtureTransferError("rsync transfer id is invalid")
    if kind not in {"data", "metadata"}:
        raise FixtureTransferError("rsync upload kind is invalid")
    if not server_arguments or server_arguments[0] != "--server":
        raise FixtureTransferError("rsync server mode is invalid")
    partial = f"--partial-dir=.fixturectl-partial-{transfer_id}"
    if kind == "data":
        expected_arguments = [
            "--server",
            _RSYNC_DATA_SERVER_OPTIONS,
            "--partial-dir",
            partial.removeprefix("--partial-dir="),
            ".",
            f"{policy.destination}/incoming/{transfer_id}/data/",
        ]
    else:
        expected_arguments = [
            "--server",
            _RSYNC_METADATA_SERVER_OPTIONS,
            ".",
            f"{policy.destination}/incoming/{transfer_id}/",
        ]
    if server_arguments != expected_arguments:
        raise FixtureTransferError("rsync argv does not match pinned upload protocol")
    return [receiver.rsync_executable, *server_arguments]


def _dispatch_ssh_original_command(
    receiver: ReceiverConfig,
    original_command: str,
) -> int:
    try:
        tokens = shlex.split(original_command, posix=True)
    except ValueError:
        raise FixtureTransferError("SSH command is invalid") from None
    if len(tokens) < 2 or tokens[0] != receiver.fixturectl_executable:
        raise FixtureTransferError("SSH command executable is not authorized")
    if tokens[1] in {"receive", "receipt"}:
        return main(tokens[1:])
    if tokens[1] != "ssh-dispatch-rsync" or len(tokens) < 11:
        raise FixtureTransferError("SSH command is not authorized")
    prefix = tokens[2:8]
    if (
        len(prefix) != 6
        or prefix[0] != "--dataset"
        or prefix[2] != "--transfer-id"
        or prefix[4] != "--kind"
    ):
        raise FixtureTransferError("rsync dispatcher prefix is invalid")
    dataset, transfer_id, kind = prefix[1], prefix[3], prefix[5]
    policy_set = load_policy(receiver.policy)
    try:
        policy = policy_set.datasets[dataset]
    except KeyError:
        raise FixtureTransferError("rsync dataset is not authorized") from None
    executable_arguments = _validate_rsync_upload_command(
        receiver,
        policy,
        dataset=dataset,
        transfer_id=transfer_id,
        kind=kind,
        server_arguments=tokens[8:],
    )
    _verify_pinned_rsync(receiver.rsync_executable, receiver.rsync_version)
    os.execv(receiver.rsync_executable, executable_arguments)
    raise FixtureTransferError("rsync dispatcher returned unexpectedly")


def _read_release_receipt(policy: DatasetPolicy, manifest_sha: str) -> Receipt:
    if not _SHA256.fullmatch(manifest_sha):
        raise FixtureTransferError("release id is invalid")
    root_fd, _ = _open_private_directory(Path(policy.destination), "dataset root")
    releases_fd: int | None = None
    release_fd: int | None = None
    receipt_fd: int | None = None
    try:
        releases_fd, _ = _open_private_child_directory(
            root_fd,
            "releases",
            "releases directory",
        )
        release_fd, _ = _open_private_child_directory(
            releases_fd,
            manifest_sha,
            "release directory",
        )
        receipt_fd, _, raw = _open_held_transfer_file_at(
            release_fd,
            "receipt.json",
            label="receipt",
            max_bytes=64 * 1024,
        )
        return _receipt_from_bytes(raw)
    finally:
        if receipt_fd is not None:
            os.close(receipt_fd)
        if release_fd is not None:
            os.close(release_fd)
        if releases_fd is not None:
            os.close(releases_fd)
        os.close(root_fd)


def _write_redacted_mapping(value: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


def _die(message: str) -> NoReturn:
    sys.stderr.write(f"fixturectl: {message}\n")
    raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fixturectl")
    parser.add_argument("--policy", type=Path)
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

    transfer = subcommands.add_parser("transfer")
    transfer.add_argument("--dataset", required=True)
    transfer.add_argument("--target", required=True)
    transfer.add_argument("--config", type=Path, required=True)
    transfer.add_argument("--source", type=Path, required=True)
    transfer.add_argument("--manifest", type=Path, required=True)
    transfer.add_argument("--signature", type=Path, required=True)
    transfer.add_argument("--public-key", type=Path, required=True)
    transfer.add_argument("--retained-releases", type=int, default=0)
    transfer_mode = transfer.add_mutually_exclusive_group()
    transfer_mode.add_argument("--apply", action="store_true")
    transfer_mode.add_argument("--dry-run", action="store_false", dest="apply")
    transfer.set_defaults(apply=False)

    transfer_status = subcommands.add_parser("transfer-status")
    transfer_status.add_argument("--dataset", required=True)
    transfer_status.add_argument("--target", required=True)
    transfer_status.add_argument("--config", type=Path, required=True)
    transfer_status.add_argument("--transfer-id", required=True)
    transfer_status.add_argument("--manifest-sha", required=True)

    receiver = subcommands.add_parser("receive")
    receiver_commands = receiver.add_subparsers(
        dest="receive_command",
        required=True,
    )
    preflight = receiver_commands.add_parser("preflight")
    preflight.add_argument("--dataset", required=True)
    preflight.add_argument("--required-bytes", type=int, required=True)
    create_staging_parser = receiver_commands.add_parser("create-staging")
    create_staging_parser.add_argument("--dataset", required=True)
    create_staging_parser.add_argument("--transfer-id", required=True)
    receive_verify = receiver_commands.add_parser("verify")
    receive_verify.add_argument("--dataset", required=True)
    receive_verify.add_argument("--transfer-id", required=True)
    receive_promote = receiver_commands.add_parser("promote")
    receive_promote.add_argument("--dataset", required=True)
    receive_promote.add_argument("--transfer-id", required=True)
    receive_promote.add_argument("--manifest-sha", required=True)
    receive_status = receiver_commands.add_parser("status")
    receive_status.add_argument("--dataset", required=True)
    receive_status.add_argument("--transfer-id", required=True)

    receipt = subcommands.add_parser("receipt")
    receipt_commands = receipt.add_subparsers(dest="receipt_command", required=True)
    receipt_print = receipt_commands.add_parser("print")
    receipt_print.add_argument("--dataset", required=True)
    receipt_print.add_argument("--manifest-sha", required=True)
    subcommands.add_parser("ssh-dispatch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "ssh-dispatch":
            receiver = load_receiver_config()
            original = os.environ.get("SSH_ORIGINAL_COMMAND")
            if original is None:
                raise FixtureTransferError("SSH original command is unavailable")
            return _dispatch_ssh_original_command(receiver, original)
        if args.command == "transfer-status":
            status, _ = run_transfer_status(
                config_path=args.config,
                dataset=args.dataset,
                target_alias=args.target,
                transfer_id=args.transfer_id,
                manifest_sha=args.manifest_sha,
            )
            _write_redacted_mapping(
                {
                    "dataset": args.dataset,
                    "status": status,
                    "target": args.target,
                    "transfer_id": args.transfer_id,
                }
            )
            return 0
        receiver: ReceiverConfig | None = None
        if args.command in {"receive", "receipt"}:
            if args.policy is not None:
                raise FixtureTransferError("receiver policy is server-owned")
            receiver = load_receiver_config()
            policy_path = receiver.policy
        else:
            if args.policy is None:
                raise FixturePolicyError("policy is required")
            policy_path = args.policy
        policy = load_policy(policy_path)
        try:
            dataset_policy = policy.datasets[args.dataset]
        except KeyError as exc:
            raise FixturePolicyError("dataset id is not authorized") from exc
        if args.command == "manifest" and args.manifest_command == "create":
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
        elif args.command == "manifest":
            verify_manifest(load_manifest(args.manifest), dataset_policy, args.root)
        elif args.command == "transfer":
            result = run_transfer(
                config_path=args.config,
                policy=dataset_policy,
                dataset=args.dataset,
                target_alias=args.target,
                source=args.source,
                manifest_path=args.manifest,
                signature_path=args.signature,
                public_key=args.public_key,
                apply=args.apply,
                retained_releases=args.retained_releases,
                verified_at=datetime.now(UTC),
            )
            sys.stdout.write(result.summary)
            if result.applied:
                _write_redacted_mapping(
                    {
                        "dataset": args.dataset,
                        "status": "applied",
                        "transfer_id": result.transfer_id,
                    }
                )
        elif args.command == "receive" and args.receive_command == "preflight":
            assert receiver is not None
            _verify_pinned_rsync(
                receiver.rsync_executable,
                receiver.rsync_version,
            )
            _receiver_preflight(
                dataset_policy,
                required_bytes=args.required_bytes,
            )
            _write_redacted_mapping(
                {
                    "dataset": args.dataset,
                    "protocol_version": receiver.protocol_version,
                    "status": "ready",
                    "target_id": receiver.target_id,
                }
            )
        elif args.command == "receive" and args.receive_command == "create-staging":
            _create_staging_data(Path(dataset_policy.destination), args.transfer_id)
            _write_redacted_mapping(
                {
                    "dataset": args.dataset,
                    "status": "staging-created",
                    "transfer_id": args.transfer_id,
                }
            )
        elif args.command == "receive" and args.receive_command == "verify":
            assert receiver is not None
            staging = Path(dataset_policy.destination) / "incoming" / args.transfer_id
            receipt = receive(
                staging,
                staging / "manifest.json",
                staging / "manifest.minisig",
                receiver.public_key,
                dataset_policy,
                target_id=receiver.target_id,
                verified_at=datetime.now(UTC),
            )
            sys.stdout.write(_receipt_json(receipt).decode("utf-8"))
        elif args.command == "receive" and args.receive_command == "promote":
            assert receiver is not None
            staging = Path(dataset_policy.destination) / "incoming" / args.transfer_id
            release = promote(
                Path(dataset_policy.destination),
                staging,
                args.manifest_sha,
                dataset_policy,
                receiver.public_key,
            )
            _write_redacted_mapping(
                {
                    "dataset": args.dataset,
                    "release": release.name,
                    "status": "promoted",
                }
            )
        elif args.command == "receive":
            status = _receiver_status(dataset_policy, args.transfer_id)
            _write_redacted_mapping(
                {
                    "dataset": args.dataset,
                    "status": status,
                    "transfer_id": args.transfer_id,
                }
            )
        else:
            receipt = _read_release_receipt(dataset_policy, args.manifest_sha)
            sys.stdout.write(_receipt_json(receipt).decode("utf-8"))
    except FixtureTransferApplyError as exc:
        _die(str(exc))
    except (
        FixturePolicyError,
        FixtureManifestError,
        FixtureSignatureError,
        FixtureTransferError,
        OSError,
    ):
        _die("validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
