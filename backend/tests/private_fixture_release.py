"""Fail-closed discovery of immutable private-fixture test releases."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping
import unicodedata


APPROVED_ROOT_ENV = "REAL_MEDICAL_FIXTURES_APPROVED_ROOT"
CURRENT_LINK_ENV = "REAL_MEDICAL_FIXTURES_DIR"
TARGET_ID_ENV = "REAL_MEDICAL_FIXTURES_TARGET_ID"

_DATASET_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DATASET_TOP_LEVEL = {
    "medtimeline": frozenset({"raw"}),
    "medtimeline-fidelity-v2": frozenset({"Run-v5-71a2f50"}),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_MANIFEST_FIELDS = {
    "schema_version",
    "dataset",
    "created_at",
    "hash_algorithm",
    "file_count",
    "total_bytes",
    "files",
}
_MANIFEST_FILE_FIELDS = {"path", "size", "mode", "sha256"}
_RECEIPT_FIELDS = {
    "schema_version",
    "dataset",
    "manifest_sha256",
    "file_count",
    "total_bytes",
    "verified_at",
    "target_id",
    "status",
}
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024


class PrivateFixtureReleaseError(RuntimeError):
    """A requested private fixture release failed closed."""


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
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
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_uid,
        stat.S_IMODE(metadata.st_mode),
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _file_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _open_approved_root(path: Path) -> tuple[int, Path, os.stat_result]:
    if not path.is_absolute():
        raise PrivateFixtureReleaseError(
            "private fixture approved root must be absolute"
        )
    try:
        leaf = path.lstat()
        resolved = path.resolve(strict=True)
        if path != resolved:
            raise PrivateFixtureReleaseError(
                "private fixture approved root must be canonical"
            )
        if stat.S_ISLNK(leaf.st_mode) or not stat.S_ISDIR(leaf.st_mode):
            raise PrivateFixtureReleaseError(
                "private fixture approved root is not a real directory"
            )
        if leaf.st_uid not in {0, os.geteuid()} or stat.S_IMODE(leaf.st_mode) & 0o022:
            raise PrivateFixtureReleaseError("private fixture approved root is unsafe")
        descriptor = os.open(path, _directory_flags())
        opened = os.fstat(descriptor)
        if _directory_identity(opened) != _directory_identity(leaf):
            os.close(descriptor)
            raise PrivateFixtureReleaseError(
                "private fixture approved root changed during validation"
            )
        return descriptor, resolved, opened
    except PrivateFixtureReleaseError:
        raise
    except OSError:
        raise PrivateFixtureReleaseError(
            "private fixture approved root is unavailable"
        ) from None


def _open_private_directory_at(
    parent_fd: int,
    name: str,
    label: str,
) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or _directory_identity(opened) != _directory_identity(linked)
        ):
            raise PrivateFixtureReleaseError(f"{label} is unsafe")
        return descriptor, opened
    except PrivateFixtureReleaseError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise PrivateFixtureReleaseError(f"{label} is unavailable or unsafe") from None


def _require_linked_directory(
    descriptor: int,
    expected: os.stat_result,
    parent_fd: int,
    name: str,
    label: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise PrivateFixtureReleaseError(f"{label} changed during validation") from None
    if _directory_identity(opened) != _directory_identity(
        expected
    ) or _directory_identity(linked) != _directory_identity(expected):
        raise PrivateFixtureReleaseError(f"{label} changed during validation")


def _read_private_file_at(
    parent_fd: int,
    name: str,
    label: str,
    max_bytes: int,
) -> bytes:
    descriptor: int | None = None
    try:
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size > max_bytes
            or _identity(opened) != _identity(linked)
        ):
            raise PrivateFixtureReleaseError(f"{label} is unsafe")
        body = bytearray()
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, max_bytes + 1 - len(body)),
        ):
            body.extend(chunk)
            if len(body) > max_bytes:
                raise PrivateFixtureReleaseError(f"{label} is too large")
        after = os.fstat(descriptor)
        relinked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(after) != _identity(opened) or _identity(relinked) != _identity(
            opened
        ):
            raise PrivateFixtureReleaseError(f"{label} changed during validation")
        return bytes(body)
    except PrivateFixtureReleaseError:
        raise
    except OSError:
        raise PrivateFixtureReleaseError(f"{label} is unavailable or unsafe") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PrivateFixtureReleaseError("private fixture JSON has duplicate keys")
        result[key] = value
    return result


def _canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except PrivateFixtureReleaseError:
        raise
    except (UnicodeError, json.JSONDecodeError):
        raise PrivateFixtureReleaseError(f"{label} JSON is invalid") from None
    if not isinstance(value, dict):
        raise PrivateFixtureReleaseError(f"{label} JSON is invalid")
    canonical = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PrivateFixtureReleaseError(f"{label} JSON is not canonical")
    return value


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _validate_receipt(
    raw: bytes,
    *,
    dataset: str,
    manifest_sha: str,
    target_id: str,
) -> dict[str, Any]:
    receipt = _canonical_json(raw, "private fixture receipt")
    if set(receipt) != _RECEIPT_FIELDS:
        raise PrivateFixtureReleaseError("private fixture receipt fields are invalid")
    if receipt["dataset"] != dataset:
        raise PrivateFixtureReleaseError("private fixture receipt dataset is invalid")
    if receipt["target_id"] != target_id:
        raise PrivateFixtureReleaseError("private fixture target identity is invalid")
    if receipt["manifest_sha256"] != manifest_sha:
        raise PrivateFixtureReleaseError("private fixture manifest identity is invalid")
    if (
        receipt["schema_version"] != 1
        or not _nonnegative_integer(receipt["file_count"])
        or not _nonnegative_integer(receipt["total_bytes"])
        or not _valid_timestamp(receipt["verified_at"])
        or receipt["status"] != "verified"
    ):
        raise PrivateFixtureReleaseError("private fixture receipt values are invalid")
    return receipt


def _safe_manifest_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        return False
    if unicodedata.normalize("NFC", value) != value:
        return False
    path = PurePosixPath(value)
    return (
        str(path) == value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _validate_manifest(raw: bytes, *, dataset: str) -> dict[str, Any]:
    manifest = _canonical_json(raw, "private fixture manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise PrivateFixtureReleaseError("private fixture manifest fields are invalid")
    if manifest["dataset"] != dataset:
        raise PrivateFixtureReleaseError("private fixture manifest dataset is invalid")
    files = manifest["files"]
    if (
        manifest["schema_version"] != 1
        or manifest["hash_algorithm"] != "sha256"
        or not _valid_timestamp(manifest["created_at"])
        or not _nonnegative_integer(manifest["file_count"])
        or not _nonnegative_integer(manifest["total_bytes"])
        or not isinstance(files, list)
        or manifest["file_count"] != len(files)
    ):
        raise PrivateFixtureReleaseError("private fixture manifest values are invalid")

    paths: list[str] = []
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != _MANIFEST_FILE_FIELDS:
            raise PrivateFixtureReleaseError("private fixture manifest file is invalid")
        if _safe_manifest_path(entry["path"]) and (
            PurePosixPath(entry["path"]).parts[0] not in _DATASET_TOP_LEVEL[dataset]
        ):
            raise PrivateFixtureReleaseError(
                "private fixture manifest top-level path is invalid"
            )
        if (
            not _safe_manifest_path(entry["path"])
            or not _nonnegative_integer(entry["size"])
            or entry["mode"] != "0600"
            or not isinstance(entry["sha256"], str)
            or not _SHA256.fullmatch(entry["sha256"])
        ):
            raise PrivateFixtureReleaseError("private fixture manifest file is invalid")
        paths.append(entry["path"])
        total_bytes += entry["size"]
    if (
        paths != sorted(paths)
        or len({path.casefold() for path in paths}) != len(paths)
        or total_bytes != manifest["total_bytes"]
    ):
        raise PrivateFixtureReleaseError(
            "private fixture manifest aggregates are invalid"
        )
    return manifest


def discover_private_release(
    *,
    expected_dataset: str,
    current_link: Path,
    approved_root: Path,
    expected_target_id: str,
) -> Path:
    """Resolve one verified receipt to its immutable release data directory."""

    if (
        not _DATASET_ID.fullmatch(expected_dataset)
        or expected_dataset not in _DATASET_TOP_LEVEL
    ):
        raise PrivateFixtureReleaseError("private fixture dataset identity is invalid")
    if not _DATASET_ID.fullmatch(expected_target_id):
        raise PrivateFixtureReleaseError("private fixture target identity is invalid")

    approved_fd: int | None = None
    dataset_fd: int | None = None
    releases_fd: int | None = None
    release_fd: int | None = None
    data_fd: int | None = None
    try:
        approved_fd, approved_path, approved_metadata = _open_approved_root(
            approved_root
        )
        expected_current = approved_path / expected_dataset / "current"
        if not current_link.is_absolute() or current_link != expected_current:
            raise PrivateFixtureReleaseError(
                "private fixture current link is outside the approved runtime root"
            )

        dataset_fd, dataset_metadata = _open_private_directory_at(
            approved_fd,
            expected_dataset,
            "private fixture dataset root",
        )
        try:
            current_before = os.stat(
                "current", dir_fd=dataset_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISLNK(current_before.st_mode)
                or current_before.st_uid != os.geteuid()
                or current_before.st_nlink != 1
            ):
                raise PrivateFixtureReleaseError("current release pointer is unsafe")
            target = os.readlink("current", dir_fd=dataset_fd)
            current_after = os.stat("current", dir_fd=dataset_fd, follow_symlinks=False)
        except PrivateFixtureReleaseError:
            raise
        except OSError:
            raise PrivateFixtureReleaseError(
                "current release pointer is unavailable"
            ) from None
        if _identity(current_before) != _identity(current_after):
            raise PrivateFixtureReleaseError(
                "current release pointer changed during validation"
            )
        target_path = PurePosixPath(target)
        if (
            target_path.is_absolute()
            or len(target_path.parts) != 2
            or target_path.parts[0] != "releases"
            or not _SHA256.fullmatch(target_path.parts[1])
        ):
            raise PrivateFixtureReleaseError("current release pointer is unsafe")
        manifest_sha = target_path.parts[1]

        releases_fd, releases_metadata = _open_private_directory_at(
            dataset_fd,
            "releases",
            "private fixture releases directory",
        )
        release_fd, release_metadata = _open_private_directory_at(
            releases_fd,
            manifest_sha,
            "private fixture release",
        )
        data_fd, data_metadata = _open_private_directory_at(
            release_fd,
            "data",
            "private fixture data directory",
        )
        manifest_raw = _read_private_file_at(
            release_fd,
            "manifest.json",
            "private fixture manifest",
            _MAX_MANIFEST_BYTES,
        )
        receipt_raw = _read_private_file_at(
            release_fd,
            "receipt.json",
            "private fixture receipt",
            _MAX_RECEIPT_BYTES,
        )
        signature_raw = _read_private_file_at(
            release_fd,
            "manifest.minisig",
            "private fixture signature",
            _MAX_SIGNATURE_BYTES,
        )
        if not signature_raw:
            raise PrivateFixtureReleaseError("private fixture signature is empty")

        actual_manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        if actual_manifest_sha != manifest_sha:
            raise PrivateFixtureReleaseError(
                "private fixture manifest identity is invalid"
            )
        manifest = _validate_manifest(manifest_raw, dataset=expected_dataset)
        receipt = _validate_receipt(
            receipt_raw,
            dataset=expected_dataset,
            manifest_sha=manifest_sha,
            target_id=expected_target_id,
        )
        if (
            receipt["file_count"] != manifest["file_count"]
            or receipt["total_bytes"] != manifest["total_bytes"]
        ):
            raise PrivateFixtureReleaseError(
                "private fixture receipt aggregates are invalid"
            )

        _require_linked_directory(
            data_fd,
            data_metadata,
            release_fd,
            "data",
            "private fixture data directory",
        )
        _require_linked_directory(
            release_fd,
            release_metadata,
            releases_fd,
            manifest_sha,
            "private fixture release",
        )
        _require_linked_directory(
            releases_fd,
            releases_metadata,
            dataset_fd,
            "releases",
            "private fixture releases directory",
        )
        _require_linked_directory(
            dataset_fd,
            dataset_metadata,
            approved_fd,
            expected_dataset,
            "private fixture dataset root",
        )
        if _directory_identity(os.fstat(approved_fd)) != _directory_identity(
            approved_metadata
        ):
            raise PrivateFixtureReleaseError(
                "private fixture approved root changed during validation"
            )
        current_final = os.stat("current", dir_fd=dataset_fd, follow_symlinks=False)
        if (
            _identity(current_final) != _identity(current_after)
            or os.readlink("current", dir_fd=dataset_fd) != target
        ):
            raise PrivateFixtureReleaseError(
                "current release pointer changed during validation"
            )
        _require_linked_directory(
            data_fd,
            data_metadata,
            release_fd,
            "data",
            "private fixture data directory",
        )
        _require_linked_directory(
            release_fd,
            release_metadata,
            releases_fd,
            manifest_sha,
            "private fixture release",
        )
        _require_linked_directory(
            releases_fd,
            releases_metadata,
            dataset_fd,
            "releases",
            "private fixture releases directory",
        )
        _require_linked_directory(
            dataset_fd,
            dataset_metadata,
            approved_fd,
            expected_dataset,
            "private fixture dataset root",
        )
        if _directory_identity(os.fstat(approved_fd)) != _directory_identity(
            approved_metadata
        ):
            raise PrivateFixtureReleaseError(
                "private fixture approved root changed during validation"
            )
        return approved_path / expected_dataset / "releases" / manifest_sha / "data"
    except PrivateFixtureReleaseError:
        raise
    except OSError:
        raise PrivateFixtureReleaseError(
            "private fixture release changed during validation"
        ) from None
    finally:
        for descriptor in (data_fd, release_fd, releases_fd, dataset_fd, approved_fd):
            if descriptor is not None:
                os.close(descriptor)


def private_fixture_root(
    expected_dataset: str = "medtimeline",
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Discover the requested release, or return ``None`` when not requested."""

    values = os.environ if environment is None else environment
    configured = values.get(CURRENT_LINK_ENV)
    if not configured:
        return None
    approved = values.get(APPROVED_ROOT_ENV)
    if not approved:
        raise PrivateFixtureReleaseError(
            "private fixtures require an approved runtime root"
        )
    target_id = values.get(TARGET_ID_ENV)
    if not target_id:
        raise PrivateFixtureReleaseError(
            "private fixtures require an approved target identity"
        )
    return discover_private_release(
        expected_dataset=expected_dataset,
        current_link=Path(configured).expanduser(),
        approved_root=Path(approved).expanduser(),
        expected_target_id=target_id,
    )
