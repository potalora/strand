"""Verified, atomic storage for immutable local-AI model packs."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import (
    ALLOWED_LICENSES,
    ALLOWED_SUFFIXES,
    COMMIT_RE,
    EXPECTED_ROLES,
    FORBIDDEN_SUFFIXES,
    MAX_MANIFEST_FILES,
    MAX_MANIFEST_FILE_BYTES,
    MAX_MANIFEST_PACK_BYTES,
    MAX_DECODE_TOKENS,
    PACK_REVISION_RE,
    REPOSITORY_RE,
    SHA256_RE,
    LocalAIManifest,
    ManifestArtifact,
    ManifestFile,
    manifest_path_suffix,
    safe_relative_manifest_path,
)
from app.services.local_ai.types import ModelRole

_POINTER_KEYS = frozenset({"pack_revision", "manifest_sha256"})
_STATE_KEYS = frozenset({"active", "previous"})
_ACTIVATION_STATE = "activation-state.json"
_ACTIVATION_LOCK = ".activation.lock"
_PROGRESS_KEYS = frozenset({"role", "bytes_done", "bytes_total"})
_MANIFEST_METADATA = ".manifest.json"
_OPERATION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,191}$")
_OPERATION_SUFFIX_RE = re.compile(r"^[0-9a-f]{32}$")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[Path, threading.RLock] = {}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _manifest_payload(manifest: LocalAIManifest) -> dict[str, Any]:
    return asdict(manifest)


def _manifest_from_canonical_metadata(raw: bytes) -> LocalAIManifest:
    try:
        value = json.loads(raw)
        artifacts = tuple(
            ManifestArtifact(
                role=ModelRole(artifact["role"]),
                repository=artifact["repository"],
                revision=artifact["revision"],
                quantization=artifact["quantization"],
                license=artifact["license"],
                attribution=artifact["attribution"],
                decode_limits=artifact["decode_limits"],
                files=tuple(ManifestFile(**file) for file in artifact["files"]),
            )
            for artifact in value["artifacts"]
        )
        manifest = LocalAIManifest(
            schema_version=value["schema_version"],
            pack_revision=value["pack_revision"],
            platform=value["platform"],
            runtime=value["runtime"],
            validation_suite_version=value["validation_suite_version"],
            artifacts=artifacts,
        )
        _validate_manifest(manifest)
        if _canonical_json(_manifest_payload(manifest)) != raw:
            raise LocalValidationError("Installed model manifest is invalid")
        return manifest
    except LocalValidationError:
        raise
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        raise LocalValidationError("Installed model manifest is invalid") from exc


def manifest_sha256(manifest: LocalAIManifest) -> str:
    """Return the digest of a canonical, fully validated manifest payload."""

    _validate_manifest(manifest)
    return hashlib.sha256(_canonical_json(_manifest_payload(manifest))).hexdigest()


def _validate_manifest(manifest: LocalAIManifest) -> None:
    """Revalidate dataclass manifests at the filesystem trust boundary."""

    try:
        if manifest.schema_version != 1 or manifest.platform != "apple_silicon":
            raise LocalValidationError("Model manifest is invalid")
        if (
            not isinstance(manifest.pack_revision, str)
            or PACK_REVISION_RE.fullmatch(manifest.pack_revision) is None
        ):
            raise LocalValidationError("Model pack revision is invalid")
        if not isinstance(manifest.runtime, dict) or set(manifest.runtime) != {
            "name",
            "version",
        }:
            raise LocalValidationError("Model manifest is invalid")
        if not all(
            isinstance(value, str)
            and value
            and len(value) <= 2048
            and not any(ord(character) < 32 for character in value)
            for value in manifest.runtime.values()
        ):
            raise LocalValidationError("Model manifest is invalid")
        if (
            not isinstance(manifest.validation_suite_version, str)
            or not manifest.validation_suite_version
            or len(manifest.validation_suite_version) > 2048
            or any(
                ord(character) < 32 for character in manifest.validation_suite_version
            )
        ):
            raise LocalValidationError("Model manifest is invalid")

        roles = [artifact.role for artifact in manifest.artifacts]
        if (
            len(roles) != 3
            or frozenset(roles) != EXPECTED_ROLES
            or len(set(roles)) != 3
        ):
            raise LocalValidationError("Model manifest roles are invalid")

        file_count = 0
        aggregate_size = 0
        for artifact in manifest.artifacts:
            if not isinstance(artifact.role, ModelRole):
                raise LocalValidationError("Model manifest role is invalid")
            if (
                not isinstance(artifact.repository, str)
                or REPOSITORY_RE.fullmatch(artifact.repository) is None
                or not isinstance(artifact.revision, str)
                or COMMIT_RE.fullmatch(artifact.revision) is None
            ):
                raise LocalValidationError("Model manifest source is invalid")
            text_metadata = (
                artifact.quantization,
                artifact.license,
                artifact.attribution,
            )
            if not all(
                isinstance(value, str)
                and value
                and len(value) <= 2048
                and not any(ord(character) < 32 for character in value)
                for value in text_metadata
            ):
                raise LocalValidationError("Model manifest metadata is invalid")
            if artifact.license not in ALLOWED_LICENSES:
                raise LocalValidationError("Model manifest metadata is invalid")
            if (
                not isinstance(artifact.decode_limits, dict)
                or set(artifact.decode_limits)
                != {"max_input_tokens", "max_output_tokens"}
                or not all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 < value <= MAX_DECODE_TOKENS
                    for value in artifact.decode_limits.values()
                )
            ):
                raise LocalValidationError("Model manifest metadata is invalid")
            if not artifact.files:
                raise LocalValidationError("Model manifest file set is invalid")

            paths: set[str] = set()
            for file in artifact.files:
                path = safe_relative_manifest_path(file.path)
                suffix = manifest_path_suffix(path)
                if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
                    raise LocalValidationError("Model manifest file type is forbidden")
                if path in paths:
                    raise LocalValidationError(
                        "Model manifest contains a duplicate path"
                    )
                paths.add(path)
                if (
                    not isinstance(file.sha256, str)
                    or SHA256_RE.fullmatch(file.sha256) is None
                ):
                    raise LocalValidationError("Model manifest SHA-256 is invalid")
                if (
                    not isinstance(file.size, int)
                    or isinstance(file.size, bool)
                    or file.size <= 0
                    or file.size > MAX_MANIFEST_FILE_BYTES
                ):
                    raise LocalValidationError(
                        "Model manifest file size exceeds the limit"
                    )
                file_count += 1
                aggregate_size += file.size

        if file_count > MAX_MANIFEST_FILES:
            raise LocalValidationError("Model manifest file count exceeds the limit")
        if aggregate_size > MAX_MANIFEST_PACK_BYTES:
            raise LocalValidationError(
                "Model manifest aggregate size exceeds the limit"
            )
    except LocalValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise LocalValidationError("Model manifest is invalid") from exc


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise LocalValidationError(
            "Model artifact filesystem operation failed"
        ) from exc


def _assert_real_directory(path: Path, *, missing_ok: bool = False) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise LocalValidationError("Model artifact directory is missing") from None
    except OSError as exc:
        raise LocalValidationError(
            "Model artifact filesystem operation failed"
        ) from exc
    if stat.S_ISLNK(status.st_mode):
        raise LocalValidationError("Model artifact path contains a symlink")
    if not stat.S_ISDIR(status.st_mode):
        raise LocalValidationError("Model artifact path must be a directory")
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise LocalValidationError(
            "Model artifact filesystem operation failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _assert_no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise LocalValidationError("Model artifact store path is invalid") from exc
        if stat.S_ISLNK(status.st_mode):
            raise LocalValidationError("Model artifact store path contains a symlink")


def _mutation_lock_for(root: Path) -> threading.RLock:
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(root)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[root] = lock
        return lock


class ArtifactStore:
    """Own immutable model packs, staging operations, and activation pointers."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(Path(root)))
        _assert_no_symlink_components(self.root)
        self.staging_dir = self.root / ".staging"
        self.packs_dir = self.root / "packs"
        self.operations_dir = self.root / "operations"
        self.activation_lock_path = self.root / _ACTIVATION_LOCK
        self._mutation_lock = _mutation_lock_for(self.root)
        self._initialize()

    def _initialize(self) -> None:
        try:
            if self.root.exists() or self.root.is_symlink():
                _assert_real_directory(self.root)
            else:
                self.root.mkdir(parents=True, mode=0o700)
            for path in (self.staging_dir, self.packs_dir, self.operations_dir):
                if path.exists() or path.is_symlink():
                    _assert_real_directory(path)
                else:
                    path.mkdir(mode=0o700)
                path.chmod(0o700)
            self._ensure_activation_lock()
            self.root.chmod(0o700)
        except LocalValidationError:
            raise
        except OSError as exc:
            raise LocalValidationError(
                "Model artifact store could not be initialized"
            ) from exc

    def _ensure_activation_lock(self) -> None:
        descriptor = -1
        try:
            descriptor = os.open(
                self.activation_lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise LocalValidationError("Model activation lock is invalid")
            os.fchmod(descriptor, 0o600)
        except LocalValidationError:
            raise
        except OSError as exc:
            raise LocalValidationError("Model activation lock is invalid") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def _mutation_guard(self) -> Any:
        with self._mutation_lock:
            descriptor = -1
            locked = False
            try:
                descriptor = os.open(
                    self.activation_lock_path,
                    os.O_RDWR
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                status = os.fstat(descriptor)
                if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                    raise LocalValidationError("Model activation lock is invalid")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                yield
            except LocalValidationError:
                raise
            except OSError as exc:
                raise LocalValidationError("Model activation lock failed") from exc
            finally:
                if locked:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                if descriptor >= 0:
                    os.close(descriptor)

    def validate_manifest(self, manifest: LocalAIManifest) -> None:
        """Validate a manifest before any network or filesystem path use."""

        _validate_manifest(manifest)

    def stage(self, pack_revision: str) -> Path:
        """Create an owner-only staging directory for one new pack revision."""

        if (
            not isinstance(pack_revision, str)
            or PACK_REVISION_RE.fullmatch(pack_revision) is None
        ):
            raise LocalValidationError("Model pack revision is invalid")
        destination = self.packs_dir / pack_revision
        if destination.exists() or destination.is_symlink():
            raise LocalValidationError("Model pack revision already exists")

        operation_id = f"{pack_revision}-{uuid.uuid4().hex}"
        path = self.staging_dir / operation_id
        try:
            path.mkdir(mode=0o700)
            path.chmod(0o700)
        except OSError as exc:
            raise LocalValidationError(
                "Model staging operation could not be created"
            ) from exc
        return path

    def _assert_staging_path(self, path: Path) -> None:
        if path.parent != self.staging_dir or not _OPERATION_RE.fullmatch(path.name):
            raise LocalValidationError("Model staging path is invalid")
        _assert_real_directory(path)

    def _open_directory_fd(
        self,
        path: Path | str,
        *,
        dir_fd: int | None = None,
    ) -> int:
        try:
            descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS, dir_fd=dir_fd)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                os.close(descriptor)
                raise LocalValidationError("Model artifact path must be a directory")
            return descriptor
        except LocalValidationError:
            raise
        except OSError as exc:
            raise LocalValidationError(
                "Model artifact filesystem operation failed"
            ) from exc

    def _read_regular_file_at(
        self,
        directory_fd: int,
        name: str,
        *,
        max_bytes: int,
    ) -> bytes:
        descriptor = -1
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                raise LocalValidationError("Model artifact path contains a symlink")
            descriptor = os.open(
                name,
                _FILE_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or opened.st_size > max_bytes
            ):
                raise LocalValidationError("Model artifact changed during verification")
            chunks: list[bytes] = []
            observed = 0
            while chunk := os.read(descriptor, min(1024 * 1024, max_bytes + 1)):
                observed += len(chunk)
                if observed > max_bytes:
                    raise LocalValidationError(
                        "Model artifact changed during verification"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
        except LocalValidationError:
            raise
        except OSError as exc:
            raise LocalValidationError(
                "Model artifact filesystem operation failed"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _verify_tree_fd(
        self,
        root_fd: int,
        manifest: LocalAIManifest,
        *,
        allow_manifest_metadata: bool,
        expected_metadata: bytes | None = None,
    ) -> None:
        expected_files: dict[str, tuple[int, str]] = {}
        expected_directories: set[str] = set()
        for artifact in manifest.artifacts:
            role = artifact.role.value
            expected_directories.add(role)
            for file in artifact.files:
                relative = Path(role, *file.path.split("/"))
                expected_files[relative.as_posix()] = (file.size, file.sha256)
                parent = relative.parent
                while parent != Path("."):
                    expected_directories.add(parent.as_posix())
                    parent = parent.parent

        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        aggregate = 0

        def visit(directory_fd: int, relative_directory: Path) -> None:
            nonlocal aggregate
            try:
                names = sorted(os.listdir(directory_fd))
            except OSError as exc:
                raise LocalValidationError(
                    "Model artifact filesystem operation failed"
                ) from exc
            for name in names:
                relative = relative_directory / name
                relative_text = relative.as_posix()
                try:
                    before = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise LocalValidationError(
                        "Model artifact filesystem operation failed"
                    ) from exc
                if stat.S_ISLNK(before.st_mode):
                    raise LocalValidationError("Model artifact path contains a symlink")
                if stat.S_ISDIR(before.st_mode):
                    child_fd = self._open_directory_fd(name, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (before.st_dev, before.st_ino) != (
                            opened.st_dev,
                            opened.st_ino,
                        ):
                            raise LocalValidationError(
                                "Model artifact changed during verification"
                            )
                        actual_directories.add(relative_text)
                        visit(child_fd, relative)
                    finally:
                        os.close(child_fd)
                    try:
                        after = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise LocalValidationError(
                            "Model artifact changed during verification"
                        ) from exc
                    if not stat.S_ISDIR(after.st_mode) or (
                        after.st_dev,
                        after.st_ino,
                    ) != (before.st_dev, before.st_ino):
                        raise LocalValidationError(
                            "Model artifact changed during verification"
                        )
                    continue
                if not stat.S_ISREG(before.st_mode):
                    raise LocalValidationError("Model artifact must be a regular file")

                descriptor = -1
                try:
                    descriptor = os.open(
                        name,
                        _FILE_OPEN_FLAGS,
                        dir_fd=directory_fd,
                    )
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode):
                        raise LocalValidationError(
                            "Model artifact must be a regular file"
                        )
                    if opened.st_nlink != 1:
                        raise LocalValidationError(
                            "Model artifact cannot be a hard link"
                        )
                    if (before.st_dev, before.st_ino) != (
                        opened.st_dev,
                        opened.st_ino,
                    ):
                        raise LocalValidationError(
                            "Model artifact changed during verification"
                        )
                    actual_files.add(relative_text)

                    expected = expected_files.get(relative_text)
                    if expected is not None:
                        expected_size, expected_hash = expected
                        if opened.st_size != expected_size:
                            raise LocalValidationError(
                                "Model artifact size does not match manifest"
                            )
                        aggregate += opened.st_size
                        if aggregate > MAX_MANIFEST_PACK_BYTES:
                            raise LocalValidationError(
                                "Model artifact aggregate size exceeds the limit"
                            )
                        digest = hashlib.sha256()
                        while chunk := os.read(descriptor, 1024 * 1024):
                            digest.update(chunk)
                        if digest.hexdigest() != expected_hash:
                            raise LocalValidationError(
                                "Model artifact SHA-256 does not match manifest"
                            )
                    elif (
                        allow_manifest_metadata and relative_text == _MANIFEST_METADATA
                    ):
                        metadata = b""
                        while chunk := os.read(descriptor, 1024 * 1024):
                            metadata += chunk
                            if len(metadata) > 1024 * 1024:
                                raise LocalValidationError(
                                    "Installed model manifest is invalid"
                                )
                        if (
                            expected_metadata is not None
                            and metadata != expected_metadata
                        ):
                            raise LocalValidationError(
                                "Installed model manifest changed during verification"
                            )
                    try:
                        after = os.stat(
                            name,
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise LocalValidationError(
                            "Model artifact changed during verification"
                        ) from exc
                    if (
                        not stat.S_ISREG(after.st_mode)
                        or after.st_nlink != 1
                        or after.st_size != opened.st_size
                        or (after.st_dev, after.st_ino)
                        != (opened.st_dev, opened.st_ino)
                    ):
                        raise LocalValidationError(
                            "Model artifact changed during verification"
                        )
                except LocalValidationError:
                    raise
                except OSError as exc:
                    raise LocalValidationError(
                        "Model artifact filesystem operation failed"
                    ) from exc
                finally:
                    if descriptor >= 0:
                        os.close(descriptor)
            try:
                if sorted(os.listdir(directory_fd)) != names:
                    raise LocalValidationError(
                        "Model artifact file set changed during verification"
                    )
            except LocalValidationError:
                raise
            except OSError as exc:
                raise LocalValidationError(
                    "Model artifact filesystem operation failed"
                ) from exc

        visit(root_fd, Path())
        allowed_files = set(expected_files)
        if allow_manifest_metadata:
            allowed_files.add(_MANIFEST_METADATA)
        if actual_files != allowed_files or actual_directories != expected_directories:
            raise LocalValidationError(
                "Model artifact file set does not match manifest"
            )

    def _verify_tree(
        self,
        root: Path,
        manifest: LocalAIManifest,
        *,
        allow_manifest_metadata: bool,
        expected_metadata: bytes | None = None,
    ) -> None:
        _validate_manifest(manifest)
        _assert_real_directory(root)
        root_fd = self._open_directory_fd(root)
        try:
            self._verify_tree_fd(
                root_fd,
                manifest,
                allow_manifest_metadata=allow_manifest_metadata,
                expected_metadata=expected_metadata,
            )
        finally:
            os.close(root_fd)

    def verify(self, staging_path: Path, manifest: LocalAIManifest) -> None:
        """Verify the exact per-role file set in a staging operation."""

        path = Path(staging_path)
        self._assert_staging_path(path)
        staged_revision, separator, operation_suffix = path.name.rpartition("-")
        if (
            separator != "-"
            or staged_revision != manifest.pack_revision
            or _OPERATION_SUFFIX_RE.fullmatch(operation_suffix) is None
        ):
            raise LocalValidationError("Model staging revision does not match manifest")
        self._verify_tree(path, manifest, allow_manifest_metadata=False)

    def _prepare_json_temp(self, destination: Path, payload: dict[str, Any]) -> Path:
        temp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(_canonical_json(payload))
                stream.flush()
                os.fsync(stream.fileno())
            temp.chmod(0o600)
            return temp
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            raise LocalValidationError("Model state could not be written") from exc

    def _write_json_atomic(self, destination: Path, payload: dict[str, Any]) -> None:
        temp = self._prepare_json_temp(destination, payload)
        try:
            os.replace(temp, destination)
            _fsync_directory(destination.parent)
        except LocalValidationError:
            temp.unlink(missing_ok=True)
            raise
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise LocalValidationError("Model state could not be activated") from exc

    def _validate_pointer(self, value: Any) -> dict[str, str]:
        if not isinstance(value, dict) or set(value) != _POINTER_KEYS:
            raise LocalValidationError("Model activation pointer is invalid")
        revision = value.get("pack_revision")
        digest = value.get("manifest_sha256")
        if (
            not isinstance(revision, str)
            or PACK_REVISION_RE.fullmatch(revision) is None
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise LocalValidationError("Model activation pointer is invalid")
        return {"pack_revision": revision, "manifest_sha256": digest}

    def _validate_state(self, value: Any) -> dict[str, dict[str, str] | None]:
        if not isinstance(value, dict) or set(value) != _STATE_KEYS:
            raise LocalValidationError("Model activation state is invalid")
        active = value.get("active")
        previous = value.get("previous")
        return {
            "active": self._validate_pointer(active) if active is not None else None,
            "previous": (
                self._validate_pointer(previous) if previous is not None else None
            ),
        }

    def _read_state(self) -> dict[str, dict[str, str] | None]:
        path = self.root / _ACTIVATION_STATE
        try:
            status = path.lstat()
        except FileNotFoundError:
            return {"active": None, "previous": None}
        except OSError as exc:
            raise LocalValidationError(
                "Model activation state could not be read"
            ) from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise LocalValidationError("Model activation state is invalid")
        if status.st_nlink != 1 or status.st_size > 4096:
            raise LocalValidationError("Model activation state is invalid")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise LocalValidationError("Model activation state is invalid") from exc
        return self._validate_state(value)

    def _write_state(
        self,
        *,
        active: dict[str, str] | None,
        previous: dict[str, str] | None,
    ) -> None:
        state = self._validate_state({"active": active, "previous": previous})
        self._write_json_atomic(self.root / _ACTIVATION_STATE, state)

    def activate(self, staging_path: Path, manifest: LocalAIManifest) -> None:
        """Verify and atomically activate a new immutable model pack."""

        with self._mutation_guard():
            self._activate_locked(staging_path, manifest)

    def _activate_locked(
        self,
        staging_path: Path,
        manifest: LocalAIManifest,
    ) -> None:
        self.verify(staging_path, manifest)
        destination = self.packs_dir / manifest.pack_revision
        if destination.exists() or destination.is_symlink():
            raise LocalValidationError("Model pack revision already exists")

        digest = manifest_sha256(manifest)
        state = self._read_state()
        current = state["active"]
        if current is not None:
            self._load_installed_manifest(
                self.packs_dir / current["pack_revision"],
                current,
            )
        new_pointer = {
            "pack_revision": manifest.pack_revision,
            "manifest_sha256": digest,
        }
        metadata_payload = _manifest_payload(manifest)
        metadata = _canonical_json(metadata_payload)
        self._write_json_atomic(
            Path(staging_path) / _MANIFEST_METADATA,
            metadata_payload,
        )
        try:
            os.replace(staging_path, destination)
            _fsync_directory(self.packs_dir)
            self._verify_tree(
                destination,
                manifest,
                allow_manifest_metadata=True,
                expected_metadata=metadata,
            )
            self._write_state(active=new_pointer, previous=current)
        except (OSError, LocalValidationError) as exc:
            if isinstance(exc, LocalValidationError):
                raise
            raise LocalValidationError("Model pack activation failed") from exc

    def active_revision(self) -> str | None:
        """Return the active pack revision, if one is selected."""

        pointer = self._read_state()["active"]
        return pointer["pack_revision"] if pointer is not None else None

    def _load_installed_manifest(
        self,
        pack: Path,
        pointer: dict[str, str],
    ) -> LocalAIManifest:
        root_fd = -1
        try:
            _assert_real_directory(pack)
            root_fd = self._open_directory_fd(pack)
            metadata = self._read_regular_file_at(
                root_fd,
                _MANIFEST_METADATA,
                max_bytes=1024 * 1024,
            )
            manifest = _manifest_from_canonical_metadata(metadata)
            if (
                manifest.pack_revision != pointer["pack_revision"]
                or manifest_sha256(manifest) != pointer["manifest_sha256"]
            ):
                raise LocalValidationError(
                    "Previously installed model pack is not verified"
                )
            self._verify_tree_fd(
                root_fd,
                manifest,
                allow_manifest_metadata=True,
                expected_metadata=metadata,
            )
        except LocalValidationError as exc:
            raise LocalValidationError(
                "Previously installed model pack is not verified"
            ) from exc
        finally:
            if root_fd >= 0:
                os.close(root_fd)
        return manifest

    def rollback(self) -> None:
        """Atomically reactivate the prior pack after verifying it in place."""

        with self._mutation_guard():
            self._rollback_locked()

    def _rollback_locked(self) -> None:
        state = self._read_state()
        previous = state["previous"]
        if previous is None:
            raise LocalValidationError("No verified previous model pack is available")
        pack = self.packs_dir / previous["pack_revision"]
        self._load_installed_manifest(pack, previous)
        self._write_state(active=previous, previous=state["active"])

    def operation_id(self, staging_path: Path) -> str:
        """Return a validated operation identifier for a staging path."""

        path = Path(staging_path)
        self._assert_staging_path(path)
        return path.name

    def write_progress(
        self,
        operation_id: str,
        payload: dict[str, int | str],
    ) -> None:
        """Persist a minimal, document-free progress payload atomically."""

        if (
            not isinstance(operation_id, str)
            or _OPERATION_RE.fullmatch(operation_id) is None
        ):
            raise LocalValidationError("Model download operation is invalid")
        if set(payload) != _PROGRESS_KEYS:
            raise LocalValidationError("Model download progress is invalid")
        role = payload.get("role")
        done = payload.get("bytes_done")
        total = payload.get("bytes_total")
        if (
            role not in {item.value for item in ModelRole}
            or not isinstance(done, int)
            or isinstance(done, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or done < 0
            or total <= 0
            or done > total
        ):
            raise LocalValidationError("Model download progress is invalid")
        self._write_json_atomic(self.operations_dir / f"{operation_id}.json", payload)

    def discard_staging(self, staging_path: Path) -> None:
        """Remove one failed staging operation without touching installed packs."""

        path = Path(staging_path)
        if path.parent != self.staging_dir or not _OPERATION_RE.fullmatch(path.name):
            raise LocalValidationError("Model staging path is invalid")
        if not path.exists() and not path.is_symlink():
            return
        _assert_real_directory(path)
        try:
            shutil.rmtree(path)
            _fsync_directory(self.staging_dir)
        except (OSError, LocalValidationError) as exc:
            raise LocalValidationError(
                "Failed model staging operation could not be removed"
            ) from exc

    def remove(self, role: ModelRole | str | None = None) -> None:
        """Remove model packs, or one role from the currently active pack."""

        with self._mutation_guard():
            self._remove_locked(role)

    def _remove_locked(self, role: ModelRole | str | None = None) -> None:
        if role is None:
            self._write_state(active=None, previous=None)
            try:
                for directory in (self.packs_dir, self.staging_dir):
                    _assert_real_directory(directory)
                    for child in list(directory.iterdir()):
                        status = child.lstat()
                        if stat.S_ISLNK(status.st_mode):
                            raise LocalValidationError(
                                "Model removal encountered a symlink"
                            )
                        if stat.S_ISDIR(status.st_mode):
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                    _fsync_directory(directory)
            except LocalValidationError:
                raise
            except OSError as exc:
                raise LocalValidationError("Model packs could not be removed") from exc
            return

        try:
            selected_role = role if isinstance(role, ModelRole) else ModelRole(role)
        except (TypeError, ValueError) as exc:
            raise LocalValidationError("Model role is invalid") from exc
        state = self._read_state()
        active = state["active"]
        if active is None:
            return
        previous = state["previous"]
        target_revision = active["pack_revision"]
        target_pack = self.packs_dir / target_revision
        role_path = target_pack / selected_role.value
        _assert_real_directory(target_pack)
        if role_path.exists() or role_path.is_symlink():
            _assert_real_directory(role_path)

        new_previous = (
            None
            if previous is not None and previous["pack_revision"] == target_revision
            else previous
        )
        self._write_state(active=None, previous=new_previous)
        if role_path.exists():
            try:
                shutil.rmtree(role_path)
                _fsync_directory(target_pack)
            except OSError as exc:
                raise LocalValidationError("Model role could not be removed") from exc
