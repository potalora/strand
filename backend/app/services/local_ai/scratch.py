"""Owner-only, no-follow plaintext scratch for strict-local jobs."""

from __future__ import annotations

import os
import re
import stat
import time
import math
from collections.abc import Iterable
from typing import BinaryIO
from pathlib import Path
from types import TracebackType
from uuid import uuid4

from app.services.local_ai.errors import LocalValidationError
from app.utils.file_utils import decrypt_file_stream_to

_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_SUPPORTED_SOURCE_SUFFIXES = frozenset({".pdf", ".rtf", ".tif", ".tiff"})
_JOB_MARKER_NAME = ".medtimeline-local-ai-job-v1"
_JOB_MARKER_CONTENT = b"MEDTIMELINE_LOCAL_AI_SCRATCH_V1\n"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_OPEN_FILE_FLAGS = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
_DEFAULT_MAX_ACTIVE_JOBS = 10_000


def _safe_job_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_JOB_ID.fullmatch(value)
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise LocalValidationError("Scratch job identifier is invalid.")
    return value


def _safe_file_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_FILE_NAME.fullmatch(value)
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise LocalValidationError("Scratch file name is invalid.")
    return value


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _same_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _identity(left) == _identity(right)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _open_secure_root(root: Path, *, create: bool) -> tuple[int, tuple[int, int]]:
    if create:
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError:
            raise LocalValidationError("Local scratch root is unavailable.") from None
    try:
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise LocalValidationError("Local scratch root must be a directory.")
        root_fd = os.open(root, _DIRECTORY_FLAGS)
        opened = os.fstat(root_fd)
        if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(info):
            os.close(root_fd)
            raise LocalValidationError("Local scratch root changed during validation.")
        os.fchmod(root_fd, 0o700)
        return root_fd, _identity(opened)
    except LocalValidationError:
        raise
    except OSError:
        raise LocalValidationError("Local scratch root is unavailable.") from None


def _remove_tree_contents(
    directory_fd: int,
    *,
    budget: list[int] | None = None,
    depth: int = 0,
    max_depth: int = 64,
    skip_names: frozenset[str] = frozenset(),
) -> bool:
    """Lazily unlink children; return False when a sweep budget is exhausted."""
    if depth > max_depth:
        return False
    try:
        entries = os.scandir(directory_fd)
    except OSError:
        return False
    complete = True
    with entries:
        for entry in entries:
            if entry.name in skip_names:
                continue
            if budget is not None:
                if budget[0] <= 0:
                    return False
                budget[0] -= 1
            name = entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError:
                complete = False
                continue
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                try:
                    child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue
                except OSError:
                    complete = False
                    continue
                try:
                    opened = os.fstat(child_fd)
                    if not stat.S_ISDIR(opened.st_mode) or not _same_entry(opened, info):
                        complete = False
                        continue
                    child_complete = _remove_tree_contents(
                        child_fd,
                        budget=budget,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                finally:
                    os.close(child_fd)
                if not child_complete:
                    complete = False
                    if budget is not None and budget[0] <= 0:
                        return False
                    continue
                try:
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if _identity(current) == _identity(info):
                        os.rmdir(name, dir_fd=directory_fd)
                    else:
                        complete = False
                except FileNotFoundError:
                    pass
                except OSError:
                    complete = False
            else:
                try:
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not _same_entry(current, info):
                        complete = False
                        continue
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    complete = False
    return complete


def _valid_job_marker(job_fd: int) -> bool:
    try:
        marker_fd = os.open(
            _JOB_MARKER_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=job_fd,
        )
    except OSError:
        return False
    try:
        info = os.fstat(marker_fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
        ):
            return False
        return os.read(marker_fd, len(_JOB_MARKER_CONTENT) + 1) == _JOB_MARKER_CONTENT
    finally:
        os.close(marker_fd)


def _remove_job_at(
    root_fd: int,
    job_id: str,
    expected_identity: tuple[int, int] | None = None,
    expected_mtime_ns: int | None = None,
    cutoff: float | None = None,
    *,
    budget: list[int] | None = None,
    max_depth: int = 64,
) -> bool:
    """Remove one validated direct child without following links."""
    _safe_job_id(job_id)
    try:
        job_fd = os.open(job_id, _DIRECTORY_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        info = os.fstat(job_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or (expected_identity is not None and _identity(info) != expected_identity)
            or (expected_mtime_ns is not None and info.st_mtime_ns != expected_mtime_ns)
            or (cutoff is not None and info.st_mtime > cutoff)
            or not _valid_job_marker(job_fd)
        ):
            return False
        revalidated = os.fstat(job_fd)
        if (
            _identity(revalidated) != _identity(info)
            or revalidated.st_mtime_ns != info.st_mtime_ns
        ):
            return False
        complete = _remove_tree_contents(
            job_fd,
            budget=budget,
            max_depth=max_depth,
            skip_names=frozenset({_JOB_MARKER_NAME}),
        )
        if complete:
            if budget is not None:
                if budget[0] <= 0:
                    complete = False
                else:
                    budget[0] -= 1
            if complete and _valid_job_marker(job_fd):
                try:
                    os.unlink(_JOB_MARKER_NAME, dir_fd=job_fd)
                except FileNotFoundError:
                    complete = False
            elif complete:
                complete = False
        if not complete:
            try:
                current = os.stat(job_id, dir_fd=root_fd, follow_symlinks=False)
                if _identity(current) == _identity(info):
                    os.utime(
                        job_id,
                        ns=(info.st_atime_ns, info.st_mtime_ns),
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
            except OSError:
                pass
    finally:
        os.close(job_fd)
    if not complete:
        return False
    try:
        current = os.stat(job_id, dir_fd=root_fd, follow_symlinks=False)
        if _identity(current) != _identity(info):
            return False
        os.rmdir(job_id, dir_fd=root_fd)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


class ScratchJob:
    """A single exclusive owner-only scratch directory."""

    def __init__(self, root: Path | str, job_id: str) -> None:
        self.root = Path(root)
        self.job_id = _safe_job_id(job_id)
        self.path = self.root / self.job_id
        self._root_identity: tuple[int, int] | None = None
        self._job_identity: tuple[int, int] | None = None
        self._job_fd: int | None = None
        self._files: dict[str, tuple[int, int]] = {}
        self._entered = False
        self._cleaned = False

    def __enter__(self) -> "ScratchJob":
        if self._entered:
            raise LocalValidationError("Scratch job is already active.")
        root_fd, root_identity = _open_secure_root(self.root, create=True)
        created = False
        job_fd = -1
        try:
            try:
                os.mkdir(self.job_id, mode=0o700, dir_fd=root_fd)
                created = True
            except FileExistsError:
                try:
                    existing = os.stat(self.job_id, dir_fd=root_fd, follow_symlinks=False)
                except OSError:
                    existing = None
                message = (
                    "Scratch job directory is unsafe."
                    if existing is None
                    or not stat.S_ISDIR(existing.st_mode)
                    or stat.S_ISLNK(existing.st_mode)
                    else "Scratch job directory already exists."
                )
                raise LocalValidationError(message) from None
            try:
                job_fd = os.open(self.job_id, _DIRECTORY_FLAGS, dir_fd=root_fd)
            except OSError:
                raise LocalValidationError("Scratch job directory is unsafe.") from None
            opened = os.fstat(job_fd)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(job_fd)
                raise LocalValidationError("Scratch job directory is unsafe.")
            os.fchmod(job_fd, 0o700)
            marker_fd = os.open(
                _JOB_MARKER_NAME,
                _FILE_FLAGS,
                0o600,
                dir_fd=job_fd,
            )
            try:
                os.fchmod(marker_fd, 0o600)
                os.write(marker_fd, _JOB_MARKER_CONTENT)
            finally:
                os.close(marker_fd)
            self._root_identity = root_identity
            self._job_identity = _identity(opened)
            self._job_fd = job_fd
            self._entered = True
            return self
        except BaseException:
            if job_fd >= 0 and self._job_fd is None:
                os.close(job_fd)
            if created:
                try:
                    cleanup_fd = os.open(self.job_id, _DIRECTORY_FLAGS, dir_fd=root_fd)
                except OSError:
                    cleanup_fd = -1
                if cleanup_fd >= 0:
                    try:
                        _remove_tree_contents(cleanup_fd)
                    finally:
                        os.close(cleanup_fd)
                    try:
                        os.rmdir(self.job_id, dir_fd=root_fd)
                    except OSError:
                        pass
            os.close(root_fd)
            raise
        finally:
            if self._entered:
                os.close(root_fd)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.cleanup()

    def _require_active(self) -> int:
        if not self._entered or self._cleaned or self._job_fd is None:
            raise LocalValidationError("Scratch job is not active.")
        return self._job_fd

    def _reserve_fd(self, filename: str) -> int:
        name = _safe_file_name(filename)
        job_fd = self._require_active()
        try:
            descriptor = os.open(name, _FILE_FLAGS, 0o600, dir_fd=job_fd)
        except OSError:
            raise LocalValidationError("Unsafe scratch file target.") from None
        try:
            os.fchmod(descriptor, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise LocalValidationError("Scratch file is unsafe.")
            self._files[name] = _identity(info)
            return descriptor
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=job_fd)
            except FileNotFoundError:
                pass
            raise

    def reserve_file(self, filename: str) -> Path:
        """Exclusively reserve a regular 0600 file and return its path."""
        descriptor = self._reserve_fd(filename)
        os.close(descriptor)
        return self.verify_file(filename)

    def create_file(self, filename: str, content: bytes) -> Path:
        """Exclusively create a 0600 scratch file with the supplied bytes."""
        descriptor = self._reserve_fd(filename)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as destination:
                destination.write(content)
        except BaseException:
            try:
                os.unlink(filename, dir_fd=self._require_active())
            except FileNotFoundError:
                pass
            raise
        return self.verify_file(filename)

    def open_file(self, filename: str) -> BinaryIO:
        """Open a tracked file no-follow and verify its original inode."""
        name = _safe_file_name(filename)
        expected = self._files.get(name)
        if expected is None:
            raise LocalValidationError("Scratch file is not tracked.")
        try:
            descriptor = os.open(
                name,
                _OPEN_FILE_FLAGS,
                dir_fd=self._require_active(),
            )
        except OSError:
            raise LocalValidationError("Scratch file changed unexpectedly.") from None
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or _identity(info) != expected
        ):
            os.close(descriptor)
            raise LocalValidationError("Scratch file changed unexpectedly.")
        return os.fdopen(descriptor, "r+b", closefd=True)

    def verify_file(self, filename: str) -> Path:
        """Verify a tracked pathname still names its original regular inode."""
        name = _safe_file_name(filename)
        expected = self._files.get(name)
        try:
            info = os.stat(name, dir_fd=self._require_active(), follow_symlinks=False)
        except OSError:
            raise LocalValidationError("Scratch file changed unexpectedly.") from None
        if (
            expected is None
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or _identity(info) != expected
        ):
            raise LocalValidationError("Scratch file changed unexpectedly.")
        return self.path / name

    def remove_file(self, filename: str) -> None:
        """Unlink one direct scratch child without following it."""
        name = _safe_file_name(filename)
        try:
            info = os.stat(name, dir_fd=self._require_active(), follow_symlinks=False)
        except FileNotFoundError:
            return
        expected = self._files.get(name)
        if expected is not None and _identity(info) != expected:
            return
        try:
            os.unlink(name, dir_fd=self._require_active())
        except FileNotFoundError:
            return
        self._files.pop(name, None)

    def decrypt_to_file(self, encrypted_path: Path | str) -> Path:
        """Stream-decrypt an encrypted upload into a secure scratch file."""
        source = Path(encrypted_path)
        suffix = source.suffix.lower()
        if suffix not in _SUPPORTED_SOURCE_SUFFIXES:
            raise LocalValidationError("Local source type is unsupported.")
        try:
            source_info = source.lstat()
        except OSError:
            raise LocalValidationError("Local encrypted source is unavailable.") from None
        if not stat.S_ISREG(source_info.st_mode) or stat.S_ISLNK(source_info.st_mode):
            raise LocalValidationError("Local encrypted source is unsafe.")

        filename = f"source-{uuid4().hex}{suffix}"
        self.reserve_file(filename)
        try:
            source_fd = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            opened_source = os.fstat(source_fd)
            if _identity(opened_source) != _identity(source_info):
                os.close(source_fd)
                raise LocalValidationError("Local encrypted source changed unexpectedly.")
            with os.fdopen(source_fd, "rb", closefd=True) as encrypted, self.open_file(
                filename
            ) as plain:
                decrypt_file_stream_to(encrypted, plain)
                plain.flush()
                os.fsync(plain.fileno())
            return self.verify_file(filename)
        except LocalValidationError:
            self.remove_file(filename)
            raise
        except BaseException:
            self.remove_file(filename)
            raise LocalValidationError(
                "Encrypted local document could not be decrypted."
            ) from None

    def cleanup(self) -> None:
        """Idempotently remove this job without following any directory link."""
        if self._cleaned or not self._entered:
            return
        job_fd = self._require_active()
        if not _remove_tree_contents(job_fd, max_depth=64):
            raise LocalValidationError("Local scratch cleanup is incomplete.")
        try:
            root_fd, root_identity = _open_secure_root(self.root, create=False)
        except LocalValidationError:
            os.close(job_fd)
            self._job_fd = None
            self._files.clear()
            self._cleaned = True
            raise LocalValidationError(
                "Local scratch containment changed after secure purge."
            ) from None
        try:
            if root_identity != self._root_identity:
                os.close(job_fd)
                self._job_fd = None
                self._files.clear()
                self._cleaned = True
                raise LocalValidationError("Local scratch root changed during cleanup.")
            try:
                current = os.stat(self.job_id, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            except OSError:
                raise LocalValidationError("Local scratch cleanup is incomplete.") from None
            if current is None or _identity(current) != self._job_identity:
                os.close(job_fd)
                self._job_fd = None
                self._files.clear()
                self._cleaned = True
                raise LocalValidationError("Local scratch job changed during cleanup.")
            try:
                os.rmdir(self.job_id, dir_fd=root_fd)
            except OSError:
                raise LocalValidationError("Local scratch cleanup is incomplete.") from None
            os.close(job_fd)
            self._job_fd = None
            self._files.clear()
            self._cleaned = True
        finally:
            os.close(root_fd)


def sweep_stale_scratch(
    root: Path | str,
    *,
    stale_after_seconds: float,
    active_job_ids: Iterable[str] = (),
    max_entries: int = 10_000,
    max_nodes: int = 100_000,
    max_depth: int = 64,
    max_active_jobs: int = _DEFAULT_MAX_ACTIVE_JOBS,
    now: float | None = None,
) -> int:
    """Boundedly remove stale valid job directories without following links."""
    if (
        isinstance(stale_after_seconds, bool)
        or not isinstance(stale_after_seconds, (int, float))
        or not math.isfinite(stale_after_seconds)
        or stale_after_seconds <= 0
    ):
        raise LocalValidationError("Scratch recovery age must be positive and finite.")
    if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
        raise LocalValidationError("Scratch recovery bound must be positive.")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise LocalValidationError("Scratch recovery node bound must be positive.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0:
        raise LocalValidationError("Scratch recovery depth must be positive.")
    if (
        isinstance(max_active_jobs, bool)
        or not isinstance(max_active_jobs, int)
        or max_active_jobs <= 0
    ):
        raise LocalValidationError("Scratch active job bound must be positive.")
    if now is not None and (
        isinstance(now, bool)
        or not isinstance(now, (int, float))
        or not math.isfinite(now)
    ):
        raise LocalValidationError("Scratch recovery time must be finite.")
    if isinstance(active_job_ids, (str, bytes)):
        raise LocalValidationError("Scratch active job identifiers must be an iterable.")
    try:
        active_iterator = iter(active_job_ids)
    except TypeError:
        raise LocalValidationError(
            "Scratch active job identifiers must be an iterable."
        ) from None
    active: set[str] = set()
    for index, value in enumerate(active_iterator):
        if index >= max_active_jobs:
            raise LocalValidationError("Scratch has too many active job identifiers.")
        active.add(_safe_job_id(value))
    cutoff = (time.time() if now is None else now) - stale_after_seconds
    root_fd, _ = _open_secure_root(Path(root), create=True)
    removed = 0
    node_budget = [max_nodes]
    try:
        try:
            entries = os.scandir(root_fd)
        except FileNotFoundError:
            return 0
        with entries:
            for index, entry in enumerate(entries):
                if index >= max_entries:
                    break
                name = entry.name
                if (
                    name in active
                    or not _SAFE_JOB_ID.fullmatch(name)
                    or name in {".", ".."}
                ):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o700
                    or info.st_uid != os.geteuid()
                    or info.st_mtime > cutoff
                ):
                    continue
                try:
                    removed += int(
                        _remove_job_at(
                            root_fd,
                            name,
                            _identity(info),
                            info.st_mtime_ns,
                            cutoff,
                            budget=node_budget,
                            max_depth=max_depth,
                        )
                    )
                except FileNotFoundError:
                    continue
        return removed
    finally:
        os.close(root_fd)
