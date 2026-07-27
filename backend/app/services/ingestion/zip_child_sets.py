"""Crash-safe set publication for encrypted children extracted from ZIP uploads."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.uploaded_file import UploadedFile
from app.utils.file_utils import ENC_MAGIC, EncryptedFileWriter

STAGING_EXTRACTION_STATUS = "staging_extraction"
_FINAL_SET_PATTERN = re.compile(r"^medtimeline-zip-set-[0-9a-f]{32}$")
_PENDING_SET_PATTERN = re.compile(r"^\.medtimeline-zip-set-[0-9a-f]{32}\.pending$")
_SAFE_CHILD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
    os, "O_NOFOLLOW", 0
)
_FILE_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_ENCRYPT_CHUNK = 1024 * 1024
_RECOVERY_ERROR = [
    {
        "error": "Encrypted upload set recovery failed.",
        "error_type": "LocalStorageRecoveryError",
    }
]


class _Writer(Protocol):
    def __init__(self, fileobj: BinaryIO) -> None: ...

    def write_chunk(self, plaintext: bytes) -> None: ...

    def finalize(self) -> None: ...


def _fsync(descriptor: int, event: str) -> None:
    """Durably sync a descriptor; ``event`` documents ordering for tests."""
    del event
    os.fsync(descriptor)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _safe_child_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_CHILD_NAME.fullmatch(value)
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError("Encrypted ZIP child name is invalid")
    return value


def _pending_name(final_name: str) -> str:
    if not _FINAL_SET_PATTERN.fullmatch(final_name):
        raise ValueError("Encrypted ZIP set name is invalid")
    return f".{final_name}.pending"


def _final_name(pending_name: str) -> str:
    if not _PENDING_SET_PATTERN.fullmatch(pending_name):
        raise ValueError("Encrypted ZIP pending set name is invalid")
    return pending_name[1:-8]


def _open_upload_root(root: Path, *, create: bool) -> tuple[int, tuple[int, int]]:
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.geteuid()
    ):
        raise ValueError("Encrypted upload root is unsafe")
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(info):
        os.close(descriptor)
        raise ValueError("Encrypted upload root changed during validation")
    os.fchmod(descriptor, 0o700)
    return descriptor, _identity(opened)


class ZipChildSet:
    """One owner-only pending directory published atomically as a complete set."""

    def __init__(
        self,
        upload_root: Path | str,
        *,
        writer_factory: type[_Writer] = EncryptedFileWriter,
    ) -> None:
        self.root = Path(upload_root).absolute()
        self.root_fd, self.root_identity = _open_upload_root(self.root, create=True)
        self.final_name = f"medtimeline-zip-set-{uuid4().hex}"
        self.pending_name = _pending_name(self.final_name)
        self.final_path = self.root / self.final_name
        self.pending_path = self.root / self.pending_name
        self._files: dict[str, tuple[int, int]] = {}
        self._sealed = False
        self._published = False
        self._closed = False
        self._writer_factory = writer_factory
        self.pending_fd = -1
        pending_created = False
        created_identity: tuple[int, int] | None = None

        try:
            os.mkdir(self.pending_name, mode=0o700, dir_fd=self.root_fd)
            pending_created = True
            scanned = os.stat(
                self.pending_name,
                dir_fd=self.root_fd,
                follow_symlinks=False,
            )
            created_identity = _identity(scanned)
            self.pending_fd = os.open(
                self.pending_name,
                _DIRECTORY_FLAGS,
                dir_fd=self.root_fd,
            )
            os.fchmod(self.pending_fd, 0o700)
            opened = os.fstat(self.pending_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or _identity(opened) != _identity(scanned)
            ):
                raise ValueError("Encrypted ZIP pending set changed during validation")
            self.pending_identity = _identity(opened)
            _fsync(self.root_fd, "upload_root_after_pending_create")
        except BaseException:
            try:
                if self.pending_fd >= 0:
                    try:
                        os.close(self.pending_fd)
                    except OSError:
                        pass
                    finally:
                        self.pending_fd = -1
            finally:
                try:
                    if pending_created:
                        try:
                            _remove_set_directory(
                                self.root_fd,
                                self.pending_name,
                                expected_identity=created_identity,
                                max_files=1,
                            )
                        except BaseException:
                            pass
                finally:
                    try:
                        os.close(self.root_fd)
                    except OSError:
                        pass
                    self._closed = True
            raise

    def child_path(self, name: str) -> Path:
        """Return the final durable path recorded in the database."""
        return self.final_path / _safe_child_name(name)

    def encrypt(self, source: Path, name: str) -> tuple[int, int]:
        """Stream-encrypt and fsync one child inside the pending directory."""
        if self._sealed or self._published or self._closed:
            raise ValueError("Encrypted ZIP set is no longer writable")
        name = _safe_child_name(name)
        descriptor = os.open(
            name,
            _FILE_CREATE_FLAGS,
            0o600,
            dir_fd=self.pending_fd,
        )
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        self._files[name] = identity
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as ciphertext, source.open(
                "rb"
            ) as plaintext:
                descriptor = -1
                writer = self._writer_factory(ciphertext)
                while True:
                    chunk = plaintext.read(_ENCRYPT_CHUNK)
                    if not chunk:
                        break
                    writer.write_chunk(chunk)
                writer.finalize()
                ciphertext.flush()
                _fsync(ciphertext.fileno(), "ciphertext_file")
            return identity
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def seal(self) -> None:
        """Fsync the pending directory after every child has been created."""
        if self._closed or self._published:
            raise ValueError("Encrypted ZIP set is unavailable")
        if not self._files:
            raise ValueError("Encrypted ZIP set is empty")
        valid, identity = _validate_set_directory(
            self.root_fd,
            self.pending_name,
            set(self._files),
            max_files=len(self._files),
        )
        if not valid or identity != self.pending_identity:
            raise ValueError("Encrypted ZIP pending set failed validation")
        _fsync(self.pending_fd, "pending_directory_after_files")
        self._sealed = True

    def discard(self, name: str) -> None:
        """Remove one failed staged child while the set is still private."""
        name = _safe_child_name(name)
        expected = self._files.get(name)
        if expected is None:
            return
        try:
            current = os.stat(name, dir_fd=self.pending_fd, follow_symlinks=False)
            if _identity(current) != expected:
                raise ValueError("Encrypted ZIP child changed during cleanup")
            os.unlink(name, dir_fd=self.pending_fd)
            _fsync(self.pending_fd, "pending_directory_after_child_discard")
            self._files.pop(name, None)
        except FileNotFoundError:
            self._files.pop(name, None)

    def publish(self) -> None:
        """Atomically rename the entire pending directory to its final name."""
        if not self._sealed or self._closed or self._published:
            raise ValueError("Encrypted ZIP set is not ready for publication")
        current = os.stat(
            self.pending_name,
            dir_fd=self.root_fd,
            follow_symlinks=False,
        )
        if _identity(current) != self.pending_identity:
            raise ValueError("Encrypted ZIP pending set changed before publication")
        os.rename(
            self.pending_name,
            self.final_name,
            src_dir_fd=self.root_fd,
            dst_dir_fd=self.root_fd,
        )
        _fsync(self.root_fd, "upload_root_after_set_publish")
        published = os.stat(
            self.final_name,
            dir_fd=self.root_fd,
            follow_symlinks=False,
        )
        if _identity(published) != self.pending_identity:
            raise ValueError("Encrypted ZIP set changed during publication")
        self._published = True

    def close(self, *, cleanup_pending: bool = True) -> None:
        """Close descriptors, optionally removing an uncommitted pending set."""
        if self._closed:
            return
        if cleanup_pending and not self._published:
            _remove_set_directory(
                self.root_fd,
                self.pending_name,
                expected_identity=self.pending_identity,
                max_files=max(len(self._files) + 1, 1),
            )
        os.close(self.pending_fd)
        os.close(self.root_fd)
        self._closed = True

    def remove_published(self) -> bool:
        """Remove this set after a caller-specific post-publication failure."""
        if not self._published or self._closed:
            return False
        return _remove_set_directory(
            self.root_fd,
            self.final_name,
            max_files=max(len(self._files) + 1, 1),
        )


@dataclass(frozen=True)
class ReconciliationResult:
    recovered_groups: int = 0
    failed_groups: int = 0
    removed_orphans: int = 0
    bounded: bool = False


def _parse_storage_path(root: Path, storage_path: str) -> tuple[str, str] | None:
    try:
        root_absolute = root.absolute()
        candidate = Path(storage_path)
        if not candidate.is_absolute():
            return None
        relative = candidate.relative_to(root_absolute)
    except (TypeError, ValueError):
        return None
    if len(relative.parts) != 2 or any(part in {".", ".."} for part in relative.parts):
        return None
    final_name, child_name = relative.parts
    if not _FINAL_SET_PATTERN.fullmatch(final_name):
        return None
    try:
        _safe_child_name(child_name)
    except ValueError:
        return None
    return final_name, child_name


def _validate_set_directory(
    root_fd: int,
    name: str,
    expected_files: set[str],
    *,
    max_files: int,
) -> tuple[bool, tuple[int, int] | None]:
    try:
        scanned = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(scanned.st_mode)
            or stat.S_ISLNK(scanned.st_mode)
            or stat.S_IMODE(scanned.st_mode) != 0o700
            or scanned.st_uid != os.geteuid()
        ):
            return False, None
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=root_fd)
    except OSError:
        return False, None
    try:
        opened = os.fstat(directory_fd)
        if _identity(opened) != _identity(scanned):
            return False, None
        entries: set[str] = set()
        with os.scandir(directory_fd) as iterator:
            for index, entry in enumerate(iterator):
                if index >= max_files:
                    return False, None
                child_name = entry.name
                if child_name not in expected_files:
                    return False, None
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    return False, None
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.geteuid()
                    or info.st_nlink != 1
                ):
                    return False, None
                try:
                    file_fd = os.open(
                        child_name,
                        _FILE_READ_FLAGS,
                        dir_fd=directory_fd,
                    )
                except OSError:
                    return False, None
                try:
                    current = os.fstat(file_fd)
                    if _identity(current) != _identity(info):
                        return False, None
                    if os.read(file_fd, len(ENC_MAGIC)) != ENC_MAGIC:
                        return False, None
                finally:
                    os.close(file_fd)
                entries.add(child_name)
        return entries == expected_files, _identity(opened)
    finally:
        os.close(directory_fd)


def _remove_set_directory(
    root_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int] | None = None,
    max_files: int,
    require_encrypted: bool = False,
) -> bool:
    """Remove one validated flat owner-only set directory without following links."""
    durable = True
    try:
        scanned = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(scanned.st_mode)
            or stat.S_ISLNK(scanned.st_mode)
            or stat.S_IMODE(scanned.st_mode) != 0o700
            or scanned.st_uid != os.geteuid()
            or (
                expected_identity is not None
                and _identity(scanned) != expected_identity
            )
        ):
            return False
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=root_fd)
    except OSError:
        return False
    try:
        if _identity(os.fstat(directory_fd)) != _identity(scanned):
            return False
        names: list[tuple[str, tuple[int, int]]] = []
        with os.scandir(directory_fd) as iterator:
            for index, entry in enumerate(iterator):
                if index >= max_files:
                    return False
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    return False
                if (
                    not _SAFE_CHILD_NAME.fullmatch(entry.name)
                    or not stat.S_ISREG(info.st_mode)
                    or stat.S_ISLNK(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.geteuid()
                    or info.st_nlink != 1
                ):
                    return False
                if require_encrypted:
                    try:
                        file_fd = os.open(
                            entry.name,
                            _FILE_READ_FLAGS,
                            dir_fd=directory_fd,
                        )
                    except OSError:
                        return False
                    try:
                        opened = os.fstat(file_fd)
                        if (
                            _identity(opened) != _identity(info)
                            or os.read(file_fd, len(ENC_MAGIC)) != ENC_MAGIC
                        ):
                            return False
                    finally:
                        os.close(file_fd)
                names.append((entry.name, _identity(info)))
        for child_name, expected in names:
            try:
                current = os.stat(
                    child_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if _identity(current) != expected:
                    return False
                os.unlink(child_name, dir_fd=directory_fd)
            except OSError:
                return False
        try:
            _fsync(directory_fd, "pending_directory_after_cleanup")
        except OSError:
            durable = False
    finally:
        os.close(directory_fd)
    try:
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if _identity(current) != _identity(scanned):
            return False
        os.rmdir(name, dir_fd=root_fd)
        try:
            _fsync(root_fd, "upload_root_after_pending_cleanup")
        except OSError:
            durable = False
        return durable
    except OSError:
        return False


async def _fail_rows(db: AsyncSession, rows: list[UploadedFile]) -> None:
    for row in rows:
        row.ingestion_status = "failed"
        row.ingestion_errors = list(_RECOVERY_ERROR)
    await db.commit()


async def _has_rows_for_set(
    db: AsyncSession,
    root: Path,
    final_name: str,
) -> bool:
    prefix = str(root.absolute() / final_name) + os.sep
    result = await db.execute(
        select(UploadedFile.id)
        .where(UploadedFile.storage_path.startswith(prefix, autoescape=True))
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def reconcile_zip_child_sets(
    db: AsyncSession,
    upload_root: Path | str,
    *,
    max_rows: int = 10_000,
    max_directories: int = 1_000,
    max_root_entries: int = 10_000,
    max_files_per_set: int = 1_000,
) -> ReconciliationResult:
    """Recover committed staging groups before the extraction worker starts."""
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, int)
        or max_rows <= 0
        or isinstance(max_directories, bool)
        or not isinstance(max_directories, int)
        or max_directories <= 0
        or isinstance(max_root_entries, bool)
        or not isinstance(max_root_entries, int)
        or max_root_entries <= 0
        or isinstance(max_files_per_set, bool)
        or not isinstance(max_files_per_set, int)
        or max_files_per_set <= 0
    ):
        raise ValueError("Encrypted ZIP recovery bounds are invalid")

    root = Path(upload_root).absolute()
    root_fd, _root_identity = _open_upload_root(root, create=True)
    recovered = 0
    failed = 0
    removed_orphans = 0
    try:
        result = await db.execute(
            select(UploadedFile)
            .where(
                UploadedFile.ingestion_status == STAGING_EXTRACTION_STATUS,
                UploadedFile.file_category == "unstructured",
            )
            .order_by(UploadedFile.id)
            .limit(max_rows + 1)
        )
        rows = list(result.scalars().all())
        if len(rows) > max_rows:
            await db.rollback()
            return ReconciliationResult(bounded=True)

        grouped: dict[str, list[tuple[UploadedFile, str]]] = {}
        invalid_rows: list[UploadedFile] = []
        for row in rows:
            parsed = _parse_storage_path(root, row.storage_path)
            if parsed is None:
                invalid_rows.append(row)
                continue
            final_name, child_name = parsed
            grouped.setdefault(final_name, []).append((row, child_name))

        for row in invalid_rows:
            await _fail_rows(db, [row])
            failed += 1

        for final_name, members in grouped.items():
            group_rows = [member[0] for member in members]
            child_names = [member[1] for member in members]
            expected_files = set(child_names)
            users = {row.user_id for row in group_rows}
            if (
                len(users) != 1
                or len(expected_files) != len(child_names)
                or len(expected_files) > max_files_per_set
            ):
                await _fail_rows(db, group_rows)
                failed += 1
                continue

            pending_name = _pending_name(final_name)
            try:
                pending_info = os.stat(
                    pending_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError:
                pending_info = None
            try:
                final_info = os.stat(
                    final_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError:
                final_info = None

            valid = False
            if pending_info is not None and final_info is None:
                valid, validated_identity = _validate_set_directory(
                    root_fd,
                    pending_name,
                    expected_files,
                    max_files=max_files_per_set,
                )
                if valid:
                    try:
                        current = os.stat(
                            pending_name,
                            dir_fd=root_fd,
                            follow_symlinks=False,
                        )
                        if _identity(current) != validated_identity:
                            raise OSError("Encrypted ZIP pending set changed")
                        os.rename(
                            pending_name,
                            final_name,
                            src_dir_fd=root_fd,
                            dst_dir_fd=root_fd,
                        )
                        _fsync(root_fd, "upload_root_after_recovery_publish")
                        published = os.stat(
                            final_name,
                            dir_fd=root_fd,
                            follow_symlinks=False,
                        )
                        if _identity(published) != validated_identity:
                            raise OSError("Encrypted ZIP published set changed")
                    except OSError:
                        valid = False
            elif final_info is not None and pending_info is None:
                valid, validated_identity = _validate_set_directory(
                    root_fd,
                    final_name,
                    expected_files,
                    max_files=max_files_per_set,
                )
                if valid:
                    try:
                        current = os.stat(
                            final_name,
                            dir_fd=root_fd,
                            follow_symlinks=False,
                        )
                        valid = _identity(current) == validated_identity
                    except OSError:
                        valid = False

            if not valid:
                await _fail_rows(db, group_rows)
                failed += 1
                continue

            for row in group_rows:
                row.ingestion_status = "pending_extraction"
                row.ingestion_errors = []
            await db.commit()
            recovered += 1

        try:
            directory_entries = os.scandir(root_fd)
        except OSError:
            directory_entries = None
        if directory_entries is not None:
            recognized_names: list[str] = []
            root_entry_count = 0
            with directory_entries:
                for entry in directory_entries:
                    root_entry_count += 1
                    if root_entry_count > max_root_entries:
                        break
                    name = entry.name
                    if not (
                        _PENDING_SET_PATTERN.fullmatch(name)
                        or _FINAL_SET_PATTERN.fullmatch(name)
                    ):
                        continue
                    recognized_names.append(name)
                    if len(recognized_names) > max_directories:
                        break
            if (
                root_entry_count > max_root_entries
                or len(recognized_names) > max_directories
            ):
                return ReconciliationResult(
                    recovered,
                    failed,
                    removed_orphans,
                    bounded=True,
                )
            for name in sorted(recognized_names):
                final_name = (
                    _final_name(name)
                    if _PENDING_SET_PATTERN.fullmatch(name)
                    else name
                )
                if await _has_rows_for_set(db, root, final_name):
                    continue
                if _remove_set_directory(
                    root_fd,
                    name,
                    max_files=max_files_per_set,
                    require_encrypted=True,
                ):
                    removed_orphans += 1

        return ReconciliationResult(recovered, failed, removed_orphans)
    finally:
        os.close(root_fd)
