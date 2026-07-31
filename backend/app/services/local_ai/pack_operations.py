"""Persist document-free model-pack operations and validation receipts."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import stat
import threading
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

from app.services.local_ai.artifact_store import ArtifactStore, manifest_sha256
from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import LocalAIManifest
from app.services.local_ai.types import ModelRole
from app.services.local_ai.validation_receipt import RuntimeValidationReceipt

OperationAction = Literal["install", "update", "verify", "rollback"]
OperationState = Literal["queued", "running", "paused", "failed", "completed"]

_ACTIONS = frozenset({"install", "update", "verify", "rollback"})
_STATES = frozenset({"queued", "running", "paused", "failed", "completed"})
_MESSAGES = frozenset(
    {
        "Waiting for model pack operation.",
        "Downloading verified model files.",
        "Running local validation fixtures.",
        "Model pack operation paused.",
        "Model pack operation failed.",
        "Model pack is ready.",
        "Restoring previous verified model pack.",
        "Previous verified model pack restored.",
    }
)
_OPERATION_KEYS = frozenset(
    {
        "id",
        "action",
        "state",
        "current_role",
        "bytes_done",
        "bytes_total",
        "message",
        "retryable",
        "pack_revision",
        "manifest_sha256",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "id",
        "action",
        "state",
        "current_role",
        "bytes_done",
        "bytes_total",
        "message",
        "retryable",
    }
)
_LOCK = threading.RLock()
_LIFECYCLE_LOCK = ".lifecycle.lock"


def _release_lease_descriptor(descriptor: int) -> None:
    """Release one lease descriptor without ever acting on it twice."""

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(descriptor)
    except OSError:
        pass


class OperationLease:
    """Explicitly owned runner lease that can be handed to a background task."""

    def __init__(self, descriptor: int) -> None:
        self._finalizer = weakref.finalize(
            self,
            _release_lease_descriptor,
            descriptor,
        )

    def release(self) -> None:
        """Release the cross-process lease exactly once."""

        self._finalizer()


def platform_profile() -> tuple[str, bool]:
    """Return a stable platform label and 16-GB Apple compatibility result."""

    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        return "unsupported", False
    try:
        physical_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES")
        )
    except (OSError, TypeError, ValueError):
        return "apple_silicon", False
    return "apple_silicon", physical_bytes >= 16 * 1024**3


def _uuid_text(value: object) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        raise LocalValidationError("Model pack operation is invalid") from None
    if str(parsed) != str(value):
        raise LocalValidationError("Model pack operation is invalid")
    return str(parsed)


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalValidationError("Model pack operation is invalid")
    return value


def _bounded_text(value: object, *, max_length: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_length
        or any(ord(character) < 32 for character in value)
    ):
        raise LocalValidationError("Model pack operation is invalid")
    return value


def _operation_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _OPERATION_KEYS:
        raise LocalValidationError("Model pack operation is invalid")
    operation_id = _uuid_text(value.get("id"))
    action = value.get("action")
    state = value.get("state")
    current_role = value.get("current_role")
    message = value.get("message")
    bytes_done = value.get("bytes_done")
    bytes_total = value.get("bytes_total")
    retryable = value.get("retryable")
    pack_revision = value.get("pack_revision")
    if (
        action not in _ACTIONS
        or state not in _STATES
        or current_role not in {None, *(role.value for role in ModelRole)}
        or message not in {None, *_MESSAGES}
        or not isinstance(bytes_done, int)
        or isinstance(bytes_done, bool)
        or not isinstance(bytes_total, int)
        or isinstance(bytes_total, bool)
        or bytes_done < 0
        or bytes_total < 0
        or bytes_done > bytes_total
        or not isinstance(retryable, bool)
    ):
        raise LocalValidationError("Model pack operation is invalid")
    return {
        "id": operation_id,
        "action": action,
        "state": state,
        "current_role": current_role,
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "message": message,
        "retryable": retryable,
        "pack_revision": _bounded_text(pack_revision, max_length=128),
        "manifest_sha256": _digest(value.get("manifest_sha256")),
    }


def _read_regular_json(path: Path, *, max_bytes: int = 16 * 1024) -> object:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LocalValidationError("Model pack state could not be read") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > max_bytes
    ):
        raise LocalValidationError("Model pack state is invalid")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LocalValidationError("Model pack state is invalid") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise LocalValidationError("Model pack state could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            descriptor = -1
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            os.fchmod(stream.fileno(), 0o600)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise LocalValidationError("Model pack state could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class PackOperationStore:
    """Owner-only JSON state for model-pack lifecycle operations."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.directory = artifact_store.operations_dir / "lifecycle"
        self.lock_path = self.directory / _LIFECYCLE_LOCK
        for directory in (self.directory,):
            try:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                directory.chmod(0o700)
            except OSError as exc:
                raise LocalValidationError(
                    "Model pack state could not be initialized"
                ) from exc
        self._ensure_lock_file()

    def _ensure_lock_file(self) -> None:
        descriptor = -1
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LocalValidationError("Model lifecycle lock is invalid")
            os.fchmod(descriptor, 0o600)
        except LocalValidationError:
            raise
        except OSError as exc:
            raise LocalValidationError("Model lifecycle lock is invalid") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @contextmanager
    def lifecycle_guard(self) -> Iterator[None]:
        """Serialize lifecycle state across threads and backend processes."""

        with _LOCK:
            descriptor = -1
            locked = False
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_RDWR
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise LocalValidationError("Model lifecycle lock is invalid")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
                yield
            except LocalValidationError:
                raise
            except OSError as exc:
                raise LocalValidationError("Model lifecycle lock failed") from exc
            finally:
                if locked:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                if descriptor >= 0:
                    os.close(descriptor)

    def acquire_operation_lease(
        self,
        operation_id: str,
        *,
        blocking: bool,
    ) -> OperationLease | None:
        """Acquire one transferable runner lease, or report a live peer."""

        operation_id = _uuid_text(operation_id)
        path = self.directory / f".{operation_id}.lease"
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise LocalValidationError("Model operation lease is invalid")
            os.fchmod(descriptor, 0o600)
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(descriptor, flags)
            except BlockingIOError:
                os.close(descriptor)
                return None
            return OperationLease(descriptor)
        except LocalValidationError:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise LocalValidationError("Model operation lease failed") from exc

    @contextmanager
    def operation_lease(
        self,
        operation_id: str,
        *,
        blocking: bool,
    ) -> Iterator[bool]:
        """Hold one runner lease, or report that a peer still owns it."""

        lease = self.acquire_operation_lease(operation_id, blocking=blocking)
        try:
            yield lease is not None
        finally:
            if lease is not None:
                lease.release()

    def _new_operation(
        self,
        *,
        action: OperationAction,
        manifest: LocalAIManifest,
        operation_id: str,
    ) -> dict[str, Any]:
        return _operation_payload(
            {
                "id": operation_id,
                "action": action,
                "state": "queued",
                "current_role": None,
                "bytes_done": 0,
                "bytes_total": (
                    sum(
                        item.size
                        for artifact in manifest.artifacts
                        for item in artifact.files
                    )
                    if action in {"install", "update"}
                    else 0
                ),
                "message": "Waiting for model pack operation.",
                "retryable": False,
                "pack_revision": manifest.pack_revision,
                "manifest_sha256": manifest_sha256(manifest),
            }
        )

    def create(
        self,
        *,
        action: OperationAction,
        manifest: LocalAIManifest,
    ) -> dict[str, Any]:
        with self.lifecycle_guard():
            if self.has_nonterminal_locked():
                raise LocalValidationError(
                    "A model pack lifecycle operation is already active"
                )
            operation_id = str(uuid.uuid4())
            payload = self._new_operation(
                action=action,
                manifest=manifest,
                operation_id=operation_id,
            )
            _write_json_atomic(self._path(operation_id), payload)
        return payload

    def create_claimed(
        self,
        *,
        action: OperationAction,
        manifest: LocalAIManifest,
    ) -> tuple[dict[str, Any], OperationLease]:
        """Create queued state while reserving its background runner."""

        with self.lifecycle_guard():
            if self.has_nonterminal_locked():
                raise LocalValidationError(
                    "A model pack lifecycle operation is already active"
                )
            operation_id = str(uuid.uuid4())
            lease = self.acquire_operation_lease(operation_id, blocking=False)
            if lease is None:
                raise LocalValidationError("Model operation lease is unavailable")
            try:
                payload = self._new_operation(
                    action=action,
                    manifest=manifest,
                    operation_id=operation_id,
                )
                _write_json_atomic(self._path(operation_id), payload)
            except BaseException:
                lease.release()
                raise
        return payload, lease

    def get(self, operation_id: str) -> dict[str, Any] | None:
        try:
            operation_id = _uuid_text(operation_id)
            value = _read_regular_json(self._path(operation_id))
        except FileNotFoundError:
            return None
        return _operation_payload(value)

    def transition(
        self,
        operation_id: str,
        *,
        expected_states: str | set[str] | frozenset[str],
        **changes: object,
    ) -> dict[str, Any]:
        """Compare-and-swap one operation from an explicitly expected state."""

        expected = (
            {expected_states}
            if isinstance(expected_states, str)
            else set(expected_states)
        )
        if not expected or not expected.issubset(_STATES):
            raise LocalValidationError("Model pack operation transition is invalid")
        with self.lifecycle_guard():
            current = self.get(operation_id)
            if current is None:
                raise LocalValidationError("Model pack operation was not found")
            if current["state"] not in expected:
                raise LocalValidationError("Model pack operation state changed")
            if set(changes) - _OPERATION_KEYS:
                raise LocalValidationError("Model pack operation is invalid")
            value = _operation_payload({**current, **changes})
            _write_json_atomic(self._path(operation_id), value)
            return value

    def update(self, operation_id: str, **changes: object) -> dict[str, Any]:
        """Update non-state fields without allowing an unchecked state change."""

        if "state" in changes:
            raise LocalValidationError(
                "Model pack state changes require compare-and-swap"
            )
        with self.lifecycle_guard():
            current = self.get(operation_id)
            if current is None:
                raise LocalValidationError("Model pack operation was not found")
            if set(changes) - _OPERATION_KEYS:
                raise LocalValidationError("Model pack operation is invalid")
            value = _operation_payload({**current, **changes})
            _write_json_atomic(self._path(operation_id), value)
            return value

    def latest(self) -> dict[str, Any] | None:
        candidates: list[tuple[int, str, dict[str, Any]]] = []
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError as exc:
            raise LocalValidationError("Model pack state could not be read") from exc
        for path in paths:
            try:
                value = _operation_payload(_read_regular_json(path))
                metadata = path.stat()
            except FileNotFoundError:
                continue
            candidates.append((metadata.st_mtime_ns, path.name, value))
        return max(candidates, default=(0, "", None))[2]

    def response(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = _operation_payload(value)
        return {key: payload[key] for key in _RESPONSE_KEYS}

    def has_nonterminal(self) -> bool:
        """Return whether any lifecycle operation can still mutate pack state."""

        with self.lifecycle_guard():
            return self.has_nonterminal_locked()

    def has_nonterminal_locked(
        self,
        *,
        exclude_operation_id: str | None = None,
    ) -> bool:
        """Return nonterminal state while the caller holds ``lifecycle_guard``."""

        excluded = (
            _uuid_text(exclude_operation_id)
            if exclude_operation_id is not None
            else None
        )
        return any(
            operation["id"] != excluded
            and operation["state"] in {"queued", "running", "paused"}
            for operation in self._all_operations()
        )

    def restart_claimed(
        self,
        operation_id: str,
        *,
        expected_state: Literal["failed", "paused"],
    ) -> tuple[dict[str, Any], OperationLease]:
        """Atomically requeue and reserve one restartable operation runner."""

        operation_id = _uuid_text(operation_id)
        with self.lifecycle_guard():
            lease = self.acquire_operation_lease(operation_id, blocking=False)
            if lease is None:
                raise LocalValidationError("Model operation runner is still active")
            try:
                current = self.get(operation_id)
                if current is None:
                    raise LocalValidationError("Model pack operation was not found")
                if current["state"] != expected_state or (
                    expected_state == "failed" and not current["retryable"]
                ):
                    raise LocalValidationError("Model pack operation state changed")
                if self.has_nonterminal_locked(exclude_operation_id=operation_id):
                    raise LocalValidationError(
                        "A model pack lifecycle operation is already active"
                    )
                self.artifact_store.sweep_orphan_staging()
                value = _operation_payload(
                    {
                        **current,
                        "state": "queued",
                        "current_role": None,
                        "bytes_done": 0,
                        "message": "Waiting for model pack operation.",
                        "retryable": False,
                    }
                )
                _write_json_atomic(self._path(operation_id), value)
            except BaseException:
                lease.release()
                raise
            return value, lease

    def ensure_no_nonterminal(self) -> None:
        """Reject removal while a lifecycle operation is nonterminal."""

        with self.lifecycle_guard():
            if self.has_nonterminal_locked():
                raise LocalValidationError(
                    "A model pack lifecycle operation is already active"
                )

    def reconcile_interrupted(self) -> int:
        """Make crash-interrupted operations honest and restartable."""

        reconciled = 0
        with self.lifecycle_guard():
            active: LocalAIManifest | None
            try:
                active = self.artifact_store.active_manifest()
            except LocalValidationError:
                active = None
            for operation in self._all_operations():
                if operation["state"] not in {"queued", "running", "paused"}:
                    continue
                with self.operation_lease(
                    operation["id"],
                    blocking=False,
                ) as acquired:
                    if not acquired:
                        continue
                    if (
                        operation["state"] == "running"
                        and operation["action"] in {"install", "update", "verify"}
                        and active is not None
                        and active.pack_revision == operation["pack_revision"]
                        and manifest_sha256(active) == operation["manifest_sha256"]
                    ):
                        value = {
                            **operation,
                            "state": "completed",
                            "current_role": None,
                            "bytes_done": operation["bytes_total"],
                            "message": "Model pack is ready.",
                            "retryable": False,
                        }
                    else:
                        retryable = not (
                            operation["state"] == "running"
                            and operation["action"] == "rollback"
                        )
                        value = {
                            **operation,
                            "state": "failed",
                            "current_role": None,
                            "bytes_done": 0,
                            "message": "Model pack operation failed.",
                            "retryable": retryable,
                        }
                    _write_json_atomic(
                        self._path(operation["id"]),
                        _operation_payload(value),
                    )
                    reconciled += 1
            if not self.has_nonterminal_locked():
                self.artifact_store.sweep_orphan_staging()
        return reconciled

    def mark_validated(
        self,
        manifest: LocalAIManifest,
        receipt: RuntimeValidationReceipt,
    ) -> None:
        self.artifact_store.refresh_validation_receipt(manifest, receipt)

    def is_validated(self, manifest: LocalAIManifest) -> bool:
        return self.artifact_store.has_validation_receipt(manifest)

    def remove_validation(self, manifest: LocalAIManifest) -> None:
        self.artifact_store.remove_validation_receipt(manifest)

    def _all_operations(self) -> list[dict[str, Any]]:
        operations: list[dict[str, Any]] = []
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError as exc:
            raise LocalValidationError("Model pack state could not be read") from exc
        for path in paths:
            try:
                operations.append(_operation_payload(_read_regular_json(path)))
            except FileNotFoundError:
                continue
        return operations

    def _path(self, operation_id: str) -> Path:
        return self.directory / f"{_uuid_text(operation_id)}.json"
