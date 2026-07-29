"""Serialized lifecycle manager for fresh, isolated local model workers."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import inspect
import logging
import os
import resource
import shlex
import shutil
import signal
import stat
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.services.local_ai.errors import (
    LocalInputLimitError,
    LocalWorkerError,
    LocalWorkerTimeout,
)
from app.services.local_ai.protocol import (
    MAX_MESSAGE_BYTES,
    ErrorPayload,
    ProgressPayload,
    ProtocolViolation,
    ResultPayload,
    WorkerRequest,
    encode_message,
    parse_response_line,
    validate_identifier,
)
from app.services.local_ai.types import ModelRole

ProgressCallback = Callable[[dict[str, object]], object | Awaitable[object]]
LivenessCallback = Callable[[], object | Awaitable[object]]

_GLOBAL_ROLE_PROCESS_LOCK: asyncio.Lock | None = None
_GLOBAL_ROLE_PROCESS_LOOP: asyncio.AbstractEventLoop | None = None
_GLOBAL_PROCESS_BOUNDARY_POISONED = False
_PROCESS_SHUTDOWN_SECONDS = 5.0
_MAX_PENDING_CANCELLATIONS = 1_024
# One cancellation-resistant callback never blocks a later job. A sustained
# application callback failure is capped and then fails fast before spawning.
_MAX_DETACHED_CALLBACKS = 32
_INTERPROCESS_LOCK_POLL_SECONDS = 0.05
_MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_MACOS_NETWORK_DENY_PROFILE = "(version 1) (allow default) (deny network*)"
_WORKER_PROCESS_LOCK = "worker-process.lock"
_PROGRESS_STAGE_ORDER = {
    "starting": 0,
    "loading": 1,
    "processing": 2,
    "finalizing": 3,
    "cancelling": 4,
}
logger = logging.getLogger(__name__)


@dataclass
class LocalModelMetrics:
    """Non-content lifecycle metrics for tests and operational health."""

    live_processes: int = 0
    max_live_processes: int = 0
    roles_started: list[ModelRole] = field(default_factory=list)
    pids_started: list[int] = field(default_factory=list)
    completed_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0


@dataclass
class _RunState:
    job_id: str
    running_event: asyncio.Event = field(default_factory=asyncio.Event)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: bool = False


class _ProgressCallbackFailure(RuntimeError):
    pass


class _RunCancelled(RuntimeError):
    def __init__(self, *, stopped: bool) -> None:
        super().__init__()
        self.stopped = stopped


@dataclass
class _WorkerProcessLease:
    """Local and cross-process ownership held through worker-group cleanup."""

    local_lock: asyncio.Lock
    descriptor: int

    def release(self) -> None:
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        self.local_lock.release()


def _global_role_process_lock() -> asyncio.Lock:
    """Return the process-global lock, refreshing it between closed test loops."""
    global _GLOBAL_ROLE_PROCESS_LOCK, _GLOBAL_ROLE_PROCESS_LOOP

    loop = asyncio.get_running_loop()
    if _GLOBAL_ROLE_PROCESS_LOCK is None:
        _GLOBAL_ROLE_PROCESS_LOCK = asyncio.Lock()
        _GLOBAL_ROLE_PROCESS_LOOP = loop
    elif _GLOBAL_ROLE_PROCESS_LOOP is not loop:
        if _GLOBAL_ROLE_PROCESS_LOCK.locked():
            raise LocalWorkerError("Local worker is unavailable.")
        _GLOBAL_ROLE_PROCESS_LOCK = asyncio.Lock()
        _GLOBAL_ROLE_PROCESS_LOOP = loop
    return _GLOBAL_ROLE_PROCESS_LOCK


def _disable_core_dumps() -> None:
    """Disable child core dumps before the worker executable starts."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        os._exit(126)


class LocalModelManager:
    """Own exactly one fresh local model worker process at a time."""

    def __init__(
        self,
        worker_command: str | Sequence[str] | None = None,
        *,
        worker_home: str | Path | None = None,
        timeout_seconds: float | None = None,
        hard_timeout_seconds: float | None = None,
        _allow_unisolated_test_worker: bool = False,
    ) -> None:
        if _allow_unisolated_test_worker and worker_command is None:
            raise ValueError("An unisolated test worker command is required.")
        self._worker_command_spec = worker_command
        self._worker_command: tuple[str, ...] | None = None
        self._enforce_network_sandbox = not _allow_unisolated_test_worker
        self._network_sandbox_prefix: tuple[str, ...] = ()
        self._worker_home = (
            Path(worker_home) if worker_home else self._default_worker_home()
        )
        self._worker_process_lock_path: Path | None = None
        self._worker_process_lock_fd: int | None = None
        self._timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else self._default_timeout_seconds()
        )
        self._hard_timeout_seconds = (
            float(hard_timeout_seconds)
            if hard_timeout_seconds is not None
            else self._default_hard_timeout_seconds()
        )
        if (
            self._timeout_seconds <= 0
            or self._hard_timeout_seconds < self._timeout_seconds
        ):
            raise ValueError("Local worker timeout configuration is invalid.")
        self._started = False
        self._stopping = False
        self._stopped = False
        self._active_process: asyncio.subprocess.Process | None = None
        self._active_job_id: str | None = None
        self._active_role: ModelRole | None = None
        self._jobs: dict[str, _RunState] = {}
        self._cancelled_jobs: OrderedDict[str, None] = OrderedDict()
        self._running_events: dict[str, asyncio.Event] = {}
        self._done_events: dict[str, asyncio.Event] = {}
        self._detached_callbacks: set[asyncio.Future[object]] = set()
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_poisoned = False
        self._retained_worker_process_lease: _WorkerProcessLease | None = None
        self._cleanup_recovery_task: asyncio.Task[None] | None = None
        self.metrics = LocalModelMetrics()

    @staticmethod
    def _default_worker_home() -> Path:
        from app.config import settings

        return Path(settings.local_ai_scratch_dir) / "worker-home"

    @staticmethod
    def _default_timeout_seconds() -> float:
        from app.config import settings

        return float(settings.local_ai_worker_timeout_seconds)

    @staticmethod
    def _default_hard_timeout_seconds() -> float:
        from app.config import settings

        return float(settings.local_ai_worker_hard_timeout_seconds)

    @property
    def active_pid(self) -> int | None:
        """The active PID, cleared only after its process group is gone."""
        return self._active_process.pid if self._active_process is not None else None

    async def start(self) -> None:
        """Validate owner-only state and command without spawning a worker."""
        if self._stopped:
            raise LocalWorkerError("Local worker manager is stopped.")
        if self._started:
            return
        self._prepare_worker_home()
        self._worker_command = self._resolve_command()
        self._prepare_worker_process_lock()
        self._network_sandbox_prefix = self._resolve_network_sandbox()
        self._started = True

    def _resolve_command(self) -> tuple[str, ...]:
        worker_command = self._worker_command_spec
        if worker_command is None:
            from app.config import settings

            worker_command = settings.local_ai_worker_command
        if isinstance(worker_command, str):
            parts = tuple(shlex.split(worker_command))
        else:
            try:
                parts = tuple(os.fspath(item) for item in worker_command)
            except TypeError:
                parts = ()
        if not parts or any(not part for part in parts):
            raise LocalWorkerError("Local worker is unavailable.")

        executable = Path(parts[0]).expanduser()
        if not executable.is_absolute():
            if executable.parent != Path("."):
                executable = (Path.cwd() / executable).absolute()
            else:
                located = shutil.which(parts[0], path=os.defpath)
                if located is None:
                    raise LocalWorkerError("Local worker is unavailable.")
                executable = Path(located).absolute()
        else:
            executable = executable.absolute()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise LocalWorkerError("Local worker is unavailable.")
        return (str(executable), *parts[1:])

    def _resolve_network_sandbox(self) -> tuple[str, ...]:
        if not self._enforce_network_sandbox:
            return ()
        if sys.platform != "darwin":
            raise LocalWorkerError("Local worker network isolation is unavailable.")
        try:
            metadata = _MACOS_SANDBOX_EXEC.stat()
        except OSError:
            raise LocalWorkerError(
                "Local worker network isolation is unavailable."
            ) from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
            or not os.access(_MACOS_SANDBOX_EXEC, os.X_OK)
        ):
            raise LocalWorkerError("Local worker network isolation is unavailable.")
        return (
            str(_MACOS_SANDBOX_EXEC),
            "-p",
            _MACOS_NETWORK_DENY_PROFILE,
        )

    def _process_lock_root(self) -> Path:
        if not self._enforce_network_sandbox:
            return self._worker_home.parent / ".local-ai-runtime"
        from app.config import settings

        return Path(settings.local_ai_model_dir).resolve() / ".runtime"

    def _prepare_worker_process_lock(self) -> None:
        root = self._process_lock_root()
        if root.is_symlink():
            raise LocalWorkerError("Local worker is unavailable.")
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if root.is_symlink() or not root.is_dir():
                raise LocalWorkerError("Local worker is unavailable.")
            os.chmod(root, 0o700)
            root_metadata = root.stat()
        except LocalWorkerError:
            raise
        except OSError:
            raise LocalWorkerError("Local worker is unavailable.") from None
        if stat.S_IMODE(root_metadata.st_mode) != 0o700 or (
            hasattr(os, "getuid") and root_metadata.st_uid != os.getuid()
        ):
            raise LocalWorkerError("Local worker is unavailable.")

        lock_path = root / _WORKER_PROCESS_LOCK
        descriptor = -1
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
            ):
                raise LocalWorkerError("Local worker is unavailable.")
            os.fchmod(descriptor, 0o600)
        except LocalWorkerError:
            raise
        except OSError:
            raise LocalWorkerError("Local worker is unavailable.") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self._worker_process_lock_path = lock_path

    def _prepare_worker_home(self) -> None:
        if self._worker_home.is_symlink():
            raise LocalWorkerError("Local worker is unavailable.")
        self._worker_home.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._worker_home.is_symlink() or not self._worker_home.is_dir():
            raise LocalWorkerError("Local worker is unavailable.")
        try:
            os.chmod(self._worker_home, 0o700)
            metadata = self._worker_home.stat()
        except OSError:
            raise LocalWorkerError("Local worker is unavailable.") from None
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise LocalWorkerError("Local worker is unavailable.")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise LocalWorkerError("Local worker is unavailable.")

    async def stop(self) -> None:
        """Prevent work, signal all callers, and reap the active process group."""
        if self._stopped:
            return
        self._stopping = True
        self._started = False
        states = list(self._jobs.values())
        for state in states:
            state.stop_requested = True
            state.cancel_event.set()
        active_process = self._active_process
        if active_process is not None:
            await self._terminate_process(active_process)
            if self._cleanup_poisoned:
                await self._finish_cleanup_recovery(active_process)
        if states:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(state.done_event.wait() for state in states)),
                    timeout=_PROCESS_SHUTDOWN_SECONDS,
                )
            except TimeoutError:
                pass
        self._stopped = True
        self._stopping = False

    async def run(
        self,
        role: ModelRole,
        payload: dict[str, Any],
        on_progress: ProgressCallback | None = None,
        *,
        on_liveness: LivenessCallback | None = None,
    ) -> Any:
        """Run one role in a fresh process and return only after group cleanup."""
        role = ModelRole(role)
        job_id = self._job_id(payload)
        state = self._register_job(job_id)
        try:
            if _GLOBAL_PROCESS_BOUNDARY_POISONED:
                raise LocalWorkerError("Local worker is unavailable.")
            if self._is_unavailable():
                raise LocalWorkerError("Local worker manager is stopped.")
            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.pop(job_id)
                raise _RunCancelled(stopped=False)
            return await self._run_registered(
                state,
                role,
                payload,
                on_progress,
                on_liveness,
            )
        except _RunCancelled as exc:
            self.metrics.cancelled_runs += 1
            message = (
                "Local worker manager is stopped."
                if exc.stopped
                else "Local worker cancelled."
            )
            raise LocalWorkerError(message) from None
        except asyncio.CancelledError:
            state.cancel_event.set()
            self.metrics.cancelled_runs += 1
            raise
        except LocalWorkerTimeout:
            self.metrics.failed_runs += 1
            raise
        except LocalWorkerError:
            self.metrics.failed_runs += 1
            raise
        finally:
            state.done_event.set()
            self._jobs.pop(job_id, None)
            self._running_events.pop(job_id, None)
            self._done_events.pop(job_id, None)

    def _register_job(self, job_id: str) -> _RunState:
        if job_id in self._jobs:
            raise LocalWorkerError("Local worker job is already registered.")
        state = _RunState(job_id=job_id)
        self._jobs[job_id] = state
        self._running_events[job_id] = state.running_event
        self._done_events[job_id] = state.done_event
        return state

    async def _run_registered(
        self,
        state: _RunState,
        role: ModelRole,
        payload: dict[str, Any],
        on_progress: ProgressCallback | None,
        on_liveness: LivenessCallback | None,
    ) -> Any:
        lease = await self._acquire_worker_process_lease(state)
        process: asyncio.subprocess.Process | None = None
        result: Any = None
        try:
            self._prune_detached_callbacks()
            if len(self._detached_callbacks) >= _MAX_DETACHED_CALLBACKS:
                raise LocalWorkerError("Local worker is unavailable.")
            self._raise_if_cancelled(state)
            request = self._build_request(role, payload, state.job_id)
            self._active_job_id = state.job_id
            self._active_role = role
            process = await self._spawn_worker_cancellation_safe()
            self._record_process_start(process)
            state.running_event.set()
            self._raise_if_cancelled(state)

            try:
                result = await self._exchange(
                    process,
                    request,
                    role,
                    on_progress,
                    on_liveness,
                    state,
                )
            except _ProgressCallbackFailure:
                raise LocalWorkerError(
                    "Local worker progress callback failed."
                ) from None
            except ProtocolViolation:
                raise LocalWorkerError("Local worker protocol failed.") from None
            except (BrokenPipeError, ConnectionError, OSError):
                self._raise_if_cancelled(state)
                raise LocalWorkerError("Local worker failed.") from None
            except LocalWorkerError:
                self._raise_if_cancelled(state)
                raise
            self._raise_if_cancelled(state)
        finally:
            cleanup_process = process
            if cleanup_process is None and self._cleanup_poisoned:
                cleanup_process = self._active_process
            cleanup_task = asyncio.create_task(
                self._cleanup_run_process(cleanup_process)
            )
            cleanup_error: LocalWorkerError | None = None
            try:
                caller_cancelled = await self._wait_for_task_final(cleanup_task)
                try:
                    cleanup_task.result()
                except asyncio.CancelledError:
                    cleanup_error = LocalWorkerError("Local worker failed.")
                except LocalWorkerError as exc:
                    cleanup_error = exc
                except Exception:
                    cleanup_error = LocalWorkerError("Local worker failed.")
            finally:
                if cleanup_error is None:
                    self._clear_active_process_state(cleanup_process)
                    self._clear_process_boundary_poison()
                    self._worker_process_lock_fd = None
                    lease.release()
                else:
                    self._retain_failed_cleanup(cleanup_process, lease)
            if cleanup_error is not None:
                raise cleanup_error from None
            if caller_cancelled:
                raise asyncio.CancelledError
        self.metrics.completed_runs += 1
        return result

    async def _cleanup_run_process(
        self,
        process: asyncio.subprocess.Process | None,
    ) -> None:
        if process is not None:
            await self._terminate_process(process)

    @staticmethod
    async def _wait_for_task_final(task: asyncio.Future[object]) -> bool:
        """Wait through repeated caller cancellations and consume task state later."""
        caller_cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                caller_cancelled = True
            except Exception:
                break
        return caller_cancelled

    def _clear_active_process_state(
        self,
        process: asyncio.subprocess.Process | None,
    ) -> None:
        if process is None:
            if self._active_process is None:
                self._active_job_id = None
                self._active_role = None
            return
        if self._active_process is not process:
            return
        self.metrics.live_processes = max(0, self.metrics.live_processes - 1)
        self._active_process = None
        self._active_job_id = None
        self._active_role = None

    def _record_process_start(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        if self._active_process is process:
            return
        self._active_process = process
        self.metrics.live_processes += 1
        self.metrics.max_live_processes = max(
            self.metrics.max_live_processes,
            self.metrics.live_processes,
        )
        if self._active_role is not None:
            self.metrics.roles_started.append(self._active_role)
        self.metrics.pids_started.append(process.pid)

    def _poison_process_boundary(self) -> None:
        global _GLOBAL_PROCESS_BOUNDARY_POISONED

        self._cleanup_poisoned = True
        _GLOBAL_PROCESS_BOUNDARY_POISONED = True

    def _clear_process_boundary_poison(self) -> None:
        global _GLOBAL_PROCESS_BOUNDARY_POISONED

        if not self._cleanup_poisoned:
            return
        self._cleanup_poisoned = False
        _GLOBAL_PROCESS_BOUNDARY_POISONED = False

    def _retain_failed_cleanup(
        self,
        process: asyncio.subprocess.Process | None,
        lease: _WorkerProcessLease,
    ) -> None:
        """Keep exclusivity until a failed-cleanup process group is confirmed gone."""
        if process is None or not self._process_group_exists(process.pid):
            self._clear_active_process_state(process)
            self._clear_process_boundary_poison()
            self._worker_process_lock_fd = None
            lease.release()
            return
        self._poison_process_boundary()
        self._retained_worker_process_lease = lease
        recovery_task = asyncio.create_task(
            self._recover_failed_cleanup(process),
        )
        self._cleanup_recovery_task = recovery_task

    async def _recover_failed_cleanup(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Release retained ownership only after the orphan group exits."""
        while self._process_group_exists(process.pid):
            await asyncio.sleep(_INTERPROCESS_LOCK_POLL_SECONDS)
        await self._finish_cleanup_recovery(process)

    async def _finish_cleanup_recovery(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Clear a poisoned boundary and release its retained lease once."""
        recovery_task = self._cleanup_recovery_task
        if (
            recovery_task is not None
            and recovery_task is not asyncio.current_task()
            and not recovery_task.done()
        ):
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)
        self._cleanup_recovery_task = None
        lease = self._retained_worker_process_lease
        if lease is None:
            return
        self._retained_worker_process_lease = None
        self._clear_active_process_state(process)
        self._clear_process_boundary_poison()
        self._worker_process_lock_fd = None
        lease.release()

    async def _acquire_role_process_lock(self, state: _RunState) -> asyncio.Lock:
        if _GLOBAL_PROCESS_BOUNDARY_POISONED:
            raise LocalWorkerError("Local worker is unavailable.")
        lock = _global_role_process_lock()
        acquire_task = asyncio.create_task(lock.acquire())
        cancel_task = asyncio.create_task(state.cancel_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {acquire_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                if acquire_task.done() and acquire_task.result():
                    lock.release()
                else:
                    acquire_task.cancel()
                    await asyncio.gather(acquire_task, return_exceptions=True)
                raise _RunCancelled(stopped=state.stop_requested)
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            if _GLOBAL_PROCESS_BOUNDARY_POISONED:
                lock.release()
                raise LocalWorkerError("Local worker is unavailable.")
            self._raise_if_cancelled(state)
            return lock
        except asyncio.CancelledError:
            acquire_task.cancel()
            cancel_task.cancel()
            await asyncio.gather(
                acquire_task,
                cancel_task,
                return_exceptions=True,
            )
            if lock.locked() and acquire_task.done() and not acquire_task.cancelled():
                lock.release()
            raise

    async def _acquire_worker_process_lease(
        self,
        state: _RunState,
    ) -> _WorkerProcessLease:
        local_lock = await self._acquire_role_process_lock(state)
        descriptor = -1
        try:
            descriptor = await self._acquire_interprocess_lock(state)
            self._worker_process_lock_fd = descriptor
            return _WorkerProcessLease(
                local_lock=local_lock,
                descriptor=descriptor,
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            local_lock.release()
            raise

    async def _acquire_interprocess_lock(self, state: _RunState) -> int:
        lock_path = self._worker_process_lock_path
        if lock_path is None:
            raise LocalWorkerError("Local worker is unavailable.")
        descriptor = -1
        acquired = False
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
            ):
                raise LocalWorkerError("Local worker is unavailable.")
            while True:
                self._raise_if_cancelled(state)
                try:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    acquired = True
                    return descriptor
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise LocalWorkerError("Local worker is unavailable.") from None
                try:
                    await asyncio.wait_for(
                        state.cancel_event.wait(),
                        timeout=_INTERPROCESS_LOCK_POLL_SECONDS,
                    )
                except TimeoutError:
                    continue
                raise _RunCancelled(stopped=state.stop_requested)
        except (LocalWorkerError, _RunCancelled):
            raise
        except OSError:
            raise LocalWorkerError("Local worker is unavailable.") from None
        finally:
            if descriptor >= 0 and not acquired:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    @staticmethod
    def _raise_if_cancelled(state: _RunState) -> None:
        if state.cancel_event.is_set():
            raise _RunCancelled(stopped=state.stop_requested)

    def _is_unavailable(self) -> bool:
        return (
            not self._started
            or self._stopping
            or self._stopped
            or _GLOBAL_PROCESS_BOUNDARY_POISONED
        )

    @staticmethod
    def _job_id(payload: dict[str, Any]) -> str:
        value = payload.get("job_id")
        if value is None:
            return uuid.uuid4().hex
        try:
            return validate_identifier(value)
        except ProtocolViolation:
            raise LocalWorkerError("Local worker request was rejected.") from None

    @staticmethod
    def _command_for(role: ModelRole) -> str:
        if role is ModelRole.EXTRACTION:
            return "extract"
        if role is ModelRole.SUMMARY:
            return "summarize"
        return "ocr"

    def _build_request(
        self,
        role: ModelRole,
        payload: dict[str, Any],
        job_id: str,
    ) -> WorkerRequest:
        try:
            return WorkerRequest(
                version=1,
                request_id=uuid.uuid4().hex,
                job_id=job_id,
                command=self._command_for(role),  # type: ignore[arg-type]
                payload=payload,
            )
        except ValidationError:
            raise LocalWorkerError("Local worker request was rejected.") from None

    def _child_environment(self) -> dict[str, str]:
        environment = {
            "PATH": os.defpath,
            "HOME": str(self._worker_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if self._enforce_network_sandbox:
            lock_fd = self._worker_process_lock_fd
            lock_path = self._worker_process_lock_path
            if lock_fd is None or lock_path is None:
                raise LocalWorkerError("Local worker is unavailable.")
            try:
                descriptor_metadata = os.fstat(lock_fd)
                path_metadata = os.stat(lock_path, follow_symlinks=False)
            except OSError:
                raise LocalWorkerError("Local worker is unavailable.") from None
            if (
                not stat.S_ISREG(descriptor_metadata.st_mode)
                or stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
                or descriptor_metadata.st_nlink != 1
                or (hasattr(os, "getuid") and descriptor_metadata.st_uid != os.getuid())
                or not stat.S_ISREG(path_metadata.st_mode)
                or descriptor_metadata.st_dev != path_metadata.st_dev
                or descriptor_metadata.st_ino != path_metadata.st_ino
            ):
                raise LocalWorkerError("Local worker is unavailable.")
            environment["LOCAL_AI_PARENT_PID"] = str(os.getpid())
            environment["LOCAL_AI_PROCESS_LOCK_FD"] = str(lock_fd)
            environment["LOCAL_AI_PROCESS_LOCK_DEVICE"] = str(
                descriptor_metadata.st_dev
            )
            environment["LOCAL_AI_PROCESS_LOCK_INODE"] = str(descriptor_metadata.st_ino)
        return environment

    async def _spawn_worker(self) -> asyncio.subprocess.Process:
        if self._worker_command is None:
            raise LocalWorkerError("Local worker is unavailable.")
        lock_fd = self._worker_process_lock_fd
        if lock_fd is None:
            raise LocalWorkerError("Local worker is unavailable.")
        command = (*self._network_sandbox_prefix, *self._worker_command)
        try:
            return await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._child_environment(),
                cwd=self._worker_home,
                start_new_session=True,
                preexec_fn=_disable_core_dumps,
                pass_fds=(lock_fd,),
                limit=MAX_MESSAGE_BYTES + 1,
            )
        except (OSError, ValueError):
            raise LocalWorkerError("Local worker is unavailable.") from None

    async def _spawn_worker_cancellation_safe(self) -> asyncio.subprocess.Process:
        spawn_task = asyncio.create_task(self._spawn_worker())
        try:
            return await asyncio.shield(spawn_task)
        except asyncio.CancelledError:
            process: asyncio.subprocess.Process | None = None
            await self._wait_for_task_final(spawn_task)
            try:
                process = spawn_task.result()
            except (asyncio.CancelledError, Exception):
                pass
            if process is not None:
                self._record_process_start(process)
                cleanup_task = asyncio.create_task(self._terminate_process(process))
                await self._wait_for_task_final(cleanup_task)
                try:
                    cleanup_task.result()
                except (asyncio.CancelledError, Exception):
                    self._poison_process_boundary()
                    raise LocalWorkerError("Local worker failed.") from None
                self._clear_active_process_state(process)
            raise

    async def _exchange(
        self,
        process: asyncio.subprocess.Process,
        request: WorkerRequest,
        role: ModelRole,
        on_progress: ProgressCallback | None,
        on_liveness: LivenessCallback | None,
        state: _RunState,
    ) -> Any:
        if process.stdin is None or process.stdout is None:
            raise LocalWorkerError("Local worker failed.")
        started_at = time.monotonic()
        idle_deadline = started_at + self._timeout_seconds
        hard_deadline = started_at + self._hard_timeout_seconds
        process.stdin.write(encode_message(request))
        remaining = min(
            idle_deadline - time.monotonic(),
            hard_deadline - time.monotonic(),
        )
        if remaining <= 0:
            raise LocalWorkerTimeout("Local worker timed out.")
        try:
            await asyncio.wait_for(process.stdin.drain(), timeout=remaining)
        except TimeoutError:
            raise LocalWorkerTimeout("Local worker timed out.") from None

        ready_seen = False
        terminal: ResultPayload | ErrorPayload | None = None
        last_progress: ProgressPayload | None = None
        while terminal is None:
            self._raise_if_cancelled(state)
            now = time.monotonic()
            remaining = min(idle_deadline - now, hard_deadline - now)
            if remaining <= 0:
                raise LocalWorkerTimeout("Local worker timed out.")
            try:
                line = await asyncio.wait_for(
                    self._readline(process),
                    timeout=remaining,
                )
            except TimeoutError:
                raise LocalWorkerTimeout("Local worker timed out.") from None
            if not line:
                self._raise_if_cancelled(state)
                raise LocalWorkerError("Local worker failed.")
            response = parse_response_line(line)
            if response.request_id != request.request_id:
                raise ProtocolViolation("Worker response request ID does not match.")
            if response.kind == "ready":
                if ready_seen or response.payload.role is not role:
                    raise ProtocolViolation("Worker readiness response is invalid.")
                ready_seen = True
                idle_deadline = time.monotonic() + self._timeout_seconds
                continue
            if not ready_seen:
                raise ProtocolViolation("Worker responded before readiness.")
            if response.kind == "progress":
                progress = response.payload
                if (
                    not isinstance(progress, ProgressPayload)
                    or progress.role is not role
                ):
                    raise ProtocolViolation("Worker progress response is invalid.")
                liveness_advanced = self._validate_progress_sequence(
                    last_progress,
                    progress,
                )
                visible_advance = self._progress_is_visible_advance(
                    last_progress,
                    progress,
                )
                if liveness_advanced:
                    # The worker has already proved it is live. Give any
                    # persistence callback a fresh idle window while the hard
                    # deadline remains absolute.
                    idle_deadline = time.monotonic() + self._timeout_seconds
                callback_deadline = min(idle_deadline, hard_deadline)
                if liveness_advanced and not visible_advance:
                    await self._forward_liveness(
                        on_liveness,
                        state,
                        deadline=callback_deadline,
                    )
                if visible_advance:
                    await self._forward_progress(
                        progress,
                        on_progress,
                        state,
                        deadline=callback_deadline,
                    )
                last_progress = progress
                continue
            if response.kind in {"result", "error"}:
                if not isinstance(response.payload, (ResultPayload, ErrorPayload)):
                    raise ProtocolViolation("Worker terminal response is invalid.")
                terminal = response.payload
                break
            raise ProtocolViolation("Worker response kind is invalid.")

        await self._request_shutdown(process, request)
        await self._drain_and_reap_after_terminal(process, request.request_id)
        self._raise_if_cancelled(state)
        if isinstance(terminal, ErrorPayload):
            logger.warning(
                "Local worker returned safe error code %s for role %s",
                terminal.code,
                role.value,
            )
            if terminal.code == "input_limit_exceeded":
                raise LocalInputLimitError(terminal.message)
            raise LocalWorkerError(terminal.message)
        return terminal.data

    @staticmethod
    def _validate_progress_sequence(
        previous: ProgressPayload | None,
        current: ProgressPayload,
    ) -> bool:
        """Reject regressing counters and identify deadline-extending progress."""

        if previous is None:
            return True
        previous_activity = previous.activity or 0
        current_activity = current.activity or 0
        if current_activity < previous_activity:
            raise ProtocolViolation("Worker progress response is invalid.")
        activity_advanced = current_activity > previous_activity
        previous_order = _PROGRESS_STAGE_ORDER[previous.stage]
        current_order = _PROGRESS_STAGE_ORDER[current.stage]
        if current_order < previous_order:
            raise ProtocolViolation("Worker progress response is invalid.")
        if current.stage != previous.stage:
            return True
        if current.total != previous.total or current.current < previous.current:
            raise ProtocolViolation("Worker progress response is invalid.")
        return current.current > previous.current or activity_advanced

    @staticmethod
    def _progress_is_visible_advance(
        previous: ProgressPayload | None,
        current: ProgressPayload,
    ) -> bool:
        """Return whether a stage or completed-page change should reach callers."""

        return (
            previous is None
            or current.stage != previous.stage
            or current.current > previous.current
        )

    async def _forward_progress(
        self,
        progress: ProgressPayload,
        on_progress: ProgressCallback | None,
        state: _RunState,
        *,
        deadline: float,
    ) -> None:
        if on_progress is None:
            return
        self._prune_detached_callbacks()
        safe_progress = progress.model_dump(mode="json", exclude_none=True)
        try:
            callback_result = on_progress(safe_progress)
        except Exception:
            raise _ProgressCallbackFailure from None
        await self._await_callback(
            callback_result,
            state,
            deadline=deadline,
        )

    async def _forward_liveness(
        self,
        on_liveness: LivenessCallback | None,
        state: _RunState,
        *,
        deadline: float,
    ) -> None:
        """Persist an activity-only lease without manufacturing UI progress."""

        if on_liveness is None:
            return
        self._prune_detached_callbacks()
        try:
            callback_result = on_liveness()
        except Exception:
            raise _ProgressCallbackFailure from None
        await self._await_callback(
            callback_result,
            state,
            deadline=deadline,
        )

    async def _await_callback(
        self,
        callback_result: object | Awaitable[object],
        state: _RunState,
        *,
        deadline: float,
    ) -> None:
        """Await one callback without coupling it to any later job."""

        if not inspect.isawaitable(callback_result):
            return

        callback_task = asyncio.ensure_future(callback_result)
        cancel_task = asyncio.create_task(state.cancel_event.wait())
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._detach_callback(callback_task)
                raise LocalWorkerTimeout("Local worker timed out.")
            done, _pending = await asyncio.wait(
                {callback_task, cancel_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done or time.monotonic() >= deadline:
                self._detach_callback(callback_task)
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)
                raise LocalWorkerTimeout("Local worker timed out.")
            if cancel_task in done:
                self._detach_callback(callback_task)
                raise _RunCancelled(stopped=state.stop_requested)
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            if callback_task.cancelled():
                raise _ProgressCallbackFailure
            callback_task.result()
        except asyncio.CancelledError:
            self._detach_callback(callback_task)
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            raise
        except _RunCancelled:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            raise
        except LocalWorkerTimeout:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            raise
        except Exception:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            raise _ProgressCallbackFailure from None

    def _detach_callback(self, task: asyncio.Future[object]) -> None:
        if task.done():
            self._consume_callback_result(task)
            return
        task.cancel()
        if task.done():
            self._consume_callback_result(task)
            return
        self._detached_callbacks.add(task)
        task.add_done_callback(self._consume_detached_callback)

    def _prune_detached_callbacks(self) -> None:
        for task in tuple(self._detached_callbacks):
            if task.done():
                self._consume_detached_callback(task)

    def _consume_detached_callback(self, task: asyncio.Future[object]) -> None:
        self._detached_callbacks.discard(task)
        self._consume_callback_result(task)

    @staticmethod
    def _consume_callback_result(task: asyncio.Future[object]) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    async def _readline(process: asyncio.subprocess.Process) -> bytes:
        if process.stdout is None:
            raise LocalWorkerError("Local worker failed.")
        try:
            return await process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError):
            raise ProtocolViolation(
                "Worker protocol message exceeds the size limit."
            ) from None

    async def _request_shutdown(
        self,
        process: asyncio.subprocess.Process,
        request: WorkerRequest,
    ) -> None:
        if process.stdin is None or process.stdin.is_closing():
            raise LocalWorkerError("Local worker failed.")
        shutdown = WorkerRequest(
            version=1,
            request_id=request.request_id,
            job_id=request.job_id,
            command="shutdown",
            payload={},
        )
        process.stdin.write(encode_message(shutdown))
        await process.stdin.drain()
        process.stdin.close()
        try:
            await process.stdin.wait_closed()
        except (BrokenPipeError, ConnectionError):
            pass

    async def _drain_and_reap_after_terminal(
        self,
        process: asyncio.subprocess.Process,
        request_id: str,
    ) -> None:
        async def drain() -> None:
            while True:
                line = await self._readline(process)
                if not line:
                    break
                response = parse_response_line(line)
                if response.request_id != request_id:
                    raise ProtocolViolation(
                        "Worker response request ID does not match."
                    )
                raise ProtocolViolation("Worker responded after a terminal response.")
            return_code = await process.wait()
            if return_code != 0:
                raise LocalWorkerError("Local worker failed.")

        try:
            await asyncio.wait_for(drain(), timeout=_PROCESS_SHUTDOWN_SECONDS)
        except TimeoutError:
            raise LocalWorkerError("Local worker failed.") from None

    async def cancel(self, job_id: str) -> None:
        """Cancel a registered job or reserve a bounded pre-start cancellation."""
        try:
            job_id = validate_identifier(job_id)
        except ProtocolViolation:
            raise LocalWorkerError("Local worker request was rejected.") from None
        if await self._cancel_registered(job_id):
            return
        self._reserve_cancellation(job_id)

    async def cancel_registered(self, job_id: str) -> bool:
        """Cancel only an existing registration, never reserving an unknown ID."""
        try:
            job_id = validate_identifier(job_id)
        except ProtocolViolation:
            raise LocalWorkerError("Local worker request was rejected.") from None
        return await self._cancel_registered(job_id)

    async def _cancel_registered(self, job_id: str) -> bool:
        state = self._jobs.get(job_id)
        if state is None:
            return False
        state.cancel_event.set()
        process = self._active_process if self._active_job_id == job_id else None
        if process is not None:
            await self._terminate_process(process)
        if not state.done_event.is_set():
            try:
                await asyncio.wait_for(
                    state.done_event.wait(),
                    timeout=_PROCESS_SHUTDOWN_SECONDS,
                )
            except TimeoutError:
                pass
        return True

    def _reserve_cancellation(self, job_id: str) -> None:
        if job_id in self._cancelled_jobs:
            return
        if len(self._cancelled_jobs) >= _MAX_PENDING_CANCELLATIONS:
            raise LocalWorkerError("Local worker cancellation capacity exceeded.")
        self._cancelled_jobs[job_id] = None

    async def wait_until_running(self, job_id: str, timeout: float = 5.0) -> None:
        """Wait for a registered job without allocating state for unknown IDs."""
        try:
            job_id = validate_identifier(job_id)
        except ProtocolViolation:
            raise LocalWorkerError("Local worker request was rejected.") from None
        deadline = asyncio.get_running_loop().time() + timeout
        state = self._jobs.get(job_id)
        while state is None and asyncio.get_running_loop().time() < deadline:
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.sleep(min(0.01, max(0.0, remaining)))
            state = self._jobs.get(job_id)
        if state is None:
            raise LocalWorkerError("Local worker job is not registered.")
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        try:
            await asyncio.wait_for(state.running_event.wait(), remaining)
        except TimeoutError:
            raise LocalWorkerError("Local worker job is not running.") from None

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        async with self._cleanup_lock:
            pgid = process.pid
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
            self._signal_process_group(pgid, signal.SIGTERM)
            deadline = time.monotonic() + _PROCESS_SHUTDOWN_SECONDS
            await self._wait_for_parent(process, deadline)
            if await self._wait_for_group_exit(pgid, deadline):
                return
            self._signal_process_group(pgid, signal.SIGKILL)
            kill_deadline = time.monotonic() + _PROCESS_SHUTDOWN_SECONDS
            await self._wait_for_parent(process, kill_deadline)
            if not await self._wait_for_group_exit(pgid, kill_deadline):
                raise LocalWorkerError("Local worker failed.")

    @staticmethod
    def _signal_process_group(pgid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass

    @staticmethod
    async def _wait_for_parent(
        process: asyncio.subprocess.Process,
        deadline: float,
    ) -> None:
        if process.returncode is not None:
            await process.wait()
            return
        remaining = max(0.0, deadline - time.monotonic())
        try:
            await asyncio.wait_for(process.wait(), remaining)
        except TimeoutError:
            pass

    @staticmethod
    async def _wait_for_group_exit(pgid: int, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if not LocalModelManager._process_group_exists(pgid):
                return True
            await asyncio.sleep(0.01)
        return not LocalModelManager._process_group_exists(pgid)

    @staticmethod
    def _process_group_exists(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True


def create_local_model_manager() -> LocalModelManager:
    """Build the application singleton without command validation or spawning."""
    return LocalModelManager()


local_model_manager = create_local_model_manager()
