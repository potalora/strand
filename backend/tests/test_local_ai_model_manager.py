from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI

from app.services.local_ai.errors import LocalWorkerError, LocalWorkerTimeout
from app.services.local_ai.model_manager import LocalModelManager
from app.services.local_ai.types import ModelRole


@pytest.fixture
def fake_worker_command() -> list[str]:
    backend_root = Path(__file__).resolve().parents[1]
    bootstrap = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(backend_root)!r});"
        "runpy.run_module('app.services.local_ai.fake_worker',run_name='__main__')"
    )
    return [sys.executable, "-c", bootstrap]


@pytest.fixture
def worker_home(tmp_path: Path) -> Path:
    return tmp_path / "worker-home"


@pytest.mark.asyncio
async def test_manager_never_overlaps_role_processes(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    first = asyncio.create_task(manager.run(ModelRole.OCR, {"delay_ms": 50}))
    await asyncio.sleep(0)
    second = asyncio.create_task(manager.run(ModelRole.EXTRACTION, {"delay_ms": 10}))

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result["markdown"] == "# Synthetic OCR"
    assert second_result["entities"] == []
    assert manager.metrics.max_live_processes == 1
    assert manager.metrics.roles_started == [ModelRole.OCR, ModelRole.EXTRACTION]
    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_each_run_uses_a_fresh_reaped_process(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    await manager.run(ModelRole.OCR, {})
    first_pid = manager.metrics.pids_started[-1]
    await manager.run(ModelRole.OCR, {})
    second_pid = manager.metrics.pids_started[-1]

    assert first_pid != second_pid
    assert manager.active_pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(first_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(second_pid, 0)
    await manager.stop()


@pytest.mark.asyncio
async def test_cancel_terminates_and_reaps_active_worker(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    task = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "job-1", "block": True})
    )
    await manager.wait_until_running("job-1")
    pid = manager.active_pid

    await manager.cancel("job-1")

    with pytest.raises(LocalWorkerError, match="cancelled"):
        await task
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0
    assert manager.metrics.cancelled_runs == 1
    assert manager.metrics.failed_runs == 0
    assert pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await manager.stop()


@pytest.mark.asyncio
async def test_cancel_before_startup_is_not_lost_and_never_spawns(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    await manager.cancel("queued-job")

    with pytest.raises(LocalWorkerError, match="cancelled"):
        await manager.run(
            ModelRole.OCR,
            {"job_id": "queued-job", "block": True},
        )

    assert manager.metrics.roles_started == []
    assert manager.active_pid is None
    assert manager.metrics.cancelled_runs == 1
    assert manager.metrics.failed_runs == 0
    await manager.stop()


@pytest.mark.asyncio
async def test_manager_rejects_unbounded_job_ids_before_storing_them(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    with pytest.raises(LocalWorkerError, match="rejected"):
        await manager.cancel("x" * 129)
    with pytest.raises(LocalWorkerError, match="rejected"):
        await manager.run(ModelRole.OCR, {"job_id": "x" * 129})

    assert manager._cancelled_jobs == {}
    assert manager._running_events == {}
    await manager.stop()


@pytest.mark.asyncio
async def test_prestart_cancellation_capacity_fails_without_evicting_authorization(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.model_manager as manager_module

    monkeypatch.setattr(manager_module, "_MAX_PENDING_CANCELLATIONS", 2)
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    await manager.cancel("queued-1")
    await manager.cancel("queued-2")
    with pytest.raises(LocalWorkerError, match="capacity"):
        await manager.cancel("queued-3")

    with pytest.raises(LocalWorkerError, match="cancelled"):
        await manager.run(ModelRole.OCR, {"job_id": "queued-1"})
    assert manager.metrics.roles_started == []
    await manager.stop()


@pytest.mark.asyncio
async def test_duplicate_live_job_id_is_rejected_without_second_spawn(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    first = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "duplicate", "block": True})
    )
    await manager.wait_until_running("duplicate")

    with pytest.raises(LocalWorkerError, match="already registered"):
        await manager.run(ModelRole.EXTRACTION, {"job_id": "duplicate"})

    assert manager.metrics.roles_started == [ModelRole.OCR]
    await manager.cancel("duplicate")
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await first
    await manager.stop()


@pytest.mark.asyncio
async def test_duplicate_queued_job_id_is_rejected_without_second_registration(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    active = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "active-a", "block": True})
    )
    await manager.wait_until_running("active-a")
    queued = asyncio.create_task(
        manager.run(ModelRole.EXTRACTION, {"job_id": "queued-b"})
    )
    for _ in range(100):
        if "queued-b" in manager._jobs:
            break
        await asyncio.sleep(0.01)

    with pytest.raises(LocalWorkerError, match="already registered"):
        await manager.run(ModelRole.SUMMARY, {"job_id": "queued-b"})

    await manager.cancel("queued-b")
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await queued
    await manager.cancel("active-a")
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await active
    assert manager.metrics.roles_started == [ModelRole.OCR]
    await manager.stop()


@pytest.mark.asyncio
async def test_task_cancellation_during_spawn_reaps_created_process(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    original_spawn = manager._spawn_worker
    spawned = asyncio.Event()
    spawned_pids: list[int] = []

    async def delayed_spawn() -> asyncio.subprocess.Process:
        process = await original_spawn()
        spawned_pids.append(process.pid)
        spawned.set()
        await asyncio.sleep(0.05)
        return process

    manager._spawn_worker = delayed_spawn  # type: ignore[method-assign]
    task = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "startup-cancel", "block": True})
    )
    await spawned.wait()
    task.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(spawned_pids[0], 0)
        assert manager.active_pid is None
    finally:
        try:
            os.killpg(spawned_pids[0], signal.SIGKILL)
        except ProcessLookupError:
            pass
    await manager.stop()


@pytest.mark.asyncio
async def test_task_cancellation_during_cleanup_reaps_group_and_releases_lock(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    original_terminate = manager._terminate_process
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def delayed_terminate(process: asyncio.subprocess.Process) -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        await original_terminate(process)

    manager._terminate_process = delayed_terminate  # type: ignore[method-assign]
    task = asyncio.create_task(
        manager.run(
            ModelRole.OCR,
            {"job_id": "cleanup-cancel", "spawn_descendant": True},
        )
    )
    await asyncio.wait_for(cleanup_started.wait(), 1)
    descendant_pid = int((worker_home / "descendant.pid").read_text())

    task.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0

    manager._terminate_process = original_terminate  # type: ignore[method-assign]
    assert await asyncio.wait_for(manager.run(ModelRole.OCR, {}), 1) == {
        "markdown": "# Synthetic OCR",
        "page_number": 1,
    }
    await manager.stop()


@pytest.mark.asyncio
async def test_cleanup_failure_before_group_death_poison_blocks_until_recovery(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    original_terminate = manager._terminate_process

    async def failing_terminate(process: asyncio.subprocess.Process) -> None:
        raise LocalWorkerError("Local worker failed.")

    manager._terminate_process = failing_terminate  # type: ignore[method-assign]

    with pytest.raises(LocalWorkerError, match="failed"):
        await manager.run(
            ModelRole.OCR,
            {"job_id": "cleanup-error", "spawn_descendant": True},
        )

    descendant_pid = int((worker_home / "descendant.pid").read_text())
    os.kill(descendant_pid, 0)
    assert manager.active_pid is not None
    assert manager.metrics.live_processes == 1
    assert "cleanup-error" not in manager._jobs

    roles_started = list(manager.metrics.roles_started)
    with pytest.raises(LocalWorkerError, match="unavailable"):
        await manager.run(ModelRole.OCR, {})
    assert manager.metrics.roles_started == roles_started

    other_home = worker_home.parent / "other-worker-home"
    other_manager = LocalModelManager(
        fake_worker_command,
        worker_home=other_home,
    )
    await other_manager.start()
    with pytest.raises(LocalWorkerError, match="unavailable"):
        await other_manager.run(ModelRole.EXTRACTION, {})
    assert other_manager.metrics.roles_started == []

    manager._terminate_process = original_terminate  # type: ignore[method-assign]
    await manager.stop()
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)

    recovered_manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home.parent / "recovered-worker-home",
    )
    await recovered_manager.start()
    assert await asyncio.wait_for(recovered_manager.run(ModelRole.OCR, {}), 1) == {
        "markdown": "# Synthetic OCR",
        "page_number": 1,
    }
    await recovered_manager.stop()
    await other_manager.stop()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_spawn_and_cleanup_never_loses_process(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    original_spawn = manager._spawn_worker
    original_terminate = manager._terminate_process
    spawn_created = asyncio.Event()
    release_spawn = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    spawned_pid: int | None = None

    async def delayed_spawn() -> asyncio.subprocess.Process:
        nonlocal spawned_pid
        process = await original_spawn()
        spawned_pid = process.pid
        spawn_created.set()
        await release_spawn.wait()
        return process

    async def delayed_terminate(process: asyncio.subprocess.Process) -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        await original_terminate(process)

    manager._spawn_worker = delayed_spawn  # type: ignore[method-assign]
    manager._terminate_process = delayed_terminate  # type: ignore[method-assign]
    task = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "double-spawn-cancel"})
    )
    await asyncio.wait_for(spawn_created.wait(), 1)

    task.cancel()
    release_spawn.set()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    task.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert spawned_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(spawned_pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(spawned_pid, 0)
    assert manager.metrics.max_live_processes == 1
    assert manager.metrics.pids_started == [spawned_pid]
    assert manager.metrics.roles_started == [ModelRole.OCR]
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0

    manager._spawn_worker = original_spawn  # type: ignore[method-assign]
    manager._terminate_process = original_terminate  # type: ignore[method-assign]
    assert await asyncio.wait_for(manager.run(ModelRole.OCR, {}), 1) == {
        "markdown": "# Synthetic OCR",
        "page_number": 1,
    }
    await manager.stop()


@pytest.mark.asyncio
async def test_cancel_is_idempotent_and_cannot_terminate_another_job(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    task = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "active-job", "delay_ms": 50})
    )
    await manager.wait_until_running("active-job")

    await manager.cancel("other-job")
    await manager.cancel("other-job")

    result = await task
    assert result["markdown"] == "# Synthetic OCR"
    await manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel", "stop"])
async def test_cancel_or_stop_interrupts_callback_that_suppresses_cancellation(
    fake_worker_command: list[str],
    worker_home: Path,
    operation: str,
) -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    async def uncooperative_callback(_progress: dict[str, object]) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_callback.wait()

    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    task = asyncio.create_task(
        manager.run(
            ModelRole.OCR,
            {"job_id": "hung-callback", "block": True},
            uncooperative_callback,
        )
    )
    await asyncio.wait_for(callback_started.wait(), 1)

    started_at = time.monotonic()
    if operation == "cancel":
        await asyncio.wait_for(manager.cancel("hung-callback"), 1)
    else:
        await asyncio.wait_for(manager.stop(), 1)
    with pytest.raises(LocalWorkerError, match="cancelled|stopped"):
        await asyncio.wait_for(task, 1)

    assert time.monotonic() - started_at < 1
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0
    if operation == "cancel":
        assert await manager.run(ModelRole.OCR, {}) == {
            "markdown": "# Synthetic OCR",
            "page_number": 1,
        }
        await manager.stop()
    release_callback.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_detached_callbacks_are_bounded_and_do_not_block_queue_or_stop(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    skipped_callback_calls = 0

    async def uncooperative_callback(_progress: dict[str, object]) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_callback.wait()

    def skipped_callback(_progress: dict[str, object]) -> None:
        nonlocal skipped_callback_calls
        skipped_callback_calls += 1

    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    first = asyncio.create_task(
        manager.run(
            ModelRole.OCR,
            {"job_id": "detached-first", "block": True},
            uncooperative_callback,
        )
    )
    await asyncio.wait_for(callback_started.wait(), 1)
    await manager.cancel("detached-first")
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await first

    assert len(manager._detached_callbacks) == 1
    for index in range(5):
        result = await asyncio.wait_for(
            manager.run(
                ModelRole.OCR,
                {"job_id": f"detached-repeat-{index}"},
                skipped_callback,
            ),
            1,
        )
        assert result["markdown"] == "# Synthetic OCR"
        assert len(manager._detached_callbacks) == 1
        assert manager.metrics.live_processes == 0
    assert skipped_callback_calls == 0

    assert await manager.run(ModelRole.OCR, {}) == {
        "markdown": "# Synthetic OCR",
        "page_number": 1,
    }
    await asyncio.wait_for(manager.stop(), 1)
    assert len(manager._detached_callbacks) == 1
    release_callback.set()
    for _ in range(100):
        if not manager._detached_callbacks:
            break
        await asyncio.sleep(0.01)
    assert manager._detached_callbacks == set()


@pytest.mark.asyncio
async def test_cancelled_callback_future_is_a_safe_failed_run(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    callback_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    callback_future.cancel()

    def cancelled_callback(_progress: dict[str, object]) -> asyncio.Future[None]:
        return callback_future

    with pytest.raises(LocalWorkerError, match="progress callback failed"):
        await manager.run(ModelRole.OCR, {}, cancelled_callback)

    assert manager.metrics.failed_runs == 1
    assert manager.metrics.cancelled_runs == 0
    assert manager.metrics.live_processes == 0
    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_success_reaps_descendant_after_parent_exits_cleanly(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    result = await manager.run(ModelRole.OCR, {"spawn_descendant": True})

    descendant_pid = result["descendant_pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    await manager.stop()


@pytest.mark.asyncio
async def test_cancel_kills_sigterm_ignoring_descendant_after_parent_exits(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.services.local_ai.model_manager as manager_module

    monkeypatch.setattr(manager_module, "_PROCESS_SHUTDOWN_SECONDS", 0.1)
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    task = asyncio.create_task(
        manager.run(
            ModelRole.OCR,
            {
                "job_id": "descendant-cancel",
                "spawn_descendant": True,
                "descendant_ignores_term": True,
                "block": True,
            },
        )
    )
    await manager.wait_until_running("descendant-cancel")
    pid_file = worker_home / "descendant.pid"
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    descendant_pid = int(pid_file.read_text())

    await manager.cancel("descendant-cancel")

    with pytest.raises(LocalWorkerError, match="cancelled"):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0
    await manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ({"malformed": True}, "protocol"),
        ({"oversized": True}, "protocol"),
        ({"mismatched_id": True}, "protocol"),
        ({"duplicate_terminal": True}, "protocol"),
        ({"post_terminal": True}, "protocol"),
        ({"crash": True}, "failed"),
        ({"safe_error": True}, "failed"),
    ],
)
async def test_failures_are_safe_and_always_reap_the_worker(
    fake_worker_command: list[str],
    worker_home: Path,
    mode: dict[str, bool],
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    canary = "Patient Alice 01/02/1970 secret prompt"
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    with pytest.raises(LocalWorkerError, match=message) as exc_info:
        await manager.run(ModelRole.OCR, {**mode, "content": canary})

    assert manager.active_pid is None
    assert canary not in caplog.text
    assert exc_info.value.__cause__ is None
    await manager.stop()


@pytest.mark.asyncio
async def test_timeout_reaps_worker_and_returns_safe_taxonomy(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
        timeout_seconds=0.05,
    )
    await manager.start()

    with pytest.raises(LocalWorkerTimeout, match="timed out"):
        await manager.run(ModelRole.OCR, {"block": True})

    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_progress_callback_receives_only_validated_non_content_fields(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    progress: list[dict[str, object]] = []
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    await manager.run(
        ModelRole.OCR,
        {"content": "never echo this"},
        progress.append,
    )

    assert progress
    assert all(set(item) <= {"role", "stage", "current", "total"} for item in progress)
    assert "never echo this" not in repr(progress)
    await manager.stop()


@pytest.mark.asyncio
async def test_callback_failure_reaps_worker_without_exposing_exception(
    fake_worker_command: list[str], worker_home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    canary = "callback secret"

    def fail_callback(_progress: dict[str, object]) -> None:
        raise RuntimeError(canary)

    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    with pytest.raises(LocalWorkerError, match="callback failed") as exc_info:
        await manager.run(ModelRole.OCR, {}, fail_callback)

    assert manager.active_pid is None
    assert canary not in caplog.text
    assert exc_info.value.__cause__ is None
    await manager.stop()


@pytest.mark.asyncio
async def test_worker_gets_minimal_secret_free_environment(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "provider-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-secret")
    monkeypatch.setenv("PATH", "/parent/secret/path")
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    result = await manager.run(ModelRole.OCR, {"inspect_env": True})

    assert result == {
        "home_matches": True,
        "offline": True,
        "cwd_matches": True,
        "path_is_fixed": True,
        "secret_names_present": [],
    }
    assert worker_home.stat().st_mode & 0o777 == 0o700
    await manager.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_rejects_active_and_queued_work(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    active = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "active", "block": True})
    )
    await manager.wait_until_running("active")
    queued = asyncio.create_task(
        manager.run(ModelRole.EXTRACTION, {"job_id": "queued"})
    )
    await asyncio.sleep(0)

    await manager.stop()
    await manager.stop()

    with pytest.raises(LocalWorkerError, match="stopped"):
        await active
    with pytest.raises(LocalWorkerError, match="stopped"):
        await queued
    assert manager.metrics.cancelled_runs == 2
    assert manager.metrics.failed_runs == 0
    with pytest.raises(LocalWorkerError, match="stopped"):
        await manager.run(ModelRole.OCR, {})
    assert manager.active_pid is None
    assert manager.metrics.live_processes == 0
    assert manager.metrics.cancelled_runs == 2
    assert manager.metrics.failed_runs == 1


@pytest.mark.asyncio
async def test_wait_until_running_rejects_invalid_or_unknown_ids_without_state_leak(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    with pytest.raises(LocalWorkerError, match="rejected"):
        await manager.wait_until_running("x" * 129, timeout=0.01)
    with pytest.raises(LocalWorkerError, match="not registered"):
        await manager.wait_until_running("missing", timeout=0.01)

    assert manager._running_events == {}
    assert manager._done_events == {}
    await manager.stop()


@pytest.mark.asyncio
async def test_constructor_defers_command_validation_until_start(
    worker_home: Path,
) -> None:
    manager = LocalModelManager("", worker_home=worker_home)

    with pytest.raises(LocalWorkerError, match="unavailable"):
        await manager.start()
    await manager.stop()


@pytest.mark.asyncio
async def test_start_resolves_executable_to_absolute_path(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    relative_python = os.path.relpath(fake_worker_command[0], Path.cwd())
    manager = LocalModelManager(
        [relative_python, *fake_worker_command[1:]],
        worker_home=worker_home,
    )

    await manager.start()

    assert Path(manager._worker_command[0]).is_absolute()
    await manager.stop()


@pytest.mark.asyncio
async def test_enabled_app_lifespan_starts_and_always_stops_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module
    import app.services.auth_service as auth_service
    import app.services.extraction.terminology as terminology
    from app.api import upload

    events: list[str] = []

    class FakeManager:
        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

    class FakeResult:
        rowcount = 0

    class FakeSession:
        async def execute(self, _query: object) -> FakeResult:
            return FakeResult()

        async def commit(self) -> None:
            return None

    class FakeSessionContext:
        async def __aenter__(self) -> FakeSession:
            return FakeSession()

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def no_purge(_db: object) -> int:
        return 0

    monkeypatch.setattr(main_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(main_module.settings, "phi_ner_enabled", False)
    monkeypatch.setattr(main_module.settings, "extraction_engine", "gemini")
    monkeypatch.setattr(main_module, "local_model_manager", FakeManager(), raising=False)
    monkeypatch.setattr(main_module, "async_session_factory", FakeSessionContext)
    monkeypatch.setattr(terminology, "schedule_medication_refresh", lambda: None)
    monkeypatch.setattr(auth_service, "purge_expired_revoked_tokens", no_purge)
    monkeypatch.setattr(upload, "start_extraction_worker", lambda: None)

    async with main_module.lifespan(FastAPI()):
        assert events == ["start"]

    assert events == ["start", "stop"]


@pytest.mark.asyncio
async def test_enabled_lifespan_attempts_stop_after_partial_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module

    events: list[str] = []

    class FailingManager:
        async def start(self) -> None:
            events.append("start")
            raise LocalWorkerError("Local worker is unavailable.")

        async def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(main_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(main_module.settings, "phi_ner_enabled", False)
    monkeypatch.setattr(main_module.settings, "extraction_engine", "gemini")
    monkeypatch.setattr(main_module, "local_model_manager", FailingManager())

    with pytest.raises(LocalWorkerError, match="unavailable"):
        async with main_module.lifespan(FastAPI()):
            pass

    assert events == ["start", "stop"]
