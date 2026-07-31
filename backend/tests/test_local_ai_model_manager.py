from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi import FastAPI

from app.services.local_ai.errors import (
    LocalInputLimitError,
    LocalWorkerError,
    LocalWorkerTimeout,
)
from app.services.local_ai.model_manager import (
    LocalModelManager as ProductionLocalModelManager,
)
from app.services.local_ai.types import ModelRole


class LocalModelManager(ProductionLocalModelManager):
    """Explicitly unisolated fake-worker manager for unit-level protocol tests."""

    def __init__(self, worker_command, **kwargs) -> None:
        super().__init__(
            worker_command,
            _allow_unisolated_test_worker=True,
            **kwargs,
        )


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
async def test_summary_token_count_uses_separate_worker_command(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    count = await manager.count_summary_tokens(
        {"job_id": "token-job", "fake_token_count": 173}
    )

    assert count == 173
    assert type(count) is int
    assert manager.metrics.roles_started == [ModelRole.SUMMARY]
    await manager.stop()


@pytest.mark.asyncio
async def test_summary_token_count_rejects_non_integer_result(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()

    with pytest.raises(LocalWorkerError, match="token count"):
        await manager.count_summary_tokens(
            {"job_id": "token-job", "fake_token_count": True}
        )

    await manager.stop()


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


def test_separate_backend_processes_never_overlap_workers(
    fake_worker_command: list[str],
    worker_home: Path,
    tmp_path: Path,
) -> None:
    guard_path = tmp_path / "live-worker.guard"
    start_path = tmp_path / "start"
    backend_root = Path(__file__).resolve().parents[1]
    script = "\n".join(
        [
            "import asyncio,json,time",
            "from pathlib import Path",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"start_path = Path({str(start_path)!r})",
            "while not start_path.exists(): time.sleep(0.01)",
            "async def main():",
            (
                "    manager = LocalModelManager("
                f"{fake_worker_command!r}, worker_home={str(worker_home)!r}, "
                "_allow_unisolated_test_worker=True)"
            ),
            "    await manager.start()",
            (
                "    result = await manager.run(ModelRole.OCR, "
                f"{{'delay_ms': 750, 'concurrency_guard_path': {str(guard_path)!r}}})"
            ),
            "    await manager.stop()",
            "    print(json.dumps(result), flush=True)",
            "asyncio.run(main())",
        ]
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=backend_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    start_path.touch()
    outputs = [process.communicate(timeout=15) for process in processes]

    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr
        assert json.loads(stdout)["overlap_detected"] is False


def test_worker_keeps_cross_process_lock_after_backend_parent_crash(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    started_marker = worker_home.parent / "orphan-started"
    crash_script = "\n".join(
        [
            "import asyncio,os",
            "from pathlib import Path",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"started_marker = Path({str(started_marker)!r})",
            "async def main():",
            (
                "    manager = LocalModelManager("
                f"{fake_worker_command!r}, worker_home={str(worker_home)!r}, "
                "_allow_unisolated_test_worker=True)"
            ),
            "    await manager.start()",
            (
                "    task = asyncio.create_task(manager.run("
                "ModelRole.OCR, {'job_id': 'orphaned', 'delay_ms': 2000, "
                f"'started_marker_path': {str(started_marker)!r}}}))"
            ),
            "    await manager.wait_until_running('orphaned')",
            "    while not started_marker.exists(): await asyncio.sleep(0.01)",
            "    print(manager.active_pid, flush=True)",
            "    os._exit(0)",
            "asyncio.run(main())",
        ]
    )
    crashed_parent = subprocess.Popen(
        [sys.executable, "-c", crash_script],
        cwd=backend_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert crashed_parent.stdout is not None
    worker_pid = int(crashed_parent.stdout.readline().strip())
    assert crashed_parent.wait(timeout=5) == 0

    follower_script = "\n".join(
        [
            "import asyncio,json,time",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            "async def main():",
            (
                "    manager = LocalModelManager("
                f"{fake_worker_command!r}, worker_home={str(worker_home)!r}, "
                "_allow_unisolated_test_worker=True)"
            ),
            "    await manager.start()",
            "    started = time.monotonic()",
            "    await manager.run(ModelRole.OCR, {})",
            "    print(json.dumps({'elapsed': time.monotonic() - started}), flush=True)",
            "    await manager.stop()",
            "asyncio.run(main())",
        ]
    )
    follower = subprocess.run(
        [sys.executable, "-c", follower_script],
        cwd=backend_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert follower.returncode == 0, follower.stderr
    assert json.loads(follower.stdout)["elapsed"] >= 1.25
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox contract")
def test_parent_watchdog_releases_sandboxed_lock_after_backend_crash(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    model_root = worker_home.parent / "models"
    started_marker = worker_home.parent / "hung-worker-started"
    configured_command = shlex.join(fake_worker_command)
    crash_script = "\n".join(
        [
            "import asyncio,os",
            "from pathlib import Path",
            "from app.config import settings",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"settings.local_ai_worker_command = {configured_command!r}",
            f"settings.local_ai_model_dir = {str(model_root)!r}",
            f"started_marker = Path({str(started_marker)!r})",
            "async def main():",
            f"    manager = LocalModelManager(worker_home={str(worker_home)!r})",
            "    await manager.start()",
            (
                "    asyncio.create_task(manager.run("
                "ModelRole.OCR, {'job_id': 'hung-orphan', 'block': True, "
                f"'started_marker_path': {str(started_marker)!r}}}))"
            ),
            "    await manager.wait_until_running('hung-orphan')",
            "    while not started_marker.exists(): await asyncio.sleep(0.01)",
            "    print(manager.active_pid, flush=True)",
            "    os._exit(0)",
            "asyncio.run(main())",
        ]
    )
    crashed_parent = subprocess.Popen(
        [sys.executable, "-c", crash_script],
        cwd=backend_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert crashed_parent.stdout is not None
    worker_pid = int(crashed_parent.stdout.readline().strip())
    assert crashed_parent.wait(timeout=5) == 0

    follower_script = "\n".join(
        [
            "import asyncio,json,time",
            "from app.config import settings",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"settings.local_ai_worker_command = {configured_command!r}",
            f"settings.local_ai_model_dir = {str(model_root)!r}",
            "async def main():",
            f"    manager = LocalModelManager(worker_home={str(worker_home)!r})",
            "    await manager.start()",
            "    started = time.monotonic()",
            "    await manager.run(ModelRole.OCR, {})",
            "    print(json.dumps({'elapsed': time.monotonic() - started}), flush=True)",
            "    await manager.stop()",
            "asyncio.run(main())",
        ]
    )
    follower = subprocess.run(
        [sys.executable, "-c", follower_script],
        cwd=backend_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert follower.returncode == 0, follower.stderr
    assert json.loads(follower.stdout)["elapsed"] < 3
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox contract")
def test_parent_watchdog_retains_lock_through_failed_cleanup_and_backend_crash(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    model_root = worker_home.parent / "models"
    configured_command = shlex.join(fake_worker_command)
    crash_script = "\n".join(
        [
            "import asyncio,json,os,signal,subprocess,time",
            "from pathlib import Path",
            "from app.config import settings",
            "from app.services.local_ai.errors import LocalWorkerError",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"settings.local_ai_worker_command = {configured_command!r}",
            f"settings.local_ai_model_dir = {str(model_root)!r}",
            "async def main():",
            f"    manager = LocalModelManager(worker_home={str(worker_home)!r})",
            "    await manager.start()",
            "    async def fail_after_parent_exit(process):",
            (
                "        child_pids = [int(value) for value in "
                "subprocess.check_output(['pgrep', '-P', str(process.pid)], "
                "text=True).split()]"
            ),
            (
                "        watchdog_pid = next(pid for pid in child_pids "
                "if os.getpgid(pid) != process.pid)"
            ),
            "        os.kill(watchdog_pid, signal.SIGSTOP)",
            "        process.kill()",
            "        await process.wait()",
            "        manager._test_watchdog_pid = watchdog_pid",
            "        raise LocalWorkerError('Local worker failed.')",
            "    manager._terminate_process = fail_after_parent_exit",
            "    try:",
            (
                "        await manager.run("
                "ModelRole.OCR, {'spawn_descendant': True, "
                "'descendant_ignores_term': True, 'malformed': True})"
            ),
            "    except LocalWorkerError:",
            "        pass",
            "    descendant_pid = int((Path(manager._worker_home) / 'descendant.pid').read_text())",
            (
                "    print(json.dumps({'pgid': manager.active_pid, "
                "'descendant_pid': descendant_pid, "
                "'watchdog_pid': manager._test_watchdog_pid}), flush=True)"
            ),
            "    os._exit(0)",
            "asyncio.run(main())",
        ]
    )
    crashed_parent = subprocess.Popen(
        [sys.executable, "-c", crash_script],
        cwd=backend_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    follower: subprocess.CompletedProcess[str] | None = None
    orphan_pgid: int | None = None
    try:
        assert crashed_parent.stdout is not None
        owner_state = json.loads(crashed_parent.stdout.readline())
        orphan_pgid = owner_state["pgid"]
        descendant_pid = owner_state["descendant_pid"]
        watchdog_pid = owner_state["watchdog_pid"]
        os.kill(descendant_pid, 0)
        os.kill(watchdog_pid, 0)
        assert crashed_parent.wait(timeout=5) == 0

        follower_script = "\n".join(
            [
                "import asyncio,json",
                "from app.config import settings",
                "from app.services.local_ai.model_manager import LocalModelManager",
                "from app.services.local_ai.types import ModelRole",
                f"settings.local_ai_worker_command = {configured_command!r}",
                f"settings.local_ai_model_dir = {str(model_root)!r}",
                "async def main():",
                (
                    f"    manager = LocalModelManager(worker_home="
                    f"{str(worker_home.parent / 'follower-home')!r})"
                ),
                "    await manager.start()",
                "    result = await manager.run(ModelRole.OCR, {})",
                "    print(json.dumps(result), flush=True)",
                "    await manager.stop()",
                "asyncio.run(main())",
            ]
        )
        follower_process = subprocess.Popen(
            [sys.executable, "-c", follower_script],
            cwd=backend_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.3)
        assert follower_process.poll() is None

        os.kill(watchdog_pid, signal.SIGCONT)
        follower_stdout, follower_stderr = follower_process.communicate(timeout=10)
        follower = subprocess.CompletedProcess(
            follower_process.args,
            follower_process.returncode,
            follower_stdout,
            follower_stderr,
        )

        assert follower.returncode == 0, follower.stderr
        assert json.loads(follower.stdout)["markdown"] == "# Synthetic OCR"
        with pytest.raises(ProcessLookupError):
            os.kill(descendant_pid, 0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(watchdog_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        with pytest.raises(ProcessLookupError):
            os.kill(watchdog_pid, 0)
    finally:
        if orphan_pgid is not None:
            try:
                os.killpg(orphan_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if crashed_parent.poll() is None:
            crashed_parent.kill()
            crashed_parent.communicate(timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox contract")
@pytest.mark.asyncio
async def test_default_manager_os_sandbox_denies_worker_network(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings

    monkeypatch.setattr(
        settings,
        "local_ai_worker_command",
        shlex.join(fake_worker_command),
    )
    monkeypatch.setattr(
        settings,
        "local_ai_model_dir",
        str(worker_home.parent / "models"),
    )
    manager = ProductionLocalModelManager(worker_home=worker_home)
    await manager.start()

    result = await manager.run(ModelRole.OCR, {"probe_network": True})

    assert result["network_denied"] is True
    assert result["network_errno"] in {1, 13}

    result = await manager.run(ModelRole.OCR, {"inspect_lock_fd": True})
    assert result["lock_fd_inherited"] is True
    assert result["lock_identity_matches"] is True
    assert result["lock_path_exposed"] is False

    with tempfile.TemporaryDirectory(dir="/tmp") as socket_directory:
        socket_path = str(Path(socket_directory) / "probe.sock")
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(socket_path)
            server.listen(1)
            result = await manager.run(
                ModelRole.OCR,
                {"probe_unix_socket_path": socket_path},
            )
        finally:
            server.close()
    assert result["network_denied"] is True
    assert result["network_errno"] in {1, 13}
    await manager.stop()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox contract")
@pytest.mark.asyncio
async def test_explicit_worker_command_does_not_bypass_os_sandbox(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings

    monkeypatch.setattr(
        settings,
        "local_ai_model_dir",
        str(worker_home.parent / "models"),
    )
    manager = ProductionLocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
    )
    await manager.start()

    result = await manager.run(ModelRole.OCR, {"probe_network": True})

    assert result["network_denied"] is True
    assert result["network_errno"] in {1, 13}
    await manager.stop()


@pytest.mark.asyncio
async def test_default_manager_fails_closed_without_supported_network_sandbox(
    fake_worker_command: list[str],
    worker_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import settings
    from app.services.local_ai import model_manager

    monkeypatch.setattr(
        settings,
        "local_ai_worker_command",
        shlex.join(fake_worker_command),
    )
    monkeypatch.setattr(
        settings,
        "local_ai_model_dir",
        str(worker_home.parent / "models"),
    )
    monkeypatch.setattr(model_manager.sys, "platform", "unsupported")
    manager = ProductionLocalModelManager(worker_home=worker_home)

    with pytest.raises(LocalWorkerError, match="network isolation is unavailable"):
        await manager.start()


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
async def test_cancel_registered_stops_lock_queued_job_without_reservation(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    active = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "registered-active", "block": True})
    )
    await manager.wait_until_running("registered-active")
    queued = asyncio.create_task(
        manager.run(ModelRole.EXTRACTION, {"job_id": "registered-queued"})
    )
    for _ in range(100):
        if "registered-queued" in manager._jobs:
            break
        await asyncio.sleep(0.01)

    assert await manager.cancel_registered("registered-queued") is True
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await queued
    assert await manager.cancel_registered("never-registered") is False
    assert manager._cancelled_jobs == {}

    await manager.cancel("registered-active")
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


def test_cleanup_failure_retains_cross_process_lock_until_orphan_group_exits(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    release_owner = worker_home.parent / "release-owner"
    owner_script = "\n".join(
        [
            "import asyncio,json",
            "from pathlib import Path",
            "from app.services.local_ai.errors import LocalWorkerError",
            "from app.services.local_ai.model_manager import LocalModelManager",
            "from app.services.local_ai.types import ModelRole",
            f"release_owner = Path({str(release_owner)!r})",
            "async def main():",
            (
                "    manager = LocalModelManager("
                f"{fake_worker_command!r}, worker_home={str(worker_home)!r}, "
                "_allow_unisolated_test_worker=True)"
            ),
            "    await manager.start()",
            "    async def fail_after_parent_exit(process):",
            "        process.kill()",
            "        await process.wait()",
            "        raise LocalWorkerError('Local worker failed.')",
            "    manager._terminate_process = fail_after_parent_exit",
            "    try:",
            (
                "        await manager.run("
                "ModelRole.OCR, {'spawn_descendant': True, "
                "'descendant_ignores_term': True})"
            ),
            "    except LocalWorkerError:",
            "        pass",
            "    descendant_pid = int((Path(manager._worker_home) / 'descendant.pid').read_text())",
            (
                "    print(json.dumps({'pgid': manager.active_pid, "
                "'descendant_pid': descendant_pid}), flush=True)"
            ),
            "    while not release_owner.exists():",
            "        await asyncio.sleep(0.01)",
            "asyncio.run(main())",
        ]
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", owner_script],
        cwd=backend_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    follower: subprocess.Popen[str] | None = None
    orphan_pgid: int | None = None
    try:
        assert owner.stdout is not None
        owner_state = json.loads(owner.stdout.readline())
        orphan_pgid = owner_state["pgid"]
        os.kill(owner_state["descendant_pid"], 0)

        follower_script = "\n".join(
            [
                "import asyncio,json",
                "from app.services.local_ai.model_manager import LocalModelManager",
                "from app.services.local_ai.types import ModelRole",
                "async def main():",
                (
                    "    manager = LocalModelManager("
                    f"{fake_worker_command!r}, worker_home={str(worker_home)!r}, "
                    "_allow_unisolated_test_worker=True)"
                ),
                "    await manager.start()",
                "    result = await manager.run(ModelRole.OCR, {})",
                "    print(json.dumps(result), flush=True)",
                "    await manager.stop()",
                "asyncio.run(main())",
            ]
        )
        follower = subprocess.Popen(
            [sys.executable, "-c", follower_script],
            cwd=backend_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.5)
        assert follower.poll() is None

        os.killpg(orphan_pgid, signal.SIGKILL)
        follower_stdout, follower_stderr = follower.communicate(timeout=5)
        assert follower.returncode == 0, follower_stderr
        assert json.loads(follower_stdout)["markdown"] == "# Synthetic OCR"
    finally:
        if orphan_pgid is not None:
            try:
                os.killpg(orphan_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if follower is not None and follower.poll() is None:
            follower.kill()
            follower.communicate(timeout=5)
        release_owner.touch()
        owner_stdout, owner_stderr = owner.communicate(timeout=5)
        assert owner.returncode == 0, f"{owner_stdout}\n{owner_stderr}"


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
async def test_detached_callback_is_isolated_from_later_jobs_and_stop(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    later_callback_calls = 0

    async def uncooperative_callback(_progress: dict[str, object]) -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_callback.wait()

    def later_callback(_progress: dict[str, object]) -> None:
        nonlocal later_callback_calls
        later_callback_calls += 1

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
                later_callback,
            ),
            1,
        )
        assert result["markdown"] == "# Synthetic OCR"
        assert len(manager._detached_callbacks) == 1
        assert manager.metrics.live_processes == 0
    assert later_callback_calls == 5

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
async def test_detached_callback_retention_is_bounded_before_spawning_more_workers(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    from app.services.local_ai.model_manager import _MAX_DETACHED_CALLBACKS

    manager = LocalModelManager(fake_worker_command, worker_home=worker_home)
    await manager.start()
    retained = {
        asyncio.get_running_loop().create_future()
        for _ in range(_MAX_DETACHED_CALLBACKS)
    }
    manager._detached_callbacks.update(retained)

    with pytest.raises(LocalWorkerError, match="unavailable"):
        await manager.run(ModelRole.OCR, {"job_id": "retention-limit"})

    assert manager.metrics.pids_started == []
    for callback in retained:
        callback.cancel()
    manager._prune_detached_callbacks()
    await manager.stop()


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
async def test_input_limit_error_preserves_safe_non_retryable_taxonomy(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
    )
    await manager.start()

    with pytest.raises(LocalInputLimitError) as exc_info:
        await manager.run(
            ModelRole.EXTRACTION,
            {"input_limit_error": True},
        )

    assert exc_info.value.code == "local_input_limit_exceeded"
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert manager.active_pid is None
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
async def test_forward_progress_extends_idle_timeout_but_not_hard_deadline(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
        timeout_seconds=0.25,
        hard_timeout_seconds=1.0,
    )
    await manager.start()

    result = await manager.run(
        ModelRole.EXTRACTION,
        {
            "progress_steps": 3,
            "progress_delay_ms": 150,
        },
    )

    assert result["entities"] == []
    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_repeated_progress_does_not_extend_idle_timeout(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
        timeout_seconds=0.08,
        hard_timeout_seconds=0.5,
    )
    await manager.start()

    with pytest.raises(LocalWorkerTimeout, match="timed out"):
        await manager.run(
            ModelRole.EXTRACTION,
            {
                "repeat_progress": True,
                "progress_delay_ms": 25,
            },
        )

    assert manager.active_pid is None
    await manager.stop()


def test_activity_counter_extends_idle_without_changing_page_completion() -> None:
    from app.services.local_ai.protocol import ProgressPayload, ProtocolViolation

    first = ProgressPayload(
        role=ModelRole.EXTRACTION,
        stage="processing",
        current=0,
        total=3,
        activity=1,
    )
    heartbeat = first.model_copy(update={"activity": 2})

    assert ProductionLocalModelManager._validate_progress_sequence(None, first) is True
    assert (
        ProductionLocalModelManager._validate_progress_sequence(first, heartbeat)
        is True
    )
    assert (
        ProductionLocalModelManager._validate_progress_sequence(heartbeat, heartbeat)
        is False
    )
    with pytest.raises(ProtocolViolation, match="progress"):
        ProductionLocalModelManager._validate_progress_sequence(
            heartbeat,
            heartbeat.model_copy(update={"activity": 1}),
        )


def test_final_memory_frame_may_omit_optional_activity_counter() -> None:
    from app.services.local_ai.protocol import ProgressPayload, ProtocolViolation

    processing = ProgressPayload(
        role=ModelRole.EXTRACTION,
        stage="processing",
        current=1,
        total=1,
        activity=4,
    )
    finalizing = ProgressPayload(
        role=ModelRole.EXTRACTION,
        stage="finalizing",
        current=1,
        total=1,
        active_memory_bytes=512 * 1024**2,
        peak_memory_bytes=6 * 1024**3,
    )

    assert (
        ProductionLocalModelManager._validate_progress_sequence(
            processing,
            finalizing,
        )
        is True
    )
    remembered = ProductionLocalModelManager._remember_progress_activity(
        processing,
        finalizing,
    )
    assert remembered.activity == 4
    assert (
        ProductionLocalModelManager._validate_progress_sequence(
            remembered,
            finalizing.model_copy(update={"activity": 4}),
        )
        is False
    )
    assert (
        ProductionLocalModelManager._validate_progress_sequence(
            remembered,
            finalizing.model_copy(update={"activity": 5}),
        )
        is True
    )
    with pytest.raises(ProtocolViolation, match="progress"):
        ProductionLocalModelManager._validate_progress_sequence(
            remembered,
            finalizing.model_copy(update={"activity": 3}),
        )


@pytest.mark.asyncio
async def test_activity_only_progress_extends_liveness_without_public_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only a real stage or page advance reaches the public progress callback."""

    from app.services.local_ai.model_manager import _RunState
    from app.services.local_ai.protocol import (
        ProgressPayload,
        ReadyPayload,
        ResultPayload,
        WorkerResponse,
        encode_message,
    )

    class Input:
        def write(self, _value: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    class Process:
        stdin = Input()
        stdout = object()

    manager = LocalModelManager([], worker_home=tmp_path / "worker-home")
    request = manager._build_request(ModelRole.EXTRACTION, {}, "job-1")
    first = ProgressPayload(
        role=ModelRole.EXTRACTION,
        stage="processing",
        current=54,
        total=54,
        activity=1,
    )
    heartbeat = first.model_copy(update={"activity": 2})
    finalizing = ProgressPayload(
        role=ModelRole.EXTRACTION,
        stage="finalizing",
        current=1,
        total=1,
        active_memory_bytes=512 * 1024**2,
        peak_memory_bytes=6 * 1024**3,
    )
    frames = iter(
        [
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="ready",
                    payload=ReadyPayload(role=ModelRole.EXTRACTION),
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="progress",
                    payload=first,
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="progress",
                    payload=heartbeat,
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="progress",
                    payload=finalizing,
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="result",
                    payload=ResultPayload(data={"entities": []}),
                )
            ),
        ]
    )

    async def readline(_process: object) -> bytes:
        return next(frames)

    async def noop(*_args: object) -> None:
        return None

    monkeypatch.setattr(manager, "_readline", readline)
    monkeypatch.setattr(manager, "_request_shutdown", noop)
    monkeypatch.setattr(manager, "_drain_and_reap_after_terminal", noop)
    observed: list[dict[str, object]] = []
    durable_heartbeats: list[int] = []

    result = await manager._exchange(
        Process(),  # type: ignore[arg-type]
        request,
        ModelRole.EXTRACTION,
        observed.append,
        lambda: durable_heartbeats.append(1),
        _RunState(job_id="job-1"),
    )

    assert result == {"entities": []}
    assert observed == [
        first.model_dump(mode="json", exclude_none=True),
        finalizing.model_copy(update={"current": 54, "total": 54}).model_dump(
            mode="json",
            exclude_none=True,
        ),
    ]
    assert durable_heartbeats == [1]


@pytest.mark.asyncio
async def test_worker_eof_logs_only_safe_role_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.services.local_ai.model_manager import _RunState

    class Input:
        def write(self, _value: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    class Process:
        stdin = Input()
        stdout = object()
        returncode = -9

    manager = LocalModelManager([], worker_home=tmp_path / "worker-home")
    request = manager._build_request(ModelRole.EXTRACTION, {}, "job-1")

    async def eof(_process: object) -> bytes:
        return b""

    monkeypatch.setattr(manager, "_readline", eof)
    caplog.set_level("WARNING")

    with pytest.raises(LocalWorkerError, match="failed"):
        await manager._exchange(
            Process(),  # type: ignore[arg-type]
            request,
            ModelRole.EXTRACTION,
            None,
            None,
            _RunState(job_id="job-1"),
        )

    assert "role=extraction returncode=-9" in caplog.text


@pytest.mark.asyncio
async def test_nonzero_exit_after_terminal_logs_only_safe_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Process:
        returncode = -9

        async def wait(self) -> int:
            return self.returncode

    manager = LocalModelManager([], worker_home=tmp_path / "worker-home")

    async def eof(_process: object) -> bytes:
        return b""

    monkeypatch.setattr(manager, "_readline", eof)
    caplog.set_level("WARNING")

    with pytest.raises(LocalWorkerError, match="failed"):
        await manager._drain_and_reap_after_terminal(
            Process(),  # type: ignore[arg-type]
            "request-1",
        )

    assert "category=nonzero_after_terminal returncode=-9" in caplog.text


@pytest.mark.asyncio
async def test_generation_failure_logs_only_allowlisted_category_role_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.services.local_ai.model_manager import _RunState

    class Input:
        def write(self, _value: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    class Process:
        stdin = Input()
        stdout = object()
        returncode = 0
        private = "clinical-content-must-not-be-logged"

    manager = LocalModelManager([], worker_home=tmp_path / "worker-home")
    request = manager._build_request(ModelRole.EXTRACTION, {}, "job-1")
    frames = iter(
        [
            (
                json.dumps(
                    {
                        "version": 1,
                        "request_id": request.request_id,
                        "kind": "ready",
                        "payload": {"role": "extraction"},
                    },
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            ),
            (
                json.dumps(
                    {
                        "version": 1,
                        "request_id": request.request_id,
                        "kind": "error",
                        "payload": {
                            "code": "generation_failed",
                            "message": "Local worker generation failed.",
                            "category": "output_limit",
                        },
                    },
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            ),
        ]
    )

    async def readline(_process: object) -> bytes:
        return next(frames)

    async def noop(*_args: object) -> None:
        return None

    monkeypatch.setattr(manager, "_readline", readline)
    monkeypatch.setattr(manager, "_request_shutdown", noop)
    monkeypatch.setattr(manager, "_drain_and_reap_after_terminal", noop)
    caplog.set_level("WARNING")

    with pytest.raises(LocalWorkerError, match="generation failed"):
        await manager._exchange(
            Process(),  # type: ignore[arg-type]
            request,
            ModelRole.EXTRACTION,
            None,
            None,
            _RunState(job_id="job-1"),
        )

    assert (
        "safe error code=generation_failed category=output_limit "
        "role=extraction returncode=0" in caplog.text
    )
    assert Process.private not in caplog.text


@pytest.mark.asyncio
async def test_progress_renews_idle_deadline_before_awaiting_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A timely worker frame gives its callback a fresh idle-time allowance."""

    from app.services.local_ai.model_manager import _RunState
    from app.services.local_ai.protocol import (
        ProgressPayload,
        ReadyPayload,
        ResultPayload,
        WorkerResponse,
        encode_message,
    )

    class Input:
        def write(self, _value: bytes) -> None:
            return None

        async def drain(self) -> None:
            return None

    class Process:
        stdin = Input()
        stdout = object()

    manager = LocalModelManager(
        [],
        worker_home=tmp_path / "worker-home",
        timeout_seconds=0.06,
        hard_timeout_seconds=0.5,
    )
    request = manager._build_request(ModelRole.EXTRACTION, {}, "job-renew")
    frames = iter(
        [
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="ready",
                    payload=ReadyPayload(role=ModelRole.EXTRACTION),
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="progress",
                    payload=ProgressPayload(
                        role=ModelRole.EXTRACTION,
                        stage="processing",
                        current=1,
                        total=1,
                    ),
                )
            ),
            encode_message(
                WorkerResponse(
                    version=1,
                    request_id=request.request_id,
                    kind="result",
                    payload=ResultPayload(data={"entities": []}),
                )
            ),
        ]
    )
    reads = 0

    async def readline(_process: object) -> bytes:
        nonlocal reads
        reads += 1
        if reads == 2:
            await asyncio.sleep(0.04)
        return next(frames)

    async def callback(_progress: dict[str, object]) -> None:
        await asyncio.sleep(0.04)

    async def noop(*_args: object) -> None:
        return None

    monkeypatch.setattr(manager, "_readline", readline)
    monkeypatch.setattr(manager, "_request_shutdown", noop)
    monkeypatch.setattr(manager, "_drain_and_reap_after_terminal", noop)

    result = await manager._exchange(
        Process(),  # type: ignore[arg-type]
        request,
        ModelRole.EXTRACTION,
        callback,
        None,
        _RunState(job_id="job-renew"),
    )

    assert result == {"entities": []}


@pytest.mark.asyncio
async def test_forward_progress_cannot_extend_hard_timeout(
    fake_worker_command: list[str], worker_home: Path
) -> None:
    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
        timeout_seconds=0.25,
        hard_timeout_seconds=0.5,
    )
    await manager.start()

    with pytest.raises(LocalWorkerTimeout, match="timed out"):
        await manager.run(
            ModelRole.EXTRACTION,
            {
                "progress_steps": 20,
                "progress_delay_ms": 50,
            },
        )

    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_hard_timeout_includes_blocked_request_drain(worker_home: Path) -> None:
    manager = LocalModelManager(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        worker_home=worker_home,
        timeout_seconds=0.05,
        hard_timeout_seconds=0.08,
    )
    await manager.start()

    with pytest.raises(LocalWorkerTimeout, match="timed out"):
        await asyncio.wait_for(
            manager.run(
                ModelRole.EXTRACTION,
                {"content": "x" * (2 * 1024 * 1024)},
            ),
            timeout=1,
        )

    assert manager.active_pid is None
    await manager.stop()


@pytest.mark.asyncio
async def test_hard_timeout_includes_async_progress_callback(
    fake_worker_command: list[str],
    worker_home: Path,
) -> None:
    callback_started = asyncio.Event()

    async def blocked_callback(_progress: dict[str, object]) -> None:
        callback_started.set()
        await asyncio.Event().wait()

    manager = LocalModelManager(
        fake_worker_command,
        worker_home=worker_home,
        timeout_seconds=0.3,
        hard_timeout_seconds=0.5,
    )
    await manager.start()

    with pytest.raises(LocalWorkerTimeout, match="timed out"):
        await asyncio.wait_for(
            manager.run(
                ModelRole.EXTRACTION,
                {
                    "progress_steps": 2,
                    "progress_delay_ms": 20,
                },
                blocked_callback,
            ),
            timeout=1,
        )

    assert callback_started.is_set()
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
    assert (
        "category=progress_callback_failure role=ocr "
        "exception_type=RuntimeError" in caplog.text
    )
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
    monkeypatch.setattr(
        main_module, "local_model_manager", FakeManager(), raising=False
    )
    monkeypatch.setattr(
        main_module,
        "_reconcile_model_pack_operations_on_startup",
        lambda: events.append("reconcile-pack") or 0,
    )
    monkeypatch.setattr(main_module, "async_session_factory", FakeSessionContext)
    monkeypatch.setattr(terminology, "schedule_medication_refresh", lambda: None)
    monkeypatch.setattr(auth_service, "purge_expired_revoked_tokens", no_purge)
    monkeypatch.setattr(upload, "start_extraction_worker", lambda: None)

    async with main_module.lifespan(FastAPI()):
        assert events == ["reconcile-pack", "start"]

    assert events == ["reconcile-pack", "start", "stop"]


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
