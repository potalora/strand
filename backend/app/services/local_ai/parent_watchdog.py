"""Kill an isolated worker group if its backend parent exits."""

from __future__ import annotations

import os
import select
import signal
import stat
import sys
import time
from contextlib import suppress


def _positive_pid(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed > 1 else 0


def _nonstdio_descriptor(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return -1
    return parsed if parsed >= 3 else -1


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return -1
    return parsed if parsed >= 0 else -1


def _lock_descriptor_is_valid(
    descriptor: int,
    expected_device: int,
    expected_inode: int,
) -> bool:
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
        and (not hasattr(os, "getuid") or metadata.st_uid == os.getuid())
        and metadata.st_dev == expected_device
        and metadata.st_ino == expected_inode
    )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _kill_worker_boundary(worker_pid: int, process_group_id: int) -> bool:
    if os.getpgrp() == process_group_id:
        return False
    try:
        if process_group_id == worker_pid:
            os.killpg(process_group_id, signal.SIGKILL)
        else:
            os.kill(worker_pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass
    while _process_group_exists(process_group_id):
        time.sleep(0.01)
    return True


def watch_parent(
    parent_pid: int,
    worker_pid: int,
    process_group_id: int,
    ready_descriptor: int,
    lock_descriptor: int,
    expected_device: int,
    expected_inode: int,
) -> int:
    """Retain the worker lease until its exact process boundary is gone."""

    if (
        sys.platform != "darwin"
        or min(parent_pid, worker_pid, process_group_id) <= 1
        or ready_descriptor < 3
        or lock_descriptor < 3
        or ready_descriptor == lock_descriptor
        or expected_device < 0
        or expected_inode <= 0
        or worker_pid != process_group_id
        or os.getppid() != worker_pid
        or os.getpgrp() == process_group_id
        or not _lock_descriptor_is_valid(
            lock_descriptor,
            expected_device,
            expected_inode,
        )
    ):
        return 2
    try:
        if os.getpgid(worker_pid) != process_group_id:
            return 2
        queue = select.kqueue()
        changes = [
            select.kevent(
                process_id,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_CLEAR,
                fflags=select.KQ_NOTE_EXIT,
            )
            for process_id in (parent_pid, worker_pid)
        ]
        queue.control(changes, 0, 0)
        os.write(ready_descriptor, b"1")
        os.close(ready_descriptor)
        ready_descriptor = -1
        events = queue.control(None, 2, None)
    except (OSError, ValueError):
        _kill_worker_boundary(worker_pid, process_group_id)
        return 1
    finally:
        if ready_descriptor >= 0:
            with suppress(OSError):
                os.close(ready_descriptor)
        if "queue" in locals():
            queue.close()

    if not any(event.ident in {parent_pid, worker_pid} for event in events):
        return 1
    return 0 if _kill_worker_boundary(worker_pid, process_group_id) else 1


def main(argv: list[str] | None = None) -> int:
    """Validate fixed numeric arguments and start the parent watcher."""

    values = sys.argv[1:] if argv is None else argv
    if len(values) != 7:
        return 2
    parent_pid, worker_pid, process_group_id = map(
        _positive_pid,
        values[:3],
    )
    ready_descriptor, lock_descriptor = map(_nonstdio_descriptor, values[3:5])
    expected_device, expected_inode = map(_nonnegative_integer, values[5:])
    return watch_parent(
        parent_pid,
        worker_pid,
        process_group_id,
        ready_descriptor,
        lock_descriptor,
        expected_device,
        expected_inode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
