"""Regression tests for loop-safe extraction semaphores.

A module-level ``asyncio.Semaphore`` binds to the event loop it first blocks on.
Reused across loops under contention it raises
``RuntimeError: <Semaphore> is bound to a different event loop``. That defect
made the full unstructured pipeline drop every chunk (0 records) at higher
``section_extraction_concurrency`` whenever Gemini-semaphore contention forced a
task to wait on a stale loop. These tests pin the loop-keyed fix in place.
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import upload as upload_mod
from app.api.upload import _get_extraction_semaphore, _get_gemini_semaphore


async def _coro_get_gemini_sem() -> asyncio.Semaphore:
    """Return the Gemini semaphore for the currently running loop."""
    return _get_gemini_semaphore()


def _acquire_under_contention(get_sem, n_tasks: int, hold: float) -> tuple[int, int]:
    """Run ``n_tasks`` coroutines that all contend for the semaphore.

    Returns ``(succeeded, failed)``. ``n_tasks`` is set larger than the
    semaphore limit so some tasks must actually *wait*, which is the precise
    condition that triggers cross-loop binding errors.
    """

    async def worker() -> None:
        sem = get_sem()
        async with sem:
            await asyncio.sleep(hold)

    async def run_all() -> tuple[int, int]:
        results = await asyncio.gather(
            *(worker() for _ in range(n_tasks)), return_exceptions=True
        )
        failed = sum(1 for r in results if isinstance(r, Exception))
        return len(results) - failed, failed

    return asyncio.run(run_all())


@pytest.mark.parametrize("get_sem", [_get_gemini_semaphore, _get_extraction_semaphore])
def test_semaphore_survives_separate_loops_under_contention(get_sem) -> None:
    """Each fresh event loop gets a semaphore bound to it — no RuntimeError.

    Demands more concurrent acquirers than the semaphore limit so tasks must
    block; with the old module-global the second loop raised
    ``RuntimeError: ... bound to a different event loop`` for every waiter.
    """
    # First loop creates and binds a semaphore.
    ok1, failed1 = _acquire_under_contention(get_sem, n_tasks=40, hold=0.02)
    assert failed1 == 0, "contention on first loop should not fail"
    assert ok1 == 40

    # Second, independent loop must NOT reuse the stale (now-closed) semaphore.
    ok2, failed2 = _acquire_under_contention(get_sem, n_tasks=40, hold=0.02)
    assert failed2 == 0, "second loop must get a freshly-bound semaphore"
    assert ok2 == 40


def test_distinct_semaphore_per_running_loop() -> None:
    """The same loop reuses one semaphore; a different loop gets a different one."""
    upload_mod._gemini_semaphores.clear()

    async def grab_two_in_one_loop() -> bool:
        return _get_gemini_semaphore() is _get_gemini_semaphore()

    assert asyncio.run(grab_two_in_one_loop()) is True

    # Run two loops back-to-back and keep BOTH loops alive simultaneously so the
    # caches cannot collapse to one entry. A fresh, dedicated loop per call gives
    # each its own semaphore keyed by that loop object.
    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    try:
        sem_a = loop_a.run_until_complete(_coro_get_gemini_sem())
        sem_b = loop_b.run_until_complete(_coro_get_gemini_sem())
        assert sem_a is not sem_b, "each event loop must get its own semaphore instance"
        assert upload_mod._gemini_semaphores[loop_a] is sem_a
        assert upload_mod._gemini_semaphores[loop_b] is sem_b
    finally:
        loop_a.close()
        loop_b.close()


def test_closed_loops_are_pruned_from_cache() -> None:
    """Stale (closed-loop) cache entries are cleaned up, keeping caches bounded."""
    upload_mod._gemini_semaphores.clear()

    async def touch() -> None:
        _get_gemini_semaphore()

    asyncio.run(touch())
    asyncio.run(touch())
    asyncio.run(touch())

    # After three short-lived loops, only entries for live loops should remain.
    # All three loops are closed by now, so a fresh access prunes them down.
    async def final_touch() -> int:
        _get_gemini_semaphore()
        return len(upload_mod._gemini_semaphores)

    remaining = asyncio.run(final_touch())
    assert remaining == 1, "closed loops should be pruned, leaving only the live loop"
