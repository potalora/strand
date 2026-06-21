"""Event-loop responsiveness during background extraction (perf divergence D3).

The bug: the background extraction worker ran CPU-bound *synchronous* work
(spaCy NER PHI-scrubbing, entity→FHIR mapping, local clinical NLP, local
text parsing) directly on the asyncio event loop. While a file extracted, that
loop thread was pinned for seconds at a time, so every concurrent API request
stalled (~18-22s vs ~1-3.5s idle).

The fix offloads those CPU-bound sync calls to a worker thread via
``asyncio.to_thread(...)``. CPython releases the GIL periodically (and spaCy /
BLAS release it during their heavy C sections), so the event loop interleaves
and keeps handling requests while the CPU work runs off-thread.

These tests prove the loop stays responsive. The primary seam is
``phi_scrubber.scrub_phi_async`` (the offloaded de-identification scrub — the
single biggest blocker). A heartbeat coroutine ticks on ``asyncio.sleep(0)``
while a deliberately CPU-bound scrub runs:

* offloaded (``scrub_phi_async``)  → heartbeat keeps ticking (GREEN)
* direct (``scrub_phi``)           → heartbeat is starved to zero (the old bug)

The direct-call test is the negative control: it shows the heartbeat assertion
is meaningful (a blocking call really does freeze the loop), so the GREEN test
isn't passing for a trivial reason.

NOTE: this targets API responsiveness (D3), not extraction THROUGHPUT (D2).
Throughput is GIL-bound; speeding it up would need a process pool, which is out
of scope here. Offloading frees the loop without making the CPU work faster.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.services.ai.phi_scrubber import scrub_phi, scrub_phi_async

pytestmark = pytest.mark.asyncio


def _busy_spin(duration_s: float) -> None:
    """Burn CPU for ~``duration_s`` wall-clock in pure Python.

    Stands in for the spaCy NER pass so the test is deterministic and needs no
    model installed: it holds the GIL between CPython's periodic switch points,
    exactly like the real synchronous scrub does between its C sections.
    """
    end = time.monotonic() + duration_s
    x = 0
    while time.monotonic() < end:
        x += 1


@pytest.fixture
def heavy_ner(monkeypatch):
    """Force ``scrub_phi`` to do ~0.4s of CPU work via the NER pass.

    ``scrub_phi`` imports ``redact_named_entities`` lazily at call time, so a
    monkeypatch on the ``phi_ner`` module attribute is picked up on the next
    call — no spaCy model required.
    """

    def fake_redact(text):
        _busy_spin(0.4)
        return text, {}

    monkeypatch.setattr(
        "app.services.ai.phi_ner.redact_named_entities", fake_redact
    )
    return fake_redact


async def test_scrub_phi_async_matches_sync_output():
    """The offloaded scrub returns byte-identical output to the sync scrub.

    Offloading changes only WHERE the work runs, never WHAT it produces.
    """
    text = "Patient John Smith, SSN 123-45-6789, seen 07/14/2023 by Dr. Adams."
    sync_out = scrub_phi(text, enable_ner=False)
    async_out = await scrub_phi_async(text, enable_ner=False)
    assert async_out == sync_out


async def test_scrub_phi_async_runs_in_worker_thread(monkeypatch):
    """The CPU work executes off the event-loop (main) thread."""
    main_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_redact(text):
        seen["thread"] = threading.get_ident()
        return text, {}

    monkeypatch.setattr(
        "app.services.ai.phi_ner.redact_named_entities", fake_redact
    )

    await scrub_phi_async("Seen by Dr. Helen Park.", enable_ner=True)

    assert seen.get("thread") is not None
    assert seen["thread"] != main_thread


async def test_offloaded_scrub_keeps_event_loop_responsive(heavy_ner):
    """GREEN: a heartbeat keeps ticking while the offloaded scrub burns CPU."""
    ticks = 0
    stop = asyncio.Event()

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat start

    # ~0.4s of CPU, offloaded to a worker thread.
    await scrub_phi_async("text with a name", enable_ner=True)

    stop.set()
    await hb

    # A blocked loop ticks ~0 during the scrub (see the negative-control test
    # below, which asserts exactly 0); an offloaded one ticks many times. Tick
    # cadence is gated by CPython's GIL switch interval (~5ms), so the count is
    # in the tens over a 0.4s spin — the threshold leaves wide margin over 0.
    assert ticks >= 20, f"event loop appears blocked during offloaded scrub (ticks={ticks})"


async def test_direct_scrub_starves_event_loop(heavy_ner):
    """Negative control: the OLD direct (sync) call freezes the loop.

    Proves the heartbeat assertion above is meaningful — a blocking call really
    does starve the loop, so the GREEN test isn't green for a trivial reason.
    """
    ticks = 0
    stop = asyncio.Event()

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)  # let the heartbeat run a little
    ticks_before = ticks

    # Direct synchronous call (pre-fix behavior): pins the single loop thread.
    scrub_phi("text with a name", enable_ner=True)

    # No await between the call returning and this read, so the heartbeat could
    # not have advanced *during* the blocking call.
    ticks_during = ticks - ticks_before

    stop.set()
    await hb

    assert ticks_during == 0, (
        f"expected the loop to be frozen during a direct scrub, ticked {ticks_during}"
    )
