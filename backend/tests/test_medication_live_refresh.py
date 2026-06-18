"""Tests for the live medication-index refresh (RxNorm).

Covers: the committed baseline vs gitignored live-cache loader precedence, the
staleness gate (fresh cache no-ops with no network), fail-open behavior on a
fetch error, and the non-blocking startup scheduler. All offline — the RxNorm
rebuild is mocked.
"""
from __future__ import annotations

import gzip
import json
import os
import time

import pytest

from app.services.extraction import terminology as t


def _write_med_cache(path, *, alias="metformin", code="111111", display="LIVE Metformin"):
    """Write a minimal but valid medications index gz to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "codes": {code: [t.RXNORM_SYSTEM, code, display]},
        "index": {alias: code},
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


@pytest.fixture(autouse=True)
def _isolate_med_index(monkeypatch, tmp_path):
    """Point the live cache at a tmp path and reset the medication cache slot."""
    monkeypatch.setattr(t, "_LIVE_MED_CACHE", tmp_path / "terminology" / "medications.json.gz")
    # Fresh cache dict so we don't leak the real baseline-loaded index across tests.
    monkeypatch.setattr(t, "_INDEX_CACHE", dict(t._INDEX_CACHE))
    t._INDEX_CACHE.pop("medication", None)
    yield


class TestLoaderPrecedence:
    def test_baseline_used_when_no_live_cache(self):
        # No live cache at the tmp path -> committed baseline serves.
        assert not t._LIVE_MED_CACHE.exists()
        assert t.lookup_medication("Metformin").code == "6809"

    def test_live_cache_wins_over_baseline(self):
        _write_med_cache(t._LIVE_MED_CACHE, alias="metformin", code="111111")
        t._INDEX_CACHE.pop("medication", None)
        c = t.lookup_medication("Metformin")
        assert c is not None and c.code == "111111", "live cache should override baseline"

    def test_other_categories_unaffected_by_live_cache(self):
        # Scope is medications only — conditions still come from the baseline.
        _write_med_cache(t._LIVE_MED_CACHE)
        assert t.lookup_condition("Hypertension").code == "I10"


class TestStalenessGate:
    def test_fresh_cache_no_rebuild_no_network(self, monkeypatch):
        _write_med_cache(t._LIVE_MED_CACHE)  # mtime = now
        called = []
        monkeypatch.setattr(t, "_build_medication_cache",
                            lambda out: called.append(out))
        result = t.refresh_medication_index(max_age_days=7)
        assert result is False, "fresh cache should not rebuild"
        assert called == [], "no rebuild/network when cache is fresh"

    def test_missing_cache_triggers_rebuild(self, monkeypatch):
        assert not t._LIVE_MED_CACHE.exists()
        calls = []

        def fake_build(out):
            calls.append(out)
            _write_med_cache(out, alias="metformin", code="222222")

        monkeypatch.setattr(t, "_build_medication_cache", fake_build)
        result = t.refresh_medication_index(max_age_days=7)
        assert result is True
        assert calls == [t._LIVE_MED_CACHE]
        # Hot-swap: the freshly built cache is now what lookups serve.
        assert t.lookup_medication("Metformin").code == "222222"

    def test_old_cache_triggers_rebuild(self, monkeypatch):
        _write_med_cache(t._LIVE_MED_CACHE)
        old = time.time() - 8 * 86400
        os.utime(t._LIVE_MED_CACHE, (old, old))
        calls = []
        monkeypatch.setattr(t, "_build_medication_cache",
                            lambda out: (calls.append(out), _write_med_cache(out)))
        result = t.refresh_medication_index(max_age_days=7)
        assert result is True and calls == [t._LIVE_MED_CACHE]


class TestFailOpen:
    def test_fetch_error_does_not_raise_and_keeps_serving(self, monkeypatch):
        # Prime the baseline medication index so an index already exists.
        assert t.lookup_medication("Metformin").code == "6809"

        def boom(out):
            raise RuntimeError("simulated RxNorm network failure")

        monkeypatch.setattr(t, "_build_medication_cache", boom)
        # Must not raise.
        result = t.refresh_medication_index(max_age_days=0)
        assert result is False
        # Existing index still served.
        assert t.lookup_medication("Metformin").code == "6809"

    def test_no_connectivity_missing_cache_still_serves_baseline(self, monkeypatch):
        assert not t._LIVE_MED_CACHE.exists()
        monkeypatch.setattr(t, "_build_medication_cache",
                            lambda out: (_ for _ in ()).throw(OSError("no network")))
        assert t.refresh_medication_index(max_age_days=0) is False
        assert t.lookup_medication("Metformin").code == "6809"  # baseline fallback


class TestStartupScheduler:
    async def test_schedule_is_non_blocking(self, monkeypatch):
        ran = []
        monkeypatch.setattr(t, "refresh_medication_index",
                            lambda max_age_days=7: ran.append(max_age_days))
        task = t.schedule_medication_refresh()
        # Created, not awaited: it hasn't run synchronously at return time.
        assert not task.done()
        assert ran == []
        await task
        assert ran, "background refresh eventually ran"

    async def test_schedule_swallows_refresh_errors(self, monkeypatch):
        def boom(max_age_days=7):
            raise RuntimeError("should be swallowed by fail-open refresh")

        # The real refresh fail-opens; here we ensure the scheduler awaits cleanly
        # even if the underlying callable raises (defense in depth).
        monkeypatch.setattr(t, "refresh_medication_index", boom)
        task = t.schedule_medication_refresh()
        # Awaiting should not raise out of the scheduler.
        try:
            await task
        except RuntimeError:
            pytest.fail("scheduler must not propagate refresh errors")
