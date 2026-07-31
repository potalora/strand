"""Durable wake and graceful-drain behavior for strict-local summaries."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.ai_summary import AISummaryPrompt
from app.models.local_ai import LocalAIJob
from app.services.local_ai.manifest import canonicalize_manifest_snapshot
from tests.conftest import auth_headers, create_test_patient


@pytest.mark.asyncio
async def test_duplicate_wakeups_run_one_summary_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process registry coalesces duplicate durable queue wakeups."""
    from app.services.local_ai.summary_runner import LocalSummaryRunner

    release = asyncio.Event()

    async def wait_for_release(_job_ids):
        await release.wait()

    resume = AsyncMock(side_effect=wait_for_release)
    monkeypatch.setattr(
        "app.services.local_ai.summary_runner.resume_grounded_local_summary_jobs",
        resume,
    )
    runner = LocalSummaryRunner()
    job_id = uuid4()

    runner.enqueue(job_id)
    runner.enqueue(job_id)
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*runner._tasks.values())

    resume.assert_awaited_once_with([job_id])


@pytest.mark.asyncio
async def test_stop_cancels_tasks_then_runs_durable_requeue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planned shutdown drains live runners before moving work to recovery."""
    from app.services.local_ai.summary_runner import LocalSummaryRunner

    cancelled = asyncio.Event()

    async def run_one(_job_id):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    requeue = AsyncMock()
    runner = LocalSummaryRunner()
    monkeypatch.setattr(runner, "_run_one", run_one)
    monkeypatch.setattr(
        "app.services.local_ai.summary_runner.requeue_interrupted_summary_jobs",
        requeue,
    )
    runner.enqueue(uuid4())
    await asyncio.sleep(0)

    await runner.stop_and_requeue()

    assert cancelled.is_set()
    requeue.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_failed_shutdown_requeue_clears_registry_before_a_new_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed recovery write cannot leave cancelled tasks blocking a restart."""
    from app.services.local_ai.summary_runner import LocalSummaryRunner

    async def wait_forever(_job_id):
        await asyncio.Event().wait()

    requeue = AsyncMock(side_effect=RuntimeError("database unavailable"))
    runner = LocalSummaryRunner()
    monkeypatch.setattr(runner, "_run_one", wait_forever)
    monkeypatch.setattr(
        "app.services.local_ai.summary_runner.requeue_interrupted_summary_jobs",
        requeue,
    )
    first = uuid4()
    runner.enqueue(first)
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await runner.stop_and_requeue()

    assert runner._tasks == {}
    runner.start([])
    second = uuid4()
    runner.enqueue(second)
    await asyncio.sleep(0)
    assert second in runner._tasks

    requeue.side_effect = None
    await runner.stop_and_requeue()


@pytest.mark.asyncio
async def test_unexpected_runner_failure_is_consumed_and_logged_without_content(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing wake is cleaned up without leaking its exception text to logs."""
    from app.services.local_ai.summary_runner import LocalSummaryRunner

    secret = "sensitive patient content"

    async def fail(_job_id):
        raise RuntimeError(secret)

    runner = LocalSummaryRunner()
    monkeypatch.setattr(runner, "_run_one", fail)
    job_id = uuid4()
    runner.enqueue(job_id)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert runner._tasks == {}
    assert "Strict-local summary runner task failed" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_shutdown_requeue_stops_at_the_summary_recovery_safety_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Shutdown recovery handles a bounded batch and leaves later work queued."""
    from app.services.ai.summarizer import requeue_interrupted_summary_jobs

    jobs = [
        SimpleNamespace(
            id=uuid4(),
            cancel_requested=False,
            status="processing",
            stage="summary",
            progress={"stage": "summary"},
            failure=None,
            completed_at=None,
        )
        for _ in range(1001)
    ]

    class Result:
        def scalars(self):
            return self

        def all(self):
            return jobs

    class Session:
        committed = False

        async def execute(self, _statement):
            return Result()

        async def commit(self):
            self.committed = True

    class SessionContext:
        session = Session()

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, *_args):
            return None

    resumed = await requeue_interrupted_summary_jobs(session_factory=SessionContext)

    assert len(resumed) == 1000
    assert jobs[0].stage == "recovery"
    assert jobs[-1].stage == "summary"
    assert SessionContext.session.committed is True
    assert (
        "Strict-local summary shutdown recovery reached its safety bound" in caplog.text
    )


@pytest.mark.asyncio
async def test_requeue_preserves_explicit_cancel_but_recovers_planned_stop(
    client,
    db_session: AsyncSession,
) -> None:
    """A planned stop never converts uncancelled work into a user cancellation."""
    from app.services.ai.summarizer import requeue_interrupted_summary_jobs
    from tests.test_summarization import _strict_manifest

    _headers, user_id = await auth_headers(client, email="summary-drain@example.com")
    patient = await create_test_patient(db_session, user_id)
    snapshot, digest = canonicalize_manifest_snapshot(_strict_manifest())
    prompts = [
        AISummaryPrompt(
            id=uuid4(),
            user_id=UUID(user_id),
            patient_id=patient.id,
            summary_type="full",
            processing_mode="validated_strict_local",
            scope_filter={},
            system_prompt="Locked policy",
            user_prompt="Grounded facts",
            target_model="locked-local-summary",
            suggested_config={},
            record_count=0,
            generated_at=datetime.now(timezone.utc),
        )
        for _ in range(2)
    ]
    planned = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompts[0].id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        status="processing",
        stage="summary",
        progress={"stage": "summary"},
    )
    explicit = LocalAIJob(
        user_id=UUID(user_id),
        summary_prompt_id=prompts[1].id,
        kind="summary",
        processing_mode="validated_strict_local",
        manifest_snapshot=snapshot,
        manifest_sha256=digest,
        status="processing",
        stage="summary",
        progress={"stage": "summary"},
        cancel_requested=True,
    )
    db_session.add_all([*prompts, planned, explicit])
    await db_session.commit()

    resumed = await requeue_interrupted_summary_jobs(
        session_factory=async_sessionmaker(
            db_session.bind,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    )
    await db_session.refresh(planned)
    await db_session.refresh(explicit)

    assert resumed == [planned.id]
    assert (planned.status, planned.stage) == ("queued", "recovery")
    assert (explicit.status, explicit.stage) == ("cancelled", "cancelled")
