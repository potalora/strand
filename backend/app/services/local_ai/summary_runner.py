"""In-process wake and graceful-drain registry for durable local summaries."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app.config import settings
from app.services.ai.summarizer import (
    requeue_interrupted_summary_jobs,
    resume_grounded_local_summary_jobs,
)

logger = logging.getLogger(__name__)


class LocalSummaryRunner:
    """Wake persisted summary jobs without becoming their source of truth."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._draining = False

    def enqueue(self, job_id: UUID) -> None:
        """Schedule one durable job unless shutdown or an equivalent wake owns it."""
        if self._draining or job_id in self._tasks:
            return
        task = asyncio.create_task(self._run_one(job_id))
        self._tasks[job_id] = task

        def _discard(completed: asyncio.Task[None]) -> None:
            if not completed.cancelled() and completed.exception() is not None:
                logger.error("Strict-local summary runner task failed")
            if self._tasks.get(job_id) is completed:
                self._tasks.pop(job_id, None)

        task.add_done_callback(_discard)

    def start(self, job_ids: list[UUID]) -> None:
        """Open a new lifespan and wake its recovered summary jobs."""
        self._draining = False
        for job_id in job_ids:
            self.enqueue(job_id)

    async def _run_one(self, job_id: UUID) -> None:
        """Run one database-authoritative summary claim."""
        await resume_grounded_local_summary_jobs([job_id])

    async def stop_and_requeue(self) -> None:
        """Drain live wakes, then transactionally preserve unfinished work."""
        self._draining = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        try:
            if tasks:
                drain = asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(drain),
                        timeout=settings.local_ai_shutdown_drain_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "Strict-local summary runner shutdown drain timed out"
                    )
            await requeue_interrupted_summary_jobs()
        finally:
            self._tasks.clear()


local_summary_runner = LocalSummaryRunner()
