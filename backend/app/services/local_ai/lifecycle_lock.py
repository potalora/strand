"""PostgreSQL transaction lock shared by pack mutations and strict job admission."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Stable signed BIGINT namespace reserved for the process-global local-AI pack.
LOCAL_AI_LIFECYCLE_LOCK_KEY = 0x4D454454494D45


async def acquire_local_ai_lifecycle_lock(db: AsyncSession) -> None:
    """Hold the global lifecycle lock until the current transaction ends."""

    statement = text(
        "SELECT pg_advisory_xact_lock(:local_ai_lifecycle_lock_key)"
    ).bindparams(local_ai_lifecycle_lock_key=LOCAL_AI_LIFECYCLE_LOCK_KEY)
    await db.execute(statement)
