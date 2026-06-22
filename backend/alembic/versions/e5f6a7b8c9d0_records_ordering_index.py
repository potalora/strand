"""records ordering index + fix partial-index predicate (D4 perf)

Divergence D4: ``GET /records`` / ``GET /timeline`` were slow (~3.5s / ~1.4s on
~1750 active records) because every read full-scanned ``health_records`` and
top-N-sorted the result.

Two root causes, both fixed here:

1. **Predicate mismatch made the existing partial index unusable.** Every read
   path filters with ``HealthRecord.is_duplicate.is_(False)`` (SQL
   ``is_duplicate IS FALSE``), but ``idx_health_records_user_active`` was created
   ``WHERE deleted_at IS NULL AND is_duplicate = false``. Postgres will not match
   an ``IS FALSE`` query against a ``= false`` partial-index predicate, so the
   index was never used and queries fell back to a sequential scan of the WHOLE
   table — including every soft-deleted and duplicate row (the dedup pipeline
   produces many ``is_duplicate = true`` rows from cumulative re-exports).
   Recreated with an ``IS FALSE`` predicate so it actually matches the queries.

2. **No index served the default ordering.** ``GET /records`` /
   ``GET /timeline`` order by ``effective_date DESC NULLS LAST, id`` (with a
   ``LIMIT``). Added a partial b-tree on ``(user_id, effective_date DESC NULLS
   LAST, id)`` so the ordered page is an index range scan that touches only the
   page's rows (no full scan, no sort) and skips dead rows. The index column
   order/nulls placement matches the query's ``ORDER BY`` exactly so the planner
   can drop the sort entirely.

Revision ID: e5f6a7b8c9d0
Revises: a1b2c3d4e5f7
Create Date: 2026-06-21 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Recreate the composite active-rows index with a predicate that MATCHES
    #    the ``is_duplicate IS FALSE`` the query layer emits (was ``= false``,
    #    which Postgres treated as a different predicate → index never used).
    op.execute("DROP INDEX IF EXISTS idx_health_records_user_active")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_health_records_user_active
        ON health_records (user_id, record_type)
        WHERE deleted_at IS NULL AND is_duplicate IS FALSE
        """
    )

    # 2. Ordering index for the default newest-first page (records + timeline).
    #    Column order + DESC NULLS LAST mirror the ORDER BY so the sort is
    #    eliminated; partial predicate keeps it small and skips dead rows.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_health_records_user_eff_active
        ON health_records (user_id, effective_date DESC NULLS LAST, id)
        WHERE deleted_at IS NULL AND is_duplicate IS FALSE
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_health_records_user_eff_active")
    # Restore the original (= false) predicate form on the composite index.
    op.execute("DROP INDEX IF EXISTS idx_health_records_user_active")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_health_records_user_active
        ON health_records (user_id, record_type)
        WHERE deleted_at IS NULL AND is_duplicate = false
        """
    )
