"""add_performance_indexes

Revision ID: a1b2c3d4e5f6
Revises: 9ac4081003fc
Create Date: 2026-02-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '9ac4081003fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add performance indexes for common query patterns."""
    # Composite partial index for the most common query pattern
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_health_records_user_active
        ON health_records (user_id, record_type)
        WHERE deleted_at IS NULL AND is_duplicate = false
    """)

    # Dedup candidate indexes
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup_candidates_pair
        ON dedup_candidates (record_a_id, record_b_id)
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_dedup_candidates_status
        ON dedup_candidates (record_a_id, status)
    """)


def downgrade() -> None:
    """Remove performance indexes."""
    op.execute("DROP INDEX IF EXISTS idx_dedup_candidates_status")
    op.execute("DROP INDEX IF EXISTS idx_dedup_candidates_pair")
    op.execute("DROP INDEX IF EXISTS idx_health_records_user_active")
