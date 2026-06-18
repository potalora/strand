"""uploaded_files: add cancel_requested + section-progress columns

Supports cooperative cancel (POST /upload/cancel sets cancel_requested; the
extraction worker aborts and marks the file ``cancelled``) and section-level
progress (progress_stage + progress_detail surfaced on the per-file status).

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "e1f2a3b4c5d6"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "uploaded_files",
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("progress_stage", sa.Text(), nullable=True),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("progress_detail", JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("uploaded_files", "progress_detail")
    op.drop_column("uploaded_files", "progress_stage")
    op.drop_column("uploaded_files", "cancel_requested")
