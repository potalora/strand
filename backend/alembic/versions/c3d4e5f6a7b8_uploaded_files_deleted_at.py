"""uploaded_files soft-delete: add deleted_at

Supports DELETE /upload/:id, which soft-deletes the uploaded_file and cascade
soft-deletes the health_records it produced (via source_file_id).

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "uploaded_files",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("uploaded_files", "deleted_at")
