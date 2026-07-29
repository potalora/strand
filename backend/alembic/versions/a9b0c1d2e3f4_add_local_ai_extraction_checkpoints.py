"""Add encrypted durable strict-local extraction checkpoints.

Revision ID: a9b0c1d2e3f4
Revises: a8b9c0d1e2f3
Create Date: 2026-07-28
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a9b0c1d2e3f4"
down_revision: Union[str, Sequence[str], None] = "a8b9c0d1e2f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "local_ai_extraction_checkpoints",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("checkpoint_key", sa.String(length=64), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("ocr_text_sha256", sa.String(length=64), nullable=False),
        sa.Column("page_bindings_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=128), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("raw_result_sha256", sa.String(length=64), nullable=True),
        sa.Column("raw_extraction_result", sa.LargeBinary(), nullable=True),
        sa.Column("extraction_result", sa.LargeBinary(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["local_ai_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            name="uq_local_ai_extraction_checkpoints_job",
        ),
    )
    op.create_index(
        "ix_local_ai_extraction_checkpoints_job",
        "local_ai_extraction_checkpoints",
        ["job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_local_ai_extraction_checkpoints_job",
        table_name="local_ai_extraction_checkpoints",
    )
    op.drop_table("local_ai_extraction_checkpoints")
