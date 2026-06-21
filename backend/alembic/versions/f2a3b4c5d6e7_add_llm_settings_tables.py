"""add llm settings tables

Per-user LLM provider credentials (encrypted API keys) and routing
preferences. Backs the Admin -> System "AI providers" settings pane.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f2a3b4c5d6e7"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_configs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("api_key_encrypted", sa.LargeBinary(), nullable=True),
        sa.Column("base_url", sa.String(512), nullable=True),
        sa.Column("model", sa.String(128), nullable=True),
        sa.Column(
            "enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "provider", name="uq_llm_user_provider"),
    )
    op.create_index(
        "ix_llm_provider_configs_user_id", "llm_provider_configs", ["user_id"]
    )

    op.create_table(
        "user_llm_preferences",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("default_provider", sa.String(32), nullable=True),
        sa.Column("summary_provider", sa.String(32), nullable=True),
        sa.Column("section_provider", sa.String(32), nullable=True),
        sa.Column("dedup_provider", sa.String(32), nullable=True),
        sa.Column("extraction_provider", sa.String(32), nullable=True),
        sa.Column("vision_provider", sa.String(32), nullable=True),
        sa.Column("extraction_engine", sa.String(16), nullable=True),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_user_llm_preferences_user"),
    )
    op.create_index(
        "ix_user_llm_preferences_user_id", "user_llm_preferences", ["user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_user_llm_preferences_user_id", "user_llm_preferences")
    op.drop_table("user_llm_preferences")
    op.drop_index("ix_llm_provider_configs_user_id", "llm_provider_configs")
    op.drop_table("llm_provider_configs")
