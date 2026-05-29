"""idempotency fields: external_id, source_system, content_hash, version, record_versions

Revision ID: b2c3d4e5f6a7
Revises: f1a2b3c4d5e6
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b2c3d4e5f6a7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("health_records", sa.Column("external_id", sa.Text(), nullable=True))
    op.add_column("health_records", sa.Column("source_system", sa.Text(), nullable=True))
    op.add_column("health_records", sa.Column("content_hash", sa.Text(), nullable=True))
    op.add_column(
        "health_records",
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
    )
    op.create_index(
        "uq_health_records_identity",
        "health_records",
        ["user_id", "source_system", "external_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND external_id IS NOT NULL"),
    )
    op.create_table(
        "record_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "record_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("health_records.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("fhir_resource", postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("changed_fields", postgresql.JSONB(), nullable=True),
        sa.Column("source_file_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_record_versions_record_id", "record_versions", ["record_id"])


def downgrade() -> None:
    op.drop_index("ix_record_versions_record_id", table_name="record_versions")
    op.drop_table("record_versions")
    op.drop_index("uq_health_records_identity", table_name="health_records")
    op.drop_column("health_records", "version")
    op.drop_column("health_records", "content_hash")
    op.drop_column("health_records", "source_system")
    op.drop_column("health_records", "external_id")
