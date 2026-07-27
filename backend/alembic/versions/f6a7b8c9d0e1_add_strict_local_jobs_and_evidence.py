"""Add strict-local jobs, encrypted page checkpoints, and evidence.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-26
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_uploaded_files_id_user_id",
        "uploaded_files",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_ai_summary_prompts_id_user_id",
        "ai_summary_prompts",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_health_records_id_source_file_user",
        "health_records",
        ["id", "source_file_id", "user_id"],
    )

    op.add_column(
        "user_llm_preferences",
        sa.Column("processing_mode", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "uploaded_files",
        sa.Column(
            "processing_mode",
            sa.String(length=32),
            server_default="cloud_assisted",
            nullable=False,
        ),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("processing_manifest", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "uploaded_files",
        sa.Column("processing_schema_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column(
            "processing_mode",
            sa.String(length=32),
            server_default="cloud_assisted",
            nullable=False,
        ),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column("model_provenance", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "ai_summary_prompts",
        sa.Column("typed_response", sa.LargeBinary(), nullable=True),
    )

    op.create_table(
        "local_ai_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summary_prompt_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("processing_mode", sa.String(length=32), nullable=False),
        sa.Column("manifest_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column(
            "progress",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column("failure", postgresql.JSONB(), nullable=True),
        sa.Column(
            "audit_metadata",
            postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "cancel_requested",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "(kind = 'ingestion' AND upload_id IS NOT NULL "
            "AND summary_prompt_id IS NULL) "
            "OR (kind = 'summary' AND upload_id IS NULL "
            "AND summary_prompt_id IS NOT NULL)",
            name="ck_local_ai_jobs_kind_target",
        ),
        sa.ForeignKeyConstraint(
            ["summary_prompt_id", "user_id"],
            ["ai_summary_prompts.id", "ai_summary_prompts.user_id"],
            name="fk_local_ai_jobs_summary_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_local_ai_jobs_upload_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_local_ai_jobs_user_status",
        "local_ai_jobs",
        ["user_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_local_ai_jobs_upload_id",
        "local_ai_jobs",
        ["upload_id"],
        unique=False,
    )
    op.create_index(
        "ix_local_ai_jobs_summary_prompt_id",
        "local_ai_jobs",
        ["summary_prompt_id"],
        unique=False,
    )
    op.execute(
        """
        CREATE FUNCTION reject_local_ai_job_identity_update()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.processing_mode IS DISTINCT FROM OLD.processing_mode
               OR NEW.manifest_snapshot IS DISTINCT FROM OLD.manifest_snapshot
               OR NEW.manifest_sha256 IS DISTINCT FROM OLD.manifest_sha256 THEN
                RAISE EXCEPTION USING
                    ERRCODE = '23514',
                    MESSAGE = 'local AI job identity is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_local_ai_jobs_immutable_identity
        BEFORE UPDATE OF processing_mode, manifest_snapshot, manifest_sha256
        ON local_ai_jobs
        FOR EACH ROW
        EXECUTE FUNCTION reject_local_ai_job_identity_update()
        """
    )

    op.create_table(
        "local_ai_pages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("checkpoint_key", sa.String(length=64), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("ocr_result", sa.LargeBinary(), nullable=False),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default="[]",
            nullable=False,
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
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["local_ai_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            "page_number",
            name="uq_local_ai_pages_job_page",
        ),
    )
    op.create_index(
        "ix_local_ai_pages_job_page",
        "local_ai_pages",
        ["job_id", "page_number"],
        unique=False,
    )

    op.create_table(
        "extraction_evidence",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("upload_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("health_record_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("excerpt", sa.LargeBinary(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("field_paths", sa.LargeBinary(), nullable=False),
        sa.Column("source_metadata", sa.LargeBinary(), nullable=False),
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
            ["health_record_id", "upload_id", "user_id"],
            [
                "health_records.id",
                "health_records.source_file_id",
                "health_records.user_id",
            ],
            name="fk_extraction_evidence_record_lineage",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["health_record_id"],
            ["health_records.id"],
            name="fk_extraction_evidence_health_record",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id", "user_id"],
            ["uploaded_files.id", "uploaded_files.user_id"],
            name="fk_extraction_evidence_upload_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_extraction_evidence_health_record_id",
        "extraction_evidence",
        ["health_record_id"],
        unique=False,
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_local_ai_jobs_immutable_identity "
        "ON local_ai_jobs"
    )
    op.execute("DROP FUNCTION IF EXISTS reject_local_ai_job_identity_update()")
    op.drop_table("extraction_evidence")
    op.drop_table("local_ai_pages")
    op.drop_table("local_ai_jobs")

    op.drop_column("ai_summary_prompts", "typed_response")
    op.drop_column("ai_summary_prompts", "model_provenance")
    op.drop_column("ai_summary_prompts", "processing_mode")
    op.drop_column("uploaded_files", "processing_schema_version")
    op.drop_column("uploaded_files", "processing_manifest")
    op.drop_column("uploaded_files", "processing_mode")
    op.drop_column("user_llm_preferences", "processing_mode")
    op.execute(
        "ALTER TABLE health_records "
        "DROP CONSTRAINT IF EXISTS uq_health_records_id_source_file_user"
    )
    op.execute(
        "ALTER TABLE ai_summary_prompts "
        "DROP CONSTRAINT IF EXISTS uq_ai_summary_prompts_id_user_id"
    )
    op.execute(
        "ALTER TABLE uploaded_files "
        "DROP CONSTRAINT IF EXISTS uq_uploaded_files_id_user_id"
    )
