"""Default new processing snapshots to prompt-only.

Revision ID: b0c1d2e3f4a5
Revises: b4c5d6e7f8a9
Create Date: 2026-07-31
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b0c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "b4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PROMPT_ONLY_UPLOAD_FUNCTION = """
CREATE OR REPLACE FUNCTION enforce_uploaded_file_processing_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.processing_mode NOT IN (
        'prompt_only',
        'cloud_assisted',
        'validated_strict_local'
    ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = 'upload processing mode is invalid';
    END IF;
    IF TG_OP = 'UPDATE'
       AND (
            NEW.processing_mode IS DISTINCT FROM OLD.processing_mode
            OR NEW.processing_manifest IS DISTINCT FROM OLD.processing_manifest
            OR NEW.processing_schema_version
               IS DISTINCT FROM OLD.processing_schema_version
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = 'upload processing snapshot is immutable';
    END IF;
    RETURN NEW;
END;
$$
"""

_LEGACY_UPLOAD_FUNCTION = """
CREATE OR REPLACE FUNCTION enforce_uploaded_file_processing_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.processing_mode NOT IN (
        'cloud_assisted',
        'validated_strict_local'
    ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = 'upload processing mode is invalid';
    END IF;
    IF TG_OP = 'UPDATE'
       AND (
            NEW.processing_mode IS DISTINCT FROM OLD.processing_mode
            OR NEW.processing_manifest IS DISTINCT FROM OLD.processing_manifest
            OR NEW.processing_schema_version
               IS DISTINCT FROM OLD.processing_schema_version
       ) THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = 'upload processing snapshot is immutable';
    END IF;
    RETURN NEW;
END;
$$
"""


def upgrade() -> None:
    """Make omitted processing choices private without rewriting saved choices."""
    op.execute(_PROMPT_ONLY_UPLOAD_FUNCTION)
    op.alter_column(
        "ai_summary_prompts",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default="prompt_only",
        existing_nullable=False,
    )
    op.alter_column(
        "uploaded_files",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default="prompt_only",
        existing_nullable=False,
    )
    op.alter_column(
        "user_llm_preferences",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default="prompt_only",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Restore the preceding server defaults without rewriting rows."""
    op.execute(_LEGACY_UPLOAD_FUNCTION)
    op.alter_column(
        "ai_summary_prompts",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default="cloud_assisted",
        existing_nullable=False,
    )
    op.alter_column(
        "uploaded_files",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default="cloud_assisted",
        existing_nullable=False,
    )
    op.alter_column(
        "user_llm_preferences",
        "processing_mode",
        existing_type=sa.String(length=32),
        server_default=None,
        existing_nullable=True,
    )
