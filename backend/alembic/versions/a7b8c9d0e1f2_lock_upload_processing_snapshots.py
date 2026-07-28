"""Lock uploaded-file processing snapshots at the database boundary.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-27
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE FUNCTION enforce_uploaded_file_processing_snapshot()
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
    )
    op.execute(
        """
        CREATE TRIGGER trg_uploaded_files_immutable_processing_snapshot
        BEFORE INSERT OR UPDATE
        ON uploaded_files
        FOR EACH ROW
        EXECUTE FUNCTION enforce_uploaded_file_processing_snapshot()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_uploaded_files_immutable_processing_snapshot "
        "ON uploaded_files"
    )
    op.execute("DROP FUNCTION IF EXISTS enforce_uploaded_file_processing_snapshot()")
