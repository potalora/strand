"""Add the durable manual extraction flag for mixed-ZIP children.

Revision ID: b4c5d6e7f8a9
Revises: a9b0c1d2e3f4
Create Date: 2026-07-31
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4c5d6e7f8a9"
down_revision: Union[str, Sequence[str], None] = "a9b0c1d2e3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "uploaded_files",
        sa.Column(
            "manual_extraction_required",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE uploaded_files
        SET manual_extraction_required = true
        WHERE file_category = 'unstructured'
          AND ingestion_status IN ('staging_extraction', 'pending_extraction')
          AND (
            storage_path LIKE '%/medtimeline-zip-set-%/%'
            OR storage_path LIKE '%/.medtimeline-zip-set-%.pending/%'
          )
        """
    )


def downgrade() -> None:
    op.drop_column("uploaded_files", "manual_extraction_required")
