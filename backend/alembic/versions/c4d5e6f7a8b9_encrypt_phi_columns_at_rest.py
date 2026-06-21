"""Encrypt clinical PHI columns at rest + email blind index (W7/CRYPTO-01, W17).

Converts the following plaintext columns to AES-256-GCM ciphertext stored as
``BYTEA`` (backed by the ``EncryptedJSON`` / ``EncryptedText`` TypeDecorators),
data-migrating existing rows in place (read plaintext -> encrypt -> write):

- health_records.fhir_resource            (JSONB -> BYTEA, EncryptedJSON)
- uploaded_files.extracted_text           (TEXT  -> BYTEA, EncryptedText)
- uploaded_files.extraction_entities      (JSONB -> BYTEA, EncryptedJSON)
- uploaded_files.extraction_sections      (JSONB -> BYTEA, EncryptedJSON)
- uploaded_files.document_metadata        (JSONB -> BYTEA, EncryptedJSON)
- ai_summary_prompts.system_prompt        (TEXT  -> BYTEA, EncryptedText)
- ai_summary_prompts.user_prompt          (TEXT  -> BYTEA, EncryptedText)
- ai_summary_prompts.response_text        (TEXT  -> BYTEA, EncryptedText)
- users.email                             (TEXT  -> BYTEA, EncryptedText)

Because GCM uses a random nonce per value the ciphertext is non-deterministic and
cannot back an equality lookup, so ``users.email`` loses its UNIQUE constraint and
a deterministic ``users.email_hmac`` (HMAC-SHA256 blind index) is added, backfilled
and made UNIQUE NOT NULL to preserve login / uniqueness.

The encryption is done in Python (the AES-GCM transform can't run in a SQL
``USING`` clause), so each column is migrated via a temp BYTEA column that is then
swapped in. Requires ``DATABASE_ENCRYPTION_KEY`` to be configured at migration
time (already mandatory for the app).

Revision ID: c4d5e6f7a8b9
Revises: b1c2d3e4f5a6
Create Date: 2026-06-21
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

from app.middleware.encryption import blind_index, decrypt_field, encrypt_field

# revision identifiers, used by Alembic.
revision = "c4d5e6f7a8b9"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


# (table, column, is_json, not_null)
_COLUMNS = [
    ("health_records", "fhir_resource", True, True),
    ("uploaded_files", "extracted_text", False, False),
    ("uploaded_files", "extraction_entities", True, False),
    ("uploaded_files", "extraction_sections", True, False),
    ("uploaded_files", "document_metadata", True, False),
    ("ai_summary_prompts", "system_prompt", False, True),
    ("ai_summary_prompts", "user_prompt", False, True),
    ("ai_summary_prompts", "response_text", False, False),
    ("users", "email", False, True),
]


def _encrypt_column(conn, table: str, col: str, is_json: bool, not_null: bool) -> None:
    """Migrate one plaintext column to an encrypted BYTEA column in place."""
    tmp = f"{col}__enc_tmp"
    op.add_column(table, sa.Column(tmp, sa.LargeBinary(), nullable=True))

    rows = conn.execute(sa.text(f"SELECT id, {col} AS val FROM {table}")).fetchall()
    for row in rows:
        value = row.val
        if value is None:
            continue
        plaintext = json.dumps(value) if is_json else str(value)
        conn.execute(
            sa.text(f"UPDATE {table} SET {tmp} = :v WHERE id = :id").bindparams(
                sa.bindparam("v", type_=sa.LargeBinary())
            ),
            {"v": encrypt_field(plaintext), "id": row.id},
        )

    op.drop_column(table, col)
    op.alter_column(table, tmp, new_column_name=col)
    if not_null:
        op.alter_column(table, col, existing_type=sa.LargeBinary(), nullable=False)


def _decrypt_column(conn, table: str, col: str, is_json: bool, not_null: bool) -> None:
    """Reverse: migrate an encrypted BYTEA column back to plaintext."""
    sa_type = sa.dialects.postgresql.JSONB() if is_json else sa.Text()
    tmp = f"{col}__plain_tmp"
    op.add_column(table, sa.Column(tmp, sa_type, nullable=True))

    rows = conn.execute(sa.text(f"SELECT id, {col} AS val FROM {table}")).fetchall()
    for row in rows:
        value = row.val
        if value is None:
            continue
        plaintext = decrypt_field(bytes(value))
        restored = json.loads(plaintext) if is_json else plaintext
        conn.execute(
            sa.text(f"UPDATE {table} SET {tmp} = :v WHERE id = :id").bindparams(
                sa.bindparam("v", type_=sa_type)
            ),
            {"v": restored, "id": row.id},
        )

    op.drop_column(table, col)
    op.alter_column(table, tmp, new_column_name=col)
    if not_null:
        op.alter_column(table, col, existing_type=sa_type, nullable=False)


def upgrade() -> None:
    conn = op.get_bind()

    # Email is encrypted -> its UNIQUE constraint can no longer enforce
    # uniqueness on randomized ciphertext; move it to the blind index.
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_email_key")

    # Add the email blind index (nullable for backfill, then UNIQUE NOT NULL).
    op.add_column("users", sa.Column("email_hmac", sa.String(length=64), nullable=True))
    for row in conn.execute(sa.text("SELECT id, email FROM users")).fetchall():
        if row.email is None:
            continue
        conn.execute(
            sa.text("UPDATE users SET email_hmac = :h WHERE id = :id"),
            {"h": blind_index(str(row.email)), "id": row.id},
        )
    op.alter_column("users", "email_hmac", existing_type=sa.String(length=64), nullable=False)
    op.create_index("ix_users_email_hmac", "users", ["email_hmac"], unique=True)

    # Encrypt each PHI column in place.
    for table, col, is_json, not_null in _COLUMNS:
        _encrypt_column(conn, table, col, is_json, not_null)


def downgrade() -> None:
    conn = op.get_bind()

    for table, col, is_json, not_null in reversed(_COLUMNS):
        _decrypt_column(conn, table, col, is_json, not_null)

    op.drop_index("ix_users_email_hmac", table_name="users")
    op.drop_column("users", "email_hmac")
    op.create_unique_constraint("users_email_key", "users", ["email"])
