"""Rename the encrypted account email fields to a general login identifier.

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-08-23
"""

from __future__ import annotations

import hmac

import sqlalchemy as sa
from alembic import op
from cryptography.exceptions import InvalidTag
from pydantic import EmailStr, TypeAdapter, ValidationError

from app.middleware.encryption import blind_index, decrypt_field

# revision identifiers, used by Alembic.
revision = "d6e7f8a9b0c1"
down_revision = "c5d6e7f8a9b0"
branch_labels = None
depends_on = None

DOWNGRADE_BLOCKED = (
    "Downgrade blocked: account identifiers are not compatible with the legacy schema."
)
_EMAIL_ADAPTER = TypeAdapter(EmailStr)


def upgrade() -> None:
    op.alter_column("users", "email", new_column_name="login_identifier")
    op.alter_column("users", "email_hmac", new_column_name="login_identifier_hmac")
    op.execute(
        "ALTER INDEX ix_users_email_hmac RENAME TO ix_users_login_identifier_hmac"
    )


def _assert_legacy_email_compatible() -> None:
    connection = op.get_bind()
    connection.execute(sa.text("LOCK TABLE users IN SHARE ROW EXCLUSIVE MODE"))
    encrypted_identifiers = connection.execute(
        sa.text("SELECT login_identifier, login_identifier_hmac FROM users")
    )
    try:
        for ciphertext, stored_hmac in encrypted_identifiers:
            plaintext = decrypt_field(bytes(ciphertext))
            legacy_email = str(_EMAIL_ADAPTER.validate_python(plaintext))
            if not hmac.compare_digest(blind_index(legacy_email), stored_hmac):
                raise RuntimeError(DOWNGRADE_BLOCKED)
    except (InvalidTag, RuntimeError, ValueError, ValidationError):
        raise RuntimeError(DOWNGRADE_BLOCKED) from None


def downgrade() -> None:
    _assert_legacy_email_compatible()
    op.execute(
        "ALTER INDEX ix_users_login_identifier_hmac RENAME TO ix_users_email_hmac"
    )
    op.alter_column("users", "login_identifier_hmac", new_column_name="email_hmac")
    op.alter_column("users", "login_identifier", new_column_name="email")
