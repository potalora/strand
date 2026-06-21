"""SQLAlchemy ``TypeDecorator``s that encrypt column values at rest.

These wrap a ``LargeBinary`` (Postgres ``BYTEA``) column with the app-layer
AES-256-GCM helpers from :mod:`app.middleware.encryption`. Encryption happens on
the bind side (Python -> DB) and decryption on the result side (DB -> Python), so
the ORM-facing value is the plain ``dict``/``str`` and call sites never change.

- :class:`EncryptedJSON` — for JSON-shaped columns (``dict``/``list``). The
  value is ``json.dumps``'d before encryption and ``json.loads``'d after, so it
  reads back as the same structure. Replaces a ``JSONB`` column; note the data is
  opaque ciphertext at rest and therefore NOT SQL-queryable (these columns are
  fetch-and-render only).
- :class:`EncryptedText` — for free-text columns (``str``). Replaces a ``Text``
  column.

``None`` passes through unchanged in both directions so nullable columns keep
working.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import LargeBinary
from sqlalchemy.types import TypeDecorator

from app.middleware.encryption import decrypt_field, encrypt_field


class EncryptedJSON(TypeDecorator):
    """Encrypt a JSON-serializable value (dict/list) at rest as BYTEA."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> bytes | None:
        if value is None:
            return None
        return encrypt_field(json.dumps(value))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return json.loads(decrypt_field(bytes(value)))


class EncryptedText(TypeDecorator):
    """Encrypt a free-text string value at rest as BYTEA."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> bytes | None:
        if value is None:
            return None
        return encrypt_field(value)

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return decrypt_field(bytes(value))
