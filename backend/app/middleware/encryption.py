from __future__ import annotations

import hashlib
import hmac
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.config import settings


def _get_key() -> bytes:
    """Derive the AES-256 key from the configured encryption key."""
    key_hex = settings.database_encryption_key
    if not key_hex:
        raise RuntimeError("DATABASE_ENCRYPTION_KEY is not configured")
    return bytes.fromhex(key_hex)[:32]


def encrypt_field(plaintext: str) -> bytes:
    """Encrypt a string field using AES-256-GCM. Returns nonce + ciphertext."""
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext


def decrypt_field(data: bytes) -> str:
    """Decrypt AES-256-GCM encrypted data. Expects nonce (12 bytes) + ciphertext."""
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = data[:12]
    ciphertext = data[12:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def hash_value(value: str) -> str:
    """Create a SHA-256 hash for deduplication checks."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def blind_index(value: str) -> str:
    """Deterministic, keyed lookup token for an encrypted column.

    Encrypting a column with AES-256-GCM uses a random nonce, so the ciphertext
    is non-deterministic and cannot back an equality lookup (e.g. login by
    email). A blind index is an HMAC-SHA256 of the *normalized* plaintext keyed
    off the database encryption key: deterministic for the same input (so it can
    be indexed and compared) but not reversible without the key.

    The input is lowercased and stripped so lookups are stable regardless of
    surrounding whitespace or case. Returns a 64-char lowercase hex digest.
    """
    normalized = value.strip().lower().encode("utf-8")
    return hmac.new(_get_key(), normalized, hashlib.sha256).hexdigest()
