from __future__ import annotations

import hashlib
import logging
import os
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.middleware.encryption import _get_key

logger = logging.getLogger(__name__)

# --- At-rest encryption for uploaded source files (CRYPTO-02 / SEC-PHI-02) -----
#
# Uploaded PHI documents are written to disk in a framed, chunked AES-256-GCM
# format so the bytes at rest never carry plaintext. AES-GCM needs the whole
# plaintext per encryption, so we encrypt PER CHUNK (one <=1 MiB plaintext chunk
# per frame) rather than the whole file — this composes with the streaming
# upload path (SEC-DOS-01) without ever buffering the full file in RAM.
#
# On-disk layout:
#   [magic header: b"MTENC1\n"]
#   then repeated frames, one per plaintext chunk:
#     [4-byte big-endian length N][N bytes: nonce(12) + ciphertext+tag]
#
# A file WITHOUT the magic header is treated as legacy plaintext (written before
# this feature) and read back verbatim — see ``decrypt_file``.

ENC_MAGIC = b"MTENC1\n"
_NONCE_LEN = 12
_LENGTH_PREFIX = 4  # big-endian uint32 frame length
_COPY_CHUNK = 1024 * 1024  # 1 MiB streaming chunk for legacy pass-through

# Log the legacy-plaintext warning once per process to avoid spamming when a
# directory still holds many pre-encryption files (run encrypt_existing_uploads.py).
_legacy_warned = False


def encrypt_chunk(plaintext: bytes) -> bytes:
    """Encrypt one plaintext chunk into a single length-prefixed frame.

    Frame = ``[4-byte big-endian len(nonce+ciphertext)][nonce(12)][ciphertext+tag]``.
    Each frame uses a fresh random nonce (AES-256-GCM via the database encryption
    key, matching ``encrypt_field`` semantics).
    """
    aesgcm = AESGCM(_get_key())
    nonce = os.urandom(_NONCE_LEN)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    payload = nonce + ciphertext
    return struct.pack(">I", len(payload)) + payload


def encrypt_stream(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Yield the magic header, then one encrypted frame per plaintext chunk.

    The caller streams ``chunks`` (each <=1 MiB of plaintext) so the whole file
    is never materialized in memory; this generator yields the magic header
    followed by the per-chunk frames, ready to be written straight to disk.
    """
    yield ENC_MAGIC
    for chunk in chunks:
        if not chunk:
            continue
        yield encrypt_chunk(chunk)


class EncryptedFileWriter:
    """Frame-by-frame encrypting writer for the streaming upload path.

    Wraps an open binary file object. The magic header is written lazily on the
    first ``write_chunk`` (or ``finalize``); each ``write_chunk`` encrypts a
    single plaintext chunk and writes its frame immediately, so only one chunk's
    worth of plaintext is ever held in memory.
    """

    def __init__(self, fileobj: BinaryIO) -> None:
        self._f = fileobj
        self._header_written = False

    def _ensure_header(self) -> None:
        if not self._header_written:
            self._f.write(ENC_MAGIC)
            self._header_written = True

    def write_chunk(self, plaintext: bytes) -> None:
        """Encrypt and write a single plaintext chunk as one frame."""
        self._ensure_header()
        if plaintext:
            self._f.write(encrypt_chunk(plaintext))

    def finalize(self) -> None:
        """Ensure the magic header is written even for an empty body."""
        self._ensure_header()


def is_encrypted_file(file_path: Path | str) -> bool:
    """Return True when the file begins with the encrypted-format magic header."""
    try:
        with open(file_path, "rb") as f:
            return f.read(len(ENC_MAGIC)) == ENC_MAGIC
    except OSError:
        return False


def _warn_legacy_once(file_path: Path | str) -> None:
    global _legacy_warned
    if not _legacy_warned:
        _legacy_warned = True
        logger.warning(
            "Reading legacy PLAINTEXT upload(s) at rest (no encryption header), "
            "e.g. %s. New uploads are encrypted; run "
            "scripts/encrypt_existing_uploads.py to encrypt existing files.",
            file_path,
        )


def decrypt_file(file_path: Path | str) -> bytes:
    """Read an uploaded file back as plaintext bytes.

    Detects the magic header: an encrypted file is decrypted frame-by-frame and
    concatenated; a file WITHOUT the header is legacy plaintext and returned
    verbatim (a one-time warning is logged). This lets the read path serve both
    pre-encryption files and new encrypted writes transparently.
    """
    with open(file_path, "rb") as f:
        magic = f.read(len(ENC_MAGIC))
        if magic != ENC_MAGIC:
            _warn_legacy_once(file_path)
            # Legacy plaintext: return the whole file unchanged (header bytes we
            # already consumed + the rest).
            return magic + f.read()

        aesgcm = AESGCM(_get_key())
        out = bytearray()
        while True:
            len_bytes = f.read(_LENGTH_PREFIX)
            if not len_bytes:
                break
            if len(len_bytes) != _LENGTH_PREFIX:
                raise ValueError("Corrupt encrypted file: truncated frame length")
            (frame_len,) = struct.unpack(">I", len_bytes)
            payload = f.read(frame_len)
            if len(payload) != frame_len:
                raise ValueError("Corrupt encrypted file: truncated frame payload")
            nonce, ciphertext = payload[:_NONCE_LEN], payload[_NONCE_LEN:]
            out += aesgcm.decrypt(nonce, ciphertext, None)
    return bytes(out)


def decrypt_file_to(src_path: Path | str, dst_path: Path | str) -> None:
    """Stream-decrypt an at-rest upload to a plaintext file with BOUNDED memory.

    Reads the encrypted ``src_path`` ONE frame at a time and writes the decrypted
    plaintext straight to ``dst_path``, so the whole file is never materialized in
    RAM — only a single frame's worth (<=1 MiB plaintext + GCM nonce/tag) is held
    at any moment. This is the OOM-safety property the structured-ingest path
    depends on: a 5 GB encrypted Epic export decrypts within the same bounded
    memory the streaming parsers (ijson / member-by-member zip) already assume.

    A source WITHOUT the magic header is legacy plaintext (written before
    CRYPTO-02) and is stream-COPIED through verbatim (1 MiB chunks), so the
    decrypt-to-temp entry works uniformly for both encrypted and legacy files.

    Unlike ``decrypt_file`` (which returns the whole plaintext as ``bytes``), this
    never builds a full in-memory buffer — use it for large structured archives.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        magic = src.read(len(ENC_MAGIC))
        if magic != ENC_MAGIC:
            _warn_legacy_once(src_path)
            # Legacy plaintext: emit the header bytes we already consumed, then
            # stream the remainder in bounded chunks (no whole-file buffering).
            if magic:
                dst.write(magic)
            while True:
                chunk = src.read(_COPY_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
            return

        aesgcm = AESGCM(_get_key())
        while True:
            len_bytes = src.read(_LENGTH_PREFIX)
            if not len_bytes:
                break
            if len(len_bytes) != _LENGTH_PREFIX:
                raise ValueError("Corrupt encrypted file: truncated frame length")
            (frame_len,) = struct.unpack(">I", len_bytes)
            payload = src.read(frame_len)
            if len(payload) != frame_len:
                raise ValueError("Corrupt encrypted file: truncated frame payload")
            nonce, ciphertext = payload[:_NONCE_LEN], payload[_NONCE_LEN:]
            dst.write(aesgcm.decrypt(nonce, ciphertext, None))


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_file_type(filename: str) -> str:
    """Detect the format type from a filename."""
    lower = filename.lower()
    if lower.endswith(".json"):
        return "fhir_r4"
    elif lower.endswith(".tsv"):
        return "epic_ehi"
    elif lower.endswith(".zip"):
        return "zip_archive"
    elif lower.endswith(".pdf"):
        return "pdf"
    elif lower.endswith((".png", ".jpg", ".jpeg", ".tiff")):
        return "image"
    return "unknown"
