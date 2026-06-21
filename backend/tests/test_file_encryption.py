"""Tests for at-rest encryption of uploaded source files (CRYPTO-02 / SEC-PHI-02).

Uploaded files (PDF/TIFF/RTF) are written to disk through a framed, chunked
AES-256-GCM format so the on-disk bytes never carry plaintext PHI, while the
write path stays streaming (one <=1 MiB plaintext chunk encrypted at a time —
never the whole file buffered in RAM). Legacy plaintext files (written before
this feature) must still be readable: the read helper detects the magic header
and falls back to a verbatim plaintext read when it is absent.
"""

from __future__ import annotations

import hashlib
import io

import pytest

from app.utils.file_utils import (
    ENC_MAGIC,
    EncryptedFileWriter,
    decrypt_file,
    encrypt_chunk,
    encrypt_stream,
    is_encrypted_file,
)

PLAINTEXT_MARKER = b"SECRET_PHI_MARKER_Jane_Q_Public_MRN_0001234567"


def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _make_payload(total: int) -> bytes:
    """A payload that embeds a known plaintext marker and exceeds ``total`` bytes."""
    body = bytearray()
    while len(body) < total:
        body.extend(PLAINTEXT_MARKER)
        body.extend(bytes(range(256)))
    return bytes(body[:total])


# ---------------------------------------------------------------------------
# Framed format round-trip
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_multichunk_roundtrip(tmp_path):
    """A >2 MiB payload spanning multiple 1 MiB frames round-trips exactly."""
    data = _make_payload(3 * 1024 * 1024 + 777)  # spans 4 frames at 1 MiB chunks
    dest = tmp_path / "note.enc"

    with open(dest, "wb") as f:
        for blob in encrypt_stream(_chunks(data, 1024 * 1024)):
            f.write(blob)

    assert decrypt_file(dest) == data


def test_encrypted_disk_bytes_hide_plaintext_marker(tmp_path):
    """The bytes written to disk must not contain the known plaintext marker."""
    data = _make_payload(2 * 1024 * 1024 + 13)
    dest = tmp_path / "note.enc"

    with open(dest, "wb") as f:
        for blob in encrypt_stream(_chunks(data, 1024 * 1024)):
            f.write(blob)

    on_disk = dest.read_bytes()
    assert PLAINTEXT_MARKER not in on_disk
    assert on_disk != data
    # Sanity: round-trips back to the marker-bearing plaintext.
    assert PLAINTEXT_MARKER in decrypt_file(dest)


def test_encrypted_file_has_magic_header(tmp_path):
    """Encrypted files start with the magic header; legacy files do not."""
    enc = tmp_path / "enc.bin"
    with open(enc, "wb") as f:
        for blob in encrypt_stream(_chunks(b"hello world", 4)):
            f.write(blob)
    assert enc.read_bytes().startswith(ENC_MAGIC)
    assert is_encrypted_file(enc) is True

    legacy = tmp_path / "legacy.bin"
    legacy.write_bytes(b"%PDF-1.4 plaintext content")
    assert is_encrypted_file(legacy) is False


def test_decrypt_file_reads_legacy_plaintext_unchanged(tmp_path):
    """A file without the magic header is treated as legacy plaintext, read as-is."""
    legacy = tmp_path / "legacy.rtf"
    original = rb"{\rtf1\ansi Patient visit note with PHI.}" + b"x" * 5000
    legacy.write_bytes(original)

    assert decrypt_file(legacy) == original


def test_encrypt_chunk_is_a_single_self_describing_frame(tmp_path):
    """encrypt_chunk emits one length-prefixed frame that decrypts in isolation."""
    import struct

    frame = encrypt_chunk(b"abc123")
    # [4-byte length][nonce(12)][ciphertext+tag] — length covers nonce+ct.
    (declared,) = struct.unpack(">I", frame[:4])
    assert declared == len(frame) - 4
    # Write magic + this frame and confirm the file decrypts.
    dest = tmp_path / "one.enc"
    dest.write_bytes(ENC_MAGIC + frame)
    assert decrypt_file(dest) == b"abc123"


def test_empty_payload_round_trips(tmp_path):
    """An empty body still produces a valid (header-only) encrypted file."""
    dest = tmp_path / "empty.enc"
    with open(dest, "wb") as f:
        writer = EncryptedFileWriter(f)
        writer.finalize()
    assert is_encrypted_file(dest)
    assert decrypt_file(dest) == b""


# ---------------------------------------------------------------------------
# Streaming writer (frame-by-frame, no whole-file buffering)
# ---------------------------------------------------------------------------


def test_streaming_writer_round_trips_and_frames_incrementally(tmp_path):
    """EncryptedFileWriter encrypts each chunk into its own frame as it streams."""
    import struct

    data = _make_payload(3 * 1024 * 1024)  # 3 chunks of 1 MiB
    dest = tmp_path / "stream.enc"

    with open(dest, "wb") as f:
        writer = EncryptedFileWriter(f)
        for chunk in _chunks(data, 1024 * 1024):
            writer.write_chunk(chunk)
        writer.finalize()

    raw = dest.read_bytes()
    assert raw.startswith(ENC_MAGIC)

    # Count frames by walking the length prefixes — must be one per 1 MiB chunk.
    pos = len(ENC_MAGIC)
    frames = 0
    while pos < len(raw):
        (n,) = struct.unpack(">I", raw[pos : pos + 4])
        pos += 4 + n
        frames += 1
    assert frames == 3
    assert decrypt_file(dest) == data


# ---------------------------------------------------------------------------
# _stream_upload_to_disk integration (write path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_upload_to_disk_encrypts_and_returns_plaintext_hash(tmp_path):
    """Encrypt mode writes a framed file, returns the PLAINTEXT size/header/hash."""
    from app.api.upload import _stream_upload_to_disk
    from starlette.datastructures import UploadFile as StarletteUploadFile

    data = b"%PDF-1.4\n" + PLAINTEXT_MARKER + b"y" * (2 * 1024 * 1024)
    uf = StarletteUploadFile(file=io.BytesIO(data), filename="x.pdf")
    dest = tmp_path / "x.pdf"

    total, header, digest = await _stream_upload_to_disk(
        uf, dest, max_bytes=10 * 1024 * 1024, encrypt=True
    )

    assert total == len(data)
    assert header == data[:16]
    assert digest == hashlib.sha256(data).hexdigest()
    # On disk: encrypted (no plaintext marker), but decrypts to the original.
    assert is_encrypted_file(dest)
    assert PLAINTEXT_MARKER not in dest.read_bytes()
    assert decrypt_file(dest) == data


@pytest.mark.asyncio
async def test_stream_upload_to_disk_plaintext_mode_for_structured(tmp_path):
    """Plaintext mode (structured ingestion) leaves the bytes on disk verbatim."""
    from app.api.upload import _stream_upload_to_disk
    from starlette.datastructures import UploadFile as StarletteUploadFile

    data = b'{"resourceType": "Bundle", "entry": []}'
    uf = StarletteUploadFile(file=io.BytesIO(data), filename="bundle.json")
    dest = tmp_path / "bundle.json"

    total, header, digest = await _stream_upload_to_disk(
        uf, dest, max_bytes=10 * 1024 * 1024, encrypt=False
    )

    assert total == len(data)
    assert dest.read_bytes() == data
    assert not is_encrypted_file(dest)
    assert digest == hashlib.sha256(data).hexdigest()


@pytest.mark.asyncio
async def test_stream_upload_to_disk_encrypt_aborts_over_budget(tmp_path):
    """Over-budget body aborts mid-stream and unlinks the partial encrypted file."""
    from fastapi import HTTPException
    from app.api.upload import _stream_upload_to_disk
    from starlette.datastructures import UploadFile as StarletteUploadFile

    body = b"\x00" * (3 * 1024 * 1024)  # 3 MiB
    uf = StarletteUploadFile(file=io.BytesIO(body), filename="x.bin")
    dest = tmp_path / "x.bin"

    with pytest.raises(HTTPException) as exc:
        await _stream_upload_to_disk(uf, dest, max_bytes=1024 * 1024, encrypt=True)
    assert exc.value.status_code == 413
    assert not dest.exists()


# ---------------------------------------------------------------------------
# Read-side wiring: text_extractor decrypts (and tolerates legacy plaintext)
# ---------------------------------------------------------------------------


def test_text_extractor_rtf_reads_encrypted_file(tmp_path):
    """extract_text_from_rtf transparently decrypts an encrypted RTF on disk."""
    from app.services.extraction.text_extractor import extract_text_from_rtf

    rtf = rb"{\rtf1\ansi Encrypted clinical note body.}"
    dest = tmp_path / "note.rtf"
    with open(dest, "wb") as f:
        writer = EncryptedFileWriter(f)
        writer.write_chunk(rtf)
        writer.finalize()

    text = extract_text_from_rtf(dest)
    assert "Encrypted clinical note body." in text


def test_text_extractor_rtf_reads_legacy_plaintext(tmp_path):
    """A legacy plaintext RTF (no magic header) still extracts unchanged."""
    from app.services.extraction.text_extractor import extract_text_from_rtf

    dest = tmp_path / "legacy.rtf"
    dest.write_bytes(rb"{\rtf1\ansi Legacy plaintext note.}")

    text = extract_text_from_rtf(dest)
    assert "Legacy plaintext note." in text


# ---------------------------------------------------------------------------
# Re-encryption migration script
# ---------------------------------------------------------------------------


def test_encrypt_existing_uploads_script_is_idempotent(tmp_path):
    """The migration encrypts legacy files in place and skips already-encrypted ones."""
    from scripts.encrypt_existing_uploads import encrypt_file_in_place

    legacy = tmp_path / "legacy.pdf"
    original = b"%PDF-1.4\n" + PLAINTEXT_MARKER + b"z" * 4096
    legacy.write_bytes(original)

    # First pass encrypts in place.
    changed = encrypt_file_in_place(legacy)
    assert changed is True
    assert is_encrypted_file(legacy)
    assert PLAINTEXT_MARKER not in legacy.read_bytes()
    assert decrypt_file(legacy) == original

    # Second pass is a no-op (already encrypted).
    changed_again = encrypt_file_in_place(legacy)
    assert changed_again is False
    assert decrypt_file(legacy) == original
