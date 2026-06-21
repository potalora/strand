"""At-rest encryption for STRUCTURED uploads (CRYPTO-02 / SEC-PHI-02, issue #54).

CRYPTO-02 already encrypts UNSTRUCTURED uploads (PDF/RTF/TIFF) at rest via the
framed ``MTENC1`` AES-256-GCM format. STRUCTURED uploads (FHIR JSON / Epic EHI
ZIP / standalone CDA XML / XDM ZIP) historically wrote PLAINTEXT because the
ingestion coordinator reads them via streaming (``ijson`` for big bundles,
member-by-member ``zipfile`` extraction with the W9 zip-bomb caps).

This suite covers the decrypt-to-temp approach: the coordinator stream-decrypts
an encrypted source to a temp plaintext file at the ingest entry, runs ALL the
existing streaming logic (including the W9 caps) on that temp, and keeps the
ORIGINAL encrypted path as ``storage_path`` so future reads decrypt again.

Key OOM-safety property: ``decrypt_file_to`` streams the decrypt frame-by-frame
(never loading the whole file into memory), so a 5 GB file decrypts with bounded
memory — preserving the W9 streaming guarantees.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch

from app.models.record import HealthRecord
from app.models.uploaded_file import UploadedFile
from app.utils.file_utils import (
    ENC_MAGIC,
    EncryptedFileWriter,
    compute_file_hash,
    decrypt_file,
    is_encrypted_file,
)
from tests.conftest import FIXTURES_DIR, auth_headers

PLAINTEXT_MARKER = b"SECRET_PHI_MARKER_Jane_Q_Public_MRN_0001234567"

# The dedup scan runs in a fire-and-forget background task on its OWN session
# (the real DB, not the test session). Patch it so the full ingest_file path
# stays self-contained and deterministic in tests.
PATCH_DEDUP_BG = "app.services.ingestion.coordinator._run_dedup_background"


def _encrypt_bytes_to(path: Path, data: bytes, chunk: int = 1024 * 1024) -> None:
    """Write ``data`` to ``path`` in the framed MTENC1 encrypted format."""
    with open(path, "wb") as f:
        writer = EncryptedFileWriter(f)
        for i in range(0, len(data), chunk):
            writer.write_chunk(data[i : i + chunk])
        writer.finalize()


def _make_payload(total: int) -> bytes:
    """A payload embedding a known plaintext marker, padded to ``total`` bytes."""
    body = bytearray()
    while len(body) < total:
        body.extend(PLAINTEXT_MARKER)
        body.extend(bytes(range(256)))
    return bytes(body[:total])


def _sample_bundle_bytes() -> bytes:
    return (FIXTURES_DIR / "sample_fhir_bundle.json").read_bytes()


def _make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# decrypt_file_to — streaming, bounded-memory decrypt-to-file
# ---------------------------------------------------------------------------


def test_decrypt_file_to_roundtrips_multiframe_and_source_stays_ciphertext(tmp_path):
    """A >2 MiB multi-frame file decrypts exactly; the source stays ciphertext."""
    from app.utils.file_utils import decrypt_file_to

    data = _make_payload(3 * 1024 * 1024 + 777)  # spans 4 frames at 1 MiB
    src = tmp_path / "src.enc"
    dst = tmp_path / "dst.plain"
    _encrypt_bytes_to(src, data)

    decrypt_file_to(src, dst)

    # Decrypted output matches the original plaintext exactly.
    assert dst.read_bytes() == data
    # The on-disk SOURCE is still ciphertext (header present, marker absent).
    on_disk = src.read_bytes()
    assert on_disk.startswith(ENC_MAGIC)
    assert PLAINTEXT_MARKER not in on_disk
    # Cross-check against the whole-file decrypt helper.
    assert dst.read_bytes() == decrypt_file(src)


def test_decrypt_file_to_passes_through_legacy_plaintext(tmp_path):
    """A source WITHOUT the magic header is copied through verbatim."""
    from app.utils.file_utils import decrypt_file_to

    original = b'{"resourceType": "Bundle", "entry": []}' + b"x" * 5000
    src = tmp_path / "legacy.json"
    dst = tmp_path / "out.json"
    src.write_bytes(original)

    decrypt_file_to(src, dst)

    assert dst.read_bytes() == original
    assert not is_encrypted_file(src)


def test_same_content_encrypted_twice_yields_same_plaintext_hash(tmp_path):
    """Different nonces ⇒ different ciphertext on disk, but identical plaintext hash."""
    from app.utils.file_utils import decrypt_file_to

    data = _sample_bundle_bytes()
    enc_a = tmp_path / "a.json"
    enc_b = tmp_path / "b.json"
    _encrypt_bytes_to(enc_a, data)
    _encrypt_bytes_to(enc_b, data)

    # The on-disk encrypted bytes differ (random per-frame nonces)...
    assert enc_a.read_bytes() != enc_b.read_bytes()

    out_a = tmp_path / "a.plain"
    out_b = tmp_path / "b.plain"
    decrypt_file_to(enc_a, out_a)
    decrypt_file_to(enc_b, out_b)

    # ...but the decrypted plaintext hashes match each other and the raw bytes.
    plaintext_hash = hashlib.sha256(data).hexdigest()
    assert compute_file_hash(out_a) == plaintext_hash
    assert compute_file_hash(out_b) == plaintext_hash


# ---------------------------------------------------------------------------
# coordinator.ingest_file — decrypt-to-temp at the ingest entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encrypted_fhir_bundle_ingests_equal_to_plaintext(
    client: AsyncClient, db_session: AsyncSession, tmp_path
):
    """An ENCRYPTED FHIR bundle ingests the same records as the plaintext one."""
    from app.services.ingestion.coordinator import ingest_file

    data = _sample_bundle_bytes()

    # Two users so the idempotent inserter doesn't collapse the second (identical)
    # bundle as "unchanged".
    _, uid_plain = await auth_headers(client, "plain@example.com")
    _, uid_enc = await auth_headers(client, "enc@example.com")

    plain_path = tmp_path / "plain.json"
    plain_path.write_bytes(data)
    enc_path = tmp_path / "enc.json"
    _encrypt_bytes_to(enc_path, data)
    assert is_encrypted_file(enc_path)

    with patch(PATCH_DEDUP_BG, new_callable=AsyncMock):
        plain_result = await ingest_file(
            db=db_session, user_id=UUID(uid_plain),
            file_path=plain_path, original_filename="bundle.json",
        )
        enc_result = await ingest_file(
            db=db_session, user_id=UUID(uid_enc),
            file_path=enc_path, original_filename="bundle.json",
        )

    assert plain_result["records_inserted"] > 0
    assert enc_result["records_inserted"] == plain_result["records_inserted"]

    # The encrypted upload's stored path is the ORIGINAL ciphertext file (so
    # future reads decrypt again), and that file is still encrypted on disk.
    enc_row = (
        await db_session.execute(
            select(UploadedFile).where(UploadedFile.user_id == UUID(uid_enc))
        )
    ).scalar_one()
    assert enc_row.storage_path == str(enc_path)
    assert is_encrypted_file(Path(enc_row.storage_path))

    # Records actually landed for the encrypted user.
    enc_records = (
        await db_session.execute(
            select(HealthRecord).where(HealthRecord.user_id == UUID(uid_enc))
        )
    ).scalars().all()
    assert len(enc_records) == enc_result["records_inserted"]


@pytest.mark.asyncio
async def test_encrypted_zip_bomb_still_rejected_by_w9_caps(
    client: AsyncClient, db_session: AsyncSession, monkeypatch, tmp_path
):
    """The W9 zip-bomb caps fire on the DECRYPTED temp of an encrypted ZIP."""
    import app.services.ingestion.coordinator as coord
    from app.services.ingestion.coordinator import ingest_file

    # Tighten the uncompressed budget so the bomb trips it deterministically.
    monkeypatch.setattr(coord, "_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 1024 * 1024, raising=False)

    _, uid = await auth_headers(client, "zipbomb@example.com")

    bomb = _make_zip({"bomb.txt": b"\x00" * (5 * 1024 * 1024)})  # 5 MiB of zeros
    enc_zip = tmp_path / "bomb.zip"
    _encrypt_bytes_to(enc_zip, bomb)
    assert is_encrypted_file(enc_zip)

    # An HTTPException (413/400) — NOT a BadZipFile — proves the source was
    # decrypted first (else zipfile couldn't open it) AND the W9 cap fired on the
    # decrypted content.
    with pytest.raises(HTTPException) as exc:
        await ingest_file(
            db=db_session, user_id=UUID(uid),
            file_path=enc_zip, original_filename="bomb.zip",
        )
    assert exc.value.status_code in (400, 413)


@pytest.mark.asyncio
async def test_legacy_plaintext_fhir_still_ingests(
    client: AsyncClient, db_session: AsyncSession, tmp_path
):
    """A LEGACY plaintext structured file (no magic header) still ingests."""
    from app.services.ingestion.coordinator import ingest_file

    _, uid = await auth_headers(client, "legacy@example.com")

    plain_path = tmp_path / "legacy.json"
    plain_path.write_bytes(_sample_bundle_bytes())
    assert not is_encrypted_file(plain_path)

    with patch(PATCH_DEDUP_BG, new_callable=AsyncMock):
        result = await ingest_file(
            db=db_session, user_id=UUID(uid),
            file_path=plain_path, original_filename="legacy.json",
        )

    assert result["records_inserted"] > 0
    # storage_path is the plaintext original (no temp involved for legacy files).
    row = (
        await db_session.execute(
            select(UploadedFile).where(UploadedFile.user_id == UUID(uid))
        )
    ).scalar_one()
    assert row.storage_path == str(plain_path)


@pytest.mark.asyncio
async def test_reupload_dedup_hash_matches_for_encrypted_copies(
    client: AsyncClient, db_session: AsyncSession, tmp_path
):
    """The dedup file_hash is computed on PLAINTEXT, so two encrypted copies match."""
    from app.services.ingestion.coordinator import ingest_file

    data = _sample_bundle_bytes()
    plaintext_hash = hashlib.sha256(data).hexdigest()

    _, uid_a = await auth_headers(client, "copyA@example.com")
    _, uid_b = await auth_headers(client, "copyB@example.com")

    enc_a = tmp_path / "a.json"
    enc_b = tmp_path / "b.json"
    _encrypt_bytes_to(enc_a, data)
    _encrypt_bytes_to(enc_b, data)
    # Different nonces ⇒ the encrypted files differ byte-for-byte on disk.
    assert enc_a.read_bytes() != enc_b.read_bytes()

    with patch(PATCH_DEDUP_BG, new_callable=AsyncMock):
        await ingest_file(
            db=db_session, user_id=UUID(uid_a),
            file_path=enc_a, original_filename="bundle.json",
        )
        await ingest_file(
            db=db_session, user_id=UUID(uid_b),
            file_path=enc_b, original_filename="bundle.json",
        )

    row_a = (
        await db_session.execute(
            select(UploadedFile).where(UploadedFile.user_id == UUID(uid_a))
        )
    ).scalar_one()
    row_b = (
        await db_session.execute(
            select(UploadedFile).where(UploadedFile.user_id == UUID(uid_b))
        )
    ).scalar_one()

    assert row_a.file_hash == plaintext_hash
    assert row_b.file_hash == plaintext_hash
    assert row_a.file_hash == row_b.file_hash
