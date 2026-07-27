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


def test_mixed_zip_empty_child_still_finalizes_encrypted_header(tmp_path: Path) -> None:
    import app.services.ingestion.coordinator as coord

    source = tmp_path / "empty.pdf"
    source.write_bytes(b"")
    destination = tmp_path / "durable.pdf"

    coord._encrypt_zip_child(source, destination)

    assert destination.read_bytes() == ENC_MAGIC
    assert decrypt_file(destination) == b""
    assert destination.stat().st_mode & 0o777 == 0o600


def test_mixed_zip_exclusive_create_failure_preserves_preexisting_file(
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coord

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\nPHI")
    destination = tmp_path / "durable.pdf"
    destination.write_bytes(b"KEEP")

    with pytest.raises(FileExistsError):
        coord._encrypt_zip_child(source, destination)

    assert destination.read_bytes() == b"KEEP"


def test_mixed_zip_failure_has_no_public_check_then_unlink_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed writer never exposes a partial at the public destination name."""
    import app.services.ingestion.coordinator as coord

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\nPHI")
    upload_dir = tmp_path / "uploads"
    destination = upload_dir / "durable.pdf"
    real_lstat = Path.lstat
    public_cleanup_window_seen = False

    def swap_after_public_identity_check(path: Path):
        nonlocal public_cleanup_window_seen
        info = real_lstat(path)
        if path == destination:
            public_cleanup_window_seen = True
            path.rename(upload_dir / "attacker-held-partial.pdf")
            path.write_bytes(b"KEEP")
        return info

    class FailingWriter:
        def __init__(self, _fileobj: object) -> None:
            pass

        def write_chunk(self, _plaintext: bytes) -> None:
            raise OSError("injected writer failure")

        def finalize(self) -> None:
            raise AssertionError("finalize must not run")

    monkeypatch.setattr(Path, "lstat", swap_after_public_identity_check)
    monkeypatch.setattr(coord, "EncryptedFileWriter", FailingWriter)

    with pytest.raises(OSError, match="writer failure"):
        coord._encrypt_zip_child(source, destination)

    assert public_cleanup_window_seen is False
    assert not destination.exists()
    assert list(upload_dir.iterdir()) == []


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


@pytest.mark.asyncio
async def test_encrypted_mixed_zip_child_is_ciphertext_and_inherits_parent_policy(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A PDF child never becomes durable plaintext and keeps its parent's policy."""
    import app.services.ingestion.coordinator as coord
    from app.services.ingestion.coordinator import ingest_file

    _, uid_text = await auth_headers(client, "mixed-zip@example.com")
    user_id = UUID(uid_text)
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(coord.settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(coord.settings, "temp_extract_dir", str(tmp_path / "extract"))

    pdf = b"%PDF-1.7\n" + PLAINTEXT_MARKER + b"\n%%EOF"
    encrypted_zip = tmp_path / "mixed.zip"
    _encrypt_bytes_to(encrypted_zip, _make_zip({"clinical-note.pdf": pdf}))

    original_ingest_zip = coord._ingest_zip
    manifest = {"manifest_digest": "d" * 64, "pack_revision": "test-pack"}

    async def stamp_parent_then_ingest(
        db: AsyncSession,
        scoped_user_id: UUID,
        patient_id: UUID,
        upload_id: UUID,
        zip_path: Path,
    ) -> dict:
        parent = (
            await db.execute(
                select(UploadedFile).where(
                    UploadedFile.id == upload_id,
                    UploadedFile.user_id == scoped_user_id,
                )
            )
        ).scalar_one()
        parent.processing_mode = "validated_strict_local"
        parent.processing_manifest = manifest
        await db.commit()
        return await original_ingest_zip(
            db,
            scoped_user_id,
            patient_id,
            upload_id,
            zip_path,
        )

    monkeypatch.setattr(coord, "_ingest_zip", stamp_parent_then_ingest)

    with patch(PATCH_DEDUP_BG, new_callable=AsyncMock):
        result = await ingest_file(
            db=db_session,
            user_id=user_id,
            file_path=encrypted_zip,
            original_filename="mixed.zip",
        )

    assert len(result["unstructured_uploads"]) == 1
    rows = (
        await db_session.execute(
            select(UploadedFile).where(UploadedFile.user_id == user_id)
        )
    ).scalars().all()
    child = next(row for row in rows if row.file_category == "unstructured")
    durable_bytes = Path(child.storage_path).read_bytes()
    assert durable_bytes.startswith(ENC_MAGIC)
    assert PLAINTEXT_MARKER not in durable_bytes
    assert decrypt_file(child.storage_path) == pdf
    assert child.file_size_bytes == len(pdf)
    assert child.file_hash == hashlib.sha256(pdf).hexdigest()
    assert child.processing_mode == "validated_strict_local"
    assert child.processing_manifest == manifest


@pytest.mark.asyncio
async def test_mixed_zip_writer_failure_removes_partial_durable_child(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coord

    _, uid_text = await auth_headers(client, "mixed-zip-failure@example.com")
    user_id = UUID(uid_text)
    patient = await coord.get_or_create_patient(db_session, user_id)
    parent = UploadedFile(
        user_id=user_id,
        filename="parent.zip",
        mime_type="application/zip",
        file_size_bytes=1,
        file_hash="f" * 64,
        storage_path=str(tmp_path / "parent.zip"),
        ingestion_status="processing",
        processing_mode="validated_strict_local",
        processing_manifest={"manifest_digest": "d" * 64},
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(coord.settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(coord.settings, "temp_extract_dir", str(tmp_path / "extract"))
    zip_path = tmp_path / "plain.zip"
    zip_path.write_bytes(_make_zip({"clinical-note.pdf": b"%PDF-1.7\nPHI"}))

    class FailingWriter:
        def __init__(self, _fileobj: object) -> None:
            pass

        def write_chunk(self, _plaintext: bytes) -> None:
            raise OSError("injected writer failure")

        def finalize(self) -> None:
            raise AssertionError("finalize should not run after write failure")

    monkeypatch.setattr(coord, "EncryptedFileWriter", FailingWriter)
    result = await coord._ingest_zip(
        db_session,
        user_id,
        patient.id,
        parent.id,
        zip_path,
    )

    assert result["unstructured_files"] == []
    assert result["errors"]
    assert list(upload_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_mixed_zip_commit_failure_rolls_back_before_parent_failure_update(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coord

    _, uid_text = await auth_headers(client, "mixed-zip-commit@example.com")
    user_id = UUID(uid_text)
    patient = await coord.get_or_create_patient(db_session, user_id)
    parent = UploadedFile(
        user_id=user_id,
        filename="parent.zip",
        mime_type="application/zip",
        file_size_bytes=1,
        file_hash="e" * 64,
        storage_path=str(tmp_path / "parent.zip"),
        ingestion_status="processing",
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(coord.settings, "upload_dir", str(upload_dir))
    monkeypatch.setattr(coord.settings, "temp_extract_dir", str(tmp_path / "extract"))
    zip_path = tmp_path / "plain.zip"
    zip_path.write_bytes(_make_zip({"clinical-note.pdf": b"%PDF-1.7\nPHI"}))

    original_commit = db_session.commit
    original_rollback = db_session.rollback
    rolled_back = False

    async def failing_commit() -> None:
        raise RuntimeError("injected commit failure")

    async def tracking_rollback() -> None:
        nonlocal rolled_back
        rolled_back = True
        await original_rollback()

    monkeypatch.setattr(db_session, "commit", failing_commit)
    monkeypatch.setattr(db_session, "rollback", tracking_rollback)
    with pytest.raises(RuntimeError, match="commit failure"):
        await coord._ingest_zip(
            db_session,
            user_id,
            patient.id,
            parent.id,
            zip_path,
        )

    assert rolled_back
    assert list(upload_dir.iterdir()) == []

    monkeypatch.setattr(db_session, "commit", original_commit)
    parent.ingestion_status = "failed"
    await db_session.commit()
    await db_session.refresh(parent)
    assert parent.ingestion_status == "failed"
