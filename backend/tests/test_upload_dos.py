"""W9 — resource-exhaustion DoS hardening (SEC-DOS-01/-02, SEC-INJ-03).

Three classes of fix are exercised here:

* **SEC-DOS-01** — upload endpoints must stream the body to disk in bounded
  chunks (never buffer the whole ≤5GB body into a single in-memory ``bytes``)
  and reject early on an oversized declared ``Content-Length``.
* **SEC-DOS-02** — the ZIP ingestion path must defend against zip bombs: cap the
  declared/streamed uncompressed total, per-member size, member count, and the
  per-member compression ratio — *before* writing the decompressed bytes.
* **SEC-INJ-03** — the FHIR coordinator must locate the bundle's Patient via
  streaming (ijson), not ``json.load`` of the whole ≤500MB bundle.
"""
from __future__ import annotations

import io
import json
import zipfile
from uuid import uuid4

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from tests.conftest import auth_headers

# Avoid kicking off the real background extraction worker / processor in the
# size/streaming endpoint tests (mirrors test_unstructured_upload.py).
from unittest.mock import AsyncMock, MagicMock, patch

PATCH_BG_TASK = patch("app.api.upload._process_unstructured", new_callable=AsyncMock)
PATCH_WORKER = patch("app.api.upload.start_extraction_worker")


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory DEFLATE zip from ``{name: bytes}``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SEC-DOS-02 — zip-bomb defenses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zip_bomb_rejected_by_upload_endpoint(
    client: AsyncClient, db_session: AsyncSession, monkeypatch
):
    """A small zip declaring a large uncompressed size is rejected (413/400)."""
    import app.services.ingestion.coordinator as coord

    # Tighten the uncompressed budget so the test stays fast/deterministic.
    monkeypatch.setattr(coord, "_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 1024 * 1024, raising=False)

    headers, _ = await auth_headers(client)
    bomb = _make_zip({"bomb.txt": b"\x00" * (5 * 1024 * 1024)})  # 5 MiB of zeros

    resp = await client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("bomb.zip", bomb, "application/zip")},
    )
    assert resp.status_code in (400, 413), resp.text


@pytest.mark.asyncio
async def test_zip_bomb_not_extracted_to_disk(db_session: AsyncSession, monkeypatch, tmp_path):
    """The bomb is rejected by the budget pre-scan, never via ``extractall``."""
    import app.services.ingestion.coordinator as coord

    monkeypatch.setattr(coord, "_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 1024 * 1024, raising=False)
    extractall_spy = MagicMock()
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extractall_spy)

    zip_path = tmp_path / "bomb.zip"
    zip_path.write_bytes(_make_zip({"bomb.txt": b"\x00" * (5 * 1024 * 1024)}))

    with pytest.raises(HTTPException) as exc:
        await coord._ingest_zip(db_session, uuid4(), uuid4(), uuid4(), zip_path)

    assert exc.value.status_code in (400, 413)
    extractall_spy.assert_not_called()


@pytest.mark.asyncio
async def test_zip_high_compression_ratio_rejected(db_session: AsyncSession, monkeypatch, tmp_path):
    """An absurd per-member compression ratio is rejected even under the total budget."""
    import app.services.ingestion.coordinator as coord

    # Total budget huge so ONLY the ratio guard can trip.
    monkeypatch.setattr(
        coord, "_ZIP_MAX_TOTAL_UNCOMPRESSED_BYTES", 1024 ** 4, raising=False
    )
    monkeypatch.setattr(coord, "_ZIP_MAX_COMPRESSION_RATIO", 50.0, raising=False)

    zip_path = tmp_path / "ratio.zip"
    zip_path.write_bytes(_make_zip({"bomb.txt": b"\x00" * (5 * 1024 * 1024)}))

    with pytest.raises(HTTPException) as exc:
        await coord._ingest_zip(db_session, uuid4(), uuid4(), uuid4(), zip_path)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_zip_too_many_entries_rejected(db_session: AsyncSession, monkeypatch, tmp_path):
    """A zip with more members than the cap is rejected (400)."""
    import app.services.ingestion.coordinator as coord

    monkeypatch.setattr(coord, "_ZIP_MAX_ENTRIES", 3, raising=False)
    zip_path = tmp_path / "many.zip"
    zip_path.write_bytes(_make_zip({f"f{i}.txt": b"x" for i in range(10)}))

    with pytest.raises(HTTPException) as exc:
        await coord._ingest_zip(db_session, uuid4(), uuid4(), uuid4(), zip_path)
    assert exc.value.status_code == 400


def test_safe_extract_zip_handles_normal_zip(tmp_path):
    """A normal small zip extracts member-by-member, preserving structure."""
    from app.services.ingestion.coordinator import _safe_extract_zip

    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(_make_zip({"a/b.txt": b"hello", "c.json": b"{}"}))
    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract_zip(zf, out)

    assert (out / "a" / "b.txt").read_bytes() == b"hello"
    assert (out / "c.json").read_bytes() == b"{}"


def test_safe_extract_zip_skips_zip_slip(tmp_path):
    """A member path escaping the extract dir is skipped, not written outside."""
    from app.services.ingestion.coordinator import _safe_extract_zip

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.txt", b"pwned")
        zf.writestr("safe.txt", b"ok")
    buf.seek(0)
    zip_path = tmp_path / "slip.zip"
    zip_path.write_bytes(buf.getvalue())

    out = tmp_path / "out"
    out.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract_zip(zf, out)

    assert (out / "safe.txt").read_bytes() == b"ok"
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------------------
# SEC-INJ-03 — stream the bundle Patient lookup (no whole-file json.load)
# ---------------------------------------------------------------------------


def test_find_patient_resource_streaming(tmp_path):
    """The Patient resource is located by streaming the bundle."""
    from app.services.ingestion.coordinator import _find_patient_resource_streaming

    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Observation", "id": "o1"}},
            {"resource": {"resourceType": "Patient", "id": "p1"}},
        ],
    }
    fpath = tmp_path / "bundle.json"
    fpath.write_text(json.dumps(bundle))

    found = _find_patient_resource_streaming(fpath)
    assert found is not None
    assert found["resourceType"] == "Patient"
    assert found["id"] == "p1"


def test_find_patient_resource_streaming_none_when_absent(tmp_path):
    """No Patient in the bundle yields None (not an error)."""
    from app.services.ingestion.coordinator import _find_patient_resource_streaming

    bundle = {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Observation"}}]}
    fpath = tmp_path / "bundle.json"
    fpath.write_text(json.dumps(bundle))
    assert _find_patient_resource_streaming(fpath) is None


@pytest.mark.asyncio
async def test_ingest_fhir_does_not_json_load_whole_bundle(
    db_session: AsyncSession, monkeypatch, tmp_path
):
    """``_ingest_fhir`` finds the Patient by streaming — never via json.load."""
    import app.services.ingestion.coordinator as coord

    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Observation", "id": "o1"}},
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "name": [{"family": "Doe", "given": ["Jane"]}],
                }
            },
        ],
    }
    fpath = tmp_path / "bundle.json"
    fpath.write_text(json.dumps(bundle))

    captured: dict = {}

    async def fake_get_or_create(db, user_id, fhir_data=None):
        captured["resource"] = fhir_data
        p = MagicMock()
        p.id = uuid4()
        return p

    # coordinator no longer imports json; patch the global json.load so any
    # reintroduction of a whole-bundle json.load (it resolves to this same module
    # object) is caught as a regression.
    import json as _json

    json_load_spy = MagicMock(side_effect=AssertionError("json.load must not be called"))
    monkeypatch.setattr(_json, "load", json_load_spy)

    with patch.object(coord, "get_or_create_patient", side_effect=fake_get_or_create), \
         patch.object(coord, "parse_fhir_bundle", new=AsyncMock(return_value={"records_inserted": 0})):
        await coord._ingest_fhir(db_session, uuid4(), uuid4(), uuid4(), fpath)

    assert captured.get("resource", {}).get("resourceType") == "Patient"
    json_load_spy.assert_not_called()


# ---------------------------------------------------------------------------
# SEC-DOS-01 — streaming uploads + early Content-Length rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_upload_to_disk_writes_and_returns_header(tmp_path):
    """Happy path: streams the whole body to disk, returns size + header + hash.

    CRYPTO-02: the default path encrypts at rest, so the on-disk bytes are NOT the
    plaintext — they decrypt back to it. The returned header/hash are PLAINTEXT.
    """
    import hashlib

    from app.api.upload import _stream_upload_to_disk
    from app.utils.file_utils import decrypt_file, is_encrypted_file
    from starlette.datastructures import UploadFile as StarletteUploadFile

    data = b"%PDF-1.4\n" + b"abcdefghijklmnop" + b"x" * 1000
    uf = StarletteUploadFile(file=io.BytesIO(data), filename="x.pdf")
    dest = tmp_path / "x.pdf"

    total, header, digest = await _stream_upload_to_disk(uf, dest, max_bytes=10 * 1024 * 1024)
    assert total == len(data)
    assert header == data[:16]
    assert digest == hashlib.sha256(data).hexdigest()
    # Encrypted at rest, but round-trips to the original plaintext.
    assert is_encrypted_file(dest)
    assert dest.read_bytes() != data
    assert decrypt_file(dest) == data


@pytest.mark.asyncio
async def test_stream_upload_to_disk_aborts_over_budget(tmp_path):
    """Over-budget body aborts mid-stream and unlinks the partial file."""
    from app.api.upload import _stream_upload_to_disk
    from starlette.datastructures import UploadFile as StarletteUploadFile

    body = b"\x00" * (3 * 1024 * 1024)  # 3 MiB
    uf = StarletteUploadFile(file=io.BytesIO(body), filename="x.bin")
    dest = tmp_path / "x.bin"

    with pytest.raises(HTTPException) as exc:
        await _stream_upload_to_disk(uf, dest, max_bytes=1024 * 1024)  # 1 MiB cap
    assert exc.value.status_code == 413
    assert not dest.exists()


@pytest.mark.asyncio
async def test_unstructured_upload_streams_body_in_chunks(
    client: AsyncClient, db_session: AsyncSession, monkeypatch
):
    """The unstructured endpoint reads the body in bounded chunks, not whole."""
    from starlette.datastructures import UploadFile as StarletteUploadFile

    read_sizes: list = []
    orig_read = StarletteUploadFile.read

    async def spy_read(self, size=-1):
        read_sizes.append(size)
        return await orig_read(self, size)

    monkeypatch.setattr(StarletteUploadFile, "read", spy_read)

    headers, _ = await auth_headers(client)
    rtf = rb"{\rtf1\ansi Patient visit note.}"
    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("note.rtf", io.BytesIO(rtf), "application/rtf")},
            headers=headers,
        )
    assert resp.status_code == 202, resp.text
    assert any(
        s is not None and s > 0 for s in read_sizes
    ), f"body must be read in bounded chunks, got read() calls: {read_sizes}"


@pytest.mark.asyncio
async def test_oversized_content_length_rejected_before_reading_body(
    client: AsyncClient, db_session: AsyncSession, monkeypatch
):
    """A too-large declared Content-Length is rejected before the body is read."""
    monkeypatch.setattr(settings, "max_file_size_mb", 1)  # 1 MiB cap

    from starlette.datastructures import UploadFile as StarletteUploadFile

    read_calls: list = []
    orig_read = StarletteUploadFile.read

    async def spy_read(self, size=-1):
        read_calls.append(size)
        return await orig_read(self, size)

    monkeypatch.setattr(StarletteUploadFile, "read", spy_read)

    headers, _ = await auth_headers(client)
    big = b"%PDF-1.4\n" + b"\x00" * (2 * 1024 * 1024)  # 2 MiB > 1 MiB cap
    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")},
            headers=headers,
        )
    assert resp.status_code == 413, resp.text
    assert read_calls == [], (
        "oversized upload must be rejected on Content-Length before the handler "
        f"reads the body; got read() calls: {read_calls}"
    )
