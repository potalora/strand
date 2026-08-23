from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, UploadFile
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.uploaded_file import UploadedFile
from app.services.local_ai.processing_snapshot import ProcessingSnapshot
from app.services.local_ai.types import ProcessingMode
from tests.conftest import auth_headers, create_test_patient

HAS_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))

# Patch the background task to avoid event loop conflicts with the test DB session.
# The _process_unstructured function creates its own DB session via async_session_factory
# which uses the production engine — incompatible with the test session override.
PATCH_BG_TASK = patch(
    "app.api.upload._process_unstructured",
    new_callable=AsyncMock,
)

# Patch the extraction worker so trigger-extraction tests don't start a real background loop.
PATCH_WORKER = patch("app.api.upload.start_extraction_worker")


@pytest.mark.asyncio
async def test_upload_rtf_creates_record(client: AsyncClient, db_session: AsyncSession):
    """Upload an RTF file to the unstructured endpoint."""
    headers, user_id = await auth_headers(client)

    rtf_content = rb"""{\rtf1\ansi\deff0 Patient visit note. Assessment: Hypertension. Plan: Continue medication.}"""

    with PATCH_BG_TASK:
        resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("note.rtf", io.BytesIO(rtf_content), "application/rtf")},
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["upload_id"]
    assert data["filename"] == "note.rtf"
    assert data["status"] == "pending_extraction"
    assert data["file_type"] == "rtf"


@pytest.mark.asyncio
async def test_reject_unsupported_file_type(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify .doc files are rejected with 400."""
    headers, user_id = await auth_headers(client)

    resp = await client.post(
        "/api/v1/upload/unstructured",
        files={"file": ("doc.doc", io.BytesIO(b"content"), "application/msword")},
        headers=headers,
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_reject_txt_file(client: AsyncClient, db_session: AsyncSession):
    """Verify .txt files are rejected."""
    headers, user_id = await auth_headers(client)

    resp = await client.post(
        "/api/v1/upload/unstructured",
        files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
        headers=headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_extraction_results_endpoint_not_found(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify 404 for non-existent upload."""
    headers, user_id = await auth_headers(client)

    resp = await client.get(
        "/api/v1/upload/00000000-0000-0000-0000-000000000000/extraction",
        headers=headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_extraction_results_for_uploaded_file(
    client: AsyncClient, db_session: AsyncSession
):
    """Upload RTF, then check extraction endpoint returns valid response."""
    headers, user_id = await auth_headers(client)

    rtf_content = rb"""{\rtf1\ansi Patient has diabetes and takes Metformin.}"""
    with PATCH_BG_TASK:
        upload_resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("note.rtf", io.BytesIO(rtf_content), "application/rtf")},
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    upload_id = upload_resp.json()["upload_id"]

    # File is queued, so status will be "pending_extraction"
    resp = await client.get(
        f"/api/v1/upload/{upload_id}/extraction",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["upload_id"] == upload_id
    assert data["status"] in (
        "pending_extraction",
        "processing",
        "awaiting_confirmation",
        "completed",
        "failed",
    )


@pytest.mark.asyncio
async def test_confirm_extraction_missing_patient(
    client: AsyncClient, db_session: AsyncSession
):
    """Verify confirmation fails without patient_id."""
    headers, user_id = await auth_headers(client)

    rtf_content = rb"""{\rtf1\ansi Test note.}"""
    with PATCH_BG_TASK:
        upload_resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("note.rtf", io.BytesIO(rtf_content), "application/rtf")},
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    upload_id = upload_resp.json()["upload_id"]

    resp = await client.post(
        f"/api/v1/upload/{upload_id}/confirm-extraction",
        json={"confirmed_entities": [], "patient_id": ""},
        headers=headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_confirm_extraction_creates_records(
    client: AsyncClient, db_session: AsyncSession
):
    """Confirm extracted entities and verify HealthRecords are created."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)

    rtf_content = rb"""{\rtf1\ansi Test note.}"""
    with PATCH_BG_TASK:
        upload_resp = await client.post(
            "/api/v1/upload/unstructured",
            files={"file": ("note.rtf", io.BytesIO(rtf_content), "application/rtf")},
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    upload_id = upload_resp.json()["upload_id"]

    # Manually confirm with synthetic entities
    resp = await client.post(
        f"/api/v1/upload/{upload_id}/confirm-extraction",
        json={
            "confirmed_entities": [
                {
                    "entity_class": "condition",
                    "text": "Hypertension",
                    "attributes": {"status": "active"},
                    "confidence": 0.85,
                },
                {
                    "entity_class": "medication",
                    "text": "Lisinopril",
                    "attributes": {"medication_group": "Lisinopril"},
                    "confidence": 0.9,
                },
                {
                    "entity_class": "dosage",
                    "text": "10mg",
                    "attributes": {},
                    "confidence": 0.8,
                },
            ],
            "patient_id": str(patient.id),
        },
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # dosage should be skipped (non-storable), so 2 records
    assert data["records_created"] == 2
    assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_upload_pdf_accepted(client: AsyncClient, db_session: AsyncSession):
    """Verify PDF files are accepted by the unstructured endpoint."""
    headers, user_id = await auth_headers(client)

    # Minimal PDF-like content (won't actually parse but tests routing)
    with PATCH_BG_TASK:
        resp = await client.post(
            "/api/v1/upload/unstructured",
            files={
                "file": ("report.pdf", io.BytesIO(b"%PDF-1.4 test"), "application/pdf")
            },
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["file_type"] == "pdf"


@pytest.mark.asyncio
async def test_concurrent_uploads_respect_semaphore(
    client: AsyncClient, db_session: AsyncSession
):
    """Upload 3 RTF files simultaneously, verify all are accepted."""
    headers, user_id = await auth_headers(client)

    rtf_files = [
        rb"""{\rtf1\ansi Note one: Patient has hypertension.}""",
        rb"""{\rtf1\ansi Note two: Patient takes Metformin 500mg.}""",
        rb"""{\rtf1\ansi Note three: Lab results show elevated glucose.}""",
    ]

    upload_ids = []
    with PATCH_BG_TASK:
        for i, content in enumerate(rtf_files):
            resp = await client.post(
                "/api/v1/upload/unstructured",
                files={
                    "file": (f"note_{i}.rtf", io.BytesIO(content), "application/rtf")
                },
                headers=headers,
                data={"processing_mode": "cloud_assisted"},
            )
            assert resp.status_code == 202
            data = resp.json()
            assert data["upload_id"]
            assert data["status"] == "pending_extraction"
            upload_ids.append(data["upload_id"])

    # All 3 should have unique upload IDs
    assert len(set(upload_ids)) == 3


@pytest.mark.asyncio
async def test_batch_upload_endpoint(client: AsyncClient, db_session: AsyncSession):
    """Verify /unstructured-batch accepts multiple files and returns list of upload IDs."""
    headers, user_id = await auth_headers(client)

    rtf1 = rb"""{\rtf1\ansi Batch note one: Diabetes type 2.}"""
    rtf2 = rb"""{\rtf1\ansi Batch note two: Allergic to penicillin.}"""
    pdf1 = b"%PDF-1.4 batch pdf content"

    with PATCH_BG_TASK:
        resp = await client.post(
            "/api/v1/upload/unstructured-batch",
            files=[
                ("files", ("batch1.rtf", io.BytesIO(rtf1), "application/rtf")),
                ("files", ("batch2.rtf", io.BytesIO(rtf2), "application/rtf")),
                ("files", ("batch3.pdf", io.BytesIO(pdf1), "application/pdf")),
            ],
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["total"] == 3
    assert len(data["uploads"]) == 3
    assert [item["filename"] for item in data["uploads"]] == [
        "batch1.rtf",
        "batch2.rtf",
        "batch3.pdf",
    ]
    assert data["rejected"] == []

    # Each upload should have a unique ID
    upload_ids = [u["upload_id"] for u in data["uploads"]]
    assert len(set(upload_ids)) == 3

    # Check file types
    file_types = [u["file_type"] for u in data["uploads"]]
    assert file_types.count("rtf") == 2
    assert file_types.count("pdf") == 1


@pytest.mark.asyncio
async def test_pending_extraction_lists_pending_files(
    client: AsyncClient, db_session: AsyncSession
):
    """GET /upload/pending-extraction returns files with pending_extraction status."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="test_note.pdf",
        mime_type="application/pdf",
        file_size_bytes=1234,
        file_hash="abc123pendingtest",
        storage_path="/tmp/test.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        manual_extraction_required=True,
    )
    db_session.add(upload)
    await db_session.commit()

    resp = await client.get("/api/v1/upload/pending-extraction", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["files"][0]["filename"] == "test_note.pdf"
    assert data["files"][0]["id"] == str(upload.id)
    assert data["files"][0]["manual_extraction_required"] is True


@pytest.mark.asyncio
async def test_pending_extraction_excludes_other_users(
    client: AsyncClient, db_session: AsyncSession
):
    """Pending extraction only returns files owned by the current user."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from app.models.user import User
    from uuid import uuid4

    other_user = User(
        id=uuid4(),
        login_identifier="other_pending_encrypted",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(other_user)
    await db_session.flush()

    upload = UploadedFile(
        id=uuid4(),
        user_id=other_user.id,
        filename="other_note.pdf",
        mime_type="application/pdf",
        file_size_bytes=1234,
        file_hash="xyz789pendingother",
        storage_path="/tmp/other.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    resp = await client.get("/api/v1/upload/pending-extraction", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_batch_upload_skips_invalid_files(
    client: AsyncClient, db_session: AsyncSession
):
    """Batch endpoint skips unsupported file types and invalid magic bytes."""
    headers, user_id = await auth_headers(client)

    rtf_valid = rb"""{\rtf1\ansi Later valid RTF note.}"""
    pdf_valid = b"%PDF-1.4 final valid PDF"
    txt_invalid = b"plain text not allowed"

    with PATCH_BG_TASK:
        resp = await client.post(
            "/api/v1/upload/unstructured-batch",
            files=[
                ("files", ("first-invalid.txt", io.BytesIO(txt_invalid), "text/plain")),
                (
                    "files",
                    ("later-valid.rtf", io.BytesIO(rtf_valid), "application/rtf"),
                ),
                (
                    "files",
                    ("final-valid.pdf", io.BytesIO(pdf_valid), "application/pdf"),
                ),
            ],
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["total"] == 2
    assert [(item["filename"], item["file_type"]) for item in data["uploads"]] == [
        ("later-valid.rtf", "rtf"),
        ("final-valid.pdf", "pdf"),
    ]
    assert data["rejected"] == [
        {"filename": "first-invalid.txt", "code": "unsupported_type"}
    ]


@pytest.mark.asyncio
async def test_batch_upload_all_rejected_uses_stable_bounded_codes_without_snapshot(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _headers, _user_id = await auth_headers(client)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_file_size_mb", 1)
    resolver = AsyncMock(
        side_effect=HTTPException(status_code=409, detail="unavailable")
    )
    monkeypatch.setattr(
        "app.api.upload._resolve_ingestion_snapshot_or_409",
        resolver,
    )
    oversized = rb"{\rtf1" + (b"x" * (1024 * 1024 + 1))

    from app.api.upload import upload_unstructured_batch

    result = await upload_unstructured_batch(
        files=[
            UploadFile(filename=None, file=io.BytesIO(rb"{\rtf1}")),
            UploadFile(filename="unsupported.txt", file=io.BytesIO(b"plain")),
            UploadFile(filename="oversized.rtf", file=io.BytesIO(oversized)),
            UploadFile(filename="bad-signature.pdf", file=io.BytesIO(b"not a pdf")),
        ],
        processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
        user_id=UUID(str(_user_id)),
        db=db_session,
    )

    payload = result.model_dump()
    assert payload == {
        "uploads": [],
        "rejected": [
            {"filename": "", "code": "missing_filename"},
            {"filename": "unsupported.txt", "code": "unsupported_type"},
            {"filename": "oversized.rtf", "code": "file_too_large"},
            {"filename": "bad-signature.pdf", "code": "invalid_signature"},
        ],
        "total": 0,
    }
    serialized = json.dumps(payload)
    assert "/tmp/" not in serialized
    assert "HTTPException" not in serialized
    assert "unavailable" not in serialized
    resolver.assert_not_awaited()
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_batch_rejection_cap_does_not_block_later_valid_file(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, _user_id = await auth_headers(client)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    invalid_files = [
        ("files", (f"invalid-{index}.txt", io.BytesIO(b"plain"), "text/plain"))
        for index in range(51)
    ]
    invalid_files.append(
        (
            "files",
            (
                "accepted-after-cap.rtf",
                io.BytesIO(rb"{\rtf1 valid}"),
                "application/rtf",
            ),
        )
    )

    response = await client.post(
        "/api/v1/upload/unstructured-batch",
        files=invalid_files,
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["total"] == 1
    assert [item["filename"] for item in payload["uploads"]] == [
        "accepted-after-cap.rtf"
    ]
    assert len(payload["rejected"]) == 50
    assert payload["rejected"][0] == {
        "filename": "invalid-0.txt",
        "code": "unsupported_type",
    }
    assert payload["rejected"][-1] == {
        "filename": "invalid-49.txt",
        "code": "unsupported_type",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("http_409", "cancelled"))
async def test_batch_snapshot_failure_removes_staged_ciphertext(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_kind: str,
) -> None:
    _headers, _user_id = await auth_headers(client)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    failure: BaseException = (
        HTTPException(status_code=409, detail="unavailable")
        if failure_kind == "http_409"
        else asyncio.CancelledError()
    )
    monkeypatch.setattr(
        "app.api.upload._resolve_ingestion_snapshot_or_409",
        AsyncMock(side_effect=failure),
    )

    from app.api.upload import upload_unstructured_batch

    request = upload_unstructured_batch(
        files=[
            UploadFile(
                filename="valid-before-snapshot.rtf",
                file=io.BytesIO(rb"{\rtf1 valid}"),
            )
        ],
        processing_mode=ProcessingMode.VALIDATED_STRICT_LOCAL,
        user_id=UUID(str(_user_id)),
        db=db_session,
    )
    if failure_kind == "http_409":
        with pytest.raises(HTTPException) as exc_info:
            await request
        assert exc_info.value.status_code == 409
    else:
        with pytest.raises(asyncio.CancelledError):
            await request
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_batch_non_size_http_error_is_not_classified_as_rejection(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _headers, _user_id = await auth_headers(client)
    monkeypatch.setattr(
        "app.api.upload._stream_upload_to_disk",
        AsyncMock(side_effect=HTTPException(status_code=409, detail="private")),
    )

    from app.api.upload import upload_unstructured_batch

    with pytest.raises(HTTPException) as exc_info:
        await upload_unstructured_batch(
            files=[UploadFile(filename="valid.rtf", file=io.BytesIO(rb"{\rtf1}"))],
            processing_mode=ProcessingMode.CLOUD_ASSISTED,
            user_id=UUID(str(_user_id)),
            db=db_session,
        )

    assert exc_info.value.status_code == 409


def _cloud_snapshot() -> ProcessingSnapshot:
    return ProcessingSnapshot(
        mode=ProcessingMode.CLOUD_ASSISTED,
        manifest_snapshot=None,
        manifest_sha256=None,
        schema_version=None,
    )


@pytest.mark.asyncio
async def test_reprocess_new_upload_returns_stored_filename(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, user_id = await auth_headers(client)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    source_path = tmp_path / "server-source.rtf"
    source_path.write_bytes(b"synthetic encrypted source")
    source = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="server-source.rtf",
        mime_type="application/rtf",
        file_size_bytes=27,
        file_hash=uuid4().hex,
        storage_path=str(source_path),
        ingestion_status="completed",
        file_category="unstructured",
        processing_mode="prompt_only",
    )
    db_session.add(source)
    await db_session.commit()
    monkeypatch.setattr(
        "app.api.upload._resolve_ingestion_snapshot_or_409",
        AsyncMock(return_value=_cloud_snapshot()),
    )

    response = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        json={"processing_mode": "cloud_assisted"},
        headers=headers,
    )

    assert response.status_code == 202
    assert response.json()["filename"] == source.filename


@pytest.mark.asyncio
async def test_reprocess_existing_upload_returns_existing_stored_filename(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, user_id = await auth_headers(client)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    source_path = tmp_path / "canonical-source.rtf"
    source_path.write_bytes(b"synthetic encrypted source")
    source = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="canonical-source.rtf",
        mime_type="application/rtf",
        file_size_bytes=27,
        file_hash=uuid4().hex,
        storage_path=str(source_path),
        ingestion_status="completed",
        file_category="unstructured",
        processing_mode="prompt_only",
    )
    db_session.add(source)
    await db_session.flush()
    existing = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="existing-server-name.rtf",
        mime_type="application/rtf",
        file_size_bytes=27,
        file_hash=source.file_hash,
        storage_path=str(source_path),
        ingestion_status="pending_extraction",
        ingestion_progress={"reprocesses_upload_id": str(source.id)},
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(existing)
    await db_session.commit()
    monkeypatch.setattr(
        "app.api.upload._resolve_ingestion_snapshot_or_409",
        AsyncMock(return_value=_cloud_snapshot()),
    )

    response = await client.post(
        f"/api/v1/upload/{source.id}/reprocess",
        json={"processing_mode": "cloud_assisted"},
        headers=headers,
    )

    assert response.status_code == 202
    assert response.json()["filename"] == existing.filename


@pytest.mark.asyncio
async def test_trigger_extraction_starts_processing(
    client: AsyncClient, db_session: AsyncSession
):
    """Manual ZIP children become worker-claimable only after the trigger commit."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    uploads = []
    for i in range(3):
        upload = UploadedFile(
            id=uuid4(),
            user_id=user_id,
            filename=f"note_{i}.rtf",
            mime_type="application/rtf",
            file_size_bytes=500,
            file_hash=f"hash_trigger_{i}",
            storage_path=f"/tmp/note_{i}.rtf",
            ingestion_status="pending_extraction",
            file_category="unstructured",
            processing_mode="cloud_assisted",
            manual_extraction_required=True,
        )
        db_session.add(upload)
        uploads.append(upload)
    await db_session.commit()

    upload_ids = [str(u.id) for u in uploads]

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": upload_ids},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 3
    assert data["failed"] == 0
    assert len(data["results"]) == 3
    for upload in uploads:
        await db_session.refresh(upload)
        assert upload.manual_extraction_required is False


@pytest.mark.asyncio
async def test_trigger_extraction_rejects_auto_claimable_pending_upload(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Pending status alone never grants the manual extraction control."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="direct.pdf",
        mime_type="application/pdf",
        file_size_bytes=1000,
        file_hash="hash_direct_auto_claim",
        storage_path="/tmp/direct.pdf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    response = await client.post(
        "/api/v1/upload/trigger-extraction",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "upload_id": str(upload.id),
            "status": "manual_extraction_not_required",
        }
    ]


@pytest.mark.asyncio
async def test_trigger_extraction_releases_failed_manual_zip_child(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """An explicit retry clears the durable manual hold before worker claim."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="failed-zip-child.pdf",
        mime_type="application/pdf",
        file_size_bytes=1000,
        file_hash="hash_failed_manual_child",
        storage_path="/tmp/medtimeline-zip-set-example/failed-zip-child.pdf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="cloud_assisted",
        manual_extraction_required=True,
    )
    db_session.add(upload)
    await db_session.commit()

    response = await client.post(
        "/api/v1/upload/trigger-extraction",
        json={"upload_ids": [str(upload.id)]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["triggered"] == 1
    await db_session.refresh(upload)
    assert upload.ingestion_status == "pending_extraction"
    assert upload.manual_extraction_required is False


@pytest.mark.asyncio
async def test_upload_openapi_publishes_typed_manual_extraction_contracts(
    client: AsyncClient,
) -> None:
    """Mixed-ZIP and trigger payloads are explicit in generated OpenAPI."""
    schema = (await client.get("/openapi.json")).json()
    upload_response = schema["components"]["schemas"]["UploadResponse"]
    child_schema = upload_response["properties"]["unstructured_uploads"]["items"]
    assert child_schema == {"$ref": "#/components/schemas/UnstructuredUploadItem"}
    child_properties = schema["components"]["schemas"]["UnstructuredUploadItem"][
        "properties"
    ]
    assert child_properties["manual_extraction_required"]["type"] == "boolean"

    trigger = schema["paths"]["/api/v1/upload/trigger-extraction"]["post"]
    request_schema = trigger["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {"$ref": "#/components/schemas/TriggerExtractionRequest"}
    response_schema = trigger["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert response_schema == {"$ref": "#/components/schemas/TriggerExtractionResponse"}


@pytest.mark.asyncio
async def test_trigger_extraction_rejects_wrong_status(
    client: AsyncClient, db_session: AsyncSession
):
    """Trigger extraction rejects completed files."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="completed.pdf",
        mime_type="application/pdf",
        file_size_bytes=1000,
        file_hash="hash_done_trigger",
        storage_path="/tmp/completed.pdf",
        ingestion_status="completed",
        file_category="structured",
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 0
    assert data["failed"] == 1


@pytest.mark.asyncio
async def test_trigger_extraction_allows_retry_of_processing(
    client: AsyncClient, db_session: AsyncSession
):
    """Trigger extraction allows retrying files stuck in processing status."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="stuck.pdf",
        mime_type="application/pdf",
        file_size_bytes=1000,
        file_hash="hash_stuck_processing",
        storage_path="/tmp/stuck.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 1
    assert data["failed"] == 0


@pytest.mark.asyncio
async def test_trigger_extraction_skips_actively_processing(
    client: AsyncClient, db_session: AsyncSession
):
    """Re-triggering a file a worker is ACTIVELY processing (recent
    processing_started_at) must be a no-op. Resetting it to pending_extraction
    would let the worker re-claim it and run a second concurrent extraction
    pass (wasting Gemini calls, risking duplicate records). Genuinely stuck
    files — old/absent processing_started_at — stay retriable (covered by
    test_trigger_extraction_allows_retry_of_processing).
    """
    from app.models.uploaded_file import UploadedFile
    from datetime import datetime, timezone
    from uuid import uuid4

    headers, user_id = await auth_headers(client)

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="active.pdf",
        mime_type="application/pdf",
        file_size_bytes=1000,
        file_hash="hash_active_processing",
        storage_path="/tmp/active.pdf",
        ingestion_status="processing",
        file_category="unstructured",
        processing_mode="cloud_assisted",
        processing_started_at=datetime.now(timezone.utc),  # actively processing now
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 0
    assert data["failed"] == 1
    # Status must be untouched — not reset to pending_extraction
    await db_session.refresh(upload)
    assert upload.ingestion_status == "processing"


@pytest.mark.asyncio
async def test_trigger_extraction_allows_retry_of_failed(
    client: AsyncClient, db_session: AsyncSession
):
    """Trigger extraction allows retrying files with failed status."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="failed.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash="hash_failed_retry",
        storage_path="/tmp/failed.rtf",
        ingestion_status="failed",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 1
    assert data["failed"] == 0


@pytest.mark.asyncio
async def test_trigger_extraction_allows_retry_of_awaiting_confirmation(
    client: AsyncClient, db_session: AsyncSession
):
    """Trigger extraction allows retrying files stuck in awaiting_confirmation status."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="awaiting.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash="hash_awaiting_retry",
        storage_path="/tmp/awaiting.rtf",
        ingestion_status="awaiting_confirmation",
        file_category="unstructured",
        processing_mode="cloud_assisted",
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 1
    assert data["failed"] == 0


@pytest.mark.asyncio
async def test_auto_confirm_creates_records(
    client: AsyncClient, db_session: AsyncSession
):
    """Auto-confirm creates health records when patient exists."""
    headers, user_id = await auth_headers(client)
    patient = await create_test_patient(db_session, user_id)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    upload = UploadedFile(
        id=uuid4(),
        user_id=user_id,
        filename="autoconfirm.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash="hash_autoconfirm",
        storage_path="/tmp/autoconfirm.rtf",
        ingestion_status="processing",
        file_category="unstructured",
        extraction_entities=[
            {
                "entity_class": "condition",
                "text": "Hypertension",
                "attributes": {"status": "active"},
                "start_pos": 0,
                "end_pos": 12,
                "confidence": 0.9,
            },
            {
                "entity_class": "medication",
                "text": "Lisinopril",
                "attributes": {"medication_group": "Lisinopril"},
                "start_pos": 13,
                "end_pos": 23,
                "confidence": 0.85,
            },
        ],
    )
    db_session.add(upload)
    await db_session.commit()

    # Simulate what _process_unstructured does after extraction
    from app.services.extraction.entity_extractor import ExtractedEntity
    from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
    from app.models.record import HealthRecord
    from sqlalchemy import select

    created = 0
    for entity_data in upload.extraction_entities:
        entity = ExtractedEntity(
            entity_class=entity_data["entity_class"],
            text=entity_data["text"],
            attributes=entity_data["attributes"],
            start_pos=entity_data.get("start_pos"),
            end_pos=entity_data.get("end_pos"),
            confidence=entity_data.get("confidence", 0.8),
        )
        record_dict = entity_to_health_record_dict(
            entity=entity,
            user_id=user_id,
            patient_id=patient.id,
            source_file_id=upload.id,
        )
        if record_dict is not None:
            db_session.add(HealthRecord(**record_dict))
            created += 1

    upload.ingestion_status = "completed"
    upload.record_count = created
    await db_session.commit()

    assert created == 2
    assert upload.ingestion_status == "completed"
    assert upload.record_count == 2

    # Verify records were created in DB
    result = await db_session.execute(
        select(HealthRecord).where(HealthRecord.user_id == user_id)
    )
    records = result.scalars().all()
    assert len(records) >= 2


@pytest.mark.asyncio
async def test_ensure_patient_creates_placeholder_when_missing(
    client: AsyncClient, db_session: AsyncSession
):
    """Unstructured-first users have no Patient yet. `_ensure_patient` must
    create a blank (no-PHI) placeholder so the auto-confirm step produces
    records instead of stalling at awaiting_confirmation — the "Reimagined"
    frontend has no manual-confirm UI, so that fallback is a dead end.
    A later structured upload backfills this blank patient via
    get_or_create_patient.
    """
    from app.api.upload import _ensure_patient
    from app.models.patient import Patient
    from sqlalchemy import select

    headers, user_id = await auth_headers(client)

    # No patient initially
    existing = (
        (await db_session.execute(select(Patient).where(Patient.user_id == user_id)))
        .scalars()
        .all()
    )
    assert existing == []

    patient = await _ensure_patient(db_session, user_id)
    await db_session.commit()
    assert patient is not None
    assert str(patient.user_id) == str(user_id)
    # Placeholder carries no PHI
    assert patient.name_encrypted is None
    assert patient.mrn_encrypted is None

    # Idempotent: a second call returns the same patient, never a duplicate
    again = await _ensure_patient(db_session, user_id)
    assert again.id == patient.id
    all_patients = (
        (await db_session.execute(select(Patient).where(Patient.user_id == user_id)))
        .scalars()
        .all()
    )
    assert len(all_patients) == 1


@pytest.mark.asyncio
async def test_extraction_progress_returns_counts(
    client: AsyncClient, db_session: AsyncSession
):
    """GET /upload/extraction-progress returns status counts for unstructured files."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    # Create files in various statuses
    for i, status in enumerate(
        [
            "completed",
            "completed",
            "processing",
            "dedup_scanning",
            "dedup_processing",
            "failed",
            "pending_extraction",
        ]
    ):
        upload = UploadedFile(
            id=uuid4(),
            user_id=user_id,
            filename=f"progress_{i}.rtf",
            mime_type="application/rtf",
            file_size_bytes=500,
            file_hash=f"hash_progress_{i}_{status}",
            storage_path=f"/tmp/progress_{i}.rtf",
            ingestion_status=status,
            file_category="unstructured",
            record_count=3 if status == "completed" else 0,
        )
        db_session.add(upload)
    await db_session.commit()

    resp = await client.get("/api/v1/upload/extraction-progress", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 7
    assert data["completed"] == 2
    assert data["processing"] == 3
    assert data["failed"] == 1
    assert data["pending"] == 1
    assert data["records_created"] == 6


@pytest.mark.asyncio
async def test_extraction_progress_excludes_duplicate_files(
    client: AsyncClient, db_session: AsyncSession
):
    """Duplicate-file uploads are skipped (no extraction), so they must not
    inflate the progress denominator ('1 of 2 processed' when 1 is a dup)."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from uuid import uuid4

    for i, status in enumerate(["completed", "duplicate_file"]):
        db_session.add(
            UploadedFile(
                id=uuid4(),
                user_id=user_id,
                filename=f"dup_{i}.rtf",
                mime_type="application/rtf",
                file_size_bytes=500,
                file_hash=f"hash_dup_{i}_{status}",
                storage_path=f"/tmp/dup_{i}.rtf",
                ingestion_status=status,
                file_category="unstructured",
                record_count=3 if status == "completed" else 0,
            )
        )
    await db_session.commit()

    data = (
        await client.get("/api/v1/upload/extraction-progress", headers=headers)
    ).json()
    # The duplicate is excluded entirely: 1 file, completed, not 2.
    assert data["total"] == 1
    assert data["completed"] == 1


@pytest.mark.asyncio
async def test_trigger_extraction_rejects_other_users_files(
    client: AsyncClient, db_session: AsyncSession
):
    """Trigger extraction rejects files not owned by the current user."""
    headers, user_id = await auth_headers(client)

    from app.models.uploaded_file import UploadedFile
    from app.models.user import User
    from uuid import uuid4

    other_user = User(
        id=uuid4(),
        login_identifier="trigger_other_user_encrypted",
        password_hash="$2b$12$fakefakefakefakefakefuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        is_active=True,
    )
    db_session.add(other_user)
    await db_session.flush()

    upload = UploadedFile(
        id=uuid4(),
        user_id=other_user.id,
        filename="other_note.rtf",
        mime_type="application/rtf",
        file_size_bytes=500,
        file_hash="hash_other_trigger_test",
        storage_path="/tmp/other_trigger.rtf",
        ingestion_status="pending_extraction",
        file_category="unstructured",
    )
    db_session.add(upload)
    await db_session.commit()

    with PATCH_BG_TASK, PATCH_WORKER:
        resp = await client.post(
            "/api/v1/upload/trigger-extraction",
            json={"upload_ids": [str(upload.id)]},
            headers=headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["triggered"] == 0
    assert data["failed"] == 1
