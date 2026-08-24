from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.uploaded_file import UploadedFile
from tests.conftest import auth_headers, FIXTURES_DIR


def _deny_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    import app.services.dedup.llm_judge as llm_judge

    operations: list[str] = []

    def _unexpected_get_provider(
        _operation: object = None,
        _config: object = None,
    ) -> None:
        operations.append("get_provider")

    monkeypatch.setattr(llm_judge, "get_provider", _unexpected_get_provider)
    return operations


async def _owned_upload(
    db_session: AsyncSession,
    upload_id: str,
    user_id: str,
) -> UploadedFile:
    db_session.expire_all()
    return (
        await db_session.execute(
            select(UploadedFile).where(
                UploadedFile.id == UUID(upload_id),
                UploadedFile.user_id == UUID(user_id),
            )
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_upload_unauthenticated(client: AsyncClient):
    """POST /upload without token returns 401."""
    resp = await client.post("/api/v1/upload")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_no_file(client: AsyncClient, db_session: AsyncSession):
    """POST /upload without file returns 422."""
    headers, _ = await auth_headers(client)
    resp = await client.post("/api/v1/upload", headers=headers)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_synthetic_fhir(client: AsyncClient, db_session: AsyncSession):
    """Upload synthetic FHIR bundle, verify records inserted."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    resp = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("sample_fhir_bundle.json", fhir_data, "application/json")},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "upload_id" in data
    assert data["status"] in ("completed", "dedup_scanning")
    # Sample bundle has 1 Patient (skipped) + 17 clinical resources = 17 records
    assert data["records_inserted"] == 17
    assert isinstance(data["errors"], list)


@pytest.mark.asyncio
async def test_upload_creates_patient(client: AsyncClient, db_session: AsyncSession):
    """Upload auto-creates a patient record."""
    headers, uid = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("test.json", fhir_data, "application/json")},
    )

    # Verify patient exists via dashboard
    overview = await client.get("/api/v1/dashboard/overview", headers=headers)
    assert overview.json()["total_patients"] >= 1


@pytest.mark.asyncio
async def test_upload_records_appear_in_records(
    client: AsyncClient, db_session: AsyncSession
):
    """Uploaded records appear in GET /records."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("test.json", fhir_data, "application/json")},
    )

    resp = await client.get("/api/v1/records", headers=headers)
    data = resp.json()
    assert data["total"] == 17
    types = {item["record_type"] for item in data["items"]}
    assert "condition" in types
    assert "observation" in types
    assert "medication" in types
    assert "encounter" in types


@pytest.mark.asyncio
async def test_upload_status(client: AsyncClient, db_session: AsyncSession):
    """GET /upload/:id/status returns status with total_file_count."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    upload_resp = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("test.json", fhir_data, "application/json")},
    )
    upload_id = upload_resp.json()["upload_id"]

    status_resp = await client.get(
        f"/api/v1/upload/{upload_id}/status", headers=headers
    )
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["upload_id"] == upload_id
    assert data["ingestion_status"] in ("completed", "dedup_scanning")
    assert data["record_count"] == 17
    # Fix 4: total_file_count must be present
    assert "total_file_count" in data
    assert data["total_file_count"] >= 1


@pytest.mark.asyncio
async def test_upload_status_not_found(client: AsyncClient, db_session: AsyncSession):
    """GET /upload/:id/status with invalid UUID returns 404."""
    headers, _ = await auth_headers(client)
    resp = await client.get(
        "/api/v1/upload/00000000-0000-0000-0000-000000000000/status", headers=headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_history(client: AsyncClient, db_session: AsyncSession):
    """GET /upload/history returns upload list."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("test.json", fhir_data, "application/json")},
    )

    resp = await client.get("/api/v1/upload/history", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    item = data["items"][0]
    assert "id" in item
    assert "filename" in item
    assert "ingestion_status" in item
    assert "record_count" in item


@pytest.mark.asyncio
async def test_zip_upload_returns_unstructured_uploads(
    client: AsyncClient, db_session: AsyncSession
):
    """Uploading a ZIP with unstructured files returns them in unstructured_uploads."""
    headers, user_id = await auth_headers(client)

    import zipfile
    from io import BytesIO

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Add a minimal PDF (magic bytes)
        zf.writestr("doc.pdf", b"%PDF-1.4 minimal test content")
    buf.seek(0)

    resp = await client.post(
        "/api/v1/upload",
        files={"file": ("mixed.zip", buf, "application/zip")},
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "unstructured_uploads" in data
    assert len(data["unstructured_uploads"]) >= 1
    # Check structure of each entry
    entry = data["unstructured_uploads"][0]
    assert "upload_id" in entry
    assert "filename" in entry
    assert entry["status"] == "pending_extraction"


@pytest.mark.asyncio
async def test_upload_errors_endpoint(client: AsyncClient, db_session: AsyncSession):
    """GET /upload/:id/errors returns error list."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    upload_resp = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={"file": ("test.json", fhir_data, "application/json")},
    )
    upload_id = upload_resp.json()["upload_id"]

    resp = await client.get(f"/api/v1/upload/{upload_id}/errors", headers=headers)
    assert resp.status_code == 200
    assert "errors" in resp.json()


@pytest.mark.asyncio
async def test_cloud_assisted_tracked_fhir_and_identical_reupload_finish_without_provider(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    provider_operations = _deny_provider_construction(monkeypatch)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "temp_extract_dir", str(tmp_path / "temp-extract"))
    headers, user_id = await auth_headers(
        client,
        email="provider-free-tracked-fhir@example.com",
    )
    fhir_bytes = (FIXTURES_DIR / "sample_fhir_bundle.json").read_bytes()

    try:
        first = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={
                "file": (
                    "sample_fhir_bundle.json",
                    fhir_bytes,
                    "application/fhir+json",
                )
            },
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert first.status_code == 202
    assert first.json()["records_inserted"] == 17
    first_upload = await _owned_upload(
        db_session,
        first.json()["upload_id"],
        user_id,
    )
    assert first_upload.processing_mode == "cloud_assisted"
    assert first_upload.ingestion_status == "completed"
    assert first_upload.processing_completed_at is not None
    assert first_upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }

    records_after_first = (await client.get("/api/v1/records", headers=headers)).json()[
        "total"
    ]
    try:
        second = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={
                "file": (
                    "sample_fhir_bundle.json",
                    fhir_bytes,
                    "application/fhir+json",
                )
            },
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert second.status_code == 202
    assert second.json()["records_inserted"] == 0
    second_upload = await _owned_upload(
        db_session,
        second.json()["upload_id"],
        user_id,
    )
    records_after_second = (
        await client.get("/api/v1/records", headers=headers)
    ).json()["total"]

    assert second_upload.processing_mode == "cloud_assisted"
    assert second_upload.ingestion_status == "completed"
    assert second_upload.processing_completed_at is not None
    assert second_upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }
    assert records_after_second == records_after_first
    assert provider_operations == []
    assert coordinator._dedup_tasks == set()


@pytest.mark.asyncio
async def test_cloud_assisted_synthetic_cda_finishes_without_provider(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    provider_operations = _deny_provider_construction(monkeypatch)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "temp_extract_dir", str(tmp_path / "temp-extract"))
    headers, user_id = await auth_headers(
        client,
        email="provider-free-synthetic-cda@example.com",
    )
    cda_path = FIXTURES_DIR / "synthetic_cda" / "DOC0001.XML"

    try:
        response = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={"file": ("DOC0001.XML", cda_path.read_bytes(), "application/xml")},
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert response.status_code == 202
    assert response.json()["records_inserted"] > 0
    upload = await _owned_upload(db_session, response.json()["upload_id"], user_id)

    assert upload.processing_mode == "cloud_assisted"
    assert upload.ingestion_status == "completed"
    assert upload.processing_completed_at is not None
    assert upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }
    assert provider_operations == []
    assert coordinator._dedup_tasks == set()
