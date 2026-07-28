from __future__ import annotations


import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import auth_headers, FIXTURES_DIR


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
        files={"file": ("test.json", fhir_data, "application/json")},
    )

    # Verify patient exists via dashboard
    overview = await client.get("/api/v1/dashboard/overview", headers=headers)
    assert overview.json()["total_patients"] >= 1


@pytest.mark.asyncio
async def test_upload_records_appear_in_records(client: AsyncClient, db_session: AsyncSession):
    """Uploaded records appear in GET /records."""
    headers, _ = await auth_headers(client)
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    fhir_data = fhir_path.read_bytes()

    await client.post(
        "/api/v1/upload",
        headers=headers,
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
        files={"file": ("test.json", fhir_data, "application/json")},
    )
    upload_id = upload_resp.json()["upload_id"]

    status_resp = await client.get(f"/api/v1/upload/{upload_id}/status", headers=headers)
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
        files={"file": ("test.json", fhir_data, "application/json")},
    )
    upload_id = upload_resp.json()["upload_id"]

    resp = await client.get(f"/api/v1/upload/{upload_id}/errors", headers=headers)
    assert resp.status_code == 200
    assert "errors" in resp.json()
