"""Upload audit rows must not persist client-controlled filenames as plaintext."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from tests.conftest import auth_headers

_PHI_FILENAME = "Jane-Doe-MRN-123456-discharge-summary.rtf"
_STRUCTURED_PHI_FILENAME = "Jane-Doe-MRN-123456-records.json"


async def _upload_audit_details(
    db: AsyncSession,
    *,
    resource_ids: set[UUID],
    action: str = "file.upload.unstructured",
) -> list[dict]:
    rows = (
        (
            await db.execute(
                select(AuditLog).where(
                    AuditLog.action == action,
                    AuditLog.resource_id.in_(resource_ids),
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == len(resource_ids)
    return [row.details or {} for row in rows]


@pytest.mark.asyncio
async def test_single_unstructured_upload_audit_excludes_raw_filename(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, _user_id = await auth_headers(client, "upload-audit-single@example.com")
    monkeypatch.setattr("app.api.upload.settings.upload_dir", str(tmp_path))

    response = await client.post(
        "/api/v1/upload/unstructured",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={
            "file": (
                _PHI_FILENAME,
                b"{\\rtf1\\ansi bounded synthetic note}",
                "application/rtf",
            )
        },
    )

    assert response.status_code == 202, response.text
    upload_id = UUID(response.json()["upload_id"])
    details = (await _upload_audit_details(db_session, resource_ids={upload_id}))[0]
    assert _PHI_FILENAME.casefold() not in json.dumps(details).casefold()
    assert "filename" not in details
    assert details == {"file_type": ".rtf", "file_category": "unstructured"}


@pytest.mark.asyncio
async def test_structured_upload_audit_excludes_raw_filename(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, _user_id = await auth_headers(
        client, "upload-audit-structured@example.com"
    )
    upload_id = UUID("00000000-0000-4000-8000-000000000123")
    monkeypatch.setattr("app.api.upload.settings.upload_dir", str(tmp_path))
    ingest = AsyncMock(
        return_value={
            "upload_id": str(upload_id),
            "status": "completed",
            "records_inserted": 2,
            "errors": [],
        }
    )
    monkeypatch.setattr("app.services.ingestion.coordinator.ingest_file", ingest)

    response = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files={
            "file": (
                _STRUCTURED_PHI_FILENAME,
                b'{"resourceType":"Bundle","entry":[]}',
                "application/fhir+json",
            )
        },
    )

    assert response.status_code == 202, response.text
    details = (
        await _upload_audit_details(
            db_session,
            resource_ids={upload_id},
            action="file.upload",
        )
    )[0]
    assert _STRUCTURED_PHI_FILENAME.casefold() not in json.dumps(details).casefold()
    assert "filename" not in details
    assert details == {
        "file_type": ".json",
        "file_category": "structured",
        "records": 2,
    }


@pytest.mark.asyncio
async def test_batch_unstructured_upload_audits_exclude_raw_filenames(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    headers, _user_id = await auth_headers(client, "upload-audit-batch@example.com")
    monkeypatch.setattr("app.api.upload.settings.upload_dir", str(tmp_path))
    second_filename = "John-Smith-DOB-1970-01-01-labs.rtf"

    response = await client.post(
        "/api/v1/upload/unstructured-batch",
        headers=headers,
        data={"processing_mode": "cloud_assisted"},
        files=[
            (
                "files",
                (
                    _PHI_FILENAME,
                    b"{\\rtf1\\ansi first bounded synthetic note}",
                    "application/rtf",
                ),
            ),
            (
                "files",
                (
                    second_filename,
                    b"{\\rtf1\\ansi second bounded synthetic note}",
                    "application/rtf",
                ),
            ),
        ],
    )

    assert response.status_code == 202, response.text
    upload_ids = {UUID(item["upload_id"]) for item in response.json()["uploads"]}
    details = await _upload_audit_details(db_session, resource_ids=upload_ids)
    serialized = json.dumps(details).casefold()
    assert _PHI_FILENAME.casefold() not in serialized
    assert second_filename.casefold() not in serialized
    assert all("filename" not in detail for detail in details)
    assert details == [
        {"file_type": ".rtf", "file_category": "unstructured"},
        {"file_type": ".rtf", "file_category": "unstructured"},
    ]
