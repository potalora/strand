"""Integration tests for XDM package detection and ingestion via coordinator."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from app.services.dedup.orchestrator import DedupSummary
from tests.conftest import FIXTURES_DIR, auth_headers

SYNTHETIC_CDA_DIR = FIXTURES_DIR / "synthetic_cda"

# Patch path for dedup (local import inside ingest_file)
PATCH_DEDUP = "app.services.dedup.orchestrator.run_upload_dedup"


def _create_xdm_zip() -> bytes:
    """Create a ZIP with IHE XDM structure from synthetic CDA fixtures."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in SYNTHETIC_CDA_DIR.iterdir():
            if f.is_file() and f.suffix.upper() == ".XML":
                zf.write(f, f"IHE_XDM/Patient1/{f.name}")
    buf.seek(0)
    return buf.getvalue()


def _create_plain_json_zip() -> bytes:
    """Create a ZIP with a plain FHIR JSON file (no XDM metadata)."""
    fhir_path = FIXTURES_DIR / "sample_fhir_bundle.json"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(fhir_path, "bundle.json")
    buf.seek(0)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_coordinator_detects_xdm_in_zip(
    client: AsyncClient, db_session: AsyncSession
):
    """ZIP with METADATA.XML routes through XDM pipeline, records inserted."""
    headers, uid = await auth_headers(client)
    zip_data = _create_xdm_zip()

    with (
        patch(PATCH_DEDUP, new_callable=AsyncMock) as mock_dedup,
        patch("app.services.ingestion.coordinator.parse_cda_document") as mock_parse,
    ):
        mock_dedup.return_value = DedupSummary()

        # Return some synthetic records from the CDA parser
        mock_parse.return_value = [
            {
                "record_type": "condition",
                "fhir_resource_type": "Condition",
                "fhir_resource": {
                    "resourceType": "Condition",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "38341003",
                                "display": "Hypertension",
                            }
                        ]
                    },
                    "_extraction_metadata": {
                        "source_format": "cda_r2",
                        "source_document": "DOC0001.XML",
                        "source_institution": "Synthetic Health Clinic",
                    },
                },
                "source_format": "cda_r2",
                "effective_date": None,
                "effective_date_end": None,
                "status": "active",
                "category": ["encounter-diagnosis"],
                "code_system": "http://snomed.info/sct",
                "code_value": "38341003",
                "code_display": "Hypertension",
                "display_text": "Hypertension",
            }
        ]

        resp = await client.post(
            "/api/v1/upload",
            headers=headers,
            files={"file": ("xdm_export.zip", zip_data, "application/zip")},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] in ("completed", "dedup_scanning")
    # parse_cda_document called once per XML doc (2 docs in manifest)
    assert mock_parse.call_count == 2
    # 1 record per doc x 2 docs = 2 records, dedup may collapse some
    assert data["records_inserted"] >= 1


@pytest.mark.asyncio
async def test_xdm_skips_pdf_in_package(client: AsyncClient, db_session: AsyncSession):
    """PDF entry in XDM manifest is skipped (not processed as unstructured)."""
    headers, uid = await auth_headers(client)
    zip_data = _create_xdm_zip()

    with (
        patch(PATCH_DEDUP, new_callable=AsyncMock) as mock_dedup,
        patch("app.services.ingestion.coordinator.parse_cda_document") as mock_parse,
    ):
        mock_dedup.return_value = DedupSummary()
        mock_parse.return_value = [
            {
                "record_type": "condition",
                "fhir_resource_type": "Condition",
                "fhir_resource": {
                    "resourceType": "Condition",
                    "_extraction_metadata": {
                        "source_format": "cda_r2",
                        "source_document": "DOC0001.XML",
                        "source_institution": "",
                    },
                },
                "source_format": "cda_r2",
                "effective_date": None,
                "effective_date_end": None,
                "status": None,
                "category": [],
                "code_system": None,
                "code_value": None,
                "code_display": None,
                "display_text": "Condition",
            }
        ]

        resp = await client.post(
            "/api/v1/upload",
            headers=headers,
            files={"file": ("xdm_export.zip", zip_data, "application/zip")},
        )

    assert resp.status_code == 202
    data = resp.json()
    # The PDF should show up in errors with structured_preferred reason
    pdf_errors = [
        e for e in data.get("errors", []) if e.get("reason") == "structured_preferred"
    ]
    assert len(pdf_errors) == 1
    assert "SCAN0001.PDF" in pdf_errors[0]["file"]
    # No unstructured uploads created for XDM path
    assert data.get("unstructured_uploads", []) == []


@pytest.mark.asyncio
async def test_xdm_creates_cda_records(client: AsyncClient, db_session: AsyncSession):
    """Records from XDM ingestion have source_format='cda_r2'."""
    headers, uid = await auth_headers(client)
    zip_data = _create_xdm_zip()

    with (
        patch(PATCH_DEDUP, new_callable=AsyncMock) as mock_dedup,
        patch("app.services.ingestion.coordinator.parse_cda_document") as mock_parse,
    ):
        mock_dedup.return_value = DedupSummary()
        mock_parse.return_value = [
            {
                "record_type": "medication",
                "fhir_resource_type": "MedicationRequest",
                "fhir_resource": {
                    "resourceType": "MedicationRequest",
                    "status": "active",
                    "medicationCodeableConcept": {"text": "Metformin 500mg"},
                    "_extraction_metadata": {
                        "source_format": "cda_r2",
                        "source_document": "DOC0001.XML",
                        "source_institution": "Synthetic Health Clinic",
                    },
                },
                "source_format": "cda_r2",
                "effective_date": None,
                "effective_date_end": None,
                "status": "active",
                "category": [],
                "code_system": None,
                "code_value": None,
                "code_display": "Metformin 500mg",
                "display_text": "Metformin 500mg",
            }
        ]

        resp = await client.post(
            "/api/v1/upload",
            headers=headers,
            files={"file": ("xdm_export.zip", zip_data, "application/zip")},
        )

    assert resp.status_code == 202

    # Query DB for inserted records
    from uuid import UUID as _UUID

    result = await db_session.execute(
        select(HealthRecord).where(HealthRecord.user_id == _UUID(uid))
    )
    records = result.scalars().all()
    assert len(records) >= 1
    for rec in records:
        assert rec.source_format == "cda_r2"


@pytest.mark.asyncio
async def test_xdm_intra_upload_dedup(client: AsyncClient, db_session: AsyncSession):
    """Dedup collapses duplicate records across CDA documents."""
    headers, uid = await auth_headers(client)
    zip_data = _create_xdm_zip()

    # Simulate: each doc returns the SAME record (a duplicate)
    shared_record = {
        "record_type": "condition",
        "fhir_resource_type": "Condition",
        "fhir_resource": {
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "44054006",
                        "display": "Type 2 diabetes",
                    }
                ]
            },
            "_extraction_metadata": {
                "source_format": "cda_r2",
                "source_document": "DOC0001.XML",
                "source_institution": "Synthetic Health Clinic",
            },
        },
        "source_format": "cda_r2",
        "effective_date": None,
        "effective_date_end": None,
        "status": "active",
        "category": ["encounter-diagnosis"],
        "code_system": "http://snomed.info/sct",
        "code_value": "44054006",
        "code_display": "Type 2 diabetes",
        "display_text": "Type 2 diabetes mellitus",
    }

    call_count = 0

    def mock_parse_side_effect(file_path, manifest_doc=None):
        nonlocal call_count
        call_count += 1
        # Return a copy with different source_document metadata
        import copy

        rec = copy.deepcopy(shared_record)
        doc_name = f"DOC000{call_count}.XML"
        rec["fhir_resource"]["_extraction_metadata"]["source_document"] = doc_name
        return [rec]

    with (
        patch(PATCH_DEDUP, new_callable=AsyncMock) as mock_dedup,
        patch(
            "app.services.ingestion.coordinator.parse_cda_document",
            side_effect=mock_parse_side_effect,
        ),
    ):
        mock_dedup.return_value = DedupSummary()

        resp = await client.post(
            "/api/v1/upload",
            headers=headers,
            files={"file": ("xdm_export.zip", zip_data, "application/zip")},
        )

    assert resp.status_code == 202
    data = resp.json()
    # 2 docs parsed, but dedup collapses the identical record
    assert data["records_inserted"] == 1


@pytest.mark.asyncio
async def test_non_xdm_zip_uses_existing_pipeline(
    client: AsyncClient, db_session: AsyncSession
):
    """ZIP without METADATA.XML falls through to existing FHIR/Epic routing."""
    headers, uid = await auth_headers(client)
    zip_data = _create_plain_json_zip()

    with patch(PATCH_DEDUP, new_callable=AsyncMock) as mock_dedup:
        mock_dedup.return_value = DedupSummary()

        resp = await client.post(
            "/api/v1/upload",
            headers=headers,
            files={"file": ("fhir_bundle.zip", zip_data, "application/zip")},
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] in ("completed", "dedup_scanning")
    # The sample FHIR bundle has 17 clinical resources
    assert data["records_inserted"] == 17
