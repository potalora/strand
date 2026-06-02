"""TDD for frontend TODO C2: GET /records/export?format=fhir-bundle.

Bulk-exports all of the user's (non-deleted, non-duplicate) records as one
FHIR R4 collection Bundle. Powers Admin -> System "Export all (FHIR)".
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient, seed_test_records


@pytest.mark.asyncio
async def test_export_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/records/export?format=fhir-bundle")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_export_returns_collection_bundle(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=5)

    resp = await client.get("/api/v1/records/export?format=fhir-bundle", headers=headers)
    assert resp.status_code == 200
    bundle = resp.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] == 5
    assert len(bundle["entry"]) == 5
    # each entry wraps the stored FHIR resource
    assert all("resource" in e and e["resource"].get("resourceType") for e in bundle["entry"])


@pytest.mark.asyncio
async def test_export_excludes_deleted_and_duplicates(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    await seed_test_records(db_session, uid, patient.id, count=3)
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=patient.id, user_id=UUID(uid), record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", display_text="deleted", deleted_at=datetime.now(timezone.utc),
        )
    )
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=patient.id, user_id=UUID(uid), record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", display_text="dup", is_duplicate=True,
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/records/export?format=fhir-bundle", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 3


@pytest.mark.asyncio
async def test_export_rejects_unsupported_format(client: AsyncClient, db_session: AsyncSession):
    headers, _ = await auth_headers(client)
    resp = await client.get("/api/v1/records/export?format=csv", headers=headers)
    assert resp.status_code == 400
