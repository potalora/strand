"""TDD for frontend TODO A3: surface the record subject's identity.

- GET /auth/me must include created_at (powers "Member since").
- GET /dashboard/patients must include the decrypted name + birth_date
  (powers the Overview masthead + System account card). This is the user's
  OWN data, user-scoped, so returning it is appropriate.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.encryption import encrypt_field
from app.models.patient import Patient
from tests.conftest import auth_headers


@pytest.mark.asyncio
async def test_auth_me_includes_created_at(client: AsyncClient):
    headers, _ = await auth_headers(client)
    resp = await client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "created_at" in body
    assert body["created_at"] is not None


@pytest.mark.asyncio
async def test_patients_includes_decrypted_name_and_dob(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = Patient(
        id=uuid4(),
        user_id=UUID(uid),
        fhir_id="subject-1",
        gender="female",
        name_encrypted=encrypt_field("Eleanor M. Reyes"),
        birth_date_encrypted=encrypt_field("1967-03-14"),
    )
    db_session.add(patient)
    await db_session.commit()

    resp = await client.get("/api/v1/dashboard/patients", headers=headers)
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["name"] == "Eleanor M. Reyes"
    assert item["birth_date"] == "1967-03-14"
    assert item["gender"] == "female"


@pytest.mark.asyncio
async def test_patients_without_encrypted_name_returns_null(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = Patient(id=uuid4(), user_id=UUID(uid), fhir_id="subject-2", gender="male")
    db_session.add(patient)
    await db_session.commit()

    resp = await client.get("/api/v1/dashboard/patients", headers=headers)
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["name"] is None
    assert item["birth_date"] is None
