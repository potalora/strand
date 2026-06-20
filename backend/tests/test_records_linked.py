"""GET /records/{id}/linked — the "From this visit" read-path.

Surfaces records sharing an encounter via ``linked_encounter_id`` (the existing
AI-extraction link), user-scoped, in the rich timeline-event shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient


def _rec(uid, patient_id, rt, fhir_type, fhir, text, *, linked=None, date=(2026, 2, 28)):
    return HealthRecord(
        id=uuid4(),
        patient_id=patient_id,
        user_id=UUID(uid),
        record_type=rt,
        fhir_resource_type=fhir_type,
        fhir_resource=fhir,
        source_format="ai_extracted",
        display_text=text,
        effective_date=datetime(*date, tzinfo=timezone.utc),
        linked_encounter_id=linked,
    )


@pytest.mark.asyncio
async def test_linked_records_returns_visit_siblings(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    enc = _rec(uid, patient.id, "encounter", "Encounter",
               {"resourceType": "Encounter", "class": {"code": "AMB"}}, "Office visit")
    db_session.add(enc)
    await db_session.flush()

    db_session.add(_rec(uid, patient.id, "condition", "Condition",
                        {"resourceType": "Condition", "clinicalStatus": {"coding": [{"code": "active"}]}},
                        "reflux", linked=enc.id))
    db_session.add(_rec(uid, patient.id, "medication", "MedicationRequest",
                        {"resourceType": "MedicationRequest", "status": "active"},
                        "omeprazole", linked=enc.id))
    # Unlinked record must NOT appear.
    db_session.add(_rec(uid, patient.id, "condition", "Condition",
                        {"resourceType": "Condition"}, "unrelated", date=(2026, 1, 1)))
    await db_session.commit()

    resp = await client.get(f"/api/v1/records/{enc.id}/linked", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    texts = {e["display_text"] for e in body}
    assert texts == {"reflux", "omeprazole"}
    assert "unrelated" not in texts
    # Rich timeline-event shape: previews populated (condition status flag).
    by = {e["display_text"]: e for e in body}
    assert by["reflux"]["preview"]["flag"] == "ACTIVE"


@pytest.mark.asyncio
async def test_linked_records_empty_for_unlinked_encounter(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    enc = _rec(uid, patient.id, "encounter", "Encounter", {"resourceType": "Encounter"}, "Lonely visit")
    db_session.add(enc)
    await db_session.commit()
    resp = await client.get(f"/api/v1/records/{enc.id}/linked", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_linked_records_user_scoped(client: AsyncClient, db_session: AsyncSession):
    """User B must never see records linked under user A's encounter."""
    headers_a, uid_a = await auth_headers(client)
    patient_a = await create_test_patient(db_session, uid_a)
    enc = _rec(uid_a, patient_a.id, "encounter", "Encounter", {"resourceType": "Encounter"}, "A's visit")
    db_session.add(enc)
    await db_session.flush()
    db_session.add(_rec(uid_a, patient_a.id, "condition", "Condition",
                        {"resourceType": "Condition"}, "A private dx", linked=enc.id))
    await db_session.commit()

    headers_b, _ = await auth_headers(client, email="userb_linked@example.com")
    resp = await client.get(f"/api/v1/records/{enc.id}/linked", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []  # scoped to user B → nothing


@pytest.mark.asyncio
async def test_linked_records_unauthenticated(client: AsyncClient):
    resp = await client.get(f"/api/v1/records/{uuid4()}/linked")
    assert resp.status_code == 401
