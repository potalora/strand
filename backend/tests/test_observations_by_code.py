"""TDD for GET /api/v1/observations/by-code (BACKEND-TODOs #1, #2, #6).

One entry per distinct observation code_value, sorted by recency of the latest
reading (latest.date desc; tie -> higher count first). Each entry carries
latest/prior readings + a numeric, unit-normalized, ascending series.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.record import HealthRecord
from tests.conftest import auth_headers, create_test_patient


async def _add_obs(
    db,
    uid,
    pid,
    code_value,
    when,
    *,
    value=None,
    unit="%",
    display="Hemoglobin A1c",
    fhir_extra=None,
    source_section=None,
    category=None,
    source_format="fhir_r4",
):
    fhir = {"resourceType": "Observation"}
    if value is not None:
        fhir["valueQuantity"] = {"value": value, "unit": unit}
    if fhir_extra:
        fhir.update(fhir_extra)
    db.add(
        HealthRecord(
            id=uuid4(),
            patient_id=pid,
            user_id=uid,
            record_type="observation",
            fhir_resource_type="Observation",
            fhir_resource=fhir,
            source_format=source_format,
            code_system="http://loinc.org",
            code_value=code_value,
            code_display=display,
            display_text=f"{display}: {value}{unit}" if value is not None else display,
            effective_date=when,
            source_section=source_section,
            category=category,
        )
    )


@pytest.mark.asyncio
async def test_by_code_requires_auth(client: AsyncClient):
    resp = await client.get("/api/v1/observations/by-code")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_by_code_groups_one_entry_per_code(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(db_session, u, p, "4548-4", datetime(2021, 1, 1, tzinfo=timezone.utc), value=7.0)
    await _add_obs(db_session, u, p, "4548-4", datetime(2022, 1, 1, tzinfo=timezone.utc), value=6.6)
    await _add_obs(
        db_session, u, p, "13457-7", datetime(2023, 1, 1, tzinfo=timezone.utc),
        value=120, unit="mg/dL", display="LDL",
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    codes = {item["code"] for item in data["items"]}
    assert codes == {"4548-4", "13457-7"}
    a1c = next(i for i in data["items"] if i["code"] == "4548-4")
    assert a1c["count"] == 2
    assert a1c["display"] == "Hemoglobin A1c"


@pytest.mark.asyncio
async def test_by_code_latest_and_prior(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(db_session, u, p, "4548-4", datetime(2021, 1, 1, tzinfo=timezone.utc), value=7.8)
    await _add_obs(db_session, u, p, "4548-4", datetime(2023, 6, 1, tzinfo=timezone.utc), value=6.6)
    await _add_obs(db_session, u, p, "4548-4", datetime(2022, 3, 1, tzinfo=timezone.utc), value=7.1)
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    data = resp.json()
    a1c = next(i for i in data["items"] if i["code"] == "4548-4")
    assert a1c["latest"]["value"] == 6.6
    assert a1c["latest"]["date"].startswith("2023-06-01")
    assert a1c["prior"]["value"] == 7.1
    assert a1c["prior"]["date"].startswith("2022-03-01")
    # series ascending by date, numeric only
    assert [pt["value"] for pt in a1c["series"]] == [7.8, 7.1, 6.6]


@pytest.mark.asyncio
async def test_by_code_single_reading_prior_is_null(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(db_session, u, p, "4548-4", datetime(2023, 6, 1, tzinfo=timezone.utc), value=6.6)
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    a1c = next(i for i in resp.json()["items"] if i["code"] == "4548-4")
    assert a1c["prior"] is None
    assert a1c["count"] == 1
    assert len(a1c["series"]) == 1


@pytest.mark.asyncio
async def test_by_code_reference_range_and_interpretation(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(
        db_session, u, p, "4548-4", datetime(2023, 6, 1, tzinfo=timezone.utc), value=6.6,
        fhir_extra={
            "referenceRange": [{"low": {"value": 4.0}, "high": {"value": 5.6}}],
            "interpretation": [{"coding": [{"code": "H"}]}],
        },
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    a1c = next(i for i in resp.json()["items"] if i["code"] == "4548-4")
    assert a1c["latest"]["ref_low"] == 4.0
    assert a1c["latest"]["ref_high"] == 5.6
    assert a1c["latest"]["interpretation"] == "H"


@pytest.mark.asyncio
async def test_by_code_non_numeric_value_kept_in_latest_excluded_from_series(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    # Blood pressure as components -> "128/78", non-numeric
    await _add_obs(
        db_session, u, p, "85354-9", datetime(2023, 6, 1, tzinfo=timezone.utc),
        value=None, display="Blood pressure",
        fhir_extra={
            "component": [
                {
                    "code": {"coding": [{"code": "8480-6"}]},
                    "valueQuantity": {"value": 128, "unit": "mmHg"},
                },
                {
                    "code": {"coding": [{"code": "8462-4"}]},
                    "valueQuantity": {"value": 78, "unit": "mmHg"},
                },
            ]
        },
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    bp = next(i for i in resp.json()["items"] if i["code"] == "85354-9")
    assert bp["latest"]["value"] == "128/78"
    assert bp["series"] == []  # non-numeric excluded


@pytest.mark.asyncio
async def test_by_code_series_unit_normalized(client: AsyncClient, db_session: AsyncSession):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    # One A1c reported in % and one in mmol/mol; series should be in %.
    await _add_obs(
        db_session, u, p, "4548-4", datetime(2021, 1, 1, tzinfo=timezone.utc), value=7.0, unit="%"
    )
    await _add_obs(
        db_session, u, p, "4548-4", datetime(2022, 1, 1, tzinfo=timezone.utc),
        value=53.0, unit="mmol/mol",
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    a1c = next(i for i in resp.json()["items"] if i["code"] == "4548-4")
    vals = [pt["value"] for pt in a1c["series"]]
    assert vals[0] == 7.0
    assert vals[1] == pytest.approx(7.0, abs=0.05)  # 53 mmol/mol normalized to %


@pytest.mark.asyncio
async def test_by_code_sorted_by_recency_then_count(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    # code A: most-recent reading 2024; code B: most-recent 2022
    await _add_obs(db_session, u, p, "AAA", datetime(2024, 1, 1, tzinfo=timezone.utc), value=1)
    await _add_obs(db_session, u, p, "BBB", datetime(2022, 1, 1, tzinfo=timezone.utc), value=2)
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    codes = [i["code"] for i in resp.json()["items"]]
    assert codes == ["AAA", "BBB"]  # most-recent latest reading first


@pytest.mark.asyncio
async def test_by_code_category_prefers_source_section(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(
        db_session, u, p, "4548-4", datetime(2023, 1, 1, tzinfo=timezone.utc), value=6.6,
        source_section="Diabetes Panel", category=["laboratory"],
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    a1c = next(i for i in resp.json()["items"] if i["code"] == "4548-4")
    assert a1c["category"] == "Diabetes Panel"


@pytest.mark.asyncio
async def test_by_code_excludes_non_observations_and_other_users(
    client: AsyncClient, db_session: AsyncSession
):
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(db_session, u, p, "4548-4", datetime(2023, 1, 1, tzinfo=timezone.utc), value=6.6)
    # a non-observation record sharing a code_value must NOT appear
    db_session.add(
        HealthRecord(
            id=uuid4(), patient_id=p, user_id=u, record_type="condition",
            fhir_resource_type="Condition", fhir_resource={"resourceType": "Condition"},
            source_format="fhir_r4", code_value="4548-4", display_text="cond",
            effective_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["count"] == 1


@pytest.mark.asyncio
async def test_coded_value_uses_coding_display_fallback(client: AsyncClient, db_session: AsyncSession):
    """A coded observation carrying only valueCodeableConcept.coding[].display
    (no .text) must surface that display as the value, not the code's name."""
    headers, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)
    u, p = UUID(uid), patient.id
    await _add_obs(
        db_session, u, p, "72166-2",
        datetime(2026, 1, 6, tzinfo=timezone.utc),
        value=None, display="Tobacco smoking status NHIS", category=["social-history"],
        fhir_extra={"valueCodeableConcept": {"coding": [{"display": "Never smoker", "code": "266919005"}]}},
    )
    await db_session.commit()

    resp = await client.get("/api/v1/observations/by-code", headers=headers)
    assert resp.status_code == 200
    entry = next(i for i in resp.json()["items"] if i["code"] == "72166-2")
    assert entry["latest"]["value"] == "Never smoker"
