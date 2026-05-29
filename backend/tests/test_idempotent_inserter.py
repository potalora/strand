from __future__ import annotations

import pytest
from app.models.record import HealthRecord
from app.models.record_version import RecordVersion
from app.services.ingestion.idempotent_inserter import plan_batch
from sqlalchemy import select
from tests.conftest import auth_headers, create_test_patient


def _rec(fmt="fhir_r4", rid="c1", code="active"):
    return {
        "source_format": fmt,
        "fhir_resource": {
            "resourceType": "Condition", "id": rid,
            "clinicalStatus": {"coding": [{"code": code}]},
        },
    }


def test_new_record_is_insert():
    plan = plan_batch([_rec(rid="c1")], existing={})
    assert [p.action for p in plan] == ["insert"]
    assert plan[0].identity is not None
    assert plan[0].content_hash is not None


def test_identical_existing_is_skip():
    rec = _rec(rid="c1", code="active")
    key = ("fhir", "Condition/c1")
    from app.services.ingestion.content_hash import content_hash
    existing = {key: ("row-uuid", content_hash(rec["fhir_resource"]), 1)}
    plan = plan_batch([rec], existing=existing)
    assert [p.action for p in plan] == ["skip"]


def test_changed_existing_is_update():
    rec = _rec(rid="c1", code="resolved")
    key = ("fhir", "Condition/c1")
    from app.services.ingestion.content_hash import content_hash
    old_hash = content_hash(_rec(rid="c1", code="active")["fhir_resource"])
    existing = {key: ("row-uuid", old_hash, 1)}
    plan = plan_batch([rec], existing=existing)
    assert plan[0].action == "update"
    assert plan[0].existing_id == "row-uuid"
    assert plan[0].new_version == 2


def test_no_identity_is_insert_fallthrough():
    rec = {"source_format": "fhir_r4", "fhir_resource": {"resourceType": "Condition"}}
    plan = plan_batch([rec], existing={})
    assert plan[0].action == "insert"
    assert plan[0].identity is None


def test_within_batch_duplicate_identity_second_is_skip():
    r1 = _rec(rid="dup", code="active")
    r2 = _rec(rid="dup", code="active")
    plan = plan_batch([r1, r2], existing={})
    assert [p.action for p in plan] == ["insert", "skip"]


def test_within_batch_duplicate_changed_second_is_update_of_first():
    r1 = _rec(rid="dup", code="active")
    r2 = _rec(rid="dup", code="resolved")
    plan = plan_batch([r1, r2], existing={})
    assert plan[0].action == "insert"
    assert plan[1].action == "update_pending"  # updates the just-planned insert


@pytest.mark.asyncio
async def test_insert_then_reingest_identical_is_unchanged(client, db_session):
    from app.services.ingestion.idempotent_inserter import idempotent_insert_records
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    def batch():
        return [{
            "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
            "record_type": "condition", "fhir_resource_type": "Condition",
            "fhir_resource": {"resourceType": "Condition", "id": "c1",
                              "clinicalStatus": {"coding": [{"code": "active"}]}},
            "source_format": "fhir_r4", "display_text": "Cond",
        }]

    s1 = await idempotent_insert_records(db_session, batch())
    await db_session.commit()
    assert s1["inserted"] == 1 and s1["unchanged"] == 0

    s2 = await idempotent_insert_records(db_session, batch())
    await db_session.commit()
    assert s2["inserted"] == 0 and s2["unchanged"] == 1

    rows = (await db_session.execute(select(HealthRecord))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_changed_reingest_updates_and_snapshots(client, db_session):
    from app.services.ingestion.idempotent_inserter import idempotent_insert_records
    _, uid = await auth_headers(client)
    patient = await create_test_patient(db_session, uid)

    def batch(code):
        return [{
            "user_id": patient.user_id, "patient_id": patient.id, "source_file_id": None,
            "record_type": "condition", "fhir_resource_type": "Condition",
            "fhir_resource": {"resourceType": "Condition", "id": "c2",
                              "clinicalStatus": {"coding": [{"code": code}]}},
            "source_format": "fhir_r4", "display_text": "Cond", "status": code,
        }]

    await idempotent_insert_records(db_session, batch("active"))
    await db_session.commit()
    s2 = await idempotent_insert_records(db_session, batch("resolved"))
    await db_session.commit()
    assert s2["updated"] == 1

    row = (await db_session.execute(select(HealthRecord))).scalars().one()
    assert row.version == 2
    assert row.fhir_resource["clinicalStatus"]["coding"][0]["code"] == "resolved"

    versions = (await db_session.execute(select(RecordVersion))).scalars().all()
    assert len(versions) == 1  # the prior (active) snapshot
    assert versions[0].fhir_resource["clinicalStatus"]["coding"][0]["code"] == "active"
