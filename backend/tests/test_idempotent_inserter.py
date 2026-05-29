from __future__ import annotations

from app.services.ingestion.idempotent_inserter import Plan, plan_batch


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
