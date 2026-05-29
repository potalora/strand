from __future__ import annotations

from app.services.ingestion.content_hash import canonicalize, content_hash


def _base_resource() -> dict:
    return {
        "resourceType": "Condition",
        "id": "abc",
        "code": {"text": "Type 2 Diabetes"},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "meta": {"lastUpdated": "2024-01-01T00:00:00Z"},
    }


def test_hash_is_deterministic():
    assert content_hash(_base_resource()) == content_hash(_base_resource())


def test_meta_lastupdated_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["meta"]["lastUpdated"] = "2025-12-31T23:59:59Z"
    assert content_hash(a) == content_hash(b)


def test_extraction_metadata_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["_extraction_metadata"] = {"source_format": "cda_r2", "source_document": "DOC0001.XML"}
    assert content_hash(a) == content_hash(b)


def test_narrative_text_div_does_not_change_hash():
    a = _base_resource()
    b = _base_resource()
    b["text"] = {"status": "generated", "div": "<div>rendered html</div>"}
    assert content_hash(a) == content_hash(b)


def test_clinical_status_change_changes_hash():
    a = _base_resource()
    b = _base_resource()
    b["clinicalStatus"]["coding"][0]["code"] = "resolved"
    assert content_hash(a) != content_hash(b)


def test_key_order_does_not_change_hash():
    a = {"resourceType": "Observation", "code": {"text": "BP"}, "valueString": "120/80"}
    b = {"valueString": "120/80", "code": {"text": "BP"}, "resourceType": "Observation"}
    assert content_hash(a) == content_hash(b)


def test_canonicalize_strips_noise_keys():
    res = _base_resource()
    res["_extraction_metadata"] = {"x": 1}
    res["text"] = {"div": "<div/>"}
    out = canonicalize(res)
    assert "_extraction_metadata" not in out
    assert "text" not in out
    assert "lastUpdated" not in out.get("meta", {})


def test_hash_is_hex_sha256():
    h = content_hash(_base_resource())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
