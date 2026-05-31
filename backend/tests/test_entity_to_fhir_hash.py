from __future__ import annotations

from uuid import uuid4

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.entity_to_fhir import entity_to_health_record_dict
from app.services.ingestion.content_hash import content_hash


def _entity() -> ExtractedEntity:
    return ExtractedEntity(
        entity_class="condition",
        text="Type 2 Diabetes",
        attributes={"status": "active"},
        start_pos=0,
        end_pos=15,
        confidence=0.9,
    )


def test_extracted_record_has_content_hash():
    rec = entity_to_health_record_dict(_entity(), uuid4(), uuid4(), uuid4())
    assert rec is not None
    assert rec["content_hash"] == content_hash(rec["fhir_resource"])


def test_content_hash_is_hex_sha256():
    rec = entity_to_health_record_dict(_entity(), uuid4(), uuid4(), uuid4())
    assert rec is not None
    assert len(rec["content_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in rec["content_hash"])


def test_non_storable_entity_still_returns_none():
    e = ExtractedEntity(entity_class="dosage", text="10mg", attributes={}, start_pos=0, end_pos=4, confidence=0.5)
    assert entity_to_health_record_dict(e, uuid4(), uuid4(), uuid4()) is None
