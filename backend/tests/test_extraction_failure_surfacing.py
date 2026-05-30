from __future__ import annotations

from app.api.upload import _collect_entities
from app.services.extraction.entity_extractor import ExtractedEntity, ExtractionResult


def _ok(n, section="other"):
    ents = [ExtractedEntity(entity_class="condition", text=f"c{i}") for i in range(n)]
    return (ExtractionResult(source_file="f", source_text="t", entities=ents), section)


def test_collect_counts_ok_entities():
    results = [_ok(2), _ok(3)]
    entities, failed = _collect_entities(results, total_chunks=2)
    assert len(entities) == 5
    assert failed == 0
    assert all(e.attributes.get("_source_section") == "other" for e in entities)


def test_collect_counts_error_result_as_failure():
    err = (ExtractionResult(source_file="f", source_text="t", error="429 rate exceeded"), "labs")
    entities, failed = _collect_entities([_ok(2), err], total_chunks=2)
    assert len(entities) == 2
    assert failed == 1


def test_collect_counts_raised_exception_as_failure():
    results = [_ok(1), RuntimeError("boom")]
    entities, failed = _collect_entities(results, total_chunks=2)
    assert len(entities) == 1
    assert failed == 1
