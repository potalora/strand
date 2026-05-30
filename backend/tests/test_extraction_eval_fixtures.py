from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.extraction.entity_to_fhir import ENTITY_TO_RECORD_TYPE

_FIX = Path(__file__).resolve().parent / "fixtures" / "extraction_eval"
_PAIRS = [("transcript_visit.txt", "transcript_visit.expected.json"),
          ("phone_note.txt", "phone_note.expected.json")]
_VALID = set(ENTITY_TO_RECORD_TYPE.keys())


@pytest.mark.parametrize("txt,gt", _PAIRS)
def test_fixture_pair_exists_and_parses(txt, gt):
    assert (_FIX / txt).read_text().strip()
    data = json.loads((_FIX / gt).read_text())
    assert "expected" in data


@pytest.mark.parametrize("txt,gt", _PAIRS)
def test_expected_entity_classes_are_valid(txt, gt):
    data = json.loads((_FIX / gt).read_text())
    for item in data.get("expected", []) + data.get("must_not_extract", []):
        assert item["entity_class"] in _VALID, f"unknown class {item['entity_class']}"
