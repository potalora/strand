"""WS-A fidelity: on-device extraction over REAL Epic RTF clinical notes.

Uses the real, gitignored corpus in ``test_data/Requested Record/Rich Text/``.
Validates that the now-default local/hybrid path produces clinical entities on
real notes — fully on-device (no Gemini, no network). Skips when the corpus or
the optional clinical-NLP models aren't present.
"""

from __future__ import annotations

import pathlib

import pytest
from striprtf.striprtf import rtf_to_text

from tests.conftest import private_fixture_root

# Real Epic RTF corpus lives off-repo (gitignored), under
# <REAL_MEDICAL_FIXTURES_DIR>/raw/Requested Record/Rich Text/. No in-repo fallback.
_FIXROOT = private_fixture_root()
RTF_DIR = (_FIXROOT / "raw" / "Requested Record" / "Rich Text") if _FIXROOT else None

pytestmark = [pytest.mark.fidelity, pytest.mark.asyncio]


def _real_notes() -> list[pathlib.Path]:
    return sorted(RTF_DIR.glob("*.RTF")) if RTF_DIR and RTF_DIR.is_dir() else []


def _models_available() -> bool:
    from app.services.extraction.clinical_context import get_clinical_context
    from app.services.extraction.local_ner import get_local_ner

    return get_local_ner().available and get_clinical_context().available


async def test_local_extraction_on_real_rtf_notes():
    notes = _real_notes()
    if not notes:
        pytest.skip("real RTF notes absent (set REAL_MEDICAL_FIXTURES_DIR)")
    if not _models_available():
        pytest.skip("scispaCy/medspaCy not installed (.[clinical-nlp])")

    from app.services.extraction.clinical_context import get_clinical_context
    from app.services.extraction.extraction_engine import run_clinical_extraction
    from app.services.extraction.local_ner import get_local_ner

    sample = notes[:5]
    notes_with_entities = 0
    total = 0
    classes: set[str] = set()
    for path in sample:
        text = rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        result = await run_clinical_extraction(
            text,
            engine="local",
            ner=get_local_ner(),
            context=get_clinical_context(),
            gemini_section_extract=None,
            confidence_threshold=0.6,
        )
        if result.entities:
            notes_with_entities += 1
        total += len(result.entities)
        classes.update(e.entity_class for e in result.entities)

    assert notes_with_entities >= 1, "local extraction found nothing in real notes"
    assert total >= 5, f"expected several entities across {len(sample)} real notes, got {total}"
    assert classes & {"medication", "condition"}, f"no clinical classes extracted: {classes}"
