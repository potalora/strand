from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.extraction.section_parser import SectionType, parse_sections


@pytest.mark.asyncio
async def test_large_document_parses_into_multiple_sections():
    """A ~40KB doc the old full-text path would truncate now parses into N sections."""
    body = "x" * 13000
    text = f"MEDICATIONS\n{body}\nLABS\n{body}\nASSESSMENT\n{body}"  # ~39KB, 3 sections
    raw = {
        "document_type": "clinical_note",
        "primary_visit_date": None, "provider": None, "facility": None,
        "sections": [
            {"type": "medications", "anchor": "MEDICATIONS"},
            {"type": "labs", "anchor": "LABS"},
            {"type": "assessment", "anchor": "ASSESSMENT"},
        ],
    }
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new=AsyncMock(return_value=raw),
    ):
        doc = await parse_sections(text, "key")

    assert [s.section_type for s in doc.sections] == [
        SectionType.MEDICATIONS, SectionType.LABS, SectionType.ASSESSMENT,
    ]
    # full coverage, no text lost on a large doc
    assert "".join(s.text for s in doc.sections) == text
    # each section is substantial (proves real slicing, not single-section fallback)
    assert all(len(s.text) > 10000 for s in doc.sections)
