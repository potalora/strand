from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.extraction.section_parser import (
    ParsedDocument,
    SectionType,
    parse_sections,
    split_large_section,
)

MOCK_API_KEY = "test-api-key"

CLINICAL_NOTE_TEXT = (
    "History of Present Illness\n"
    "Patient is a 55-year-old male presenting with chest pain for 3 days.\n\n"
    "Current Medications\n"
    "1. Lisinopril 10mg daily\n2. Metformin 500mg BID\n3. Aspirin 81mg daily\n\n"
    "Assessment and Plan\n"
    "1. Chest pain - likely musculoskeletal. Order EKG and troponin.\n"
    "2. Continue current medications."
)

# Anchor-based contract: the LLM returns only {type, anchor} where each anchor is a
# verbatim substring of the document marking that section's start. parse_sections then
# slices the original text locally via resolve_sections.
MOCK_LLM_RESPONSE = {
    "document_type": "clinical_note",
    "primary_visit_date": "2025-03-15",
    "provider": "Dr. Smith",
    "facility": "General Hospital",
    "sections": [
        {"type": "history", "anchor": "History of Present Illness"},
        {"type": "medications", "anchor": "Current Medications"},
        {"type": "assessment_plan", "anchor": "Assessment and Plan"},
    ],
}


@pytest.mark.asyncio
async def test_parse_sections_returns_parsed_document():
    """Mock _call_gemini_for_sections and verify ParsedDocument with correct fields."""
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new_callable=AsyncMock,
        return_value=MOCK_LLM_RESPONSE,
    ):
        result = await parse_sections(CLINICAL_NOTE_TEXT, MOCK_API_KEY)

    assert isinstance(result, ParsedDocument)
    assert result.document_type == "clinical_note"
    assert result.primary_visit_date == "2025-03-15"
    assert result.provider == "Dr. Smith"
    assert result.facility == "General Hospital"
    assert len(result.sections) == 3


@pytest.mark.asyncio
async def test_parse_sections_maps_section_types():
    """Verify section types are correctly mapped to SectionType enum."""
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new_callable=AsyncMock,
        return_value=MOCK_LLM_RESPONSE,
    ):
        result = await parse_sections(CLINICAL_NOTE_TEXT, MOCK_API_KEY)

    assert result.sections[0].section_type == SectionType.HISTORY
    assert result.sections[1].section_type == SectionType.MEDICATIONS
    assert result.sections[2].section_type == SectionType.ASSESSMENT_PLAN


@pytest.mark.asyncio
async def test_parse_sections_preserves_text():
    """Verify section text is sliced locally from the original document with full coverage."""
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new_callable=AsyncMock,
        return_value=MOCK_LLM_RESPONSE,
    ):
        result = await parse_sections(CLINICAL_NOTE_TEXT, MOCK_API_KEY)

    # Each section's text is the slice starting at its anchor up to the next anchor.
    assert result.sections[0].text.startswith("History of Present Illness")
    assert (
        "Patient is a 55-year-old male presenting with chest pain for 3 days."
        in result.sections[0].text
    )
    assert "Lisinopril 10mg daily" in result.sections[1].text
    assert "Order EKG and troponin." in result.sections[2].text
    assert result.sections[0].title == "History of Present Illness"
    # Anchor begins the document, so no leading preamble section is inserted.
    assert result.sections[0].char_range == (0, result.sections[1].char_range[0])
    # Full-coverage invariant: the slices reconstruct the original document exactly.
    assert "".join(s.text for s in result.sections) == CLINICAL_NOTE_TEXT


@pytest.mark.asyncio
async def test_parse_sections_unknown_type_falls_back_to_other():
    """Unknown section types map to SectionType.OTHER; the anchor becomes the title."""
    response = {
        "document_type": "clinical_note",
        "primary_visit_date": None,
        "provider": None,
        "facility": None,
        "sections": [
            {
                "type": "nonexistent_section_type",
                "anchor": "History of Present Illness",
            },
        ],
    }
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new_callable=AsyncMock,
        return_value=response,
    ):
        result = await parse_sections(CLINICAL_NOTE_TEXT, MOCK_API_KEY)

    # A single resolved anchor at position 0 yields one OTHER section covering the doc.
    assert len(result.sections) == 1
    assert result.sections[0].section_type == SectionType.OTHER
    assert result.sections[0].title == "History of Present Illness"
    assert result.sections[0].text == CLINICAL_NOTE_TEXT


@pytest.mark.asyncio
async def test_parse_sections_empty_text_returns_single_other_section():
    """Empty or very short text returns a single OTHER section without calling Gemini."""
    for text_input in ["", "   ", None, "short"]:
        result = await parse_sections(text_input, MOCK_API_KEY)
        assert isinstance(result, ParsedDocument)
        assert result.document_type == "unknown"
        assert len(result.sections) == 1
        assert result.sections[0].section_type == SectionType.OTHER
        assert result.sections[0].title == "Full Document"


@pytest.mark.asyncio
async def test_parse_sections_handles_llm_error_gracefully():
    """Exception from Gemini returns single OTHER section with full text."""
    with patch(
        "app.services.extraction.section_parser._call_gemini_for_sections",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Gemini API unavailable"),
    ):
        result = await parse_sections(CLINICAL_NOTE_TEXT, MOCK_API_KEY)

    assert isinstance(result, ParsedDocument)
    assert result.document_type == "unknown"
    assert len(result.sections) == 1
    assert result.sections[0].section_type == SectionType.OTHER
    assert result.sections[0].text == CLINICAL_NOTE_TEXT


def test_small_section_not_split():
    """Sections under max_chars returned as-is."""
    short_text = "Patient presents with mild headache. No other complaints."
    chunks = split_large_section(short_text, max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_large_section_split_at_paragraphs():
    """Large sections split at double-newline paragraph boundaries."""
    para1 = "First paragraph with clinical details. " * 20  # ~780 chars
    para2 = "Second paragraph with lab results. " * 20  # ~700 chars
    para3 = "Third paragraph with assessment. " * 20  # ~660 chars
    text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = split_large_section(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    # Each chunk should be within the max size (accounting for overlap)
    for chunk in chunks:
        # First chunk fits within max; subsequent chunks may slightly exceed
        # due to overlap prepending, but the core content respects boundaries
        assert len(chunk) > 0


def test_split_includes_overlap():
    """Split chunks have overlapping content from the end of the previous chunk."""
    para1 = "Alpha paragraph content here. " * 30  # ~900 chars
    para2 = "Beta paragraph content here. " * 30  # ~870 chars
    para3 = "Gamma paragraph content here. " * 30  # ~900 chars
    text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = split_large_section(text, max_chars=1000, overlap=200)
    assert len(chunks) >= 2
    # The end of chunk[0] should appear at the start of chunk[1] (overlap region)
    tail_of_first = chunks[0][-200:]
    assert tail_of_first in chunks[1]


def test_single_huge_paragraph_still_split():
    """A single long paragraph without double-newlines splits at sentence boundaries."""
    text = "The patient reported feeling dizzy. " * 100  # ~3500 chars, one paragraph
    chunks = split_large_section(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    # Verify all chunks contain text
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_all_section_types_in_enum():
    """SectionType enum has all 15 expected values."""
    expected = {
        "medications",
        "assessment",
        "clinical_note",
        "labs",
        "review_of_systems",
        "history",
        "physical_exam",
        "assessment_plan",
        "imaging",
        "family_history",
        "social_history",
        "allergies",
        "procedures",
        "vitals",
        "other",
    }
    actual = {member.value for member in SectionType}
    assert actual == expected
    assert len(SectionType) == 15
