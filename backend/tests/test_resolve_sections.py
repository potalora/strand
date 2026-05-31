from __future__ import annotations

from app.services.extraction.section_parser import SectionType, resolve_sections


def test_anchors_found_in_order_full_coverage():
    text = "MEDICATIONS\nmetformin 500mg\nLABS\nA1c 6.1\nASSESSMENT\nstable"
    raw = [
        {"type": "medications", "anchor": "MEDICATIONS"},
        {"type": "labs", "anchor": "LABS"},
        {"type": "assessment", "anchor": "ASSESSMENT"},
    ]
    secs = resolve_sections(text, raw)
    assert [s.section_type for s in secs] == [
        SectionType.MEDICATIONS, SectionType.LABS, SectionType.ASSESSMENT,
    ]
    assert "".join(s.text for s in secs) == text  # full-coverage invariant
    assert secs[0].char_range == (0, text.index("LABS"))


def test_leading_text_becomes_other_preamble():
    text = "Patient John Doe, DOB 1/1/1970\nMEDICATIONS\nmetformin"
    raw = [{"type": "medications", "anchor": "MEDICATIONS"}]
    secs = resolve_sections(text, raw)
    assert secs[0].section_type == SectionType.OTHER
    assert secs[0].text.startswith("Patient John Doe")
    assert secs[1].section_type == SectionType.MEDICATIONS
    assert "".join(s.text for s in secs) == text


def test_unfound_anchor_is_dropped():
    text = "MEDICATIONS\nmetformin\nLABS\nA1c"
    raw = [
        {"type": "medications", "anchor": "MEDICATIONS"},
        {"type": "imaging", "anchor": "IMAGING"},
        {"type": "labs", "anchor": "LABS"},
    ]
    secs = resolve_sections(text, raw)
    assert [s.section_type for s in secs] == [SectionType.MEDICATIONS, SectionType.LABS]
    assert "".join(s.text for s in secs) == text


def test_out_of_order_anchors_sorted_by_position():
    text = "MEDICATIONS\nm\nLABS\nl"
    raw = [
        {"type": "labs", "anchor": "LABS"},
        {"type": "medications", "anchor": "MEDICATIONS"},
    ]
    secs = resolve_sections(text, raw)
    assert [s.section_type for s in secs] == [SectionType.MEDICATIONS, SectionType.LABS]
    assert "".join(s.text for s in secs) == text


def test_repeated_heading_resolves_forward():
    text = "NOTE\nfirst\nNOTE\nsecond"
    raw = [
        {"type": "clinical_note", "anchor": "NOTE"},
        {"type": "clinical_note", "anchor": "NOTE"},
    ]
    secs = resolve_sections(text, raw)
    assert len(secs) == 2
    assert secs[0].char_range == (0, text.index("NOTE", 1))
    assert "".join(s.text for s in secs) == text


def test_no_anchors_resolve_returns_single_other():
    text = "some clinical text with no recognizable headings"
    secs = resolve_sections(text, [{"type": "labs", "anchor": "NOPE"}])
    assert len(secs) == 1
    assert secs[0].section_type == SectionType.OTHER
    assert secs[0].text == text
    assert secs[0].char_range == (0, len(text))


def test_unknown_type_falls_back_to_other():
    text = "WEIRD\ncontent"
    secs = resolve_sections(text, [{"type": "not_a_real_type", "anchor": "WEIRD"}])
    assert secs[0].section_type == SectionType.OTHER
    assert "".join(s.text for s in secs) == text


def test_empty_raw_returns_single_other():
    text = "anything"
    secs = resolve_sections(text, [])
    assert len(secs) == 1 and secs[0].section_type == SectionType.OTHER
    assert secs[0].text == text
