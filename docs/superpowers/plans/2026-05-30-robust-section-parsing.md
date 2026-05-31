# Robust Section Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `section_parser` robust to document size — the model returns section type + a short verbatim anchor (not the full text), the code locates anchors and slices locally — so large notes parse into multiple sections (unlocking the Phase 2d concurrency speedup) and never crash on truncated JSON.

**Architecture:** A pure `resolve_sections(text, raw_sections)` helper does anchor-location + local slicing with a full-coverage guarantee. `_call_gemini_for_sections` uses Gemini structured output (`response_schema`) and an anchor-based prompt. `parse_sections` wires them together; output shape (`ParsedDocument`) is unchanged so `api/upload.py` needs no change.

**Tech Stack:** Python 3.12, google-genai 1.63 (structured output), pydantic v2, pytest. Spec: `docs/superpowers/specs/2026-05-30-robust-section-parsing-design.md`. Branch `feat/robust-section-parsing` (stacked on `feat/faster-entity-extraction`).

**Conventions:** `from __future__ import annotations`. Type hints. No `print()` — `logging` (indices/types only, NO PHI / no anchor text in logs). **Run pytest via `cd backend && .venv/bin/python -m pytest …`** (global pyenv broken). Ruff 100. All changes confined to `backend/app/services/extraction/section_parser.py` + its tests.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/app/services/extraction/section_parser.py` | `resolve_sections` helper; anchor-based prompt + `response_schema`; `parse_sections` wiring | Modify |
| `backend/tests/test_resolve_sections.py` | Exhaustive unit tests for the pure helper | Create |
| `backend/tests/test_section_parser.py` | Update existing tests to the anchor contract | Modify |
| `backend/tests/test_section_parser_integration.py` | Long-doc → multiple sections (mocked Gemini) | Create |
| `backend/tests/test_section_parsing_fidelity.py` | Slow: real note → multiple sections + speedup + records preserved | Create |

---

## Task 1: `resolve_sections` pure helper (the core)

**Files:**
- Modify: `backend/app/services/extraction/section_parser.py` (add the helper; nothing else yet)
- Test: `backend/tests/test_resolve_sections.py`

- [ ] **Step 1: Write the failing tests (expected-output-first)**

```python
# backend/tests/test_resolve_sections.py
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
    # full-coverage invariant: concatenation reconstructs the original text exactly
    assert "".join(s.text for s in secs) == text
    assert secs[0].char_range == (0, text.index("LABS"))


def test_leading_text_becomes_other_preamble():
    text = "Patient John Doe, DOB 1/1/1970\nMEDICATIONS\nmetformin"
    raw = [{"type": "medications", "anchor": "MEDICATIONS"}]
    secs = resolve_sections(text, raw)
    assert secs[0].section_type == SectionType.OTHER
    assert secs[0].text.startswith("Patient John Doe")
    assert secs[1].section_type == SectionType.MEDICATIONS
    assert "".join(s.text for s in secs) == text  # nothing lost


def test_unfound_anchor_is_dropped():
    text = "MEDICATIONS\nmetformin\nLABS\nA1c"
    raw = [
        {"type": "medications", "anchor": "MEDICATIONS"},
        {"type": "imaging", "anchor": "IMAGING"},  # not in text → dropped
        {"type": "labs", "anchor": "LABS"},
    ]
    secs = resolve_sections(text, raw)
    assert [s.section_type for s in secs] == [SectionType.MEDICATIONS, SectionType.LABS]
    assert "".join(s.text for s in secs) == text


def test_out_of_order_anchors_sorted_by_position():
    text = "MEDICATIONS\nm\nLABS\nl"
    raw = [
        {"type": "labs", "anchor": "LABS"},          # model returned out of order
        {"type": "medications", "anchor": "MEDICATIONS"},
    ]
    secs = resolve_sections(text, raw)
    assert [s.section_type for s in secs] == [SectionType.MEDICATIONS, SectionType.LABS]
    assert "".join(s.text for s in secs) == text


def test_repeated_heading_resolves_forward():
    text = "NOTE\nfirst\nNOTE\nsecond"
    raw = [
        {"type": "clinical_note", "anchor": "NOTE"},
        {"type": "clinical_note", "anchor": "NOTE"},  # second occurrence
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_resolve_sections.py -v`
Expected: FAIL (`cannot import name 'resolve_sections'`).

- [ ] **Step 3: Implement `resolve_sections` in `section_parser.py`**

Add this pure function (e.g. just below the dataclasses, above `parse_sections`):
```python
def resolve_sections(text: str, raw_sections: list[dict]) -> list[ParsedSection]:
    """Locate each section's anchor in `text` and slice locally (boundaries-only).

    Each raw section is {"type": <str>, "anchor": <verbatim snippet from the doc>}.
    Anchors are located in document order via forward search; an anchor not found is
    dropped. Full-coverage invariant: any text before the first anchor becomes a
    leading OTHER section, and each section runs to the next section's start, so the
    concatenation of section texts reconstructs `text` exactly. If no anchor resolves,
    the whole document is returned as a single OTHER section.
    """
    located: list[tuple[int, SectionType, str]] = []  # (position, type, anchor)
    search_from = 0
    for s in raw_sections:
        if not isinstance(s, dict):
            continue
        anchor = s.get("anchor") or ""
        if not anchor:
            continue
        pos = text.find(anchor, search_from)
        if pos == -1:
            pos = text.find(anchor)  # model may have returned slightly out of order
        if pos == -1:
            logger.debug("section anchor not found (type=%s); dropping", s.get("type"))
            continue
        try:
            stype = SectionType(s.get("type"))
        except (ValueError, KeyError):
            stype = SectionType.OTHER
        located.append((pos, stype, anchor))
        search_from = pos + len(anchor)

    if not located:
        return [ParsedSection(SectionType.OTHER, "Full Document", text, (0, len(text)))]

    located.sort(key=lambda t: t[0])
    sections: list[ParsedSection] = []

    first_pos = located[0][0]
    if first_pos > 0:
        sections.append(
            ParsedSection(SectionType.OTHER, "Preamble", text[0:first_pos], (0, first_pos))
        )

    for i, (pos, stype, anchor) in enumerate(located):
        end = located[i + 1][0] if i + 1 < len(located) else len(text)
        sections.append(ParsedSection(stype, anchor, text[pos:end], (pos, end)))

    return sections
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_resolve_sections.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/section_parser.py tests/test_resolve_sections.py
git commit -m "feat: resolve_sections — anchor-based local section slicing with full coverage"
```

---

## Task 2: Anchor-based prompt + structured output + wire into `parse_sections`

**Files:**
- Modify: `backend/app/services/extraction/section_parser.py` (`_SECTION_PARSER_PROMPT`, `_call_gemini_for_sections`, `parse_sections`)
- Modify: `backend/tests/test_section_parser.py` (update mocks/assertions to the anchor contract)

- [ ] **Step 1: Rewrite the prompt**

Replace `_SECTION_PARSER_PROMPT` so it asks for type + anchor only (NOT full text). Use:
```python
_SECTION_PARSER_PROMPT = """\
You are a clinical document parser. Given the full text of a medical document, identify its
logical sections in the order they appear and return structured JSON.

For EACH section return:
- "type": one of medications, assessment, clinical_note, labs, review_of_systems, history,
  physical_exam, assessment_plan, imaging, family_history, social_history, allergies,
  procedures, vitals, other
- "anchor": the section's heading or first line, copied VERBATIM from the document (the exact
  characters as they appear, ~10-60 chars). Do NOT paraphrase. Do NOT return the section body.

Also return top-level "document_type", "primary_visit_date", "provider", "facility" when
identifiable (else null). Return sections in document order. Do NOT echo the document text;
return only types and verbatim anchors.
"""
```

- [ ] **Step 2: Add the response schema + rewrite `_call_gemini_for_sections`**

Add pydantic models near the top of the file (after imports):
```python
from pydantic import BaseModel


class _SectionAnchor(BaseModel):
    type: str
    anchor: str


class _SectionParseSchema(BaseModel):
    document_type: str = "unknown"
    primary_visit_date: str | None = None
    provider: str | None = None
    facility: str | None = None
    sections: list[_SectionAnchor]
```
Rewrite `_call_gemini_for_sections` to pass the schema and parse:
```python
async def _call_gemini_for_sections(text: str, api_key: str) -> dict:
    """Call Gemini Flash to parse document sections (type + verbatim anchor only)."""
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=f"{_SECTION_PARSER_PROMPT}\n\n---\n\nDOCUMENT TEXT:\n{text}",
        config=genai.types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_SectionParseSchema,
        ),
    )
    return json.loads(response.text)
```
> Implementer note: google-genai 1.63 accepts a pydantic `BaseModel` subclass for
> `response_schema` and returns schema-valid JSON in `response.text`. If the installed SDK
> rejects the pydantic class, fall back to an equivalent `genai.types.Schema(...)` dict — verify
> with a quick `python -c "from google.genai import types; help(types.GenerateContentConfig)"`
> and adjust; the contract (object with a `sections` list of `{type, anchor}`) stays the same.

- [ ] **Step 3: Rewrite `parse_sections` to use `resolve_sections`**

Replace the body of `parse_sections` after the LLM call (the `raw_sections` extraction + the
per-section loop, currently `:108-151`) so it extracts metadata and delegates slicing:
```python
    if isinstance(llm_response, list):
        raw_sections = llm_response
        doc_type, visit_date, provider, facility = "unknown", None, None, None
    else:
        raw_sections = llm_response.get("sections", [])
        doc_type = llm_response.get("document_type", "unknown")
        visit_date = llm_response.get("primary_visit_date")
        provider = llm_response.get("provider")
        facility = llm_response.get("facility")

    sections = resolve_sections(text, raw_sections)

    return ParsedDocument(
        document_type=doc_type,
        primary_visit_date=visit_date,
        provider=provider,
        facility=facility,
        sections=sections,
    )
```
Keep the early-return for short/empty text and the `try/except` → single-OTHER-section fallback
(`:87-106`) exactly as they are.

- [ ] **Step 4: Update the existing tests to the anchor contract**

`tests/test_section_parser.py` currently mocks `_call_gemini_for_sections` to return sections
with `text`/`char_range`. Update those mock return values to the new contract
(`{"type": ..., "anchor": <verbatim substring of the test doc>}`) and update assertions to
expect locally-sliced text. Read the file; for each test that builds a mocked response, change
each section dict from `{"type","title","text","char_range"}` to `{"type","anchor"}` where
`anchor` is an exact substring of `CLINICAL_NOTE_TEXT` that begins that section, and assert
`result.sections[i].section_type` + that the sliced `.text` contains the expected content +
the full-coverage invariant `"".join(s.text for s in result.sections) == CLINICAL_NOTE_TEXT`.
Preserve the intent of each existing test (type mapping, unknown→OTHER, empty→single OTHER,
LLM-error→fallback).

- [ ] **Step 5: Run**

Run: `cd backend && .venv/bin/python -m pytest tests/test_section_parser.py tests/test_resolve_sections.py -v`
Expected: PASS (updated existing tests + the 8 helper tests).

- [ ] **Step 6: Commit**

```bash
cd backend && git add app/services/extraction/section_parser.py tests/test_section_parser.py
git commit -m "feat: anchor-based section prompt + structured output; slice locally via resolve_sections"
```

---

## Task 3: Integration test — long doc parses into multiple sections (mocked Gemini)

**Files:**
- Create: `backend/tests/test_section_parser_integration.py`

- [ ] **Step 1: Write the test**

```python
# backend/tests/test_section_parser_integration.py
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
```

- [ ] **Step 2: Run**

Run: `cd backend && .venv/bin/python -m pytest tests/test_section_parser_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_section_parser_integration.py
git commit -m "test: large-document section parsing yields multiple sections (no truncation)"
```

---

## Task 4: Fidelity — real note parses + speedup + records preserved (slow)

**Files:**
- Create: `backend/tests/test_section_parsing_fidelity.py`

- [ ] **Step 1: Write the slow test**

Mirror the harness in `tests/test_faster_extraction_fidelity.py` (real note, `GEMINI_API_KEY`
via `settings.gemini_api_key`, `@pytest.mark.slow`, patched `async_session_factory` +
`_run_dedup_background`, drive `_process_unstructured`, count records). Add a direct
`parse_sections` assertion + timing:

```python
# backend/tests/test_section_parsing_fidelity.py  (mirror the faster-extraction-fidelity harness)
# Two checks:
#  (a) parse_sections on the real note returns >1 section (no JSONDecodeError fallback):
#        from app.services.extraction.section_parser import parse_sections
#        doc = await parse_sections(real_note_text, settings.gemini_api_key)
#        assert len(doc.sections) > 1, "real note should parse into multiple sections now"
#  (b) full pipeline at conc=10: records preserved, no entity_extraction failures, and
#        wall-clock materially below the 603s Phase 2d observation:
#        assert n_records > 0
#        assert not any(e.get("stage") == "entity_extraction" for e in upload.ingestion_errors)
#        assert elapsed < 300, f"expected well under the 603s single-section run, got {elapsed:.0f}s"
```
Use `Path(__file__).resolve().parents[2] / "test_data"` glob `note_*.pdf`; the note text comes
from `extract_text_from_pdf_local` (local, fast) or the upload path. The 300s ceiling is
generous (target ~60–90s) to avoid flakiness while proving a real cut from 603s.

- [ ] **Step 2: Run ONCE (slow; needs key). No retry loops.**

Run: `cd backend && .venv/bin/python -m pytest tests/test_section_parsing_fidelity.py -v -m slow -rs`
Report: number of sections parsed, wall-clock, record count, any failures. If `parse_sections`
still returns 1 section, capture the Gemini response/error and report — the anchor prompt may
need tightening.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_section_parsing_fidelity.py
git commit -m "test: real-note section parsing fidelity — multiple sections, wall-clock cut"
```

---

## Task 5: Regression + lint

- [ ] **Step 1: Lint**

Run: `cd backend && .venv/bin/ruff check app/services/extraction/section_parser.py tests/test_resolve_sections.py tests/test_section_parser_integration.py tests/test_section_parsing_fidelity.py`
Fix feature-introduced errors; report pre-existing.

- [ ] **Step 2: Full fast suite**

Run: `cd backend && .venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (baseline 527 + new fast tests; updated `test_section_parser.py` green). Fix real regressions; report unrelated pre-existing failures.

- [ ] **Step 3: Final commit (if changes)**

```bash
cd backend && git add -A && git commit -m "chore: lint + regression pass for robust section parsing" || echo "nothing to commit"
```

---

## Self-Review Notes (author)

- **Spec coverage:** boundaries-only + structured output → Task 2; `resolve_sections` (find/slice/coverage/degradation) → Task 1; unchanged `ParsedDocument` shape (no upload.py change) → Task 2 Step 3; full-coverage invariant → Task 1 tests; integration (large doc) → Task 3; fidelity (sections + speedup + records) → Task 4; regression → Task 5. All covered.
- **Type consistency:** `resolve_sections(text: str, raw_sections: list[dict]) -> list[ParsedSection]`; raw section dict shape `{type, anchor}`; `_SectionParseSchema.sections: list[_SectionAnchor]`; `ParsedSection(section_type, title, text, char_range)` constructor positional order matches the dataclass. Title = anchor (or "Preamble"/"Full Document").
- **No PHI in logs:** `resolve_sections` logs only `type` on a dropped anchor, never the anchor text or document content.
- **Open implementation-time check:** `response_schema` form for the installed google-genai (pydantic class vs `types.Schema`) — Task 2 Step 2 notes the verification + fallback.
