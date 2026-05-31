# Robust Section Parsing — Design (Phase 2e)

**Date:** 2026-05-30
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/robust-section-parsing` (stacked on `feat/faster-entity-extraction` → … → `feat/idempotent-incremental-ingestion`)
**Predecessor:** Phase 2d (`2026-05-29-faster-entity-extraction-design.md`)

## Problem

Phase 2d made concurrency safe but the real note still ran ~603s (no speedup). Root cause
(measured): `section_parser._call_gemini_for_sections` (`section_parser.py:154-165`) asks
Gemini to return JSON whose every section contains the **full section text** (the prompt:
"Preserve ALL text — every character of the original document must appear in exactly one
section"). It sets `response_mime_type="application/json"` but **no `response_schema`**, then
does `json.loads(response.text)`.

For a 35 KB note the JSON must echo all ~35 KB → it **exceeds the output-token limit → the
JSON is truncated** (`JSONDecodeError: Unterminated string` ~char 35425) → the `except`
fallback returns a **single OTHER section**. With one giant section, the per-section
concurrency (`section_extraction_concurrency=10`) has nothing to parallelize, so large notes
never get the speedup. Downstream, `upload.py` uses `section.text` directly.

## Goal & Contract

Make section parsing robust to document size so large notes parse into **multiple real
sections** (unlocking the Phase 2d concurrency speedup) and never crash on truncated JSON.
Output shape (`ParsedDocument` / `ParsedSection`) is unchanged so `upload.py` needs no change.
Full text coverage is preserved (every character lands in exactly one section). Target: the
real note parses into N sections and the full pipeline drops toward ~60–90s, with record count
preserved.

## Approach (A — boundaries-only + structured output)

Stop making the model echo text. The model returns only **section type + a short verbatim
anchor** (the section heading / first line, ~40 chars) per section. The code locates each
anchor in the **original** text and slices locally. Output is a few hundred bytes regardless
of document size → no truncation, cheaper, faster.

**Why anchors, not character offsets:** LLMs count characters unreliably; locating a short
verbatim anchor string via `str.find` in the original text is robust and deterministic.

## Components & Changes (`backend/app/services/extraction/section_parser.py` only)

### 1. Prompt + call (`_call_gemini_for_sections`, `_SECTION_PARSER_PROMPT`)
- Rewrite the prompt: return a JSON array of objects `{ "type": <SectionType>, "anchor": <verbatim first line / heading of the section, copied exactly from the document> }`, in document order. Do NOT return section text. Keep the existing document-metadata fields (`document_type`, `primary_visit_date`, `provider`, `facility`) if present, but the sections carry only type + anchor.
- Add Gemini **structured output**: pass `response_schema` (a list-of-objects schema) in
  `GenerateContentConfig` alongside `response_mime_type="application/json"` so the response is
  always schema-valid JSON. Keep `temperature=0.1`.

### 2. New pure helper `resolve_sections(text: str, raw_sections: list[dict]) -> list[ParsedSection]`
- For each raw section in order, find its `anchor` in `text` via `text.find(anchor, search_from)`
  where `search_from` starts at 0 and advances to each found position (so anchors resolve in
  document order, handling repeated headings).
- Drop anchors not found (`find` returns -1); log at debug (no PHI — log the type/index, not
  the anchor text).
- Sort the resolved `(position, type)` by position; the section's text is
  `text[position : next_position]` (last section runs to end-of-text). The first resolved
  position > 0 → prepend a leading OTHER section with `text[0:first_position]` so **no leading
  text is lost** (full coverage invariant).
- If zero anchors resolve → return a single `OTHER` section with the full text (existing
  fallback behavior).
- `char_range` is set to `(position, next_position)` for each section.

### 3. `parse_sections`
- Call `_call_gemini_for_sections`, then build sections via `resolve_sections(text, raw)`.
- Same `ParsedDocument` output (document metadata + `sections`), so `api/upload.py` is
  unchanged.
- Keep the try/except → single-OTHER-section fallback for any Gemini/parse failure.

## Error Handling & Safety

- Structured output makes truncated/invalid JSON near-impossible; the existing fallback stays
  as a backstop.
- `resolve_sections` is pure and total: any anchor mismatch degrades gracefully (drop /
  fallback), never raises.
- **Full-coverage invariant:** the concatenation of section texts (in position order, plus any
  leading-OTHER prefix) reconstructs the original `text` with no characters lost — unit-tested.
- No PHI in logs (indices/types only). No schema/model/config changes outside this file.

## Test Strategy (expected-output-first)

- **Unit (no Gemini) — `resolve_sections`:**
  - anchors found in order → correct slices; concatenation == original text (coverage invariant).
  - leading text before the first anchor → preserved as a leading OTHER section.
  - an unfound anchor → dropped; remaining sections still slice correctly.
  - repeated heading text → second occurrence resolves after the first (forward search).
  - empty / no anchors resolve → single OTHER section with full text.
  - out-of-order anchors (model returned wrong order) → sorted by found position.
- **Integration (mocked Gemini):** a long synthetic doc (e.g. 40 KB) where the old path would
  truncate; mock `_call_gemini_for_sections` to return anchors → `parse_sections` yields N
  sections covering the whole doc, no exception.
- **Fidelity / slow (real note, `GEMINI_API_KEY`):** `parse_sections` on the real note returns
  **>1 section** (no `JSONDecodeError` fallback), and the full `_process_unstructured` pipeline
  wall-clock drops materially (target ~60–90s; assert a generous ceiling well below the 603s
  Phase 2d observation), with `ai_extracted` record count preserved and no `entity_extraction`
  failures. This is the Phase 2d speedup finally realized.
- Full fast-suite regression (baseline 527 passed) + existing `test_section_parser.py` updated
  for the anchor-based contract.

## Out of Scope

- Extraction *quality* / which entities are produced (Phase 2b).
- A no-LLM heuristic header splitter (considered; kept only as the existing single-section
  fallback, not built here).
- Changes to `api/upload.py`, models, or other services.

## Validation Plan (post-build)

1. Run the slow fidelity test → confirm multiple sections + wall-clock cut + records preserved.
2. Upload the real note via the app → confirm fast extraction and sensible section breakdown.
3. Confirm a small note (< `small_doc_threshold`) still skips section parsing (Phase 2c path),
   and a mid-size multi-section note parses correctly.
