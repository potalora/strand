# Extraction Performance — Design (Phase 2c)

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/extraction-performance` (stacked on `feat/unstructured-idempotency` → `feat/idempotent-incremental-ingestion`)
**Predecessors:** Phase 2a (`2026-05-29-unstructured-idempotency-design.md`)

## Problem

First-time unstructured extraction is slow and costly. Measured: the real clinical note
PDF took **~7 minutes** through the live pipeline (Phase 2a fidelity run). Root causes
(verified in code):

- `text_extractor.extract_text_from_pdf` (`:44`) sends the **whole PDF to Gemini vision**
  (`gemini_model`) — no local text-layer attempt. Most clinical PDFs have an embedded text
  layer extractable locally in milliseconds.
- `entity_extractor.extract_entities` calls `lx.extract(..., max_workers=1)` (`:24`) — chunks
  (`max_char_buffer=2000`) are processed **sequentially**, one Gemini call at a time.
- `_process_unstructured` calls `parse_sections` (a Gemini round-trip). NOTE: a short-doc skip
  guard ALREADY EXISTS at `upload.py:611` (`if len(scrubbed_text) < settings.small_doc_threshold`,
  default 3000) — so optimization #3 is already implemented; Phase 2c only adds a regression
  test pinning it.

Pipeline order: **text-extract → PHI scrub → section-parse → per-section entity-extract.**

This is **Phase 2c**. It changes *how fast/cheaply* we obtain text and entities, **not which
entities are extracted** (that is Phase 2b, extraction quality).

## Goal & Contract

Cut first-time extraction latency and Gemini cost — **minutes → seconds for text-layer
PDFs** — **without degrading clinical fidelity**. When local extraction is untrustworthy
(scanned/image PDFs, sparse or mangled output, or any error), fall back to the existing
Gemini-vision path. TIFFs remain on Gemini vision (scanned images by definition).

## Optimization #1 — Local-first PDF text (`pdfplumber`)

New `extract_text_from_pdf_local(file_path) -> tuple[str, float]` in `text_extractor.py`:
- Per page: `page.extract_text()` **plus** `page.extract_tables()` rendered as pipe-delimited
  rows appended to the page text (preserves lab panels / vitals grids — the fidelity-critical
  case that motivated choosing `pdfplumber` over `pypdf`).
- Returns `(text, confidence)` where `confidence` = average characters per page.

`extract_text_from_pdf` becomes a **router**:
1. Run `extract_text_from_pdf_local`.
2. If `confidence >= LOCAL_TEXT_MIN_CHARS_PER_PAGE` (**default 50**) and no error → **use the
   local text** ($0, milliseconds).
3. Otherwise (sparse / scanned / `pdfplumber` raised) → **fall back** to the existing
   Gemini-vision path, renamed `_extract_text_from_pdf_gemini` (logic unchanged).

Conservative by design: **prefer Gemini when uncertain** — fidelity over speed. Add
`pdfplumber>=0.11` (MIT license — satisfies Absolute Rule 17; explicitly NOT PyMuPDF/`fitz`,
which is AGPL).

## Optimization #2 — Parallelize LangExtract

`entity_extractor.extract_entities`: change `max_workers=1` →
`max_workers=settings.gemini_concurrency_limit` (already configured = 10). LangExtract chunks
the text (`max_char_buffer=2000`) and processes chunks concurrently with a thread pool,
bounded by the existing concurrency limit so Gemini rate limits are respected. ~N× faster on
multi-chunk documents.

## Optimization #3 — Skip section-parse for short docs (ALREADY IMPLEMENTED)

This already exists: `_process_unstructured` (`upload.py:611`) skips the `parse_sections`
Gemini call and uses the single-section fallback when
`len(scrubbed_text) < settings.small_doc_threshold` (config default **3000**). Phase 2c does
NOT re-implement it — it adds a **regression test** pinning the behavior (short text → no
`parse_sections` call; long text → call made), so a future change can't silently undo it.

## Components & Boundaries

- `app/services/extraction/text_extractor.py`:
  - `extract_text_from_pdf_local(file_path) -> tuple[str, float]` (new; pure, no API).
  - `_render_tables(tables) -> str` (new helper; pure).
  - `_extract_text_from_pdf_gemini(file_path, api_key) -> str` (renamed from the current
    `extract_text_from_pdf` body).
  - `extract_text_from_pdf(file_path, api_key) -> str` (router; local-first with Gemini
    fallback).
  - Module constant `LOCAL_TEXT_MIN_CHARS_PER_PAGE = 50`.
- `app/services/extraction/entity_extractor.py`: one-line `max_workers` change (config-driven).
- `app/api/upload.py`: short-doc guard already present (`settings.small_doc_threshold`); no
  code change — regression test only.

The confidence/routing logic is isolated in `text_extractor.py` and unit-testable without any
Gemini call.

## Error Handling & Safety

- `pdfplumber` raising on a malformed PDF → log (no PHI; counts only) and fall back to Gemini.
  Never crash ingestion.
- Confidence threshold is conservative; a borderline doc goes to Gemini, not to a degraded
  local result.
- No schema change. No new env var required (constants; `gemini_concurrency_limit` already
  exists).

## Test Strategy (expected-output-first)

- **Unit (no Gemini):**
  - Router/confidence decision: a fixture with good text-per-page → router uses local and does
    NOT call Gemini; a sparse/empty-text fixture → router falls back (assert the Gemini
    function is invoked). Use `unittest.mock` to stub `pdfplumber.open` and the Gemini call.
  - `_render_tables`: given a sample `extract_tables()` structure → expected pipe-delimited
    string (pinned).
  - `extract_entities` passes `max_workers = settings.gemini_concurrency_limit` to `lx.extract`
    (patch `lx.extract`, assert kwarg).
  - Section-skip: text below threshold → `parse_sections` not called (single-section result);
    above threshold → called. Patch `parse_sections`.
- **Spy test:** a real/representative text-layer PDF extracts with **zero** Gemini-vision
  calls (assert `_extract_text_from_pdf_gemini` not invoked) — proves the optimization engages.
- **Fidelity / slow (`@pytest.mark.slow`, real data + `GEMINI_API_KEY`):**
  - Text-layer note (`test_data/note_*.pdf`) → local path; records extracted (n>0); wall-clock
    **well under the ~7-min Gemini baseline** (assert a generous ceiling, e.g. < 90s, to catch
    regression without being flaky).
  - **Scanned PDF (`test_data/ibs_smart.pdf`)** → low confidence → **falls back to Gemini
    vision** → records extracted. This is the explicit **Gemini vision/OCR-path** verification.

## Out of Scope

- Extraction *quality/coverage* tuning (Phase 2b): entity precision/recall, hallucination and
  negation handling on non-standard documents.
- Streaming/progressive results to the UI.
- Changing `max_char_buffer` or the extraction prompts.

## Validation Plan (post-build)

1. Upload the text-layer note → confirm fast (seconds), records present, no Gemini vision call
   in logs.
2. Upload `ibs_smart.pdf` (scanned) → confirm Gemini-vision fallback fired and records present.
3. Re-run the Phase 2a fidelity test (`test_unstructured_idempotency_fidelity.py`) — now fast,
   giving the deferred green observation of the full Gemini entity path.
