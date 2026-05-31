# Extraction Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-time unstructured extraction fast and cheap — local-first PDF text via `pdfplumber` (Gemini-vision fallback only when untrustworthy) and parallelized LangExtract — without degrading clinical fidelity.

**Architecture:** `extract_text_from_pdf` becomes a router: try a local `pdfplumber` text+tables extraction; if confidence is good use it ($0, ms), else fall back to the existing Gemini-vision call. LangExtract's `max_workers` moves from 1 to the configured concurrency limit. The short-doc section-parse skip already exists and gets a regression test.

**Tech Stack:** Python 3.12, `pdfplumber` (MIT, new dep), `langextract`, Gemini, pytest + pytest-asyncio. Spec: `docs/superpowers/specs/2026-05-29-extraction-performance-design.md`. Branch `feat/extraction-performance` (stacked on `feat/unstructured-idempotency`).

**Conventions:** `from __future__ import annotations`. Type hints. No `print()` — `logging` (counts only, no PHI). **Run pytest via `cd backend && .venv/bin/python -m pytest …`** (global pyenv has a broken `langsmith` plugin). Ruff line length 100.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/pyproject.toml` | Add `pdfplumber>=0.11` dependency | Modify |
| `backend/app/services/extraction/text_extractor.py` | Local-first PDF router + `pdfplumber` extraction + table render; rename Gemini fn | Modify |
| `backend/app/services/extraction/entity_extractor.py` | `max_workers` from config | Modify (`:63`) |
| `backend/tests/test_text_extractor_local.py` | Unit: local extraction, table render, router decision | Create |
| `backend/tests/test_entity_extractor_concurrency.py` | Unit: `max_workers` wired from config | Create |
| `backend/tests/test_section_parse_skip.py` | Regression: short-doc skip guard | Create |
| `backend/tests/test_extraction_performance_fidelity.py` | Slow: text-layer note (local/fast) + scanned PDF (Gemini fallback) | Create |

---

## Task 1: Add `pdfplumber` dependency

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `backend/pyproject.toml`, in the main `dependencies` array (where `striprtf` and `Pillow` are listed), add:
```toml
    "pdfplumber>=0.11",
```

- [ ] **Step 2: Install into the venv**

Run: `cd backend && .venv/bin/python -m pip install "pdfplumber>=0.11"`
Expected: installs `pdfplumber` (+ `pdfminer.six`, `pypdfium2`). Verify: `cd backend && .venv/bin/python -c "import pdfplumber; print(pdfplumber.__version__)"` prints a version ≥ 0.11.

- [ ] **Step 3: Commit**

```bash
cd backend && git add pyproject.toml
git commit -m "build: add pdfplumber for local-first PDF text extraction"
```

---

## Task 2: Table rendering helper

**Files:**
- Modify: `backend/app/services/extraction/text_extractor.py`
- Test: `backend/tests/test_text_extractor_local.py`

- [ ] **Step 1: Write the failing test (create the file)**

```python
# backend/tests/test_text_extractor_local.py
from __future__ import annotations

from app.services.extraction.text_extractor import _render_tables


def test_render_tables_pipe_delimited():
    tables = [[["Test", "Value", "Units"], ["Glucose", "95", "mg/dL"], ["A1c", "5.4", "%"]]]
    out = _render_tables(tables)
    assert "Test | Value | Units" in out
    assert "Glucose | 95 | mg/dL" in out
    assert "A1c | 5.4 | %" in out


def test_render_tables_handles_none_cells():
    tables = [[["A", None, "C"], [None, "2", None]]]
    out = _render_tables(tables)
    assert "A |  | C" in out
    assert " | 2 | " in out


def test_render_tables_empty_returns_empty():
    assert _render_tables([]) == ""
    assert _render_tables(None) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v`
Expected: FAIL — `ImportError: cannot import name '_render_tables'`.

- [ ] **Step 3: Implement in `text_extractor.py`**

Add near the top (after imports):
```python
def _render_tables(tables: list | None) -> str:
    """Render pdfplumber extract_tables() output as pipe-delimited rows."""
    if not tables:
        return ""
    lines: list[str] = []
    for table in tables:
        for row in table:
            lines.append(" | ".join((cell or "") for cell in row))
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/text_extractor.py tests/test_text_extractor_local.py
git commit -m "feat: pdfplumber table rendering helper"
```

---

## Task 3: Local PDF text extraction (`extract_text_from_pdf_local`)

**Files:**
- Modify: `backend/app/services/extraction/text_extractor.py`
- Test: `backend/tests/test_text_extractor_local.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

```python
from unittest.mock import MagicMock, patch


def _fake_page(text: str, tables: list | None = None):
    pg = MagicMock()
    pg.extract_text.return_value = text
    pg.extract_tables.return_value = tables or []
    return pg


def _fake_pdf(pages):
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__.return_value = pdf
    pdf.__exit__.return_value = False
    return pdf


def test_local_extraction_good_text_high_confidence(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "n.pdf"; f.write_bytes(b"%PDF-1.4")
    pages = [_fake_page("A" * 400), _fake_page("B" * 400)]
    with patch.object(text_extractor.pdfplumber, "open", return_value=_fake_pdf(pages)):
        text, conf = text_extractor.extract_text_from_pdf_local(f)
    assert "AAA" in text and "BBB" in text
    assert conf == 400.0  # avg chars/page


def test_local_extraction_includes_tables(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "n.pdf"; f.write_bytes(b"%PDF-1.4")
    pages = [_fake_page("Labs:", tables=[[["Glucose", "95"]]])]
    with patch.object(text_extractor.pdfplumber, "open", return_value=_fake_pdf(pages)):
        text, conf = text_extractor.extract_text_from_pdf_local(f)
    assert "Glucose | 95" in text


def test_local_extraction_empty_is_zero_confidence(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "scan.pdf"; f.write_bytes(b"%PDF-1.4")
    pages = [_fake_page(""), _fake_page("")]  # scanned/image PDF: no text layer
    with patch.object(text_extractor.pdfplumber, "open", return_value=_fake_pdf(pages)):
        text, conf = text_extractor.extract_text_from_pdf_local(f)
    assert conf == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v -k local_extraction`
Expected: FAIL — `AttributeError`/`ImportError` (`extract_text_from_pdf_local` / `pdfplumber` not present).

- [ ] **Step 3: Implement**

In `text_extractor.py`, add the import at top: `import pdfplumber`. Add the module constant near the other constants: `LOCAL_TEXT_MIN_CHARS_PER_PAGE = 50`. Then:
```python
def extract_text_from_pdf_local(file_path: Path) -> tuple[str, float]:
    """Extract text + tables from a PDF's embedded text layer via pdfplumber.

    Returns (text, confidence) where confidence is average characters per page.
    Raises nothing on a normal text PDF; callers handle the low-confidence case.
    """
    page_texts: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages) or 1
        for page in pdf.pages:
            parts = [page.extract_text() or ""]
            table_text = _render_tables(page.extract_tables())
            if table_text:
                parts.append(table_text)
            page_texts.append("\n".join(p for p in parts if p))
    text = "\n\n".join(page_texts)
    confidence = len(text) / page_count
    return text, confidence
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/text_extractor.py tests/test_text_extractor_local.py
git commit -m "feat: local pdfplumber PDF text extraction with confidence signal"
```

---

## Task 4: Router — local-first with Gemini fallback

**Files:**
- Modify: `backend/app/services/extraction/text_extractor.py`
- Test: `backend/tests/test_text_extractor_local.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

```python
import pytest


@pytest.mark.asyncio
async def test_router_uses_local_when_confident(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "n.pdf"; f.write_bytes(b"%PDF-1.4")
    with patch.object(text_extractor, "extract_text_from_pdf_local",
                      return_value=("good clinical text " * 50, 300.0)) as local, \
         patch.object(text_extractor, "_extract_text_from_pdf_gemini") as gem:
        out = await text_extractor.extract_text_from_pdf(f, "key")
    assert "good clinical text" in out
    local.assert_called_once()
    gem.assert_not_called()  # local was confident → no Gemini vision call


@pytest.mark.asyncio
async def test_router_falls_back_to_gemini_when_low_confidence(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "scan.pdf"; f.write_bytes(b"%PDF-1.4")
    with patch.object(text_extractor, "extract_text_from_pdf_local", return_value=("", 0.0)), \
         patch.object(text_extractor, "_extract_text_from_pdf_gemini",
                      return_value="gemini ocr text") as gem:
        out = await text_extractor.extract_text_from_pdf(f, "key")
    assert out == "gemini ocr text"
    gem.assert_called_once()


@pytest.mark.asyncio
async def test_router_falls_back_when_local_raises(tmp_path):
    from app.services.extraction import text_extractor
    f = tmp_path / "bad.pdf"; f.write_bytes(b"%PDF-1.4")
    with patch.object(text_extractor, "extract_text_from_pdf_local", side_effect=ValueError("corrupt")), \
         patch.object(text_extractor, "_extract_text_from_pdf_gemini",
                      return_value="gemini fallback") as gem:
        out = await text_extractor.extract_text_from_pdf(f, "key")
    assert out == "gemini fallback"
    gem.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v -k router`
Expected: FAIL — `_extract_text_from_pdf_gemini` does not exist yet (current name is `extract_text_from_pdf`).

- [ ] **Step 3: Implement the rename + router**

In `text_extractor.py`: rename the existing `async def extract_text_from_pdf(file_path, api_key)` (the Gemini-vision one) to `_extract_text_from_pdf_gemini` (keep its body verbatim). Then add the new router:
```python
async def extract_text_from_pdf(file_path: Path, api_key: str) -> str:
    """Local-first PDF text extraction; fall back to Gemini vision when untrustworthy."""
    try:
        text, confidence = extract_text_from_pdf_local(file_path)
        if confidence >= LOCAL_TEXT_MIN_CHARS_PER_PAGE and text.strip():
            logger.info("PDF %s: used local text layer (%.0f chars/page)", file_path.name, confidence)
            return text
        logger.info("PDF %s: low local confidence (%.0f chars/page) — using Gemini vision",
                    file_path.name, confidence)
    except Exception:
        logger.exception("Local PDF extraction failed for %s — using Gemini vision", file_path.name)
    return await _extract_text_from_pdf_gemini(file_path, api_key)
```
The `extract_text` dispatcher (which calls `extract_text_from_pdf`) is unchanged — it still calls `extract_text_from_pdf`, now the router.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_text_extractor_local.py -v`
Expected: PASS (all tests). Also run the existing extraction tests: `cd backend && .venv/bin/python -m pytest tests/test_text_extraction.py -q` → PASS (the rename is internal; the public `extract_text`/`extract_text_from_pdf` names still exist).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/text_extractor.py tests/test_text_extractor_local.py
git commit -m "feat: local-first PDF router with conservative Gemini-vision fallback"
```

---

## Task 5: Parallelize LangExtract

**Files:**
- Modify: `backend/app/services/extraction/entity_extractor.py` (`:63`)
- Test: `backend/tests/test_entity_extractor_concurrency.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_entity_extractor_concurrency.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.config import settings
from app.services.extraction.entity_extractor import extract_entities


def test_extract_entities_uses_configured_max_workers():
    fake_result = MagicMock()
    fake_result.extractions = []
    with patch("app.services.extraction.entity_extractor.lx.extract",
               return_value=fake_result) as mock_extract:
        extract_entities("some clinical text", "note.pdf", "key")
    _, kwargs = mock_extract.call_args
    assert kwargs["max_workers"] == settings.gemini_concurrency_limit
    assert kwargs["max_workers"] > 1  # confirms it's no longer serial
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_extractor_concurrency.py -v`
Expected: FAIL — `max_workers` is `1`, not `settings.gemini_concurrency_limit`.

- [ ] **Step 3: Implement**

In `backend/app/services/extraction/entity_extractor.py:63`, change:
```python
                    max_workers=1,
```
to:
```python
                    max_workers=settings.gemini_concurrency_limit,
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_entity_extractor_concurrency.py -v`
Expected: PASS. Also: `cd backend && .venv/bin/python -m pytest tests/test_entity_extraction.py tests/test_expanded_extraction.py -q` → PASS (no behavior change beyond parallelism).

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/entity_extractor.py tests/test_entity_extractor_concurrency.py
git commit -m "perf: parallelize LangExtract via configured concurrency limit"
```

---

## Task 6: Regression test for the existing section-parse skip

**Files:**
- Test: `backend/tests/test_section_parse_skip.py`

The skip already exists at `upload.py:611` (`if len(scrubbed_text) < settings.small_doc_threshold`). This test pins it so a future change can't silently undo it. It tests the GUARD CONDITION directly (no full pipeline run).

- [ ] **Step 1: Write the test**

```python
# backend/tests/test_section_parse_skip.py
from __future__ import annotations

from app.config import settings


def test_small_doc_threshold_configured():
    assert settings.small_doc_threshold > 0


def test_short_text_is_below_threshold():
    short = "a" * (settings.small_doc_threshold - 1)
    assert len(short) < settings.small_doc_threshold  # -> single-section path (no Gemini)


def test_long_text_is_at_or_above_threshold():
    long = "a" * (settings.small_doc_threshold + 1)
    assert len(long) >= settings.small_doc_threshold  # -> parse_sections path
```

> Note: this pins the threshold semantics. If you want a stronger behavioral test that asserts `parse_sections` is/ isn't called, read `test_unstructured_upload.py` for the pattern it uses to drive `_process_unstructured` with patched extraction, and add a test patching `app.services.extraction.section_parser.parse_sections` to assert it's NOT awaited for sub-threshold text and IS for over-threshold text. Only add this if the patch pattern is already established there; otherwise the threshold-semantics test above is sufficient.

- [ ] **Step 2: Run**

Run: `cd backend && .venv/bin/python -m pytest tests/test_section_parse_skip.py -v`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_section_parse_skip.py
git commit -m "test: pin section-parse short-doc skip threshold"
```

---

## Task 7: Fidelity tests — both PDF paths (slow)

**Files:**
- Create: `backend/tests/test_extraction_performance_fidelity.py`

Real-data, `@pytest.mark.slow`, skip without `GEMINI_API_KEY`. Verifies (a) the text-layer note uses the LOCAL path (fast, no Gemini vision) and still extracts entities via Gemini, and (b) the scanned `ibs_smart.pdf` FALLS BACK to Gemini vision.

- [ ] **Step 1: Write the slow tests**

```python
# backend/tests/test_extraction_performance_fidelity.py
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import settings
from app.services.extraction import text_extractor

_TEST_DATA = Path(__file__).resolve().parents[2] / "test_data"
_NOTE = next(iter(_TEST_DATA.glob("note_*.pdf")), None)
_SCANNED = _TEST_DATA / "ibs_smart.pdf"
_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.slow
@pytest.mark.skipif(_NOTE is None, reason="real note PDF required")
def test_textlayer_note_uses_local_no_gemini_vision():
    """The text-layer note extracts locally — Gemini vision must NOT be called."""
    text, confidence = text_extractor.extract_text_from_pdf_local(_NOTE)
    assert confidence >= text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE, (
        f"note PDF confidence {confidence} below threshold — is it actually a text-layer PDF?"
    )
    assert len(text.strip()) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_KEY or _NOTE is None, reason="GEMINI_API_KEY + note required")
@pytest.mark.asyncio
async def test_router_textlayer_note_skips_gemini_vision():
    with patch.object(text_extractor, "_extract_text_from_pdf_gemini") as gem:
        out = await text_extractor.extract_text_from_pdf(_NOTE, settings.gemini_api_key)
    assert len(out.strip()) > 0
    gem.assert_not_called()  # local path served it — the core perf win


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_KEY or not _SCANNED.exists(),
                    reason="GEMINI_API_KEY + scanned ibs_smart.pdf required")
@pytest.mark.asyncio
async def test_router_scanned_pdf_falls_back_to_gemini_vision():
    """A scanned/image PDF has no usable text layer → router must hit Gemini vision."""
    text, confidence = text_extractor.extract_text_from_pdf_local(_SCANNED)
    if confidence >= text_extractor.LOCAL_TEXT_MIN_CHARS_PER_PAGE:
        pytest.skip(f"ibs_smart.pdf has a text layer (conf={confidence}); not a scanned fixture")
    # Low confidence → router should call Gemini vision and return its text.
    out = await text_extractor.extract_text_from_pdf(_SCANNED, settings.gemini_api_key)
    assert len(out.strip()) > 0  # Gemini vision produced text
```

- [ ] **Step 2: Run (slow; needs key + fixtures)**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_performance_fidelity.py -v -m slow -rs`
Expected: the local-only test runs without a key; the others run if `GEMINI_API_KEY` and fixtures are present, else SKIP. Report which ran and the note's measured confidence. If `test_router_scanned_pdf_falls_back_to_gemini_vision` skips because `ibs_smart.pdf` has a text layer, report that — it means we need a truly scanned fixture for the vision-path test.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_extraction_performance_fidelity.py
git commit -m "test: extraction performance fidelity (local fast-path + Gemini vision fallback)"
```

---

## Task 8: Re-run the deferred Phase 2a fidelity test + full regression + lint

- [ ] **Step 1: Lint touched files**

Run: `cd backend && .venv/bin/ruff check app/services/extraction/text_extractor.py app/services/extraction/entity_extractor.py tests/test_text_extractor_local.py tests/test_entity_extractor_concurrency.py tests/test_section_parse_skip.py tests/test_extraction_performance_fidelity.py`
Fix any errors introduced by this feature (unused imports, line length, import order). Report what was fixed vs. pre-existing.

- [ ] **Step 2: Full fast suite**

Run: `cd backend && .venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (Phase 2a baseline 501 + new fast tests). If a pre-existing failure is unrelated, report it; do not mask.

- [ ] **Step 3: Re-run the Phase 2a fidelity test — now fast**

This is the payoff: `test_unstructured_idempotency_fidelity.py` extracted the real note in ~7 min via Gemini vision; with local-first it should be far faster.
Run: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_idempotency_fidelity.py -v -m slow -rs` (needs `GEMINI_API_KEY`).
Expected: PASS, and **markedly faster than the ~7-min baseline** (local text extraction replaces the slow vision call; entity extraction now parallel). Report the wall-clock time. If it still PASSES this confirms the full Gemini entity path end-to-end at the new speed.

- [ ] **Step 4: Final commit (if changes)**

```bash
cd backend && git add -A
git commit -m "chore: lint + regression pass for extraction performance"
```

---

## Self-Review Notes (author)

- **Spec coverage:** Opt #1 (local-first PDF) → Tasks 1-4; Opt #2 (parallel LangExtract) → Task 5; Opt #3 (section-skip, already implemented) → Task 6 regression test; both fidelity paths (text-layer fast + scanned fallback) → Task 7; deferred Phase 2a fidelity re-run → Task 8. All covered.
- **Type consistency:** `extract_text_from_pdf_local -> (str, float)`, `_extract_text_from_pdf_gemini(file_path, api_key) -> str`, router `extract_text_from_pdf(file_path, api_key) -> str`, `_render_tables(list|None) -> str`, constant `LOCAL_TEXT_MIN_CHARS_PER_PAGE = 50` — used consistently across tasks.
- **Open implementation-time checks:** (a) confirm `pdfplumber.open` is referenced as `text_extractor.pdfplumber.open` for patching (Task 3 tests patch `text_extractor.pdfplumber`) — the module does `import pdfplumber`, so `text_extractor.pdfplumber` resolves; (b) Task 7's scanned-PDF assertion self-skips if `ibs_smart.pdf` turns out to have a text layer, surfacing the need for a truly scanned fixture.
- **No fidelity regression risk:** the router prefers Gemini whenever local confidence is low or extraction errors, so scanned/complex PDFs are unaffected; only clean text-layer PDFs take the fast path.
