# Faster Entity Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut entity-extraction wall-clock (~467s → ~60–90s on the reference note) by safely raising concurrency, while guaranteeing records are never silently dropped; adopt `gemini-3.5-flash`; investigate per-call overhead reduction.

**Architecture:** Surface extraction-chunk failures first (so nothing vanishes), then reproduce and fix the full-pipeline conc=10 bug, then raise `section_extraction_concurrency` 3→10. Adopt `gemini-3.5-flash` for both model configs. Investigate Gemini context caching for the static few-shot prompt as a stretch.

**Tech Stack:** Python 3.12, FastAPI, LangExtract + Gemini (`gemini-3.5-flash`), asyncio, pytest. Spec: `docs/superpowers/specs/2026-05-29-faster-entity-extraction-design.md`. Branch `feat/faster-entity-extraction` (stacked on `feat/extraction-performance`).

**Conventions:** `from __future__ import annotations`. Type hints. No `print()` — `logging` (counts only, no PHI). **Run pytest via `cd backend && .venv/bin/python -m pytest …`** (global pyenv broken). Ruff 100. Measured facts: per-call ≈22s (model-independent), concurrency=10 is safe (12/12 ok), the wall is call count × latency.

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/app/config.py` | Model IDs → `gemini-3.5-flash`; later `section_extraction_concurrency` 3→10 | Modify (`:51-52`, `:59`) |
| `backend/app/api/upload.py` | Count + surface failed extraction chunks (Step 4 gather loop `:677-687`) | Modify |
| `backend/tests/test_model_config.py` | Assert configured model IDs | Create |
| `backend/tests/test_extraction_failure_surfacing.py` | Failed chunk → surfaced on upload, not swallowed | Create |
| `backend/tests/test_extraction_concurrency_setting.py` | `section_extraction_concurrency` value pinned | Create |
| `backend/tests/test_faster_extraction_fidelity.py` | Slow: record count preserved + wall-clock cut at raised concurrency | Create |
| `docs/superpowers/specs/2026-05-29-entity-overhead-findings.md` | (B) context-caching investigation findings | Create |
| `CLAUDE.md` | Rule 16 + model refs → 3.5-flash | Modify (LOCAL ONLY, never committed/pushed) |

---

## Task 1: Adopt `gemini-3.5-flash` (config)

**Files:**
- Modify: `backend/app/config.py` (`:51-52`)
- Test: `backend/tests/test_model_config.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_model_config.py
from __future__ import annotations

from app.config import settings


def test_extraction_model_is_3_5_flash():
    assert settings.gemini_extraction_model == "gemini-3.5-flash"


def test_text_summary_model_is_3_5_flash():
    assert settings.gemini_model == "gemini-3.5-flash"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_model_config.py -v`
Expected: FAIL (currently `gemini-2.5-flash` / `gemini-3-flash-preview`).

- [ ] **Step 3: Implement**

In `backend/app/config.py`:
- Line 51: `gemini_model: str = "gemini-3-flash-preview"` → `gemini_model: str = "gemini-3.5-flash"`
- Line 52: `gemini_extraction_model: str = "gemini-2.5-flash"` → `gemini_extraction_model: str = "gemini-3.5-flash"`

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_model_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit (config + test only — NOT CLAUDE.md)**

```bash
cd backend && git add app/config.py tests/test_model_config.py
git commit -m "feat: adopt gemini-3.5-flash for entity + text/summary models"
```

> CLAUDE.md Rule 16 + model refs are updated in Task 7 **locally and left uncommitted** per the maintainer's standing rule. Do not stage CLAUDE.md in any commit.

---

## Task 2: Surface failed extraction chunks (never silently drop)

Make a chunk that fails after retries **visible** on the upload instead of vanishing. This also makes Task 3's conc=10 diagnosis trivial (failures become inspectable).

**Files:**
- Modify: `backend/app/api/upload.py` (Step 4 gather loop, currently `:677-687`)
- Test: `backend/tests/test_extraction_failure_surfacing.py`

- [ ] **Step 1: Write the failing test**

Use the real-user fixture pattern + drive `_process_unstructured` with extraction patched to fail one chunk. Mirror how `tests/test_unstructured_idempotency_fidelity.py` patches `async_session_factory` + `_run_dedup_background` and reads back the upload. Read that file first to copy the harness exactly; then:

```python
# backend/tests/test_extraction_failure_surfacing.py
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.models.uploaded_file import UploadedFile
from app.services.extraction.entity_extractor import ExtractionResult


@pytest.mark.asyncio
async def test_failed_chunk_is_recorded_not_swallowed(db_session, client, tmp_path):
    """A chunk that errors must be counted in upload.ingestion_errors, not silently dropped."""
    from app.api import upload as upload_mod
    from tests.conftest import auth_headers, create_test_patient

    headers, uid = await auth_headers(client)
    await create_test_patient(db_session, uid)

    # An RTF file extracts locally (no Gemini for text) → deterministic text path.
    f = tmp_path / "note.rtf"
    f.write_text(r"{\rtf1\ansi Patient on metformin. BP 120/80. A1c 6.0.}")

    # Make every entity-extraction call return an error result (simulates exhausted retries).
    async def _failing_extract(text, source_file, api_key):
        return ExtractionResult(source_file=source_file, source_text=text, error="429 rate exceeded")

    # Build an upload row + drive _process_unstructured with patched pieces.
    # (Mirror the fidelity test's patching of async_session_factory / _run_dedup_background.)
    # ... harness setup identical to test_unstructured_idempotency_fidelity.py ...
    # After running _process_unstructured, reload the upload and assert:
    #   - upload.ingestion_errors contains an entry with stage == "entity_extraction"
    #     and failed_chunks >= 1
    #   - the upload did NOT end in a plain "completed"/"dedup_scanning" with zero trace of failure
```

> NOTE for the implementer: the exact harness (session factory patch, status finalization) must match `test_unstructured_idempotency_fidelity.py`. If driving the full `_process_unstructured` is too heavy, instead unit-test the **gather-result loop** by extracting it into a small pure helper (see Step 3) and testing that helper directly with a mix of OK / error / Exception results. Prefer the helper approach if the full-pipeline harness is fragile — it's more focused and deterministic.

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_failure_surfacing.py -v`
Expected: FAIL (failures currently only `logger.warning`, never recorded on the upload).

- [ ] **Step 3: Implement**

In `backend/app/api/upload.py`, refactor the results loop (currently `:677-687`) to count failures, and record them on the upload. Replace:
```python
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Section extraction failed: %s", r)
                    continue
                extraction_result, section_type = r
                if extraction_result.error:
                    logger.warning("Extraction error in section %s: %s", section_type, extraction_result.error)
                    continue
                for entity in extraction_result.entities:
                    entity.attributes["_source_section"] = section_type
                    all_entities.append(entity)
```
with:
```python
            failed_chunks = 0
            for r in results:
                if isinstance(r, Exception):
                    failed_chunks += 1
                    logger.error("Section extraction raised: %s", r)
                    continue
                extraction_result, section_type = r
                if extraction_result.error:
                    failed_chunks += 1
                    logger.warning("Extraction error in section %s: %s", section_type, extraction_result.error)
                    continue
                for entity in extraction_result.entities:
                    entity.attributes["_source_section"] = section_type
                    all_entities.append(entity)

            if failed_chunks:
                errors = list(upload.ingestion_errors or [])
                errors.append({
                    "stage": "entity_extraction",
                    "failed_chunks": failed_chunks,
                    "total_chunks": len(extraction_tasks),
                    "error_type": "ExtractionChunkFailure",
                })
                upload.ingestion_errors = errors
                logger.warning(
                    "Extraction for %s: %d/%d chunks failed (records may be incomplete)",
                    upload.id, failed_chunks, len(extraction_tasks),
                )
```
(If a pure helper was chosen in Step 1, factor the loop into `def _collect_entities(results, extraction_task_count) -> tuple[list, int]` returning `(entities, failed_chunks)` and call it here; test the helper directly.)

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_failure_surfacing.py -v`
Expected: PASS. Regression: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_upload.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/api/upload.py tests/test_extraction_failure_surfacing.py
git commit -m "feat: surface failed entity-extraction chunks on the upload (no silent drops)"
```

---

## Task 3: Pin (and fix) the conc=10 full-pipeline bug — GATING

The isolated spike proved entity extraction at conc=10 works (12/12). The full pipeline produced 0 records at `section_extraction_concurrency=10`. With Task 2 in place, the failure is now inspectable. Reproduce, identify the real cause, fix it.

**Files:**
- Investigate: `backend/app/api/upload.py` (the `_get_gemini_semaphore`/`section_sem` usage), `backend/app/services/extraction/*`
- Doc/test: a regression test for the confirmed cause

- [ ] **Step 1: Reproduce at conc=10 and capture the real errors**

Temporarily set `section_extraction_concurrency=10` (env override, not committed) and run the failure-surfacing path or the fidelity test, reading `upload.ingestion_errors` and logs:
```bash
cd backend && SECTION_EXTRACTION_CONCURRENCY=10 .venv/bin/python -m pytest tests/test_unstructured_idempotency_fidelity.py -v -m slow -rs --tb=long 2>&1 | tail -60
```
Capture the actual per-chunk error text. **Report the confirmed cause before fixing.**

- [ ] **Step 2: Identify the root cause (candidate hypotheses to confirm/reject)**

The most likely cause — verify it first: the module-level asyncio semaphores `_gemini_semaphore` / `_extraction_semaphore` (`upload.py:18-39`, created lazily via `asyncio.Semaphore(...)`) are **bound to the event loop that first created them**. If created in one loop and reused under a different loop / higher concurrency, asyncio raises `RuntimeError: ... bound to a different event loop`, which would make every `extract_chunk` fail → 0 records. Confirm by inspecting the captured error text. Other candidates if that's wrong: `split_large_section`/`parse_sections` returning 0 usable chunks at this input; a DB-session/connection issue; or the test-harness task-cancellation loop cancelling in-flight extraction. Use the captured errors to decide.

- [ ] **Step 3: Implement the minimal fix for the confirmed cause**

If it is the event-loop-bound semaphore (most likely): make the semaphores loop-safe — e.g. create them lazily **per running loop** (key a dict by `asyncio.get_running_loop()`), or construct the semaphore inside the async context that uses it rather than as a module singleton. Implement the smallest change that makes the semaphore valid for the current loop, preserving the `gemini_concurrency_limit` cap. If the cause is different, implement the minimal corresponding fix and note it.

- [ ] **Step 4: Add a regression test**

Add a test (in `tests/test_extraction_concurrency_setting.py` or a focused new file) that exercises the fixed mechanism so the bug can't silently return. For the semaphore case: a test that `_get_gemini_semaphore()` (or its replacement) returns a usable semaphore under a freshly-running event loop without raising. Keep it Gemini-free.

- [ ] **Step 5: Verify + commit**

Run the failure-surfacing + unstructured suites: `cd backend && .venv/bin/python -m pytest tests/test_unstructured_upload.py tests/test_extraction_failure_surfacing.py -q` → PASS.
```bash
cd backend && git add app/api/upload.py tests/  # plus any service file touched
git commit -m "fix: make extraction semaphores event-loop-safe (conc>3 dropped all records)"
```
(Adjust the message to the confirmed cause if it differs.)

---

## Task 4: Raise `section_extraction_concurrency` 3 → 10

Only after Task 3 confirms conc=10 works end-to-end (records preserved).

**Files:**
- Modify: `backend/app/config.py` (`:59`)
- Test: `backend/tests/test_extraction_concurrency_setting.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_extraction_concurrency_setting.py  (add if file exists from Task 3)
from __future__ import annotations

from app.config import settings


def test_section_extraction_concurrency_raised():
    assert settings.section_extraction_concurrency == 10
    # never exceed the global Gemini concurrency cap
    assert settings.section_extraction_concurrency <= settings.gemini_concurrency_limit
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_concurrency_setting.py -v`
Expected: FAIL (currently 3).

- [ ] **Step 3: Implement**

In `backend/app/config.py:59`, change `section_extraction_concurrency: int = 3` → `= 10` and update the explanatory comment to note conc=10 is now verified safe (Task 3 fixed the loop-binding bug; the global `gemini_concurrency_limit` remains the hard cap).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_concurrency_setting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/config.py tests/test_extraction_concurrency_setting.py
git commit -m "perf: raise entity-extraction concurrency to 10 (verified safe)"
```

---

## Task 5: (B) Investigate per-call overhead (context caching)

Findings-first; implement only if low-risk and behavior-preserving.

**Files:**
- Create: `docs/superpowers/specs/2026-05-29-entity-overhead-findings.md`

- [ ] **Step 1: Measure the prompt size**

Quantify the static prompt cost: `cd backend && .venv/bin/python -c "from app.services.extraction.clinical_examples import CLINICAL_EXTRACTION_PROMPT, CLINICAL_EXAMPLES; print('prompt chars:', len(CLINICAL_EXTRACTION_PROMPT)); print('examples:', len(CLINICAL_EXAMPLES))"` and estimate tokens (~chars/4). This confirms how much of the ~22s/call is the fixed few-shot payload.

- [ ] **Step 2: Investigate LangExtract caching support**

Determine whether the installed `langextract` exposes a way to (a) pass a Gemini cached-content handle, or (b) reduce examples per call. Inspect `lx.extract` signature/params: `cd backend && .venv/bin/python -c "import langextract as lx, inspect; print(inspect.signature(lx.extract))"` and check for any caching / model-config passthrough. Check whether the project can create a Gemini cached content (`client.caches.create(...)`) for the static prefix and feed it through.

- [ ] **Step 3: Write the findings doc**

Create `docs/superpowers/specs/2026-05-29-entity-overhead-findings.md` recording: measured prompt size/tokens, whether LangExtract supports caching, the estimated latency/cost saving, and a recommendation (implement now / defer / not feasible). If a safe, behavior-preserving win exists (e.g., LangExtract accepts a cached-content handle), note the concrete change; if it would alter which entities are produced, defer to Phase 2b.

- [ ] **Step 4: Commit**

```bash
cd backend && git add ../docs/superpowers/specs/2026-05-29-entity-overhead-findings.md
git commit -m "docs: entity-extraction per-call overhead findings (context caching investigation)"
```
(Implement a caching change as a follow-up task only if Step 3 concludes it is safe — otherwise this task is investigation-only.)

---

## Task 6: Fidelity re-measure — speedup with records preserved (slow)

**Files:**
- Create: `backend/tests/test_faster_extraction_fidelity.py`

- [ ] **Step 1: Write the slow test**

Mirror the harness in `test_unstructured_idempotency_fidelity.py` (real note, `GEMINI_API_KEY`, `@pytest.mark.slow`, patched session factory + `_run_dedup_background`). Drive `_process_unstructured` once on the real note and assert:
```python
# Pseudostructure — copy the exact harness from test_unstructured_idempotency_fidelity.py
#   import time; t0 = time.monotonic(); await _process_unstructured(...); elapsed = time.monotonic() - t0
#   n = count ai_extracted records for the patient
#   assert n > 0, "records must be produced"
#   assert n >= EXPECTED_BASELINE * 0.9, "record count must not regress vs conc=3 baseline"
#   assert elapsed < 180, f"entity extraction should be well under the ~467s baseline, got {elapsed:.0f}s"
#   assert no entity_extraction failure entry in upload.ingestion_errors
```
Set `EXPECTED_BASELINE` from the conc=3 baseline (the implementer captures the conc=3 record count once during Task 3/6 and pins it as a constant with a comment). The 180s ceiling is generous (target ~60–90s) to avoid flakiness while still catching a regression to multi-minute.

- [ ] **Step 2: Run (slow; needs key)**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faster_extraction_fidelity.py -v -m slow -rs`
Report: record count vs baseline, and **wall-clock** (the headline). One run, no retry loop.

- [ ] **Step 3: Commit**

```bash
cd backend && git add tests/test_faster_extraction_fidelity.py
git commit -m "test: faster-extraction fidelity — record count preserved, wall-clock cut"
```

---

## Task 7: Regression + lint + local CLAUDE.md update

- [ ] **Step 1: Lint touched files**

Run: `cd backend && .venv/bin/ruff check app/config.py app/api/upload.py tests/test_model_config.py tests/test_extraction_failure_surfacing.py tests/test_extraction_concurrency_setting.py tests/test_faster_extraction_fidelity.py`
Fix feature-introduced errors only; report pre-existing.

- [ ] **Step 2: Full fast suite**

Run: `cd backend && .venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (baseline 515 + new fast tests). Investigate/fix real regressions; report unrelated pre-existing failures.

- [ ] **Step 3: Update CLAUDE.md LOCALLY (do NOT commit/push)**

Edit `CLAUDE.md` to replace `gemini-3-flash-preview` and `gemini-2.5-flash` with `gemini-3.5-flash` at: the AI Models list (lines ~158-159), the env example (lines ~360-361, and `PROMPT_TARGET_MODEL` ~367), and **Rule 16** (~436). Leave the file **unstaged/uncommitted** — the maintainer commits it if they choose. Verify with `git status` that `CLAUDE.md` shows as modified-but-unstaged and is NOT part of any commit.

- [ ] **Step 4: Final commit (code only, NOT CLAUDE.md)**

```bash
cd backend && git add -A ':!../CLAUDE.md'
git commit -m "chore: lint + regression pass for faster entity extraction" || echo "nothing to commit"
```
Then confirm: `git status` shows only `CLAUDE.md` as uncommitted.

---

## Self-Review Notes (author)

- **Spec coverage:** adopt 3.5-flash → Task 1 (+ Task 7 CLAUDE.md local); never-drop-records → Task 2; pin conc=10 bug → Task 3 (gating); raise concurrency → Task 4; investigate B → Task 5; fidelity speedup+no-drops → Task 6; CLAUDE.md local-only → Task 7. All covered.
- **Ordering:** Task 2 (surface failures) deliberately precedes Task 3 (diagnose) so the conc=10 failure is inspectable; Task 4 (raise concurrency) is gated on Task 3's fix.
- **Investigative honesty:** Task 3's fix code can't be fully pre-written (depends on the confirmed cause); the reproduction + the top hypothesis (event-loop-bound module-level semaphore) are concrete, and the task requires confirming the cause before fixing.
- **CLAUDE.md:** every commit step explicitly excludes it; Task 7 Step 4 uses a pathspec exclude and a final `git status` check.
- **Type consistency:** `failed_chunks: int`, `ingestion_errors` entry shape `{stage, failed_chunks, total_chunks, error_type}`, config names `gemini_model`/`gemini_extraction_model`/`section_extraction_concurrency` used consistently.
