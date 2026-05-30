# Faster Entity Extraction — Design (Phase 2d)

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/faster-entity-extraction` (stacked on `feat/extraction-performance` → `feat/unstructured-idempotency` → `feat/idempotent-incremental-ingestion`)
**Predecessor:** Phase 2c (`2026-05-29-extraction-performance-design.md`)

## Problem

Phase 2c made PDF *text* extraction fast (local-first) but the real wall is **entity
extraction**. Measured (live, this session):

- **Per-call latency ≈ 22s** for a single LangExtract→Gemini call — even for a 128-char chunk.
  This is essentially constant whether the chunk is tiny or full-size, which means the **static
  few-shot prompt (`CLINICAL_EXAMPLES`) dominates every call**, not the chunk content.
- **Per-call latency is model-independent:** `gemini-2.5-flash` 21.5s vs `gemini-3.5-flash`
  22.1s — identical, same 9 entities. A model upgrade does NOT help speed.
- **Concurrency is safe:** 12 concurrent extraction calls all succeeded (12/12, no 429s, no
  dropped records). The earlier "rate-limit ceiling" hypothesis was **disproven** by this spike.
- The note (35.5k chars) → ~18 chunks. At `section_extraction_concurrency=3` that's ~6 serial
  waves × ~22s ≈ the observed 467s. At concurrency ≈ the chunk count, it would be ~1–2 waves
  ≈ **60–90s**.
- **Unexplained:** the Phase 2c full-pipeline run at `section_extraction_concurrency=10`
  produced **0 records** in 73s. The isolated spike proves entity extraction at conc=10 is
  fine, so this failure has a *different* cause (section-parse/split, semaphore interaction,
  or a test-harness artifact) — **must be pinned before raising production concurrency.**

Account is on a **paid Gemini tier** (~1000+ RPM for flash), so concurrency headroom is real.

## Goal & Contract

Cut entity-extraction wall-clock toward **~60–90s** for the reference note (from ~467s), with
**record count unchanged** vs the `section_extraction_concurrency=3` baseline. Adopt
`gemini-3.5-flash`. **Hard rule: never silently drop records** — a transient/rate failure must
be surfaced on the upload, not swallowed into a "completed" upload with missing data.

## Scope

- **(A) Raise concurrency safely** — the proven primary lever. Gated on first pinning the
  conc=10 pipeline bug.
- **(B) Investigate per-call overhead reduction** — context caching / prompt trimming.
  Findings report; implement only if low-risk and behavior-preserving.
- **Adopt `gemini-3.5-flash`** for both model configs.

Out of scope: extraction *quality* / model-quality evaluation (Phase 2b); PDF/idempotency
(shipped 2a/2c).

## Components & Changes

### 1. Adopt `gemini-3.5-flash`
- `app/config.py`: `gemini_extraction_model` `"gemini-2.5-flash"` → `"gemini-3.5-flash"`;
  `gemini_model` `"gemini-3-flash-preview"` → `"gemini-3.5-flash"` (consolidate on GA, drop
  the preview). Confirmed available on the key (`models/gemini-3.5-flash`, supports
  `generateContent` + `batchGenerateContent`).
- `.env.example`: update the model defaults to match.
- **`CLAUDE.md` Rule 16**: update locally to name `gemini-3.5-flash`. **Left uncommitted /
  never pushed** per the maintainer's standing rule — the maintainer commits it if desired.
- Speed-neutral by measurement; adopted for currency/quality.

### 2. (A) Pin the conc=10 pipeline bug — GATING first task
A diagnostic that reproduces the **full** `_process_unstructured` path on the reference note at
`section_extraction_concurrency=10` and identifies why it produced 0 records (isolated entity
extraction at conc=10 works). Likely suspects: section parse/split returning unusable sections,
the global `_gemini_semaphore` vs `section_sem` interaction, the test-harness task-cancellation
loop, or DB-session/connection handling. Output: the confirmed root cause + the minimal fix.
**No production concurrency increase ships until this is understood and fixed.**

### 3. (A) Raise concurrency + surface failures
- After the bug is fixed: raise `section_extraction_concurrency` 3 → 10 (bounded by the global
  `gemini_concurrency_limit = 10`).
- **Stop silent drops** (`entity_extractor.extract_entities` + the gather loop in
  `api/upload.py` Step 4): today a chunk that fails after retries returns `error` + 0 entities,
  logged at WARNING, but the upload still finishes "completed" with missing records. Change so
  failed chunks are **counted and recorded on the upload** — e.g. add the failure count to
  `ingestion_errors` and set status `completed_with_errors` when any chunk failed. Optionally
  retry a failed chunk once more (with jittered backoff) before recording it as failed.

### 4. (B) Investigate per-call overhead (spike → report; implement only if cheap/safe)
~22s for 128 chars ⟹ the static prompt + `CLINICAL_EXAMPLES` dominate each call. Investigate:
(a) whether LangExtract exposes a way to use Gemini **context caching** (`createCachedContent`)
for the identical prompt/examples prefix across all chunks; (b) trimming the example set.
Deliverable: a findings note (`docs/superpowers/specs/2026-05-29-entity-overhead-findings.md`)
and an implementation **only if** it is low-risk and does not change extraction behavior.
Anything that affects which entities are produced is deferred to Phase 2b.

## Error Handling & Safety

- **Never silently drop records** (the central correctness rule of this phase).
- All Gemini calls stay bounded by `gemini_concurrency_limit` (global semaphore) — no unbounded
  fan-out (respects the project's DB-pool / concurrency gotchas).
- De-identification path unchanged (PHI scrubbing still precedes any Gemini call).
- No schema change required for surfacing failures (reuse `ingestion_errors` JSONB +
  `ingestion_status`); if a dedicated field is cleaner, it is additive only.

## Test Strategy (expected-output-first)

- **Diagnostic (gating):** reproduce the conc=10 full-pipeline behavior; commit the finding +
  a regression test that the fixed cause stays fixed.
- **Model-id config tests:** assert `gemini_extraction_model == "gemini-3.5-flash"` and
  `gemini_model == "gemini-3.5-flash"`.
- **No-silent-drop tests (no live Gemini):** patch `extract_entities_async` so one chunk raises
  / returns an error → assert the upload surfaces it (failure count recorded, status reflects
  partial failure) rather than silently completing clean.
- **Concurrency unit test:** N concurrent chunks via the pipeline's gather path all return their
  entities (mirrors the spike, mocked).
- **Fidelity / slow (real note, `GEMINI_API_KEY`):** re-run the note pipeline at raised
  concurrency → assert (1) `ai_extracted` record count **== the conc=3 baseline** (no drops),
  (2) wall-clock **materially below 467s** (assert a generous ceiling, e.g. < 180s, to catch
  regression without flakiness). This is the verified before/after.
- Full fast-suite regression (baseline 515 passed).

## Validation Plan (post-build)

1. Re-run the slow note fidelity test → confirm record count preserved and wall-clock cut.
2. Upload the note via the app → confirm fast extraction and that any failed chunk surfaces in
   the upload status rather than silently reducing the record count.
3. Confirm summaries / PDF-vision fallback still work on `gemini-3.5-flash`.

## Key Lesson Carried Forward

Every performance claim in this phase is grounded in a measured spike, not inference — after
Phase 2c's "rate-limit" misdiagnosis, the rule is: reproduce and measure the real failure
before designing the fix. A faster *failing* run is not a speedup.
