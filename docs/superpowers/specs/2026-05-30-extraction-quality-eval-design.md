# Extraction Quality Evaluation Harness — Design (Phase 2b-1)

**Date:** 2026-05-30
**Status:** Approved (design); pending implementation plan
**Branch:** `feat/extraction-quality-eval` (stacked on `feat/robust-section-parsing` → … → `feat/idempotent-incremental-ingestion`)
**Context:** First half of Phase 2b (extraction quality on non-standard docs). 2b-2 (the actual
prompt/extraction improvements) is a separate spec, designed AFTER this harness reports the baseline.

## Problem

We want to improve clinical entity-extraction quality on non-standard documents (transcripts,
phone/iPhone notes), but there is currently **no way to measure extraction quality**:

- The prompt has guards (negation, family history, educational-text, confidence) + 4 few-shot
  examples (`clinical_examples.py`), but tests are assertion-based on synthetic FHIR mapping —
  **no ground-truth labeled fixtures and no precision/recall harness**.
- **No transcript or phone-note fixtures exist.**
- Predicted degradations on conversational/informal text (unmeasured): conversational negation
  ("stopped taking", "never had"), relative dates ("since last week"), patient-reported meds
  without doses, abbreviations (HTN, DM2), and **speaker attribution** (a transcript mixes
  clinician + patient speech; a clinician-mentioned or family-member condition must not be
  extracted as the patient's).

This session repeatedly showed that **unmeasured "improvements" are guesses** (two illusory
speedups). So Phase 2b is **measure-first**: build the evaluation harness and a baseline before
tuning anything. This is the standing TDD rule ("define expected output, compare to real
output") applied to extraction quality.

## Goal & Contract

Deliver a deterministic, committable harness that quantifies extraction quality on non-standard
documents, plus a **measured baseline** identifying exactly where extraction fails — the input
to 2b-2. Plus a **plain-language explanation** of the eval system for non-technical readers.

No changes to extraction behavior in this spec (measurement only).

## Components

### 1. Synthetic labeled fixtures (committed, no PHI) — `backend/tests/fixtures/extraction_eval/`
- `transcript_visit.txt` — a clinician↔patient visit transcript with speaker turns, containing
  deliberate **traps**:
  - a condition the **clinician** mentions generically (not the patient's) → must NOT be the
    patient's condition;
  - a **family member's** condition → must be `family_history`, not a patient condition;
  - **conversational negation** ("I stopped taking lisinopril", "never had diabetes");
  - a **relative date** ("since about last week");
  - a **patient-reported med without dose** ("something for my stomach").
- `phone_note.txt` — terse iPhone-note style: abbreviations (HTN, DM2), fragments, no structure.
- `transcript_visit.expected.json` / `phone_note.expected.json` — ground truth per fixture:
  - `expected`: list of entities that SHOULD be extracted — each `{entity_class, text,
    status?}` (text normalized; `status` for negated/active where it matters).
  - `must_not_extract`: list of trap items that must NOT appear as patient entities (e.g. the
    clinician-mentioned or family-member condition), each `{entity_class, text, reason}`.
  - `expected_family_history`: family-member conditions that SHOULD appear as `family_history`.

### 2. Pure scorer — `backend/app/services/extraction/eval/scorer.py`
- Dataclasses: `EvalReport` (precision, recall, f1 overall; `per_type: dict[str, PRF]`;
  `negation_accuracy: float`; `attribution_accuracy: float`; `false_extractions: list`;
  `missed: list`).
- `normalize(text) -> str`: lowercase, collapse whitespace, strip punctuation; apply a small
  synonym map (e.g. `htn→hypertension`, `dm2→type 2 diabetes`, `t2dm→type 2 diabetes`,
  `sob→shortness of breath`).
- `match(expected_item, extracted_item) -> bool`: same `entity_class` AND
  `normalize` text equal (or one contains the other for partial patient-reported meds).
- `score(ground_truth: dict, extracted: list[ExtractedEntity]) -> EvalReport`:
  - precision/recall/F1 overall and per `entity_class` from `expected` vs `extracted`;
  - **negation_accuracy**: of the `expected` items marked negated + the negation traps,
    fraction handled correctly (skipped or status negated/inactive);
  - **attribution_accuracy**: fraction of `must_not_extract` + `expected_family_history` items
    correctly handled (trap not extracted as patient entity; family items present as
    `family_history`);
  - **false_extractions**: extracted items matching any `must_not_extract` trap (hard signal).
- Pure and fully unit-tested with fixed `(ground_truth, extracted)` pairs — **no Gemini**.

### 3. Baseline eval (slow, real Gemini) — `backend/tests/test_extraction_eval.py`
- For each synthetic fixture: run the real extractor (`extract_entities_async` on the fixture
  text → entities), `score(...)`, and **log the full `EvalReport`**.
- Loose floor-assertions only (extraction is non-deterministic): assert `false_extractions == []`
  (no hard hallucination of trap content) and overall `recall >= a documented floor`; everything
  else is logged, not gated. `@pytest.mark.slow`, skip without `GEMINI_API_KEY`.
- Also evaluates real gitignored fixtures under `test_data/extraction_eval/` when present
  (skip otherwise) — mirrors the project's synthetic-always / real-when-present pattern.

### 4. Baseline findings doc — `docs/superpowers/specs/2026-05-30-extraction-quality-baseline.md`
Committed report of the measured numbers (overall + per-type P/R/F1, negation accuracy,
attribution accuracy, the specific traps missed/wrongly-extracted) → the concrete input to 2b-2.

### 5. Plain-language explanation (requested) — `docs/extraction-eval-explained.md`
A non-technical explainer: what "extraction quality" means, what a transcript/phone-note is and
why they're hard, what precision/recall/negation/attribution measure (in plain words, with a
tiny worked example like "the note says the patient's *father* had cancer — the system must NOT
record that as the patient's cancer"), what the synthetic fixtures + ground truth are, and how
to read a baseline report. ~1 page, no jargon (define any term used).

## Error Handling & Safety

- PHI scrubbing path unchanged; real fixtures gitignored; scorer pure/deterministic.
- The slow eval logs numbers and asserts only loose floors (no brittle exact-match against
  non-deterministic LLM output).
- Absolute Rule 1 holds: the eval **measures organization/extraction accuracy only** — no
  diagnosis or medical advice is produced or evaluated.

## Test Strategy (expected-output-first)

- **Scorer unit tests (deterministic, no Gemini):** perfect match → P/R/F1 == 1.0; a missed
  expected entity → recall < 1; a trap extracted → `false_extractions` non-empty + attribution
  accuracy < 1; a negated item correctly skipped → negation_accuracy == 1; synonym match
  (HTN↔hypertension) works; partial med match ("omeprazole" vs "omeprazole 20mg") matches.
- **Fixture validity test (no Gemini):** each `.expected.json` parses and references entity
  classes that exist in `ENTITY_TO_RECORD_TYPE`; fixture text contains each expected `text`
  (so the ground truth is self-consistent with its document).
- **Slow baseline eval:** runs real extraction on synthetic fixtures, scores, logs the report,
  asserts the loose floors. Writes/refreshes the baseline findings doc.

## Out of Scope (→ Phase 2b-2)

- Any change to `clinical_examples.py` prompt/examples or extraction logic — that is 2b-2,
  designed from this baseline.
- New entity types or FHIR mapping changes.

## Validation Plan (post-build)

1. Run the scorer unit tests → deterministic, green.
2. Run the slow baseline eval (with `GEMINI_API_KEY`) → capture the report; commit the baseline
   findings doc; confirm it surfaces concrete gaps (e.g. attribution/negation numbers).
3. Have a non-technical reader sanity-check `docs/extraction-eval-explained.md`.
4. Hand the baseline numbers to Phase 2b-2 design.
