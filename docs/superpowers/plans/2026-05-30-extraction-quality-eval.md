# Extraction Quality Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, committable harness that measures clinical entity-extraction quality on non-standard documents (transcripts, phone notes), produce a baseline, and explain the system in plain language — WITHOUT changing extraction behavior (that's Phase 2b-2).

**Architecture:** A pure, unit-tested scorer (`eval/scorer.py`) computes precision/recall/F1 + negation + speaker-attribution metrics from ground-truth labels vs extracted entities. Synthetic labeled fixtures (transcript + phone note + `.expected.json`) feed it. A slow test runs the real extractor on the fixtures, scores, and logs a baseline. A plain-language doc explains it.

**Tech Stack:** Python 3.12, pytest, LangExtract/Gemini (only in the slow baseline test). Spec: `docs/superpowers/specs/2026-05-30-extraction-quality-eval-design.md`. Branch `feat/extraction-quality-eval` (stacked on `feat/robust-section-parsing`).

**Conventions:** `from __future__ import annotations`. Type hints. No `print()` — `logging`. **Run pytest via `cd backend && .venv/bin/python -m pytest …`** (global pyenv broken). Ruff `check` is the gate. `ExtractedEntity(entity_class, text, attributes, start_pos, end_pos, confidence)`. Valid entity classes: medication, condition, lab_result, vital, procedure, allergy, encounter, imaging_result, family_history, assessment_plan, social_history (+ non-storable provider/dosage/route/frequency/duration/date).

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `backend/app/services/extraction/eval/__init__.py` | package marker | Create |
| `backend/app/services/extraction/eval/scorer.py` | pure scorer: normalize, match, score → EvalReport | Create |
| `backend/tests/test_extraction_scorer.py` | exhaustive scorer unit tests (no Gemini) | Create |
| `backend/tests/fixtures/extraction_eval/transcript_visit.txt` | synthetic transcript w/ traps | Create |
| `backend/tests/fixtures/extraction_eval/transcript_visit.expected.json` | ground truth | Create |
| `backend/tests/fixtures/extraction_eval/phone_note.txt` | synthetic terse note | Create |
| `backend/tests/fixtures/extraction_eval/phone_note.expected.json` | ground truth | Create |
| `backend/tests/test_extraction_eval_fixtures.py` | fixture-validity tests (no Gemini) | Create |
| `backend/tests/test_extraction_eval.py` | slow baseline eval (real Gemini) | Create |
| `docs/superpowers/specs/2026-05-30-extraction-quality-baseline.md` | measured baseline findings | Create (by Task 4 run) |
| `docs/extraction-eval-explained.md` | plain-language explainer | Create |

---

## Task 1: The pure scorer

**Files:**
- Create: `backend/app/services/extraction/eval/__init__.py` (empty), `backend/app/services/extraction/eval/scorer.py`
- Test: `backend/tests/test_extraction_scorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_extraction_scorer.py
from __future__ import annotations

from app.services.extraction.entity_extractor import ExtractedEntity
from app.services.extraction.eval.scorer import normalize, score


def _ext(cls, text, **attrs):
    return ExtractedEntity(entity_class=cls, text=text, attributes=attrs)


def test_normalize_lowercases_and_collapses():
    assert normalize("  Type 2  DIABETES. ") == "type 2 diabetes"


def test_normalize_synonyms():
    assert normalize("HTN") == "hypertension"
    assert normalize("DM2") == "type 2 diabetes"


def test_perfect_match_scores_one():
    gt = {"expected": [
        {"entity_class": "condition", "text": "hypertension"},
        {"entity_class": "medication", "text": "lisinopril"},
    ]}
    extracted = [_ext("condition", "Hypertension"), _ext("medication", "lisinopril")]
    rep = score(gt, extracted)
    assert rep.overall.precision == 1.0 and rep.overall.recall == 1.0 and rep.overall.f1 == 1.0


def test_missed_entity_drops_recall():
    gt = {"expected": [
        {"entity_class": "condition", "text": "hypertension"},
        {"entity_class": "medication", "text": "lisinopril"},
    ]}
    rep = score(gt, [_ext("condition", "hypertension")])
    assert rep.overall.recall == 0.5
    assert any(m["text"] == "lisinopril" for m in rep.missed)


def test_extra_entity_drops_precision():
    gt = {"expected": [{"entity_class": "condition", "text": "hypertension"}]}
    rep = score(gt, [_ext("condition", "hypertension"), _ext("condition", "asthma")])
    assert rep.overall.precision == 0.5


def test_synonym_match_counts_as_hit():
    gt = {"expected": [{"entity_class": "condition", "text": "hypertension"}]}
    rep = score(gt, [_ext("condition", "HTN")])
    assert rep.overall.recall == 1.0


def test_partial_medication_match():
    # patient-reported med without dose still matches the expected med name
    gt = {"expected": [{"entity_class": "medication", "text": "omeprazole"}]}
    rep = score(gt, [_ext("medication", "omeprazole 20mg")])
    assert rep.overall.recall == 1.0


def test_per_type_breakdown():
    gt = {"expected": [
        {"entity_class": "condition", "text": "hypertension"},
        {"entity_class": "medication", "text": "lisinopril"},
    ]}
    rep = score(gt, [_ext("condition", "hypertension")])
    assert rep.per_type["condition"].recall == 1.0
    assert rep.per_type["medication"].recall == 0.0


def test_false_extraction_of_attribution_trap():
    # father's cancer must NOT be extracted as the patient's condition
    gt = {
        "expected": [],
        "must_not_extract": [
            {"entity_class": "condition", "text": "colon cancer", "reason": "attribution"}
        ],
    }
    rep = score(gt, [_ext("condition", "colon cancer")])
    assert rep.false_extractions  # non-empty
    assert rep.attribution_accuracy == 0.0


def test_attribution_trap_respected():
    gt = {
        "expected": [],
        "must_not_extract": [
            {"entity_class": "condition", "text": "colon cancer", "reason": "attribution"}
        ],
    }
    rep = score(gt, [])  # correctly not extracted
    assert rep.false_extractions == []
    assert rep.attribution_accuracy == 1.0


def test_negation_trap_respected_and_violated():
    gt = {
        "expected": [],
        "must_not_extract": [
            {"entity_class": "condition", "text": "diabetes", "reason": "negation"}
        ],
    }
    assert score(gt, []).negation_accuracy == 1.0
    bad = score(gt, [_ext("condition", "diabetes")])
    assert bad.negation_accuracy == 0.0
    assert bad.false_extractions


def test_expected_family_history_present():
    gt = {"expected": [], "expected_family_history": [{"text": "colon cancer"}]}
    assert score(gt, [_ext("family_history", "colon cancer")]).attribution_accuracy == 1.0
    assert score(gt, []).attribution_accuracy == 0.0  # family history missed
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_scorer.py -v`
Expected: FAIL (`ModuleNotFoundError: ...eval.scorer`).

- [ ] **Step 3: Implement**

Create `backend/app/services/extraction/eval/__init__.py` (empty). Create `scorer.py`:
```python
"""Pure scorer for clinical entity-extraction quality evaluation.

Compares ground-truth labels against extracted entities and returns precision/recall/F1
(overall + per type) plus negation and speaker-attribution accuracy. Deterministic; no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.extraction.entity_extractor import ExtractedEntity

# Light clinical synonym map (extend as the baseline reveals more).
_SYNONYMS = {
    "htn": "hypertension",
    "dm2": "type 2 diabetes",
    "t2dm": "type 2 diabetes",
    "dm": "diabetes",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "gerd": "gastroesophageal reflux disease",
}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, apply synonyms (whole-string)."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    return _SYNONYMS.get(t, t)


def _texts_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na  # partial (e.g. "omeprazole" in "omeprazole 20mg")


@dataclass
class PRF:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(round(precision, 4), round(recall, 4), round(f1, 4))


@dataclass
class EvalReport:
    overall: PRF
    per_type: dict[str, PRF]
    negation_accuracy: float
    attribution_accuracy: float
    false_extractions: list[dict] = field(default_factory=list)
    missed: list[dict] = field(default_factory=list)


def _matches(item: dict, extracted: list[ExtractedEntity]) -> bool:
    return any(
        e.entity_class == item["entity_class"] and _texts_match(e.text, item["text"])
        for e in extracted
    )


def score(ground_truth: dict[str, Any], extracted: list[ExtractedEntity]) -> EvalReport:
    expected: list[dict] = ground_truth.get("expected", [])
    must_not: list[dict] = ground_truth.get("must_not_extract", [])
    expected_fh: list[dict] = ground_truth.get("expected_family_history", [])

    # Precision/recall over the expected set.
    matched_expected = [e for e in expected if _matches(e, extracted)]
    missed = [e for e in expected if e not in matched_expected]
    tp = len(matched_expected)
    fn = len(missed)
    # extracted entities that match no expected item = false positives
    fp = sum(
        1 for e in extracted
        if not any(e.entity_class == x["entity_class"] and _texts_match(e.text, x["text"]) for x in expected)
    )
    overall = _prf(tp, fp, fn)

    per_type: dict[str, PRF] = {}
    classes = {x["entity_class"] for x in expected}
    for cls in classes:
        exp_c = [x for x in expected if x["entity_class"] == cls]
        ext_c = [e for e in extracted if e.entity_class == cls]
        t = sum(1 for x in exp_c if _matches(x, extracted))
        f = sum(1 for e in ext_c if not any(_texts_match(e.text, x["text"]) for x in exp_c))
        per_type[cls] = _prf(t, f, len(exp_c) - t)

    # Traps: anything in must_not_extract that WAS extracted is a false extraction.
    false_extractions = [m for m in must_not if _matches(m, extracted)]

    neg = [m for m in must_not if m.get("reason") == "negation"]
    neg_violated = [m for m in neg if _matches(m, extracted)]
    negation_accuracy = 1.0 if not neg else round(1 - len(neg_violated) / len(neg), 4)

    attr = [m for m in must_not if m.get("reason") == "attribution"]
    attr_violated = [m for m in attr if _matches(m, extracted)]
    fh_present = sum(
        1 for fh in expected_fh
        if any(e.entity_class == "family_history" and _texts_match(e.text, fh["text"]) for e in extracted)
    )
    attr_total = len(attr) + len(expected_fh)
    attr_correct = (len(attr) - len(attr_violated)) + fh_present
    attribution_accuracy = 1.0 if attr_total == 0 else round(attr_correct / attr_total, 4)

    return EvalReport(
        overall=overall,
        per_type=per_type,
        negation_accuracy=negation_accuracy,
        attribution_accuracy=attribution_accuracy,
        false_extractions=false_extractions,
        missed=missed,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_scorer.py -v`
Expected: PASS (all tests). Iterate the implementation against the tests until green.

- [ ] **Step 5: Commit**

```bash
cd backend && git add app/services/extraction/eval/ tests/test_extraction_scorer.py
git commit -m "feat: pure extraction-quality scorer (P/R/F1, negation, attribution)"
```

---

## Task 2: Synthetic labeled fixtures + validity tests

**Files:**
- Create: the 4 fixture files under `backend/tests/fixtures/extraction_eval/`
- Test: `backend/tests/test_extraction_eval_fixtures.py`

- [ ] **Step 1: Author the transcript fixture** — `backend/tests/fixtures/extraction_eval/transcript_visit.txt`

A short clinician↔patient transcript with speaker turns and the traps from the spec. Author realistic content, e.g.:
```
Dr. Lee: Good morning. What brings you in today?
Patient: I've had this burning stomach pain since about last week.
Dr. Lee: Are you taking anything for it?
Patient: Just something over the counter for my stomach. I stopped taking my lisinopril though.
Dr. Lee: Okay. Any history of diabetes?
Patient: No, never had diabetes. But my father had colon cancer.
Dr. Lee: Your blood pressure today is 142 over 90. I'm noting hypertension.
Dr. Lee: Generally, untreated reflux can cause esophagitis, but let's not get ahead of ourselves.
```
Traps embedded: patient-reported med w/o dose ("something ... for my stomach"); discontinued med ("stopped taking my lisinopril"); negation ("never had diabetes"); family-member condition ("father had colon cancer"); educational mention ("reflux can cause esophagitis" — must NOT be a patient condition); a real vital (142/90) + condition (hypertension).

- [ ] **Step 2: Author its ground truth** — `transcript_visit.expected.json`

```json
{
  "expected": [
    {"entity_class": "vital", "text": "142/90"},
    {"entity_class": "condition", "text": "hypertension"}
  ],
  "must_not_extract": [
    {"entity_class": "condition", "text": "diabetes", "reason": "negation"},
    {"entity_class": "condition", "text": "colon cancer", "reason": "attribution"},
    {"entity_class": "condition", "text": "esophagitis", "reason": "educational"}
  ],
  "expected_family_history": [
    {"text": "colon cancer"}
  ]
}
```
(Note: "stopped taking lisinopril" — the med may legitimately be extracted as a discontinued/historical medication; leave it out of both lists to avoid penalizing either choice, OR add it to `expected` with a note. Keep the ground truth conservative and defensible; document choices in a top-of-file comment is not possible in JSON, so record rationale in the layman doc / baseline doc.)

- [ ] **Step 3: Author the phone-note fixture** — `phone_note.txt`

```
pt c/o HTN, on lisinopril 10mg. DM2 dx 2019, metformin 500 bid. allergic PCN.
no chest pain. mom breast ca.
```

- [ ] **Step 4: Author its ground truth** — `phone_note.expected.json`

```json
{
  "expected": [
    {"entity_class": "condition", "text": "hypertension"},
    {"entity_class": "medication", "text": "lisinopril"},
    {"entity_class": "condition", "text": "type 2 diabetes"},
    {"entity_class": "medication", "text": "metformin"},
    {"entity_class": "allergy", "text": "penicillin"}
  ],
  "must_not_extract": [
    {"entity_class": "condition", "text": "chest pain", "reason": "negation"},
    {"entity_class": "condition", "text": "breast cancer", "reason": "attribution"}
  ],
  "expected_family_history": [
    {"text": "breast cancer"}
  ]
}
```

- [ ] **Step 5: Write fixture-validity tests** — `backend/tests/test_extraction_eval_fixtures.py`

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.extraction.entity_to_fhir import ENTITY_TO_RECORD_TYPE

_FIX = Path(__file__).resolve().parent / "fixtures" / "extraction_eval"
_PAIRS = [("transcript_visit.txt", "transcript_visit.expected.json"),
          ("phone_note.txt", "phone_note.expected.json")]
_VALID = set(ENTITY_TO_RECORD_TYPE.keys())


@pytest.mark.parametrize("txt,gt", _PAIRS)
def test_fixture_pair_exists_and_parses(txt, gt):
    assert (_FIX / txt).read_text().strip()
    data = json.loads((_FIX / gt).read_text())
    assert "expected" in data


@pytest.mark.parametrize("txt,gt", _PAIRS)
def test_expected_entity_classes_are_valid(txt, gt):
    data = json.loads((_FIX / gt).read_text())
    for item in data.get("expected", []) + data.get("must_not_extract", []):
        assert item["entity_class"] in _VALID, f"unknown class {item['entity_class']}"
```

- [ ] **Step 6: Run + commit**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_eval_fixtures.py -v` → PASS.
```bash
cd backend && git add tests/fixtures/extraction_eval/ tests/test_extraction_eval_fixtures.py
git commit -m "test: synthetic labeled fixtures for extraction-quality eval (transcript + phone note)"
```

---

## Task 3: Slow baseline eval (real Gemini) + baseline findings doc

**Files:**
- Create: `backend/tests/test_extraction_eval.py`
- Create: `docs/superpowers/specs/2026-05-30-extraction-quality-baseline.md` (written from the run output)

- [ ] **Step 1: Write the slow eval test**

```python
# backend/tests/test_extraction_eval.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.config import settings
from app.services.extraction.entity_extractor import extract_entities_async
from app.services.extraction.eval.scorer import score

logger = logging.getLogger(__name__)
_FIX = Path(__file__).resolve().parent / "fixtures" / "extraction_eval"
_REAL = Path(__file__).resolve().parents[2] / "test_data" / "extraction_eval"
_PAIRS = [("transcript_visit.txt", "transcript_visit.expected.json"),
          ("phone_note.txt", "phone_note.expected.json")]


@pytest.mark.slow
@pytest.mark.skipif(not settings.gemini_api_key, reason="GEMINI_API_KEY required")
@pytest.mark.parametrize("txt,gt", _PAIRS)
@pytest.mark.asyncio
async def test_baseline_extraction_quality(txt, gt):
    text = (_FIX / txt).read_text()
    ground_truth = json.loads((_FIX / gt).read_text())
    result = await extract_entities_async(text, txt, settings.gemini_api_key)
    rep = score(ground_truth, result.entities)
    logger.warning(
        "[eval %s] overall=%s negation=%.2f attribution=%.2f false_extractions=%s missed=%s",
        txt, rep.overall, rep.negation_accuracy, rep.attribution_accuracy,
        [f"{m['entity_class']}:{m['text']}" for m in rep.false_extractions],
        [f"{m['entity_class']}:{m['text']}" for m in rep.missed],
    )
    # Loose floors only (extraction is non-deterministic): no hard hallucination of trap content.
    assert rep.false_extractions == [], (
        f"hallucinated trap content for {txt}: {rep.false_extractions}"
    )
    assert rep.overall.recall >= 0.5, f"recall floor breached for {txt}: {rep.overall}"
```

> If `assert rep.false_extractions == []` fails at baseline (the model DOES hallucinate a trap), that is a real finding, not a flaky test — capture it in the baseline doc and SOFTEN this to a logged warning + a documented known-gap for 2b-2, rather than leaving the suite red. Decide based on the actual run.

- [ ] **Step 2: Run the baseline (slow, needs key). ONE run, no retry loops.**

Run: `cd backend && .venv/bin/python -m pytest tests/test_extraction_eval.py -v -m slow -rs -s 2>&1 | grep -E "eval |PASSED|FAILED"`
Capture the logged `EvalReport` lines for both fixtures.

- [ ] **Step 3: Write the baseline findings doc**

Create `docs/superpowers/specs/2026-05-30-extraction-quality-baseline.md` with the captured numbers: per-fixture overall P/R/F1, negation accuracy, attribution accuracy, the specific missed entities and any false extractions, plus a short "Top gaps for 2b-2" list (e.g. "attribution: father's colon cancer extracted as patient condition" if observed). This is the concrete input to the 2b-2 design.

- [ ] **Step 4: Commit**

```bash
cd backend && git add tests/test_extraction_eval.py ../docs/superpowers/specs/2026-05-30-extraction-quality-baseline.md
git commit -m "test: baseline extraction-quality eval + findings doc"
```

---

## Task 4: Plain-language explainer

**Files:**
- Create: `docs/extraction-eval-explained.md`

- [ ] **Step 1: Write the explainer**

Write `docs/extraction-eval-explained.md` for a non-technical reader. Cover, in plain words (define any term):
- What the extractor does (turns a document into structured health records) and why non-standard docs (a recorded visit *transcript*, a quick *phone note*) are harder than clean clinical notes.
- What we measure and why, each with a one-line plain definition + a tiny example:
  - **Recall** — "of the things that should have been found, how many did we find?"
  - **Precision** — "of the things we found, how many were actually right?"
  - **Negation accuracy** — "when the note says the patient does NOT have something, did we correctly avoid recording it? (e.g. 'never had diabetes')"
  - **Attribution accuracy** — "when a condition belongs to someone else, did we keep it off the patient's record? Worked example: the note says the patient's *father* had colon cancer — the system must record that as *family history*, NOT as the patient's cancer."
  - **False extractions** — "things we wrongly recorded (the dangerous case for a health record)."
- What the synthetic fixtures + ground-truth files are ("a fake transcript we wrote, plus the list of correct answers, so we can grade the system").
- How to read a baseline report (point at the baseline doc).
- A sentence on why this matters: it lets us *prove* an extraction change is an improvement instead of guessing.
Keep it ~1 page, friendly, no jargon left undefined.

- [ ] **Step 2: Commit**

```bash
cd backend && git add ../docs/extraction-eval-explained.md
git commit -m "docs: plain-language explanation of the extraction-quality eval system"
```

---

## Task 5: Regression + lint

- [ ] **Step 1: Lint**

Run: `cd backend && .venv/bin/ruff check app/services/extraction/eval/ tests/test_extraction_scorer.py tests/test_extraction_eval_fixtures.py tests/test_extraction_eval.py`
Fix feature-introduced errors; report pre-existing.

- [ ] **Step 2: Full fast suite**

Run: `cd backend && .venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (baseline 536 + new fast tests: scorer + fixture-validity). Fix real regressions; report unrelated pre-existing.

- [ ] **Step 3: Final commit (if changes)**

```bash
cd backend && git add -A && git commit -m "chore: lint + regression pass for extraction-quality eval" || echo "nothing to commit"
```

---

## Self-Review Notes (author)

- **Spec coverage:** synthetic fixtures + traps → Task 2; pure scorer (P/R/F1, negation, attribution, false-extractions) → Task 1; slow baseline + findings doc → Task 3; plain-language explainer (deliverable #5) → Task 4; fixture-validity → Task 2; regression → Task 5. All covered.
- **Measure-only scope:** no change to `clinical_examples.py`/extraction logic — improvements are Phase 2b-2, designed from Task 3's baseline.
- **Non-determinism handled:** the slow eval logs numbers and asserts only loose floors; Step 1's note tells the implementer to soften the false-extraction assertion to a documented known-gap if the baseline genuinely hallucinates (a finding, not a flaky test).
- **Type consistency:** `score(ground_truth: dict, extracted: list[ExtractedEntity]) -> EvalReport`; `EvalReport(overall: PRF, per_type, negation_accuracy, attribution_accuracy, false_extractions, missed)`; `normalize()`/`_texts_match()` used consistently; fixture ground-truth keys `expected` / `must_not_extract` (with `reason`) / `expected_family_history` match the scorer.
- **Parallelizable:** Task 1 (scorer), Task 2 (fixtures), Task 4 (explainer) are disjoint files → can run concurrently; Task 3 depends on 1+2; Task 5 last.
