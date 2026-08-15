# Strict-local extraction remediation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make validated strict-local extraction reject administrative procedure assertions that lack performance wording, scope medication lifecycle signals to their own subject, and prove every extraction completes or fails before its 12-call budget is exceeded.

**Architecture:** Track B changes two deliberately independent validation boundaries: the backend remains authoritative and the contained MLX worker applies mirrored conservative grounding before it emits a fact. A versioned deterministic JSON case matrix is loaded by both test suites so each package proves the same procedure and medication decisions. The worker pre-fits all pages before inference, then uses one ordered pending-batch queue whose split admission reserves one minimum call for every untouched batch and both children.

**Tech Stack:** Python 3.11, FastAPI/Pydantic backend validation, contained Apple MLX worker, pytest, Ruff, deterministic synthetic fixtures.

## Global Constraints

- Validated strict-local routing must branch before any cloud-capable provider is constructed and must never fall back to cloud.
- The backend is the authoritative clinical-validation boundary; worker grounding is a second, mirrored guard and cannot replace backend rejection.
- Keep progress, errors, and operation results content-free; do not log source text, prompts, model output, identities, paths, or raw exceptions.
- Keep the hard extraction limits at **12 generation attempts** and **32,768 generated tokens**. No thirteenth generation call is permitted.
- Keep `MAX_EXTRACTION_FRAGMENT_DEPTH = 5`; reduce the static split ceiling to **11**. The dynamic queue-capacity admission is authoritative.
- Use deterministic synthetic text only. Do not call providers, download models, invoke private fixtures, or run live-cloud tests.
- Preserve existing assertion/status precedence inside the newly bounded source span.
- Keep `from __future__ import annotations`, type hints, Google-style docstrings, and Ruff's 100-column limit.
- This plan owns Track B only and is independently executable in parallel with Tracks A and C. Do not modify their API, authorization, durable-job, migration, frontend, or runtime-attestation files.
- Subagents must not commit. The root agent reviews and verifies Track B. Creating the single Track
  B commit requires Pedro's explicit authorization; once accepted, that commit is Track D's
  prerequisite.
- Start this plan in a Codex-managed **Worktree** task based on
  `codex/pr62-pr63-remediation-planning`; retain it as
  `codex/strict-local-extraction-remediation` only when the implementation is ready for root
  review.

---

## File map

- Create: `workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json`: shared synthetic procedure/lifecycle decision matrix consumed by both packages.
- Modify: `backend/app/services/local_ai/extraction_validator.py`: administrative procedure gate and subject-local lifecycle blockers in the authoritative validator.
- Modify: `backend/tests/test_local_ai_extraction_validation.py`: loads the shared matrix and proves backend acceptance/rejection.
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py`: mirrored grounding rules plus pre-fit and runtime queue-capacity admission.
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`: loads the shared matrix and proves worker grounding, pre-fit rejection, and a maximum of 12 stubbed generations.
- Do not modify: `backend/app/api/`, `backend/app/services/ai/`, `backend/alembic/`, `backend/app/models/`, frontend files, manifests, locks, private fixtures, or provider configuration.

### Task 1: Add the shared deterministic clinical-grounding case matrix

**Files:**
- Create: `workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json`
- Modify: `backend/tests/test_local_ai_extraction_validation.py:1-31`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py:1-39`

**Interfaces:**
- Produces: JSON object `{"version": 1, "procedure_cases": list[dict[str, object]], "medication_cases": list[dict[str, object]]}`.
- Each procedure item has `id`, `name`, `context`, `page`, `date`, `backend_present_valid`, and `worker_assertion`.
- Each medication item has `id`, `name`, `context`, `frequency`, `candidate_status`, `backend_valid`, and `worker_status`. `frequency` is either the source-grounded frequency string or `null`; it prevents a lifecycle regression from failing first on an unrelated critical-field locator.
- Consumed by: `_ground_explicit_assertions()` worker tests and `validate_clinical_extraction()` backend tests in later tasks.

- [ ] **Step 1: Write the failing matrix-loader tests**

Add this helper near the imports in both test files. The backend test uses
`parents[2]`; the worker test uses `parents[4]`.

```python
from pathlib import Path


def _grounding_cases() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (
            root
            / "workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json"
        ).read_text(encoding="utf-8")
    )


def test_strict_local_grounding_case_matrix_is_versioned_and_nonempty() -> None:
    cases = _grounding_cases()
    assert cases["version"] == 1
    assert {case["id"] for case in cases["procedure_cases"]} == {
        "authorization_validity_date_is_not_performance",
        "authorization_with_status_post_is_performance",
        "ordinary_clinical_date_remains_compatible",
    }
    assert {case["id"] for case in cases["medication_cases"]} == {
        "other_medication_blocker_cannot_reject_subject",
        "other_medication_active_cue_cannot_promote_subject",
        "own_dosing_continuation_supports_subject",
        "repeated_subject_fails_closed",
    }
```

In `workers/local_ai/apple_mlx/tests/test_protocol.py`, the identical helper
uses this root declaration instead:

```python
    root = Path(__file__).resolve().parents[4]
```

- [ ] **Step 2: Run the two loaders to verify they fail**

Run: `cd backend && uv run pytest -q tests/test_local_ai_extraction_validation.py::test_strict_local_grounding_case_matrix_is_versioned_and_nonempty`

Expected: FAIL with `FileNotFoundError` for `strict_local_extraction_grounding_cases.json`.

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py::test_strict_local_grounding_case_matrix_is_versioned_and_nonempty`

Expected: FAIL with the same missing-fixture error.

- [ ] **Step 3: Create the complete synthetic matrix**

Create `workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json` with exactly this content. It deliberately has no patient identifiers, provider input, or private medical text.

```json
{
  "version": 1,
  "procedure_cases": [
    {
      "id": "authorization_validity_date_is_not_performance",
      "name": "Colonoscopy",
      "context": "Authorization for Colonoscopy valid from 2026-01-01 through 2026-12-31.",
      "page": "Authorization for Colonoscopy valid from 2026-01-01 through 2026-12-31.",
      "date": "2026-01-01",
      "backend_present_valid": false,
      "worker_assertion": "mentioned_not_performed"
    },
    {
      "id": "authorization_with_status_post_is_performance",
      "name": "Colonoscopy",
      "context": "Authorization record notes status post Colonoscopy performed on 2026-01-01.",
      "page": "Authorization record notes status post Colonoscopy performed on 2026-01-01.",
      "date": "2026-01-01",
      "backend_present_valid": true,
      "worker_assertion": "present"
    },
    {
      "id": "ordinary_clinical_date_remains_compatible",
      "name": "Colonoscopy",
      "context": "Colonoscopy on 2026-01-01.",
      "page": "Colonoscopy on 2026-01-01.",
      "date": "2026-01-01",
      "backend_present_valid": true,
      "worker_assertion": "present"
    }
  ],
  "medication_cases": [
    {
      "id": "other_medication_blocker_cannot_reject_subject",
      "name": "Metformin",
      "context": "Metformin 500 mg oral daily; Lisinopril discontinued.",
      "frequency": "daily",
      "candidate_status": "active",
      "backend_valid": true,
      "worker_status": "active"
    },
    {
      "id": "other_medication_active_cue_cannot_promote_subject",
      "name": "Metformin",
      "context": "Metformin 500 mg oral; Lisinopril active daily.",
      "frequency": null,
      "candidate_status": "active",
      "backend_valid": false,
      "worker_status": "unknown"
    },
    {
      "id": "own_dosing_continuation_supports_subject",
      "name": "Metformin",
      "context": "Metformin 500 mg oral, Take one tablet daily; Lisinopril discontinued.",
      "frequency": "daily",
      "candidate_status": "active",
      "backend_valid": true,
      "worker_status": "active"
    },
    {
      "id": "repeated_subject_fails_closed",
      "name": "Metformin",
      "context": "Metformin active daily; Metformin discontinued.",
      "frequency": null,
      "candidate_status": "active",
      "backend_valid": false,
      "worker_status": "unknown"
    }
  ]
}
```

- [ ] **Step 4: Run the loader tests to verify they pass**

Run: `cd backend && uv run pytest -q tests/test_local_ai_extraction_validation.py::test_strict_local_grounding_case_matrix_is_versioned_and_nonempty`

Expected: PASS.

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py::test_strict_local_grounding_case_matrix_is_versioned_and_nonempty`

Expected: PASS.

### Task 2: Make backend procedure and medication validation subject-local

**Files:**
- Modify: `backend/app/services/local_ai/extraction_validator.py:1178-1208`
- Modify: `backend/app/services/local_ai/extraction_validator.py:1298-1386`
- Modify: `backend/tests/test_local_ai_extraction_validation.py`
- Test: `backend/tests/test_local_ai_extraction_validation.py`

**Interfaces:**
- Produces: `_procedure_requires_explicit_performance(context: str, page_text: str | None) -> bool` returning `True` only for an administrative/billing/authorization context without `_PERFORMED_RE` wording in the procedure's own context.
- Produces: `_lifecycle_subject_source(fact: EvidenceFact, category: str, path: str) -> str`, an unambiguous semantic clause plus only `_is_lifecycle_continuation()` tails; it is the sole status-support, blocker, and uncertainty source. An ambiguous repeated subject returns `""`, so promotion fails closed rather than scanning a broader span.
- Consumes: existing `_BILLED_RE`, `_BILLING_FORM_SIGNATURE_RE`, `_PERFORMED_RE`, `_UNCERTAIN_RE`, `_LIFECYCLE_BLOCKERS`, `_STATUS_SIGNALS`, `_semantic_boundaries`, and `_is_lifecycle_continuation`.
- Preserves: `validate_clinical_extraction(raw, pages, *, upload_id, strict_local=False) -> ClinicalDocumentExtraction` and existing assertion/status enum precedence.

- [ ] **Step 1: Write failing authoritative-validator tests from the matrix**

Add the following parameterized tests after the existing billed-procedure tests and medication lifecycle tests. Build a complete fact, so failure is caused only by the remediation rule.

```python
@pytest.mark.parametrize("case", _grounding_cases()["procedure_cases"], ids=lambda case: str(case["id"]))
def test_administrative_procedure_grounding_matches_shared_matrix(
    case: dict[str, object],
) -> None:
    context = str(case["context"])
    raw = {
        "procedures": [{
            "name": str(case["name"]),
            "assertion": "present",
            "date": str(case["date"]),
            "verbatim": context,
            "page_number": 1,
            "evidence_excerpt": context,
        }]
    }
    if case["backend_present_valid"]:
        assert _validate(raw, page=str(case["page"])).procedures[0].assertion == AssertionState.PRESENT
    else:
        with pytest.raises(LocalValidationError, match=r"procedures\[0\].*performance"):
            _validate(raw, page=str(case["page"]))


@pytest.mark.parametrize("case", _grounding_cases()["medication_cases"], ids=lambda case: str(case["id"]))
def test_medication_lifecycle_scope_matches_shared_matrix(case: dict[str, object]) -> None:
    context = str(case["context"])
    fact = _medication(
        name=str(case["name"]),
        frequency=case["frequency"],
        status=str(case["candidate_status"]),
        verbatim=context,
        evidence_excerpt=context,
    )
    if case["backend_valid"]:
        assert _validate({"medications": [fact]}, page=context).medications[0].status.value == str(case["candidate_status"])
    else:
        with pytest.raises(LocalValidationError, match=r"medications\[0\].status.*lacks source support"):
            _validate({"medications": [fact]}, page=context)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `cd backend && uv run pytest -q tests/test_local_ai_extraction_validation.py -k "administrative_procedure_grounding_matches_shared_matrix or medication_lifecycle_scope_matches_shared_matrix"`

Expected: the validity-date procedure is incorrectly accepted and the cross-subject blocker case is incorrectly rejected.

- [ ] **Step 3: Implement the narrow backend rules**

In `backend/app/services/local_ai/extraction_validator.py`, add these pure helpers immediately before `_validate_assertion_guards`; do not widen `_PERFORMED_RE` or alter ordinary clinical-document behavior.

```python
def _procedure_requires_explicit_performance(context: str, page_text: str | None) -> bool:
    """Return whether administrative procedure text lacks performance wording."""
    administrative = _BILLED_RE.search(context) is not None or (
        _BILLING_FORM_SIGNATURE_RE.search(page_text or "") is not None
    )
    return administrative and _PERFORMED_RE.search(context) is None
```

Replace the present-procedure condition with this exact policy. A date continues to support a present procedure in ordinary clinical text, but never substitutes for performance wording on administrative text.

```python
        if assertion == AssertionState.PRESENT and (
            _procedure_requires_explicit_performance(context, page_text)
            or (
                getattr(fact, "date", None) is None
                and _PERFORMED_RE.search(context) is None
            )
        ):
            _fail(f"{path}.assertion", "performed assertion lacks source performance evidence")
```

Rename `_lifecycle_support_source` to `_lifecycle_subject_source` and keep its existing clause/dosing behavior only for a uniquely located subject. In `_validate_lifecycle_status`, remove `blocker_source = _verbatim_context(fact, path)` and use the one bounded source for every signal:

```python
    source = _lifecycle_subject_source(fact, category, path)
    detected = next((status for status, pattern in signals if pattern.search(source)), None)
    claimed_value = getattr(fact, "status", None)
    claimed = claimed_value.value if hasattr(claimed_value, "value") else claimed_value
    promoted = _PROMOTED_LIFECYCLE_STATES[category]
    blocker = _LIFECYCLE_BLOCKERS.get(category)
    if claimed == promoted and (
        _UNCERTAIN_RE.search(source) is not None
        or (blocker is not None and blocker.search(source) is not None)
    ):
        _fail(f"{path}.status", "lifecycle state contradicts source evidence")
```

Change the old repeated-subject fallback. It must never return `_fact_context()`
because that can reintroduce cross-subject support or blockers. An occurrence
count other than exactly one is ambiguous; return the empty string and let a
promoted state fail with the existing `lifecycle state lacks source support`
error:

```python
    matches = list(re.finditer(pattern, excerpt, re.IGNORECASE))
    if len(matches) != 1:
        return ""
```

Add a matrix case where the selected medication appears twice in one excerpt
and a blocker belongs to the other occurrence. Assert the backend rejects
`active` for lack of unambiguous subject-local support, never because a wider
blocker scan was used.

Update the helper docstring to state that support, blockers, and uncertainty all share this subject-local source. Do not change the order of `_STATUS_SIGNALS` or `_LIFECYCLE_BLOCKERS`.

- [ ] **Step 4: Run backend RED/GREEN regression coverage**

Run: `cd backend && uv run pytest -q tests/test_local_ai_extraction_validation.py -k "administrative_procedure_grounding_matches_shared_matrix or medication_lifecycle_scope_matches_shared_matrix or billed or medication_lifecycle"`

Expected: PASS, including (1) validity-dated authorization rejected, (2) administrative `status post` accepted, (3) ordinary clinical dated procedure accepted, (4) another medication's blocker ignored, and (5) the selected medication's dosing continuation accepted.

Run: `cd backend && uv run ruff check app/services/local_ai/extraction_validator.py tests/test_local_ai_extraction_validation.py`

Expected: `All checks passed!`.

### Task 3: Mirror the clinical grounding rules in the contained worker

**Files:**
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:74-105`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:858-990`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Test: `workers/local_ai/apple_mlx/tests/test_protocol.py`

**Interfaces:**
- Produces: `_procedure_requires_explicit_performance(context: str, page_markdown: str) -> bool`, intentionally mirrored with the backend's pure rule.
- Produces: `_lifecycle_subject_source(fact: Mapping[str, object]) -> str`, which uses an unambiguous medication name, semantic separators, and only safe dosing continuations; it returns `""` for repeated/ambiguous subjects so the worker assigns `unknown`.
- Produces: `EXTRACTION_INSTRUCTIONS: str`, a tested worker-local prompt rule that requires explicit administrative performance wording and says a date/name alone is insufficient.
- Preserves: `_ground_explicit_assertions(value, pages=None) -> dict[str, object]` and `_ground_lifecycle_statuses(value) -> dict[str, object]` mutate only their supplied synthetic worker result.
- Consumes: the Task 1 matrix; no backend module is imported into the isolated worker.

- [ ] **Step 1: Write failing worker tests from the same matrix**

Add tests beside `test_grounding_downgrades_billed_procedure_line_items_to_mentioned_not_performed`. They invoke pure worker transformations, never `generate_content` or a model.

```python
@pytest.mark.parametrize("case", _grounding_cases()["procedure_cases"], ids=lambda case: str(case["id"]))
def test_worker_administrative_procedure_grounding_matches_shared_matrix(
    case: dict[str, object],
) -> None:
    from local_ai_mlx_worker.nuextract3 import _ground_explicit_assertions

    context = str(case["context"])
    value = {"procedures": [{
        "name": str(case["name"]), "assertion": "present", "date": str(case["date"]),
        "verbatim": context, "evidence_excerpt": context, "page_number": 1,
    }]}
    grounded = _ground_explicit_assertions(value, [{"page_number": 1, "markdown": str(case["page"])}])
    assert grounded["procedures"][0]["assertion"] == case["worker_assertion"]  # type: ignore[index]


@pytest.mark.parametrize("case", _grounding_cases()["medication_cases"], ids=lambda case: str(case["id"]))
def test_worker_medication_lifecycle_scope_matches_shared_matrix(
    case: dict[str, object],
) -> None:
    from local_ai_mlx_worker.nuextract3 import _ground_lifecycle_statuses

    context = str(case["context"])
    grounded = _ground_lifecycle_statuses({"medications": [{
        "name": str(case["name"]), "status": str(case["candidate_status"]),
        "verbatim": context, "evidence_excerpt": context,
    }]})
    assert grounded["medications"][0]["status"] == case["worker_status"]  # type: ignore[index]


def test_worker_prompt_requires_explicit_administrative_performance_wording() -> None:
    from local_ai_mlx_worker.nuextract3 import EXTRACTION_INSTRUCTIONS

    lowered = EXTRACTION_INSTRUCTIONS.casefold()
    assert "underwent" in lowered
    assert "performed" in lowered
    assert "status post" in lowered
    assert "date or procedure name alone is insufficient" in lowered
```

- [ ] **Step 2: Run the worker grounding tests to verify they fail**

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py -k "worker_administrative_procedure_grounding_matches_shared_matrix or worker_medication_lifecycle_scope_matches_shared_matrix or worker_prompt_requires_explicit_administrative_performance_wording"`

Expected: the validity date leaves `assertion == "present"`; medication status follows a cue belonging to the other drug; and the prompt constant does not yet exist or still permits an administrative date to stand in for performance wording.

- [ ] **Step 3: Implement mirrored, package-local pure rules**

Keep this code local to `nuextract3.py`; do not import backend code into the network-denied worker. Add worker-local semantic-boundary and continuation helpers matching the backend's existing algorithm, then implement:

```python
def _procedure_requires_explicit_performance(context: str, page_markdown: str) -> bool:
    """Return whether administrative procedure text lacks performance wording."""
    administrative = re.search(rf"\b{_BILLED_MARKER}\b", context, re.IGNORECASE) is not None
    administrative = administrative or _BILLING_FORM_SIGNATURE_RE.search(page_markdown) is not None
    return administrative and _PERFORMED_MARKER_RE.search(context) is None
```

In `_ground_explicit_assertions`, replace the dated administrative exception with the mirror rule:

```python
            elif category == "procedures" and _procedure_requires_explicit_performance(
                context,
                page_markdown_by_number.get(int(fact.get("page_number", 0)), ""),
            ):
                expected = "mentioned_not_performed"
```

Use a safe conversion helper rather than calling `int()` inline if the page number is absent or malformed. Preserve the existing explicit negation, planned/not-done, uncertain, and family-history precedence before this branch.

Move the `run_extraction()` instruction text into this module constant and replace
the stale administrative sentence. The prompt must reinforce the same rule as
the post-processing guard; it may not say that a date establishes performance:

```python
EXTRACTION_INSTRUCTIONS = (
    "Extract only values present in the document. Use JSON null only for optional "
    "text fields without evidence, and use empty lists for absent categories. Never "
    "use JSON null for enum-valued fields. Use unknown for status enums, uncertain "
    "for assertion enums, and other for an unestablished category or visit type. "
    "Set assertion to negated for explicit no, denies, absent, or negative evidence. "
    "Set assertion to family_history only for explicit family-history context. Set "
    "procedure assertion to mentioned_not_performed for planned, cancelled, deferred, "
    "or not-done procedures, and for procedures listed only as billing, claim, or "
    "authorization line items unless the source uses explicit performance wording "
    "such as underwent, performed, or status post. A date or procedure name alone "
    "is insufficient in administrative context. Use uncertain for explicit possible, "
    "suspected, or unclear evidence; otherwise use present only when the source "
    "affirms the fact. Preserve verbatim clinical values."
)
```

Set `instructions = EXTRACTION_INSTRUCTIONS` in `run_extraction()`. Keep all
other existing prompt restrictions verbatim; only replace the old phrase
`without a statement or date that the procedure occurred`.

Implement `_lifecycle_subject_source(fact)` by accepting `evidence_excerpt` only when it contains exactly one word-bounded medication name. Do not fall back to a broader `verbatim` span when the evidence is ambiguous; return `""` and let `_ground_lifecycle_statuses` assign `unknown`. Split the accepted source on the worker mirror of `(?<!\\d)[.,](?!\\d)|[;|•●▪◦]|\\b(?:and|but|while|whereas)\\b`; return the subject segment plus only tails whose nonnumeric words are in the backend-compatible dosing continuation allowlist. In `_ground_lifecycle_statuses`, replace the whole-excerpt `context` search with:

```python
            source = _lifecycle_subject_source(fact)
            fact["status"] = next(
                (status for status, pattern in signals if pattern.search(source)),
                "unknown",
            )
```

Add the repeated-subject matrix row to the worker parameterization and assert
that it becomes `unknown`. This is intentionally conservative: worker output
must not use a wider source than the authoritative backend can validate.

- [ ] **Step 4: Run worker clinical verification**

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py -k "grounding or worker_administrative_procedure_grounding_matches_shared_matrix or worker_medication_lifecycle_scope_matches_shared_matrix or worker_prompt_requires_explicit_administrative_performance_wording"`

Expected: PASS; no model artifact or provider is loaded because these tests use only pure transforms and stub data.

Run: `cd workers/local_ai/apple_mlx && uv run ruff check src/local_ai_mlx_worker/nuextract3.py tests/test_protocol.py`

Expected: `All checks passed!`.

### Task 4: Reserve attempt capacity during pre-fit and runtime splitting

**Files:**
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:35-42`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:268-329`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:748-855`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py:1388-1481`
- Modify: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Test: `workers/local_ai/apple_mlx/tests/test_protocol.py`

**Interfaces:**
- Changes constant: `MAX_EXTRACTION_RUNTIME_SPLITS = 11`.
- Produces: `_ExtractionWorkBudget.require_prefit_capacity(batch_count: int) -> None`, raising `GenerationError(category="work_attempt_limit")` when `batch_count > 12` before inference.
- Produces: `_ExtractionWorkBudget.reserve_prefit_split() -> None`, which maps a twelfth deterministic pre-fit split to `work_attempt_limit` rather than exposing the stale `work_split_limit`; eleven splits can produce at most twelve isolated fragment batches.
- Produces: `_ExtractionWorkBudget.require_runtime_split_capacity(queued_batches: int) -> None`, raising `GenerationError(category="work_attempt_limit")` unless `attempts + queued_batches + 2 <= 12`.
- Produces: `_split_runtime_batch(pages, *, budget) -> tuple[list[dict[str, object]], list[dict[str, object]]]`, which spends one split and returns two ordered children without generating.
- Preserves: `_run_extraction_batch(pages, *, images_by_page, selected, template, instructions, max_tokens, generate_fn, attempt_progress_fn, budget) -> dict[str, object]`, syntax-only retry semantics, exact token charging, `work_token_limit`, `fragment_depth_limit`, content-free failure categories, ordered result consolidation, and public `run_extraction(payload, *, loaded=None, generate_fn=generate_content, progress_fn=None, attempt_progress_fn=None, lifecycle_progress_fn=None, budget_progress_fn=None) -> dict[str, object]`.

- [ ] **Step 1: Write failing bounded-work tests**

Add both tests beside the existing extraction-work-budget tests. They patch only local functions and pass a stub generator, so no inference occurs.

```python
def test_prefit_batches_over_attempt_limit_fail_before_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from local_ai_mlx_worker import nuextract3
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import run_extraction

    pages = [{"page_number": index, "markdown": "bounded OCR"} for index in range(1, 14)]
    monkeypatch.setattr(nuextract3, "_extraction_batches", lambda *_args, **_kwargs: [[page] for page in pages])
    calls = 0

    def generate(**_kwargs: object) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("must not generate")

    with pytest.raises(GenerationError, match="attempt work limit") as error:
        run_extraction(
            {"page_markdown": pages, "scratch_dir": str(tmp_path), "image_paths": {},
             "schema": {"schema_version": "clinical-document-extraction.v1"}},
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=generate,
        )
    assert calls == 0
    assert error.value.category == "work_attempt_limit"


def test_runtime_splits_never_make_a_thirteenth_generation_call(tmp_path: Path) -> None:
    from local_ai_mlx_worker.common import GenerationError
    from local_ai_mlx_worker.nuextract3 import MAX_EXTRACTION_GENERATION_ATTEMPTS, run_extraction

    calls: list[int] = []
    def invalid_json(**_kwargs: object) -> str:
        calls.append(1)
        return "{"

    with pytest.raises(GenerationError) as error:
        run_extraction(
            {"page_markdown": [{"page_number": 1, "markdown": "bounded clinical text " * 128}],
             "scratch_dir": str(tmp_path), "image_paths": {},
             "schema": {"schema_version": "clinical-document-extraction.v1"}},
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            generate_fn=invalid_json,
    )
    assert error.value.category == "work_attempt_limit"
    assert len(calls) <= MAX_EXTRACTION_GENERATION_ATTEMPTS


def test_prefit_fragment_overflow_uses_work_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_ai_mlx_worker import nuextract3
    from local_ai_mlx_worker.common import GenerationError

    monkeypatch.setattr(nuextract3, "_fits_extraction_batch", lambda *_args, **_kwargs: False)
    budget = nuextract3._ExtractionWorkBudget(
        runtime_splits=nuextract3.MAX_EXTRACTION_RUNTIME_SPLITS
    )
    with pytest.raises(GenerationError) as error:
        nuextract3._fit_page_fragments(
            {"page_number": 1, "markdown": "bounded text " * 2048},
            loaded=_loaded("extraction"),  # type: ignore[arg-type]
            template="{}",
            instructions="bounded",
            max_tokens=128,
            budget=budget,
        )
    assert error.value.category == "work_attempt_limit"
```

- [ ] **Step 2: Run the bounded-work tests to verify they fail**

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py -k "prefit_batches_over_attempt_limit_fail_before_generation or prefit_fragment_overflow_uses_work_attempt_limit or runtime_splits_never_make_a_thirteenth_generation_call"`

Expected: the pre-fit test attempts a generation; the fragment-overflow test reports the stale `work_split_limit` rather than `work_attempt_limit`; and the split test either reaches the stale split/depth path or records more than 12 calls.

- [ ] **Step 3: Add explicit admission checks and replace recursive runtime execution with an ordered queue**

Set the static split ceiling to 11:

```python
MAX_EXTRACTION_RUNTIME_SPLITS = 11
```

Add these two methods to `_ExtractionWorkBudget` after `reserve_attempt`. They explicitly model the minimum remaining one call per queued batch and one call per newly split child; syntax retries remain enforced by `reserve_attempt()`.

```python
    def require_prefit_capacity(self, batch_count: int) -> None:
        """Reject deterministic work that needs more first calls than remain."""
        if batch_count > MAX_EXTRACTION_GENERATION_ATTEMPTS - self.attempts:
            raise GenerationError(
                "Local extraction exceeded its attempt work limit.",
                category="work_attempt_limit",
            )

    def require_runtime_split_capacity(self, queued_batches: int) -> None:
        """Reserve one minimum generation for queued work and both split children."""
        if self.attempts + queued_batches + 2 > MAX_EXTRACTION_GENERATION_ATTEMPTS:
            raise GenerationError(
                "Local extraction exceeded its attempt work limit.",
                category="work_attempt_limit",
            )

    def reserve_prefit_split(self) -> None:
        """Spend one deterministic split without masking work-cap overflow."""
        if self.runtime_splits >= MAX_EXTRACTION_RUNTIME_SPLITS:
            raise GenerationError(
                "Local extraction exceeded its attempt work limit.",
                category="work_attempt_limit",
            )
        self.reserve_runtime_split()
```

Import `deque` with `from collections import deque`. In `_fit_page_fragments`,
call `budget.reserve_prefit_split()` before every deterministic `_split_page`
call, and add a `reserve_split=False` argument to `_split_page()` so that
pre-fit does not reserve twice. This establishes `work_attempt_limit` as the
public category whenever a twelfth pre-fit split would create a thirteenth
isolated fragment batch. Immediately after assigning the result of
`_extraction_batches` to `batches`, call
`budget.require_prefit_capacity(len(batches))`. This call must run before any
progress callback or `_run_extraction_batch` call, so thirteen original
image-isolated batches also fail with `work_attempt_limit` and zero generation
calls.

Use this exact reservation switch in `_split_page()` and its pre-fit caller:

```python
def _split_page(
    page: dict[str, object],
    *,
    budget: _ExtractionWorkBudget | None = None,
    reserve_split: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    # Keep the existing markdown/depth validation unchanged.
    if budget is not None and reserve_split:
        budget.reserve_runtime_split()
    # Keep the existing left/right fragment construction unchanged.


# Inside _fit_page_fragments(), after a candidate does not fit:
budget.reserve_prefit_split()
left, right = _split_page(candidate, budget=budget, reserve_split=False)
pending[0:0] = [left, right]
```

Replace `_run_extraction_batch_with_runtime_splits` recursion with a non-generating `_split_runtime_batch()` plus an ordered `collections.deque` loop in `run_extraction`. On a split-eligible failure, call `budget.require_runtime_split_capacity(len(pending_batches))` **before** `_split_runtime_batch`; put left then right at the front of the deque so output ordering remains unchanged. The loop's success path appends `(pages, result)` to `resolved_batches`; non-split failures immediately re-raise their existing bounded category. The only generation call remains `_run_extraction_batch`, so every call still reserves and charges the existing budget.

Implement the required split helper completely; it performs no generation and
reserves exactly one runtime split for either shape:

```python
def _split_runtime_batch(
    pages: list[dict[str, object]], *, budget: _ExtractionWorkBudget
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split one failed batch into ordered children after capacity admission."""
    if len(pages) > 1:
        budget.reserve_runtime_split()
        midpoint = len(pages) // 2
        return pages[:midpoint], pages[midpoint:]
    left, right = _split_page(pages[0], budget=budget)
    return [left], [right]
```

Use this complete queue loop, retaining the current exception categorization:

```python
    pending_batches: deque[list[dict[str, object]]] = deque(batches)
    resolved_batches: list[tuple[list[dict[str, object]], dict[str, object]]] = []
    while pending_batches:
        batch = pending_batches.popleft()
        failure: GenerationError | None = None
        should_split = False
        try:
            result = _run_extraction_batch(
                batch,
                images_by_page=images_by_page,
                selected=selected,
                template=template,
                instructions=instructions,
                max_tokens=max_tokens,
                generate_fn=generate_fn,
                attempt_progress_fn=attempt_progress_fn,
                budget=budget,
            )
        except _TruncatedGenerationError as exc:
            failure = exc
            should_split = True
        except _FormattedInputLimitError:
            should_split = True
        except GenerationError as exc:
            failure = exc
            should_split = exc.category == "invalid_structured_output"
        if should_split:
            budget.require_runtime_split_capacity(len(pending_batches))
            left, right = _split_runtime_batch(batch, budget=budget)
            pending_batches.extendleft((right, left))
        elif failure is not None:
            raise failure
        else:
            if all(
                not _is_fragment(page) or page.get(_FRAGMENT_FINAL_KEY) is True
                for page in batch
            ):
                publish_batch_progress(len(batch))
            resolved_batches.append((batch, result))
```

Do not literally add `_SplitEligibleExtractionError` unless the existing exception taxonomy is refactored to define it. A small local predicate/function that maps the current `_TruncatedGenerationError`, `_FormattedInputLimitError`, and `GenerationError(category="invalid_structured_output")` branches to `split=True` is preferred; retain the exact current categories for all failures exposed to the caller.

- [ ] **Step 4: Run bounded-work RED/GREEN and existing fragmentation regressions**

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py -k "prefit_batches_over_attempt_limit_fail_before_generation or prefit_fragment_overflow_uses_work_attempt_limit or runtime_splits_never_make_a_thirteenth_generation_call or extraction_recursively_splits or extraction_work_budget_enforces_exact_independent_limits or fragment_depth_limit"`

Expected: PASS. Both non-fragment and fragment-heavy pre-fit overflow cases raise content-free `work_attempt_limit` with zero generation calls; every runtime-split path records at most 12 calls; the token ceiling and depth-5 shape guard remain covered.

Run: `cd workers/local_ai/apple_mlx && uv run ruff check src/local_ai_mlx_worker/nuextract3.py tests/test_protocol.py`

Expected: `All checks passed!`.

### Task 5: Run Track B verification and prepare the dependency checkpoint

**Files:**
- Verify only: all files in Tasks 1-4
- Proposed root-only commit if Pedro authorizes it: shared fixture, backend validator/tests, MLX
  worker/tests

**Interfaces:**
- Verifies: `validate_clinical_extraction(raw, pages, *, upload_id, strict_local=False)` rejects the administrative date-as-performance regression and remains authoritative.
- Verifies: `run_extraction(payload, *, loaded=None, generate_fn=generate_content, progress_fn=None, attempt_progress_fn=None, lifecycle_progress_fn=None, budget_progress_fn=None)` either returns ordered grounded chunks or fails with a content-free bounded error before a 13th call.
- Produces: one reviewed Track B diff and, only if explicitly authorized, a commit that Track D may
  use as the immutable worker-source baseline for its new runtime identity and release receipt.

- [ ] **Step 1: Run focused backend and worker suites without external calls**

Run: `cd backend && uv run pytest -q tests/test_local_ai_extraction_validation.py`

Expected: PASS. This uses only deterministic fixtures and validator code; no provider or model call occurs.

Run: `cd workers/local_ai/apple_mlx && uv run pytest -q tests/test_protocol.py`

Expected: PASS. Tests use stubs/fakes unless separately marked `local_model`; do not add `--run-local-model` or invoke any pack-management command.

- [ ] **Step 2: Run static checks and the ordinary backend regression gate**

Run: `cd backend && uv run ruff check app/services/local_ai/extraction_validator.py tests/test_local_ai_extraction_validation.py && uv run pytest -m "not slow and not fidelity and not local_model and not hardware" -q`

Expected: Ruff reports `All checks passed!`; the ordinary deterministic backend suite passes with its normal skips/deselections. Do not pass flags that enable provider, private-fixture, model-download, or live-cloud work.

Run: `cd workers/local_ai/apple_mlx && uv run ruff check src/local_ai_mlx_worker/nuextract3.py tests/test_protocol.py`

Expected: `All checks passed!`.

- [ ] **Step 3: Root-agent review and prepare the proposed dependency checkpoint**

Subagents stop after reporting test output and must not run `git commit`. The root agent inspects
the exact diff and confirms that only Track B files changed. The `git add` and `git commit` commands
below are run only after Pedro explicitly authorizes them:

```bash
git status --short
git diff --check
git diff -- backend/app/services/local_ai/extraction_validator.py \
  backend/tests/test_local_ai_extraction_validation.py \
  workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py \
  workers/local_ai/apple_mlx/tests/test_protocol.py \
  workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json
git add backend/app/services/local_ai/extraction_validator.py \
  backend/tests/test_local_ai_extraction_validation.py \
  workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py \
  workers/local_ai/apple_mlx/tests/test_protocol.py \
  workers/local_ai/apple_mlx/tests/fixtures/strict_local_extraction_grounding_cases.json
git commit -m "fix(local-ai): bound strict extraction grounding and work"
```

Expected before authorization: `git diff --check` is silent and the worktree contains only Track B.
If a commit is authorized, it contains only Track B. Do not merge, push, attest a worker runtime,
regenerate a model pack, or start Track D in this task. After the root commit is accepted, hand its
commit SHA to Track D as the required final-worker-source baseline.

## Plan self-review

- [x] Track B administrative procedure grounding maps to Task 2 (backend) and Task 3 (worker), including the date-only authorization, explicit performance, and ordinary-clinical cases.
- [x] Subject-scoped medication lifecycle maps to Tasks 2 and 3, including both cross-subject leakage directions and a valid dosing continuation.
- [x] Attempt-aware pre-fit, runtime queue reservation, static 11-split ceiling, unchanged depth-5/token rules, and no-thirteenth-call proof map to Task 4.
- [x] Both packages load the one deterministic synthetic matrix created in Task 1; no provider/private/model call is introduced.
- [x] Track A/C isolation, no-subagent-commit ownership, Track D dependency, focused checks, full
  deterministic checks, and the separately authorized root-only commit are explicit in Tasks 1-5.

Plan complete and saved to `docs/superpowers/plans/2026-08-13-strict-local-extraction-remediation.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, and iterate quickly.
2. Inline Execution - execute task-by-task in one session with checkpoints.
