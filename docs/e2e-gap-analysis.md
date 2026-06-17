# E2E Gap Analysis — why ~115 Playwright tests caught none of the 7 bugs

This document explains, per bug, the concrete reason the existing end-to-end suite
could not have caught it, then rolls the reasons up into systemic categories and
lists the hardening added in response.

The evidence is drawn from the specs as they stood before this pass
(`frontend/e2e/*.spec.ts`) and the helpers in `frontend/e2e/helpers/`.

---

## Per-bug root cause

### #1 — Pagination instability (missing `id` tiebreaker in `ORDER BY`)

**Bug:** `GET /records` ordered only by `effective_date` (+ type), with no stable
final tiebreaker. With >100 records (the page-size cap) and tie groups straddling
a page boundary, Postgres could return a row on two pages or drop one between
pages. Admin → Records pages through the whole set at `page_size=100` and
accumulates, so a duplicated row produced two React children with the same `key`,
emitting a `console.error` ("Encountered two children with the same key").

**Why no test caught it:**
- **Dataset too small.** Every data-seeding path uploads exactly one fixture
  bundle — `helpers/seed.ts` and every `beforeAll` call
  `api.uploadStructured(PATHS.fhirBundle, …)` once. `sample_fhir_bundle.json`
  yields well under 100 records, so the second page (and therefore the
  cross-page tie boundary) never existed in any test. The bug is *structurally*
  unreachable below the page-size cap.
- **No console-error gate.** The duplicate-key symptom is a React `console.error`
  in dev, not a thrown exception. The only spec watching the console
  (`error-handling.spec.ts`) listened to `page.on("pageerror")` (uncaught
  exceptions) — which never fires for a React key warning — filtered out
  `ResizeObserver`, and ran only on `/` and `/timeline`. Admin → Records was
  never in scope, and even if it had been, `pageerror` ≠ `console.error`.

**Category:** dataset-scale gap **+** observability gap.

---

### #2 — Stuck-PDF hang (out-of-range tz offset + un-rolled-back session)

**Bug:** A date-*range* string parsed to a `-24:00` tz offset that Postgres
rejects; the `_process_unstructured` except handler then tried to mark the file
`failed` on the *poisoned* session without rolling back, so the write failed, the
file stayed `processing`, was retried 3×, and got mislabeled `TimeoutError`.

**Why no test caught it:**
- **Clean, happy-path inputs.** The unstructured specs feed real RTF/PDF files
  that extract cleanly (`upload-unstructured.spec.ts`, `upload-progress.spec.ts`),
  or run purely at the API level. None feeds a document whose extracted text
  produces a date-*range* token, so the poisoned-session path is never reached.
- **No terminal-state assertion.** The polling helper
  (`api-client.ts::pollUploadStatus`) waits until *any* terminal status or throws
  on timeout. A test that times out simply fails with "did not complete"; no spec
  asserts the invariant that a *failing* document must reach `failed` (and never
  sit in `processing` forever). "Stuck in processing" and "slow" are
  indistinguishable to a timeout-based poll.

**Category:** happy-path/clean inputs **+** missing terminal-state invariant.

---

### #3 — Frozen per-row extraction labels

**Bug:** The upload page's poll loop refreshed the aggregate progress bar
(`/upload/extraction-progress`) but not each file's row, so a row in the
"N unstructured files detected" table stayed frozen at its trigger-time status
(e.g. `processing`) forever while the bar moved.

**Why no test caught it:**
- **Steady-state-only assertions.** The upload specs assert the *final* outcome —
  `"records inserted"`, or an API-level terminal status via `pollUploadStatus`.
  The frozen label is a purely client-side derived value
  (`extractionStatuses[id] || file.status`) that the API-level helpers never
  read. No spec observed the *row* advancing across successive poll ticks, so a
  row that never advanced looked identical to one the test never looked at.

**Category:** steady-state-only assertions / no live-transition coverage.
**Now covered** by `upload-row-status-polling.spec.ts`.

---

### #4 — Blank account name (transient 401 on `/auth/me` swallowed)

**Bug:** A transient 401 on `/auth/me` (expired access token) was swallowed, so
the account name rendered blank instead of refreshing-and-retrying.

**Why no test caught it:**
- **Auth tokens are always fresh.** Every spec logs in immediately before
  asserting (`browserLogin`), so the 15-minute access token is always valid and
  `/auth/me` never 401s mid-test — the refresh race simply cannot occur.
- **The one expiry spec asserted the wrong surface.** `auth-refresh.spec.ts`
  corrupts the access token to force a 401, but only asserts that *dashboard data*
  renders after refresh. It never asserts that the **account name** specifically
  survives a transient 401 on `/auth/me` — the exact field that blanked.

**Category:** always-fresh auth / refresh-race never simulated on the affected
surface. **Now covered** by `admin-consolidation.spec.ts` (e) (`meSequence: [401, 200]`).

---

### #5 — Duplicates + Merges collapsed into one "Deduplication" tab

This is a deliberate IA refactor, not a latent bug, so "why not caught" is N/A.
Its *testing* consequence is the relevant gap:
- **Stale IA assertions + vacuous passes.** `admin-dedup.spec.ts` clicked
  `getByRole("button", { name: "Dedup" })`, a label that does not exist in the
  current admin (tabs use `role="tab"`, label "Deduplication"). Worse, the spec's
  body returned early ("No candidates available — all were auto-resolved")
  whenever it found nothing, so it could pass without ever exercising the pane.
  An IA-level regression here would not have failed loudly.

**Category:** stale IA assertions / vacuous-pass tolerance.
**Now covered** by `admin-consolidation.spec.ts` (b)(c).

---

### #6 — Settings folded into Admin → System; nav bell removed; `/settings` redirect

Also a deliberate IA refactor. Same testing consequence as #5: the pre-existing
`admin-system.spec.ts` and `navigation.spec.ts` asserted the *old* IA (a
`role="button"` System tab, nav links "Home"/"Summarize"/"Upload", a
`Toggle theme` aria-label, a notifications bell). They would only ever fail
*stale*, never catch the intended new behavior.

**Category:** stale IA assertions.
**Now covered** by `admin-consolidation.spec.ts` (a)(d)(f).

---

### #7 — Extraction date attribution (79% of extracted records were dateless)

**Bug:** `entity_to_fhir` failed to recover effective dates from most extracted
entities, so ~79% of AI-extracted records had no `effective_date` and never
appeared on the date-based timeline.

**Why no test caught it:**
- **Data quality is unmeasured.** The extraction specs assert a terminal status
  and total record counts — never the *quality* property "what fraction of
  extracted records carry a date." A run where every record is dateless passes a
  status/count assertion just as cleanly as a correct one.
- **Fidelity tests skip without fixtures.** The real-data assertions live behind
  `@pytest.mark.fidelity` / `hasTestData(...)` guards that skip when the
  gitignored fixtures are absent, so in any environment without the private
  exports there is no quality guard at all.

**Category:** data-quality unguarded **+** fidelity tests skipped without fixtures.
Backend integration tests now cover this directly
(`tests/test_extraction_date_attribution.py`); e2e leans on them.

---

## Systemic categories (the "why", rolled up)

| # | Category | Bugs |
|---|----------|------|
| A | **Dataset scale** — synthetic fixtures never exceed a few dozen records, so page-boundary / tie-group / ordering bugs are structurally unreachable. | #1 |
| B | **Happy-path / clean inputs** — extraction is fed files that parse cleanly; malformed-date and dateless real-world inputs are never exercised. | #2, #7 |
| C | **Steady-state-only assertions** — tests wait for the final state; live transitions and polling propagation are never observed. | #2, #3 |
| D | **Always-fresh auth** — login immediately precedes every assertion, so token-expiry / refresh races never occur on the real surfaces. | #4 |
| E | **No observability gate** — React warnings and `console.error` (the visible symptom of #1) print to the console but fail nothing; the lone listener watched only `pageerror`, filtered noise, and on two pages. | #1 |
| F | **Stale IA assertions / vacuous passes** — specs assert an obsolete IA and/or return early on "nothing found", so they pass without exercising the behavior. | #5, #6 |
| G | **Data-quality unguarded** — quality/fidelity assertions are gated on gitignored fixtures and skip in their absence. | #7 |

---

## Hardening added in response

- **Global console-error gate** (category E) — `frontend/e2e/fixtures/console-gate.ts`
  extends Playwright's `test` so ANY `console.error`, React key/duplicate-key
  warning, or uncaught `pageerror` fails the test, with a small documented
  allowlist (HTTP-status "Failed to load resource" noise, ResizeObserver loop
  notices) for genuinely benign output. A focused proof test
  (`console-gate.spec.ts`) asserts the gate both passes on a clean page and trips
  on an injected `console.error` and a React duplicate-key warning.
- **Large-dataset pagination integrity** (category A, bug #1) —
  `pagination-integrity.spec.ts` seeds >100 records with many duplicate AND null
  `effective_date`s, opens Admin → Records, and asserts zero duplicate row keys,
  zero console errors, and visible-rows == reported-total. Fails on the pre-fix
  ordering, passes now.
- **Extraction terminal-state** (categories B/C, bugs #2/#3) —
  `extraction-terminal-state.spec.ts` drives the Admin → Extractions and Upload
  UI with mocked status endpoints to assert a failing document surfaces `failed`
  (terminal) and `processing` is never treated as terminal, plus a document that
  completes reaches a completed state.
- **Confirmed/retained** the #3 polling spec (`upload-row-status-polling.spec.ts`)
  and the #4 transient-401 name spec (`admin-consolidation.spec.ts` (e)).
- **Repaired stale IA specs** (category F) to assert the *new* consolidated IA
  without weakening coverage.
