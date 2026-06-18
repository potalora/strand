# MedTimeline — Session Summary & Open Issues (2026-06-17)

A compaction of this session's work plus the open issues found during the e2e pass.
Detailed companions: [`docs/e2e-gap-analysis.md`](./e2e-gap-analysis.md) and
[`docs/extraction-remediation.md`](./extraction-remediation.md).

---

## Part 1 — Shipped this session ✅ (PR #31, merged to `main`)

An e2e pass surfaced 7 issues; all were fixed with TDD, the Playwright suite was
hardened to catch this class going forward, and the lot was merged.

**The 7 fixes**
1. **Pagination instability** — `/records` `ORDER BY` now has a stable `id` tiebreaker (was duplicating/dropping rows across pages in large tie groups).
2. **Stuck-PDF "hang"** — `parse_datetime` rejects out-of-range tz offsets (date-range strings made a Postgres-invalid `timestamptz`); `_process_unstructured` now `rollback()`s the poisoned session before marking `failed` (no more stuck `processing` + misleading TimeoutError).
3. **Frozen per-row extraction labels** — upload page now polls per-file status (was frozen at trigger-time).
4. **Blank account name** — shared `useUserStore` + `/auth/me` 401→refresh; name no longer blanks on a transient 401; the single-flight refresh won't resurrect a logged-out session.
5. **Duplicates + Merges → one "Deduplication" tab** (sub-tabs: Pending review / Merge ledger).
6. **Settings folded into System**; bell/notifications button removed; `/settings` → `/admin?tab=sys`.
7. **Extraction date attribution** — recover effective dates from broadened keys / text / document fallback (was ~79% dateless).

**E2E hardening** — global console-error/React-key gate, large-dataset pagination-integrity spec, extraction terminal-state spec, polling-transition + logout/refresh-race specs; ~20 stale specs repaired to the new IA; `docs/e2e-gap-analysis.md` explains *why* the old suite missed all 7 (tiny datasets, no console gate, steady-state-only assertions, never-expiring auth, mocked extraction).

**Verification** — backend **745 passed**; frontend `tsc` clean; full Playwright suite **129 passed (workers=1)**. Note: the suite is flaky at `workers=3` on this machine (backend contention — every test passes in isolation); `workers=1` is the deterministic gate. Login rate-limit default is 30/60s (relaxed for e2e). `CLAUDE.md` is gitignored (its updates are local-only).

**Environment** — dev DB (`medtimeline`) wiped clean + upload storage cleared; backend (normal config) + frontend running fresh.

---

## Part 2 — Open issues (found during the continued e2e pass; **not started**)

### 2a. Upload / extraction UX — `upload/page.tsx` state lifecycle
- **(i)** The "N unstructured files detected" panel never reaches a terminal/"all done" state and can't be dismissed once everything is `completed`.
- **(ii)** A new upload doesn't reset the prior panel/rows — stale files from the previous batch persist; a 1-file upload still shows the carried-over set.
- **(iii)** The "Extracting clinical entities" progress is **user-global** (`/upload/extraction-progress` counts all unstructured files), so a 1-file upload reads "84 of 85". Should be scoped to the current upload.
- **(iv)** *(feature)* **Global hovering status bar** when anything is processing + **detailed progress for long LLM extracts** (stage/section-level) + **cancel**. Cancel needs a real backend abort (the DB-polling worker has none) and section-level progress needs new backend instrumentation (only file-level status today).

### 2b. Display / labeling
- **Labs show a generic "Observation" badge.** Not a data bug — labs *are* FHIR `Observation`/`laboratory`. The badge should show the sub-type (**Lab / Vital / Social**) using the category already on each record (icons already differentiate). Lives in `RetroBadge` + Timeline/Records/RecordDetail.
- **Providers aren't surfaced clearly.** Where present (structured encounters ~94%), the UI doesn't render `participant`/`performer`. See remediation C1.

### 2c. Extraction quality — richness & precision → **[`docs/extraction-remediation.md`](./extraction-remediation.md)**
Audited the full test-data load (2,153 records / 16 types). Two problem classes:

**Precision (LangExtract over-extracts false records)** — confirmed: `2mg` and `5'9"` as observations, `COLONOSCOPY` as a procedure (never performed — mentioned in the doc), `"Go"`/`PPI`/`LDN`/`b12` as medications, `Exercise:`/`Diet:`/`Alcohol: avoid…` as observations, `[NAME]` (PHI placeholder) as a procedure, `Cystectomy` ×9 (within-doc dup). Fixes: performed-evidence guard for procedures, reject fragment/recommendation/garbage entities, drop scrubber-placeholder entities, collapse near-dups.

**Recall (richness dropped, esp. on the unstructured path)** — **36% of records carry a code; AI-extracted records are 0% coded** (no SNOMED/LOINC/RxNorm/ICD); even structured meds are 0% RxNorm. AI records also drop providers (0%), lab units (0%), reference ranges, and structured dosage. Fixes: terminology coding step in `entity_to_fhir`, attach provider/performer, parse value+unit+range, structure dosage, populate facility.

**Priority (from the remediation doc):** P0 = over-extraction guards (A1–A5) + coding (B1); P1 = providers/units/dosage richness (B2–B4) + provider UI (C1); P2 = dup-collapse, onset, facility, panel-orders.

---

## Suggested next step
Implement in two phases: **(1) extraction precision + coding** (entity_extractor prompt/few-shot + entity_to_fhir + a terminology map), then **(2) the upload/extraction UX + provider/sub-type display**. Both decompose cleanly into a parallel agent team.
