# Strict-Local E2E Runtime-Identity Harness Repair Design

**Date:** 2026-08-23

**Status:** Revised design; implementation plan awaiting root approval

**Baseline:** `origin/main` at
`32075826694e1a24bfb36699fb14374b7916b29a`

**Scope:** Local-only Playwright harness and test/documentation coverage only.
Production strict-local runtime identity, admission, manifests, releases, and
processing behavior do not change.

## Problem

A fresh single-worker run of the complete `E2E_LOCAL_ONLY` Playwright profile
against an isolated migrated database reproduced this result on the baseline
above:

- 206 tests enumerated
- 127 passed
- 15 failed
- 10 explicitly skipped
- 54 did not run after setup failures

All 15 failures belong to populated-record setup or upload. Thirteen fail
directly with the bounded response
`409 {"detail":"Strict-local worker runtime identity is required."}`. The other
two are downstream symptoms: no record types exist after failed setup, and the
upload page waits for a success message while showing the same bounded 409.
Authentication, authorization, zero-state, and mocked-network cases pass.

The failed run left zero `patients`, `health_records`, `uploaded_files`, and
`local_ai_jobs` in its disposable database. The 15 retained traces and JSON
results are diagnostic evidence only. They remain ignored and must not be
committed.

## Root cause

This is a stale harness contract and a coverage gap, not a production admission
bug.

The local-only profile still selects the legacy v1 manifest and release paths
and presents an argument-bearing Python fake-worker command. Its shared API
client stores `validated_strict_local` at login and explicitly submits that mode
for each structured fixture upload.

Current admission correctly treats legacy-v1 jobs as
`runtime_identity_required`. Strict ingestion loads the legacy manifest only far
enough to diagnose the missing worker bundle identity, then fails before file
storage, ingestion, artifact validation, release-evidence loading, or worker
execution. A current identity can be derived only from the one-token installed
worker entry point and its real project, lockfile, and effective package tree.
The argument-bearing fake wrapper cannot truthfully satisfy that contract.

Changing the profile to name the v2 manifest would not create a valid strict
runtime. Admission would still require a matching active manifest, an immutable
validation receipt, no nonterminal pack operation, a matching runtime identity,
and release evidence backed by the required benchmark and model-fidelity gates.
No v2 release manifest exists in this checkout, by design. Tests must not invent
one.

The harness conflates two independent claims:

1. browser and application processes cannot make non-loopback network calls;
2. a release-ready, runtime-attested strict-local model pack is installed.

`E2E_LOCAL_ONLY` proves the first claim on an ordinary test machine. It does not
prove the second.

## Decision

Repair the harness by separating network confinement, deterministic structured
fixture setup, and real model execution.

- Keep `validated_strict_local` as the local-only account's default preference.
- Add a fixed tracked-FHIR helper that always submits
  `processing_mode=cloud_assisted`. It accepts no caller path or filename.
- Add a separate content-based helper for the generated pagination FHIR bundle.
  It accepts JSON content, not a path, and uses a fixed filename.
- Keep the generic path-based structured helper only for private CDA callers.
  Those callers skip because the profile forces the private-fixture root empty.
- Add exact negative strict-local upload and summary admission regressions.
- Keep all OS, socket, and browser network-denial guards active.
- Gate the three summary tests that execute a model on a real attested worker and
  pack. The current legacy-v1 profile does not qualify, so those cases skip with
  a pack-required reason. The seven non-execution summary UI cases continue to
  run.

`cloud_assisted` is the correct explicit fixture mode because ingestion admits
`cloud_assisted` or `validated_strict_local`. `prompt_only` is a summary mode,
not an ingestion mode. Selecting `cloud_assisted` does not authorize a provider
call. The allowed inputs use deterministic FHIR/CDA parsers; backend tests run
the complete downstream dedup flow and fail if provider construction occurs;
provider credentials are empty; and local-only socket guards deny non-loopback
connections.

This is a narrow fixture override. Unstructured uploads, strict admission tests,
normal user preferences, and production requests retain their existing mode
semantics.

## Data flow

### Deterministic structured-fixture positive path

1. A test creates and logs into a fresh synthetic account.
2. Local-only login persists `validated_strict_local` as the account default.
3. A fixed helper reads only
   `backend/tests/fixtures/sample_fhir_bundle.json`, or the pagination test sends
   generated synthetic FHIR JSON through the content helper.
4. The helper adds `processing_mode=cloud_assisted` to that single request.
5. Admission captures the explicit cloud-assisted mode without constructing a
   strict runtime snapshot.
6. The coordinator dispatches to the deterministic structured parser, commits
   the upload, and schedules the real background dedup pass.
7. Backend regression coverage drains that background pass, proves the upload
   reaches a terminal status, and asserts that
   `app.services.dedup.llm_judge.get_provider` was called zero times. It covers
   the tracked FHIR upload, the identical FHIR re-upload/idempotency sequence,
   and the repository's synthetic CDA parser fixture.
8. Browser coverage keeps the backend socket guard, the macOS Next.js sandbox,
   and the closed browser proxy active for the same FHIR setup path.

The account preference remains strict after the request. Only the fixture upload
carries the explicit override.

### Strict-local upload negative path

1. A test creates and logs into a synthetic account and confirms its stored
   `validated_strict_local` preference.
2. It snapshots that account's upload history, local-AI jobs, patients, records,
   dashboard ingestion totals, and account-prefixed upload storage files.
3. It posts the tracked synthetic FHIR bundle without a mode override.
4. Legacy-v1 admission returns exactly HTTP 409 with
   `Strict-local worker runtime identity is required.`
5. It repeats every snapshot and proves the request created no database or
   storage side effects.

The backend regression checks the same contract through HTTP and SQLAlchemy. It
compares user-scoped `uploaded_files`, `local_ai_jobs`, `health_records`, and
`patients`, plus absence-sensitive, content-free metadata for the exact
temporary upload directory. The snapshot records path, entry type, size, and
SHA-256 but never raw bytes. Account, authentication, and audit rows may exist,
so the assertion is request-scoped rather than a claim that the database is
empty.

### Strict-local summary negative and UI paths

The summary endpoint resolves the strict snapshot before it creates an
`AISummaryPrompt` or `LocalAIJob`. Under the legacy-v1 profile it must return the
exact bounded HTTP 400 detail
`Strict-local worker runtime identity is required.` The endpoint regression
also proves that no summary prompt or local-AI job is created for the request.

The summary page still gets populated synthetic records through the fixed FHIR
helper. These seven cases remain ordinary UI coverage and run without executing
a model:

- `patient selector loads patients`
- `summary type tabs exist`
- `category dropdown appears for By category type`
- `date range inputs appear for Date range type`
- `output format options work`
- `generate button is present and enabled with a patient`
- `AI disclaimer always visible`

These three cases require actual model execution and skip in the current
legacy-v1 local-only profile:

- `generate produces a result`
- `history entry reopens a saved summary without regenerating`
- `generation reports the selected privacy boundary`

They may run only in a separately configured environment that has already
satisfied the real strict worker/pack admission gates. A boolean test flag is
not release evidence; the legacy-v1 profile forcibly clears the flag and cannot
turn it on.

## Startup-safe non-worker profile

`LOCAL_AI_ENABLED=true` causes application lifespan to call
`LocalModelManager.start()`. That method normalizes the configured worker command
and project even when no job will execute, so empty bindings prevent the backend
from starting.

The local-only profile therefore uses:

- `LOCAL_AI_WORKER_COMMAND=/usr/bin/false`
- an explicit absolute `LOCAL_AI_WORKER_PROJECT_DIR` under the task-owned E2E
  runtime root

`/usr/bin/false` is an inert startup sentinel. It is a real one-token executable,
so command normalization succeeds, but it is not an installed local-AI worker,
an attested runtime, a fake success seam, or evidence that a pack can run. The
legacy-v1 manifest guarantees strict upload and summary admission reject before
any spawn or runtime-identity use. Cloud-assisted structured parsing does not
use the worker.

A manager-level startup test proves the sentinel/project pair reaches the
started state without a child PID. A profile-level Playwright regression asserts
the sentinel, project containment, and empty real-pack gate, then reaches the
backend health endpoint. A static profile test ties those behaviors to the
configured environment and keeps the legacy-v1 negative contract explicit.

## Isolated runtime state

Each local-only Playwright command supplies a newly created absolute
`E2E_RUNTIME_ROOT` owned by that command. The profile validates that it is
absolute, creates only these children, and overwrites inherited settings:

- `UPLOAD_DIR=<runtime-root>/uploads`
- `TEMP_EXTRACT_DIR=<runtime-root>/temp-extract`
- `LOCAL_AI_SCRATCH_DIR=<runtime-root>/scratch`
- `LOCAL_AI_MODEL_DIR=<runtime-root>/models`
- `LOCAL_AI_WORKER_PROJECT_DIR=<runtime-root>/non-worker-project`

The negative browser test reads the effective `UPLOAD_DIR` from the profile and
verifies it is inside `E2E_RUNTIME_ROOT`; it does not guess a repository data
path. Command preflight and cleanup operate only on the exact runtime root they
created. Inherited `UPLOAD_DIR` and `TEMP_EXTRACT_DIR` cannot redirect the
backend.

Playwright's `outputDir` is also per run, under
`frontend/test-results/executions/<runtime-token>`. This is separate from the
runtime root so runtime cleanup cannot delete a retained failure trace. It also
prevents Playwright from clearing the parent `frontend/test-results` directory,
where the original Phase 1 traces and JSON results remain untouched. New
execution outputs stay ignored and uncommitted until the root captures or
discards them.

The public browser-test command follows the same ownership boundary. It creates
an absolute runtime root under `frontend/test-results/runtime`, passes it as
`E2E_RUNTIME_ROOT`, and removes only that exact child after a prefix check. It
uses explicit loopback PostgreSQL host and port arguments, records whether it
created the named test database, and drops the database only when that flag is
set. An existing database makes the command fail instead of being deleted.

## Local-only profile contract

The profile fixes these values inside `if (localOnly)`:

- `APP_ENV=test`
- a synthetic test-only 64-hex-character `DATABASE_ENCRYPTION_KEY`
- empty `REAL_MEDICAL_FIXTURES_DIR`
- empty provider credentials, provider project fields, and telemetry inputs
- `LLM_PROVIDER=gemini` with every operation-specific provider override empty,
  so inherited Ollama or LM Studio routing cannot reach a live loopback model
- offline Hugging Face and Transformers flags
- an empty real-attested-pack execution gate
- the task-owned upload, temp, scratch, model, and sentinel project paths
- a task-owned Playwright output directory that cannot clear Phase 1 evidence

The v1 manifest/release paths remain only for negative admission coverage. The
backend socket denial, macOS Next.js sandbox, closed browser proxy, loopback-only
database validation, and disabled service workers remain active.

The profile proves operating-system, socket, and browser egress denial. It does
not claim that a release-ready validated-strict pack is installed, that a model
ran, or that release, benchmark, fidelity, pack-verification, promotion, or
deployment evidence was produced.

## Security and privacy invariants

- Production runtime-identity and upload-admission code remains byte-for-byte
  unchanged.
- Strict-local admission keeps failing closed for legacy-v1, missing, drifted,
  or unattested worker identities.
- No strict request is silently relabeled or allowed to fall back to cloud.
- No release manifest, validation receipt, benchmark, fidelity artifact, model
  result, or promotion evidence is created or fabricated.
- Verification reads only deterministic synthetic repository fixtures.
- Private-fixture configuration is forcibly blank; private tests skip before
  reading their paths.
- No provider credential or shell encryption key is inherited, stored, or
  logged.
- No provider routing is inherited. The credential-free loopback providers are
  not selected by the local-only profile.
- Provider construction and non-loopback connections fail the positive proof.
- Existing content-free errors, owner scoping, encryption at rest, immutable
  artifacts, migration/create-all parity, and runtime-attestation semantics
  remain intact.

## Failure handling

- Positive helpers throw the bounded upload error and do not retry under another
  mode.
- Strict negative tests compare the complete response body.
- A strict upload rejection that creates an owner upload, job, patient, record,
  ingestion total, or account-prefixed storage file fails the regression.
- A strict summary rejection that creates an `AISummaryPrompt` or `LocalAIJob`
  fails the regression.
- The provider-free backend proof runs real background dedup, drains it without
  cancellation, checks terminal upload status, and fails on any provider call.
- Missing private fixtures and the three real-pack summary cases are the only
  legitimate skips.
- Full-suite acceptance requires zero failures and zero not-run tests.

## Exact behavioral file allowlist

No file outside this list may change during behavioral implementation without a
new root approval. No production service, API, model, migration, manifest,
catalog, release, artifact, worker, or pack file is in scope.

Harness and helpers:

- `frontend/playwright.config.ts`
- `frontend/e2e/helpers/api-client.ts`
- `frontend/e2e/strict-local-admission.spec.ts` (new)

Synthetic structured-fixture callers and summary gating:

- `frontend/e2e/admin-records.spec.ts`
- `frontend/e2e/dashboard-home.spec.ts`
- `frontend/e2e/display-badges-providers.spec.ts`
- `frontend/e2e/pagination-integrity.spec.ts`
- `frontend/e2e/record-ai-metadata.spec.ts`
- `frontend/e2e/record-detail-page.spec.ts`
- `frontend/e2e/record-detail-sheet.spec.ts`
- `frontend/e2e/record-renderers.spec.ts`
- `frontend/e2e/setup.spec.ts`
- `frontend/e2e/summaries.spec.ts`
- `frontend/e2e/timeline.spec.ts`
- `frontend/e2e/upload-dedup.spec.ts`
- `frontend/e2e/upload-progress.spec.ts`
- `frontend/e2e/upload-structured.spec.ts`

Backend-only regression coverage:

- `backend/tests/test_local_ai_ci_workflows.py`
- `backend/tests/test_local_ai_model_manager.py`
- `backend/tests/test_processing_mode_snapshot.py`
- `backend/tests/test_upload.py`

Public documentation:

- `docs/operations-strict-local-ai.md`

Planning documents are limited to this design and its companion implementation
plan under `docs/superpowers/plans/`.

The private CDA callers in `upload-dedup.spec.ts` and
`upload-structured.spec.ts` retain the generic path-based helper. The
pagination test stops writing a temporary file and uses the content helper.

## Test strategy

Implementation uses strict TDD.

1. Add static profile and manager-startup tests. Observe the profile test fail
   against the stale fake-worker/runtime-root contract, then add the minimal
   startup-safe sentinel profile.
2. Add upload and summary endpoint characterization regressions for the exact
   legacy-v1 rejection and request-scoped absence of database/storage side
   effects. These should pass before the harness repair; a failure means the
   diagnosis is incomplete.
3. Add the real structured upload/dedup provider-call regressions. Do not patch
   background scheduling. Cover tracked FHIR, identical FHIR re-upload, and
   synthetic CDA, drain dedup, and assert terminal status and zero provider
   construction.
4. Add the fixed tracked-FHIR and generated-pagination content helpers. Migrate
   one populated case, observe the reproduced 409 turn green, then migrate the
   mechanically enumerated synthetic callers.
5. Add the strict browser negative regression using the effective runtime upload
   root.
6. Split summary UI coverage from the three real-pack execution cases and add
   the exact endpoint negative regression.
7. Run focused backend runtime-attestation, startup, upload, summary, profile,
   and migration/create-all parity tests with `APP_ENV=test` and explicit
   loopback database commands.
8. Recreate the focused backend database before the final integration gate.
9. Run focused populated-upload Playwright tests with a new per-command runtime
   root and per-run Playwright output directory.
10. Enumerate the updated suite and record the exact total, then run the complete
    local-only suite with one worker and retained-on-failure traces.

Run the humanizer workflow on the edited public operations prose before it is
accepted. It may improve voice but must not weaken security or attestation
language.

## Full-suite acceptance

The exact enumerated total is recorded after the new spec is added. No passed
count is predicted in advance. Acceptance is:

- zero failures;
- zero not-run tests;
- every non-private, non-pack-required case passes;
- skips come only from the two private-fixture families below and the three
  named real-pack summary execution cases.

Private structured-fixture family:

- `Cross-format dedup › CDA then FHIR upload detects cross-format duplicates`
- `Structured file uploads › upload XDM/CDA ZIP package`
- `Structured file uploads › upload standalone CDA XML`

Private unstructured-fixture family:

- `Extraction progress tracking › extraction progress counts are accurate for batch upload`
- `Mixed content upload classification › unstructured upload goes to extraction pipeline`
- `Unstructured Upload › upload single RTF via API`
- `Unstructured Upload › upload batch RTFs via UI`
- `Unstructured Upload › upload PDF via API`
- `Unstructured Upload › extraction progress tracking via API`
- `Duplicate file upload (idempotency) › re-uploading identical unstructured file returns duplicate_file`

Real-pack summary execution cases:

- `Summaries page › generate produces a result`
- `Summaries page › history entry reopens a saved summary without regenerating`
- `Summaries page › generation reports the selected privacy boundary`

The final report includes the exact enumerated, passed, skipped, failed, and
not-run counts. It also confirms no provider, external network, model, download,
pack, release, promotion, or deployment call occurred.

Any unrelated baseline must be identified by exact test name, before/after
status, and reproduction on the unchanged baseline. Generated traces, JSON
results, databases, caches, uploaded bytes, and runtime artifacts are not
committed.

## Rejected approaches

### Fabricate a v2 E2E pack and release

Rejected. A truthful v2 strict runtime requires real worker identity,
installation, receipts, benchmark and model-backed fidelity evidence, and
release promotion. Placeholders would violate the trust boundary and would not
prove strict-local operation.

### Weaken or bypass strict-local admission in tests

Rejected. The legacy-v1 `runtime_identity_required` invariant is deliberate. A
test flag, monkeypatch, permissive manifest, or swallowed rejection would remove
the behavior the regression protects.

### Present the startup sentinel as a worker

Rejected. `/usr/bin/false` exists only so application lifespan can normalize a
valid executable. It cannot produce a worker protocol response and must never be
spawned in the approved flows.

### Fake summary success

Rejected. The current profile has no attested worker or pack. Model-execution
summary tests skip; they do not use a fabricated v2 receipt, fake worker output,
or browser interception that claims generation succeeded.

### Globally change local-only accounts to cloud-assisted

Rejected. That would erase coverage of the strict preference. The override
belongs only on clearly named synthetic structured-fixture calls.

### Use `prompt_only` for structured fixture ingestion

Rejected. `prompt_only` is not an admitted ingestion mode.
