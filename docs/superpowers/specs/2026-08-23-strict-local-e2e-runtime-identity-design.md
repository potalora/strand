# Strict-Local E2E Runtime-Identity Harness Repair Design

**Date:** 2026-08-23

**Status:** Approved design; implementation plan awaiting root approval

**Baseline:** `origin/main` at
`32075826694e1a24bfb36699fb14374b7916b29a`

**Scope:** Local-only Playwright harness and coverage only. Production strict-local
runtime identity, admission, manifests, releases, and processing behavior do not
change.

## Problem

A fresh, single-worker run of the complete `E2E_LOCAL_ONLY` Playwright profile
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
results are diagnostic evidence only; they remain ignored and must not be
committed.

## Root cause

This is a stale harness contract and a coverage gap, not a production admission
bug.

The local-only profile still selects the legacy v1 manifest and release paths
and presents an argument-bearing Python fake-worker command. Its shared API
client then stores `validated_strict_local` at login and explicitly submits that
mode for each structured fixture upload.

Current admission correctly treats legacy-v1 jobs as
`runtime_identity_required`. For strict ingestion it loads the legacy manifest
only far enough to diagnose the missing worker bundle identity, then fails
before file storage, ingestion, artifact validation, release-evidence loading,
or worker execution. A valid current identity could only be derived from the
one-token installed worker entry point together with its real project,
`uv.lock`, and effective package tree. The argument-bearing fake wrapper cannot
truthfully satisfy that contract.

Changing the profile to name the v2 manifest would not create a valid strict
runtime. Admission would still require a matching active manifest, an immutable
validation receipt, no nonterminal pack operation, a matching runtime identity,
and release evidence backed by the required benchmark and model-fidelity gates.
No v2 release manifest exists in this checkout, by design. Tests must not invent
one.

The product is therefore behaving as intended: strict-local admission is
fail-closed. The harness incorrectly conflates two independent claims:

1. browser and application processes cannot make non-loopback network calls;
2. a release-ready, runtime-attested strict-local model pack is installed.

`E2E_LOCAL_ONLY` can and should prove the first claim on an ordinary test
machine. It must not pretend to prove the second.

## Decision

Repair the harness by separating network confinement from processing-mode
admission.

- Keep `validated_strict_local` as the account's local-only default preference.
  Existing setup coverage will continue to prove that preference is persisted.
- Add an explicitly named structured-fixture helper that always submits
  `processing_mode=cloud_assisted` for deterministic synthetic FHIR/CDA parser
  inputs.
- Use that helper only where a test needs records populated from deterministic
  structured fixtures.
- Add a real negative strict-local upload regression that submits the legacy-v1
  profile's default strict preference and proves the exact fail-closed 409 with
  no request side effects.
- Keep the OS/socket/browser network-denial guards active for all of these
  paths.

`cloud_assisted` is the correct explicit positive-fixture mode because current
ingestion admits exactly `cloud_assisted` and `validated_strict_local`.
`prompt_only` is a summary mode, not an ingestion mode, and is correctly rejected
for uploads. Selecting `cloud_assisted` here does not authorize a provider call:
the allowed positive inputs use the deterministic FHIR/CDA parser routes, the
backend tests patch provider construction to fail if reached, provider
credentials are empty, and the local-only socket guard denies non-loopback
connections.

This is a narrow fixture override, not a global relabel. Unstructured uploads,
strict admission tests, normal user preferences, and production requests retain
their existing mode semantics.

## Data flow

### Deterministic structured-fixture positive path

1. A test creates and logs into a fresh synthetic account.
2. Local-only login persists the user's default
   `validated_strict_local` preference.
3. The explicitly named fixture helper reads only an in-repository synthetic
   FHIR/CDA file, adds `processing_mode=cloud_assisted` to that upload request,
   and posts it to the loopback backend.
4. Admission captures the explicit cloud-assisted mode without constructing a
   strict runtime snapshot.
5. The structured ingestion coordinator decrypts the temporary upload and
   dispatches directly to the deterministic FHIR or CDA parser.
6. Records remain owner-scoped and encrypted-at-rest behavior remains intact.
7. Existing deduplication, status polling, and record UI assertions continue to
   run under browser, Next.js, and backend network denial.

The account preference remains strict after this request; only the single
fixture upload carries the explicit override.

### Strict-local negative path

1. A test creates and logs into a fresh synthetic account and confirms the
   stored `validated_strict_local` preference.
2. It snapshots the account's table-backed upload history, local-AI jobs,
   patients, records, and dashboard ingestion totals, plus the set of upload
   storage files prefixed with that account's UUID.
3. It posts the synthetic FHIR bundle without the cloud-assisted fixture
   override, so the stored strict preference governs admission.
4. Legacy-v1 admission returns exactly HTTP 409 with
   `Strict-local worker runtime identity is required.`
5. It repeats every snapshot and proves no account-scoped database or storage
   state changed.

The backend regression repeats the same contract at the HTTP/SQLAlchemy
boundary, directly querying `uploaded_files`, `local_ai_jobs`, `health_records`,
and `patients` for the request's user and comparing the isolated upload
directory before and after. Account and audit rows may legitimately exist, so
the assertion is request-scoped rather than a claim that the entire database is
empty.

## Local-only profile contract

The local-only Playwright profile will make these semantics executable:

- `APP_ENV=test` is fixed.
- `DATABASE_ENCRYPTION_KEY` is a fixed 64-hex-character synthetic test key set by
  the profile. It never inherits or prints a developer's real shell key.
- `REAL_MEDICAL_FIXTURES_DIR`, every provider credential/project field, and
  telemetry inputs are forced empty or disabled.
- Hugging Face and Transformers offline flags remain enabled.
- backend socket denial, the macOS Next.js sandbox, the closed browser proxy,
  loopback-only database validation, and disabled service workers remain
  unchanged.
- The v1 manifest/release paths remain only to exercise the negative admission
  contract.
- `LOCAL_AI_WORKER_COMMAND` and `LOCAL_AI_WORKER_PROJECT_DIR` are forced empty.
  The argument-bearing Python fake wrapper is not presented as an installed,
  runnable, or attested worker.

The profile proves loopback-only operating-system, socket, and browser egress
denial. It does not claim that a release-ready validated-strict pack is present,
that a model ran, or that any release, benchmark, fidelity, pack verification,
promotion, or deployment evidence was produced.

## Security and privacy invariants

- Production runtime-identity and upload-admission code remains byte-for-byte
  unchanged.
- Strict-local admission continues failing closed for legacy-v1, missing,
  drifted, or unattested worker identities.
- No strict request is silently relabeled or allowed to fall back to cloud.
- No release manifest, validation receipt, benchmark, fidelity artifact, model
  result, or promotion evidence is created or fabricated.
- Only deterministic synthetic repository fixtures are read during verification.
- Private-fixture configuration is forcibly blank; the legitimate private tests
  skip without reading their paths.
- No provider credentials are inherited, stored, or logged, and provider
  construction is a test failure on the structured parser proof path.
- All external network paths stay blocked. No model, provider, download, pack,
  cloud, or telemetry call is allowed.
- Existing content-free policy errors, owner scoping, upload encryption,
  immutable artifacts, database migration/create-all parity, and runtime
  attestation semantics remain intact.

## Failure handling

- The positive fixture helper throws the current bounded upload error if the
  structured request does not return success; it does not retry under another
  mode.
- The strict negative test compares the complete response body, not a substring.
- Any strict rejection that creates a user-owned upload row, job, patient,
  health record, ingestion total, or account-prefixed storage file fails the
  regression.
- Any provider construction or non-loopback connection attempt fails the
  positive structured-fixture proof.
- Missing private fixtures remain the only legitimate skips. They never turn
  into failures or uploads.
- Full-suite acceptance requires zero failures and zero not-run tests; a setup
  failure cannot be hidden by weakening assertions or accepting downstream
  skips.

## Exact behavioral file allowlist

No file outside this list may change during behavioral implementation without a
new root approval. In particular, no production service, API, model, migration,
manifest, catalog, release, artifact, worker, or pack file is in scope.

Harness and helpers:

- `frontend/playwright.config.ts`
- `frontend/e2e/helpers/api-client.ts`
- `frontend/e2e/strict-local-admission.spec.ts` (new)

Synthetic structured-fixture callers:

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

- `backend/tests/test_processing_mode_snapshot.py`
- `backend/tests/test_upload.py`
- `backend/tests/test_local_ai_ci_workflows.py`

Public documentation:

- `docs/operations-strict-local-ai.md`

Planning documents are limited to this design and its companion implementation
plan under `docs/superpowers/plans/`.

The two private CDA callers in `upload-dedup.spec.ts` and
`upload-structured.spec.ts` retain their existing generic upload helper and are
not converted into positive local-only fixture uploads. The local-only profile's
forced-empty `REAL_MEDICAL_FIXTURES_DIR` makes them deterministic skips. The
synthetic FHIR caller and browser dropzone case in those same files are in scope.

## Test strategy and acceptance

Implementation uses strict TDD.

1. Add the smallest static profile regression and observe it fail because the
   stale profile presents the argument-bearing fake worker and does not force a
   synthetic encryption key or blank private-fixture configuration.
2. Add the backend and Playwright characterization regressions for the exact
   legacy-v1 409 and request-scoped absence of database/storage side effects.
   Observe these pass before the harness fix, proving that production admission
   is already correct.
3. Add structured FHIR and CDA provider-construction guards. Confirm the
   provider-free parser contract before changing fixture routing.
4. Introduce the explicit cloud-assisted fixture helper and migrate one focused
   populated-record case. Observe that the previously reproduced 409 turns
   green, then migrate the mechanically enumerated synthetic callers.
5. Implement only the remaining profile/helper/call-site changes above.
6. Run focused processing-snapshot, upload, runtime-identity, and profile-static
   backend tests.
7. Run fresh disposable-database Alembic-upgrade and metadata-create-all parity
   checks where the existing suite defines them; no migration changes are made.
8. Run focused populated-upload Playwright specs and the new negative admission
   spec with synthetic-only inputs.
9. Enumerate the updated Playwright suite and record its exact total. Do not
   hard-code a future passed count in advance.
10. Run the complete local-only suite with one worker and retained-on-failure
   traces.

Before the public operations document is accepted, run the repository's
humanizer workflow and review its suggestions without allowing it to weaken any
security or attestation language.

Acceptance is:

- every non-private-fixture case passes;
- exactly the legitimate private-fixture cases skip;
- zero tests fail;
- zero tests are not run;
- the exact new enumerated, passed, and skipped counts are reported from the
  final run;
- no provider, external network, model, download, pack, release, promotion, or
  deployment call occurs;
- generated traces, JSON results, databases, caches, uploaded bytes, and runtime
  artifacts are not committed.

Any unrelated baseline must be identified with an exact before/after test name,
status, and reproduction on the unchanged baseline rather than being folded
into acceptance.

## Rejected approaches

### Fabricate a v2 E2E pack and release

Rejected. A truthful v2 strict runtime requires real worker identity,
installation, receipts, benchmark and model-backed fidelity evidence, and
release promotion. Generating placeholders would violate the production trust
boundary and still would not prove real strict-local operation.

### Weaken or bypass strict-local admission in tests

Rejected. The legacy-v1 `runtime_identity_required` invariant is deliberate and
must remain production-identical. A test flag, monkeypatch, permissive manifest,
or swallowed 409 would remove the behavior the regression needs to protect.

### Globally change local-only accounts to cloud-assisted

Rejected. That would erase coverage of the user's strict preference and obscure
which requests are deliberately using the deterministic structured parser. The
mode override belongs only on clearly named synthetic fixture calls.

### Use `prompt_only` for structured fixture ingestion

Rejected. `prompt_only` is not an admitted ingestion mode. Converting a known
policy conflict into the test harness contract would be both incorrect and
unstable.
