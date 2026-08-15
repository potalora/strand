# PR #62 and PR #63 remediation design

**Status:** Approved

**Date:** 2026-08-13

**Scope:** Eight findings from the post-merge review of PR #62 and PR #63

## Goal

Fix the reviewed regressions without weakening strict-local privacy, clinical
grounding, bounded inference, user ownership, or durable job behavior. The work
must be split so three independent tracks can start in parallel and the runtime
attestation release can follow the worker corrections.

## Findings in scope

1. Authorization validity dates can be treated as evidence that a procedure
   occurred.
2. Medication lifecycle blockers can leak across subjects in one evidence span.
3. Fragment depth and split limits can schedule more calls than the global
   12-attempt ceiling permits.
4. Validation receipts do not bind the strict-local worker code that will run.
5. Any authenticated account can mutate the machine-global validated model
   pack.
6. Cancelling a queued strict-local ingestion through the job API leaves the
   upload runnable.
7. Retryable strict-local summary failures disappear after reload.
8. Compacted batch-upload responses can be associated with the wrong files and
   rejected files are not reported clearly.

## Constraints

- Validated strict-local work must branch before any cloud-capable provider is
  constructed and must never fall back to cloud.
- The backend remains the authoritative clinical-validation boundary. Worker
  grounding reduces invalid output but cannot replace backend rejection.
- Progress, failure, runtime-attestation, and operation responses remain
  content-free. They must not expose document text, prompt text, model output,
  patient identity, paths, or raw exceptions.
- Strict ingestion job and upload lifecycle transitions must be atomic when
  both rows represent the same operation.
- The 12-attempt and 32,768-generated-token extraction ceilings remain hard
  limits unless a separate measured change is approved.
- The test suite must use deterministic synthetic cases. No private medical
  fixtures, provider calls, model downloads, or live cloud tests are part of
  ordinary remediation verification.
- Alembic and `Base.metadata.create_all()` strict-local guards must remain
  semantically identical.
- Existing unrelated worktrees and untracked files must remain untouched.

## Approaches considered

### One combined remediation PR

This would make final integration simple, but it would mix authorization,
clinical validation, job lifecycle, frontend behavior, database guards, and
release evidence. A reviewer could not approve one risk area independently,
and runtime-attestation evidence would be regenerated while worker behavior was
still changing.

### Minimal line-by-line hotfixes

This would land quickly, but it would preserve duplicated cancellation logic,
leave rejected batch files silent, and bind validation to another manually
bumped version rather than the worker that actually runs.

### Four focused PRs with an integration gate

This is the selected approach. Three PRs can start in parallel. The fourth
depends on the corrected worker code and produces a newly attested strict-local
release. Each PR has its own regression tests and can be reviewed on one set of
invariants.

## Parallel execution model

Wave 1 starts in three native Codex worktrees from the same planning commit. That commit adds only
the approved design and execution plans, so its product tree matches the reviewed `main` commit:

- **Track A: Global pack authorization**
- **Track B: Clinical extraction and bounded work**
- **Track C: Durable jobs and upload identity**

Track B owns worker grounding and scheduling behavior. Track D starts only
after Track B is accepted because the runtime fingerprint must cover the final
worker source and dependency lock:

- **Track D: Worker runtime attestation and release promotion**

Track A and Track C both touch `backend/app/api/local_ai.py`, but they modify
separate endpoint groups. Each stays independently reviewable; the root
integrator resolves import or nearby-line conflicts when combining branches.
No subagent commits. The root agent reviews and verifies each track; commits require Pedro's
explicit authorization.

## Track A: Global pack authorization

### Authorization model

The validated model pack is machine-global, while medical records and jobs are
user-owned. Pack lifecycle mutation therefore requires a machine-operator
capability rather than ordinary authentication.

Add `LOCAL_AI_OPERATOR_USER_IDS`, parsed by application settings into an
allowlist of UUIDs. An empty allowlist fails closed for web-based pack
mutation. A syntactically invalid value prevents application startup. The app
does not silently promote the first registered user. Local CLI maintenance
remains an operating-system owner action and is documented as outside
web-account authorization.

Add `require_local_ai_operator()` in `backend/app/dependencies.py`. It preserves
401 for unauthenticated requests and returns 403 for authenticated accounts not
present in the configured allowlist.

Apply the dependency to every machine-global mutation or restart endpoint:

- install, verify, update, and rollback;
- pack-operation resume and retry;
- removal of one model role or the complete pack.

Machine-global operation detail is operator-only. Authenticated users may still
read pack readiness because they need it when selecting a processing mode.
`LocalPackStatusResponse` gains `can_manage_pack`, and the Local AI settings card
shows status to everyone. It hides lifecycle controls from non-operators and
shows fixed copy explaining that the machine operator manages the pack.

### Acceptance behavior

- A non-operator cannot create, restart, or mutate a pack operation and cannot
  change model-store files.
- An operator retains the current successful behavior and remains the audit
  actor.
- An empty allowlist denies mutation rights, and malformed configuration fails
  at startup.
- User-owned job list, get, cancel, and retry routes retain owner scoping and do
  not require machine-operator access.

## Track B: Clinical extraction and bounded work

### Administrative procedure grounding

In billing, claims, and authorization context, a date can describe validity,
service windows, or expiration. It is not proof that a procedure occurred.

The backend accepts `assertion=present` for a procedure in administrative
context only when the procedure's subject-local evidence contains explicit
performance wording such as `underwent`, `performed`, or `status post`. A date,
the procedure name alone, or a procedure-name suffix is insufficient in this
context. Ordinary clinical documents retain existing compatible date and
performance handling.

The worker applies the same conservative rule before returning grounded facts,
and its prompt no longer describes a date by itself as performance evidence.
The worker and backend remain separate packages, so they use small mirrored
pure rules backed by the same deterministic case matrix in both test suites.

### Subject-scoped medication lifecycle

Positive lifecycle support and blocking or uncertain cues must use the same
subject-local source. The source consists of the medication's semantic clause
plus safe dosing continuations and stops at a separator or another medication
subject.

The backend reuses its existing subject-scoped lifecycle source for blockers.
The worker adds equivalent subject-clause selection before assigning active,
stopped, or unknown status. Existing status precedence stays unchanged inside
that bounded source.

### Attempt-aware fragmentation

The scheduler must not create work that cannot fit within the request's global
attempt budget.

After deterministic pre-fitting, the worker rejects a batch set larger than 12
before calling the model. Runtime work uses an ordered queue. Before splitting,
the budget checks attempts already spent, untouched queued batches, and the two
new child calls. A split is admitted only when every minimum required future
call can still fit.

The static split ceiling becomes at most 11, which is the largest full binary
split count compatible with 12 first-call leaves. Depth 5 remains a shape
guard; the dynamic capacity check is authoritative. Syntax retry and token
charging behavior remain unchanged.

### Acceptance behavior

- A validity-dated authorization does not produce a performed Procedure.
- Explicit subject-local performance wording remains accepted.
- A cue for medication B cannot promote or reject medication A.
- An oversized pre-fit batch fails before inference with the existing
  content-free `work_attempt_limit` category.
- No thirteenth generation call can occur, including after retries or runtime
  splits.

## Track C: Durable jobs and upload identity

### Paired ingestion cancellation

Create a focused strict-ingestion lifecycle helper under
`backend/app/services/local_ai/`. Both the upload cancellation endpoint and the
job cancellation endpoint call it.

The helper locks the paired upload and job in one consistent order. For a
queued ingestion it sets both cancellation flags and terminalizes both rows as
cancelled in one transaction. For processing ingestion it sets both flags but
leaves terminalization to cooperative worker cleanup. Summary cancellation
remains job-only. Database commit occurs before best-effort worker IPC and
audit follow-up.

### Failed-job recovery after reload

Extend `GET /local-ai/jobs` with an additive
`include_retryable_failed=false` query parameter. When combined with
`active_only=true`, the query returns active jobs plus owner-scoped failed jobs
whose bounded failure payload marks them retryable. It excludes completed,
cancelled, non-retryable, and cross-owner jobs before applying the 50-row
limit.

The background monitor uses this query only for initial hydration. It polls
active jobs as it does today, shows recovered retryable failures without an
initial toast, and exposes the existing Retry action.

### Self-identifying batch results

`UnstructuredUploadResponse` gains the server-stored filename. Both single and
batch endpoints populate it. The frontend builds accepted upload rows from
each response item's filename and upload ID, never from the original array
index.

The batch response also gains a bounded `rejected` list with the submitted
filename and a stable rejection code such as `unsupported_type`,
`file_too_large`, or `invalid_signature`. The endpoint remains best-effort and
keeps accepted items in `uploads`; the frontend displays fixed local copy for
rejected items. An all-rejected batch therefore produces an actionable result
instead of appearing to succeed with no work.

### Acceptance behavior

- Cancelling a queued ingestion through either endpoint leaves both rows
  cancelled and unclaimable.
- Reloading after a retryable strict-summary failure restores the failure and
  Retry action.
- A compacted response cannot shift later upload IDs onto earlier filenames.
- Rejected files are reported without stack traces or server paths.

## Track D: Worker runtime attestation and release promotion

### Threat boundary

Runtime attestation protects against stale validation after worker source,
entrypoint, or locked dependency changes. It is not a defense against an
operating-system owner or root attacker who can replace application code and
its trust metadata together.

### Portable runtime identity

Use a canonical worker-bundle digest rather than hashing the generated venv
launcher verbatim, because launcher shebangs contain installation-specific
paths. The digest covers sorted relative paths and bytes for:

- the installed or editable `local_ai_mlx_worker/**/*.py` package tree;
- `workers/local_ai/apple_mlx/pyproject.toml`;
- `workers/local_ai/apple_mlx/uv.lock`;
- the fixed console-entrypoint declaration and identity scheme version.

The resolved command must still be a regular, non-symlink executable in the
expected worker environment. The runtime identity contains digests only, never
local paths.

Add the worker-bundle digest to a v2 manifest runtime contract and to the
validation receipt. Pack verification compares the observed runtime identity
before running fixtures. The model manager recomputes and compares it before
every worker spawn, including OCR, extraction, summary, token counting,
fidelity, and pack verification. Drift fails closed before PHI reaches the
worker.

Old v1 receipts remain readable only so the application can report that the
pack requires revalidation. They cannot admit new strict-local jobs or execute
queued work after the v2 requirement is enabled.

Deployment of the v2 requirement needs a no-active-strict-jobs preflight.
Existing v1 queued jobs are never rewritten to claim a v2 identity. If one is
found after upgrade, it fails closed with a bounded `runtime_identity_required`
code and the user must resubmit the upload or summary after the v2 pack is
ready. Stored records and source uploads are not deleted.

### Database and release parity

Update both the Alembic strict-local manifest guard and
`backend/app/models/local_ai_ddl.py` with semantically identical v2 checks.
Test a migrated database and a fresh `create_all` database.

Because the manifest digest changes, old benchmark, fidelity, release-evidence,
and receipt files cannot be relabeled. After Track B is final, generate the new
worker digest, promote a new locked profile revision, and rerun the documented
offline synthetic runtime verification, fidelity, and benchmark gates. No
private record is required for this release receipt. Any separate local-versus-
cloud decision campaign retains its existing private-fixture opt-in rules.

### Acceptance behavior

- Changing one worker source file or the worker lock invalidates admission and
  execution until validation is rerun against a matching manifest.
- A receipt with a missing or substituted worker digest is rejected.
- A source mutation between manager startup and process spawn is detected.
- Fresh-schema and upgraded-schema guards enforce the same v2 snapshot shape.
- The promoted release contains fresh evidence bound to the new manifest
  digest and corrected worker code.

## Verification and integration gate

Each track starts with failing regression tests and ends with focused backend,
worker, frontend, formatting, and contract checks for its scope. The final
integration branch then runs:

- the ordinary backend suite excluding explicitly gated slow, fidelity,
  local-model, and hardware markers;
- the complete deterministic Apple MLX worker protocol suite;
- frontend unit Playwright tests, TypeScript, lint, and production build;
- strict-local migration and fresh `create_all` parity tests;
- strict-local egress and log-privacy tests;
- the documented offline pack verification, synthetic fidelity, and benchmark
  gates for the new attested profile.

No merge is complete while an expected test is skipped because of a missing
ordinary dependency. Hardware and model-backed gates are reported separately
with their exact artifact receipts.

## Planned documents

After this design is approved in written form, create five implementation
plans:

1. `2026-08-13-local-ai-operator-authorization.md`
2. `2026-08-13-strict-local-extraction-remediation.md`
3. `2026-08-13-durable-jobs-and-upload-identity.md`
4. `2026-08-13-worker-runtime-attestation.md`
5. `2026-08-13-pr62-pr63-remediation-integration.md`

Plans 1 through 3 are executable in parallel worktrees. Plan 4 declares Plan 2
as a prerequisite. Plan 5 is the root-owned integration, evidence, and release
gate.
