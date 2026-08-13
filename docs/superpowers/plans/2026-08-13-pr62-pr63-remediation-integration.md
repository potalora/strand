# PR #62 and PR #63 remediation integration plan

> **For the root integrator:** Execute this plan after the focused track plans. Track A, B, and C may run concurrently. Track D must not start until Track B has been reviewed, verified, and committed with Pedro's explicit authorization. Use `subagent-driven-development` inside each focused task and `verification-before-completion` before every handoff.

**Goal:** Land the eight reviewed fixes as four focused, reviewable changes while preserving one reproducible final verification and release gate.

**Architecture:** The three Wave 1 branches start from the same planning commit, whose product tree is identical to reviewed `main`. Track A owns machine-global pack authorization, Track B owns clinical grounding and bounded extraction, and Track C owns durable job/upload behavior. Track D starts from the accepted Track B branch and binds the corrected worker to a v2 runtime identity. A root-owned integration branch resolves the expected shared-file seams and runs the combined deterministic and offline release gates.

**Tech stack:** Git worktrees, Python 3.11, FastAPI, PostgreSQL 16, Alembic, Apple MLX worker, Next.js 16, TypeScript, Playwright, pytest, Ruff.

## Non-negotiable boundaries

- Do not implement focused-track changes from the integration task. Each track owns its plan and regression tests.
- Do not start Track D from the original base or generate its runtime digest before Track B is final.
- Do not commit, push, open a pull request, merge, deploy, download a model, call a provider, or use a private medical fixture unless the user separately authorizes that action.
- The root agent owns commits, conflict resolution, combined verification, release evidence review, and merge decisions. Subagents may investigate, implement, or verify, but do not commit.
- Preserve unrelated worktrees and changes. Never repurpose the main checkout for focused implementation.
- Do not weaken strict-local egress denial, clinical validation, row ownership, database guard parity, the 12-attempt ceiling, or the 32,768-generated-token ceiling.
- All routine tests use deterministic synthetic inputs. Hardware/model-backed gates and private local-versus-cloud fidelity remain separately gated.

## Native worktree and branch map

| Track | Native Codex starting branch | Retained branch name | May start |
| --- | --- | --- | --- |
| A: operator authorization | `codex/pr62-pr63-remediation-planning` | `codex/local-ai-operator-authorization` | Wave 1 |
| B: extraction remediation | `codex/pr62-pr63-remediation-planning` | `codex/strict-local-extraction-remediation` | Wave 1 |
| C: durable jobs and upload identity | `codex/pr62-pr63-remediation-planning` | `codex/durable-jobs-upload-identity` | Wave 1 |
| D: runtime attestation | accepted Track B branch | `codex/worker-runtime-attestation` | Only after Track B acceptance |
| Integration | `codex/pr62-pr63-remediation-planning` | `codex/pr62-pr63-remediation-integration` | Root only |

The planning branch is `codex/pr62-pr63-remediation-planning`. It already contains reviewed
`main` as an ancestor. The native Worktree flow also needs the five plan files to be present in a
commit; untracked planning files do not travel with a branch selected from another checkout. After
Pedro explicitly authorizes that planning commit, verify `git merge-base --is-ancestor main HEAD`
and record `git rev-parse HEAD` as `<PLANNING_BASE>`. Start each Wave 1 task by selecting
**Worktree** under the Codex composer and choosing that branch. Codex creates a separate managed
worktree at the planning commit in detached-HEAD state. Use **Create branch here** with the
retained branch name only when a task's implementation is ready to preserve for review.

## Ownership and expected seams

| Shared path or concern | Focused owner | Integration rule |
| --- | --- | --- |
| `backend/app/api/local_ai.py` | Track A for pack lifecycle/detail authorization; Track C for job cancellation/list hydration | Preserve both changes. Resolve only imports or nearby-line conflicts; do not redesign either track during merge. |
| `backend/tests/test_local_ai_api.py` | Track A for operator policy; Track C for owner-scoped jobs | Retain both test groups and shared fixtures. Never make an operator fixture the default authenticated user. |
| `backend/app/config.py` and `.env.example` | Track A operator allowlist; Track D worker project identity | Preserve both independent settings and documentation. Empty operator allowlist must still fail closed. |
| `docs/backend-handoff.md` | Tracks A, C, and D | Keep all additive contract and runtime-attestation changes, then run a prose consistency pass. |
| Worker source and lock | Track B behavior, then Track D identity/release | Track D must compute identity from the accepted Track B tree; no manual digest carry-forward. |
| Strict-local database guards | Track D only | Migration and create-all DDL must remain semantically identical. |

## Task 1: Launch the three independent tasks

**Files:**

- Read: `docs/superpowers/specs/2026-08-13-pr62-pr63-remediation-design.md`
- Execute: `docs/superpowers/plans/2026-08-13-local-ai-operator-authorization.md`
- Execute: `docs/superpowers/plans/2026-08-13-strict-local-extraction-remediation.md`
- Execute: `docs/superpowers/plans/2026-08-13-durable-jobs-and-upload-identity.md`

- [ ] Verify each managed worktree reports `<PLANNING_BASE>` and is clean before editing:

  ```bash
  git branch --show-current
  git rev-parse HEAD
  git status --short
  ```

  Expected: an empty branch name because native Worktree tasks begin at detached `HEAD`, the
  shared planning commit, and no output from `git status --short`.

- [ ] Start three new Codex tasks in **Worktree** mode. Select
  `codex/pr62-pr63-remediation-planning` as the starting branch for each task.

- [ ] Give each task only its focused plan. Require test-first implementation, focused
  verification, a final diff review, and a stop before commit. When the task is ready to retain,
  use **Create branch here** with the branch name in the native worktree map.

- [ ] Keep Track D and the integration task idle while A, B, and C run.

## Task 2: Review and accept Wave 1 independently

For each track, the root agent must inspect the full diff, rerun the plan's focused checks, and confirm the branch contains no out-of-scope files.

- [ ] Review Track A against the machine-global versus user-owned authorization boundary. Confirm all nine lifecycle/detail endpoints are gated, status remains readable, and user-owned jobs retain owner scoping.

- [ ] Review Track B against the shared clinical matrix and attempt-budget invariants. Confirm the backend remains authoritative, worker rules are mirrored, and no thirteenth generation is possible.

- [ ] Review Track C against atomic paired cancellation, reload recovery, and server-authored upload identity. Confirm commit-before-IPC ordering and bounded content-free rejection responses.

- [ ] After using **Create branch here**, run in every Wave 1 worktree:

  ```bash
  git diff --check
  git status --short
  git diff --stat <PLANNING_BASE>...HEAD
  ```

- [ ] Commit only after the user authorizes commits. Use one focused commit per track unless a plan explicitly needs separable database migration and application commits.

Track B's accepted commit SHA becomes `<TRACK_B_COMMIT>`. Track D may not proceed without that exact SHA.

## Copy-paste prompts for Wave 1

Start each as a separate Codex desktop task in **Worktree** mode from
`codex/pr62-pr63-remediation-planning`.

### Track A prompt

```text
Implement Track A only by executing
docs/superpowers/plans/2026-08-13-local-ai-operator-authorization.md.

First verify that this Codex-managed worktree is clean and that HEAD matches the selected
planning branch. Follow the plan test-first. Use subagents only for bounded independent
investigation or verification, and do not let concurrent agents edit overlapping files. Do not
touch Tracks B, C, or D. Do not use provider calls, private medical fixtures, model downloads, or
live services. Run every focused and track-level gate in the plan, inspect the complete diff, and
stop before committing, pushing, opening a PR, merging, or deploying. Report changed files, exact
test results, remaining risks, and that the work should be retained as
codex/local-ai-operator-authorization with Create branch here.
```

### Track B prompt

```text
Implement Track B only by executing
docs/superpowers/plans/2026-08-13-strict-local-extraction-remediation.md.

First verify that this Codex-managed worktree is clean and that HEAD matches the selected
planning branch. Follow the plan test-first. Use subagents only for bounded independent
investigation or verification, and do not let concurrent agents edit overlapping files. Preserve
the backend as the authoritative clinical-validation boundary and the hard 12-attempt and 32,768-
generated-token ceilings. Do not touch Tracks A, C, or D. Do not use provider calls, private
medical fixtures, model downloads, or live services. Run every focused and track-level gate in
the plan, inspect the complete diff, and stop before committing, pushing, opening a PR, merging,
or deploying. Report changed files, exact test results, remaining risks, and that the work should
be retained as codex/strict-local-extraction-remediation with Create branch here.
```

### Track C prompt

```text
Implement Track C only by executing
docs/superpowers/plans/2026-08-13-durable-jobs-and-upload-identity.md.

First verify that this Codex-managed worktree is clean and that HEAD matches the selected
planning branch. Follow the plan test-first. Use subagents only for bounded independent
investigation or verification, and do not let concurrent agents edit overlapping files. Preserve
owner scoping, atomic paired-ingestion transitions, commit-before-IPC ordering, and content-free
failure/rejection payloads. Do not touch Tracks A, B, or D. Do not use provider calls, private
medical fixtures, model downloads, or live services. Run every focused and track-level gate in
the plan, inspect the complete diff, and stop before committing, pushing, opening a PR, merging,
or deploying. Report changed files, exact test results, remaining risks, and that the work should
be retained as codex/durable-jobs-upload-identity with Create branch here.
```

## Task 3: Launch Track D from accepted Track B and execute attestation

**Files:**

- Execute: `docs/superpowers/plans/2026-08-13-worker-runtime-attestation.md`

- [ ] Start a new native **Worktree** task from the accepted Track B branch. Do not start from the
  planning branch and do not copy only the worker file; the full accepted Track B commit must be
  an ancestor.

- [ ] Verify the dependency:

  ```bash
  git merge-base --is-ancestor <TRACK_B_COMMIT> HEAD
  git status --short
  ```

  Expected: exit code 0 and a clean worktree before Track D edits begin.

- [ ] Execute Track D test-first. Its v2 manifest, receipt, database guards, and release artifacts must all be generated from the same observed worker-bundle identity.

- [ ] Treat v1 receipts as readable diagnostics only. Do not rewrite queued v1 snapshots or relabel old benchmark/fidelity/release artifacts.

- [ ] Before any pack verification or release promotion, prove there are no active strict-local jobs in the target environment. If that cannot be proven, stop; do not mutate runtime state.

## Task 4: Assemble the root-owned integration branch

The integration branch is a temporary proving ground. It does not replace the focused branches or hide their review history.

- [ ] After Pedro explicitly authorizes integration Git operations, start a native **Worktree**
  task from `codex/pr62-pr63-remediation-planning` and use **Create branch here** to retain it as
  `codex/pr62-pr63-remediation-integration`.

- [ ] Verify Track D contains the accepted Track B commit before any integration merge:

  ```bash
  git merge-base --is-ancestor \
    <TRACK_B_COMMIT> codex/worker-runtime-attestation
  ```

  Expected: exit code 0. Stop if Track D was created from any other baseline.

- [ ] Merge the accepted branches in this fixed order, resolving only the ownership seams listed
  above:

  ```bash
  git merge --no-ff codex/local-ai-operator-authorization
  git merge --no-ff codex/durable-jobs-upload-identity
  git merge --no-ff codex/strict-local-extraction-remediation
  git merge --no-ff codex/worker-runtime-attestation
  ```

  Merging Track B before Track D preserves the dependency visibly; because Track B is already an
  ancestor of Track D, the final merge contributes only Track D's attestation work rather than
  replaying Track B as an unrelated patch.

- [ ] Inspect the combined graph and diff:

  ```bash
  git log --graph --decorate --oneline --all -30
  git diff --check
  git diff --stat <PLANNING_BASE>...HEAD
  git status --short
  ```

- [ ] Re-read the approved design and map every finding to at least one regression test. Stop if a finding has implementation but no deterministic regression.

## Task 5: Run the combined deterministic gate

Install ordinary dependencies locally if the clean worktree does not yet have them. Missing routine dependencies are setup failures, not acceptable skips.

- [ ] Backend formatting and lint:

  ```bash
  cd backend
  uv run ruff format --check app scripts tests
  uv run ruff check app scripts tests
  ```

- [ ] Ordinary backend suite:

  ```bash
  cd backend
  uv run pytest -q -m "not slow and not fidelity and not local_model and not hardware"
  ```

- [ ] Complete deterministic worker suite:

  ```bash
  cd workers/local_ai/apple_mlx
  uv run ruff format --check src tests
  uv run ruff check src tests
  uv run pytest -q
  ```

- [ ] Frontend static and unit checks:

  ```bash
  cd frontend
  npx tsc --noEmit
  npm run lint
  npx playwright test --config playwright.unit.config.ts
  npm run build
  ```

- [ ] Migration and fresh-schema parity:

  ```bash
  cd backend
  uv run pytest -q tests/test_local_ai_models.py tests/test_local_ai_migrations.py
  ```

  Also run the repository's migration CI procedure against a disposable PostgreSQL database. Do not point it at the development or user-data database.

- [ ] Strict-local privacy checks:

  ```bash
  cd backend
  uv run pytest -q \
    tests/test_strict_local_egress.py \
    tests/test_ocr_egress.py \
    tests/test_local_ai_log_privacy.py
  ```

If exact test filenames drift during implementation, locate the canonical equivalents with `rg --files backend/tests` and record the substitution in the handoff. Never silently omit a gate.

## Task 6: Run the attested offline release gate

This task is separate from the deterministic suite because it may require the retained Apple model pack and hardware profile.

- [ ] Verify the active manifest lock, runtime receipt, model root, and worker command belong to the same worktree and target profile. Do not infer availability from an empty `main/backend/data/local-ai/models` directory; inspect retained worktree roots first.

- [ ] Run the plan's documented offline pack verification, synthetic fidelity, and three-run benchmark commands. Do not enable live-cloud or private-fidelity flags.

- [ ] Generate fresh release evidence. Confirm every artifact names the new manifest digest and worker-bundle digest, and that no output contains paths or clinical content.

- [ ] Record hardware/model-backed results separately from ordinary CI. A deterministic gate passing does not imply hardware evidence was regenerated.

## Task 7: Final review and handoff

- [ ] Run `git diff --check` and confirm the integration worktree is clean except for intentionally uncommitted release evidence.

- [ ] Review public-facing prose with the `humanizer` skill. Preserve exact security terms, endpoint names, environment variables, stable error codes, digests, and command lines.

- [ ] Prepare a finding-to-test matrix, exact commit SHAs, commands run, results, skipped gated checks, and artifact paths. State explicitly whether hardware/model-backed release evidence was regenerated.

- [ ] Ask for separate authorization before committing, pushing, opening pull requests, merging, or deploying. Do not turn approval to implement into approval to publish or deploy.

## Completion criteria

- Tracks A, B, and C are independently reviewable and have no cross-track implementation leakage.
- Track D contains the accepted Track B commit as an ancestor and binds validation plus every spawn to the corrected worker identity.
- Both cancellation entry points leave paired ingestion rows in a coherent durable state.
- Reloaded retryable failures and compacted upload results remain correctly identifiable in the UI.
- Operator authorization fails closed without changing user-owned job permissions.
- Backend, worker, frontend, migration, fresh-schema, egress, and log-privacy gates pass with no unexpected skips.
- Fresh runtime/release evidence is bound to the v2 manifest and worker bundle, or the exact unmet hardware/model prerequisite is reported without claiming release completion.
