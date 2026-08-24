# Strict-Local E2E Runtime-Identity Harness Repair Implementation Plan

**Status:** Approved and executed; verification complete, with implementation
uncommitted pending root publication review

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by task.
> Give each task to a fresh bounded implementer, then use fresh spec and quality
> reviewers. Agents must not commit.

**Goal:** Restore all non-private, non-pack-required local-only Playwright cases
by separating egress denial from strict-pack attestation while preserving the
production fail-closed runtime-identity boundary.

**Architecture:** Keep the account's `validated_strict_local` preference and the
legacy-v1 manifest as exact negative-admission coverage. Use fixed or
content-only synthetic FHIR helpers with an explicit per-request
`cloud_assisted` mode for deterministic parser setup, run real background dedup
with provider construction guarded, and use an inert startup sentinel that is
never represented or executed as a worker. Model-backed summary tests run only
with a real attested worker/pack; ordinary local-only runs retain the seven
non-execution summary UI cases and skip the three named pack-required cases.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async, PostgreSQL, pytest,
Next.js 16, TypeScript, Playwright, macOS sandbox/socket denial, Markdown.

## Global constraints

- Work only on branch `codex/strict-local-e2e-runtime-identity`.
- Baseline is `origin/main` at
  `32075826694e1a24bfb36699fb14374b7916b29a`.
- Latest revised design commit is
  `c85eb929482b3bcb80da610990f0fb61ea19e775`.
- Do not change production services, APIs, models, configuration, migrations,
  DDL, manifests, catalogs, releases, artifacts, workers, packs, or
  runtime-attestation code.
- Do not weaken, bypass, swallow, or relabel `runtime_identity_required`.
  Legacy-v1 strict upload and summary admission must keep failing closed.
- Positive structured-fixture uploads use only explicit `cloud_assisted`.
  `prompt_only` is not an ingestion mode. Do not change the account default or
  globally relabel strict requests.
- Positive files are limited to tracked synthetic FHIR, generated pagination
  FHIR JSON, and backend synthetic CDA test fixtures. Do not read or upload a
  private fixture.
- Force `REAL_MEDICAL_FIXTURES_DIR`, every provider credential/project field,
  and the real-pack execution gate empty in the local-only profile.
- Force `LLM_PROVIDER=gemini` and every operation-specific `LLM_*_PROVIDER`
  override empty. Do not inherit credential-free Ollama or LM Studio routing;
  the socket guard intentionally permits loopback.
- Use a fixed synthetic 64-hex-character `DATABASE_ENCRYPTION_KEY` only with
  `APP_ENV=test`. Never inherit or log a shell key.
- Every database/test shell block uses `set -euo pipefail`. It calls `createdb`
  without a preflight drop, records ownership immediately after success, and
  drops only while that invocation's ownership flag is set. A pre-existing
  database must make the command fail unchanged.
- Every Playwright invocation receives separately task-created
  `E2E_RUNTIME_ROOT` and `E2E_OUTPUT_ROOT` directories. Reject symlinked
  parents/roots, validate real paths, owner, type, and mode, capture device and
  inode, and revalidate the captured identity before any recursive deletion.
  Never chmod, replace, or recursively delete `frontend/test-results`.
- Keep backend socket denial, the macOS Next.js sandbox, browser closed proxy,
  service-worker blocking, offline flags, and telemetry denial active.
- `/usr/bin/false` is only a startup-valid non-worker sentinel. Do not call it a
  runnable or attested worker, and do not spawn it.
- Do not create, download, validate, promote, modify, or describe as verified a
  model pack, v2 release manifest, receipt, benchmark, fidelity result,
  artifact, or deployment.
- Preserve content-free policy errors, owner scoping, encryption at rest,
  immutable artifacts, migration/create-all parity, and runtime-attestation
  semantics.
- Preserve the original ignored
  `frontend/test-results/phase1-full-results.json` and its 15 traces. Do not
  stage generated traces, results, databases, upload bytes, caches, keys, or
  runtime artifacts.
- Audit ignored paths before and after work. Exclude only the explicitly named
  dependency trees `frontend/node_modules/` and `backend/.venv/`; inventory
  test results, runtime/execution/report roots, Python/pytest caches, Ruff
  caches, TypeScript incremental caches, Next output, backend data, and
  residual ignored paths separately.
- Do not commit behavioral/test changes, push, open a PR, merge, deploy, or
  change GitHub metadata. The plan also remains uncommitted.
- Stop for root approval if implementation needs any file outside the allowlist.

## Exact behavioral file allowlist

Harness and helpers:

- `frontend/playwright.config.ts`
- `frontend/e2e/helpers/api-client.ts`
- `frontend/e2e/strict-local-admission.spec.ts` (new)

Synthetic fixture callers and summary gating:

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

Backend regression coverage:

- `backend/tests/test_local_ai_ci_workflows.py`
- `backend/tests/test_local_ai_model_manager.py`
- `backend/tests/test_processing_mode_snapshot.py`
- `backend/tests/test_upload.py`

Public documentation:

- `docs/operations-strict-local-ai.md`

Planning documents:

- `docs/superpowers/specs/2026-08-23-strict-local-e2e-runtime-identity-design.md`
- `docs/superpowers/plans/2026-08-23-strict-local-e2e-runtime-identity.md`

No backend `app/` path, migration, manifest, release, catalog, worker, pack, or
artifact path is allowed.

## Agent and review protocol

Before Task 1, the root reads and invokes
`superpowers:subagent-driven-development` and
`superpowers:test-driven-development`. For every task:

1. Spawn a fresh implementer with only that task, the revised design, this plan,
   and the global constraints.
2. The implementer uses `apply_patch`, runs the stated RED/GREEN commands, does
   not commit, and returns exact output and changed paths.
3. Spawn a fresh spec reviewer. Resolve every finding before quality review.
4. Spawn a fresh quality reviewer. Resolve every finding and rerun focused
   tests.
5. The root runs
   `git diff --name-only c85eb929482b3bcb80da610990f0fb61ea19e775`
   and
   `git ls-files --others --exclude-standard`, then rejects tracked or
   untracked scope outside the allowlist.

---

### Task 1: Make the negative profile startup-safe and state-isolated

**Files:**

- Modify: `backend/tests/test_local_ai_ci_workflows.py`
- Modify: `backend/tests/test_local_ai_model_manager.py`
- Modify: `frontend/playwright.config.ts`
- Create: `frontend/e2e/strict-local-admission.spec.ts`

**Interfaces:**

- Consumes: `E2E_LOCAL_ONLY`, `E2E_DATABASE_URL`, existing loopback database
  validation, legacy-v1 manifest paths, and `LocalModelManager.start()`.
- Produces: required absolute `E2E_RUNTIME_ROOT` and `E2E_OUTPUT_ROOT`; fixed
  upload/temp/scratch/model children; `/usr/bin/false` plus an explicit
  non-worker project directory; an empty `E2E_ATTESTED_STRICT_PACK` gate; and a
  per-run Playwright output directory that cannot clear Phase 1 evidence.
- Preserves: `LOCAL_AI_ENABLED=true` and legacy-v1 paths so strict admission
  returns the exact current policy errors.
- Adds: a profile-level health regression whose test cannot begin unless the
  configured backend completes application lifespan startup.

- [ ] **Step 0: Capture the bounded ignored-artifact baseline**

Run this once from the worktree root before any behavioral edit. The fixed
task-specific audit path must be absent; do not delete or reuse an existing
path. `--ignored --exclude-standard` enumerates ignored files, after which only
the two explicit dependency-tree prefixes are removed:

```bash
set -euo pipefail
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
test ! -e "$task_audit_root" && test ! -L "$task_audit_root"
test ! -e "$task_audit_identity_file" && test ! -L "$task_audit_identity_file"
umask 077
mkdir -m 700 -- "$task_audit_root"
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test "$(stat -f '%u' "$task_audit_root")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_root")" = 700
test "$(cd -P -- "$task_audit_root" && pwd -P)" = "$task_audit_root"
task_audit_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")"
printf '%s\n' "$task_audit_identity" > "$task_audit_identity_file"
test -f "$task_audit_identity_file" && test ! -L "$task_audit_identity_file"
test "$(stat -f '%u' "$task_audit_identity_file")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_identity_file")" = 600
test ! -e backend/.venv && test ! -L backend/.venv
printf 'absent-before-behavior\n' > "$task_audit_root/backend-venv.before"

capture_ignored_inventory() {
  task_inventory_suffix="$1"
  task_inventory_all="$task_audit_root/ignored.$task_inventory_suffix"
  git -c core.quotepath=false ls-files \
    --others --ignored --exclude-standard |
    awk '
      !/^frontend\/node_modules\// &&
      !/^backend\/\.venv(\/|$)/ { print }
    ' | LC_ALL=C sort > "$task_inventory_all"
  for task_inventory_class in \
    phase1 e2e-generated pytest-cache python-cache ruff-cache \
    typescript-cache next-output backend-data residual
  do
    : > "$task_audit_root/$task_inventory_class.$task_inventory_suffix"
  done
  while IFS= read -r task_ignored_path; do
    case "$task_ignored_path" in
      frontend/test-results/runtime/*|frontend/test-results/executions/*)
        task_inventory_class=e2e-generated
        ;;
      frontend/test-results/*)
        task_inventory_class=phase1
        ;;
      */.pytest_cache/*)
        task_inventory_class=pytest-cache
        ;;
      */__pycache__/*|*.pyc)
        task_inventory_class=python-cache
        ;;
      backend/.ruff_cache/*)
        task_inventory_class=ruff-cache
        ;;
      frontend/tsconfig.tsbuildinfo)
        task_inventory_class=typescript-cache
        ;;
      frontend/.next/*|frontend/next-env.d.ts)
        task_inventory_class=next-output
        ;;
      backend/data/*)
        task_inventory_class=backend-data
        ;;
      *)
        task_inventory_class=residual
        ;;
    esac
    printf '%s\n' "$task_ignored_path" \
      >> "$task_audit_root/$task_inventory_class.$task_inventory_suffix"
  done < "$task_inventory_all"
}

capture_ignored_inventory before
for task_generated_root in \
  frontend/test-results/runtime frontend/test-results/executions
do
  if [ -e "$task_generated_root" ] || [ -L "$task_generated_root" ]; then
    find "$task_generated_root" -print
  fi
done | LC_ALL=C sort > "$task_audit_root/e2e-tree.before"
find frontend/test-results \
  \( -path frontend/test-results/runtime \
     -o -path frontend/test-results/executions \) -prune \
  -o -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 > "$task_audit_root/phase1-files.before.sha256"
find backend/data -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 > "$task_audit_root/backend-data.before.sha256"
test "$(shasum -a 256 frontend/test-results/phase1-full-results.json | awk '{print $1}')" = \
  ef23b233169bbafe4735169fa031ba7a1dcb4630378f2035e06fa9ef6db47eb0
task_original_trace_count="$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort | wc -l | tr -d ' ')"
test "$task_original_trace_count" -eq 15
test "$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort |
  while IFS= read -r task_trace_path; do
    shasum -a 256 "$task_trace_path"
  done | shasum -a 256 | awk '{print $1}')" = \
  1c8501bf87e94eb874b316271facbbf9af087e84b967ff4ad6bd764ef8a8c827
printf 'audit_root=%s identity=%s\n' "$task_audit_root" "$task_audit_identity"
```

Expected: the command exits 0 and leaves only the owner-only audit directory
and identity file in `/private/tmp`. The root records their exact paths in the
task handoff. It independently proves the Phase-1 diagnostic dependency link is
absent before behavioral work, without using historical ownership as cleanup
authority. `residual.before` is not assumed empty; it is the exact baseline that
must remain unchanged. No repository file changes.

- [ ] **Step 1: Add the failing static profile contract**

Add to `backend/tests/test_local_ai_ci_workflows.py`:

```python
def test_local_only_playwright_profile_is_network_denial_not_attested_worker() -> None:
    content = (REPOSITORY_ROOT / "frontend/playwright.config.ts").read_text(
        encoding="utf-8"
    )
    local_profile = content.split("if (localOnly) {", 1)[1].split(
        "} else {", 1
    )[0]

    assert 'APP_ENV: "test"' in local_profile
    assert 'DATABASE_ENCRYPTION_KEY: "00".repeat(32)' in local_profile
    assert 'REAL_MEDICAL_FIXTURES_DIR: ""' in local_profile
    assert 'E2E_ATTESTED_STRICT_PACK: ""' in local_profile
    assert 'LOCAL_AI_WORKER_COMMAND: "/usr/bin/false"' in local_profile
    assert "LOCAL_AI_WORKER_PROJECT_DIR: runtimePaths.nonWorkerProject" in local_profile
    assert "UPLOAD_DIR: runtimePaths.uploads" in local_profile
    assert "TEMP_EXTRACT_DIR: runtimePaths.tempExtract" in local_profile
    assert "LOCAL_AI_SCRATCH_DIR: runtimePaths.scratch" in local_profile
    assert "LOCAL_AI_MODEL_DIR: runtimePaths.models" in local_profile
    assert "E2E_RUNTIME_ROOT" in content
    assert "E2E_OUTPUT_ROOT" in content
    assert "fs.lstatSync" in content
    assert "fs.realpathSync.native" in content
    assert "stats.dev" in content
    assert "stats.ino" in content
    assert "stats.uid" in content
    assert "stats.mode & 0o777" in content
    assert "path.dirname(rootIdentity.realPath) !== runtimeParentIdentity.realPath" in content
    assert 'path.join(outputIdentity.realPath, "artifacts")' in content
    assert "outputArtifactsIdentity.realPath" in content
    assert "e2e_local_ai_worker.py" not in local_profile
    assert "apple-m4-16gb-v1.lock.json" in local_profile
    assert "apple-m4-16gb-v1.release.json" in local_profile
    assert "backend/artifacts/local-ai-benchmark.json" in local_profile
    assert "backend/artifacts/local-ai-fidelity.json" in local_profile
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "VERTEX_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        assert f'{name}: ""' in local_profile
    assert 'LLM_PROVIDER: "gemini"' in local_profile
    for name in (
        "LLM_SUMMARY_PROVIDER",
        "LLM_SECTION_PROVIDER",
        "LLM_DEDUP_PROVIDER",
        "LLM_EXTRACTION_PROVIDER",
    ):
        assert f'{name}: ""' in local_profile
```

- [ ] **Step 2: Run the static test and observe RED**

Run:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_ci_workflows.py::test_local_only_playwright_profile_is_network_denial_not_attested_worker
```

Expected: FAIL because the current profile inherits the private-fixture root,
uses repository data paths, lacks a fixed encryption key/runtime root, and
advertises the argument-bearing fake wrapper.

- [ ] **Step 3: Add the startup-focused manager characterization**

Add beside the manager startup tests in
`backend/tests/test_local_ai_model_manager.py`:

```python
@pytest.mark.asyncio
async def test_start_with_inert_executable_does_not_spawn_worker(
    worker_home: Path,
) -> None:
    manager = LocalModelManager(
        ["/usr/bin/false"],
        worker_home=worker_home,
        worker_project_dir=worker_home.parent / "non-worker-project",
    )

    try:
        await manager.start()

        assert manager._started is True
        assert manager._worker_command == ("/usr/bin/false",)
        assert manager._worker_project_dir == worker_home.parent / "non-worker-project"
        assert manager.active_pid is None
        assert manager.metrics.live_processes == 0
        assert manager.metrics.pids_started == []
        assert manager.metrics.roles_started == []
        assert list(worker_home.glob(".worker-pycache-*")) == []
    finally:
        await manager.stop()
```

Run it before changing the profile:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_model_manager.py::test_start_with_inert_executable_does_not_spawn_worker
```

Expected: PASS. This is a startup characterization, not evidence that the
sentinel can execute the worker protocol.

- [ ] **Step 4: Implement the minimal profile contract**

In `frontend/playwright.config.ts`, add
`let localOnlyOutputDir: string | undefined;` before the profile branch. Remove
the local-only `.env.test.local` load. Add this local helper before the profile
branch; it rejects symlinks, wrong types, wrong owner, and non-`0700`
task-created roots while retaining device/inode identity for validation:

```ts
type DirectoryIdentity = {
  realPath: string;
  dev: number;
  ino: number;
  uid: number;
  mode: number;
};

function ownedDirectoryIdentity(
  rawPath: string,
  label: string,
  taskCreated: boolean
): DirectoryIdentity {
  if (!path.isAbsolute(rawPath)) {
    throw new Error(`${label} must be absolute.`);
  }
  const lexicalPath = path.resolve(rawPath);
  const linkStats = fs.lstatSync(lexicalPath);
  if (linkStats.isSymbolicLink() || !linkStats.isDirectory()) {
    throw new Error(`${label} must be a non-symlink directory.`);
  }
  const realPath = fs.realpathSync.native(lexicalPath);
  const stats = fs.statSync(realPath);
  if (!stats.isDirectory() || typeof process.getuid !== "function") {
    throw new Error(`${label} must be an owned directory.`);
  }
  const mode = stats.mode & 0o777;
  const unsafeExistingMode = !taskCreated && (mode & 0o022) !== 0;
  if (
    stats.uid !== process.getuid() ||
    (taskCreated && mode !== 0o700) ||
    unsafeExistingMode
  ) {
    throw new Error(`${label} has unsafe ownership or mode.`);
  }
  return {
    realPath,
    dev: stats.dev,
    ino: stats.ino,
    uid: stats.uid,
    mode,
  };
}
```

After `requireDedicatedLoopbackDatabase`, validate the real worktree/evidence
chain and both shell-created roots. Require each approved parent and root to be
an exact real child, not merely a lexical prefix:

```ts
  const configuredRuntimeRoot = process.env.E2E_RUNTIME_ROOT;
  const configuredOutputRoot = process.env.E2E_OUTPUT_ROOT;
  if (!configuredRuntimeRoot || !configuredOutputRoot) {
    throw new Error("E2E runtime and output roots are required.");
  }
  const repoIdentity = ownedDirectoryIdentity(repoRoot, "repository root", false);
  if (repoIdentity.realPath !== repoRoot) {
    throw new Error("Repository root must not traverse a symlink.");
  }
  const evidenceIdentity = ownedDirectoryIdentity(
    path.resolve(__dirname, "test-results"),
    "Playwright evidence parent",
    false
  );
  if (path.dirname(evidenceIdentity.realPath) !== path.join(repoIdentity.realPath, "frontend")) {
    throw new Error("Playwright evidence parent escaped the worktree.");
  }
  const runtimeParentIdentity = ownedDirectoryIdentity(
    path.join(evidenceIdentity.realPath, "runtime"),
    "runtime parent",
    false
  );
  const outputParentIdentity = ownedDirectoryIdentity(
    path.join(evidenceIdentity.realPath, "executions"),
    "execution parent",
    false
  );
  if (
    path.dirname(runtimeParentIdentity.realPath) !== evidenceIdentity.realPath ||
    path.dirname(outputParentIdentity.realPath) !== evidenceIdentity.realPath
  ) {
    throw new Error("E2E root parent escaped the evidence directory.");
  }
  const rootIdentity = ownedDirectoryIdentity(
    configuredRuntimeRoot,
    "E2E_RUNTIME_ROOT",
    true
  );
  const outputIdentity = ownedDirectoryIdentity(
    configuredOutputRoot,
    "E2E_OUTPUT_ROOT",
    true
  );
  if (path.dirname(rootIdentity.realPath) !== runtimeParentIdentity.realPath) {
    throw new Error("E2E_RUNTIME_ROOT is not an exact runtime-parent child.");
  }
  if (path.dirname(outputIdentity.realPath) !== outputParentIdentity.realPath) {
    throw new Error("E2E_OUTPUT_ROOT is not an exact execution-parent child.");
  }
  if (rootIdentity.dev === outputIdentity.dev && rootIdentity.ino === outputIdentity.ino) {
    throw new Error("E2E runtime and output roots must be distinct.");
  }
  const runtimePaths = {
    uploads: path.join(rootIdentity.realPath, "uploads"),
    tempExtract: path.join(rootIdentity.realPath, "temp-extract"),
    scratch: path.join(rootIdentity.realPath, "scratch"),
    models: path.join(rootIdentity.realPath, "models"),
    nonWorkerProject: path.join(rootIdentity.realPath, "non-worker-project"),
  };
  for (const directory of Object.values(runtimePaths)) {
    if (!fs.existsSync(directory)) fs.mkdirSync(directory, { mode: 0o700 });
    const childIdentity = ownedDirectoryIdentity(directory, "runtime child", true);
    if (path.dirname(childIdentity.realPath) !== rootIdentity.realPath) {
      throw new Error("Runtime child escaped E2E_RUNTIME_ROOT.");
    }
  }
  const outputArtifacts = path.join(outputIdentity.realPath, "artifacts");
  if (!fs.existsSync(outputArtifacts)) {
    fs.mkdirSync(outputArtifacts, { mode: 0o700 });
  }
  const outputArtifactsIdentity = ownedDirectoryIdentity(
    outputArtifacts,
    "Playwright output directory",
    true
  );
  if (path.dirname(outputArtifactsIdentity.realPath) !== outputIdentity.realPath) {
    throw new Error("Playwright output directory escaped E2E_OUTPUT_ROOT.");
  }
  localOnlyOutputDir = outputArtifactsIdentity.realPath;
```

Replace only the local profile assignments with these values while retaining
all v1/offline/network-denial fields:

```ts
    DATABASE_ENCRYPTION_KEY: "00".repeat(32),
    REAL_MEDICAL_FIXTURES_DIR: "",
    E2E_ATTESTED_STRICT_PACK: "",
    UPLOAD_DIR: runtimePaths.uploads,
    TEMP_EXTRACT_DIR: runtimePaths.tempExtract,
    LOCAL_AI_MODEL_DIR: runtimePaths.models,
    LOCAL_AI_SCRATCH_DIR: runtimePaths.scratch,
    LOCAL_AI_WORKER_COMMAND: "/usr/bin/false",
    LOCAL_AI_WORKER_PROJECT_DIR: runtimePaths.nonWorkerProject,
    LOCAL_AI_MANIFEST_PATH: path.resolve(
      repoRoot,
      "backend/app/model_manifests/apple-m4-16gb-v1.lock.json"
    ),
    LOCAL_AI_RELEASE_EVIDENCE_PATH: path.resolve(
      repoRoot,
      "backend/app/model_manifests/apple-m4-16gb-v1.release.json"
    ),
    LOCAL_AI_BENCHMARK_PATH: path.resolve(
      repoRoot,
      "backend/artifacts/local-ai-benchmark.json"
    ),
    LOCAL_AI_FIDELITY_PATH: path.resolve(
      repoRoot,
      "backend/artifacts/local-ai-fidelity.json"
    ),
    LLM_PROVIDER: "gemini",
    LLM_SUMMARY_PROVIDER: "",
    LLM_SECTION_PROVIDER: "",
    LLM_DEDUP_PROVIDER: "",
    LLM_EXTRACTION_PROVIDER: "",
```

Keep the comment explicit: this sentinel exists only so lifespan startup can
normalize a command; legacy-v1 admission rejects before runtime identity or
spawn. Do not call it an E2E worker.

In `defineConfig`, add:

```ts
  outputDir: localOnlyOutputDir,
```

Each local-only config evaluation creates or revalidates
`<E2E_OUTPUT_ROOT>/artifacts` as an exact non-symlink `0700` child and captures
its filesystem identity immediately before handing its real path to Playwright.
Playwright clears only that child, not the task-owned output root or parent
`frontend/test-results`. The shell later revalidates the output root and its
approved parent before cleanup or retention. The output root is not inside
`E2E_RUNTIME_ROOT`, so runtime cleanup cannot delete retained traces or the
token-bound list/JSON files. Do not call `chmod` on an existing parent or
recursively chmod `frontend/test-results`.

- [ ] **Step 5: Run GREEN and review the exact diff**

Run:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_ci_workflows.py::test_local_only_playwright_profile_is_network_denial_not_attested_worker \
  tests/test_local_ai_model_manager.py::test_start_with_inert_executable_does_not_spawn_worker
cd ..
git diff --check
git diff -- frontend/playwright.config.ts \
  backend/tests/test_local_ai_ci_workflows.py \
  backend/tests/test_local_ai_model_manager.py
```

Expected: `2 passed`; no production file, provider value, private path, real
key, v2 manifest, receipt, release evidence, or model result appears.

- [ ] **Step 6: Add and run the profile-level startup regression**

Create `frontend/e2e/strict-local-admission.spec.ts` with:

```ts
import * as fs from "node:fs";
import * as path from "node:path";
import { test, expect } from "@playwright/test";

test("local-only non-worker profile starts backend without attesting a worker", async () => {
  test.skip(
    process.env.E2E_LOCAL_ONLY !== "1",
    "startup sentinel contract requires the local-only profile"
  );
  const runtimeRoot = process.env.E2E_RUNTIME_ROOT;
  const outputRoot = process.env.E2E_OUTPUT_ROOT;
  const workerProject = process.env.LOCAL_AI_WORKER_PROJECT_DIR;
  if (!runtimeRoot || !outputRoot || !workerProject) {
    throw new Error("Local-only task roots and sentinel project must be configured");
  }
  const runtimeReal = fs.realpathSync.native(runtimeRoot);
  const outputReal = fs.realpathSync.native(outputRoot);
  const relativeProject = path.relative(runtimeReal, fs.realpathSync.native(workerProject));

  expect(process.env.LOCAL_AI_WORKER_COMMAND).toBe("/usr/bin/false");
  expect(process.env.E2E_ATTESTED_STRICT_PACK).toBe("");
  expect(process.env.LOCAL_AI_MANIFEST_PATH).toContain("apple-m4-16gb-v1.lock.json");
  expect(process.env.LOCAL_AI_RELEASE_EVIDENCE_PATH).toContain(
    "apple-m4-16gb-v1.release.json"
  );
  expect(process.env.LOCAL_AI_BENCHMARK_PATH).toContain(
    "backend/artifacts/local-ai-benchmark.json"
  );
  expect(process.env.LOCAL_AI_FIDELITY_PATH).toContain(
    "backend/artifacts/local-ai-fidelity.json"
  );
  for (const root of [runtimeRoot, outputRoot]) {
    const linkStats = fs.lstatSync(root);
    const stats = fs.statSync(root);
    expect(linkStats.isSymbolicLink()).toBe(false);
    expect(stats.isDirectory()).toBe(true);
    expect(stats.uid).toBe(process.getuid?.());
    expect(stats.mode & 0o777).toBe(0o700);
  }
  expect(runtimeReal).not.toBe(outputReal);
  expect(relativeProject).not.toBe("");
  expect(relativeProject.startsWith("..")).toBe(false);
  expect(path.isAbsolute(relativeProject)).toBe(false);

  const response = await fetch("http://127.0.0.1:8000/api/v1/health");
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ status: "healthy", version: "0.1.0" });
});
```

Run it on an explicit fresh DB and owned runtime. The command's link flag is the
only authority to unlink its dependency link:

```bash
set -euo pipefail
task_worktree_root="$PWD"
task_worktree_real="$(pwd -P)"
test "$task_worktree_root" = "$task_worktree_real"
task_database=medtimeline_df11_strict_identity_startup_e2e
task_created_database=0
task_backend_link="$task_worktree_root/backend/.venv"
task_dependency_source=/Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv
task_created_backend_link=0
task_backend_link_identity=""
task_backend_link_target=""
task_evidence_parent="$task_worktree_root/frontend/test-results"
task_runtime_parent="$task_evidence_parent/runtime"
task_output_parent="$task_evidence_parent/executions"
task_created_runtime_parent=0
task_created_output_parent=0
task_runtime_root=""
task_output_root=""
task_runtime_parent_real=""
task_output_parent_real=""
task_runtime_parent_identity=""
task_output_parent_identity=""
task_runtime_identity=""
task_output_identity=""

safe_remove_owned_directory() {
  task_remove_target="$1"
  task_remove_parent="$2"
  task_remove_parent_real="$3"
  task_expected_parent_identity="$4"
  task_expected_target_identity="$5"
  test -d "$task_remove_parent" && test ! -L "$task_remove_parent" || return 1
  test "$(cd -P -- "$task_remove_parent" && pwd -P)" = \
    "$task_remove_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_remove_parent")" = \
    "$task_expected_parent_identity" || return 1
  test -d "$task_remove_target" && test ! -L "$task_remove_target" || return 1
  task_remove_real="$(cd -P -- "$task_remove_target" && pwd -P)" || return 1
  test "$(dirname "$task_remove_real")" = "$task_remove_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_remove_target")" = \
    "$task_expected_target_identity" || return 1
  rm -rf -- "$task_remove_target"
}

cleanup_task1_startup() {
  task_cleanup_original_status=$?
  trap - EXIT INT TERM
  set +e
  task_cleanup_status=0
  if [ -n "$task_output_root" ]; then
    safe_remove_owned_directory \
      "$task_output_root" "$task_output_parent" "$task_output_parent_real" \
      "$task_output_parent_identity" "$task_output_identity" || task_cleanup_status=1
  fi
  if [ -n "$task_runtime_root" ]; then
    safe_remove_owned_directory \
      "$task_runtime_root" "$task_runtime_parent" "$task_runtime_parent_real" \
      "$task_runtime_parent_identity" "$task_runtime_identity" || task_cleanup_status=1
  fi
  if [ "$task_created_backend_link" -eq 1 ]; then
    if [ -L "$task_backend_link" ] && \
       [ "$(stat -f '%d:%i:%u:%HT' "$task_backend_link")" = "$task_backend_link_identity" ] && \
       [ "$(readlink "$task_backend_link")" = "$task_backend_link_target" ]; then
      unlink "$task_backend_link" || task_cleanup_status=1
    else
      echo "Refusing to unlink changed backend dependency link" >&2
      task_cleanup_status=1
    fi
  fi
  if [ "$task_created_database" -eq 1 ]; then
    if dropdb -h 127.0.0.1 -p 5432 "$task_database"; then
      task_created_database=0
    else
      task_cleanup_status=1
    fi
  fi
  if [ "$task_cleanup_original_status" -ne 0 ]; then
    exit "$task_cleanup_original_status"
  fi
  exit "$task_cleanup_status"
}
trap cleanup_task1_startup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test -d "$task_evidence_parent" && test ! -L "$task_evidence_parent"
test "$(stat -f '%u' "$task_evidence_parent")" -eq "$(id -u)"
test "$(cd -P -- "$task_evidence_parent" && pwd -P)" = \
  "$task_worktree_real/frontend/test-results"
if [ ! -e "$task_runtime_parent" ] && [ ! -L "$task_runtime_parent" ]; then
  mkdir -m 700 -- "$task_runtime_parent"
  task_created_runtime_parent=1
fi
if [ ! -e "$task_output_parent" ] && [ ! -L "$task_output_parent" ]; then
  mkdir -m 700 -- "$task_output_parent"
  task_created_output_parent=1
fi
for task_parent in "$task_runtime_parent" "$task_output_parent"; do
  test -d "$task_parent" && test ! -L "$task_parent"
  test "$(stat -f '%u' "$task_parent")" -eq "$(id -u)"
done
if [ "$task_created_runtime_parent" -eq 1 ]; then
  test "$(stat -f '%Lp' "$task_runtime_parent")" = 700
fi
if [ "$task_created_output_parent" -eq 1 ]; then
  test "$(stat -f '%Lp' "$task_output_parent")" = 700
fi
task_runtime_parent_real="$(cd -P -- "$task_runtime_parent" && pwd -P)"
task_output_parent_real="$(cd -P -- "$task_output_parent" && pwd -P)"
test "$(dirname "$task_runtime_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
test "$(dirname "$task_output_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
task_runtime_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")"
task_output_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")"

umask 077
task_runtime_root="$(mktemp -d "$task_runtime_parent/startup.XXXXXX")"
task_run_token="$(basename "$task_runtime_root")"
task_output_root="$task_output_parent/$task_run_token"
mkdir -m 700 -- "$task_output_root"
for task_root in "$task_runtime_root" "$task_output_root"; do
  test -d "$task_root" && test ! -L "$task_root"
  test "$(stat -f '%u' "$task_root")" -eq "$(id -u)"
  test "$(stat -f '%Lp' "$task_root")" = 700
done
test "$(dirname "$(cd -P -- "$task_runtime_root" && pwd -P)")" = \
  "$task_runtime_parent_real"
test "$(dirname "$(cd -P -- "$task_output_root" && pwd -P)")" = \
  "$task_output_parent_real"
task_runtime_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")"
task_output_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")"

if [ ! -e "$task_backend_link" ] && [ ! -L "$task_backend_link" ]; then
  test -x "$task_dependency_source/bin/python"
  ln -s "$task_dependency_source" "$task_backend_link"
  task_created_backend_link=1
  task_backend_link_identity="$(stat -f '%d:%i:%u:%HT' "$task_backend_link")"
  task_backend_link_target="$(readlink "$task_backend_link")"
fi
test -x "$task_backend_link/bin/python"

createdb -h 127.0.0.1 -p 5432 "$task_database"
task_created_database=1
cd frontend
env -u DATABASE_URL -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_database" \
  ./node_modules/.bin/playwright test \
  e2e/strict-local-admission.spec.ts \
  --workers=1 --trace=retain-on-failure
```

Expected: one pass. Playwright's web-server gate proves the real backend reached
health under the exact production profile. The unit characterization used the
test module's unisolated subclass and did not touch `settings.local_ai_model_dir`;
the profile test did not instantiate that subclass. Neither path spawns the
sentinel or a production/model worker, and neither produces attestation.

- [ ] **Step 7: Complete spec and quality review**

The spec reviewer confirms startup succeeds without spawning the sentinel or a
production/model worker and that inherited
`UPLOAD_DIR`/`TEMP_EXTRACT_DIR` are overwritten. The quality reviewer confirms
the runtime/output containment checks reject the parent itself and paths
outside it, the validated `artifacts` child is the only Playwright-cleared
directory, and Playwright cannot clear the original Phase 1 output. Leave the
task uncommitted.

---

### Task 2: Characterize exact legacy upload and summary admission

**Files:**

- Modify: `backend/tests/test_processing_mode_snapshot.py`

**Interfaces:**

- Consumes: existing `_manifest()`, `/api/v1/upload`,
  `/api/v1/summary/generate`, `resolve_new_*_snapshot`, and SQLAlchemy models.
- Produces:
  `test_structured_upload_rejects_legacy_strict_runtime_without_request_side_effects`
  and
  `test_summary_endpoint_rejects_legacy_strict_runtime_before_prompt_or_job`.
- Preserves: production HTTP mappings: upload is exact 409; summary is exact
  400.

- [ ] **Step 1: Add content-free request-state helpers**

Add imports for `hashlib`, `os`, `stat`, SQLAlchemy `func`, `AISummaryPrompt`,
`Patient`, `HealthRecord`, and `create_test_patient`. Add:

```python
def _configure_legacy_v1_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[Path]:
    import app.services.local_ai.processing_snapshot as snapshot_module

    legacy_manifest = replace(
        _manifest(),
        schema_version=1,
        pack_revision="apple-m4-16gb-v1",
        runtime={"name": "mlx-vlm", "version": "0.5.0"},
    )
    manifest_path = tmp_path / "legacy-manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(legacy_manifest), sort_keys=True),
        encoding="utf-8",
    )
    release_calls: list[Path] = []
    monkeypatch.setattr(snapshot_module.settings, "local_ai_enabled", True)
    monkeypatch.setattr(
        snapshot_module.settings,
        "local_ai_manifest_path",
        str(manifest_path),
    )
    monkeypatch.setattr(
        snapshot_module,
        "load_release_evidence",
        lambda path, **_kwargs: release_calls.append(Path(path)),
    )
    return release_calls


async def _owned_ingestion_counts(
    db_session: AsyncSession,
    user_id: UUID,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in (UploadedFile, LocalAIJob, HealthRecord, Patient):
        counts[model.__tablename__] = (
            await db_session.execute(
                select(func.count()).select_from(model).where(model.user_id == user_id)
            )
        ).scalar_one()
    return counts


def _content_free_storage_snapshot(root: Path) -> dict[str, object]:
    if not root.exists():
        return {"root_exists": False, "entries": []}
    entries: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*")):
        metadata = os.lstat(candidate)
        entry: dict[str, object] = {
            "path": candidate.relative_to(root).as_posix(),
            "type": stat.S_IFMT(metadata.st_mode),
            "size": metadata.st_size,
        }
        if stat.S_ISREG(metadata.st_mode):
            entry["sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        entries.append(entry)
    return {"root_exists": True, "entries": entries}
```

The helper reads deterministic synthetic test bytes only and returns hashes, not
raw content. It distinguishes an absent root from an empty one.

- [ ] **Step 2: Add the strict upload no-side-effects regression**

Add:

```python
@pytest.mark.asyncio
async def test_structured_upload_rejects_legacy_strict_runtime_without_request_side_effects(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator
    from app.config import settings

    headers, user_id_text = await auth_headers(
        client,
        email="legacy-strict-no-side-effects@example.com",
    )
    user_id = UUID(user_id_text)
    release_calls = _configure_legacy_v1_admission(monkeypatch, tmp_path)
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(settings, "upload_dir", str(upload_root))

    async def _unexpected_ingestion(**_kwargs: object) -> dict[str, object]:
        pytest.fail("legacy strict rejection reached ingestion")

    monkeypatch.setattr(coordinator, "ingest_file", _unexpected_ingestion)
    before_counts = await _owned_ingestion_counts(db_session, user_id)
    before_storage = _content_free_storage_snapshot(upload_root)

    response = await client.post(
        "/api/v1/upload",
        headers=headers,
        data={"processing_mode": "validated_strict_local"},
        files={
            "file": (
                "bundle.json",
                b'{"resourceType":"Bundle","entry":[]}',
                "application/fhir+json",
            )
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Strict-local worker runtime identity is required."
    }
    assert await _owned_ingestion_counts(db_session, user_id) == before_counts
    assert _content_free_storage_snapshot(upload_root) == before_storage
    assert release_calls == []
```

- [ ] **Step 3: Add the exact strict-summary endpoint regression**

Add:

```python
@pytest.mark.asyncio
async def test_summary_endpoint_rejects_legacy_strict_runtime_before_prompt_or_job(
    client,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.api.summary as summary_api

    headers, user_id_text = await auth_headers(
        client,
        email="legacy-strict-summary@example.com",
    )
    user_id = UUID(user_id_text)
    patient = await create_test_patient(db_session, user_id)
    release_calls = _configure_legacy_v1_admission(monkeypatch, tmp_path)
    enqueue_calls: list[UUID] = []

    monkeypatch.setattr(
        summary_api.local_summary_runner,
        "enqueue",
        lambda job_id: enqueue_calls.append(job_id),
    )

    async def _unexpected_generate(*_args: object, **_kwargs: object) -> object:
        pytest.fail("legacy strict summary reached provider generation")

    monkeypatch.setattr(
        "app.services.ai.summarizer.generate_summary",
        _unexpected_generate,
    )

    async def _summary_counts() -> tuple[int, int]:
        prompts = (
            await db_session.execute(
                select(func.count())
                .select_from(AISummaryPrompt)
                .where(AISummaryPrompt.user_id == user_id)
            )
        ).scalar_one()
        jobs = (
            await db_session.execute(
                select(func.count())
                .select_from(LocalAIJob)
                .where(LocalAIJob.user_id == user_id)
            )
        ).scalar_one()
        return prompts, jobs

    before = await _summary_counts()
    response = await client.post(
        "/api/v1/summary/generate",
        headers=headers,
        json={
            "patient_id": str(patient.id),
            "summary_type": "full",
            "processing_mode": "validated_strict_local",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Strict-local worker runtime identity is required."
    }
    assert await _summary_counts() == before
    assert enqueue_calls == []
    assert release_calls == []
```

- [ ] **Step 4: Run both characterizations before any harness call-site fix**

Create a fresh database on the same host/port used by the URL, then run with
`APP_ENV=test` and blank external inputs:

```bash
set -euo pipefail
task2_database=medtimeline_df11_strict_identity_admission_test
task2_created_database=0
cleanup_task2_database() {
  task2_original_status=$?
  trap - EXIT INT TERM
  set +e
  task2_cleanup_status=0
  if [ "$task2_created_database" -eq 1 ]; then
    if dropdb -h 127.0.0.1 -p 5432 "$task2_database"; then
      task2_created_database=0
    else
      task2_cleanup_status=1
    fi
  fi
  if [ "$task2_original_status" -ne 0 ]; then
    exit "$task2_original_status"
  fi
  exit "$task2_cleanup_status"
}
trap cleanup_task2_database EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
createdb -h 127.0.0.1 -p 5432 "$task2_database"
task2_created_database=1
cd backend
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task2_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_processing_mode_snapshot.py::test_structured_upload_rejects_legacy_strict_runtime_without_request_side_effects \
  tests/test_processing_mode_snapshot.py::test_summary_endpoint_rejects_legacy_strict_runtime_before_prompt_or_job
```

Expected: `2 passed`. A failure means the diagnosis is incomplete; stop rather
than editing production admission.

- [ ] **Step 5: Complete spec and quality review**

The spec reviewer checks exact 409/400 bodies, all four upload-side tables,
summary prompt/job counts, absence-sensitive storage metadata, zero release
loads, zero enqueue/provider generation, and request-scoped assertions. The
quality reviewer confirms no raw storage bytes enter assertion output. Leave the
task uncommitted.

---

### Task 3: Prove the real structured upload and dedup flow is provider-free

**Files:**

- Modify: `backend/tests/test_upload.py`

**Interfaces:**

- Consumes: tracked `sample_fhir_bundle.json`, tracked
  `synthetic_cda/DOC0001.XML`, real `schedule_dedup_background`, and
  `stop_dedup_background_tasks(cancel=False)`.
- Produces: one tracked-FHIR plus identical re-upload/idempotency proof and one
  synthetic-CDA proof, each with terminal dedup state and zero
  `llm_judge.get_provider` calls. Instrumentation retains only the fixed label
  `get_provider`, never arguments, keyword arguments, `LLMConfig`, prompts, or
  credentials.
- Preserves: the production dedup scheduler and provider routing code. Do not
  monkeypatch either scheduler or `run_upload_dedup`.

- [ ] **Step 1: Add exact provider-call and terminal-state helpers**

Add `Path`, `UUID`, `select`, `settings`, and `UploadedFile` imports. Add:

```python
def _deny_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    import app.services.dedup.llm_judge as llm_judge

    operations: list[str] = []

    def _unexpected_get_provider(
        _operation: object = None,
        _config: object = None,
    ) -> None:
        operations.append("get_provider")

    monkeypatch.setattr(llm_judge, "get_provider", _unexpected_get_provider)
    return operations


async def _owned_upload(
    db_session: AsyncSession,
    upload_id: str,
    user_id: str,
) -> UploadedFile:
    db_session.expire_all()
    return (
        await db_session.execute(
            select(UploadedFile).where(
                UploadedFile.id == UUID(upload_id),
                UploadedFile.user_id == UUID(user_id),
            )
        )
    ).scalar_one()
```

The wrapper deliberately does not raise or retain either argument. If called,
the production judge falls back after attempting to use the `None` result, and
the test's final `operations == []` assertion fails with only the fixed label in
its output. No provider config can enter a failing assertion payload.

- [ ] **Step 2: Add tracked FHIR plus identical re-upload coverage**

Add:

```python
@pytest.mark.asyncio
async def test_cloud_assisted_tracked_fhir_and_identical_reupload_finish_without_provider(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    provider_operations = _deny_provider_construction(monkeypatch)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "temp_extract_dir", str(tmp_path / "temp-extract"))
    headers, user_id = await auth_headers(
        client,
        email="provider-free-tracked-fhir@example.com",
    )
    fhir_bytes = (FIXTURES_DIR / "sample_fhir_bundle.json").read_bytes()

    try:
        first = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={
                "file": (
                    "sample_fhir_bundle.json",
                    fhir_bytes,
                    "application/fhir+json",
                )
            },
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert first.status_code == 202
    assert first.json()["records_inserted"] == 17
    first_upload = await _owned_upload(
        db_session,
        first.json()["upload_id"],
        user_id,
    )
    assert first_upload.processing_mode == "cloud_assisted"
    assert first_upload.ingestion_status == "completed"
    assert first_upload.processing_completed_at is not None
    assert first_upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }

    records_after_first = (await client.get("/api/v1/records", headers=headers)).json()[
        "total"
    ]
    try:
        second = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={
                "file": (
                    "sample_fhir_bundle.json",
                    fhir_bytes,
                    "application/fhir+json",
                )
            },
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert second.status_code == 202
    assert second.json()["records_inserted"] == 0
    second_upload = await _owned_upload(
        db_session,
        second.json()["upload_id"],
        user_id,
    )
    records_after_second = (
        await client.get("/api/v1/records", headers=headers)
    ).json()["total"]

    assert second_upload.processing_mode == "cloud_assisted"
    assert second_upload.ingestion_status == "completed"
    assert second_upload.processing_completed_at is not None
    assert second_upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }
    assert records_after_second == records_after_first
    assert provider_operations == []
    assert coordinator._dedup_tasks == set()
```

This is the exact browser-positive FHIR sequence, including the identical
re-upload used by `upload-dedup.spec.ts`.

- [ ] **Step 3: Add the backend synthetic CDA flow**

Add:

```python
@pytest.mark.asyncio
async def test_cloud_assisted_synthetic_cda_finishes_without_provider(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import app.services.ingestion.coordinator as coordinator

    provider_operations = _deny_provider_construction(monkeypatch)
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "temp_extract_dir", str(tmp_path / "temp-extract"))
    headers, user_id = await auth_headers(
        client,
        email="provider-free-synthetic-cda@example.com",
    )
    cda_path = FIXTURES_DIR / "synthetic_cda" / "DOC0001.XML"

    try:
        response = await client.post(
            "/api/v1/upload",
            headers=headers,
            data={"processing_mode": "cloud_assisted"},
            files={"file": ("DOC0001.XML", cda_path.read_bytes(), "application/xml")},
        )
    finally:
        await coordinator.stop_dedup_background_tasks(cancel=False)
    assert response.status_code == 202
    assert response.json()["records_inserted"] > 0
    upload = await _owned_upload(db_session, response.json()["upload_id"], user_id)

    assert upload.processing_mode == "cloud_assisted"
    assert upload.ingestion_status == "completed"
    assert upload.processing_completed_at is not None
    assert upload.dedup_summary == {
        "total_candidates": 0,
        "auto_merged": 0,
        "needs_review": 0,
        "dismissed": 0,
        "by_type": {},
    }
    assert provider_operations == []
    assert coordinator._dedup_tasks == set()
```

- [ ] **Step 4: Run the real downstream tests**

Use a distinct database owned only by this invocation and the explicit test
environment:

```bash
set -euo pipefail
task3_database=medtimeline_df11_strict_identity_provider_test
task3_created_database=0
cleanup_task3_database() {
  task3_original_status=$?
  trap - EXIT INT TERM
  set +e
  task3_cleanup_status=0
  if [ "$task3_created_database" -eq 1 ]; then
    if dropdb -h 127.0.0.1 -p 5432 "$task3_database"; then
      task3_created_database=0
    else
      task3_cleanup_status=1
    fi
  fi
  if [ "$task3_original_status" -ne 0 ]; then
    exit "$task3_original_status"
  fi
  exit "$task3_cleanup_status"
}
trap cleanup_task3_database EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
createdb -h 127.0.0.1 -p 5432 "$task3_database"
task3_created_database=1
cd backend
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task3_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_upload.py::test_cloud_assisted_tracked_fhir_and_identical_reupload_finish_without_provider \
  tests/test_upload.py::test_cloud_assisted_synthetic_cda_finishes_without_provider
```

Expected: `2 passed`; real background dedup drains, both flows are terminal,
the identical FHIR upload inserts zero records, and the non-sensitive provider
operation list is empty. A provider call is a test failure even if production
fallback catches the raised exception; assertion output can reveal only the
fixed label, never decrypted configuration.

- [ ] **Step 5: Complete spec and quality review**

The spec reviewer rejects any patch to `schedule_dedup_background`,
`run_upload_dedup`, or production files. The quality reviewer checks separate
users isolate FHIR and CDA candidates, every background task is drained with
`cancel=False` in `finally`, the retained task set is empty, and terminal
assertions occur after the drain. Leave the task uncommitted.

---

### Task 4: Add constrained fixture APIs and the strict browser regression

**Files:**

- Modify: `backend/tests/test_local_ai_ci_workflows.py`
- Modify: `frontend/e2e/helpers/api-client.ts`
- Modify: `frontend/e2e/setup.spec.ts`
- Modify: `frontend/e2e/strict-local-admission.spec.ts`

**Interfaces:**

- Produces:
  `uploadTrackedSyntheticFhirCloudAssisted()` with no arguments,
  `uploadGeneratedPaginationFhirCloudAssisted(bundleJson: string)`, and
  `attemptTrackedSyntheticFhirUsingStoredPreference()` with no arguments.
- Preserves: generic `uploadStructured(filePath, filename)` only for the two
  private CDA callers; local-only login still persists
  `validated_strict_local`.
- Browser storage checks consume the effective `process.env.UPLOAD_DIR` and
  verify it is contained by `process.env.E2E_RUNTIME_ROOT`.

- [ ] **Step 1: Change one setup caller first and observe TypeScript RED**

In `frontend/e2e/setup.spec.ts`, replace only the tracked FHIR call with:

```ts
    const result = await api.uploadTrackedSyntheticFhirCloudAssisted();
```

Run:

```bash
set -euo pipefail
cd frontend
./node_modules/.bin/tsc --noEmit
```

Expected: FAIL because the method does not exist.

- [ ] **Step 2: Add fixed/content-only positive methods and fixed negative method**

In `frontend/e2e/helpers/api-client.ts`, add `node:path` and define the tracked
path internally:

```ts
const TRACKED_SYNTHETIC_FHIR = path.resolve(
  __dirname,
  "..",
  "..",
  "..",
  "backend",
  "tests",
  "fixtures",
  "sample_fhir_bundle.json"
);
```

Add a private byte helper, then expose only these constrained methods:

```ts
  private async uploadStructuredBytes(
    bytes: Uint8Array,
    filename: string,
    processingMode?: "cloud_assisted"
  ): Promise<{ upload_id: string; status: string; records_inserted: number }> {
    const formData = new FormData();
    formData.append(
      "file",
      new Blob([bytes], { type: "application/fhir+json" }),
      filename
    );
    if (processingMode) formData.append("processing_mode", processingMode);
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.token}` },
      body: formData,
    });
    if (!res.ok) {
      throw new Error(
        `Structured fixture upload failed: ${res.status} ${await res.text()}`
      );
    }
    return res.json();
  }

  async uploadTrackedSyntheticFhirCloudAssisted(): Promise<{
    upload_id: string;
    status: string;
    records_inserted: number;
  }> {
    return this.uploadStructuredBytes(
      fs.readFileSync(TRACKED_SYNTHETIC_FHIR),
      "sample_fhir_bundle.json",
      "cloud_assisted"
    );
  }

  async uploadGeneratedPaginationFhirCloudAssisted(
    bundleJson: string
  ): Promise<{ upload_id: string; status: string; records_inserted: number }> {
    return this.uploadStructuredBytes(
      new TextEncoder().encode(bundleJson),
      "pagination-seed.json",
      "cloud_assisted"
    );
  }

  async attemptTrackedSyntheticFhirUsingStoredPreference(): Promise<{
    status: number;
    body: unknown;
  }> {
    const formData = new FormData();
    formData.append(
      "file",
      new Blob([fs.readFileSync(TRACKED_SYNTHETIC_FHIR)], {
        type: "application/fhir+json",
      }),
      "sample_fhir_bundle.json"
    );
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.token}` },
      body: formData,
    });
    return { status: res.status, body: await res.json() };
  }
```

Do not expose a positive helper that accepts a caller path or filename. Do not
make processing mode optional in a public method. The generic path method stays
unchanged for private CDA callers.

- [ ] **Step 3: Add owner-scoped API state readers**

Ensure the authenticated user type includes `id`. Add:

```ts
type PatientsResponse = {
  items: { id: string }[];
  total: number;
};

type DashboardOverviewSnapshot = {
  total_records: number;
  total_patients: number;
  total_uploads: number;
  records_by_type: Record<string, number>;
  recent_records: { id: string }[];
};
```

Add methods using existing owner-scoped endpoints:

```ts
  async getLocalAIJobs(): Promise<LocalAIJobStatus[]> {
    const res = await fetch(`${API_BASE}/local-ai/jobs?active_only=false`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`List local AI jobs failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }

  async getPatients(): Promise<PatientsResponse> {
    const res = await fetch(`${API_BASE}/dashboard/patients`, {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`Get patients failed: ${res.status} ${await res.text()}`);
    }
    return res.json();
  }
```

Give the existing overview reader the `DashboardOverviewSnapshot` return type.
Do not add a database backdoor or cross-owner endpoint.

- [ ] **Step 4: Add the strict local-only browser negative**

In the startup spec created by Task 1, keep the startup test unchanged. Add
these imports beside its existing `node:fs`, `node:path`, and Playwright
imports:

```ts
import { createHash } from "node:crypto";
import { ApiClient } from "./helpers/api-client";
import { TEST_PASSWORD, uniqueEmail } from "./helpers/test-data";
```

Append the negative helpers and test:

```ts

type StorageSnapshot = {
  rootExists: boolean;
  entries: { name: string; type: string; size: number; sha256?: string }[];
};

function effectiveUploadRoot(): string {
  const runtimeRoot = process.env.E2E_RUNTIME_ROOT;
  const uploadRoot = process.env.UPLOAD_DIR;
  if (
    !runtimeRoot ||
    !uploadRoot ||
    !path.isAbsolute(runtimeRoot) ||
    !path.isAbsolute(uploadRoot)
  ) {
    throw new Error("Local-only runtime and upload roots must be absolute");
  }
  const runtimeLinkStats = fs.lstatSync(runtimeRoot);
  const uploadLinkStats = fs.lstatSync(uploadRoot);
  const runtimeReal = fs.realpathSync.native(runtimeRoot);
  const uploadReal = fs.realpathSync.native(uploadRoot);
  const runtimeStats = fs.statSync(runtimeReal);
  const uploadStats = fs.statSync(uploadReal);
  if (
    runtimeLinkStats.isSymbolicLink() ||
    uploadLinkStats.isSymbolicLink() ||
    !runtimeStats.isDirectory() ||
    !uploadStats.isDirectory() ||
    runtimeStats.uid !== process.getuid?.() ||
    uploadStats.uid !== process.getuid?.() ||
    (runtimeStats.mode & 0o777) !== 0o700 ||
    (uploadStats.mode & 0o777) !== 0o700 ||
    path.dirname(uploadReal) !== runtimeReal
  ) {
    throw new Error("UPLOAD_DIR must be an owned real child of E2E_RUNTIME_ROOT");
  }
  return uploadReal;
}

function ownerStorageSnapshot(userId: string): StorageSnapshot {
  const uploadRoot = effectiveUploadRoot();
  if (!fs.existsSync(uploadRoot)) return { rootExists: false, entries: [] };
  const entries = fs
    .readdirSync(uploadRoot)
    .filter((name) => name.startsWith(`${userId}_`))
    .sort()
    .map((name) => {
      const fullPath = path.join(uploadRoot, name);
      const metadata = fs.lstatSync(fullPath);
      const entry: {
        name: string;
        type: string;
        size: number;
        sha256?: string;
      } = {
        name,
        type: metadata.isFile()
          ? "file"
          : metadata.isDirectory()
            ? "directory"
            : "other",
        size: metadata.size,
      };
      if (metadata.isFile()) {
        entry.sha256 = createHash("sha256")
          .update(fs.readFileSync(fullPath))
          .digest("hex");
      }
      return entry;
    });
  return { rootExists: true, entries };
}

async function ownerIngestionSnapshot(api: ApiClient) {
  const [history, jobs, records, patients, overview] = await Promise.all([
    api.getUploadHistory(),
    api.getLocalAIJobs(),
    api.getRecords({ page: 1, page_size: 100 }),
    api.getPatients(),
    api.getDashboardOverview(),
  ]);
  return {
    uploadIds: history.items.map((item) => item.id).sort(),
    jobIds: jobs.map((item) => item.id).sort(),
    recordIds: records.items.map((item) => item.id).sort(),
    recordTotal: records.total,
    patientIds: patients.items.map((item) => item.id).sort(),
    patientTotal: patients.total,
    overview,
  };
}

test("legacy strict-local upload fails before owner database or storage side effects", async () => {
  test.skip(
    process.env.E2E_LOCAL_ONLY !== "1",
    "strict-local negative admission requires the local-only profile"
  );
  const api = new ApiClient();
  const loginIdentifier = uniqueEmail("strict-local-admission");
  await api.register(loginIdentifier, TEST_PASSWORD);
  await api.login(loginIdentifier, TEST_PASSWORD);
  expect((await api.getLlmSettings()).routing.processing_mode).toBe(
    "validated_strict_local"
  );
  const user = await api.getMe();
  const beforeState = await ownerIngestionSnapshot(api);
  const beforeStorage = ownerStorageSnapshot(user.id);

  const response = await api.attemptTrackedSyntheticFhirUsingStoredPreference();

  expect(response).toEqual({
    status: 409,
    body: { detail: "Strict-local worker runtime identity is required." },
  });
  expect(await ownerIngestionSnapshot(api)).toEqual(beforeState);
  expect(ownerStorageSnapshot(user.id)).toEqual(beforeStorage);
});
```

The test never reads another owner's file or includes raw bytes in assertion
output.

- [ ] **Step 5: Add a static helper-boundary regression**

Add to `backend/tests/test_local_ai_ci_workflows.py`:

```python
def test_local_only_structured_fixture_helpers_are_constrained() -> None:
    content = (REPOSITORY_ROOT / "frontend/e2e/helpers/api-client.ts").read_text(
        encoding="utf-8"
    )
    assert "async uploadTrackedSyntheticFhirCloudAssisted():" in content
    assert "async attemptTrackedSyntheticFhirUsingStoredPreference():" in content
    assert "uploadGeneratedPaginationFhirCloudAssisted(\n    bundleJson: string" in content
    assert "uploadDeterministicStructuredFixtureCloudAssisted" not in content
    assert "sample_fhir_bundle.json" in content
```

- [ ] **Step 6: Verify static and TypeScript GREEN**

Run:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_ci_workflows.py::test_local_only_structured_fixture_helpers_are_constrained
cd ../frontend
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/eslint \
  e2e/helpers/api-client.ts \
  e2e/setup.spec.ts \
  e2e/strict-local-admission.spec.ts
```

Expected: all commands exit 0.

- [ ] **Step 7: Run the first focused E2E gate with owned cleanup**

Create the DB on explicit loopback. The shell tracks ownership of the dependency
link and runtime root. It never unlinks a pre-existing symlink:

```bash
set -euo pipefail
task_worktree_root="$PWD"
task_worktree_real="$(pwd -P)"
test "$task_worktree_root" = "$task_worktree_real"
task4_database=medtimeline_df11_strict_identity_task4_e2e
task4_created_database=0
task_backend_link="$task_worktree_root/backend/.venv"
task_dependency_source=/Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv
task_created_backend_link=0
task_backend_link_identity=""
task_backend_link_target=""
task_evidence_parent="$task_worktree_root/frontend/test-results"
task_runtime_parent="$task_evidence_parent/runtime"
task_output_parent="$task_evidence_parent/executions"
task_runtime_root=""
task_output_root=""
task_runtime_parent_real=""
task_output_parent_real=""
task_runtime_parent_identity=""
task_output_parent_identity=""
task_runtime_identity=""
task_output_identity=""

safe_remove_task4_root() {
  task4_target="$1"
  task4_parent="$2"
  task4_parent_real="$3"
  task4_parent_identity="$4"
  task4_target_identity="$5"
  test -d "$task4_parent" && test ! -L "$task4_parent" || return 1
  test "$(cd -P -- "$task4_parent" && pwd -P)" = \
    "$task4_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task4_parent")" = \
    "$task4_parent_identity" || return 1
  test -d "$task4_target" && test ! -L "$task4_target" || return 1
  task4_target_real="$(cd -P -- "$task4_target" && pwd -P)" || return 1
  test "$(dirname "$task4_target_real")" = "$task4_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task4_target")" = \
    "$task4_target_identity" || return 1
  rm -rf -- "$task4_target"
}

cleanup_task4() {
  task4_original_status=$?
  trap - EXIT INT TERM
  set +e
  task4_cleanup_status=0
  if [ -n "$task_output_root" ]; then
    safe_remove_task4_root \
      "$task_output_root" "$task_output_parent" "$task_output_parent_real" \
      "$task_output_parent_identity" "$task_output_identity" || task4_cleanup_status=1
  fi
  if [ -n "$task_runtime_root" ]; then
    safe_remove_task4_root \
      "$task_runtime_root" "$task_runtime_parent" "$task_runtime_parent_real" \
      "$task_runtime_parent_identity" "$task_runtime_identity" || task4_cleanup_status=1
  fi
  if [ "$task_created_backend_link" -eq 1 ]; then
    if [ -L "$task_backend_link" ] && \
       [ "$(stat -f '%d:%i:%u:%HT' "$task_backend_link")" = "$task_backend_link_identity" ] && \
       [ "$(readlink "$task_backend_link")" = "$task_backend_link_target" ]; then
      unlink "$task_backend_link" || task4_cleanup_status=1
    else
      task4_cleanup_status=1
    fi
  fi
  if [ "$task4_created_database" -eq 1 ]; then
    if dropdb -h 127.0.0.1 -p 5432 "$task4_database"; then
      task4_created_database=0
    else
      task4_cleanup_status=1
    fi
  fi
  if [ "$task4_original_status" -ne 0 ]; then
    exit "$task4_original_status"
  fi
  exit "$task4_cleanup_status"
}
trap cleanup_task4 EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test -d "$task_evidence_parent" && test ! -L "$task_evidence_parent"
test "$(stat -f '%u' "$task_evidence_parent")" -eq "$(id -u)"
test "$(cd -P -- "$task_evidence_parent" && pwd -P)" = \
  "$task_worktree_real/frontend/test-results"
for task_parent in "$task_runtime_parent" "$task_output_parent"; do
  if [ ! -e "$task_parent" ] && [ ! -L "$task_parent" ]; then
    mkdir -m 700 -- "$task_parent"
  fi
  test -d "$task_parent" && test ! -L "$task_parent"
  test "$(stat -f '%u' "$task_parent")" -eq "$(id -u)"
done
task_runtime_parent_real="$(cd -P -- "$task_runtime_parent" && pwd -P)"
task_output_parent_real="$(cd -P -- "$task_output_parent" && pwd -P)"
test "$(dirname "$task_runtime_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
test "$(dirname "$task_output_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
task_runtime_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")"
task_output_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")"
umask 077
task_runtime_root="$(mktemp -d "$task_runtime_parent/task4.XXXXXX")"
task_run_token="$(basename "$task_runtime_root")"
task_output_root="$task_output_parent/$task_run_token"
mkdir -m 700 -- "$task_output_root"
for task_root in "$task_runtime_root" "$task_output_root"; do
  test -d "$task_root" && test ! -L "$task_root"
  test "$(stat -f '%u' "$task_root")" -eq "$(id -u)"
  test "$(stat -f '%Lp' "$task_root")" = 700
done
test "$(dirname "$(cd -P -- "$task_runtime_root" && pwd -P)")" = \
  "$task_runtime_parent_real"
test "$(dirname "$(cd -P -- "$task_output_root" && pwd -P)")" = \
  "$task_output_parent_real"
task_runtime_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")"
task_output_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")"

if [ ! -e "$task_backend_link" ] && [ ! -L "$task_backend_link" ]; then
  test -x "$task_dependency_source/bin/python"
  ln -s "$task_dependency_source" "$task_backend_link"
  task_created_backend_link=1
  task_backend_link_identity="$(stat -f '%d:%i:%u:%HT' "$task_backend_link")"
  task_backend_link_target="$(readlink "$task_backend_link")"
fi
test -x "$task_backend_link/bin/python"
createdb -h 127.0.0.1 -p 5432 "$task4_database"
task4_created_database=1
cd frontend
env -u DATABASE_URL -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task4_database" \
  ./node_modules/.bin/playwright test \
  e2e/setup.spec.ts e2e/strict-local-admission.spec.ts \
  --workers=1 --trace=retain-on-failure
```

Expected: both setup tests and the new negative test pass. The backend starts
with the sentinel, the positive setup uses explicit cloud-assisted FHIR, and the
negative returns exact 409 without owner database/storage changes. The trap
drops only the database this invocation proved it created and removes only roots
and the dependency link whose captured identities still match.

- [ ] **Step 8: Complete spec and quality review**

The spec reviewer checks the public positive helpers accept no path, the
negative helper sends no mode, the generic helper remains for private CDA, and
the browser reads the effective upload root. The quality reviewer checks
absence-sensitive/content-free storage metadata and exactly one account
registration. Leave the task uncommitted.

---

### Task 5: Migrate synthetic callers and split summary UI from model execution

**Files:**

- Modify: `frontend/e2e/admin-records.spec.ts`
- Modify: `frontend/e2e/dashboard-home.spec.ts`
- Modify: `frontend/e2e/display-badges-providers.spec.ts`
- Modify: `frontend/e2e/pagination-integrity.spec.ts`
- Modify: `frontend/e2e/record-ai-metadata.spec.ts`
- Modify: `frontend/e2e/record-detail-page.spec.ts`
- Modify: `frontend/e2e/record-detail-sheet.spec.ts`
- Modify: `frontend/e2e/record-renderers.spec.ts`
- Modify: `frontend/e2e/summaries.spec.ts`
- Modify: `frontend/e2e/timeline.spec.ts`
- Modify: `frontend/e2e/upload-dedup.spec.ts`
- Modify: `frontend/e2e/upload-progress.spec.ts`
- Modify: `frontend/e2e/upload-structured.spec.ts`

`frontend/e2e/setup.spec.ts` was migrated in Task 4.

**Interfaces:**

- Consumes: the three constrained helper methods from Task 4.
- Produces: 14 fixed tracked-FHIR calls, one generated-pagination content call,
  two unchanged generic private CDA calls, and one explicit browser dropzone
  mode override.
- Summary output: seven non-execution UI tests run in ordinary local-only. Only
  local-only plus `E2E_ATTESTED_STRICT_PACK=1` configures the three named model
  execution cases; every other profile uses the exact generic skip reason.
  External credentials alone do not configure stored cloud-assisted execution.
  The primary-action case first proves the Record subject selector has a value,
  then validates the no-pack local-only, attested-pack local-only, or non-local
  prompt-only action without changing stored mode.

- [ ] **Step 1: Re-enumerate all current generic calls before editing**

Run:

```bash
set -euo pipefail
rg -n "uploadStructured\(" frontend/e2e --glob '*.ts'
```

Expected before editing: one method definition and 17 calls:

- 14 tracked `PATHS.fhirBundle` calls: `admin-records`;
  `dashboard-home`; `display-badges-providers`; `record-ai-metadata`;
  `record-detail-page`; `record-detail-sheet`; `record-renderers`; `setup`;
  `summaries`; `timeline`; three in `upload-dedup`; and one in
  `upload-progress`.
- one generated pagination temp-file call.
- two private CDA path calls: `upload-dedup` and `upload-structured`.

Task 4 has already moved `setup`, so expect 16 remaining calls at Task 5 entry.
Stop if the mechanical set differs.

- [ ] **Step 2: Replace the remaining fixed tracked-FHIR calls**

For each fixed tracked FHIR call, replace the complete argument-bearing call
with:

```ts
await api.uploadTrackedSyntheticFhirCloudAssisted()
```

Apply this in:

```text
frontend/e2e/admin-records.spec.ts
frontend/e2e/dashboard-home.spec.ts
frontend/e2e/display-badges-providers.spec.ts
frontend/e2e/record-ai-metadata.spec.ts
frontend/e2e/record-detail-page.spec.ts
frontend/e2e/record-detail-sheet.spec.ts
frontend/e2e/record-renderers.spec.ts
frontend/e2e/summaries.spec.ts
frontend/e2e/timeline.spec.ts
frontend/e2e/upload-dedup.spec.ts (three tracked FHIR calls only)
frontend/e2e/upload-progress.spec.ts
```

Keep each surrounding assignment, polling call, and assertion unchanged. Do not
change the private `cdaPath` call in `upload-dedup.spec.ts`.

- [ ] **Step 3: Move pagination from a temp path to generated content**

In `frontend/e2e/pagination-integrity.spec.ts`, replace the temp-file write,
try/finally upload, and unlink with:

```ts
    const up = await api.uploadGeneratedPaginationFhirCloudAssisted(
      JSON.stringify(buildBundle())
    );
    await api.pollUploadStatus(up.upload_id, 90_000);
```

Remove now-unused `fs`, `os`, or path imports. The generated JSON never enters a
caller-controlled path helper.

- [ ] **Step 4: Make the browser FHIR dropzone request explicit**

In `frontend/e2e/upload-structured.spec.ts`, add:

```ts
async function useCloudAssistedForTrackedSyntheticFhir(
  page: import("@playwright/test").Page
): Promise<void> {
  await page.route("**/api/v1/settings/llm", async (route) => {
    if (route.request().method() !== "GET") {
      await route.continue();
      return;
    }
    const response = await route.fetch();
    const body = (await response.json()) as {
      routing?: Record<string, unknown>;
    };
    await route.fulfill({
      response,
      json: {
        ...body,
        routing: {
          ...body.routing,
          processing_mode: "cloud_assisted",
        },
      },
    });
  });
}
```

Call it only in `upload FHIR JSON bundle via dropzone`, after the synthetic user
is ready and before `page.goto("/upload")`. After upload/history assertions, add:

```ts
    expect((await api.getLlmSettings()).routing.processing_mode).toBe(
      "validated_strict_local"
    );
```

This changes the real browser form request for that tracked fixture; it does not
modify the account preference or production upload page. Leave the private
standalone CDA `api.uploadStructured(xmlPath, xmlFile)` unchanged.

- [ ] **Step 5: Gate only the three real model-execution summary cases**

Replace the current `modelExecutionConfigured` definition in
`frontend/e2e/summaries.spec.ts` with:

```ts
const localOnly = process.env.E2E_LOCAL_ONLY === "1";
const realModelExecutionConfigured =
  localOnly && process.env.E2E_ATTESTED_STRICT_PACK === "1";
const modelExecutionSkipReason =
  "No E2E model execution profile is configured";
```

Use this exact gate and reason only in:

```ts
test.skip(!realModelExecutionConfigured, modelExecutionSkipReason);
```

The three test titles are:

- `generate produces a result`
- `history entry reopens a saved summary without regenerating`
- `generation reports the selected privacy boundary`

Do not add the gate to `beforeAll` or to these seven UI cases:

- `patient selector loads patients`
- `summary type tabs exist`
- `category dropdown appears for By category type`
- `date range inputs appear for Date range type`
- `output format options work`
- `primary summary action reflects execution availability with a patient`
- `AI disclaimer always visible`

Do not change `generateAndOpenSummary` into a fake success path and do not submit
the summary as cloud-assisted. The exact legacy summary rejection is covered by
Task 2.

- [ ] **Step 6: Run the mechanical helper and summary audit**

Run:

```bash
set -euo pipefail
rg -n "uploadTrackedSyntheticFhirCloudAssisted\(" frontend/e2e --glob '*.ts'
rg -n "uploadGeneratedPaginationFhirCloudAssisted\(" frontend/e2e --glob '*.ts'
rg -n "uploadStructured\(" frontend/e2e --glob '*.ts'
rg -n "realModelExecutionConfigured|modelExecutionSkipReason|test\.skip" \
  frontend/e2e/summaries.spec.ts
```

Expected:

- fixed tracked helper: one definition and 14 call sites;
- generated pagination helper: one definition and one call site;
- generic helper: one definition and exactly two private CDA calls;
- the summary execution gate appears in exactly the three named tests.

- [ ] **Step 7: Run TypeScript and focused lint**

Run:

```bash
set -euo pipefail
cd frontend
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/eslint \
  playwright.config.ts \
  e2e/helpers/api-client.ts \
  e2e/admin-records.spec.ts \
  e2e/dashboard-home.spec.ts \
  e2e/display-badges-providers.spec.ts \
  e2e/pagination-integrity.spec.ts \
  e2e/record-ai-metadata.spec.ts \
  e2e/record-detail-page.spec.ts \
  e2e/record-detail-sheet.spec.ts \
  e2e/record-renderers.spec.ts \
  e2e/setup.spec.ts \
  e2e/strict-local-admission.spec.ts \
  e2e/summaries.spec.ts \
  e2e/timeline.spec.ts \
  e2e/upload-dedup.spec.ts \
  e2e/upload-progress.spec.ts \
  e2e/upload-structured.spec.ts
```

Expected: both commands exit 0.

- [ ] **Step 8: Create an owned focused E2E DB and run every populated spec**

Use the same explicit host/port and ownership-safe cleanup pattern:

```bash
set -euo pipefail
task_worktree_root="$PWD"
task_worktree_real="$(pwd -P)"
test "$task_worktree_root" = "$task_worktree_real"
task5_database=medtimeline_df11_strict_identity_task5_e2e
task5_created_database=0
task_backend_link="$task_worktree_root/backend/.venv"
task_dependency_source=/Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv
task_created_backend_link=0
task_backend_link_identity=""
task_backend_link_target=""
task_evidence_parent="$task_worktree_root/frontend/test-results"
task_runtime_parent="$task_evidence_parent/runtime"
task_output_parent="$task_evidence_parent/executions"
task_runtime_root=""
task_output_root=""
task_runtime_parent_real=""
task_output_parent_real=""
task_runtime_parent_identity=""
task_output_parent_identity=""
task_runtime_identity=""
task_output_identity=""

safe_remove_task5_root() {
  task5_target="$1"
  task5_parent="$2"
  task5_parent_real="$3"
  task5_parent_identity="$4"
  task5_target_identity="$5"
  test -d "$task5_parent" && test ! -L "$task5_parent" || return 1
  test "$(cd -P -- "$task5_parent" && pwd -P)" = \
    "$task5_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task5_parent")" = \
    "$task5_parent_identity" || return 1
  test -d "$task5_target" && test ! -L "$task5_target" || return 1
  task5_target_real="$(cd -P -- "$task5_target" && pwd -P)" || return 1
  test "$(dirname "$task5_target_real")" = "$task5_parent_real" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task5_target")" = \
    "$task5_target_identity" || return 1
  rm -rf -- "$task5_target"
}

cleanup_task5() {
  task5_original_status=$?
  trap - EXIT INT TERM
  set +e
  task5_cleanup_status=0
  if [ -n "$task_output_root" ]; then
    safe_remove_task5_root \
      "$task_output_root" "$task_output_parent" "$task_output_parent_real" \
      "$task_output_parent_identity" "$task_output_identity" || task5_cleanup_status=1
  fi
  if [ -n "$task_runtime_root" ]; then
    safe_remove_task5_root \
      "$task_runtime_root" "$task_runtime_parent" "$task_runtime_parent_real" \
      "$task_runtime_parent_identity" "$task_runtime_identity" || task5_cleanup_status=1
  fi
  if [ "$task_created_backend_link" -eq 1 ]; then
    if [ -L "$task_backend_link" ] && \
       [ "$(stat -f '%d:%i:%u:%HT' "$task_backend_link")" = "$task_backend_link_identity" ] && \
       [ "$(readlink "$task_backend_link")" = "$task_backend_link_target" ]; then
      unlink "$task_backend_link" || task5_cleanup_status=1
    else
      task5_cleanup_status=1
    fi
  fi
  if [ "$task5_created_database" -eq 1 ]; then
    if dropdb -h 127.0.0.1 -p 5432 "$task5_database"; then
      task5_created_database=0
    else
      task5_cleanup_status=1
    fi
  fi
  if [ "$task5_original_status" -ne 0 ]; then
    exit "$task5_original_status"
  fi
  exit "$task5_cleanup_status"
}
trap cleanup_task5 EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test -d "$task_evidence_parent" && test ! -L "$task_evidence_parent"
test "$(stat -f '%u' "$task_evidence_parent")" -eq "$(id -u)"
test "$(cd -P -- "$task_evidence_parent" && pwd -P)" = \
  "$task_worktree_real/frontend/test-results"
for task_parent in "$task_runtime_parent" "$task_output_parent"; do
  if [ ! -e "$task_parent" ] && [ ! -L "$task_parent" ]; then
    mkdir -m 700 -- "$task_parent"
  fi
  test -d "$task_parent" && test ! -L "$task_parent"
  test "$(stat -f '%u' "$task_parent")" -eq "$(id -u)"
done
task_runtime_parent_real="$(cd -P -- "$task_runtime_parent" && pwd -P)"
task_output_parent_real="$(cd -P -- "$task_output_parent" && pwd -P)"
test "$(dirname "$task_runtime_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
test "$(dirname "$task_output_parent_real")" = \
  "$task_worktree_real/frontend/test-results"
task_runtime_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")"
task_output_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")"
umask 077
task_runtime_root="$(mktemp -d "$task_runtime_parent/task5.XXXXXX")"
task_run_token="$(basename "$task_runtime_root")"
task_output_root="$task_output_parent/$task_run_token"
mkdir -m 700 -- "$task_output_root"
for task_root in "$task_runtime_root" "$task_output_root"; do
  test -d "$task_root" && test ! -L "$task_root"
  test "$(stat -f '%u' "$task_root")" -eq "$(id -u)"
  test "$(stat -f '%Lp' "$task_root")" = 700
done
test "$(dirname "$(cd -P -- "$task_runtime_root" && pwd -P)")" = \
  "$task_runtime_parent_real"
test "$(dirname "$(cd -P -- "$task_output_root" && pwd -P)")" = \
  "$task_output_parent_real"
task_runtime_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")"
task_output_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")"

if [ ! -e "$task_backend_link" ] && [ ! -L "$task_backend_link" ]; then
  test -x "$task_dependency_source/bin/python"
  ln -s "$task_dependency_source" "$task_backend_link"
  task_created_backend_link=1
  task_backend_link_identity="$(stat -f '%d:%i:%u:%HT' "$task_backend_link")"
  task_backend_link_target="$(readlink "$task_backend_link")"
fi
test -x "$task_backend_link/bin/python"
createdb -h 127.0.0.1 -p 5432 "$task5_database"
task5_created_database=1
cd frontend
env -u DATABASE_URL -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task5_database" \
  ./node_modules/.bin/playwright test \
  e2e/admin-records.spec.ts \
  e2e/dashboard-home.spec.ts \
  e2e/display-badges-providers.spec.ts \
  e2e/pagination-integrity.spec.ts \
  e2e/record-ai-metadata.spec.ts \
  e2e/record-detail-page.spec.ts \
  e2e/record-detail-sheet.spec.ts \
  e2e/record-renderers.spec.ts \
  e2e/setup.spec.ts \
  e2e/strict-local-admission.spec.ts \
  e2e/summaries.spec.ts \
  e2e/timeline.spec.ts \
  e2e/upload-dedup.spec.ts \
  e2e/upload-progress.spec.ts \
  e2e/upload-structured.spec.ts \
  --workers=1 --trace=retain-on-failure
```

Expected: zero failures and zero not-run tests. Only private-fixture cases in
this selection plus the three named summary model-execution cases skip. Record
exact focused counts rather than predicting them.

- [ ] **Step 9: Complete spec and quality review**

The spec reviewer reconciles every helper hit with the 14+1+2 split and checks
the exact seven-run/three-skip summary boundary. The primary-action case must
non-vacuously select a Record subject and cover the disabled no-pack
strict-local action, enabled attested-pack strict-local action, and enabled
non-local prompt-only action without mutating mode. The quality reviewer checks
the GET-only upload-page route, stored strict preference assertion, and removal
of pagination temp-file writes. Leave the task uncommitted.

---

### Task 6: Correct and humanize the public local-only operations contract

**Files:**

- Modify: `backend/tests/test_local_ai_ci_workflows.py`
- Modify: `docs/operations-strict-local-ai.md`

**Interfaces:**

- Consumes: Task 1's profile, Task 3's real dedup proof, and Task 5's summary
  gate.
- Produces: operator documentation that distinguishes OS/socket/browser egress
  denial from a real attested strict pack and names the honest skip boundary.
- Preserves: all production install, migration, release, rollback, and pack
  operations guidance outside the browser-test section.

- [ ] **Step 1: Add a failing document-semantics regression**

Add to `backend/tests/test_local_ai_ci_workflows.py`:

```python
def test_local_only_browser_docs_separate_network_denial_from_pack_attestation() -> None:
    content = (REPOSITORY_ROOT / "docs/operations-strict-local-ai.md").read_text(
        encoding="utf-8"
    )
    section = content.split("## Run browser tests with local-only enforcement", 1)[
        1
    ].split("## ", 1)[0]

    assert "proves network confinement" in section
    assert "does not prove that an attested strict-local model pack is installed" in section
    assert "`/usr/bin/false`" in section
    assert "startup sentinel" in section
    assert "explicit `cloud_assisted`" in section
    assert "background dedup" in section
    assert "provider construction" in section
    assert "three summary model-execution cases" in section
    assert "createdb -h 127.0.0.1 -p 5432" in section
    assert "dropdb -h 127.0.0.1 -p 5432" in section
    assert "dropdb --if-exists" not in section
    assert "E2E_RUNTIME_ROOT" in section
    assert "E2E_OUTPUT_ROOT" in section
    assert "mktemp -d" in section
    assert "stat -f '%d:%i:%u:%HT:%Lp'" in section
    assert "pwd -P" in section
    assert "test ! -L" in section
    assert "chmod -R" not in section
    assert "env -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR" in section
    assert "LLM_PROVIDER=gemini" in section
    assert "LLM_SUMMARY_PROVIDER=" in section
    assert "LLM_SECTION_PROVIDER=" in section
    assert "LLM_DEDUP_PROVIDER=" in section
    assert "LLM_EXTRACTION_PROVIDER=" in section
    assert "E2E worker returns" not in section
    assert "already sandboxed model worker" not in section
```

- [ ] **Step 2: Run the document test and observe RED**

Run:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_ci_workflows.py::test_local_only_browser_docs_separate_network_denial_from_pack_attestation
```

Expected: FAIL because the current section claims a deterministic worker runs
and does not separate network confinement from attestation.

- [ ] **Step 3: Replace the inaccurate prose and make the public command own its state**

Preserve the explanations of the socket guard, macOS sandbox, closed browser
proxy, service-worker blocking, telemetry, and offline flags. Replace the
current browser command with this ownership-safe equivalent. It validates or
creates the shared evidence/runtime/output parents without changing their
permissions or removing them. Each task root is identity-captured immediately.
The command uses a validated 24-character random token for a lowercase loopback
database name, records the created database's OID and owner, and drops it only
when a cleanup query returns the same identity. A missing database is already
clean; a query failure or identity mismatch retains it and fails cleanup. It
cannot inherit a shell encryption key, upload path, temp path, private fixture
root, provider credential, loopback model route, or real-pack gate:

```bash
set -euo pipefail
task_repo_root="$(pwd -P)"
task_evidence_parent="$task_repo_root/frontend/test-results"
task_runtime_parent="$task_evidence_parent/runtime"
task_output_parent="$task_evidence_parent/executions"
task_runtime_root=""
task_output_root=""
task_runtime_parent_identity=""
task_output_parent_identity=""
task_runtime_root_identity=""
task_output_root_identity=""
task_run_token=""
task_e2e_database=""
task_e2e_database_identity=""
task_created_e2e_database=0
validate_owned_parent() {
  task_parent="$1"
  test -d "$task_parent" && test ! -L "$task_parent"
  test "$(cd -P -- "$task_parent" && pwd -P)" = "$task_parent"
  test "$(stat -f '%u' "$task_parent")" -eq "$(id -u)"
  task_parent_mode="$(stat -f '%Lp' "$task_parent")"
  case "$task_parent_mode" in
    ""|*[!0-7]*) return 1 ;;
  esac
  test "$((8#$task_parent_mode & 022))" -eq 0
}
validate_owned_root() {
  task_root="$1"
  task_parent="$2"
  test -d "$task_root" && test ! -L "$task_root"
  test "$(stat -f '%u' "$task_root")" -eq "$(id -u)"
  test "$(stat -f '%Lp' "$task_root")" = 700
  task_root_real="$(cd -P -- "$task_root" && pwd -P)"
  test "$task_root_real" = "$task_root"
  test "$(dirname -- "$task_root_real")" = "$task_parent"
}
query_task_database_identity() {
  psql -X -qAt -v ON_ERROR_STOP=1 \
    -v task_database="$task_e2e_database" \
    -h 127.0.0.1 -p 5432 -d postgres <<'SQL'
SELECT oid::text || ':' || datdba::text
FROM pg_database
WHERE datname = :'task_database';
SQL
}
safe_remove_owned_directory() {
  task_remove_root="$1"
  task_remove_parent="$2"
  task_expected_parent_identity="$3"
  task_expected_root_identity="$4"
  test -n "$task_remove_root" && test -n "$task_remove_parent" || return 1
  test -n "$task_expected_parent_identity" && \
    test -n "$task_expected_root_identity" || return 1
  test -d "$task_remove_parent" && test ! -L "$task_remove_parent" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_remove_parent")" = \
    "$task_expected_parent_identity" || return 1
  task_remove_parent_real="$(cd -P -- "$task_remove_parent" && pwd -P)" || return 1
  test "$task_remove_parent_real" = "$task_remove_parent" || return 1
  test -d "$task_remove_root" && test ! -L "$task_remove_root" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_remove_root")" = \
    "$task_expected_root_identity" || return 1
  task_remove_root_real="$(cd -P -- "$task_remove_root" && pwd -P)" || return 1
  test "$task_remove_root_real" = "$task_remove_root" || return 1
  test "$(dirname -- "$task_remove_root_real")" = \
    "$task_remove_parent_real" || return 1
  rm -rf -- "$task_remove_root" >/dev/null 2>&1
}
cleanup_local_e2e_docs() {
  task_original_status=$?
  trap - EXIT INT TERM
  task_cleanup_status=0
  if [ "$task_created_e2e_database" -eq 1 ]; then
    task_current_e2e_database_identity=""
    if task_current_e2e_database_identity="$(query_task_database_identity 2>/dev/null)"; then
      if [ -z "$task_current_e2e_database_identity" ]; then
        :
      elif [ -n "$task_e2e_database_identity" ] && \
        [ "$task_current_e2e_database_identity" = "$task_e2e_database_identity" ]; then
        if ! dropdb -h 127.0.0.1 -p 5432 "$task_e2e_database" \
          >/dev/null 2>&1; then
          task_cleanup_status=1
        fi
      else
        task_cleanup_status=1
      fi
    else
      task_cleanup_status=1
    fi
  fi
  if [ -n "$task_runtime_root" ]; then
    if ! safe_remove_owned_directory \
      "$task_runtime_root" "$task_runtime_parent" \
      "$task_runtime_parent_identity" "$task_runtime_root_identity" \
      2>/dev/null; then
      task_cleanup_status=1
    fi
  fi
  if [ -n "$task_output_root" ]; then
    if ! safe_remove_owned_directory \
      "$task_output_root" "$task_output_parent" \
      "$task_output_parent_identity" "$task_output_root_identity" \
      2>/dev/null; then
      task_cleanup_status=1
    fi
  fi
  if [ "$task_cleanup_status" -ne 0 ]; then
    printf '%s\n' \
      'local-only browser cleanup could not remove all task-owned state' >&2
  fi
  if [ "$task_original_status" -ne 0 ]; then
    exit "$task_original_status"
  fi
  exit "$task_cleanup_status"
}
trap cleanup_local_e2e_docs EXIT
test -d "$task_repo_root/frontend" && test ! -L "$task_repo_root/frontend"
test "$(cd -P -- "$task_repo_root/frontend" && pwd -P)" = \
  "$task_repo_root/frontend"
if [ ! -e "$task_evidence_parent" ] && [ ! -L "$task_evidence_parent" ]; then
  mkdir -m 700 -- "$task_evidence_parent"
fi
validate_owned_parent "$task_evidence_parent"
if [ ! -e "$task_runtime_parent" ] && [ ! -L "$task_runtime_parent" ]; then
  mkdir -m 700 -- "$task_runtime_parent"
fi
validate_owned_parent "$task_runtime_parent"
task_runtime_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")"
if [ ! -e "$task_output_parent" ] && [ ! -L "$task_output_parent" ]; then
  mkdir -m 700 -- "$task_output_parent"
fi
validate_owned_parent "$task_output_parent"
task_output_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")"
umask 077
task_runtime_root="$(mktemp -d "$task_runtime_parent/docs.XXXXXXXXXXXXXXXXXXXXXXXX")"
validate_owned_root "$task_runtime_root" "$task_runtime_parent"
task_runtime_root_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")"
task_output_root="$(mktemp -d "$task_output_parent/docs.XXXXXXXXXXXXXXXXXXXXXXXX")"
validate_owned_root "$task_output_root" "$task_output_parent"
task_output_root_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")"
task_run_token="${task_runtime_root##*.}"
test "${#task_run_token}" -eq 24
case "$task_run_token" in
  ""|*[!abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]*) exit 1 ;;
esac
task_run_token="$(printf '%s' "$task_run_token" | tr '[:upper:]' '[:lower:]')"
task_e2e_database="medtimeline_e2e_$task_run_token"
test "${#task_e2e_database}" -le 63
case "$task_e2e_database" in
  medtimeline_e2e_*) ;;
  *) exit 1 ;;
esac
case "$task_e2e_database" in
  ""|*[!a-z0-9_]*) exit 1 ;;
esac
createdb -h 127.0.0.1 -p 5432 "$task_e2e_database"
task_created_e2e_database=1
task_e2e_database_identity="$(query_task_database_identity)"
task_e2e_database_oid="${task_e2e_database_identity%%:*}"
task_e2e_database_owner="${task_e2e_database_identity#*:}"
case "$task_e2e_database_oid" in
  ""|*[!0-9]*) exit 1 ;;
esac
case "$task_e2e_database_owner" in
  ""|*[!0-9]*) exit 1 ;;
esac
cd frontend
env -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_e2e_database" \
  ./node_modules/.bin/playwright test --workers=1
```

Do not add a `dropdb` preflight to this public command. The cleanup flag becomes
`1` immediately after this invocation creates the database. Capture its
`oid:datdba` identity through fixed SQL sent to `psql` on standard input. Before
the single structural `dropdb`, query again and require the same identity. The
random name plus OID/owner comparison detects ordinary collisions and
replacement, but it is not absolute protection from a malicious PostgreSQL
cluster administrator. Cleanup reports one generic warning when state must be
retained. An earlier nonzero status wins; otherwise cleanup status is returned.

Replace only the inaccurate worker/model claims with prose that states all of
the following:

```markdown
This profile proves network confinement for the test database, backend, Next.js
server, and browser. It does not prove that an attested strict-local model pack
is installed. The legacy v1 manifest remains only for bounded negative
admission tests. `/usr/bin/false` is a startup sentinel so application lifespan
can validate a one-token executable; it is not a worker and the suite never
spawns it.

Tests that need populated records use an explicit `cloud_assisted` request for
the tracked synthetic FHIR fixture or generated pagination FHIR. Backend tests
also cover the repository's synthetic CDA parser. They run background dedup to
terminal state and fail if provider construction occurs. Provider credentials
are empty and all local-only network guards stay active. The account's default
preference remains `validated_strict_local`.

Strict upload and summary regressions check the exact legacy-v1 rejection before
request side effects. Seven summary UI cases do not execute a model and still
run. The three summary model-execution cases require a real attested worker and
pack, so this profile skips them. A boolean test gate is not release evidence.

Use the release and fidelity gates above when you need evidence for a real
strict-local pack. A browser run is not release, benchmark, fidelity,
pack-verification, promotion, or deployment evidence.
```

Do not broaden the provider-free claim beyond the tracked FHIR/identical
re-upload/synthetic CDA sequences exercised by Task 3 and the browser network
guards exercised in E2E.

- [ ] **Step 4: Run the humanizer workflow on the edited public prose**

Use the humanizer skill on only the changed browser-test paragraphs:

1. Audience: self-hosting operator; restrained technical reference voice.
2. Mark promotion, empty trailing clauses, vague attribution, AI vocabulary,
   copula avoidance, forced ranges/trios, synonym cycling, mechanical
   punctuation, filler, and generic endings.
3. Rewrite only marked text; preserve every security and attestation claim.
4. Read for rhythm and prefer direct constructions.
5. Ask: `What makes the text below so obviously AI generated?`
6. List remaining tells in the task handoff, revise again, and keep that audit
   out of the public document.
7. Confirm the final prose invents no provider, model, pack, release, metric,
   benchmark, promotion, or deployment evidence.

If wording changes, update the static test to assert the same concrete semantics
without pinning incidental prose.

- [ ] **Step 5: Run the full static workflow file and inspect scope**

Run:

```bash
set -euo pipefail
cd backend
env -u DATABASE_URL -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q tests/test_local_ai_ci_workflows.py
cd ..
git diff --check
git diff -- docs/operations-strict-local-ai.md \
  backend/tests/test_local_ai_ci_workflows.py
```

Expected: all static workflow tests pass. Only the local-only browser section
changes; production operations guidance remains unchanged.

- [ ] **Step 6: Complete spec and quality review**

The spec reviewer checks every required claim and the exact summary skip
boundary. The quality reviewer repeats the humanizer audit and rejects inflated,
formulaic, or fabricated claims. Leave the task uncommitted.

---

### Task 7: Root integration, parity, full local-only verification, and handoff

**Files:**

- Verify only. Do not add implementation paths.

**Interfaces:**

- Consumes: Tasks 1–6 after both review stages are clean.
- Produces: exact backend, migration/create-all, focused E2E, full-suite, privacy,
  and scope evidence for orchestrating-root review.

- [ ] **Step 1: Audit ancestry, scope, staging, and diagnostic preservation**

Run:

```bash
set -euo pipefail
git merge-base --is-ancestor 32075826694e1a24bfb36699fb14374b7916b29a HEAD
git log --oneline --decorate -8
git diff --check
git status --short --branch
git diff --name-only c85eb929482b3bcb80da610990f0fb61ea19e775
git -c core.quotepath=false ls-files --others --exclude-standard
git diff --cached --name-only
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test -f "$task_audit_identity_file" && test ! -L "$task_audit_identity_file"
test "$(stat -f '%u' "$task_audit_root")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_root")" = 700
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")" = \
  "$(cat "$task_audit_identity_file")"
test "$(cd -P -- "$task_audit_root" && pwd -P)" = "$task_audit_root"
for task_baseline_inventory in \
  ignored phase1 e2e-generated pytest-cache python-cache ruff-cache \
  typescript-cache next-output backend-data residual
do
  test -f "$task_audit_root/$task_baseline_inventory.before"
done
test -f "$task_audit_root/phase1-files.before.sha256"
test -f "$task_audit_root/backend-data.before.sha256"
test "$(cat "$task_audit_root/backend-venv.before")" = absent-before-behavior
test ! -e backend/.venv && test ! -L backend/.venv
test -f frontend/test-results/phase1-full-results.json
test "$(shasum -a 256 frontend/test-results/phase1-full-results.json | awk '{print $1}')" = \
  ef23b233169bbafe4735169fa031ba7a1dcb4630378f2035e06fa9ef6db47eb0
find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort
task_original_trace_count="$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort | wc -l | tr -d ' ')"
test "$task_original_trace_count" -eq 15
test "$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort | \
  while IFS= read -r task_trace_path; do
    shasum -a 256 "$task_trace_path"
  done | shasum -a 256 | awk '{print $1}')" = \
  1c8501bf87e94eb874b316271facbbf9af087e84b967ff4ad6bd764ef8a8c827
```

Expected: baseline ancestry exits 0; only the seven authorized design commits are
present after baseline; the plan and behavioral/test work are uncommitted;
staging is empty; every ordinary untracked path is allowlisted; the bounded
ignored baseline is intact; and the original JSON and exact 15-trace manifest
retain their recorded hashes.

- [ ] **Step 2: Create a fresh backend test DB immediately before final focused tests**

This is a fresh final integration gate, not reuse of Task 2/3 schema state. It
does not delete a prior database: if the distinct name exists, `createdb`
fails and the gate stops. The cleanup trap drops only after this invocation
records successful creation:

```bash
set -euo pipefail
task_backend_database=medtimeline_df11_strict_identity_backend_integration_test
task_created_backend_database=0
cleanup_backend_integration_database() {
  task_original_status=$?
  trap - EXIT
  task_cleanup_status=0
  if [ "$task_created_backend_database" -eq 1 ]; then
    dropdb -h 127.0.0.1 -p 5432 "$task_backend_database" || \
      task_cleanup_status=$?
  fi
  if [ "$task_original_status" -ne 0 ]; then
    exit "$task_original_status"
  fi
  exit "$task_cleanup_status"
}
trap cleanup_backend_integration_database EXIT
createdb -h 127.0.0.1 -p 5432 "$task_backend_database"
task_created_backend_database=1
cd backend
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_backend_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  NEXT_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1 \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_model_manager.py::test_start_with_inert_executable_does_not_spawn_worker \
  tests/test_processing_mode_snapshot.py::test_strict_admission_rejects_v1_before_release_evidence \
  tests/test_processing_mode_snapshot.py::test_structured_upload_rejects_legacy_strict_runtime_without_request_side_effects \
  tests/test_processing_mode_snapshot.py::test_summary_endpoint_rejects_legacy_strict_runtime_before_prompt_or_job \
  tests/test_upload.py::test_cloud_assisted_tracked_fhir_and_identical_reupload_finish_without_provider \
  tests/test_upload.py::test_cloud_assisted_synthetic_cda_finishes_without_provider \
  tests/test_local_ai_runtime_identity.py \
  tests/test_local_ai_ci_workflows.py
```

Expected: all selected tests pass. Record exact counts. No provider, release
evidence, production/model worker spawn, model call, or network call occurs.
The runtime-identity test module may invoke its bounded console-script
subprocess; that is not a production or model worker.

- [ ] **Step 3: Run fresh Alembic and create-all parity gates**

Use two new explicit loopback databases. Each creation fails closed if its name
already exists; cleanup drops only names this invocation successfully created:

```bash
set -euo pipefail
task_migration_database=medtimeline_df11_strict_identity_migration_parity_test
task_create_all_database=medtimeline_df11_strict_identity_create_all_parity_test
task_created_migration_database=0
task_created_create_all_database=0
cleanup_parity_databases() {
  task_original_status=$?
  trap - EXIT
  task_cleanup_status=0
  if [ "$task_created_create_all_database" -eq 1 ]; then
    dropdb -h 127.0.0.1 -p 5432 "$task_create_all_database" || \
      task_cleanup_status=$?
  fi
  if [ "$task_created_migration_database" -eq 1 ]; then
    dropdb -h 127.0.0.1 -p 5432 "$task_migration_database" || \
      task_cleanup_status=$?
  fi
  if [ "$task_original_status" -ne 0 ]; then
    exit "$task_original_status"
  fi
  exit "$task_cleanup_status"
}
trap cleanup_parity_databases EXIT
createdb -h 127.0.0.1 -p 5432 "$task_migration_database"
task_created_migration_database=1
createdb -h 127.0.0.1 -p 5432 "$task_create_all_database"
task_created_create_all_database=1
cd backend
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_migration_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m alembic upgrade head
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_create_all_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_processing_mode_snapshot.py::test_structured_upload_rejects_legacy_strict_runtime_without_request_side_effects
env \
  APP_ENV=test \
  DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_create_all_database" \
  DATABASE_ENCRYPTION_KEY=0000000000000000000000000000000000000000000000000000000000000000 \
  REAL_MEDICAL_FIXTURES_DIR= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  /Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv/bin/python \
  -m pytest -q \
  tests/test_local_ai_migrations.py::test_attested_manifest_guard_is_an_exact_migration_create_all_clone \
  tests/test_local_ai_migrations.py::test_attested_job_trigger_is_an_exact_migration_create_all_clone \
  tests/test_local_ai_migrations.py::test_legacy_validator_is_a_full_migration_create_all_clone
```

Expected: Alembic reaches head; the create-all-backed strict admission test
passes; the three exact SQL-clone tests pass. No migration or production model
is modified.

- [ ] **Step 4: Run frontend static gates**

Run:

```bash
set -euo pipefail
cd frontend
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/eslint \
  playwright.config.ts \
  e2e/helpers/api-client.ts \
  e2e/admin-records.spec.ts \
  e2e/dashboard-home.spec.ts \
  e2e/display-badges-providers.spec.ts \
  e2e/pagination-integrity.spec.ts \
  e2e/record-ai-metadata.spec.ts \
  e2e/record-detail-page.spec.ts \
  e2e/record-detail-sheet.spec.ts \
  e2e/record-renderers.spec.ts \
  e2e/setup.spec.ts \
  e2e/strict-local-admission.spec.ts \
  e2e/summaries.spec.ts \
  e2e/timeline.spec.ts \
  e2e/upload-dedup.spec.ts \
  e2e/upload-progress.spec.ts \
  e2e/upload-structured.spec.ts
```

Expected: both commands exit 0.

- [ ] **Step 5: Create the full-suite DB, enumerate, and run with one owned runtime**

The command below uses one new runtime root and one new output root for
enumeration and execution. The full-run token comes from the unique output
root, and both the list and JSON report include that token. Each report path is
proved absent before execution. The command tracks database and dependency-link
ownership, never removes a pre-existing link, and recursively deletes only the
runtime root after its real path and captured filesystem identity are
revalidated. It retains the exact output root for acceptance and audit without
touching `frontend/test-results/phase1-*` or the original traces:

```bash
set -euo pipefail
task_worktree_root="$(pwd -P)"
task_backend_link="$task_worktree_root/backend/.venv"
task_dependency_source=/Users/potalora/ai_workspace/test_autonomous_ai_web_records/backend/.venv
task_created_backend_link=0
task_backend_link_identity=""
task_full_database=medtimeline_df11_strict_identity_full_e2e_test
task_created_full_database=0
task_runtime_parent="$task_worktree_root/frontend/test-results/runtime"
task_output_parent="$task_worktree_root/frontend/test-results/executions"
task_runtime_root=""
task_output_root=""
task_runtime_parent_identity=""
task_output_parent_identity=""
task_runtime_root_identity=""
task_output_root_identity=""
safe_remove_full_runtime() {
  test -d "$task_runtime_parent" && test ! -L "$task_runtime_parent" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")" = \
    "$task_runtime_parent_identity" || return 1
  test "$(cd -P -- "$task_runtime_parent" && pwd -P)" = \
    "$task_runtime_parent" || return 1
  test -d "$task_runtime_root" && test ! -L "$task_runtime_root" || return 1
  test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")" = \
    "$task_runtime_root_identity" || return 1
  task_runtime_root_real="$(cd -P -- "$task_runtime_root" && pwd -P)" || return 1
  test "$task_runtime_root_real" = "$task_runtime_root" || return 1
  test "$(dirname -- "$task_runtime_root_real")" = \
    "$task_runtime_parent" || return 1
  rm -rf -- "$task_runtime_root"
}
cleanup_full_e2e() {
  task_original_status=$?
  trap - EXIT INT TERM
  task_cleanup_status=0
  if [ "$task_created_backend_link" -eq 1 ] && [ -L "$task_backend_link" ]; then
    if test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_backend_link")" = \
        "$task_backend_link_identity" && \
      test "$(readlink "$task_backend_link")" = "$task_dependency_source"; then
      unlink "$task_backend_link" || task_cleanup_status=$?
    else
      echo "Refusing to unlink changed backend dependency link" >&2
      task_cleanup_status=1
    fi
  elif [ "$task_created_backend_link" -eq 1 ]; then
    echo "Owned backend dependency link disappeared or changed type" >&2
    task_cleanup_status=1
  fi
  if [ -n "$task_runtime_root" ]; then
    safe_remove_full_runtime || task_cleanup_status=$?
  fi
  if [ -n "$task_output_root" ]; then
    test -d "$task_output_parent" && test ! -L "$task_output_parent" || \
      task_cleanup_status=1
    test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")" = \
      "$task_output_parent_identity" || task_cleanup_status=1
    test -d "$task_output_root" && test ! -L "$task_output_root" || \
      task_cleanup_status=1
    test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")" = \
      "$task_output_root_identity" || task_cleanup_status=1
    test "$(cd -P -- "$task_output_root" && pwd -P)" = \
      "$task_output_root" || task_cleanup_status=1
  fi
  if [ "$task_created_full_database" -eq 1 ]; then
    dropdb -h 127.0.0.1 -p 5432 "$task_full_database" || \
      task_cleanup_status=$?
  fi
  if [ "$task_original_status" -ne 0 ]; then
    exit "$task_original_status"
  fi
  exit "$task_cleanup_status"
}
trap cleanup_full_e2e EXIT
if [ ! -e "$task_backend_link" ] && [ ! -L "$task_backend_link" ]; then
  test -x "$task_dependency_source/bin/python"
  ln -s "$task_dependency_source" "$task_backend_link"
  task_created_backend_link=1
  task_backend_link_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_backend_link")"
fi
test -x "$task_backend_link/bin/python"
test -d "$task_worktree_root/frontend" && test ! -L "$task_worktree_root/frontend"
test -d "$task_worktree_root/frontend/test-results" && \
  test ! -L "$task_worktree_root/frontend/test-results"
if [ ! -e "$task_runtime_parent" ] && [ ! -L "$task_runtime_parent" ]; then
  mkdir -m 700 -- "$task_runtime_parent"
fi
if [ ! -e "$task_output_parent" ] && [ ! -L "$task_output_parent" ]; then
  mkdir -m 700 -- "$task_output_parent"
fi
for task_parent in "$task_runtime_parent" "$task_output_parent"; do
  test -d "$task_parent" && test ! -L "$task_parent"
  test "$(stat -f '%u' "$task_parent")" -eq "$(id -u)"
  test "$(cd -P -- "$task_parent" && pwd -P)" = "$task_parent"
done
task_runtime_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_parent")"
task_output_parent_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")"
umask 077
task_runtime_root="$(mktemp -d "$task_runtime_parent/full.XXXXXX")"
task_output_root="$(mktemp -d "$task_output_parent/full.XXXXXX")"
for task_root in "$task_runtime_root" "$task_output_root"; do
  test -d "$task_root" && test ! -L "$task_root"
  test "$(stat -f '%u' "$task_root")" -eq "$(id -u)"
  test "$(stat -f '%Lp' "$task_root")" = 700
  test "$(cd -P -- "$task_root" && pwd -P)" = "$task_root"
done
task_runtime_root_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_runtime_root")"
task_output_root_identity="$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")"
task_run_token="$(basename -- "$task_output_root")"
task_list_report="$task_output_root/full-list-$task_run_token.txt"
task_json_report="$task_output_root/full-results-$task_run_token.json"
test ! -e "$task_list_report" && test ! -L "$task_list_report"
test ! -e "$task_json_report" && test ! -L "$task_json_report"
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")" = \
  "$(cat "$task_audit_identity_file")"
task_full_output_record="$task_audit_root/approved-full-output"
test ! -e "$task_full_output_record" && test ! -L "$task_full_output_record"
printf '%s\n%s\n%s\n%s\n%s\n%s\n' \
  "$task_output_root" "$task_output_parent_identity" \
  "$task_output_root_identity" "$task_run_token" \
  "$task_list_report" "$task_json_report" > "$task_full_output_record"
test -f "$task_full_output_record" && test ! -L "$task_full_output_record"
test "$(stat -f '%u' "$task_full_output_record")" -eq "$(id -u)"
createdb -h 127.0.0.1 -p 5432 "$task_full_database"
task_created_full_database=1
cd frontend
env -u DATABASE_URL -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_full_database" \
  ./node_modules/.bin/playwright test --list \
  > "$task_list_report"
test -f "$task_list_report" && test ! -L "$task_list_report"
env -u DATABASE_URL -u DATABASE_ENCRYPTION_KEY -u UPLOAD_DIR -u TEMP_EXTRACT_DIR \
  APP_ENV=test REAL_MEDICAL_FIXTURES_DIR= E2E_ATTESTED_STRICT_PACK= \
  GEMINI_API_KEY= GOOGLE_API_KEY= OPENAI_API_KEY= OPENROUTER_API_KEY= \
  ANTHROPIC_API_KEY= VERTEX_PROJECT= GOOGLE_CLOUD_PROJECT= \
  GOOGLE_APPLICATION_CREDENTIALS= \
  LLM_PROVIDER=gemini LLM_SUMMARY_PROVIDER= LLM_SECTION_PROVIDER= \
  LLM_DEDUP_PROVIDER= LLM_EXTRACTION_PROVIDER= \
  E2E_LOCAL_ONLY=1 \
  E2E_RUNTIME_ROOT="$task_runtime_root" \
  E2E_OUTPUT_ROOT="$task_output_root" \
  E2E_DATABASE_URL="postgresql+asyncpg://127.0.0.1:5432/$task_full_database" \
  PLAYWRIGHT_JSON_OUTPUT_FILE="$task_json_report" \
  ./node_modules/.bin/playwright test \
  --workers=1 --reporter=list,json --trace=retain-on-failure
test -f "$task_json_report" && test ! -L "$task_json_report"
```

Record the exact token and report paths from `approved-full-output`. Do not
encode the enumeration into a test or acceptance rule. The output root remains
ignored and uncommitted for the acceptance parser and final evidence audit.

- [ ] **Step 6: Apply the exact full-suite acceptance matrix**

Acceptance is zero failures and zero not-run tests. Every other case passes.
Skips may be only these two private-fixture families and three real-pack summary
cases:

Read the reporter output without modifying it and make the distinction between
an intentional skip and a test that Playwright did not run explicit:

```bash
set -euo pipefail
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
task_full_output_record="$task_audit_root/approved-full-output"
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")" = \
  "$(cat "$task_audit_identity_file")"
test -f "$task_full_output_record" && test ! -L "$task_full_output_record"
task_output_root="$(sed -n '1p' "$task_full_output_record")"
task_output_parent_identity="$(sed -n '2p' "$task_full_output_record")"
task_output_root_identity="$(sed -n '3p' "$task_full_output_record")"
task_run_token="$(sed -n '4p' "$task_full_output_record")"
task_list_report="$(sed -n '5p' "$task_full_output_record")"
task_json_report="$(sed -n '6p' "$task_full_output_record")"
task_output_parent="$(dirname -- "$task_output_root")"
test -d "$task_output_parent" && test ! -L "$task_output_parent"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")" = \
  "$task_output_parent_identity"
test -d "$task_output_root" && test ! -L "$task_output_root"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")" = \
  "$task_output_root_identity"
test "$(cd -P -- "$task_output_root" && pwd -P)" = "$task_output_root"
test "$(dirname -- "$task_list_report")" = "$task_output_root"
test "$(dirname -- "$task_json_report")" = "$task_output_root"
test "$(basename -- "$task_list_report")" = "full-list-$task_run_token.txt"
test "$(basename -- "$task_json_report")" = "full-results-$task_run_token.json"
test -f "$task_list_report" && test ! -L "$task_list_report"
test -f "$task_json_report" && test ! -L "$task_json_report"
TASK_LIST_REPORT="$task_list_report" TASK_JSON_REPORT="$task_json_report" node <<'NODE'
const fs = require("node:fs");
const list = fs.readFileSync(process.env.TASK_LIST_REPORT, "utf8");
const report = JSON.parse(fs.readFileSync(process.env.TASK_JSON_REPORT, "utf8"));
const cases = [];
function visit(suite, parents = []) {
  const nextParents = suite.title.endsWith(".spec.ts")
    ? parents
    : [...parents, suite.title].filter(Boolean);
  for (const spec of suite.specs ?? []) {
    for (const test of spec.tests ?? []) {
      cases.push({
        title: [...nextParents, spec.title].join(" › "),
        expectedStatus: test.expectedStatus,
        status: test.status,
      });
    }
  }
  for (const child of suite.suites ?? []) visit(child, nextParents);
}
for (const suite of report.suites) visit(suite);

const intentionalSkips = cases.filter(
  (item) => item.expectedStatus === "skipped" && item.status === "skipped"
);
const notRun = cases.filter(
  (item) => item.expectedStatus !== "skipped" && item.status === "skipped"
);
const failed = cases.filter((item) => item.status === "unexpected");
const passed = cases.filter((item) => item.status === "expected");
const unaccepted = cases.filter(
  (item) => !["expected", "unexpected", "skipped"].includes(item.status)
);
const allowedSkips = [
  "Cross-format dedup › CDA then FHIR upload detects cross-format duplicates",
  "Structured file uploads › upload XDM/CDA ZIP package",
  "Structured file uploads › upload standalone CDA XML",
  "Extraction progress tracking › extraction progress counts are accurate for batch upload",
  "Mixed content upload classification › unstructured upload goes to extraction pipeline",
  "Unstructured Upload › upload single RTF via API",
  "Unstructured Upload › upload batch RTFs via UI",
  "Unstructured Upload › upload PDF via API",
  "Unstructured Upload › extraction progress tracking via API",
  "Duplicate file upload (idempotency) › re-uploading identical unstructured file returns duplicate_file",
  "Summaries page › generate produces a result",
  "Summaries page › history entry reopens a saved summary without regenerating",
  "Summaries page › generation reports the selected privacy boundary",
].sort();
const actualSkips = intentionalSkips.map((item) => item.title).sort();
const listedTotalMatch = list.match(/Total:\s+(\d+)\s+tests?\s+in\s+/);
if (!listedTotalMatch) throw new Error("Playwright list total is missing");
const listedTotal = Number(listedTotalMatch[1]);
console.log({
  listed: listedTotal,
  enumerated: cases.length,
  passed: passed.length,
  skipped: intentionalSkips.length,
  failed: failed.length,
  notRun: notRun.length,
  unaccepted: unaccepted.length,
  skipTitles: actualSkips,
});
if (listedTotal !== cases.length) process.exit(1);
if (failed.length !== 0 || notRun.length !== 0 || unaccepted.length !== 0) {
  process.exit(1);
}
if (JSON.stringify(actualSkips) !== JSON.stringify(allowedSkips)) process.exit(1);
NODE
```

The parser consumes only the token-bound paths recorded by Step 5 and compares
the reporter case count with that invocation's exact `--list` total. Treat an
`expectedStatus` other than `skipped` with final `status: "skipped"` as not-run,
not as a legitimate skip.

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

Report the exact enumerated, passed, skipped, failed, and not-run counts from the
new run. Do not predict the passed count. If anything else skips or fails,
reproduce that exact test on the same code and DB. Compare its title/status/error
with the preserved baseline result before classifying it as unrelated.

- [ ] **Step 7: Run final privacy, scope, and staged-content checks**

Run:

```bash
set -euo pipefail
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
task_full_output_record="$task_audit_root/approved-full-output"
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test -f "$task_audit_identity_file" && test ! -L "$task_audit_identity_file"
test "$(stat -f '%u' "$task_audit_root")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_root")" = 700
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")" = \
  "$(cat "$task_audit_identity_file")"
test "$(cd -P -- "$task_audit_root" && pwd -P)" = "$task_audit_root"
test -f "$task_full_output_record" && test ! -L "$task_full_output_record"
task_output_root="$(sed -n '1p' "$task_full_output_record")"
task_output_parent_identity="$(sed -n '2p' "$task_full_output_record")"
task_output_root_identity="$(sed -n '3p' "$task_full_output_record")"
task_run_token="$(sed -n '4p' "$task_full_output_record")"
task_list_report="$(sed -n '5p' "$task_full_output_record")"
task_json_report="$(sed -n '6p' "$task_full_output_record")"
task_output_parent="$(dirname -- "$task_output_root")"
test -d "$task_output_parent" && test ! -L "$task_output_parent"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_parent")" = \
  "$task_output_parent_identity"
test -d "$task_output_root" && test ! -L "$task_output_root"
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_output_root")" = \
  "$task_output_root_identity"
test "$(cd -P -- "$task_output_root" && pwd -P)" = "$task_output_root"
test "$(dirname -- "$task_list_report")" = "$task_output_root"
test "$(dirname -- "$task_json_report")" = "$task_output_root"
test "$(basename -- "$task_list_report")" = "full-list-$task_run_token.txt"
test "$(basename -- "$task_json_report")" = "full-results-$task_run_token.json"
test -f "$task_list_report" && test ! -L "$task_list_report"
test -f "$task_json_report" && test ! -L "$task_json_report"

capture_ignored_inventory() {
  task_inventory_suffix="$1"
  task_inventory_all="$task_audit_root/ignored.$task_inventory_suffix"
  git -c core.quotepath=false ls-files \
    --others --ignored --exclude-standard |
    awk '
      !/^frontend\/node_modules\// &&
      !/^backend\/\.venv(\/|$)/ { print }
    ' | LC_ALL=C sort > "$task_inventory_all"
  for task_inventory_class in \
    phase1 e2e-generated pytest-cache python-cache ruff-cache \
    typescript-cache next-output backend-data residual
  do
    : > "$task_audit_root/$task_inventory_class.$task_inventory_suffix"
  done
  while IFS= read -r task_ignored_path; do
    case "$task_ignored_path" in
      frontend/test-results/runtime/*|frontend/test-results/executions/*)
        task_inventory_class=e2e-generated
        ;;
      frontend/test-results/*)
        task_inventory_class=phase1
        ;;
      */.pytest_cache/*)
        task_inventory_class=pytest-cache
        ;;
      */__pycache__/*|*.pyc)
        task_inventory_class=python-cache
        ;;
      backend/.ruff_cache/*)
        task_inventory_class=ruff-cache
        ;;
      frontend/tsconfig.tsbuildinfo)
        task_inventory_class=typescript-cache
        ;;
      frontend/.next/*|frontend/next-env.d.ts)
        task_inventory_class=next-output
        ;;
      backend/data/*)
        task_inventory_class=backend-data
        ;;
      *)
        task_inventory_class=residual
        ;;
    esac
    printf '%s\n' "$task_ignored_path" \
      >> "$task_audit_root/$task_inventory_class.$task_inventory_suffix"
  done < "$task_inventory_all"
}

capture_ignored_inventory after
for task_generated_root in \
  frontend/test-results/runtime frontend/test-results/executions
do
  if [ -e "$task_generated_root" ] || [ -L "$task_generated_root" ]; then
    find "$task_generated_root" -print
  fi
done | LC_ALL=C sort > "$task_audit_root/e2e-tree.after"
find frontend/test-results \
  \( -path frontend/test-results/runtime \
     -o -path frontend/test-results/executions \) -prune \
  -o -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 > "$task_audit_root/phase1-files.after.sha256"
find backend/data -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 > "$task_audit_root/backend-data.after.sha256"

cmp "$task_audit_root/phase1.before" "$task_audit_root/phase1.after"
cmp "$task_audit_root/residual.before" "$task_audit_root/residual.after"
cmp "$task_audit_root/backend-data.before" "$task_audit_root/backend-data.after"
cmp "$task_audit_root/phase1-files.before.sha256" \
  "$task_audit_root/phase1-files.after.sha256"
cmp "$task_audit_root/backend-data.before.sha256" \
  "$task_audit_root/backend-data.after.sha256"
test ! -s <(comm -23 \
  "$task_audit_root/e2e-generated.before" \
  "$task_audit_root/e2e-generated.after")
task_approved_output_relative="${task_output_root#"$(pwd -P)/"}"
test "$task_approved_output_relative" != "$task_output_root"
comm -13 \
  "$task_audit_root/e2e-generated.before" \
  "$task_audit_root/e2e-generated.after" \
  > "$task_audit_root/e2e-generated.added"
while IFS= read -r task_added_e2e_path; do
  case "$task_added_e2e_path" in
    "$task_approved_output_relative"/*) ;;
    *)
      echo "Unexpected generated E2E path: $task_added_e2e_path" >&2
      exit 1
      ;;
  esac
done < "$task_audit_root/e2e-generated.added"
test ! -s <(comm -23 \
  "$task_audit_root/e2e-tree.before" "$task_audit_root/e2e-tree.after")
comm -13 "$task_audit_root/e2e-tree.before" "$task_audit_root/e2e-tree.after" \
  > "$task_audit_root/e2e-tree.added"
while IFS= read -r task_added_tree_path; do
  case "$task_added_tree_path" in
    frontend/test-results/runtime|frontend/test-results/executions|\
    "$task_approved_output_relative"|"$task_approved_output_relative"/*) ;;
    *)
      echo "Unexpected runtime/execution tree path: $task_added_tree_path" >&2
      exit 1
      ;;
  esac
done < "$task_audit_root/e2e-tree.added"
if [ -d frontend/test-results/runtime ]; then
  test ! -L frontend/test-results/runtime
  test -z "$(find frontend/test-results/runtime -mindepth 1 -print -quit)"
fi
for task_cache_class in \
  pytest-cache python-cache ruff-cache typescript-cache next-output
do
  echo "Added $task_cache_class paths:"
  comm -13 \
    "$task_audit_root/$task_cache_class.before" \
    "$task_audit_root/$task_cache_class.after"
  echo "Removed $task_cache_class paths:"
  comm -23 \
    "$task_audit_root/$task_cache_class.before" \
    "$task_audit_root/$task_cache_class.after"
done

git diff --check
git status --short --branch
git diff --name-only c85eb929482b3bcb80da610990f0fb61ea19e775
git -c core.quotepath=false ls-files --others --exclude-standard
git -c core.quotepath=false ls-files --others --ignored --exclude-standard |
  awk '
    !/^frontend\/node_modules\// &&
    !/^backend\/\.venv(\/|$)/ { print }
  '
git diff --cached --name-only
test "$(shasum -a 256 frontend/test-results/phase1-full-results.json | awk '{print $1}')" = \
  ef23b233169bbafe4735169fa031ba7a1dcb4630378f2035e06fa9ef6db47eb0
task_original_trace_count="$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort | wc -l | tr -d ' ')"
test "$task_original_trace_count" -eq 15
test "$(find frontend/test-results \
  -path frontend/test-results/executions -prune -o \
  -name trace.zip -print | sort | \
  while IFS= read -r task_trace_path; do
    shasum -a 256 "$task_trace_path"
  done | shasum -a 256 | awk '{print $1}')" = \
  1c8501bf87e94eb874b316271facbbf9af087e84b967ff4ad6bd764ef8a8c827
if rg -n \
  "(REAL_MEDICAL_FIXTURES_DIR|GEMINI_API_KEY|GOOGLE_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|OPENROUTER_API_KEY)=[^[:space:]]" \
  docs/superpowers/plans/2026-08-23-strict-local-e2e-runtime-identity.md \
  frontend/playwright.config.ts \
  frontend/e2e \
  backend/tests/test_processing_mode_snapshot.py \
  backend/tests/test_upload.py \
  docs/operations-strict-local-ai.md
then
  echo "Forbidden nonempty credential or private-fixture assignment" >&2
  exit 1
fi
```

Expected: no whitespace errors; only allowlisted tracked and ordinary untracked
changes; staging empty; phase-1 and backend-data inventories and bytes
unchanged; every new E2E artifact under the exact approved token-bound output
root; runtime roots gone; Python, pytest, Ruff, TypeScript, and Next cache
changes printed as bounded classes; residual ignored paths byte-for-byte equal
to baseline; and no nonempty credential/private path. The all-zero test key may
appear only in test-profile/test-command contexts.

- [ ] **Step 8: Verify cleanup and retain the owned audit evidence for review**

Every database-owning command has already run its local conditional cleanup
trap. Do not issue a final `dropdb`: a name alone cannot prove this shell owns a
database. Query PostgreSQL read-only and fail if any exact task-owned fixed
name remains. Query the public example's literal `medtimeline_e2e_` prefix
separately as an unowned-residue report. This plan does not execute that
example, so a prefix match is never treated as ownership or deletion authority:

```bash
set -euo pipefail
task_remaining_owned_databases="$(psql -X -qAt \
  -h 127.0.0.1 -p 5432 -d postgres -c \
  "SELECT datname FROM pg_database WHERE datname IN (\
  'medtimeline_df11_strict_identity_startup_e2e',\
  'medtimeline_df11_strict_identity_admission_test',\
  'medtimeline_df11_strict_identity_provider_test',\
  'medtimeline_df11_strict_identity_task4_e2e',\
  'medtimeline_df11_strict_identity_task5_e2e',\
  'medtimeline_df11_strict_identity_backend_integration_test',\
  'medtimeline_df11_strict_identity_migration_parity_test',\
  'medtimeline_df11_strict_identity_create_all_parity_test',\
  'medtimeline_df11_strict_identity_full_e2e_test'\
  ) ORDER BY datname")"
test -z "$task_remaining_owned_databases"

task_unowned_public_prefix_databases="$(psql -X -qAt \
  -h 127.0.0.1 -p 5432 -d postgres -c \
  "SELECT datname || '|' || oid::text || '|' || datdba::text || '|' || \
  pg_get_userbyid(datdba) FROM pg_database WHERE \
  left(datname, length('medtimeline_e2e_')) = 'medtimeline_e2e_' \
  ORDER BY datname")"
printf 'Unowned public-prefix database residue (read-only):\n%s\n' \
  "$task_unowned_public_prefix_databases"
```

Step 0 independently records that the Phase-1 diagnostic `backend/.venv` link
is absent before behavioral work; the plan does not rely on a historical
ownership claim. Every later command unlinks only when its invocation-local
creation flag is `1`. At final cleanup, verify the recorded baseline and inspect
but do not remove an unowned path:

```bash
set -euo pipefail
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
test "$(cat "$task_audit_root/backend-venv.before")" = absent-before-behavior
if [ -e backend/.venv ] || [ -L backend/.venv ]; then
  ls -ld backend/.venv
else
  echo "No backend/.venv link remains in the worktree"
fi
```

If a link remains unexpectedly, report it to the root. Never remove it merely
because it is a symlink. Retain the approved full-run output root and report its
exact path; do not stage it. Retain the external audit root through root review
after revalidating its exact real path and creation identity:

```bash
set -euo pipefail
task_audit_root=/private/tmp/medtimeline-df11-strict-identity-audit
task_audit_identity_file=/private/tmp/medtimeline-df11-strict-identity-audit.identity
test -d "$task_audit_root" && test ! -L "$task_audit_root"
test -f "$task_audit_identity_file" && test ! -L "$task_audit_identity_file"
test "$(stat -f '%u' "$task_audit_root")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_root")" = 700
test "$(stat -f '%u' "$task_audit_identity_file")" -eq "$(id -u)"
test "$(stat -f '%Lp' "$task_audit_identity_file")" = 600
test "$(stat -f '%d:%i:%u:%HT:%Lp' "$task_audit_root")" = \
  "$(cat "$task_audit_identity_file")"
test "$(cd -P -- "$task_audit_root" && pwd -P)" = "$task_audit_root"
test "$(dirname -- "$task_audit_root")" = /private/tmp
printf 'Retained audit evidence: %s %s\n' \
  "$task_audit_root" "$task_audit_identity_file"
```

Do not delete original or final traces/results; keep them ignored and
uncommitted.

- [ ] **Step 9: Return the uncommitted implementation for root approval**

Report:

- baseline SHA and all design commit SHAs;
- exact diagnosis and confirmation that production files are unchanged;
- RED/GREEN outputs for each task;
- backend, runtime-identity, migration, and create-all counts;
- focused and full Playwright enumeration/pass/skip/fail/not-run counts;
- all exact legitimate skips;
- any exact baseline comparison evidence;
- final allowlisted changed-file list and empty staged area;
- humanizer audit result for the public document;
- runtime-root/database/dependency-link cleanup result.

Pause. Do not commit, push, open a PR, merge, deploy, or change GitHub metadata
until the orchestrating root separately authorizes publication.

## Plan self-review checklist

- Spec coverage: each revised-design requirement maps to Tasks 1–7.
- Startup: no empty worker binding; `/usr/bin/false` is normalized, never
  spawned, and never described as attested.
- Summary honesty: seven UI tests run; exactly three named model cases are
  pack-gated; exact HTTP 400 has backend coverage.
- Provider-free proof: real scheduling, non-cancelling drain, terminal state,
  tracked FHIR, identical re-upload, synthetic CDA, zero `get_provider` calls,
  and failure-safe task draining.
- Provider routing: every profile and command clears operation overrides and
  selects credential-bound Gemini, so inherited loopback Ollama/LM Studio
  routes cannot escape the intended proof.
- Helper boundary: no public positive path parameter; generated pagination uses
  content; generic path use remains exactly two private CDA calls.
- Filesystem isolation: each Playwright command creates absolute owned runtime
  and output roots; profile overwrites upload/temp/scratch/model paths; browser
  reads effective `UPLOAD_DIR`; cleanup revalidates real containment plus the
  captured parent/root device and inode; each run's output cannot clear Phase 1
  evidence.
- Database consistency: every lifecycle command specifies
  `-h 127.0.0.1 -p 5432`; every backend pytest/Alembic command sets
  `APP_ENV=test`; every creation fails if the name exists, sets an immediate
  local ownership flag, and is dropped only by that command's trap.
- Dependency ownership: every link command tracks creation and only unlinks its
  own link; Step 0 and final inspection independently account for the absent
  Phase-1 diagnostic link without treating historical provenance as cleanup
  authority.
- Public command ownership: the documented database is dropped only if that
  invocation created it; its runtime and output roots use real-path and
  filesystem-identity cleanup guards.
- Diagnostic evidence: both scope gates enumerate ordinary and bounded ignored
  paths, compare explicit Python, pytest, Ruff, TypeScript, Next, and E2E
  generated classes, and verify the original JSON hash plus exact 15-trace
  manifest hash.
- Database residue: exact task-owned fixed names must be absent. Public-example
  `medtimeline_e2e_` prefix matches are reported read-only and never treated as
  ownership or deletion authority.
- Acceptance: no predicted passed count; zero failures/not-run; exact private
  and real-pack skip titles are enumerated.
- Scope: no production, migration, manifest, release, catalog, worker, pack, or
  artifact file enters the allowlist.
- Placeholder scan: no deferred marker, alternative test path, or unresolved
  file choice remains.
- Publication: only design-document commits are authorized; the plan and later
  implementation remain uncommitted.
