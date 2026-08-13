# Worker Runtime Attestation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind strict-local validation and every worker spawn to the exact corrected worker source and locked dependencies, then promote a fresh v2 Apple model-pack release.

**Architecture:** Compute a portable SHA-256 identity over the worker package, `pyproject.toml`, `uv.lock`, console entry point, and identity-scheme version. Put that digest in a schema-v2 manifest and validation receipt, compare it before fixture verification and immediately before each worker spawn, and reject v1 or drifted snapshots before PHI enters the worker. Keep the Alembic and create-all database guards identical, then regenerate release evidence instead of relabeling v1 artifacts.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2.x async, PostgreSQL 16, Alembic, Apple MLX worker, JSON Schema 2020-12, pytest, Ruff.

## Global Constraints

- This plan starts only after the final Track B extraction-remediation commit is present in this branch.
- Validated strict-local work must branch before any cloud-capable provider is constructed and must never fall back to cloud.
- Runtime identity and errors contain digests and stable codes only; never persist or return paths, source text, prompts, model output, or patient data.
- The operating-system owner/root threat boundary is explicit: attestation detects stale or drifted runtime inputs, not a privileged attacker replacing code and trust metadata together.
- Existing schema-v1 receipts and snapshots may be parsed only to report `runtime_identity_required`; they cannot admit or execute work.
- Deploy only with no active strict-local jobs. Do not rewrite old queued snapshots to claim a v2 identity.
- Alembic and `Base.metadata.create_all()` strict-local guards must remain semantically identical. Literal SQL percent signs in SQLAlchemy `DDL` strings use `%%`.
- Ordinary tests use synthetic files and fake workers only. Do not use private medical fixtures, provider calls, model downloads, or live cloud tests.
- Subagents do not commit. The root agent reviews and verifies each task. Every commit block is a
  proposed checkpoint only; do not stage or commit unless Pedro explicitly authorizes it.
- Start this plan in a new Codex-managed **Worktree** task only after the accepted Track B branch
  is available. Select that Track B branch as the starting branch; retain the completed work as
  `codex/worker-runtime-attestation` only when it is ready for root review.

---

### Task 1: Canonical worker-bundle identity and schema-v2 manifest

**Files:**
- Create: `backend/app/services/local_ai/runtime_identity.py`
- Create: `backend/app/model_manifests/schema-v2.json`
- Modify: `backend/app/services/local_ai/manifest.py`
- Modify: `backend/app/config.py`
- Modify: `.env.example`
- Test: `backend/tests/test_local_ai_runtime_identity.py`
- Test: `backend/tests/test_local_ai_manifest.py`

**Interfaces:**
- Produces: `WORKER_IDENTITY_SCHEME = "local-ai-worker-bundle.v1"`.
- Produces: `WorkerRuntimeIdentity(scheme: str, bundle_sha256: str)`.
- Produces: `resolve_worker_runtime_identity(command, worker_project_dir) -> WorkerRuntimeIdentity`.
- Produces: `require_manifest_runtime_identity(manifest, observed) -> None`.
- Produces: schema-v2 `runtime = {name, version, worker_identity_scheme, worker_bundle_sha256}`.
- Consumes: `Settings.local_ai_worker_command` and new `Settings.local_ai_worker_project_dir`.

- [ ] **Step 1: Write failing portable-identity tests**

Create both an editable fixture (the worker package resolves from
`src/local_ai_mlx_worker`) and an installed fixture (the package resolves from
`.venv/lib/python*/site-packages/local_ai_mlx_worker`). Both fixtures have the
exact regular, non-symlink launcher
`.venv/bin/local-ai-mlx-worker`, `pyproject.toml`, and `uv.lock`. Add concrete
tests equivalent to:

```python
def test_worker_bundle_identity_is_stable_and_path_independent(tmp_path: Path) -> None:
    left = build_worker_project(tmp_path / "left")
    right = build_worker_project(tmp_path / "right")

    left_identity = resolve_worker_runtime_identity(
        left.command, left.project_dir
    )
    right_identity = resolve_worker_runtime_identity(
        right.command, right.project_dir
    )

    assert left_identity == right_identity
    assert left_identity.scheme == "local-ai-worker-bundle.v1"
    assert re.fullmatch(r"[0-9a-f]{64}", left_identity.bundle_sha256)


@pytest.mark.parametrize("relative_path", [
    "effective_package/local_ai_mlx_worker/nuextract3.py",
    "pyproject.toml",
    "uv.lock",
])
def test_worker_bundle_identity_changes_with_runtime_input(
    tmp_path: Path, relative_path: str
) -> None:
    project = build_worker_project(tmp_path / "worker")
    before = resolve_worker_runtime_identity(project.command, project.project_dir)
    target = project.project_dir / relative_path
    target.write_bytes(target.read_bytes() + b"\n# changed\n")

    after = resolve_worker_runtime_identity(project.command, project.project_dir)

    assert after.bundle_sha256 != before.bundle_sha256
```

For the non-editable fixture, mutate a source-tree file and prove it does not
change the identity; mutate the installed package file and prove it does. Also
assert rejection of a symlink launcher, a symlink selected source/package
`.py` file, a non-regular selected `.py` file, a launcher outside
`<project>/.venv/bin`, a launcher with a different basename, an entry-point
declaration other than the exact value below, duplicate or noncanonical
relative paths, and a missing lock file. Add a regression which creates
`effective_package/local_ai_mlx_worker/__pycache__/nuextract3.cpython-311.pyc`
after the first identity calculation and asserts the digest is unchanged: a
normal worker must remain attestable after Python writes bytecode caches.

- [ ] **Step 2: Run identity tests and verify RED**

Run:

```bash
cd backend
uv run pytest -q tests/test_local_ai_runtime_identity.py
```

Expected: collection fails because `app.services.local_ai.runtime_identity`
does not exist.

- [ ] **Step 3: Implement bounded canonical hashing**

Implement the public boundary with these exact data shapes:

```python
WORKER_IDENTITY_SCHEME = "local-ai-worker-bundle.v1"
WORKER_ENTRY_POINT = "local-ai-mlx-worker=local_ai_mlx_worker.__main__:main"
_MAX_IDENTITY_FILES = 64
_MAX_IDENTITY_FILE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class WorkerRuntimeIdentity:
    scheme: str
    bundle_sha256: str


def resolve_worker_runtime_identity(
    command: str | Sequence[str],
    worker_project_dir: str | Path,
) -> WorkerRuntimeIdentity:
    project = _regular_real_directory(Path(worker_project_dir))
    executable = _resolve_worker_executable(command)
    expected_bin = _regular_real_directory(project / ".venv" / "bin")
    expected_launcher = expected_bin / "local-ai-mlx-worker"
    if executable != expected_launcher or executable.is_symlink():
        raise LocalWorkerError("Local worker runtime identity is unavailable.")

    files = [project / "pyproject.toml", project / "uv.lock"]
    _require_exact_console_script(project / "pyproject.toml", WORKER_ENTRY_POINT)
    package_root = _resolve_effective_worker_package(project)
    files.extend(_regular_python_tree(package_root))
    entries = [_hash_identity_file(project, path) for path in files]
    payload = {
        "scheme": WORKER_IDENTITY_SCHEME,
        "entry_point": WORKER_ENTRY_POINT,
        "files": entries,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return WorkerRuntimeIdentity(
        scheme=WORKER_IDENTITY_SCHEME,
        bundle_sha256=hashlib.sha256(encoded).hexdigest(),
    )
```

`_resolve_worker_executable()` accepts only the configured command whose first
token is that exact launcher and whose remaining tokens match an explicitly
allowlisted fixed argument list (currently empty); it does not follow PATH,
resolve a symlink, or accept a shell string. `_resolve_effective_worker_package()`
must inspect the worker environment, not merely the checkout: for an editable
install it verifies the installed `.pth`/direct-url metadata resolves exactly
to `project/src/local_ai_mlx_worker`; for a non-editable install it verifies
the regular package directory under that environment's `site-packages`. Hash
only the resulting effective package tree's selected regular, non-symlink
`.py` files, plus `pyproject.toml` and `uv.lock`. Do not descend into an
`__pycache__` directory or hash `.pyc` files: they are normal runtime byproducts
and are not worker source. Require every traversed directory to be a regular
non-symlink directory, and reject any symlink or non-regular selected input;
unselected non-source members never enter the canonical payload.

Open every identity file with `O_NOFOLLOW`, require a regular owner-readable
file, bound file count and bytes, hash while reading, and compare pre/post
`fstat` metadata. Persist only normalized relative paths and file digests in
the temporary canonical payload; return only the aggregate digest. Do not hash
the launcher bytes (its shebang can contain a machine-specific path); instead
verify the exact launcher location, regular-file mode, and console-script
declaration before hashing the portable inputs.

- [ ] **Step 4: Add strict schema-v2 parsing**

Change `SCHEMA_VERSION` to `2`, define runtime keys exactly as:

```python
_RUNTIME_KEYS = frozenset(
    {"name", "version", "worker_identity_scheme", "worker_bundle_sha256"}
)
```

Require `worker_identity_scheme == WORKER_IDENTITY_SCHEME` and a lowercase
64-character digest. Update parser/load/canonicalization docstrings to v2.
Create `schema-v2.json` from the v1 schema with `schema_version.const = 2` and
the two required runtime properties. It is a source-controlled compatibility
and reviewer-facing schema, while `manifest.py` remains the runtime parsing
authority; do not leave it as an untested duplicate artifact. Add a parity
test that loads it and asserts its schema version, exact runtime `required`
set, `additionalProperties: false`, and identity-scheme constant match the
Python parser's `_RUNTIME_KEYS` and `WORKER_IDENTITY_SCHEME`; run the same
accept/reject fixtures through the parser. Keep `schema-v1.json` unchanged so
old evidence can be diagnosed rather than rewritten.

Add `local_ai_worker_project_dir` to settings:

```python
local_ai_worker_project_dir: str = "../workers/local_ai/apple_mlx"
```

Document `LOCAL_AI_WORKER_PROJECT_DIR` beside `LOCAL_AI_WORKER_COMMAND` in
`.env.example`.

- [ ] **Step 5: Add manifest v2 tests and verify GREEN**

Update manifest fixtures to schema v2 and assert:

```python
assert manifest.runtime == {
    "name": "mlx-vlm",
    "version": "0.5.0",
    "worker_identity_scheme": "local-ai-worker-bundle.v1",
    "worker_bundle_sha256": "a" * 64,
}
```

Assert v1, missing identity, extra runtime keys, uppercase digest, and wrong
scheme fail with the existing safe `LocalValidationError` boundary on ordinary
admission. Separately add a diagnostic-only parser path such as
`parse_manifest(..., allow_legacy_diagnostic=True)`: it can canonicalize a v1
snapshot solely to identify it as needing revalidation, never returns a
manifest usable for admission/spawn, and is not used by current-v2 loading.

Run:

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_runtime_identity.py \
  tests/test_local_ai_manifest.py
uv run ruff check \
  app/services/local_ai/runtime_identity.py \
  app/services/local_ai/manifest.py \
  tests/test_local_ai_runtime_identity.py \
  tests/test_local_ai_manifest.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 6: Root review and prepare the proposed checkpoint**

Run the following only if Pedro separately authorizes a commit:

```bash
git add \
  .env.example \
  backend/app/config.py \
  backend/app/model_manifests/schema-v2.json \
  backend/app/services/local_ai/manifest.py \
  backend/app/services/local_ai/runtime_identity.py \
  backend/tests/test_local_ai_manifest.py \
  backend/tests/test_local_ai_runtime_identity.py
git commit -m "feat(local-ai): define attested worker identity"
```

### Task 2: Bind validation receipts and strict admission to runtime identity

**Files:**
- Modify: `backend/app/services/local_ai/validation_receipt.py`
- Modify: `backend/app/services/local_ai/artifact_store.py`
- Modify: `backend/app/services/local_ai/pack_verifier.py`
- Modify: `backend/app/services/local_ai/processing_snapshot.py`
- Modify: `backend/app/services/local_ai/errors.py`
- Modify: `backend/app/models/local_ai.py`
- Modify: `backend/app/api/local_ai.py`
- Modify: `backend/app/api/upload.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Test: `backend/tests/test_local_ai_pack_verifier.py`
- Test: `backend/tests/test_local_ai_artifacts.py`
- Test: `backend/tests/test_processing_mode_snapshot.py`
- Test: `backend/tests/test_local_ai_models.py`
- Test: `backend/tests/test_local_ai_api.py`
- Test: `backend/tests/test_summarization.py`
- Test: `backend/tests/test_upload_progress_cancel.py`

**Interfaces:**
- Consumes: `WorkerRuntimeIdentity` and manifest runtime identity from Task 1.
- Produces: receipt key `worker_bundle_sha256`.
- Produces: `_issue_runtime_validation_receipt(manifest, observed_identity)`.
- Produces: bounded policy code `runtime_identity_required` for v1 queued work.
- Produces: a one-way legacy-job terminal transition which preserves the
  immutable v1 snapshot and digest without allowing its execution or promotion.

- [ ] **Step 1: Write failing receipt and admission tests**

Add exact regressions:

```python
def test_persisted_receipt_requires_worker_bundle_digest(manifest_v2) -> None:
    payload = expected_validation_payload(manifest_v2)
    payload.pop("worker_bundle_sha256")

    with pytest.raises(LocalValidationError):
        validate_persisted_receipt(payload, manifest_v2)


async def test_verifier_rejects_runtime_identity_mismatch(
    manifest_v2, candidate_pack, fake_manager
) -> None:
    fake_manager.runtime_identity = WorkerRuntimeIdentity(
        scheme="local-ai-worker-bundle.v1", bundle_sha256="b" * 64
    )

    with pytest.raises(LocalValidationError):
        await verify_pack_candidate(manifest_v2, candidate_pack, manager=fake_manager)
```

Also test a correct manifest with a substituted receipt digest, activation
pointer reuse, rejection of strict admission with a v1 receipt, and owner-safe failure of a
queued v1 ingestion job and queued v1 summary job with
`runtime_identity_required` and no raw path/content. Exercise every current
state-mutation entry point: `api/local_ai.py::_retry_local_ai_job()` must return
the existing 409 boundary for a failed v1 job without changing it;
`api/local_ai.py` cancellation plus `api/upload.py` bulk/single cancellation
must terminalize an active v1 job as `failed`, not `cancelled`; and
`api/upload.py::_recover_stuck_files()` must terminalize a discovered queued or
processing v1 job as `failed`, not requeue it or apply a timeout failure. For
each persisted legacy job, assert its `manifest_snapshot` and
`manifest_sha256` stay byte-for-byte unchanged, it cannot be retried, requeued,
cancelled, promoted, or completed, and no worker, provider, or release-evidence
loader is called.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_pack_verifier.py \
  tests/test_local_ai_artifacts.py \
  tests/test_processing_mode_snapshot.py \
  tests/test_local_ai_api.py \
  tests/test_upload_progress_cancel.py
```

Expected: failures show the receipt omits the worker digest and v1 admission is
still accepted.

- [ ] **Step 3: Bind observed identity to receipt issuance**

Make receipt keys and issuance exact:

```python
VALIDATION_RECEIPT_KEYS = frozenset(
    {
        "pack_revision",
        "manifest_sha256",
        "platform",
        "runtime_name",
        "runtime_version",
        "worker_bundle_sha256",
        "validation_suite_version",
        "verifier_version",
    }
)


def _issue_runtime_validation_receipt(
    manifest: LocalAIManifest,
    observed_identity: WorkerRuntimeIdentity,
) -> RuntimeValidationReceipt:
    require_manifest_runtime_identity(manifest, observed_identity)
    return RuntimeValidationReceipt(
        payload=expected_validation_payload(manifest), _seal=_SEAL
    )
```

`expected_validation_payload()` reads the bundle digest from the v2 manifest.
`verify_pack_candidate()` resolves or obtains the selected manager's observed
identity before the first fixture call, compares it to the candidate manifest,
and passes that identity to receipt issuance.

- [ ] **Step 4: Fail old admission and queued snapshots closed**

At `_current_strict_snapshot()`, require schema v2 and a matching validation
receipt before release-evidence loading. Use normal parsing for current
admission, and use the Task-1 diagnostic parser only to recognize an existing
canonical v1 row before any worker/provider/evidence operation. Add a fixed
error subtype rather than passing an unsupported constructor keyword to the
existing exception:

```python
class RuntimeIdentityRequiredError(LocalPolicyError):
    """A legacy strict-local snapshot cannot be admitted after v2 rollout."""

    code = "runtime_identity_required"


raise RuntimeIdentityRequiredError(
    "Strict-local worker runtime identity is required."
)
```

In `LocalAIJob`, permit diagnostic revalidation of a legacy row only so a
transactional `fail_legacy_runtime_identity_required()` helper can set its
status/stage/failure/completed timestamp. The helper must preserve the raw
snapshot and its digest, set a non-retryable bounded failure payload with this
fixed code, and make no snapshot or identity-field write. Invoke it from both
the ingestion path in `api/upload.py` and the summary path in
`services/ai/summarizer.py` before their ordinary manifest parsing.

Cover worker admission and every other existing legacy state-mutation path.
In `api/local_ai.py::_retry_local_ai_job()`, diagnose a v1 row before the
queued reset and return its existing 409 response without changing it. In the
job-cancel endpoint in `api/local_ai.py`, and the single/bulk cancel helpers in
`api/upload.py`, detect a queued or processing v1 snapshot and call the
one-way helper instead of writing `cancelled`. In
`api/upload.py::_recover_stuck_files()`, exclude canonical v1 snapshots from
the raw SQL cancellation/requeue/timeout updates and lock them through the
same helper, so a post-upgrade legacy row becomes only
`failed(runtime_identity_required)`. No legacy path may mutate the snapshot,
delete its upload/prompt, construct a provider, turn it back into `queued`, or
replace the fixed failure code. Update failure allowlists only where needed to
carry this stable code.

- [ ] **Step 5: Verify GREEN and privacy boundaries**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_pack_verifier.py \
  tests/test_local_ai_artifacts.py \
  tests/test_processing_mode_snapshot.py \
  tests/test_local_ai_models.py \
  tests/test_local_ai_api.py \
  tests/test_summarization.py \
  tests/test_upload_progress_cancel.py \
  tests/test_local_ai_log_privacy.py \
  tests/test_strict_local_egress.py
uv run ruff check \
  app/services/local_ai/validation_receipt.py \
  app/services/local_ai/pack_verifier.py \
  app/services/local_ai/processing_snapshot.py \
  app/models/local_ai.py \
  app/api/local_ai.py \
  app/api/upload.py \
  app/services/ai/summarizer.py
```

Expected: selected tests and Ruff pass; the egress tests prove no provider is
constructed on identity failure.

- [ ] **Step 6: Root review and prepare the proposed checkpoint**

Run the following only if Pedro separately authorizes a commit:

```bash
git add \
  backend/app/services/local_ai/artifact_store.py \
  backend/app/services/local_ai/errors.py \
  backend/app/services/local_ai/pack_verifier.py \
  backend/app/services/local_ai/processing_snapshot.py \
  backend/app/services/local_ai/validation_receipt.py \
  backend/app/models/local_ai.py \
  backend/app/api/local_ai.py \
  backend/app/api/upload.py \
  backend/app/services/ai/summarizer.py \
  backend/tests/test_local_ai_artifacts.py \
  backend/tests/test_local_ai_pack_verifier.py \
  backend/tests/test_processing_mode_snapshot.py \
  backend/tests/test_local_ai_models.py \
  backend/tests/test_local_ai_api.py \
  backend/tests/test_summarization.py \
  backend/tests/test_upload_progress_cancel.py
git commit -m "fix(local-ai): bind validation receipts to worker code"
```

### Task 3: Re-attest immediately before every worker spawn

**Files:**
- Modify: `backend/app/services/local_ai/model_manager.py`
- Modify: `backend/app/services/local_ai/pipeline.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Modify: `backend/app/services/local_ai/fidelity_runner.py`
- Modify: `backend/app/services/local_ai/pack_verifier.py`
- Modify: `backend/scripts/benchmark_local_ai.py`
- Test: `backend/tests/test_local_ai_model_manager.py`
- Test: `backend/tests/test_strict_local_pipeline.py`
- Test: `backend/tests/test_local_ai_summary_worker_contract.py`
- Test: `backend/tests/test_local_ai_fidelity.py`
- Test: `backend/tests/test_local_ai_benchmark.py`

**Interfaces:**
- Consumes: v2 `LocalAIManifest` and runtime identity resolver from Tasks 1-2.
- Produces: `LocalModelManager.run_attested(manifest, role, payload, ...)`.
- Produces: `LocalModelManager.count_summary_tokens_attested(manifest, payload)`.
- Preserves: raw `run()` only for deliberately unisolated fake-worker unit tests.

- [ ] **Step 1: Write spawn-time drift tests**

Add a manager test which starts with matching source, mutates one worker file
after `manager.start()`, and asserts `create_subprocess_exec` is never called:

```python
await manager.start()
worker_source.write_text("# drifted after startup\n", encoding="utf-8")

with patch("asyncio.create_subprocess_exec") as spawn:
    with pytest.raises(LocalWorkerError):
        await manager.run_attested(manifest, ModelRole.OCR, payload)

spawn.assert_not_called()
```

Add call-site contract tests which inject a manager exposing only attested
methods. OCR/extraction, summary token counting/generation, pack verification,
fidelity, and benchmark execution must all pass the manifest they already
validated.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_model_manager.py \
  tests/test_strict_local_pipeline.py \
  tests/test_local_ai_summary_worker_contract.py \
  tests/test_local_ai_fidelity.py \
  tests/test_local_ai_benchmark.py
```

Expected: tests fail because attested manager methods do not exist and current
call sites invoke raw `run()`.

- [ ] **Step 3: Add manifest-bound manager methods**

Add exact wrappers:

```python
async def run_attested(
    self,
    manifest: LocalAIManifest,
    role: ModelRole,
    payload: dict[str, Any],
    on_progress: ProgressCallback | None = None,
    *,
    on_liveness: LivenessCallback | None = None,
) -> Any:
    self._require_current_runtime(manifest)
    return await self._run_command(
        role,
        payload,
        self._command_for(ModelRole(role)),
        on_progress,
        on_liveness=on_liveness,
        manifest=manifest,
    )


async def count_summary_tokens_attested(
    self, manifest: LocalAIManifest, payload: dict[str, Any]
) -> int:
    self._require_current_runtime(manifest)
    result = await self._run_command(
        ModelRole.SUMMARY,
        payload,
        "count_summary_tokens",
        None,
        on_liveness=None,
        manifest=manifest,
    )
    if (
        type(result) is not dict
        or set(result) != {"token_count"}
        or type(result["token_count"]) is not int
        or result["token_count"] <= 0
    ):
        raise LocalWorkerError("Local worker returned an invalid token count.")
    return result["token_count"]
```

To close the check-to-spawn gap inside this process, call
`_require_current_runtime(manifest)` again in `_run_registered()` immediately
before `_spawn_worker_cancellation_safe()`. Thread the manifest through the
private `_run_command(..., manifest=manifest)` and store it in the registered
run state; never trust a startup-cached digest. Keep the legacy public raw
`run()`/`count_summary_tokens()` behind an explicit
`allow_unisolated_test_worker` test-only constructor flag that is rejected when
the network sandbox is enforced. This makes raw calls unavailable to every
production manager while preserving narrow fake-worker unit tests.

- [ ] **Step 4: Convert all production call sites**

Replace raw calls with manifest-bound calls:

```python
raw_result = await self.manager.run_attested(
    self.manifest,
    ModelRole.OCR,
    payload,
    on_liveness=self.on_liveness,
)
```

Use the same pattern for extraction, summary, summary token counting, fidelity,
and pack verification. In `benchmark_local_ai.py`, change
`_BenchmarkingManager.run()` to `run_attested(manifest, role, payload, ...)`,
thread the canonical manifest through every benchmark role invocation, and call
the underlying manager only through `run_attested`. Test-only fake managers
implement the same signature or use the explicitly unisolated raw path; no
production path may call raw `run()` or `count_summary_tokens()`.

Update the `_Manager` / `FidelityManager` and pack-verifier protocols to expose
only `run_attested(manifest, role, payload, ...)`, then update every
`selected_manager.run(...)` invocation in `fidelity_runner.py` and
`pack_verifier.py`. Pass the locked/candidate `LocalAIManifest` through each
OCR, extraction, and summary payload loop; the static transport dictionary is
not a substitute for the typed manifest argument. The benchmark wrapper must
also pass that same typed manifest down to the production manager, so its
resource sampling cannot become an attestation bypass.

- [ ] **Step 5: Verify GREEN and scan the call graph**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_model_manager.py \
  tests/test_strict_local_pipeline.py \
  tests/test_local_ai_summary_worker_contract.py \
  tests/test_local_ai_fidelity.py \
  tests/test_local_ai_pack_verifier.py \
  tests/test_local_ai_benchmark.py
rg -n --glob '*.py' "\.run\(|\.count_summary_tokens\(" app scripts
uv run ruff check \
  app/services/local_ai/model_manager.py \
  app/services/local_ai/pipeline.py \
  app/services/ai/summarizer.py \
  app/services/local_ai/fidelity_runner.py
```

Expected: tests and Ruff pass. Review every `rg` hit: production manager calls
are only private implementation definitions or attested methods, while any raw
call is a deliberately unisolated test adapter. There is no
`selected_manager.run`, `local_model_manager.run`, or benchmark wrapper that
can spawn a worker without the immediate pre-spawn identity recheck.

- [ ] **Step 6: Root review and prepare the proposed checkpoint**

Run the following only if Pedro separately authorizes a commit:

```bash
git add \
  backend/app/services/ai/summarizer.py \
  backend/app/services/local_ai/fidelity_runner.py \
  backend/app/services/local_ai/model_manager.py \
  backend/app/services/local_ai/pack_verifier.py \
  backend/app/services/local_ai/pipeline.py \
  backend/scripts/benchmark_local_ai.py \
  backend/tests/test_local_ai_fidelity.py \
  backend/tests/test_local_ai_benchmark.py \
  backend/tests/test_local_ai_model_manager.py \
  backend/tests/test_local_ai_summary_worker_contract.py \
  backend/tests/test_strict_local_pipeline.py
git commit -m "fix(local-ai): attest every worker spawn"
```

### Task 4: Enforce schema-v2 snapshots in upgraded and fresh databases

**Files:**
- Create: `backend/alembic/versions/c5d6e7f8a9b0_require_attested_local_ai_manifest.py`
- Modify: `backend/app/models/local_ai_ddl.py`
- Modify: `backend/app/models/local_ai.py`
- Test: `backend/tests/test_local_ai_migrations.py`
- Test: `backend/tests/test_local_ai_models.py`

**Interfaces:**
- Consumes: schema-v2 runtime keys from Task 1.
- Produces: database `local_ai_manifest_is_valid(jsonb)` requiring schema `2`,
  exact runtime keys, scheme constant, and lowercase bundle digest.
- Preserves: all other strict-local functions, triggers, and immutable snapshot
  rules.
- Permits: only the one-way, bounded terminal failure transition for an
  existing v1 queued/processing row; v1 remains invalid for every INSERT,
  admission, requeue, promotion, or completed-work transition.

- [ ] **Step 1: Write failing migration/create-all parity tests**

Add tests that install the migration function in an upgraded schema and the
`LOCAL_AI_DDL` listeners in a fresh schema. For each, assert:

```python
assert await manifest_is_valid(valid_v2_snapshot) is True
assert await manifest_is_valid(valid_v1_snapshot) is False
assert await manifest_is_valid(v2_without_worker_digest) is False
assert await manifest_is_valid(v2_with_uppercase_digest) is False
assert await manifest_is_valid(v2_with_extra_runtime_key) is False
```

Assert the normalized migration SQL and create-all SQL enforce the same exact
runtime key set and scheme constant. Add transition tests that insert an
existing canonical v1 row under the pre-upgrade guard, upgrade the function,
then permit only `queued|processing -> failed` with stage `failed`, a bounded
non-retryable `runtime_identity_required` failure, and `completed_at` set.
Assert the v1 snapshot/digest cannot change and every v1 requeue, promotion,
or alternate status update is rejected. Run the same assertions on the fresh
`create_all` definition.

- [ ] **Step 2: Run database tests and verify RED**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_migrations.py \
  tests/test_local_ai_models.py
```

Expected: v2 snapshot cases fail because the database function still requires
schema version 1 and two runtime keys.

- [ ] **Step 3: Replace the guard in migration and create-all definitions**

Base the new migration on current head `b0c1d2e3f4a5`. In both definitions,
require:

```sql
payload->>'schema_version' = '2'
AND local_ai_json_has_exact_keys(
    payload->'runtime',
    ARRAY[
        'name',
        'version',
        'worker_identity_scheme',
        'worker_bundle_sha256'
    ]
)
AND payload->'runtime'->>'worker_identity_scheme'
    = 'local-ai-worker-bundle.v1'
AND payload->'runtime'->>'worker_bundle_sha256' ~ '^[0-9a-f]{64}$'
```

The migration `upgrade()` uses `CREATE OR REPLACE FUNCTION`; `downgrade()`
restores the exact schema-v1 function. Copy the complete function definition;
do not depend on importing application code from Alembic.

Keep `local_ai_manifest_is_valid()` v2-only. In the job trigger itself,
special-case an UPDATE whose `OLD.manifest_snapshot` is a canonical immutable
v1 snapshot: allow it only when all identity/snapshot fields are unchanged and
the exact one-way failure transition described above is present. The trigger
still rejects v1 on INSERT and every other UPDATE. Mirror that conditional
verbatim in the Alembic function and `local_ai_ddl.py`; extend the ORM
before-update validation in `models/local_ai.py` to permit only that same
legacy terminalization, not broad legacy revalidation.

- [ ] **Step 4: Recreate two isolated test schemas and verify GREEN**

Use the same PostgreSQL assumptions as
`.github/workflows/local-ai-contract-ci.yml`: create a migrated database and a
separate empty database for `Base.metadata.create_all()`. Do not reuse the
developer's persistent `medtimeline_test`; the explicit names below must not
already exist, so a stale database fails safely instead of being overwritten.
The create-all database name deliberately ends in `_test`: `tests/conftest.py`
uses `DATABASE_URL` unchanged only when that suffix is present. Create
`pgcrypto` in that exact database before the create-all strict-local DDL is
installed.

```bash
cd backend
export PGPASSWORD=postgres
migration_db=medtimeline_runtime_attestation_migrations_ci
fresh_test_db=medtimeline_runtime_attestation_fresh_create_all_test
createdb -h 127.0.0.1 -U postgres "$migration_db"
createdb -h 127.0.0.1 -U postgres "$fresh_test_db"
trap 'dropdb -h 127.0.0.1 -U postgres --if-exists "$migration_db"; dropdb -h 127.0.0.1 -U postgres --if-exists "$fresh_test_db"' EXIT
psql -h 127.0.0.1 -U postgres -d "$fresh_test_db" -c 'CREATE EXTENSION IF NOT EXISTS pgcrypto;'
DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/$migration_db" \
  uv run alembic upgrade head
DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/$migration_db" \
  uv run pytest -q tests/test_local_ai_migrations.py
DATABASE_URL="postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/$fresh_test_db" \
  uv run pytest -q tests/test_local_ai_models.py
```

Then run the focused checks:

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_migrations.py \
  tests/test_local_ai_models.py
uv run ruff check \
  alembic/versions/c5d6e7f8a9b0_require_attested_local_ai_manifest.py \
  app/models/local_ai_ddl.py \
  tests/test_local_ai_migrations.py \
  tests/test_local_ai_models.py
uv run alembic heads
```

Expected: tests and Ruff pass; Alembic reports only
`c5d6e7f8a9b0 (head)`.

- [ ] **Step 5: Root review and prepare the proposed checkpoint**

Run the following only if Pedro separately authorizes a commit:

```bash
git add \
  backend/alembic/versions/c5d6e7f8a9b0_require_attested_local_ai_manifest.py \
  backend/app/models/local_ai_ddl.py \
  backend/app/models/local_ai.py \
  backend/tests/test_local_ai_migrations.py \
  backend/tests/test_local_ai_models.py
git commit -m "fix(local-ai): require attested manifest snapshots"
```

### Task 5: Promote the attested v2 Apple release

**Files:**
- Create: `backend/app/model_manifests/catalog-v2.json`
- Create: `backend/app/model_manifests/apple-m4-16gb-v2.lock.json`
- Create: `backend/app/model_manifests/apple-m4-16gb-v2.release.json`
- Modify: `backend/app/config.py`
- Modify: `backend/scripts/lock_local_ai_manifest.py`
- Modify: `backend/scripts/benchmark_local_ai.py`
- Modify: `backend/scripts/run_local_ai_fidelity.py`
- Modify: `backend/scripts/promote_local_ai_release.py`
- Modify: `backend/artifacts/local-ai-benchmark.json`
- Modify: `backend/artifacts/local-ai-fidelity.json`
- Modify: `justfile`
- Modify: `docs/operations-strict-local-ai.md`
- Modify: `docs/backend-handoff.md`
- Modify: `docs/third-party-local-model-pack-notices.md`
- Test: `backend/tests/test_local_ai_candidate_pack_cli.py`
- Test: `backend/tests/test_local_ai_release_evidence.py`
- Test: `backend/tests/test_local_ai_ci_workflows.py`

**Interfaces:**
- Consumes: final Track B worker tree and Tasks 1-4 attestation code.
- Produces: `apple-m4-16gb-v2` lock and release evidence bound to its exact
  worker digest, benchmark, fidelity report, and unchanged immutable model
  revisions unless separately approved.
- Produces: default config and `just` commands targeting v2 artifacts.

- [ ] **Step 1: Write failing tooling and release-binding tests**

Assert the lock tool reads the canonical local worker identity and emits:

```python
assert locked["schema_version"] == 2
assert locked["pack_revision"] == "apple-m4-16gb-v2"
assert locked["runtime"]["worker_identity_scheme"] == (
    "local-ai-worker-bundle.v1"
)
assert locked["runtime"]["worker_bundle_sha256"] == observed.bundle_sha256
```

Assert benchmark, fidelity, and release promotion reject reports whose manifest
or worker digest differs. Assert CI workflow contract tests name schema v2 and
the v2 lock/release paths. Add a CLI/`justfile` contract test that the
pre-promotion verification target invokes
`scripts/local_ai_candidate_pack.py verify` (which deliberately uses
`require_release=False`) and that `local-ai-pack-verify` remains the normal
post-promotion command requiring release evidence. Add a release-gate
regression that begins with `REAL_MEDICAL_FIXTURES_DIR` set, runs the documented
command wrapper, and asserts the serialized fidelity report has
`private_documents == 0` and no private metrics.

- [ ] **Step 2: Run tooling tests and verify RED**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_candidate_pack_cli.py \
  tests/test_local_ai_release_evidence.py \
  tests/test_local_ai_ci_workflows.py
```

Expected: tests fail because tooling still emits schema v1 and v1 paths.

- [ ] **Step 3: Update tooling and static release references**

Create `catalog-v2.json` with the same approved model repositories/revisions and
decode limits, `pack_revision = "apple-m4-16gb-v2"`, schema `2`, and the
observed worker identity. Update config and `justfile` paths to v2. Make the
lock, benchmark, fidelity, and promotion scripts independently recompute or
validate the worker digest instead of copying an arbitrary CLI value.

Add a `just local-ai-candidate-verify` pre-promotion target which runs
`cd backend && uv run python scripts/local_ai_candidate_pack.py verify` and
therefore validates the retained active pack against the v2 lock without trying
to load a v2 release-evidence file that does not exist yet. Keep
`just local-ai-pack-verify` on `scripts/local_ai_pack.py verify`: it remains
the post-promotion lifecycle check and must require the newly written v2
release evidence. Do not weaken the normal lifecycle's evidence requirement.

Do not edit the old v1 lock, schema, or release file; they remain diagnostic
history and are no longer defaults.

- [ ] **Step 4: Verify deterministic tooling GREEN**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_candidate_pack_cli.py \
  tests/test_local_ai_release_evidence.py \
  tests/test_local_ai_ci_workflows.py \
  tests/test_local_ai_manifest.py
uv run ruff check scripts tests/test_local_ai_candidate_pack_cli.py \
  tests/test_local_ai_release_evidence.py tests/test_local_ai_ci_workflows.py
```

Expected: tests and Ruff pass without loading models or accessing the network.

- [ ] **Step 5: Run the offline model-backed release gates**

Before the v2 deployment or release mutation, use the application session
factory to count active validated-strict-local jobs without selecting PHI or
logging identifiers. Refuse the deployment if any queued or processing job
exists; operators must let it finish or fail it under the old behavior before
applying the v2 guard. This is a deployment preflight, not permission to
rewrite legacy snapshots:

```bash
cd backend
uv run python - <<'PY'
import asyncio

from sqlalchemy import func, select

from app.database import async_session_factory
from app.models.local_ai import LocalAIJob


async def main() -> None:
    async with async_session_factory() as session:
        active = (await session.execute(
            select(func.count()).select_from(LocalAIJob).where(
                LocalAIJob.processing_mode == "validated_strict_local",
                LocalAIJob.status.in_(("queued", "processing")),
            )
        )).scalar_one()
    if active:
        raise SystemExit("refusing v2 deployment: active strict-local jobs exist")


asyncio.run(main())
PY
cd ..
```

First verify the active retained model root against the v2 lock with the
candidate command; do not redownload when immutable artifacts already match.
This gate intentionally precedes release promotion, so it must not use the
normal command that requires release evidence. Run the release gates with the
private-fixture environment variable removed even if it is set in the operator
shell, then assert the written report contains no private run and perform the
normal evidence-requiring verification only after promotion:

```bash
just local-ai-candidate-verify
just local-ai-benchmark
cd backend
env -u REAL_MEDICAL_FIXTURES_DIR uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-fidelity.json
uv run python - <<'PY'
from app.services.local_ai.fidelity_runner import load_fidelity_report

report = load_fidelity_report("artifacts/local-ai-fidelity.json")
assert report.private_documents == 0
assert report.private_metrics is None
PY
cd ..
just local-ai-release-promote
just local-ai-pack-verify
```

Expected:

- candidate verification succeeds offline against the v2 lock before release
  evidence exists, and the normal post-promotion verification succeeds with
  the newly written evidence;
- benchmark records three cold runs per role with no memory-pressure failure;
- the committed six-document synthetic fidelity suite meets every existing
  release threshold;
- promotion writes `apple-m4-16gb-v2.release.json` bound to the exact v2
  manifest, benchmark, and fidelity digests.

Stop and report the exact failed gate if any command fails. Do not relax a
threshold, fabricate evidence, use private fixtures, or call a cloud provider.

- [ ] **Step 6: Update operational and API documentation**

Document the worker identity inputs, v1-to-v2 no-active-job deployment
preflight, bounded `runtime_identity_required` behavior, operator revalidation,
and exact v2 artifact names. State that OS owner/root remains outside this
attestation boundary.

Run the humanizer audit: ask “What makes this text obviously AI generated?”,
remove inflated language, mechanical repetition, and em-dash spray, then rerun
`git diff --check`.

- [ ] **Step 7: Root review and prepare the proposed verified-release checkpoint**

Run the following only if Pedro separately authorizes a commit:

```bash
git add \
  backend/app/config.py \
  backend/app/model_manifests/catalog-v2.json \
  backend/app/model_manifests/apple-m4-16gb-v2.lock.json \
  backend/app/model_manifests/apple-m4-16gb-v2.release.json \
  backend/artifacts/local-ai-benchmark.json \
  backend/artifacts/local-ai-fidelity.json \
  backend/scripts/benchmark_local_ai.py \
  backend/scripts/lock_local_ai_manifest.py \
  backend/scripts/promote_local_ai_release.py \
  backend/scripts/run_local_ai_fidelity.py \
  backend/tests/test_local_ai_candidate_pack_cli.py \
  backend/tests/test_local_ai_ci_workflows.py \
  backend/tests/test_local_ai_release_evidence.py \
  docs/backend-handoff.md \
  docs/operations-strict-local-ai.md \
  docs/third-party-local-model-pack-notices.md \
  justfile
git commit -m "release(local-ai): promote attested Apple pack v2"
```

### Task 6: Track-level verification and handoff

**Files:**
- Verify only; no planned product-file changes.

**Interfaces:**
- Produces: a commit list, test transcript, v2 receipt/evidence digests, and
  explicit confirmation that Track B was in branch ancestry.
- Consumes: Tasks 1-5.

- [ ] **Step 1: Verify branch ancestry and cleanliness**

```bash
git merge-base --is-ancestor <TRACK_B_FINAL_COMMIT> HEAD
git status --short
git diff --check <TRACK_B_FINAL_COMMIT>..HEAD
```

Expected: ancestry command exits 0, status is clean, and diff check prints
nothing.

- [ ] **Step 2: Run the complete deterministic attestation suite**

```bash
cd backend
uv run pytest -q \
  tests/test_local_ai_runtime_identity.py \
  tests/test_local_ai_manifest.py \
  tests/test_local_ai_artifacts.py \
  tests/test_local_ai_pack_verifier.py \
  tests/test_local_ai_model_manager.py \
  tests/test_processing_mode_snapshot.py \
  tests/test_local_ai_models.py \
  tests/test_local_ai_migrations.py \
  tests/test_local_ai_release_evidence.py \
  tests/test_local_ai_ci_workflows.py \
  tests/test_strict_local_pipeline.py \
  tests/test_strict_local_egress.py \
  tests/test_local_ai_log_privacy.py
uv run ruff check app tests scripts alembic/versions
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 3: Record model-backed evidence separately**

Report the exact manifest SHA-256, worker bundle SHA-256, receipt SHA-256,
benchmark SHA-256, fidelity SHA-256, command exit codes, and artifact paths.
Do not describe the model-backed release as verified if any receipt is missing
or stale.
