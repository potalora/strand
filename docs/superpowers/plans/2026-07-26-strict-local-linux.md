# Strict-local Linux CPU and NVIDIA implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add separately validated native Linux CPU and NVIDIA CUDA profiles
for the existing OvisOCR2, NuExtract3, and Qwen3.5-9B strict-local stack.

**Architecture:** Keep the backend's current process manager and version-1
stdin/stdout JSONL protocol. A native Python worker starts one pinned
`llama.cpp` CLI child for one model role, validates the result, reaps that
child, then exits. CPU and CUDA use the same GGUF candidates and worker source,
but each profile has its own immutable manifest, runtime build, validation
receipt, and activation state.

**Tech Stack:** Python 3.11, pinned `llama.cpp`, GGUF, CMake, OpenBLAS for CPU,
CUDA for NVIDIA, the existing FastAPI/Pydantic worker protocol, pytest, Ruff,
and the existing synthetic/private fidelity runners.

## Status

This is a follow-on plan. It does not add Linux support to the Apple release.
The repositories, revisions, filenames, runtime commit, memory figures, and
GPU floor below are candidates. They must not appear in the UI or operator
documentation as supported, ready, verified, or validated until the exact
profile passes every release gate in Task 7 and its signed validation receipt
ships with the corresponding lock.

As of July 27, 2026, the candidate set is:

<!-- markdownlint-disable MD013 -->

| Role | Repository and candidate revision | Required files |
| --- | --- | --- |
| OCR | `bartowski/ATH-MaaS_OvisOCR2-GGUF` at `ab22420f3d44201d3aa5a62ca49a665a46b507e9` | `ATH-MaaS_OvisOCR2-Q4_K_M.gguf`, `mmproj-ATH-MaaS_OvisOCR2-bf16.gguf` |
| Extraction | `numind/NuExtract3-GGUF` at `631a32f126925ea54d031dc1cb23c9208889c529` | `NuExtract3-Q4_K_M.gguf`, `mmproj-NuExtract3-BF16.gguf` |
| Summary | `bartowski/Qwen_Qwen3.5-9B-GGUF` at `182be2fd6c7bc44887d88a91cb03ff009cc9f549` | `Qwen_Qwen3.5-9B-Q4_K_M.gguf` |
| Runtime | `ggml-org/llama.cpp` at `1cbfd1988311775425d36c0ce066590f7d3049cf` | CPU and CUDA builds from the same source revision |

<!-- markdownlint-enable MD013 -->

Do not advance a repository or runtime revision in place. Create a new
candidate pack revision and rerun all gates.

Candidate sources:

- [OvisOCR2 GGUF repository](https://huggingface.co/bartowski/ATH-MaaS_OvisOCR2-GGUF)
- [NuExtract3 GGUF repository](https://huggingface.co/numind/NuExtract3-GGUF)
- [Qwen3.5-9B GGUF repository](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF)
- [`llama.cpp` source and backend support](https://github.com/ggml-org/llama.cpp)

## Global constraints

- Preserve the four existing processing modes. This plan changes only
  `validated_strict_local`.
- Preserve the role split: OvisOCR2 does page OCR, NuExtract3 does grounded
  extraction, and Qwen3.5-9B summarizes only validated facts and evidence.
- Raw document content may enter only the worker process for its owning job.
- No cloud-capable provider is constructed and no cloud fallback exists.
- The Linux worker uses protocol version `1` over stdin/stdout JSON Lines.
- Do not add a server, Unix socket, TCP listener, HTTP API, or inference port.
- Do not run `llama-server`. The only permitted runtime executable is the
  lock-pinned local `llama-cli` build.
- Worker requests and results retain the existing 8 MiB frame ceiling and
  fixed safe error messages.
- One model role is resident at a time. The next role cannot start until the
  prior `llama-cli` process group is gone.
- Prompts go through owner-only prompt files under the owning scratch
  directory. Prompt or document text must never appear in argv, environment
  variables, logs, telemetry, crash text, or progress events.
- The worker verifies the manifest, exact artifact tree, file sizes, SHA-256
  values, profile, runtime commit, and request identity before model loading.
- Repository Python, pickle, executables, symlinks, traversal, `auto_map`, and
  undeclared files remain forbidden.
- Installation may use the network only in the existing document-free
  downloader boundary. Inference is physically offline and uses local paths.
- CPU and CUDA never substitute for each other. A job captured for
  `linux_x86_64_cuda` fails if CUDA is unavailable; it does not retry on CPU.
- CPU and CUDA use separate model directories, locks, receipts, and
  `LOCAL_AI_WORKER_COMMAND` values.
- ROCm, aarch64 Linux, Vulkan, vLLM, Transformers, safetensors inference,
  containers, and custom local endpoints are outside this plan.
- Keep `LOCAL_AI_ENABLED=false` until one exact Linux profile is promoted.
- Delegated workers never stage or commit. They return their changed-file list,
  verification output, and open risks to the root agent. Only the root agent
  may integrate or commit, and only when Pedro explicitly requests it.

## Admission targets

These figures decide whether validation may start. They are not release
claims.

<!-- markdownlint-disable MD013 -->

| Profile | Admission target | Recommended release machine |
| --- | --- | --- |
| `linux_x86_64_cpu` | Linux x86_64, AVX2, 16 GiB system RAM, 8 physical cores, no sustained swap during the gate | 16 GiB RAM and 8 or more physical cores |
| `linux_x86_64_cuda` | Linux x86_64, NVIDIA compute capability 7.5 or newer, 12 GiB VRAM, 16 GiB system RAM | 16 GiB VRAM and 16 GiB system RAM |
| Either | 35 GiB free on the model/scratch filesystem before installation | Separate encrypted upload/scratch storage and model storage |

<!-- markdownlint-enable MD013 -->

If a candidate cannot pass within these admission targets, select a smaller
compatible quantization or model and rerun the full gate. If no candidate
passes, keep the profile unvalidated. Do not raise the 16 GiB system-memory
target or weaken the fidelity, privacy, cleanup, or memory gates.

## File map

- `backend/app/services/local_ai/platforms.py`: detects an explicitly requested
  profile without silently selecting an accelerator.
- `backend/app/services/local_ai/runtime_registry.py`: maps one profile to one
  manifest and worker command.
- `backend/app/services/local_ai/manifest.py`: accepts GGUF and records an
  immutable runtime build identity.
- `backend/app/services/local_ai/pack_verifier.py`: applies profile-specific
  runtime checks, then runs the shared three-role fixture chain.
- `backend/app/services/local_ai/pack_operations.py`: reports Linux
  compatibility and keeps lifecycle operations profile-specific.
- `workers/local_ai/shared_contract/`: runtime-independent payload, token-bound,
  extraction-result, and grounded-summary validators shared by Apple and Linux.
- `workers/local_ai/linux_llamacpp/`: native Linux JSONL worker and pinned
  runtime build metadata.
- `scripts/setup-local-ai-linux.sh`: explicit CPU/CUDA runtime setup without
  model download by default.
- `backend/app/model_manifests/linux-x86_64-*.catalog.json`: unvalidated
  candidate metadata.
- `backend/app/model_manifests/linux-x86_64-*.lock.json`: release-generated
  files that do not exist until promotion.

---

### Task 1: Add explicit Linux platform profiles

**Files:**

- Create: `backend/app/services/local_ai/platforms.py`
- Create: `backend/app/services/local_ai/runtime_registry.py`
- Modify: `backend/app/services/local_ai/pack_operations.py`
- Modify: `backend/app/services/local_ai/pack_verifier.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/schemas/local_ai.py`
- Modify: `backend/app/api/local_ai.py`
- Test: `backend/tests/test_local_ai_platforms.py`
- Test: `backend/tests/test_local_ai_runtime_registry.py`

**Interfaces:**

- Produces `PlatformProfile`, `detect_platform_profile()`, and
  `runtime_for_profile()`.
- Consumed by pack CLI, API status, pack verification, benchmark, and setup
  paths.

- [ ] **Step 1: Write failing profile tests**

```python
def test_linux_cpu_requires_explicit_profile(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    profile = detect_platform_profile(
        requested="linux_x86_64_cpu",
        system_memory_bytes=16 * 1024**3,
        cpu_flags={"avx2"},
        cuda=None,
    )
    assert profile.name == "linux_x86_64_cpu"
    assert profile.compatible is True


def test_cuda_profile_never_falls_back_to_cpu() -> None:
    profile = detect_platform_profile(
        requested="linux_x86_64_cuda",
        system_memory_bytes=16 * 1024**3,
        cpu_flags={"avx2"},
        cuda=None,
    )
    assert profile.name == "linux_x86_64_cuda"
    assert profile.compatible is False
    assert profile.reason == "compatible NVIDIA CUDA device is unavailable"
```

- [ ] **Step 2: Run the tests and confirm missing-module failures**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_platforms.py \
  tests/test_local_ai_runtime_registry.py -v
```

Expected: collection fails because the two modules do not exist.

- [ ] **Step 3: Implement immutable profile selection**

```python
ProfileName = Literal[
    "apple_silicon",
    "linux_x86_64_cpu",
    "linux_x86_64_cuda",
    "unsupported",
]


@dataclass(frozen=True)
class PlatformProfile:
    name: ProfileName
    compatible: bool
    reason: str | None
    runtime_name: str


def runtime_for_profile(name: ProfileName) -> RuntimeSelection:
    try:
        return RUNTIMES[name]
    except KeyError:
        raise LocalValidationError("Local model profile is unsupported") from None
```

`RUNTIMES` maps each Linux profile to its own default manifest path and worker
command. `LOCAL_AI_PROFILE` is required on Linux. An empty setting reports
`unsupported`; it does not guess CPU or CUDA from detected hardware.

- [ ] **Step 4: Run focused backend checks**

```bash
cd backend
uv run pytest tests/test_local_ai_platforms.py \
  tests/test_local_ai_runtime_registry.py tests/test_local_ai_api.py -v
uv run ruff check app/services/local_ai/platforms.py \
  app/services/local_ai/runtime_registry.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 5: Hand verified changes to the root agent**

Report the changed files, exact test and Ruff output, and any unresolved
profile-detection risks. Do not stage or commit.

---

### Task 2: Extend immutable manifests for native GGUF runtimes

**Files:**

- Modify: `backend/app/services/local_ai/manifest.py`
- Modify: `backend/app/services/local_ai/artifact_store.py`
- Modify: `backend/scripts/lock_local_ai_manifest.py`
- Create: `backend/app/model_manifests/linux-x86_64-cpu-v1.catalog.json`
- Create: `backend/app/model_manifests/linux-x86_64-cuda-v1.catalog.json`
- Test: `backend/tests/test_local_ai_linux_manifest.py`
- Test: `backend/tests/test_local_ai_artifacts.py`

**Interfaces:**

- Produces separately loadable CPU/CUDA catalogs and, after promotion, locks.
- Consumed unchanged by the downloader and artifact store.

- [ ] **Step 1: Write failing GGUF and profile-isolation tests**

```python
def test_linux_manifest_accepts_only_declared_gguf_files() -> None:
    manifest = parse_manifest(linux_manifest("linux_x86_64_cpu"))
    observed = {
        item.path
        for artifact in manifest.artifacts
        for item in artifact.files
    }
    assert observed == {
        "ATH-MaaS_OvisOCR2-Q4_K_M.gguf",
        "mmproj-ATH-MaaS_OvisOCR2-bf16.gguf",
        "NuExtract3-Q4_K_M.gguf",
        "mmproj-NuExtract3-BF16.gguf",
        "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
    }


def test_cpu_receipt_cannot_activate_cuda_manifest() -> None:
    with pytest.raises(LocalValidationError, match="profile"):
        validate_persisted_receipt(
            receipt_for("linux_x86_64_cpu").payload,
            cuda_manifest(),
        )
```

- [ ] **Step 2: Run and verify both tests fail for the missing GGUF support**

```bash
cd backend
uv run pytest tests/test_local_ai_linux_manifest.py \
  tests/test_local_ai_artifacts.py -v
```

Expected: `.gguf` is rejected and receipts have no Linux profile identity.

- [ ] **Step 3: Add the narrow GGUF and runtime-build schema**

Permit `.gguf` only for `runtime.name == "llama.cpp"`. Keep the exact-tree and
hash checks. Extend runtime identity to:

```json
{
  "name": "llama.cpp",
  "version": "1cbfd1988311775425d36c0ce066590f7d3049cf",
  "build": "cpu-avx2-openblas"
}
```

The CUDA lock uses:

```json
{
  "name": "llama.cpp",
  "version": "1cbfd1988311775425d36c0ce066590f7d3049cf",
  "build": "cuda"
}
```

Update the manifest key allowlists in both backend and worker. Do not relax
the existing suffix rules for any other file type.

- [ ] **Step 4: Create exact candidate catalogs**

Each catalog lists the repositories, 40-character revisions, filenames from
the Status table, Apache-2.0 attribution, decode limits, and
`validation_suite_version: local-ai-fixtures-v1`. Catalogs do not contain
file hashes or claim validation. The lock script resolves each exact file,
writes its byte size and SHA-256, and refuses mutable revisions.

- [ ] **Step 5: Run manifest, downloader, and artifact checks**

```bash
cd backend
uv run pytest tests/test_local_ai_linux_manifest.py \
  tests/test_local_ai_manifest.py tests/test_local_ai_artifacts.py \
  tests/test_local_ai_downloader.py -v
uv run ruff check app/services/local_ai/manifest.py \
  app/services/local_ai/artifact_store.py scripts/lock_local_ai_manifest.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 6: Hand verified changes to the root agent**

Report the changed files, catalog revisions, exact verification output, and
any artifact-license or file-list risks. Do not stage or commit.

---

### Task 3: Extract shared worker contracts without changing Apple behavior

**Files:**

- Create: `workers/local_ai/shared_contract/pyproject.toml`
- Create: `workers/local_ai/shared_contract/src/local_ai_worker_contract/`
- Modify: `workers/local_ai/apple_mlx/pyproject.toml`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/ovisocr2.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py`
- Modify: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/qwen_summary.py`
- Test: `workers/local_ai/shared_contract/tests/test_role_contracts.py`
- Test: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Test: `backend/tests/test_local_ai_summary_worker_contract.py`

**Interfaces:**

- Produces `validate_ocr_request()`, `validate_extraction_request()`,
  `validate_extraction_result()`, `validate_summary_request()`, and
  `validate_summary_result()`.
- Both Apple and Linux workers call these functions around runtime-specific
  generation.

- [ ] **Step 1: Write parity tests using one fixture payload per role**

```python
@pytest.mark.parametrize("runtime", ["apple", "linux"])
def test_role_contract_rejects_free_text_summary_claims(runtime: str) -> None:
    contract = contract_for(runtime)
    with pytest.raises(WorkerInputError):
        contract.validate_summary_result(
            {"sections": [{"heading": "Overview", "claims": ["new diagnosis"]}]}
        )
```

- [ ] **Step 2: Run the shared and Apple tests before extraction**

```bash
cd workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py -q
cd ../../../backend
uv run pytest tests/test_local_ai_summary_worker_contract.py -q
```

Expected: existing tests pass; the new shared-contract test fails because the
package does not exist.

- [ ] **Step 3: Move only runtime-independent validation**

Keep MLX loading, tokenization, generation, and memory counters in
`local_ai_mlx_worker`. Move bounded JSON serialization, request key checks,
grounded summary schema checks, output syntax checks, and fixed safety rules
to `local_ai_worker_contract`. Preserve every existing Apple exception type
and message through thin adapters.

- [ ] **Step 4: Run Apple parity and shared tests**

```bash
cd workers/local_ai/shared_contract
uv run pytest -q
uv run ruff format --check src tests
uv run ruff check src tests
cd ../apple_mlx
uv sync --frozen
uv run pytest -q
cd ../../../backend
uv run pytest tests/test_local_ai_summary_worker_contract.py \
  tests/test_local_ai_protocol.py -q
```

Expected: all selected tests pass with no Apple protocol snapshot changes.

- [ ] **Step 5: Hand verified changes to the root agent**

Report the changed files and Apple/shared contract test output. Call out any
Apple snapshot drift. Do not stage or commit.

---

### Task 4: Build the native Linux llama.cpp worker

**Files:**

- Create: `workers/local_ai/linux_llamacpp/pyproject.toml`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/__main__.py`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/common.py`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/runner.py`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/ovisocr2.py`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/nuextract3.py`
- Create: `workers/local_ai/linux_llamacpp/src/local_ai_llamacpp_worker/qwen_summary.py`
- Create: `workers/local_ai/linux_llamacpp/tests/test_protocol.py`
- Create: `workers/local_ai/linux_llamacpp/tests/test_offline_loading.py`
- Create: `workers/local_ai/linux_llamacpp/tests/test_runner_boundary.py`

**Interfaces:**

- Provides executable `local-ai-llamacpp-worker`.
- Consumes the existing protocol-v1 request payloads.
- Invokes only the configured local `llama-cli` binary.

- [ ] **Step 1: Write protocol and no-server tests**

```python
def test_health_reports_exact_runtime(worker_process) -> None:
    response = request(worker_process, command="health", payload={})
    assert response["payload"]["data"] == {
        "status": "ready",
        "runtime": "llama.cpp-1cbfd1988311775425d36c0ce066590f7d3049cf",
    }


def test_worker_source_never_mentions_llama_server() -> None:
    source = "\n".join(path.read_text() for path in WORKER_SOURCE.rglob("*.py"))
    assert "llama-server" not in source
    assert "open_unix_connection" not in source
    assert "start_server" not in source
```

- [ ] **Step 2: Write a runner test that keeps PHI out of argv**

```python
def test_prompt_content_is_written_only_to_owner_scratch(
    tmp_path: Path, fake_llama_cli: Path
) -> None:
    runner = LlamaCppRunner(binary=fake_llama_cli)
    result = runner.generate(
        prompt="Patient Canary takes metformin",
        model=tmp_path / "model.gguf",
        scratch_dir=tmp_path,
        images=[],
        max_tokens=32,
    )
    assert result == '{"ok":true}'
    assert "Patient Canary" not in fake_llama_cli.read_argv()
    assert (tmp_path / "prompt.txt").stat().st_mode & 0o777 == 0o600
```

- [ ] **Step 3: Run and confirm the worker package is missing**

```bash
cd workers/local_ai/linux_llamacpp
uv run pytest -q
```

Expected: collection fails because the worker modules do not exist.

- [ ] **Step 4: Implement the JSONL entry point and child boundary**

The entry point must match the Apple worker's command set, frame ceiling,
fixed errors, stdout discipline, shutdown handshake, and cancellation
behavior. `runner.py`:

```python
command = [
    str(self.binary),
    "--model",
    str(model),
    "--file",
    str(prompt_file),
    "--n-predict",
    str(max_tokens),
    "--temp",
    "0",
    "--no-display-prompt",
]
```

For OCR/extraction, append the exact verified `--mmproj` and `--image` paths.
Do not append prompt text. Spawn a new process group, close inherited file
descriptors, capture bounded stdout, discard model stderr, enforce the worker
deadline, and kill/reap the whole group on timeout or cancellation.

- [ ] **Step 5: Implement role adapters**

- OCR receives one verified page PNG and returns
  `{"markdown": str, "page_number": int}`.
- Extraction receives OCR Markdown, optional selected page images, and the
  existing NuExtract template. It uses non-thinking mode and the current
  syntax-only retry, then calls the shared result validator.
- Summary receives only the server-owned fact/evidence projection and returns
  fact IDs, field paths, and evidence IDs. It never receives a document or
  page image.

- [ ] **Step 6: Run worker contracts**

```bash
cd workers/local_ai/linux_llamacpp
uv sync --frozen
uv run pytest -q
uv run ruff format --check src tests
uv run ruff check src tests
```

Expected: tests, format, and Ruff pass without model downloads.

- [ ] **Step 7: Hand verified changes to the root agent**

Report the worker files, protocol/lint output, and any unresolved runtime
boundary risks. Do not stage or commit.

---

### Task 5: Build and install pinned CPU and CUDA runtimes

**Files:**

- Create: `scripts/build-local-ai-llama-cpp.sh`
- Create: `scripts/setup-local-ai-linux.sh`
- Modify: `justfile`
- Modify: `.env.example`
- Test: `workers/local_ai/linux_llamacpp/tests/test_setup_script.py`
- Test: `backend/tests/test_local_ai_manifest.py`

**Interfaces:**

- Produces profile-specific runtime directories and worker commands.
- Does not download model files unless `--download-pack` is explicit.

- [ ] **Step 1: Write failing setup-script tests**

```python
def test_linux_setup_requires_an_explicit_profile() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--profile cpu|cuda is required" in result.stderr


def test_linux_setup_does_not_contain_server_commands() -> None:
    source = SCRIPT.read_text()
    assert "llama-server" not in source
    assert "--host" not in source
    assert "--port" not in source
```

- [ ] **Step 2: Run and confirm the setup script is missing**

```bash
cd workers/local_ai/linux_llamacpp
uv run pytest tests/test_setup_script.py -q
```

Expected: tests fail because `scripts/setup-local-ai-linux.sh` does not exist.

- [ ] **Step 3: Implement deterministic native builds**

`build-local-ai-llama-cpp.sh` fetches only
`1cbfd1988311775425d36c0ce066590f7d3049cf`, verifies the checked-out commit,
then builds:

```bash
cmake -S "$source_dir" -B "$build_dir" \
  -DGGML_NATIVE=OFF \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DLLAMA_BUILD_SERVER=OFF
```

For CUDA, replace the BLAS options with:

```bash
-DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=OFF
```

Install only the runtime libraries and `llama-cli`. Record SHA-256 values for
the binary and libraries in `runtime.lock.json`. Reject any installed
`llama-server` binary.

- [ ] **Step 4: Implement setup admission checks**

`setup-local-ai-linux.sh` accepts only `--profile cpu` or `--profile cuda`.
It checks x86_64, AVX2, RAM, disk, CMake/compiler, and for CUDA the NVIDIA
driver, compute capability, and VRAM. It sets `umask 077`, installs the worker
in a profile-specific virtual environment, builds the matching runtime, and
prints the exact `.env` values. `--download-pack` first runs the existing
candidate-lock preflight; without a shipped lock it exits with the candidate
message and does not download models.

- [ ] **Step 5: Add explicit task-runner recipes**

```just
local-ai-linux-cpu-runtime-install:
    ./scripts/setup-local-ai-linux.sh --profile cpu

local-ai-linux-cuda-runtime-install:
    ./scripts/setup-local-ai-linux.sh --profile cuda
```

Do not change the existing Apple recipe or default setup.

- [ ] **Step 6: Run shell, setup, and config checks**

```bash
bash -n scripts/build-local-ai-llama-cpp.sh
bash -n scripts/setup-local-ai-linux.sh
cd workers/local_ai/linux_llamacpp
uv run pytest tests/test_setup_script.py -q
cd ../../../backend
uv run pytest tests/test_local_ai_manifest.py -q
```

Expected: all commands pass.

- [ ] **Step 7: Hand verified changes to the root agent**

Report the setup/build files, shell and test output, and the exact runtime
commit and build flags used. Do not stage or commit.

---

### Task 6: Connect Linux profiles to the existing process manager

**Files:**

- Modify: `backend/app/services/local_ai/model_manager.py`
- Modify: `backend/app/services/local_ai/pack_verifier.py`
- Modify: `backend/app/services/local_ai/validation_receipt.py`
- Modify: `backend/scripts/local_ai_pack.py`
- Modify: `backend/scripts/benchmark_local_ai.py`
- Test: `backend/tests/test_local_ai_model_manager.py`
- Test: `backend/tests/test_local_ai_pack_verifier.py`
- Test: `backend/tests/test_local_ai_resources.py`

**Interfaces:**

- Reuses `LocalModelManager.run(role, payload)` without a second transport.
- Adds profile/runtime identity to validation and benchmark receipts.

- [ ] **Step 1: Write failing direct-process parity tests**

```python
@pytest.mark.asyncio
async def test_linux_worker_uses_the_existing_stdio_manager(
    fake_linux_worker,
) -> None:
    manager = LocalModelManager(worker_command=fake_linux_worker.command)
    await manager.start()
    result = await manager.run(ModelRole.OCR, fake_ocr_payload())
    assert result == {"markdown": "A1c 6.8 %", "page_number": 1}
    assert manager.metrics.max_live_processes == 1


def test_validation_receipt_binds_profile_and_runtime_build() -> None:
    receipt = receipt_for("linux_x86_64_cpu", build="cpu-avx2-openblas")
    with pytest.raises(LocalValidationError, match="runtime"):
        validate_persisted_receipt(receipt.payload, cuda_manifest())
```

- [ ] **Step 2: Run and confirm receipt/profile failures**

```bash
cd backend
uv run pytest tests/test_local_ai_model_manager.py \
  tests/test_local_ai_pack_verifier.py tests/test_local_ai_resources.py -v
```

Expected: new tests fail because receipts do not bind Linux profile/build.

- [ ] **Step 3: Make manager changes profile-neutral**

Keep the current stdio spawn, process-group cancellation, global one-role
lock, timeouts, bounded lines, and cleanup poisoning. Select the worker command
before constructing the manager through `runtime_for_profile()`. Do not add a
socket adapter, persistent supervisor, HTTP client, or container lifecycle.

Pack verification must check:

1. detected profile equals the requested manifest profile;
2. runtime binary/library hashes equal `runtime.lock.json`;
3. worker health returns the exact runtime commit;
4. all three roles load while external sockets are denied;
5. the synthetic OCR -> extraction -> summary chain passes.

- [ ] **Step 4: Extend content-free resource receipts**

Record profile, kernel, CPU model/flags, system RAM, NVIDIA model/compute
capability/driver for CUDA, runtime commit/build hash, model lock digest,
peak RSS, peak VRAM, swap delta, cold-load time, throughput, exit status, and
post-role process/RAM/VRAM reclamation. Never record prompts, OCR, facts,
evidence, summaries, filenames, or patient identifiers.

- [ ] **Step 5: Run manager and verifier checks**

```bash
cd backend
uv run pytest tests/test_local_ai_model_manager.py \
  tests/test_local_ai_pack_verifier.py tests/test_local_ai_resources.py \
  tests/test_strict_local_pipeline.py -q
uv run ruff check app/services/local_ai/model_manager.py \
  app/services/local_ai/pack_verifier.py \
  app/services/local_ai/validation_receipt.py \
  scripts/local_ai_pack.py scripts/benchmark_local_ai.py
```

Expected: all selected tests and Ruff pass.

- [ ] **Step 6: Hand verified changes to the root agent**

Report the changed files, manager/verifier output, and any cancellation or
resource-accounting risks. Do not stage or commit.

---

### Task 7: Run profile-specific release gates and promote locks

**Files:**

- Modify: `backend/app/services/local_ai/fidelity_runner.py`
- Modify: `backend/scripts/run_local_ai_fidelity.py`
- Create: `.github/workflows/local-ai-linux-contract-ci.yml`
- Test: `backend/tests/test_local_ai_fidelity.py`
- Test: `backend/tests/test_strict_local_egress.py`
- Test: `backend/tests/test_local_ai_log_privacy.py`
- Test: `backend/tests/test_local_ai_cleanup.py`

**Interfaces:**

- Produces one signed, aggregate-only validation receipt per exact profile.
- A profile becomes selectable only when its lock and matching receipt ship.

- [ ] **Step 1: Add failing profile-gate tests**

```python
def test_linux_promotion_requires_every_gate(
    pack_root: Path,
    staged_pack: Path,
) -> None:
    manifest = cpu_manifest()
    invalid = RuntimeValidationReceipt(
        payload=expected_validation_payload(manifest),
        _seal=object(),
    )
    with pytest.raises(LocalValidationError, match="receipt"):
        ArtifactStore(pack_root).activate_validated(
            staged_pack,
            manifest,
            invalid,
        )


def test_cuda_receipt_cannot_promote_cpu_lock(
    pack_root: Path,
    staged_pack: Path,
) -> None:
    manifest = cpu_manifest()
    cuda_receipt = _issue_runtime_validation_receipt(cuda_manifest())
    with pytest.raises(LocalValidationError, match="receipt"):
        ArtifactStore(pack_root).activate_validated(
            staged_pack,
            manifest,
            cuda_receipt,
        )
```

- [ ] **Step 2: Run and confirm promotion is unavailable**

```bash
cd backend
uv run pytest tests/test_local_ai_fidelity.py \
  tests/test_local_ai_resources.py -v
```

Expected: tests fail because Linux promotion is not implemented.

- [ ] **Step 3: Preserve the Apple fidelity thresholds**

Each exact profile must pass:

- unsupported summary facts: zero;
- critical numeric OCR exactness: at least 99%;
- critical extraction precision: at least 98%;
- critical extraction recall: at least 95%;
- all accepted extraction facts have valid evidence;
- every emitted summary claim references allowed fact fields and evidence;
- no PHI canary in stdout, stderr, audit, progress, crash, or report JSON;
- no scratch after success, failure, timeout, cancellation, or startup sweep;
- denied external sockets during every inference role;
- one role process resident at a time;
- three cold runs with no OOM, no sustained swap thrash, and no orphan process;
- final RSS and VRAM retention no more than 20% above clean baseline.

The CPU gate runs on an admission-target CPU host. The CUDA gate runs
separately on an admission-target NVIDIA host. Passing one does not provide
evidence for the other.

- [ ] **Step 4: Add public CI without model claims**

`local-ai-linux-contract-ci.yml` runs on GitHub-hosted Ubuntu and:

- builds the CPU runtime from the pinned commit;
- syntax-checks the CUDA configuration without claiming hardware validation;
- installs the worker without downloading model artifacts;
- runs protocol, fake-runner, manifest, egress, log, and cleanup contracts;
- asserts no `llama-server`, socket listener, TCP listener, or server port
  appears in source or process checks.

Exact-model fidelity and memory gates run only on dedicated Linux CPU/NVIDIA
release runners. Their aggregate receipts are reviewed before a lock is added.

- [ ] **Step 5: Run the CPU release gate**

```bash
cd backend
LOCAL_AI_PROFILE=linux_x86_64_cpu \
LOCAL_AI_ENABLED=true \
uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-linux-cpu-fidelity.json

LOCAL_AI_PROFILE=linux_x86_64_cpu \
uv run python scripts/benchmark_local_ai.py \
  --runs 3 \
  --output artifacts/local-ai-linux-cpu-benchmark.json
```

Expected: both exit zero and the aggregate reports contain no clinical
content.

- [ ] **Step 6: Run the CUDA release gate on the separate NVIDIA host**

```bash
cd backend
LOCAL_AI_PROFILE=linux_x86_64_cuda \
LOCAL_AI_ENABLED=true \
uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-linux-cuda-fidelity.json

LOCAL_AI_PROFILE=linux_x86_64_cuda \
uv run python scripts/benchmark_local_ai.py \
  --runs 3 \
  --output artifacts/local-ai-linux-cuda-benchmark.json
```

Expected: both exit zero, CUDA is used without CPU substitution, and the
aggregate reports contain no clinical content.

- [ ] **Step 7: Generate and review locks**

Only after a profile passes, generate its `.lock.json` with every exact size
and hash plus the validation receipt digest. Re-run `local_ai_pack.py verify`
physically offline. Leave any failing profile catalog-only and unavailable.

- [ ] **Step 8: Hand each promoted profile to the root agent separately**

Report the changed files, exact lock and receipt digests, full release-gate
output, and hardware identity for one profile at a time. The root agent reviews
CPU and CUDA independently. Do not stage or commit.

---

### Task 8: Document only promoted Linux profiles

**Files:**

- Modify: `docs/operations-strict-local-ai.md`
- Modify: `docs/third-party-local-model-pack-notices.md`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `justfile`
- Test: `backend/tests/test_local_ai_manifest.py`

**Interfaces:**

- Makes profile selection and limitations visible to operators.
- Does not change runtime behavior.

- [ ] **Step 1: Add a docs contract for candidate language**

```python
def test_unpromoted_linux_profiles_are_not_described_as_validated() -> None:
    operations = OPERATIONS_DOC.read_text()
    assert "Linux candidates are not validated" in operations
    assert "LOCAL_AI_PROFILE=linux_x86_64_cpu" in operations
    assert "LOCAL_AI_PROFILE=linux_x86_64_cuda" in operations
```

- [ ] **Step 2: Update operations and notices**

Document:

- exact promoted profile names and admission/recommended hardware;
- native worker setup and profile-specific manifest/command values;
- no server, socket, port, container, cloud fallback, or CPU/CUDA substitution;
- candidate versus validated status;
- CPU performance expectations measured by the release receipt;
- CUDA driver/compute/VRAM requirements measured by the release receipt;
- model and runtime license attribution;
- backup behavior for immutable model files and exclusion of scratch.

If neither profile has passed, document both as unavailable candidates and do
not add install recipes to the main README.

- [ ] **Step 3: Run the humanizer audit**

Ask: "What makes this text obviously AI generated?" Remove promotional
language, em-dash chains, forced trios, generic conclusions, and repeated
claims. Keep limitations and measured figures concrete.

- [ ] **Step 4: Run docs, link, and static checks**

```bash
git diff --check -- README.md docs/operations-strict-local-ai.md \
  docs/third-party-local-model-pack-notices.md
rg -n 'llama-server|unix socket|localhost:[0-9]+' \
  workers/local_ai/linux_llamacpp scripts/setup-local-ai-linux.sh
cd backend
uv run pytest tests/test_local_ai_manifest.py \
  tests/test_local_ai_ci_workflows.py -q
```

Expected: the prose scan has no matches in new Linux text, the runtime scan
has no matches, and tests pass.

- [ ] **Step 5: Hand verified documentation to the root agent**

Report the changed files, humanizer audit, link/static checks, and any profile
whose wording still depends on release evidence. Do not stage or commit.

## Completion criteria

- Apple behavior and protocol snapshots are unchanged.
- CPU and CUDA are explicit, separate profiles with separate locks and
  receipts.
- All roles use GGUF through one pinned `llama.cpp` source revision.
- The backend talks to Linux through the existing stdin/stdout JSONL manager.
- No server, socket, port, container, Transformers runtime, or cloud fallback
  exists in the Linux path.
- Candidate artifacts remain unavailable until exact offline, fidelity,
  privacy, cleanup, and resource gates pass.
- Each promoted profile has three cold-run evidence on its own target hardware.
- Operator docs distinguish measured release facts from candidate targets.
