# Strict-Local Medical Model Pack: Linux CPU and GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the validated strict-local pipeline to Linux with a supported 16 GB CPU-only profile and separately validated NVIDIA CUDA and AMD ROCm profiles, while preserving the Apple plan's schemas, model roles, fail-closed behavior, evidence, checkpoints, and release thresholds.

**Architecture:** Reuse the platform-neutral core, immutable manifest, artifact store, validators, database schema, APIs, and UI delivered by `2026-07-26-strict-local-apple-mlx.md`. Linux inference runs in a non-root, network-disabled container behind an owner-only Unix socket. A small supervisor spawns exactly one role process at a time: reviewed Transformers/safetensors adapters for OvisOCR2 and NuExtract3, and pinned llama.cpp/GGUF for Qwen3.5-9B. A physically separate downloader service has network and model-cache access but no upload, scratch, database, or inference access. CPU, CUDA, and ROCm use separate locked images and validation manifests.

**Tech Stack:** Python 3.11, `transformers==5.14.1`, `torch==2.13.0`, `safetensors==0.8.0`, `huggingface-hub==1.24.0`, pinned llama.cpp `b9637` source lock, GGUF, Docker Compose profiles, Unix sockets, pytest, synthetic/private fidelity fixtures.

## Global Constraints

- Complete and verify the shared/Apple plan first. This plan changes platform adapters and packaging, not clinical schemas or privacy semantics.
- Linux CPU with 16 GB system RAM is a required supported profile. GPU acceleration is optional.
- CPU, CUDA, and ROCm runtime dependencies live in separate locked worker projects/images. Never add Torch, Transformers, llama.cpp, CUDA, or ROCm to the main backend environment.
- Model roles remain fixed: OvisOCR2 OCR, NuExtract3 ingestion/extraction, Qwen3.5-9B summary only.
- Only one role process is resident at a time. The supervisor may persist, but it must reap the role process before starting another.
- Inference containers use `network_mode: none`, non-root users, read-only root filesystems, dropped capabilities, `no-new-privileges`, bounded tmpfs, read-only model mounts, and a bounded scratch mount.
- No inference service exposes TCP/HTTP or host ports. Backend-to-worker traffic uses a private Unix socket volume and protocol version `1`.
- The downloader service has network plus model-cache access only. It has no upload, scratch, database, backend secrets, or inference socket mount.
- Downloaded repositories remain untrusted data. `trust_remote_code=False`; reject Python, pickle, executable, symlink, traversal, `auto_map`, and unbounded artifact sets.
- The Linux worker may ship reviewed architecture/preprocessor code in its own source tree. It may not execute repository code. Preserve license/attribution for every ported component.
- Prefer llama.cpp only when the exact GGUF plus multimodal projector/runtime passes the full fixture suite. A model-card claim or successful load alone is insufficient.
- The initial required CPU profile uses official safetensors for OvisOCR2 and NuExtract3 if their GGUF/libmtmd paths fail; Qwen3.5-9B uses `Q4_K_M` GGUF.
- No silent runtime substitution. Each job snapshots platform/profile/backend/model revision. A failed `llama_cpp` role does not fall through to Transformers, GPU, another quantization, or cloud.
- Accelerated profiles use the same prompts, decode settings, schemas, validators, checkpoints, and evidence contracts as CPU.
- Advertise a platform/profile only after it passes egress, content-log, cleanup, fidelity, peak-memory, repeated-run, and unload/reclamation gates on real hardware.
- Public CI builds images and uses tiny/fake artifacts. Multi-GB model gates run on dedicated Linux hosts.

---

### Task 1: Add Linux platform profiles and runtime selection

**Files:**

- Create: `backend/app/services/local_ai/platforms.py`
- Create: `backend/app/services/local_ai/runtime_registry.py`
- Modify: `backend/app/services/local_ai/types.py`
- Modify: `backend/app/services/local_ai/model_manager.py`
- Modify: `backend/app/schemas/local_ai.py`
- Modify: `backend/app/api/local_ai.py`
- Modify: `backend/app/config.py`
- Create: `backend/tests/test_local_ai_platforms.py`
- Create: `backend/tests/test_local_ai_runtime_registry.py`

**Interfaces:**

- `PlatformProfile`
- `detect_platform_profile()`
- `RuntimeAdapter`
- `runtime_for_profile(profile)`
- Linux worker/downloader socket settings

- [ ] **Step 1: Write failing platform tests**

```python
# backend/tests/test_local_ai_platforms.py
def test_linux_without_gpu_is_supported_cpu(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("app.services.local_ai.platforms.detect_nvidia", lambda: False)
    monkeypatch.setattr("app.services.local_ai.platforms.detect_rocm", lambda: False)
    profile = detect_platform_profile(total_memory_bytes=16 * 1024**3)
    assert profile.name == "linux_cpu"
    assert profile.compatible is True


def test_linux_below_16gb_is_not_compatible(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    profile = detect_platform_profile(total_memory_bytes=15 * 1024**3)
    assert profile.compatible is False
    assert profile.reason == "16 GB system RAM is required"


def test_gpu_detection_does_not_silently_select_acceleration(monkeypatch) -> None:
    profile = detect_platform_profile(
        total_memory_bytes=32 * 1024**3,
        requested_profile="linux_cpu",
    )
    assert profile.name == "linux_cpu"
```

- [ ] **Step 2: Run and verify missing-module failures**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_platforms.py \
  tests/test_local_ai_runtime_registry.py -v
```

Expected: collection fails for `platforms` and `runtime_registry`.

- [ ] **Step 3: Implement explicit profiles**

```python
# backend/app/services/local_ai/platforms.py
from dataclasses import dataclass
from typing import Literal

ProfileName = Literal[
    "apple_silicon", "linux_cpu", "linux_cuda", "linux_rocm", "unsupported"
]


@dataclass(frozen=True)
class PlatformProfile:
    name: ProfileName
    compatible: bool
    total_memory_bytes: int
    accelerator: str | None
    reason: str | None = None


def detect_platform_profile(
    *,
    total_memory_bytes: int,
    requested_profile: ProfileName | None = None,
) -> PlatformProfile:
    if total_memory_bytes < 16 * 1024**3:
        return PlatformProfile(
            "unsupported", False, total_memory_bytes, None,
            "16 GB system RAM is required",
        )
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return PlatformProfile("apple_silicon", True, total_memory_bytes, "metal")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        return PlatformProfile("unsupported", False, total_memory_bytes, None)
    selected = requested_profile or "linux_cpu"
    if selected == "linux_cuda" and not detect_nvidia():
        return PlatformProfile(selected, False, total_memory_bytes, "cuda", "CUDA unavailable")
    if selected == "linux_rocm" and not detect_rocm():
        return PlatformProfile(selected, False, total_memory_bytes, "rocm", "ROCm unavailable")
    accelerator = {"linux_cpu": None, "linux_cuda": "cuda", "linux_rocm": "rocm"}[selected]
    return PlatformProfile(selected, True, total_memory_bytes, accelerator)
```

Runtime selection maps each exact profile name to an adapter and manifest; it never auto-upgrades CPU to a detected GPU:

```python
RUNTIME_PROFILES = {
    "apple_silicon": RuntimeAdapter(kind="stdio", command=settings.local_ai_worker_command),
    "linux_cpu": RuntimeAdapter(kind="unix", socket=settings.local_ai_linux_socket),
    "linux_cuda": RuntimeAdapter(kind="unix", socket=settings.local_ai_linux_socket),
    "linux_rocm": RuntimeAdapter(kind="unix", socket=settings.local_ai_linux_socket),
}
```

Add:

```python
local_ai_profile: str = ""
local_ai_linux_socket: str = "/run/local-ai/inference/worker.sock"
local_ai_downloader_socket: str = "/run/local-ai/downloader/downloader.sock"
```

The API status uses the same frontend values `linux_cpu`, `linux_cuda`, and `linux_rocm`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_platforms.py \
  tests/test_local_ai_runtime_registry.py tests/test_local_ai_api.py -v
uv run ruff check app/services/local_ai/platforms.py \
  app/services/local_ai/runtime_registry.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/local_ai/platforms.py \
  backend/app/services/local_ai/runtime_registry.py \
  backend/app/services/local_ai/types.py \
  backend/app/services/local_ai/model_manager.py \
  backend/app/schemas/local_ai.py backend/app/api/local_ai.py \
  backend/app/config.py backend/tests/test_local_ai_platforms.py \
  backend/tests/test_local_ai_runtime_registry.py
git commit -m "feat(local-ai): add explicit Linux runtime profiles"
```

---

### Task 2: Lock Linux model and runtime artifacts

**Files:**

- Create: `backend/app/model_manifests/linux-cpu-catalog-v1.json`
- Create: `backend/app/model_manifests/linux-cuda-catalog-v1.json`
- Create: `backend/app/model_manifests/linux-rocm-catalog-v1.json`
- Modify: `backend/scripts/lock_local_ai_manifest.py`
- Create: `backend/scripts/lock_llama_cpp.py`
- Create: `backend/tests/test_linux_local_ai_manifest.py`
- Create: `backend/tests/test_llama_cpp_lock.py`

**Candidates:**

- OCR primary: official `ATH-MaaS/OvisOCR2` safetensors
- OCR experimental GGUF challenger: `Abiray/OvisOCR2-GGUF`
- Extraction primary: official `numind/NuExtract3` safetensors
- Extraction experimental GGUF challenger: `numind/NuExtract3-GGUF`
- Summary: `unsloth/Qwen3.5-9B-GGUF`, file `Qwen3.5-9B-Q4_K_M.gguf`
- llama.cpp source tag: `b9637`, resolved to an immutable commit

- [ ] **Step 1: Write failing Linux manifest tests**

```python
# backend/tests/test_linux_local_ai_manifest.py
def test_cpu_manifest_has_exact_required_roles_and_backends(cpu_manifest) -> None:
    assert {item.role.value for item in cpu_manifest.artifacts} == {
        "ocr", "extraction", "summary",
    }
    assert cpu_manifest.profile == "linux_cpu"
    assert cpu_manifest.backend_by_role == {
        "ocr": "transformers",
        "extraction": "transformers",
        "summary": "llama_cpp",
    }


def test_multimodal_gguf_cannot_be_validated_without_projector_and_fixture_result() -> None:
    with pytest.raises(LocalValidationError, match="multimodal validation"):
        load_manifest(gguf_manifest_without_validation_record)
```

```python
# backend/tests/test_llama_cpp_lock.py
def test_llama_cpp_lock_rejects_mutable_tag_only(tmp_path: Path) -> None:
    with pytest.raises(LocalValidationError, match="immutable commit"):
        load_llama_cpp_lock({"tag": "b9637", "commit": ""})
```

- [ ] **Step 2: Run and verify failures**

Run:

```bash
cd backend
uv run pytest tests/test_linux_local_ai_manifest.py \
  tests/test_llama_cpp_lock.py -v
```

Expected: tests fail because Linux profile fields and llama.cpp lock support are absent.

- [ ] **Step 3: Extend manifest schema with exact backend/profile locks**

Each role records:

```json
{
  "role": "summary",
  "repository": "unsloth/Qwen3.5-9B-GGUF",
  "revision": "40-character-immutable-commit-written-by-locker",
  "backend": "llama_cpp",
  "quantization": "Q4_K_M",
  "files": [
    {
      "path": "Qwen3.5-9B-Q4_K_M.gguf",
      "sha256": "64-character-digest-written-by-locker",
      "size": 5840000000
    }
  ],
  "runtime_lock": "llama-cpp-b9637.lock.json"
}
```

The strings above describe generated schema shape; the checked-in lock is produced by the locking command and contains real immutable values, never the descriptive strings.

For Transformers roles, include official safetensors shards/config/tokenizer/processor assets only. If the worker needs architecture code, the manifest records the reviewed worker adapter version, not repository `.py` files.

- [ ] **Step 4: Lock llama.cpp source and image inputs**

`lock_llama_cpp.py` resolves tag `b9637` through the official GitHub API, verifies the commit/tag relationship, records source archive URL/hash, license hash, CMake flags, compiler/base-image digests, and writes canonical JSON. It rejects unsigned/mismatched source metadata and never downloads prebuilt third-party binaries.

Run:

```bash
cd backend
uv run python scripts/lock_llama_cpp.py \
  --tag b9637 \
  --output app/model_manifests/llama-cpp-b9637.lock.json
uv run python scripts/lock_local_ai_manifest.py \
  --catalog app/model_manifests/linux-cpu-catalog-v1.json \
  --output data/local-ai/candidates/linux-cpu-16gb-v1.lock.json
```

Expected: both exit `0`; output locks contain immutable commits and SHA-256 values. No candidate is activated.

- [ ] **Step 5: Run tests and commit catalogs/lock tooling**

Run:

```bash
cd backend
uv run pytest tests/test_linux_local_ai_manifest.py \
  tests/test_llama_cpp_lock.py tests/test_local_ai_manifest.py -v
uv run ruff check scripts/lock_llama_cpp.py scripts/lock_local_ai_manifest.py
```

Expected: all pass.

```bash
git add backend/app/model_manifests/linux-*-catalog-v1.json \
  backend/app/model_manifests/llama-cpp-b9637.lock.json \
  backend/scripts/lock_local_ai_manifest.py backend/scripts/lock_llama_cpp.py \
  backend/tests/test_linux_local_ai_manifest.py backend/tests/test_llama_cpp_lock.py
git commit -m "feat(local-ai): lock Linux models and llama cpp runtime"
```

---

### Task 3: Build the Linux CPU supervisor and role workers

**Files:**

- Create: `workers/local_ai/linux_cpu/pyproject.toml`
- Create: `workers/local_ai/linux_cpu/uv.lock`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/__init__.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/supervisor.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/protocol.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/artifacts.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/ovisocr2.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/nuextract3.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/qwen_summary.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/adapters/ovis.py`
- Create: `workers/local_ai/linux_cpu/src/local_ai_linux_worker/adapters/nuextract.py`
- Create: `workers/local_ai/linux_cpu/tests/test_supervisor.py`
- Create: `workers/local_ai/linux_cpu/tests/test_offline_loading.py`
- Create: `workers/local_ai/linux_cpu/tests/test_role_contracts.py`

**Dependencies:**

```toml
dependencies = [
    "transformers==5.14.1",
    "torch==2.13.0",
    "safetensors==0.8.0",
    "huggingface-hub==1.24.0",
    "Pillow==12.3.0",
]
```

- [ ] **Step 1: Write failing supervisor/contract tests**

```python
# workers/local_ai/linux_cpu/tests/test_supervisor.py
@pytest.mark.asyncio
async def test_supervisor_reaps_role_before_starting_next(supervisor) -> None:
    await supervisor.run("ocr", fake_request(delay_ms=20))
    first_pid = supervisor.last_reaped_pid
    await supervisor.run("extraction", fake_request(delay_ms=20))
    assert supervisor.max_live_role_processes == 1
    assert first_pid in supervisor.reaped_pids


@pytest.mark.asyncio
async def test_cancel_kills_role_process_without_killing_supervisor(supervisor) -> None:
    request = asyncio.create_task(supervisor.run("ocr", fake_request(block=True)))
    await supervisor.cancel("job-1")
    with pytest.raises(WorkerCancelled):
        await request
    assert supervisor.is_serving
    assert supervisor.active_role_pid is None
```

```python
# workers/local_ai/linux_cpu/tests/test_offline_loading.py
@pytest.mark.local_model
@pytest.mark.parametrize("role", ["ocr", "extraction", "summary"])
def test_cpu_role_loads_offline_without_repository_code(role, manifest, model_dir):
    with deny_all_network():
        loaded = load_role(role, manifest, model_dir)
    assert loaded.trust_remote_code is False
    assert loaded.repository_python_files == []
```

- [ ] **Step 2: Create the isolated project and observe failure**

Run:

```bash
cd workers/local_ai/linux_cpu
uv lock
uv sync
uv run pytest tests/test_supervisor.py tests/test_role_contracts.py -v
```

Expected: fails because the supervisor/role modules do not exist.

- [ ] **Step 3: Implement a non-content Unix-socket supervisor**

The supervisor:

- binds only `/run/local-ai/inference/worker.sock` with mode `0600`;
- accepts protocol version `1`;
- validates manifest hash and role before spawning;
- launches a fresh role subprocess with no proxy/token/provider variables;
- forwards strict progress/result/error JSON;
- enforces timeout/output/input limits;
- SIGTERM/SIGKILLs and reaps on cancel/error;
- never holds model objects itself.

```python
class RoleSupervisor:
    async def run(self, role: ModelRole, request: WorkerRequest) -> WorkerResponse:
        async with self.role_lock:
            process = await self._spawn_role(role, request)
            self.active_role_pid = process.pid
            try:
                return await self._exchange(process, request)
            finally:
                await self._terminate_and_reap(process)
                self.active_role_pid = None
```

- [ ] **Step 4: Implement reviewed offline role adapters**

- OvisOCR2 uses official safetensors and a worker-shipped `adapters/ovis.py` architecture/preprocessor when Transformers lacks native support.
- NuExtract3 uses official safetensors and worker-shipped `adapters/nuextract.py`, non-thinking mode, the exact shared JSON template, and bounded output.
- Qwen summary executes the locally built pinned `llama-cli` subprocess with the exact Q4_K_M artifact, grammar/schema-constrained JSON, fixed context/output limits, and no server.

Ported adapter code is reviewed, covered by tensor-shape/preprocessing/golden-output tests, and listed in third-party notices. Do not copy or import repository Python at runtime.

- [ ] **Step 5: Run offline artifact compatibility on a 16 GB CPU host**

Run:

```bash
cd workers/local_ai/linux_cpu
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run pytest tests/test_offline_loading.py tests/test_role_contracts.py \
  -m local_model -v -rs
```

Expected: all three roles load and produce schema-valid fixtures without network/repository code. If an adapter is missing, implement it in the worker source and rerun; do not activate or advertise the CPU profile until all three pass.

- [ ] **Step 6: Run unit tests and commit**

Run:

```bash
cd workers/local_ai/linux_cpu
uv run pytest tests/test_supervisor.py tests/test_role_contracts.py -v
uv run ruff check src tests
```

Expected: all pass.

```bash
git add workers/local_ai/linux_cpu
git commit -m "feat(local-ai): add isolated Linux CPU role workers"
```

---

### Task 4: Package isolated inference and downloader services

**Files:**

- Create: `workers/local_ai/docker/Dockerfile.cpu`
- Create: `workers/local_ai/downloader/pyproject.toml`
- Create: `workers/local_ai/downloader/uv.lock`
- Create: `workers/local_ai/downloader/src/local_ai_downloader/__init__.py`
- Create: `workers/local_ai/downloader/src/local_ai_downloader/server.py`
- Create: `workers/local_ai/downloader/tests/test_server.py`
- Create: `docker-compose.local-ai.yml`
- Modify: `.env.docker.example`
- Modify: `.dockerignore`
- Create: `backend/tests/test_linux_local_ai_compose.py`

**Services:**

- `local-ai-inference-cpu`: network disabled; model read-only; scratch and inference socket only
- `local-ai-downloader`: network enabled; model cache and downloader socket only

- [ ] **Step 1: Write failing Compose security tests**

```python
# backend/tests/test_linux_local_ai_compose.py
def test_cpu_inference_has_no_network_or_public_ports(compose_config) -> None:
    service = compose_config["services"]["local-ai-inference-cpu"]
    assert service["network_mode"] == "none"
    assert "ports" not in service
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]


def test_downloader_cannot_mount_phi_or_inference_ipc(compose_config) -> None:
    mounts = compose_mount_targets(compose_config["services"]["local-ai-downloader"])
    assert "/data/uploads" not in mounts
    assert "/run/local-ai/inference" not in mounts
    assert "/data/local-ai/scratch" not in mounts


def test_inference_model_mount_is_read_only(compose_config) -> None:
    mounts = compose_mounts(compose_config["services"]["local-ai-inference-cpu"])
    assert mounts["/models"]["read_only"] is True
```

- [ ] **Step 2: Run and verify missing Compose failure**

Run:

```bash
cd backend
uv run pytest tests/test_linux_local_ai_compose.py -v
```

Expected: fails because `docker-compose.local-ai.yml` does not exist.

- [ ] **Step 3: Build llama.cpp from the locked source**

`Dockerfile.cpu` uses a digest-pinned Debian bookworm build stage. It copies `llama-cpp-b9637.lock.json`, downloads the recorded source archive, verifies SHA-256, builds only CLI/libmtmd targets with CMake CPU flags, and copies binaries into the Python worker image. It runs the final image as uid/gid `65532`, sets read-only home/cache paths, and contains no compilers/package managers in the final stage.

The build fails if the resolved source hash or compiled binary smoke test differs from the lock.

- [ ] **Step 4: Implement the downloader socket service**

The downloader dependencies are:

```toml
dependencies = [
    "httpx==0.28.1",
    "pydantic==2.12.5",
]
```

The service accepts only protocol messages containing action, pack revision, and operation id:

```python
class DownloaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    operation_id: UUID
    action: Literal["install", "update", "verify", "rollback", "remove"]
    pack_revision: str
```

It cannot receive upload/job/patient fields. The model cache is writable; no other project volume or secret is mounted. Bind the downloader Unix socket with mode `0600`.

- [ ] **Step 5: Add hardened Compose profiles**

Core shape:

```yaml
services:
  local-ai-inference-cpu:
    profiles: ["local-ai-cpu"]
    build:
      context: .
      dockerfile: workers/local_ai/docker/Dockerfile.cpu
    network_mode: none
    read_only: true
    user: "65532:65532"
    cap_drop: ["ALL"]
    security_opt: ["no-new-privileges:true"]
    pids_limit: 128
    tmpfs:
      - /tmp:size=256m,mode=0700,uid=65532,gid=65532
    volumes:
      - local_ai_models:/models:ro
      - local_ai_scratch:/scratch
      - local_ai_inference_ipc:/run/local-ai/inference

  local-ai-downloader:
    profiles: ["local-ai-cpu", "local-ai-cuda", "local-ai-rocm"]
    build:
      context: .
      dockerfile: workers/local_ai/docker/Dockerfile.downloader
    read_only: true
    user: "65532:65532"
    cap_drop: ["ALL"]
    security_opt: ["no-new-privileges:true"]
    volumes:
      - local_ai_models:/models
      - local_ai_downloader_ipc:/run/local-ai/downloader
```

Mount `local_ai_inference_ipc` at `/run/local-ai/inference`,
`local_ai_downloader_ipc` at `/run/local-ai/downloader`,
`local_ai_scratch` at `/data/local-ai/scratch`, and `local_ai_models` read-only
at `/models` in the backend. Do not mount the downloader socket into inference.

- [ ] **Step 6: Build, inspect, and test**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu config > /tmp/medtimeline-local-ai-compose.yml
cd backend
uv run pytest tests/test_linux_local_ai_compose.py -v
cd ..
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu build local-ai-inference-cpu local-ai-downloader
```

Expected: configuration tests pass and both images build from locked inputs.

- [ ] **Step 7: Prove inference is disconnected**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu run --rm local-ai-inference-cpu \
  python -m local_ai_linux_worker.protocol --network-self-test
```

Expected: command exits `0` after proving `AF_UNIX` works and non-loopback TCP/UDP connections fail.

- [ ] **Step 8: Commit**

```bash
git add workers/local_ai/docker/Dockerfile.cpu workers/local_ai/downloader \
  docker-compose.local-ai.yml .env.docker.example .dockerignore \
  backend/tests/test_linux_local_ai_compose.py
git commit -m "feat(local-ai): isolate Linux inference and downloads"
```

---

### Task 5: Connect the backend manager to the Linux Unix-socket runtime

**Files:**

- Create: `backend/app/services/local_ai/unix_worker.py`
- Create: `backend/app/services/local_ai/downloader_client.py`
- Modify: `backend/app/services/local_ai/model_manager.py`
- Modify: `backend/app/services/local_ai/downloader.py`
- Modify: `backend/app/api/local_ai.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_linux_unix_worker.py`
- Create: `backend/tests/test_linux_downloader_boundary.py`
- Modify: `backend/tests/test_strict_local_pipeline.py`

- [ ] **Step 1: Write failing Unix-socket boundary tests**

```python
# backend/tests/test_linux_unix_worker.py
@pytest.mark.asyncio
async def test_unix_adapter_uses_shared_protocol_without_tcp(fake_unix_worker, deny_tcp):
    adapter = UnixWorkerAdapter(fake_unix_worker.socket_path)
    response = await adapter.run(ModelRole.OCR, synthetic_ocr_request())
    assert response.kind == "result"
    assert deny_tcp.attempts == []


@pytest.mark.asyncio
async def test_socket_permissions_must_be_owner_only(tmp_path: Path):
    socket_path = await bind_test_socket(tmp_path, mode=0o666)
    with pytest.raises(LocalPolicyError, match="0600"):
        await UnixWorkerAdapter(socket_path).health()
```

```python
# backend/tests/test_linux_downloader_boundary.py
def test_downloader_request_model_rejects_document_fields() -> None:
    with pytest.raises(ValidationError):
        DownloaderRequest.model_validate({
            "version": 1,
            "operation_id": str(uuid4()),
            "action": "install",
            "pack_revision": "linux-cpu-16gb-v1",
            "upload_id": str(uuid4()),
        })
```

- [ ] **Step 2: Run and verify missing modules**

Run:

```bash
cd backend
uv run pytest tests/test_linux_unix_worker.py \
  tests/test_linux_downloader_boundary.py -v
```

Expected: collection fails for the new modules.

- [ ] **Step 3: Implement strict Unix-socket clients**

`UnixWorkerAdapter` validates socket owner/mode, opens with `asyncio.open_unix_connection`, uses protocol version `1`, enforces maximum line size and timeout, and closes after each role. The server supervisor still owns role serialization; the backend global manager also retains its FIFO lock so multiple app workers cannot intentionally overlap.

`DownloaderClient` sends document-free operations only. On Linux, `api/local_ai.py` uses it instead of the in-process Apple downloader. Processing modules do not import `DownloaderClient`.

- [ ] **Step 4: Add startup readiness and fail-closed errors**

On startup, the backend:

1. Detects requested profile.
2. Validates the active manifest profile.
3. Checks inference socket ownership and protocol health.
4. Checks downloader health separately.
5. Marks pack unavailable if either required component is incompatible.

A missing inference socket produces `local_runtime_unavailable`; it never selects Apple stdio, a custom endpoint, cloud, or another Linux profile.

- [ ] **Step 5: Run fake Unix end-to-end tests**

Run:

```bash
cd backend
uv run pytest tests/test_linux_unix_worker.py \
  tests/test_linux_downloader_boundary.py tests/test_strict_local_pipeline.py \
  tests/test_strict_local_egress.py tests/test_local_ai_api.py -v
uv run ruff check app/services/local_ai/unix_worker.py \
  app/services/local_ai/downloader_client.py
```

Expected: all pass with zero TCP attempts.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/local_ai/unix_worker.py \
  backend/app/services/local_ai/downloader_client.py \
  backend/app/services/local_ai/model_manager.py \
  backend/app/services/local_ai/downloader.py backend/app/api/local_ai.py \
  backend/app/main.py backend/tests/test_linux_unix_worker.py \
  backend/tests/test_linux_downloader_boundary.py \
  backend/tests/test_strict_local_pipeline.py
git commit -m "feat(local-ai): connect Linux workers over private IPC"
```

---

### Task 6: Add separately locked NVIDIA CUDA and AMD ROCm profiles

**Files:**

- Create: `workers/local_ai/docker/Dockerfile.cuda`
- Create: `workers/local_ai/docker/Dockerfile.rocm`
- Create: `workers/local_ai/linux_cuda/pyproject.toml`
- Create: `workers/local_ai/linux_cuda/uv.lock`
- Create: `workers/local_ai/linux_rocm/pyproject.toml`
- Create: `workers/local_ai/linux_rocm/uv.lock`
- Modify: `docker-compose.local-ai.yml`
- Create: `backend/tests/test_linux_gpu_profiles.py`
- Create: `workers/local_ai/linux_cpu/tests/test_accelerator_parity.py`

- [ ] **Step 1: Write failing profile-isolation tests**

```python
# backend/tests/test_linux_gpu_profiles.py
def test_cuda_and_rocm_use_distinct_images_and_never_share_runtime_deps(compose_config):
    cuda = compose_config["services"]["local-ai-inference-cuda"]
    rocm = compose_config["services"]["local-ai-inference-rocm"]
    assert cuda["build"]["dockerfile"].endswith("Dockerfile.cuda")
    assert rocm["build"]["dockerfile"].endswith("Dockerfile.rocm")
    assert cuda["network_mode"] == rocm["network_mode"] == "none"


def test_gpu_profile_requires_explicit_selection(monkeypatch):
    monkeypatch.setattr("app.services.local_ai.platforms.detect_nvidia", lambda: True)
    profile = detect_platform_profile(total_memory_bytes=32 * 1024**3)
    assert profile.name == "linux_cpu"
```

- [ ] **Step 2: Run and verify missing-profile failures**

Run:

```bash
cd backend
uv run pytest tests/test_linux_gpu_profiles.py -v
```

Expected: fails because CUDA/ROCm services do not exist.

- [ ] **Step 3: Create separate dependency locks and images**

Each project pins the vendor-specific `torch==2.13.0` wheel source and every dependency hash. CUDA and ROCm images build llama.cpp with the matching backend flags from the same immutable `b9637` source lock. Final images keep the CPU service hardening, add only required device access, and expose no ports/network.

Compose:

```yaml
local-ai-inference-cuda:
  profiles: ["local-ai-cuda"]
  network_mode: none
  gpus: all

local-ai-inference-rocm:
  profiles: ["local-ai-rocm"]
  network_mode: none
  devices:
    - /dev/kfd
    - /dev/dri
```

Do not use `privileged: true`; retain capability drop/no-new-privileges/read-only root.

- [ ] **Step 4: Prove schema/output parity with CPU**

Run the same synthetic requests through CPU/CUDA/ROCm and compare accepted fact sets, evidence ids, unresolved/rejected fields, and summary fact/evidence references. Numeric/date/unit tokens must be exact; free-text wording may differ only after passing grounding validation.

```python
assert cuda.accepted_fact_set == cpu.accepted_fact_set
assert rocm.accepted_fact_set == cpu.accepted_fact_set
assert cuda.unsupported_claims == rocm.unsupported_claims == 0
```

- [ ] **Step 5: Build and test each available hardware profile**

Run on an NVIDIA host:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cuda build local-ai-inference-cuda
cd backend
LOCAL_AI_PROFILE=linux_cuda uv run pytest \
  tests/test_local_ai_fidelity.py -m "local_model and fidelity" -v
```

Run on an AMD ROCm host:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-rocm build local-ai-inference-rocm
cd backend
LOCAL_AI_PROFILE=linux_rocm uv run pytest \
  tests/test_local_ai_fidelity.py -m "local_model and fidelity" -v
```

Expected: the tested profile passes. A profile without a passing hardware report remains `compatible: false`/unadvertised; CPU support is unaffected.

- [ ] **Step 6: Commit**

```bash
git add workers/local_ai/docker/Dockerfile.cuda \
  workers/local_ai/docker/Dockerfile.rocm workers/local_ai/linux_cuda \
  workers/local_ai/linux_rocm docker-compose.local-ai.yml \
  backend/tests/test_linux_gpu_profiles.py \
  workers/local_ai/linux_cpu/tests/test_accelerator_parity.py
git commit -m "feat(local-ai): add isolated CUDA and ROCm profiles"
```

---

### Task 7: Validate Linux fidelity, egress, cleanup, and 16 GB resources

**Files:**

- Create: `backend/scripts/benchmark_local_ai_linux.py`
- Create: `backend/tests/test_linux_local_ai_egress.py`
- Create: `backend/tests/test_linux_local_ai_resources.py`
- Create: `backend/tests/test_linux_local_ai_cleanup.py`
- Modify: `backend/tests/test_local_ai_fidelity.py`
- Modify: `.github/workflows/local-ai-contract-ci.yml`
- Create: `.github/workflows/local-ai-linux-images.yml`

- [ ] **Step 1: Add failing container egress/cleanup tests**

Tests run strict-local PDF/TIFF ingestion and summary through Compose with every cloud key populated. They inspect:

- backend provider-construction spy;
- inference network namespace counters;
- downloader access logs;
- container stdout/stderr canaries;
- scratch volume after success/cancel/worker SIGKILL/backend restart;
- max simultaneous role PIDs.

Assertions:

```python
assert report.cloud_provider_constructions == 0
assert report.inference_external_connections == 0
assert report.downloader_document_fields == 0
assert report.phi_log_canary_hits == 0
assert report.remaining_plaintext_scratch_files == []
assert report.max_live_role_processes == 1
```

- [ ] **Step 2: Run fast fake/container contract tests**

Run:

```bash
cd backend
uv run pytest tests/test_linux_local_ai_egress.py \
  tests/test_linux_local_ai_cleanup.py -v
```

Expected: passes with fake/tiny workers.

- [ ] **Step 3: Implement content-free Linux benchmark reports**

Record CPU/GPU model, system RAM/VRAM, kernel/container/runtime versions, image digests, manifest/runtime hashes, cold-load time, peak RSS/VRAM, page/section/token throughput, swap delta, OOM events, thermal state when available, and post-role reclamation. Run three times.

CPU acceptance:

```python
assert report.profile == "linux_cpu"
assert report.system_ram_gib >= 16
assert report.oom_events == 0
assert report.sustained_swap_thrashing is False
assert report.max_live_role_processes == 1
assert report.post_unload_rss_ratio <= 0.20
```

- [ ] **Step 4: Run full CPU release gates on a physical 16 GB Linux host**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu up -d
cd backend
LOCAL_AI_PROFILE=linux_cpu uv run pytest \
  tests/test_local_ai_fidelity.py tests/test_linux_local_ai_egress.py \
  tests/test_linux_local_ai_cleanup.py tests/test_linux_local_ai_resources.py \
  -m "local_model or hardware or fidelity" -v -rs
uv run python scripts/benchmark_local_ai_linux.py \
  --profile linux_cpu --runs 3 \
  --output artifacts/local-ai-linux-cpu-benchmark.json
```

Expected: all shared hard privacy/fidelity gates pass; no OOM or sustained swap thrashing.

- [ ] **Step 5: Promote the exact CPU manifest**

Copy the tested candidate lock to `backend/app/model_manifests/linux-cpu-16gb-v1.lock.json`, add fixture-suite version, worker image digest, runtime lock hash, and benchmark-report SHA-256, then rerun pack verification. This is the only step that makes Linux CPU `validated: true`.

Repeat promotion separately for CUDA/ROCm only after their hardware gates pass. Never infer GPU validation from CPU results.

- [ ] **Step 6: Add image CI without downloading models**

`local-ai-linux-images.yml` builds CPU and syntax-checks CUDA/ROCm Dockerfiles from locked inputs, runs fake-worker contract/egress tests, and scans final dependency licenses. Pin all actions and base image digests. Hardware/model tests remain dedicated-runner release gates.

- [ ] **Step 7: Commit validation gates and promoted CPU manifest**

```bash
git add backend/scripts/benchmark_local_ai_linux.py \
  backend/tests/test_linux_local_ai_egress.py \
  backend/tests/test_linux_local_ai_resources.py \
  backend/tests/test_linux_local_ai_cleanup.py \
  backend/tests/test_local_ai_fidelity.py \
  backend/app/model_manifests/linux-cpu-16gb-v1.lock.json \
  .github/workflows/local-ai-contract-ci.yml \
  .github/workflows/local-ai-linux-images.yml
git commit -m "test(local-ai): validate the 16 GB Linux CPU profile"
```

---

### Task 8: Document Linux operation and run cross-platform verification

**Files:**

- Modify: `docs/operations-strict-local-ai.md`
- Modify: `docs/third-party-local-model-pack-notices.md`
- Modify: `docs/backend-handoff.md`
- Modify: `README.md`
- Modify: `.env.docker.example`
- Modify: `justfile`

- [ ] **Step 1: Add explicit Linux operations**

Document:

- CPU is the default Linux profile and requires 16 GB RAM.
- CUDA/ROCm require explicit profile selection and a validated hardware/runtime entry.
- Compose install/download/verify/start/stop/update/rollback/remove commands.
- Model cache, inference/downloader sockets, scratch volume, permissions, and backup behavior.
- Expected CPU slowness and resumable page/section progress.
- No automatic CPU↔GPU/runtime/model substitution.
- How to inspect exact manifest/image/runtime/benchmark identity.
- GGUF multimodal candidates remain unverified unless promoted by the exact fixture gate.

Add:

```make
local-ai-linux-cpu-up:
	docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
	  --profile local-ai-cpu up -d

local-ai-linux-cpu-verify:
	cd backend && LOCAL_AI_PROFILE=linux_cpu uv run python scripts/local_ai_pack.py verify

local-ai-linux-down:
	docker compose -f docker-compose.yml -f docker-compose.local-ai.yml down
```

- [ ] **Step 2: Humanize public prose**

Invoke the `humanizer` skill for README/user-facing operations text. Preserve exact commands, modes, hardware thresholds, qualification language, and medical disclaimer.

- [ ] **Step 3: Run backend and worker verification**

Run:

```bash
cd backend
uv run ruff check app tests scripts
uv run pytest -m "not slow and not local_model and not hardware" -x -q
cd ../workers/local_ai/linux_cpu
uv run ruff check src tests
uv run pytest tests/test_supervisor.py tests/test_role_contracts.py -v
```

Expected: all pass.

- [ ] **Step 4: Validate Compose security and images**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu config
docker compose -f docker-compose.yml -f docker-compose.local-ai.yml \
  --profile local-ai-cpu build
cd backend
uv run pytest tests/test_linux_local_ai_compose.py \
  tests/test_linux_local_ai_egress.py tests/test_linux_local_ai_cleanup.py -v
```

Expected: config/build/tests pass; inference has no network/public port.

- [ ] **Step 5: Run cross-platform contract/fidelity parity**

On the release hosts, run Apple, Linux CPU, and any advertised GPU profiles against the same synthetic/private fixture version. Compare accepted critical fact sets and evidence ids; all must meet the shared thresholds and have zero unsupported summary facts.

- [ ] **Step 6: Inspect scope and commit docs**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: no model weights, private fixtures, scratch, secrets, benchmark content with PHI, or unrelated files are staged.

```bash
git add docs/operations-strict-local-ai.md \
  docs/third-party-local-model-pack-notices.md docs/backend-handoff.md \
  README.md .env.docker.example justfile
git commit -m "docs(local-ai): document validated Linux profiles"
```

---

## Completion Criteria

- Linux CPU on a physical 16 GB host completes OCR → NuExtract validation/FHIR mapping → Qwen summary using the same contracts/evidence as Apple.
- The CPU inference container has no network, public port, database, upload mount, downloader socket, provider key, or analytics configuration.
- The downloader has network/model cache only and its request schema cannot carry document/job/patient fields.
- OvisOCR2 and NuExtract3 use official safetensors plus reviewed shipped adapters when GGUF/libmtmd is not validated; repository code is never executed.
- Qwen3.5-9B uses the exact pinned Q4_K_M GGUF and llama.cpp source/runtime lock.
- Exactly one role process exists at a time and is reaped before the next model loads.
- CPU passes all shared privacy/fidelity/cleanup gates without OOM or sustained swap thrashing.
- NVIDIA/AMD profiles are selectable only when explicitly requested and advertised only after their own hardware reports pass.
- Backend/API/frontend behavior, checkpoint keys, clinical schemas, validators, evidence ids, summary grounding, and mode semantics remain cross-platform compatible.
