# Strict-Local Medical Model Pack: Shared Core and Apple MLX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an optional, fail-closed local document-processing pack on a 16 GB Apple Silicon Mac that uses OvisOCR2 for OCR, NuExtract3 for grounded structured extraction, and Qwen3.5-9B only for grounded summaries, with no document or derived medical content sent to a cloud service.

**Architecture:** The FastAPI process owns platform-neutral contracts, immutable job snapshots, encrypted checkpoints/evidence, manifest verification, and a global FIFO model queue. An isolated native MLX worker project loads exactly one manifest-verified model at a time and communicates through versioned JSON Lines over stdin/stdout; it has no provider SDK, downloader, public port, or repository-code execution. Strict-local uploads branch before existing cloud-capable configuration is loaded. The existing entity validators and FHIR mapper remain downstream of a new NuExtract schema/evidence validator. The Next.js UI exposes the validated pack and four explicit processing modes inside Admin → System.

**Tech Stack:** Python 3.11 FastAPI/SQLAlchemy/Pydantic, PostgreSQL, `httpx`, `pypdfium2==5.12.1`, isolated Apple worker with `mlx-vlm==0.5.0`, JSON Lines IPC, Next.js/TypeScript/Zustand/Playwright, pytest/pytest-asyncio.

## Global Constraints

- Preserve backend `requires-python = ">=3.11.8,<3.12"` and keep MLX out of the main backend lock/import graph.
- Preserve exactly four Admin tabs. Local AI belongs in Admin → System; `/settings` remains a redirect.
- The accepted modes are exactly `validated_strict_local`, `custom_local`, `cloud_assisted`, and `prompt_only`.
- Default existing users to `cloud_assisted`; mode changes affect new jobs only. Every upload/summary job stores an immutable mode and model-manifest snapshot.
- A strict-local job must branch before `load_llm_config`, `_vision_candidates`, `_ocr_via_provider`, `_resolve_extraction_engine`, LangExtract, or any generic/cloud provider construction.
- A strict-local failure is terminal or resumable locally. It never changes mode, model, runtime, or provider.
- The validated worker accepts only local, manifest-verified artifact paths; set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_DISABLE_TELEMETRY=1`.
- `trust_remote_code=False` is non-negotiable. Reject repository Python, pickle weights, executables, symlinks, path traversal, and configs containing `auto_map` or remote-code requirements.
- The downloader may use the network but receives no upload/job identifiers or medical content. Processing workers cannot invoke downloader code.
- Only one heavyweight model process may exist at a time across all uploads and summaries. Kill and reap the process to release Metal memory between roles.
- Raw pages, OCR, prompts, extracted values, evidence excerpts, and summaries must never enter operational logs, analytics, notices, progress JSONB, or generic audit details.
- PHI-bearing durable values use the repository's encrypted SQLAlchemy types. Plaintext scratch is mode `0700`/`0600` and is removed on success, cancellation, failure, and startup recovery.
- Keep existing cloud-assisted fallback behavior and its tests intact; make privacy claims mode-specific.
- Keep existing no-diagnosis/no-treatment-advice rules. Strict-local custom system text can add constraints but cannot replace the server-owned safety prompt.
- Runtime dependencies and model licenses must be MIT, Apache-2.0, or BSD-family. Resolve the repository's existing `frozendict` license exception before claiming the entire product runtime is permissive-only.
- Public CI uses fake workers and synthetic fixtures. Real model fidelity/resource acceptance runs on the physical 16 GB M4 with private fixtures from `REAL_MEDICAL_FIXTURES_DIR`.
- Use test-driven development: add a focused failing test, run it and observe the stated failure, implement only the named behavior, rerun the focused test, then commit.

---

### Task 1: Establish processing-mode contracts and fail-closed policy

**Files:**

- Create: `backend/app/services/local_ai/__init__.py`
- Create: `backend/app/services/local_ai/types.py`
- Create: `backend/app/services/local_ai/errors.py`
- Create: `backend/app/services/local_ai/policy.py`
- Test: `backend/tests/test_local_ai_contracts.py`
- Test: `backend/tests/test_local_ai_policy.py`

**Interfaces:**

- `ProcessingMode`
- `ModelRole`
- `ModelIdentity`
- `OCRPageRequest` / `OCRPageResult`
- `ExtractionRequest` / `ClinicalExtraction`
- `SummaryRequest` / `GroundedSummary`
- `OCRBackend`, `ExtractionBackend`, `SummaryBackend`
- `assert_processing_route(mode, endpoint)`

- [ ] **Step 1: Write the failing contract and policy tests**

```python
# backend/tests/test_local_ai_contracts.py
from dataclasses import asdict

from app.services.local_ai.types import (
    ModelIdentity,
    ModelRole,
    OCRPageRequest,
    ProcessingMode,
)


def test_ocr_request_is_serializable_and_carries_immutable_model_identity() -> None:
    model = ModelIdentity(
        role=ModelRole.OCR,
        repository="sahilchachra/ovisocr2-int4-mlx",
        revision="0123456789abcdef0123456789abcdef01234567",
        quantization="int4",
        runtime="mlx-vlm-0.5.0",
    )
    request = OCRPageRequest(
        job_id="job-1",
        page_number=1,
        image_path="/private/job-1/page-0001.png",
        image_sha256="a" * 64,
        model=model,
        max_output_tokens=8192,
    )
    assert asdict(request)["model"]["repository"] == model.repository


def test_processing_modes_are_explicit_and_stable() -> None:
    assert {mode.value for mode in ProcessingMode} == {
        "validated_strict_local",
        "custom_local",
        "cloud_assisted",
        "prompt_only",
    }
```

```python
# backend/tests/test_local_ai_policy.py
import pytest

from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.policy import assert_processing_route, require_loopback
from app.services.local_ai.types import ProcessingMode


def test_strict_local_rejects_every_network_endpoint() -> None:
    with pytest.raises(LocalPolicyError, match="embedded worker"):
        assert_processing_route(
            ProcessingMode.VALIDATED_STRICT_LOCAL,
            "https://generativelanguage.googleapis.com",
        )


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8000"],
)
def test_custom_local_accepts_only_loopback(url: str) -> None:
    require_loopback(url)


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://192.168.1.9:11434"])
def test_custom_local_rejects_non_loopback(url: str) -> None:
    with pytest.raises(LocalPolicyError, match="loopback"):
        require_loopback(url)
```

- [ ] **Step 2: Run the tests and verify the missing-package failure**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_contracts.py tests/test_local_ai_policy.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.services.local_ai'`.

- [ ] **Step 3: Implement serializable contracts and the policy boundary**

```python
# backend/app/services/local_ai/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ProcessingMode(StrEnum):
    VALIDATED_STRICT_LOCAL = "validated_strict_local"
    CUSTOM_LOCAL = "custom_local"
    CLOUD_ASSISTED = "cloud_assisted"
    PROMPT_ONLY = "prompt_only"


class ModelRole(StrEnum):
    OCR = "ocr"
    EXTRACTION = "extraction"
    SUMMARY = "summary"


@dataclass(frozen=True)
class ModelIdentity:
    role: ModelRole
    repository: str
    revision: str
    quantization: str
    runtime: str


@dataclass(frozen=True)
class EvidenceReference:
    id: str
    upload_id: str
    page_number: int | None
    section: str | None
    excerpt: str
    start_offset: int | None
    end_offset: int | None
    field_paths: list[str]


@dataclass(frozen=True)
class OCRPageRequest:
    job_id: str
    page_number: int
    image_path: str
    image_sha256: str
    model: ModelIdentity
    max_output_tokens: int


@dataclass
class OCRPageResult:
    page_number: int
    markdown: str
    width: int
    height: int
    warnings: list[str]
    content_sha256: str
    model: ModelIdentity


@dataclass(frozen=True)
class ExtractionRequest:
    job_id: str
    upload_id: str
    page_markdown: list[dict[str, Any]]
    image_paths: dict[int, str]
    schema_version: str
    model: ModelIdentity


@dataclass
class ClinicalExtraction:
    entities: list[dict[str, Any]]
    evidence: list[EvidenceReference]
    unresolved_fields: list[str] = field(default_factory=list)
    rejected_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SummaryRequest:
    job_id: str
    summary_type: str
    facts: list[dict[str, Any]]
    evidence: list[EvidenceReference]
    model: ModelIdentity


@dataclass
class GroundedSummary:
    sections: list[dict[str, Any]]
    model: ModelIdentity


class OCRBackend(Protocol):
    async def parse_page(self, request: OCRPageRequest) -> OCRPageResult: ...


class ExtractionBackend(Protocol):
    async def extract(self, request: ExtractionRequest) -> ClinicalExtraction: ...


class SummaryBackend(Protocol):
    async def summarize(self, request: SummaryRequest) -> GroundedSummary: ...
```

```python
# backend/app/services/local_ai/errors.py
class LocalAIError(RuntimeError):
    code = "local_ai_error"
    retryable = False


class LocalPolicyError(LocalAIError):
    code = "local_policy_error"


class LocalWorkerError(LocalAIError):
    code = "local_worker_error"


class LocalWorkerTimeout(LocalWorkerError):
    code = "local_worker_timeout"
    retryable = True


class LocalValidationError(LocalAIError):
    code = "local_validation_error"
```

```python
# backend/app/services/local_ai/policy.py
from ipaddress import ip_address
from urllib.parse import urlparse

from app.services.local_ai.errors import LocalPolicyError
from app.services.local_ai.types import ProcessingMode


def require_loopback(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        raise LocalPolicyError("Custom local endpoint must be an HTTP loopback URL")
    if host == "localhost":
        return
    try:
        if ip_address(host).is_loopback:
            return
    except ValueError as exc:
        raise LocalPolicyError("Custom local endpoint must resolve to loopback") from exc
    raise LocalPolicyError("Custom local endpoint must use loopback")


def assert_processing_route(mode: ProcessingMode, endpoint: str | None) -> None:
    if mode is ProcessingMode.VALIDATED_STRICT_LOCAL:
        if endpoint is not None:
            raise LocalPolicyError("Validated local jobs use only the embedded worker")
        return
    if mode is ProcessingMode.CUSTOM_LOCAL:
        if endpoint is None:
            raise LocalPolicyError("Custom local mode requires a loopback endpoint")
        require_loopback(endpoint)
        return
    if mode is ProcessingMode.PROMPT_ONLY and endpoint is not None:
        raise LocalPolicyError("Prompt-only mode cannot call a provider")
```

Export the public types from `backend/app/services/local_ai/__init__.py`.

- [ ] **Step 4: Run focused tests and lint**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_contracts.py tests/test_local_ai_policy.py -v
uv run ruff check app/services/local_ai tests/test_local_ai_contracts.py tests/test_local_ai_policy.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/local_ai backend/tests/test_local_ai_contracts.py \
  backend/tests/test_local_ai_policy.py
git commit -m "feat(local-ai): add strict processing contracts and policy"
```

---

### Task 2: Add configuration and immutable manifest locking

**Files:**

- Modify: `backend/app/config.py`
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Create: `backend/app/services/local_ai/manifest.py`
- Create: `backend/app/model_manifests/catalog-v1.json`
- Create: `backend/app/model_manifests/schema-v1.json`
- Create: `backend/scripts/lock_local_ai_manifest.py`
- Create: `backend/tests/test_local_ai_manifest.py`
- Modify: `backend/.env.example`

**Interfaces:**

- `LocalAIManifest`
- `load_manifest(path)`
- `lock_catalog(catalog_path, output_path)`
- Settings: `local_ai_enabled`, `local_ai_model_dir`, `local_ai_scratch_dir`, `local_ai_manifest_path`, `local_ai_worker_command`, size/time limits

- [ ] **Step 1: Write failing manifest tests**

```python
# backend/tests/test_local_ai_manifest.py
import json
from pathlib import Path

import pytest

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.manifest import load_manifest


def test_manifest_requires_immutable_revision_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "artifacts": [{
            "role": "ocr",
            "repository": "sahilchachra/ovisocr2-int4-mlx",
            "revision": "main",
            "files": [{"path": "model.safetensors", "sha256": "bad", "size": 10}],
        }],
    }))
    with pytest.raises(LocalValidationError, match="immutable revision"):
        load_manifest(path)


def test_manifest_rejects_repository_code_and_pickle(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "pack_revision": "apple-m4-16gb-v1",
        "platform": "apple_silicon",
        "artifacts": [{
            "role": "ocr",
            "repository": "sahilchachra/ovisocr2-int4-mlx",
            "revision": "0" * 40,
            "files": [
                {"path": "modeling_ovis.py", "sha256": "a" * 64, "size": 10},
                {"path": "pytorch_model.bin", "sha256": "b" * 64, "size": 10},
            ],
        }],
    }))
    with pytest.raises(LocalValidationError, match="forbidden"):
        load_manifest(path)
```

- [ ] **Step 2: Run and verify the missing-module failure**

Run: `cd backend && uv run pytest tests/test_local_ai_manifest.py -v`

Expected: collection fails because `app.services.local_ai.manifest` does not exist.

- [ ] **Step 3: Add lightweight dependencies and isolated-path settings**

In `backend/pyproject.toml`, add only these core dependencies:

```toml
    "pypdfium2==5.12.1",
```

Do not add MLX, Transformers, Torch, safetensors, or Hugging Face Hub to the main backend.

Add settings to `backend/app/config.py`:

```python
    local_ai_enabled: bool = False
    local_ai_model_dir: str = "./data/local-ai/models"
    local_ai_scratch_dir: str = "./data/local-ai/scratch"
    local_ai_manifest_path: str = "./app/model_manifests/apple-m4-16gb-v1.lock.json"
    local_ai_worker_command: str = "../workers/local_ai/apple_mlx/.venv/bin/local-ai-mlx-worker"
    local_ai_max_files: int = 64
    local_ai_max_file_bytes: int = 8 * 1024 * 1024 * 1024
    local_ai_max_pack_bytes: int = 20 * 1024 * 1024 * 1024
    local_ai_worker_timeout_seconds: int = 900
    local_ai_max_page_pixels: int = 40_000_000
```

Document each variable in `backend/.env.example`. `LOCAL_AI_ENABLED=false` remains the default.

- [ ] **Step 4: Implement manifest parsing and catalog locking**

The shipped catalog contains candidates, not trusted hashes:

```json
{
  "schema_version": 1,
  "pack_revision": "apple-m4-16gb-v1",
  "platform": "apple_silicon",
  "runtime": {"name": "mlx-vlm", "version": "0.5.0"},
  "candidates": [
    {"role": "ocr", "repository": "sahilchachra/ovisocr2-int4-mlx", "quantization": "int4"},
    {"role": "extraction", "repository": "numind/NuExtract3-mlx-4bits", "quantization": "4bit"},
    {"role": "summary", "repository": "mlx-community/Qwen3.5-9B-MLX-4bit", "quantization": "4bit"}
  ]
}
```

Implement `manifest.py` with explicit allowlists:

```python
# backend/app/services/local_ai/manifest.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.services.local_ai.errors import LocalValidationError
from app.services.local_ai.types import ModelRole

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_SUFFIXES = {
    ".safetensors", ".json", ".txt", ".model", ".tiktoken", ".jinja",
    ".md", ".license",
}
FORBIDDEN_SUFFIXES = {".py", ".pyc", ".bin", ".pkl", ".pickle", ".so", ".dylib", ".exe"}


@dataclass(frozen=True)
class ManifestFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ManifestArtifact:
    role: ModelRole
    repository: str
    revision: str
    quantization: str
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class LocalAIManifest:
    schema_version: int
    pack_revision: str
    platform: str
    runtime: dict[str, str]
    artifacts: tuple[ManifestArtifact, ...]


def _safe_manifest_file(raw: dict) -> ManifestFile:
    path = PurePosixPath(raw["path"])
    if path.is_absolute() or ".." in path.parts:
        raise LocalValidationError("Manifest path traversal is forbidden")
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
        raise LocalValidationError(f"Manifest file type is forbidden: {path}")
    sha256 = raw["sha256"]
    if not SHA256_RE.fullmatch(sha256):
        raise LocalValidationError("Manifest SHA-256 must be 64 lowercase hex characters")
    return ManifestFile(str(path), sha256, int(raw["size"]))


def load_manifest(path: Path) -> LocalAIManifest:
    raw = json.loads(path.read_text())
    artifacts = []
    for item in raw["artifacts"]:
        revision = item["revision"]
        if not COMMIT_RE.fullmatch(revision):
            raise LocalValidationError("Every model must use an immutable revision")
        files = tuple(_safe_manifest_file(entry) for entry in item["files"])
        artifacts.append(ManifestArtifact(
            role=ModelRole(item["role"]),
            repository=item["repository"],
            revision=revision,
            quantization=item["quantization"],
            files=files,
        ))
    return LocalAIManifest(
        schema_version=int(raw["schema_version"]),
        pack_revision=raw["pack_revision"],
        platform=raw["platform"],
        runtime=raw["runtime"],
        artifacts=tuple(artifacts),
    )
```

`backend/scripts/lock_local_ai_manifest.py` must:

1. Resolve each repository's `main` reference through the Hugging Face REST API.
2. Fetch only repository metadata and manifest-allowed file bytes through `httpx`.
3. Reject repository entries containing `auto_map`, required Python, symlinks, LFS pointers without a resolved size/hash, or aggregate sizes above config.
4. Compute SHA-256 while streaming each allowed file into a temporary validation cache.
5. Write a canonical `*.lock.json` only after all candidates are complete.
6. Never activate the lock; Task 3 owns activation after runtime and fixture validation.

Use JSON Schema `backend/app/model_manifests/schema-v1.json` to require exactly three unique roles, a 40-character revision, a 64-character SHA-256, positive byte sizes, runtime/license/attribution fields, decode limits, and validation-suite version.

- [ ] **Step 5: Lock dependencies and run tests**

Run:

```bash
cd backend
uv lock
uv run pytest tests/test_local_ai_manifest.py -v
uv run ruff check app/services/local_ai/manifest.py scripts/lock_local_ai_manifest.py \
  tests/test_local_ai_manifest.py
```

Expected: manifest tests pass; the lockfile contains `pypdfium2==5.12.1`; Ruff passes.

- [ ] **Step 6: Exercise catalog locking without activating it**

Run:

```bash
cd backend
uv run python scripts/lock_local_ai_manifest.py \
  --catalog app/model_manifests/catalog-v1.json \
  --output data/local-ai/candidates/apple-m4-16gb-v1.lock.json
```

Expected: exits `0`, prints `locked 3 candidate artifacts`, and creates a canonical lock containing immutable revisions and hashes. If a repository requires remote code or a forbidden file, the command exits non-zero and no output lock exists; the pack remains unavailable rather than weakening the policy.

- [ ] **Step 7: Commit source and dependency lock, not downloaded artifacts**

```bash
git add backend/app/config.py backend/pyproject.toml backend/uv.lock \
  backend/.env.example backend/app/services/local_ai/manifest.py \
  backend/app/model_manifests backend/scripts/lock_local_ai_manifest.py \
  backend/tests/test_local_ai_manifest.py
git commit -m "feat(local-ai): validate and lock model manifests"
```

---

### Task 3: Implement a document-free downloader and atomic artifact store

**Files:**

- Create: `backend/app/services/local_ai/artifact_store.py`
- Create: `backend/app/services/local_ai/downloader.py`
- Create: `backend/tests/test_local_ai_artifacts.py`
- Create: `backend/tests/test_local_ai_downloader.py`

**Interfaces:**

- `ArtifactStore.stage(pack_revision)`
- `ArtifactStore.verify(staging_path, manifest)`
- `ArtifactStore.activate(staging_path, manifest)`
- `ArtifactStore.rollback()`
- `ArtifactStore.remove(role=None)`
- `download_manifest(manifest, store, progress_callback)`

- [ ] **Step 1: Write failing security and activation tests**

```python
# backend/tests/test_local_ai_artifacts.py
from pathlib import Path

import pytest

from app.services.local_ai.artifact_store import ArtifactStore
from app.services.local_ai.errors import LocalValidationError


def test_activation_is_atomic_and_preserves_previous_pack(tmp_path: Path, manifest) -> None:
    store = ArtifactStore(tmp_path)
    first = store.stage("first")
    write_manifest_files(first, manifest)
    store.activate(first, manifest)
    second = store.stage("second")
    write_manifest_files(second, manifest, corrupt_role="ocr")
    with pytest.raises(LocalValidationError, match="SHA-256"):
        store.activate(second, manifest)
    assert store.active_revision() == "first"


def test_store_rejects_symlinked_artifact(tmp_path: Path, manifest) -> None:
    store = ArtifactStore(tmp_path)
    stage = store.stage("bad")
    target = tmp_path / "outside"
    target.write_text("content")
    (stage / manifest.artifacts[0].files[0].path).symlink_to(target)
    with pytest.raises(LocalValidationError, match="symlink"):
        store.verify(stage, manifest)
```

```python
# backend/tests/test_local_ai_downloader.py
async def test_downloader_context_has_no_document_fields(fake_manifest, tmp_path) -> None:
    progress = []
    await download_manifest(fake_manifest, ArtifactStore(tmp_path), progress.append)
    assert progress
    assert all(set(item) <= {"role", "bytes_done", "bytes_total"} for item in progress)
```

- [ ] **Step 2: Run and verify missing-module failures**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_artifacts.py tests/test_local_ai_downloader.py -v
```

Expected: collection fails for the two missing modules.

- [ ] **Step 3: Implement the store and streaming downloader**

Use a cache layout that is physically separate from uploads/scratch:

```text
LOCAL_AI_MODEL_DIR/
  .staging/<operation-id>/
  packs/<pack-revision>/
  active.json
  previous.json
  operations/<operation-id>.json
```

Core activation code:

```python
# backend/app/services/local_ai/artifact_store.py
def activate(self, staging: Path, manifest: LocalAIManifest) -> None:
    self.verify(staging, manifest)
    destination = self.packs_dir / manifest.pack_revision
    if destination.exists():
        raise LocalValidationError("Pack revision already exists")
    staging.replace(destination)
    current = self._read_pointer("active.json")
    if current is not None:
        self._write_pointer_atomic("previous.json", current)
    self._write_pointer_atomic(
        "active.json",
        {"pack_revision": manifest.pack_revision, "manifest_sha256": manifest_sha256(manifest)},
    )
```

`downloader.py` uses a dedicated `httpx.AsyncClient(follow_redirects=True)` created only inside the download operation. Stream each manifest-listed URL to `<path>.partial`, enforce file/aggregate byte ceilings, fsync, verify the declared SHA-256, then rename. Its request/operation dataclasses contain only pack/model fields.

Persist operation state atomically:

```python
@dataclass(frozen=True)
class DownloadProgress:
    role: str
    bytes_done: int
    bytes_total: int


def progress_payload(value: DownloadProgress) -> dict[str, int | str]:
    return {
        "role": value.role,
        "bytes_done": value.bytes_done,
        "bytes_total": value.bytes_total,
    }
```

Never accept a caller-supplied URL; derive URLs from the repository and immutable revision in the locked candidate manifest.

- [ ] **Step 4: Run focused tests**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_artifacts.py tests/test_local_ai_downloader.py -v
uv run ruff check app/services/local_ai/artifact_store.py \
  app/services/local_ai/downloader.py tests/test_local_ai_artifacts.py \
  tests/test_local_ai_downloader.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/local_ai/artifact_store.py \
  backend/app/services/local_ai/downloader.py \
  backend/tests/test_local_ai_artifacts.py backend/tests/test_local_ai_downloader.py
git commit -m "feat(local-ai): add verified atomic model downloads"
```

---

### Task 4: Persist immutable jobs, encrypted page checkpoints, and evidence

**Files:**

- Create: `backend/app/models/local_ai.py`
- Modify: `backend/app/models/__init__.py`
- Modify: `backend/app/models/uploaded_file.py`
- Modify: `backend/app/models/llm_settings.py`
- Modify: `backend/app/models/ai_summary.py`
- Create: `backend/alembic/versions/f6a7b8c9d0e1_add_strict_local_jobs_and_evidence.py`
- Create: `backend/tests/test_local_ai_models.py`
- Modify: `backend/tests/test_llm_settings_models.py`
- Modify: `backend/tests/test_at_rest_encryption.py`

**Schema:**

- `user_llm_preferences.processing_mode VARCHAR(32) NULL`
- `uploaded_files.processing_mode VARCHAR(32) NOT NULL DEFAULT 'cloud_assisted'`
- `uploaded_files.processing_manifest JSONB NULL`
- `uploaded_files.processing_schema_version VARCHAR(32) NULL`
- `ai_summary_prompts.processing_mode VARCHAR(32) NOT NULL DEFAULT 'cloud_assisted'`
- `ai_summary_prompts.model_provenance JSONB NULL`
- `ai_summary_prompts.typed_response EncryptedJSON NULL`
- `local_ai_jobs`
- `local_ai_pages`
- `extraction_evidence`

- [ ] **Step 1: Write failing model and encryption tests**

```python
# backend/tests/test_local_ai_models.py
from app.models.local_ai import ExtractionEvidence, LocalAIJob, LocalAIPage


def test_local_job_locks_mode_and_manifest(user, upload) -> None:
    job = LocalAIJob(
        user_id=user.id,
        upload_id=upload.id,
        kind="ingestion",
        processing_mode="validated_strict_local",
        manifest_snapshot={"pack_revision": "apple-m4-16gb-v1"},
        status="queued",
        stage="preflight",
    )
    assert job.processing_mode == "validated_strict_local"
    assert job.manifest_snapshot["pack_revision"] == "apple-m4-16gb-v1"


def test_page_and_evidence_payload_columns_are_encrypted_types() -> None:
    assert LocalAIPage.__table__.c.ocr_result.type.__class__.__name__ == "EncryptedJSON"
    assert ExtractionEvidence.__table__.c.excerpt.type.__class__.__name__ == "EncryptedText"
```

Extend `test_at_rest_encryption.py` to insert canary OCR Markdown, evidence text, and typed summary output, inspect raw database values, and assert the plaintext canary is absent.

- [ ] **Step 2: Run and verify missing models**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_models.py tests/test_at_rest_encryption.py -v
```

Expected: collection fails because `app.models.local_ai` does not exist.

- [ ] **Step 3: Add SQLAlchemy models**

```python
# backend/app/models/local_ai.py
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.encrypted_types import EncryptedJSON, EncryptedText


class LocalAIJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "local_ai_jobs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    upload_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("uploaded_files.id", ondelete="CASCADE")
    )
    summary_prompt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_summary_prompts.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    processing_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    progress: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    failure: Mapped[dict | None] = mapped_column(JSONB)
    audit_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LocalAIPage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "local_ai_pages"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("local_ai_jobs.id", ondelete="CASCADE"), nullable=False
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("uploaded_files.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_key: Mapped[str] = mapped_column(String(64), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ocr_result: Mapped[dict] = mapped_column(EncryptedJSON, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (UniqueConstraint("job_id", "page_number"),)


class ExtractionEvidence(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "extraction_evidence"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("uploaded_files.id", ondelete="CASCADE"), nullable=False
    )
    health_record_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("health_records.id", ondelete="SET NULL")
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    excerpt: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    field_paths: Mapped[list] = mapped_column(EncryptedJSON, nullable=False)
    source_metadata: Mapped[dict] = mapped_column(EncryptedJSON, nullable=False)
```

Add `processing_mode`, `processing_manifest`, and `processing_schema_version` to `UploadedFile`; add `processing_mode` to `UserLLMPreferences`; add `processing_mode`, `model_provenance`, and `typed_response: EncryptedJSON` to `AISummaryPrompt`. Import all new models in `backend/app/models/__init__.py`.

- [ ] **Step 4: Write and apply the migration**

The migration revision is exactly `f6a7b8c9d0e1` with `down_revision = "e5f6a7b8c9d0"`. Use `op.create_table` definitions matching the models and indexes on `(user_id, status)`, `upload_id`, `summary_prompt_id`, `local_ai_pages(job_id, page_number)`, and `extraction_evidence(health_record_id)`. The downgrade drops new tables before columns.

Run:

```bash
cd backend
uv run alembic upgrade head
uv run alembic current
```

Expected: current revision is `f6a7b8c9d0e1 (head)`.

- [ ] **Step 5: Run model/encryption tests and regressions**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_models.py tests/test_llm_settings_models.py \
  tests/test_at_rest_encryption.py -v
uv run ruff check app/models tests/test_local_ai_models.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/models backend/alembic/versions/f6a7b8c9d0e1_add_strict_local_jobs_and_evidence.py \
  backend/tests/test_local_ai_models.py backend/tests/test_llm_settings_models.py \
  backend/tests/test_at_rest_encryption.py
git commit -m "feat(local-ai): persist encrypted jobs checkpoints and evidence"
```

---

### Task 5: Build versioned worker IPC and the single-model manager

**Files:**

- Create: `backend/app/services/local_ai/protocol.py`
- Create: `backend/app/services/local_ai/model_manager.py`
- Create: `backend/app/services/local_ai/fake_worker.py`
- Create: `backend/tests/test_local_ai_protocol.py`
- Create: `backend/tests/test_local_ai_model_manager.py`
- Modify: `backend/app/main.py`

**Interfaces:**

- Protocol version `1`
- Commands: `health`, `ocr`, `extract`, `summarize`, `cancel`, `shutdown`
- Responses: `ready`, `progress`, `result`, `error`
- `LocalModelManager.start()`, `stop()`, `run(role, payload, on_progress)`, `cancel(job_id)`

- [ ] **Step 1: Write failing protocol and lifecycle tests**

```python
# backend/tests/test_local_ai_protocol.py
import pytest
from pydantic import ValidationError

from app.services.local_ai.protocol import WorkerRequest, WorkerResponse


def test_protocol_rejects_unknown_version() -> None:
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate({
            "version": 2,
            "request_id": "r1",
            "job_id": "j1",
            "command": "ocr",
            "payload": {},
        })


def test_error_response_cannot_contain_raw_detail() -> None:
    with pytest.raises(ValidationError):
        WorkerResponse.model_validate({
            "version": 1,
            "request_id": "r1",
            "kind": "error",
            "payload": {"code": "failed", "message": "safe", "raw": "PHI"},
        })
```

```python
# backend/tests/test_local_ai_model_manager.py
@pytest.mark.asyncio
async def test_manager_never_overlaps_role_processes(fake_worker_command) -> None:
    manager = LocalModelManager(fake_worker_command)
    await manager.start()
    first = asyncio.create_task(manager.run(ModelRole.OCR, {"delay_ms": 50}))
    second = asyncio.create_task(manager.run(ModelRole.EXTRACTION, {"delay_ms": 10}))
    await asyncio.gather(first, second)
    assert manager.metrics.max_live_processes == 1
    assert manager.metrics.roles_started == [ModelRole.OCR, ModelRole.EXTRACTION]
    await manager.stop()


@pytest.mark.asyncio
async def test_cancel_terminates_and_reaps_active_worker(fake_worker_command) -> None:
    manager = LocalModelManager(fake_worker_command)
    task = asyncio.create_task(
        manager.run(ModelRole.OCR, {"job_id": "j1", "block": True})
    )
    await manager.wait_until_running("j1")
    await manager.cancel("j1")
    with pytest.raises(LocalWorkerError, match="cancelled"):
        await task
    assert manager.active_pid is None
```

- [ ] **Step 2: Run and verify missing modules**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_protocol.py tests/test_local_ai_model_manager.py -v
```

Expected: collection fails for `protocol` and `model_manager`.

- [ ] **Step 3: Define strict JSON Lines messages**

```python
# backend/app/services/local_ai/protocol.py
from typing import Literal

from pydantic import BaseModel, ConfigDict


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    request_id: str
    job_id: str
    command: Literal["health", "ocr", "extract", "summarize", "cancel", "shutdown"]
    payload: dict


class WorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    request_id: str
    kind: Literal["ready", "progress", "result", "error"]
    payload: dict
```

The manager starts a new process for each `run`, sends one request, validates every stdout line, forwards only non-content progress, then sends shutdown and waits five seconds. On timeout/cancel/protocol error it sends `SIGTERM`, waits five seconds, sends `SIGKILL` if necessary, closes pipes, and reaps the PID before releasing the global `asyncio.Lock`.

Spawn with a minimal environment:

```python
WORKER_ENV = {
    "PATH": os.environ["PATH"],
    "HOME": str(worker_home),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
}
```

Do not pass provider keys, proxy variables, analytics variables, database URLs, upload paths, or a Hugging Face token. Apply `resource.setrlimit(resource.RLIMIT_CORE, (0, 0))` in the child on macOS.

Implement `fake_worker.py` as the same protocol with fixed synthetic OCR/extraction/summary fixtures, controllable delay/crash/malformed responses, and no model dependency. It is the CI full-pipeline worker.

- [ ] **Step 4: Wire application lifespan**

Create the singleton manager without starting it at import time. In `backend/app/main.py` startup, run scratch recovery first, then `await manager.start()` only when `local_ai_enabled`; in shutdown, `await manager.stop()` before closing the database. Do not warm-load a model.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_protocol.py tests/test_local_ai_model_manager.py -v
uv run ruff check app/services/local_ai/protocol.py \
  app/services/local_ai/model_manager.py app/services/local_ai/fake_worker.py \
  tests/test_local_ai_protocol.py tests/test_local_ai_model_manager.py
```

Expected: tests pass, including `max_live_processes == 1`, and Ruff passes.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/local_ai/protocol.py \
  backend/app/services/local_ai/model_manager.py \
  backend/app/services/local_ai/fake_worker.py backend/app/main.py \
  backend/tests/test_local_ai_protocol.py backend/tests/test_local_ai_model_manager.py
git commit -m "feat(local-ai): serialize isolated model workers"
```

---

### Task 6: Add secure scratch, page-at-a-time rasterization, and checkpoint keys

**Files:**

- Create: `backend/app/services/local_ai/scratch.py`
- Create: `backend/app/services/local_ai/rasterizer.py`
- Create: `backend/app/services/local_ai/checkpoints.py`
- Create: `backend/tests/test_local_ai_scratch.py`
- Create: `backend/tests/test_local_ai_rasterizer.py`
- Create: `backend/tests/test_local_ai_checkpoints.py`
- Modify: `backend/app/services/ingestion/coordinator.py`
- Modify: `backend/tests/test_structured_encryption.py`

**Interfaces:**

- `ScratchJob`
- `iter_rasterized_pages(encrypted_path, scratch, limits)`
- `ocr_checkpoint_key(...)`
- `extraction_checkpoint_key(...)`
- `summary_checkpoint_key(...)`

- [ ] **Step 1: Write failing raster, cleanup, and key tests**

```python
# backend/tests/test_local_ai_checkpoints.py
def test_ocr_key_changes_only_for_ocr_dependencies() -> None:
    base = ocr_checkpoint_key("upload-hash", 1, "raster-v1", "manifest-a")
    assert base == ocr_checkpoint_key("upload-hash", 1, "raster-v1", "manifest-a")
    assert base != ocr_checkpoint_key("upload-hash", 2, "raster-v1", "manifest-a")
    assert base != ocr_checkpoint_key("upload-hash", 1, "raster-v2", "manifest-a")


def test_summary_key_changes_when_validated_fact_hash_changes() -> None:
    assert summary_checkpoint_key("facts-a", "summary-v1", "manifest-a") != \
        summary_checkpoint_key("facts-b", "summary-v1", "manifest-a")
```

```python
# backend/tests/test_local_ai_scratch.py
def test_scratch_context_removes_plaintext_after_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with ScratchJob(tmp_path, "job-1") as scratch:
            page = scratch.create_file("page-0001.png", b"PHI-CANARY")
            assert stat.S_IMODE(page.stat().st_mode) == 0o600
            raise RuntimeError("boom")
    assert not (tmp_path / "job-1").exists()
```

```python
# backend/tests/test_local_ai_rasterizer.py
def test_pdf_rasterizer_yields_one_bounded_page_at_a_time(encrypted_pdf, tmp_path) -> None:
    with ScratchJob(tmp_path, "job-1") as scratch:
        pages = list(iter_rasterized_pages(encrypted_pdf, scratch, max_pixels=40_000_000))
    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.width * page.height <= 40_000_000 for page in pages)


def test_tiff_over_limit_fails_instead_of_silently_truncating(encrypted_26_page_tiff, tmp_path):
    with ScratchJob(tmp_path, "job-1") as scratch:
        with pytest.raises(LocalValidationError, match="page limit"):
            list(iter_rasterized_pages(encrypted_26_page_tiff, scratch, max_pages=25))
```

- [ ] **Step 2: Run and verify missing-module failures**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_scratch.py tests/test_local_ai_rasterizer.py \
  tests/test_local_ai_checkpoints.py -v
```

Expected: collection fails for the new modules.

- [ ] **Step 3: Implement bounded scratch and deterministic keys**

`ScratchJob` creates `LOCAL_AI_SCRATCH_DIR/<job-id>` with mode `0700`, refuses symlinks, creates files with `os.open(..., O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)`, and recursively unlinks only within the resolved scratch root. `sweep_stale_scratch` deletes directories whose job is not running and whose mtime exceeds the configured recovery threshold.

Checkpoint keys use canonical JSON:

```python
def _key(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"kind": kind, **payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Implement streaming rasterization**

Use `pypdfium2.PdfDocument` for PDF and Pillow's frame iterator for TIFF. Stream-decrypt the encrypted upload into a `0600` scratch source using `iter_decrypted_file_chunks`; never call the whole-file `decrypt_file`. Render one page to a `0600` PNG, yield its path/dimensions/hash, and delete that PNG after its OCR checkpoint is committed. Reject encrypted PDFs, malformed pages, page/pixel limits, and TIFF over-limit with user-safe local errors. RTF remains text-only and bypasses image OCR.

```python
@dataclass(frozen=True)
class RasterizedPage:
    page_number: int
    path: Path
    width: int
    height: int
    sha256: str


def iter_rasterized_pages(
    encrypted_path: Path,
    scratch: ScratchJob,
    *,
    max_pages: int = 500,
    max_pixels: int,
) -> Iterator[RasterizedPage]:
    source = scratch.decrypt_to_file(encrypted_path)
    if source.suffix.lower() == ".pdf":
        document = pdfium.PdfDocument(source)
        if len(document) > max_pages:
            raise LocalValidationError("Document exceeds the page limit")
        for index in range(len(document)):
            image = document[index].render(scale=2.0).to_pil()
            if image.width * image.height > max_pixels:
                raise LocalValidationError("Rasterized page exceeds the pixel limit")
            output = scratch.reserve_file(f"page-{index + 1:04d}.png")
            image.save(output, format="PNG")
            yield RasterizedPage(
                page_number=index + 1,
                path=output,
                width=image.width,
                height=image.height,
                sha256=file_sha256(output),
            )
        return
    with Image.open(source) as image:
        frames = ImageSequence.Iterator(image)
        for index, frame in enumerate(frames, start=1):
            if index > max_pages:
                raise LocalValidationError("Document exceeds the page limit")
            rendered = frame.convert("RGB")
            if rendered.width * rendered.height > max_pixels:
                raise LocalValidationError("Rasterized page exceeds the pixel limit")
            output = scratch.reserve_file(f"page-{index:04d}.png")
            rendered.save(output, format="PNG")
            yield RasterizedPage(
                page_number=index,
                path=output,
                width=rendered.width,
                height=rendered.height,
                sha256=file_sha256(output),
            )
```

- [ ] **Step 5: Encrypt mixed-ZIP child uploads**

Replace the plaintext `shutil.copy2` path in `backend/app/services/ingestion/coordinator.py` with `EncryptedFileWriter` chunked copying. Copy the parent `processing_mode` and `processing_manifest` into each `UploadedFile` child and preserve the existing temporary-tree `finally` cleanup.

Add a regression assertion to `test_structured_encryption.py` that a PDF extracted from a ZIP is ciphertext at rest and decrypts to the original bytes.

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_scratch.py tests/test_local_ai_rasterizer.py \
  tests/test_local_ai_checkpoints.py tests/test_structured_encryption.py -v
uv run ruff check app/services/local_ai/scratch.py \
  app/services/local_ai/rasterizer.py app/services/local_ai/checkpoints.py
```

Expected: all pass; no 26-page TIFF is silently shortened; no plaintext ZIP child remains.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/local_ai/scratch.py \
  backend/app/services/local_ai/rasterizer.py \
  backend/app/services/local_ai/checkpoints.py \
  backend/app/services/ingestion/coordinator.py \
  backend/tests/test_local_ai_scratch.py backend/tests/test_local_ai_rasterizer.py \
  backend/tests/test_local_ai_checkpoints.py backend/tests/test_structured_encryption.py
git commit -m "feat(local-ai): rasterize and checkpoint encrypted documents safely"
```

---

### Task 7: Define the NuExtract clinical schema and deterministic validators

**Files:**

- Create: `backend/app/services/local_ai/extraction_schema.py`
- Create: `backend/app/services/local_ai/extraction_validator.py`
- Create: `backend/app/services/local_ai/adapters.py`
- Create: `backend/tests/test_local_ai_extraction_validation.py`
- Create: `backend/tests/test_local_ai_entity_adapter.py`
- Modify: `backend/app/services/extraction/entity_to_fhir.py`
- Modify: `backend/tests/test_entity_to_fhir_richness.py`

**Interfaces:**

- Pydantic `ClinicalDocumentExtraction`
- `validate_clinical_extraction(raw, pages)`
- `to_extracted_entities(validated)`
- Evidence ids retained in FHIR `_extraction_metadata`

- [ ] **Step 1: Write failing schema/evidence tests**

```python
# backend/tests/test_local_ai_extraction_validation.py
def test_critical_fact_requires_verbatim_evidence_and_valid_page() -> None:
    raw = {
        "medications": [{
            "name": "Metformin",
            "dose_value": 500,
            "dose_unit": "mg",
            "verbatim": "Metformin 500 mg twice daily",
            "page_number": 3,
            "evidence_excerpt": "Metformin 500 mg twice daily",
        }]
    }
    with pytest.raises(LocalValidationError, match="page 3"):
        validate_clinical_extraction(raw, pages={1: "Metformin 500 mg twice daily"})


def test_validator_rejects_normalized_value_absent_from_verbatim() -> None:
    raw = medication_result(value=50, unit="mg", verbatim="Metformin 500 mg")
    with pytest.raises(LocalValidationError, match="numeric token"):
        validate_clinical_extraction(raw, pages={1: "Metformin 500 mg"})


def test_missing_fact_is_unresolved_not_inferred() -> None:
    validated = validate_clinical_extraction(
        {"medications": [], "unresolved_fields": ["medications.dose"]},
        pages={1: "Dose not stated"},
    )
    assert validated.unresolved_fields == ["medications.dose"]
```

- [ ] **Step 2: Run and verify missing modules**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_extraction_validation.py \
  tests/test_local_ai_entity_adapter.py -v
```

Expected: collection fails for the extraction modules.

- [ ] **Step 3: Define a strict, versioned clinical extraction template**

Use NuExtract's extractive types for safety-sensitive values:

```python
NUEXTRACT_TEMPLATE_V1 = {
    "patient": {
        "name": {"type": "verbatim-string"},
        "date_of_birth": {"type": "verbatim-string"},
    },
    "medications": [{
        "name": {"type": "verbatim-string"},
        "dose_value": {"type": "verbatim-string"},
        "dose_unit": {"type": "verbatim-string"},
        "route": {"type": "verbatim-string"},
        "frequency": {"type": "verbatim-string"},
        "status": {"type": "string", "enum": ["active", "stopped", "historical", "unknown"]},
        "page_number": {"type": "number"},
        "evidence_excerpt": {"type": "verbatim-string"},
    }],
    "labs": [{
        "name": {"type": "verbatim-string"},
        "value": {"type": "verbatim-string"},
        "unit": {"type": "verbatim-string"},
        "reference_range": {"type": "verbatim-string"},
        "date": {"type": "verbatim-string"},
        "page_number": {"type": "number"},
        "evidence_excerpt": {"type": "verbatim-string"},
    }],
    "conditions": [{
        "name": {"type": "verbatim-string"},
        "assertion": {"type": "string", "enum": ["present", "negated", "family_history", "uncertain"]},
        "date": {"type": "verbatim-string"},
        "page_number": {"type": "number"},
        "evidence_excerpt": {"type": "verbatim-string"},
    }],
    "procedures": [],
    "allergies": [],
    "encounters": [],
    "immunizations": [],
    "vital_signs": [],
    "diagnostic_reports": [],
    "care_plans": [],
    "unresolved_fields": [],
}
```

Represent all categories with explicit Pydantic models; the abbreviated empty arrays above are expanded in code with the same common evidence fields. Use `extra="forbid"`, bounded string/list lengths, and enums for assertions/status. Do not accept raw `dict` past this module.

- [ ] **Step 4: Implement deterministic validation and entity adaptation**

Validation order:

1. Parse strict JSON after stripping only a single fenced-code wrapper.
2. Validate schema and output-size limits.
3. Verify page numbers exist.
4. Verify each bounded excerpt occurs on that page after whitespace normalization.
5. Verify critical numeric/date/unit/name tokens occur in the verbatim value and page.
6. Apply existing negation/mentioned-not-performed/family-history guards.
7. Normalize only into separate fields; retain original values.
8. Create stable evidence ids from upload id, page, normalized excerpt offsets, and field paths.
9. Reject an invalid fact; fail the stage if any critical category is partially malformed after one same-model syntax retry.

`to_extracted_entities` creates the existing `ExtractedEntity` objects and stores `_evidence_ids`, `_source_page`, `_verbatim`, `_normalization_method`, and `_normalization_version` inside attributes. Update `entity_to_fhir.py` to copy these keys into `_extraction_metadata`.

- [ ] **Step 5: Run focused tests and existing validator/FHIR regressions**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_extraction_validation.py \
  tests/test_local_ai_entity_adapter.py tests/test_entity_validator.py \
  tests/test_entity_to_fhir_richness.py -v
uv run ruff check app/services/local_ai/extraction_schema.py \
  app/services/local_ai/extraction_validator.py app/services/local_ai/adapters.py
```

Expected: all pass; existing FHIR shapes remain compatible and now retain evidence ids.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/local_ai/extraction_schema.py \
  backend/app/services/local_ai/extraction_validator.py \
  backend/app/services/local_ai/adapters.py \
  backend/app/services/extraction/entity_to_fhir.py \
  backend/tests/test_local_ai_extraction_validation.py \
  backend/tests/test_local_ai_entity_adapter.py \
  backend/tests/test_entity_to_fhir_richness.py
git commit -m "feat(local-ai): validate grounded clinical extraction"
```

---

### Task 8: Build the strict-local ingestion pipeline with fake backends

**Files:**

- Create: `backend/app/services/local_ai/pipeline.py`
- Create: `backend/app/services/local_ai/checkpoint_store.py`
- Create: `backend/tests/test_strict_local_pipeline.py`
- Create: `backend/tests/test_strict_local_egress.py`
- Create: `backend/tests/test_local_ai_cleanup.py`
- Modify: `backend/app/api/upload.py`
- Modify: `backend/app/schemas/upload.py`
- Modify: `backend/tests/test_upload_progress_cancel.py`
- Modify: `backend/tests/test_unstructured_failure_recovery.py`

**Interfaces:**

- `run_strict_local_ingestion(job_id, upload_id, file_path)`
- `StrictLocalPipeline`
- Structured local failure/progress payloads

- [ ] **Step 1: Write the full fake-worker fail-closed tests**

```python
# backend/tests/test_strict_local_egress.py
@pytest.mark.asyncio
async def test_strict_pipeline_constructs_no_provider_and_attempts_no_egress(
    strict_local_upload,
    fake_worker_manager,
    deny_external_sockets,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-forbidden")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-forbidden")

    def fail_provider(*args, **kwargs):
        raise AssertionError("cloud provider was constructed")

    monkeypatch.setattr("app.services.ai.llm.registry.get_provider", fail_provider)
    await run_strict_local_ingestion(
        strict_local_upload.job_id,
        strict_local_upload.upload.id,
        Path(strict_local_upload.upload.storage_path),
    )
    assert deny_external_sockets.attempts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "crash", "malformed", "timeout"])
async def test_local_worker_failure_never_falls_back(failure, strict_local_upload, fake_worker):
    fake_worker.fail_as(failure)
    with pytest.raises(LocalAIError):
        await run_strict_local_ingestion(
            strict_local_upload.job_id,
            strict_local_upload.upload.id,
            Path(strict_local_upload.upload.storage_path),
        )
    assert fake_worker.cloud_calls == 0
    assert strict_local_upload.upload.processing_mode == "validated_strict_local"
```

Add a log canary test containing patient name, DOB, OCR text, dosage, prompt, and summary; assert none appears in `caplog.text`, progress, notices, or generic audit details.

- [ ] **Step 2: Run and verify the missing pipeline failure**

Run:

```bash
cd backend
uv run pytest tests/test_strict_local_pipeline.py tests/test_strict_local_egress.py \
  tests/test_local_ai_cleanup.py -v
```

Expected: collection fails because `pipeline.py` does not exist.

- [ ] **Step 3: Implement the pipeline as a separate branch**

```python
# backend/app/services/local_ai/pipeline.py
class StrictLocalPipeline:
    def __init__(
        self,
        manager: LocalModelManager,
        checkpoints: CheckpointStore,
        manifest: LocalAIManifest,
    ) -> None:
        self.manager = manager
        self.checkpoints = checkpoints
        self.manifest = manifest

    async def run_ingestion(
        self,
        job: LocalAIJob,
        upload: UploadedFile,
        encrypted_path: Path,
    ) -> None:
        assert ProcessingMode(job.processing_mode) is ProcessingMode.VALIDATED_STRICT_LOCAL
        await self._preflight(job)
        with ScratchJob(Path(settings.local_ai_scratch_dir), str(job.id)) as scratch:
            pages = await self._ocr_pages(job, upload, encrypted_path, scratch)
            extraction = await self._extract(job, upload, pages, scratch)
            await self._validate_map_and_persist(job, upload, extraction)
```

The module may import local-AI modules, existing entity validation, and FHIR mapping. It must not import `services.ai.llm`, `text_extractor`, `entity_extractor`, `section_parser`, or LangExtract.

Progress contains only:

```python
{
    "stage": "ocr",
    "page_index": 2,
    "page_total": 9,
    "model_role": "ocr",
    "repository": "sahilchachra/ovisocr2-int4-mlx",
    "revision": "immutable-40-character-revision",
}
```

Failure contains `stage`, sanitized `code`, sanitized `message`, `model_role`, repository/revision, `retryable`, `checkpoint_preserved`, and `cloud_fallback_attempted: False`.

- [ ] **Step 4: Branch in `_process_unstructured` before provider resolution**

Load `UploadedFile.processing_mode` first:

```python
if upload.processing_mode == ProcessingMode.VALIDATED_STRICT_LOCAL:
    await run_strict_local_ingestion_for_upload(db, upload)
    return

config = await load_llm_config(db, user_id)
# Existing custom/cloud-assisted path continues below unchanged.
```

Update cancellation to call `model_manager.cancel(job_id)` after setting the DB flag. Preserve completed page checkpoints and mark the local job `cancelled`. Startup recovery requeues strict-local jobs from the last valid checkpoint rather than resetting their data.

- [ ] **Step 5: Extend upload response schemas**

Use typed Pydantic models for:

```python
class LocalProcessingFailure(BaseModel):
    stage: str
    code: str
    message: str
    model_role: str | None
    repository: str | None
    revision: str | None
    retryable: bool
    checkpoint_preserved: bool
    cloud_fallback_attempted: Literal[False] = False


class LocalRunInfo(BaseModel):
    privacy_mode: str
    models: list[dict[str, str]]
```

Add them to status/history/pending responses without returning raw checkpoints or evidence.

- [ ] **Step 6: Run the strict-local and existing cloud-assisted regressions**

Run:

```bash
cd backend
uv run pytest tests/test_strict_local_pipeline.py tests/test_strict_local_egress.py \
  tests/test_local_ai_cleanup.py tests/test_upload_progress_cancel.py \
  tests/test_unstructured_failure_recovery.py -v
uv run pytest tests/test_text_extractor_local.py tests/test_ocr_egress.py \
  tests/test_extraction_engine.py -v
```

Expected: strict-local tests show zero external attempts and no fallback; existing explicit cloud-assisted fallback tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/local_ai/pipeline.py \
  backend/app/services/local_ai/checkpoint_store.py backend/app/api/upload.py \
  backend/app/schemas/upload.py backend/tests/test_strict_local_pipeline.py \
  backend/tests/test_strict_local_egress.py backend/tests/test_local_ai_cleanup.py \
  backend/tests/test_upload_progress_cancel.py \
  backend/tests/test_unstructured_failure_recovery.py
git commit -m "feat(local-ai): run ingestion through a fail-closed local pipeline"
```

---

### Task 9: Snapshot processing mode at upload creation and make reprocessing revision-aware

**Files:**

- Modify: `backend/app/schemas/llm_settings.py`
- Modify: `backend/app/api/llm_settings.py`
- Modify: `backend/app/api/upload.py`
- Modify: `backend/app/services/ingestion/coordinator.py`
- Modify: `backend/app/services/ingestion/reextraction.py`
- Modify: `backend/tests/test_llm_settings_api.py`
- Create: `backend/tests/test_processing_mode_snapshot.py`
- Modify: `backend/tests/test_reextraction.py`

**Interfaces:**

- `RoutingUpdate.processing_mode`
- Multipart `processing_mode` on `/upload/unstructured` and `/upload/unstructured-batch`
- `resolve_new_job_snapshot(user_id, requested_mode)`
- Revision-aware duplicate lookup

- [ ] **Step 1: Write failing preference/snapshot tests**

```python
# backend/tests/test_processing_mode_snapshot.py
@pytest.mark.asyncio
async def test_upload_locks_requested_mode_and_active_manifest(client, auth_headers, pack_ready):
    response = await client.post(
        "/api/v1/upload/unstructured",
        headers=auth_headers,
        files={"file": ("record.pdf", synthetic_pdf(), "application/pdf")},
        data={"processing_mode": "validated_strict_local"},
    )
    assert response.status_code == 202
    upload = await load_upload(response.json()["upload_id"])
    assert upload.processing_mode == "validated_strict_local"
    assert upload.processing_manifest["pack_revision"] == "apple-m4-16gb-v1"


@pytest.mark.asyncio
async def test_later_preference_change_does_not_change_queued_upload(
    db_session,
    user,
    pack_ready,
):
    upload = await create_upload(mode="validated_strict_local")
    await set_user_mode("cloud_assisted")
    claimed = await claim_upload(upload.id)
    assert claimed.processing_mode == "validated_strict_local"
```

- [ ] **Step 2: Run and verify schema/API failures**

Run:

```bash
cd backend
uv run pytest tests/test_processing_mode_snapshot.py \
  tests/test_llm_settings_api.py tests/test_reextraction.py -v
```

Expected: tests fail because `processing_mode` is not accepted or persisted.

- [ ] **Step 3: Add validated settings and upload fields**

```python
# backend/app/schemas/llm_settings.py
class RoutingUpdate(BaseModel):
    default: str | None = None
    summary: str | None = None
    section: str | None = None
    dedup: str | None = None
    extraction: str | None = None
    vision: str | None = None
    extraction_engine: str | None = None
    processing_mode: ProcessingMode | None = None
```

Return `processing_mode` from `GET /settings/llm`. Add it to `_ROUTING_FIELDS` and validate custom-local endpoints before persisting that mode.

Use FastAPI `Form` on both unstructured endpoints:

```python
processing_mode: ProcessingMode | None = Form(default=None)
```

`resolve_new_job_snapshot` applies:

1. Explicit request mode.
2. Stored user preference.
3. `cloud_assisted` for existing/backward-compatible behavior.
4. For validated strict local, require the active manifest to be fully installed and validated; otherwise return HTTP 409.
5. Persist canonical manifest JSON, schema version `clinical-v1`, and prompt version before enqueue.

- [ ] **Step 4: Make ZIP children inherit and duplicate checks revision-aware**

Pass a `ProcessingSnapshot` through `ingest_file` to mixed-ZIP child creation. Change `find_prior_extracted_upload` to accept `processing_mode`, `manifest_sha256`, and `schema_version`; only deduplicate when all match. Add an explicit reprocess action that creates a new pending upload/job referencing the same encrypted source when the manifest/schema changes.

- [ ] **Step 5: Run settings, upload, ZIP, and dedup tests**

Run:

```bash
cd backend
uv run pytest tests/test_processing_mode_snapshot.py tests/test_llm_settings_api.py \
  tests/test_reextraction.py tests/test_structured_encryption.py \
  tests/test_unstructured_upload.py -v
```

Expected: all pass; a queued upload retains its original mode/revision; the same bytes can be reprocessed under a new validated manifest.

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/llm_settings.py backend/app/api/llm_settings.py \
  backend/app/api/upload.py backend/app/services/ingestion/coordinator.py \
  backend/app/services/ingestion/reextraction.py \
  backend/tests/test_llm_settings_api.py backend/tests/test_processing_mode_snapshot.py \
  backend/tests/test_reextraction.py
git commit -m "feat(local-ai): lock processing mode and model revision per job"
```

---

### Task 10: Implement typed, evidence-grounded local summaries

**Files:**

- Create: `backend/app/services/local_ai/grounded_summary.py`
- Create: `backend/tests/test_grounded_local_summary.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Modify: `backend/app/api/summary.py`
- Modify: `backend/app/schemas/summary.py`
- Modify: `backend/tests/test_summarization.py`
- Modify: `backend/tests/test_summary.py`

**Interfaces:**

- `GroundedSummaryDocument`
- `build_grounded_summary_input(...)`
- `validate_and_render_summary(...)`
- `GenerateSummaryRequest.processing_mode`
- Summary provenance and deterministic disclaimer

- [ ] **Step 1: Write failing unsupported-claim tests**

```python
# backend/tests/test_grounded_local_summary.py
def test_summary_rejects_unknown_fact_and_evidence_ids() -> None:
    raw = {
        "sections": [{
            "heading": "Medications",
            "claims": [{
                "text": "Metformin 500 mg is active.",
                "fact_ids": ["fact-unknown"],
                "evidence_ids": ["evidence-unknown"],
            }],
        }],
        "uncertainties": [],
    }
    with pytest.raises(LocalValidationError, match="unknown fact"):
        validate_and_render_summary(raw, facts={"fact-1": {}}, evidence={"evidence-1": {}})


def test_every_factual_claim_requires_support() -> None:
    raw = {
        "sections": [{
            "heading": "Overview",
            "claims": [{"text": "Diabetes is present.", "fact_ids": [], "evidence_ids": []}],
        }],
        "uncertainties": [],
    }
    with pytest.raises(LocalValidationError, match="support"):
        validate_and_render_summary(raw, facts={}, evidence={})


def test_server_owned_disclaimer_is_always_rendered(valid_summary, facts, evidence) -> None:
    rendered = validate_and_render_summary(valid_summary, facts=facts, evidence=evidence)
    assert "not medical advice" in rendered.markdown.lower()
```

- [ ] **Step 2: Run and verify missing-module failure**

Run:

```bash
cd backend
uv run pytest tests/test_grounded_local_summary.py -v
```

Expected: collection fails because `grounded_summary.py` does not exist.

- [ ] **Step 3: Define typed summary input/output**

```python
# backend/app/services/local_ai/grounded_summary.py
class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    fact_ids: list[str] = Field(min_length=1, max_length=32)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)


class GroundedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1, max_length=120)
    claims: list[GroundedClaim] = Field(max_length=100)


class GroundedSummaryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[GroundedSection] = Field(max_length=20)
    uncertainties: list[str] = Field(max_length=100)
```

Create stable `fact_id` values from health-record id plus content hash. Input to Qwen includes only validated facts, bounded evidence excerpts, requested scope, and server safety rules. It does not include raw uploads or rejected/unresolved values except a list of uncertainty labels.

Validation rejects unknown ids, empty support, evidence not linked to each fact, model-created facts, excessive output, diagnoses/advice language that violates policy, and invalid types. Rendering is deterministic Markdown with citation labels and a server-owned medical disclaimer appended after validation.

- [ ] **Step 4: Add the strict-local summary branch**

In `GenerateSummaryRequest`, add:

```python
processing_mode: ProcessingMode = ProcessingMode.CLOUD_ASSISTED
```

In `generate_summary_endpoint`, create and commit `AISummaryPrompt` plus `LocalAIJob` before invoking the worker so cancellation/recovery is possible. For `validated_strict_local`, call `generate_grounded_local_summary`; do not call the existing `generate_summary` provider path. Reject arbitrary `provider`, `model`, and replacement system prompts in this mode. For `prompt_only`, reject `/generate` with HTTP 400 and direct the client to `/build-prompt`.

Store the validated typed document in encrypted `typed_response`, rendered Markdown in encrypted `response_text`, and model identity in `model_provenance`.

- [ ] **Step 5: Run local and existing cloud summary tests**

Run:

```bash
cd backend
uv run pytest tests/test_grounded_local_summary.py tests/test_summary.py \
  tests/test_summarization.py tests/test_summarizer_provider.py -v
uv run ruff check app/services/local_ai/grounded_summary.py \
  app/api/summary.py app/schemas/summary.py
```

Expected: local unsupported claims are rejected; prompt-only never invokes generation; existing cloud-assisted summaries still pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/local_ai/grounded_summary.py \
  backend/app/services/ai/summarizer.py backend/app/api/summary.py \
  backend/app/schemas/summary.py backend/tests/test_grounded_local_summary.py \
  backend/tests/test_summarization.py backend/tests/test_summary.py
git commit -m "feat(local-ai): validate evidence-grounded local summaries"
```

---

### Task 11: Add model-pack operations and record-evidence APIs

**Files:**

- Create: `backend/app/schemas/local_ai.py`
- Create: `backend/app/api/local_ai.py`
- Modify: `backend/app/api/router.py`
- Modify: `backend/app/api/records.py`
- Create: `backend/tests/test_local_ai_api.py`
- Create: `backend/tests/test_record_evidence_api.py`

**Endpoints:**

- `GET /api/v1/local-ai/status`
- `POST /api/v1/local-ai/install`
- `GET /api/v1/local-ai/operations/{operation_id}`
- `POST /api/v1/local-ai/operations/{operation_id}/resume`
- `POST /api/v1/local-ai/operations/{operation_id}/retry`
- `POST /api/v1/local-ai/verify`
- `POST /api/v1/local-ai/update`
- `POST /api/v1/local-ai/rollback`
- `DELETE /api/v1/local-ai/models/{role}`
- `DELETE /api/v1/local-ai`
- `GET /api/v1/records/{record_id}/evidence`

- [ ] **Step 1: Write failing authorization/state tests**

```python
# backend/tests/test_local_ai_api.py
@pytest.mark.asyncio
async def test_install_returns_document_free_operation(client, auth_headers, catalog_ready):
    response = await client.post("/api/v1/local-ai/install", headers=auth_headers)
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"operation_id", "state"}
    assert body["state"] == "queued"


@pytest.mark.asyncio
async def test_remove_refuses_active_job(client, auth_headers, active_local_job):
    response = await client.delete("/api/v1/local-ai", headers=auth_headers)
    assert response.status_code == 409
```

```python
# backend/tests/test_record_evidence_api.py
@pytest.mark.asyncio
async def test_user_cannot_read_another_users_evidence(client, other_user_record, auth_headers):
    response = await client.get(
        f"/api/v1/records/{other_user_record.id}/evidence",
        headers=auth_headers,
    )
    assert response.status_code == 404
```

- [ ] **Step 2: Run and verify 404 failures**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_api.py tests/test_record_evidence_api.py -v
```

Expected: endpoints return 404 because routers do not exist.

- [ ] **Step 3: Define strict response models**

```python
class LocalModelArtifactResponse(BaseModel):
    role: ModelRole
    repository: str
    revision: str
    quantization: str
    runtime: str
    license: str
    download_bytes: int
    expected_memory_bytes: int
    installed: bool
    validated: bool


class LocalPackStatusResponse(BaseModel):
    platform: Literal["apple_silicon", "unsupported"]
    compatible: bool
    state: Literal[
        "not_installed", "downloading", "verifying", "ready",
        "update_available", "failed",
    ]
    active_revision: str | None
    available_revision: str
    models: list[LocalModelArtifactResponse]
    operation: LocalPackOperationResponse | None
```

Operation status includes action/state/current role/byte counts/sanitized message/retryable. No upload/job/document fields are permitted.

Evidence response includes page/section/bounded excerpt/offsets/field paths, unresolved/rejected field names, mode, and model identities. It is user-scoped through `HealthRecord.user_id`; do not log the response body.

- [ ] **Step 4: Implement operations and auditing**

The install/update operations call the downloader and artifact store in a background task, then run `verify` before activation. `verify` performs hash/runtime/architecture smoke tests and the synthetic fixture subset. Only a verified pack can become `ready`. Rollback points to the prior already-verified revision. Removal refuses an active job and never deletes uploads/checkpoints/records.

Generic audit events contain action, pack revision, role, operation id, success/failure category, and timestamps only.

- [ ] **Step 5: Run API tests**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_api.py tests/test_record_evidence_api.py \
  tests/test_audit_middleware.py -v
uv run ruff check app/api/local_ai.py app/schemas/local_ai.py app/api/records.py
```

Expected: all pass; cross-user evidence is 404; audit details contain no excerpts.

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/local_ai.py backend/app/api/local_ai.py \
  backend/app/api/router.py backend/app/api/records.py \
  backend/tests/test_local_ai_api.py backend/tests/test_record_evidence_api.py
git commit -m "feat(local-ai): expose pack operations and extraction evidence"
```

---

### Task 12: Create the isolated native Apple MLX worker and validate artifact loading

**Files:**

- Create: `workers/local_ai/apple_mlx/pyproject.toml`
- Create: `workers/local_ai/apple_mlx/uv.lock`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/__init__.py`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/__main__.py`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/common.py`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/ovisocr2.py`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/nuextract3.py`
- Create: `workers/local_ai/apple_mlx/src/local_ai_mlx_worker/qwen_summary.py`
- Create: `workers/local_ai/apple_mlx/tests/test_protocol.py`
- Create: `workers/local_ai/apple_mlx/tests/test_offline_loading.py`
- Create: `scripts/setup-local-ai-macos.sh`
- Modify: `scripts/setup-local.sh`
- Modify: `justfile`

**Runtime:**

- Python `>=3.11,<3.13`
- `mlx-vlm==0.5.0`
- No server framework
- No downloader dependency

- [ ] **Step 1: Write worker protocol and offline-load tests**

```python
# workers/local_ai/apple_mlx/tests/test_protocol.py
def test_worker_stdout_is_protocol_only(worker_process) -> None:
    response = request(worker_process, command="health", payload={})
    assert response == {
        "version": 1,
        "request_id": "test-1",
        "kind": "result",
        "payload": {"status": "ready", "runtime": "mlx-vlm-0.5.0"},
    }
```

```python
# workers/local_ai/apple_mlx/tests/test_offline_loading.py
@pytest.mark.local_model
@pytest.mark.parametrize("role", ["ocr", "extraction", "summary"])
def test_exact_manifest_artifact_loads_without_remote_code(role, locked_manifest, model_dir):
    with deny_all_network():
        loaded = load_role(role, locked_manifest, model_dir, trust_remote_code=False)
    assert loaded.repository_files_used.isdisjoint(
        {path for path in loaded.repository_files_used if path.endswith(".py")}
    )
```

- [ ] **Step 2: Create the isolated project and observe the initial failure**

Use:

```toml
[project]
name = "local-ai-mlx-worker"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "mlx-vlm==0.5.0",
]

[project.scripts]
local-ai-mlx-worker = "local_ai_mlx_worker.__main__:main"
```

Run:

```bash
cd workers/local_ai/apple_mlx
uv lock
uv sync
uv run pytest tests/test_protocol.py -v
```

Expected: fails because the worker entry point is not implemented.

- [ ] **Step 3: Implement protocol-only stdout and role loaders**

`__main__.py` parses one JSON Line at a time, writes logs to stderr with non-content metadata only, dispatches by role, and returns strict protocol messages. `common.py` resolves each model path beneath the active manifest root, rehashes files before load, rejects config `auto_map`, and calls MLX loaders with local paths only.

Role behavior:

- OvisOCR2: image + fixed OCR instruction, `temperature=0`, sampling disabled, bounded 8192 output tokens, one page.
- NuExtract3: non-thinking chat template, schema from the request, OCR Markdown first and only explicitly selected image paths, bounded JSON output, one syntax retry at `temperature=0`.
- Qwen3.5-9B: validated facts/evidence only, non-thinking JSON output, `temperature=0`, bounded context selected by preflight.

Every role returns content only inside the `result` protocol payload. Exceptions map to fixed codes; exception strings, prompts, outputs, and image paths are not logged.

- [ ] **Step 4: Prove each exact candidate is loadable without repository code**

Run on the Apple M4:

```bash
cd workers/local_ai/apple_mlx
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run pytest tests/test_offline_loading.py -m local_model -v -rs
```

Expected: all three roles pass using the locked candidate files and no network. If any exact artifact cannot load with repository code disabled, do not enable `trust_remote_code`; keep the pack unavailable, port the required architecture/preprocessor into reviewed worker source with focused tests, and rerun this same gate before continuing.

- [ ] **Step 5: Add explicit install commands without changing normal setup**

`scripts/setup-local-ai-macos.sh` verifies Darwin/arm64, at least 16 GB physical memory, available disk, creates the isolated worker environment, and does not download models unless passed `--download-pack`.

Fix `scripts/setup-local.sh` to install Python 3.11.8 instead of 3.12 so it matches `backend/pyproject.toml`.

Add:

```make
local-ai-runtime-install:
	./scripts/setup-local-ai-macos.sh

local-ai-pack-download:
	cd backend && uv run python scripts/local_ai_pack.py install

local-ai-pack-verify:
	cd backend && uv run python scripts/local_ai_pack.py verify

local-ai-pack-remove:
	cd backend && uv run python scripts/local_ai_pack.py remove
```

- [ ] **Step 6: Run worker tests and backend protocol integration**

Run:

```bash
cd workers/local_ai/apple_mlx
uv run pytest tests/test_protocol.py -v
uv run ruff check src tests
cd ../../../backend
uv run pytest tests/test_local_ai_model_manager.py tests/test_strict_local_pipeline.py -v
```

Expected: all pass; stdout contains protocol JSON only.

- [ ] **Step 7: Commit**

```bash
git add workers/local_ai/apple_mlx scripts/setup-local-ai-macos.sh \
  scripts/setup-local.sh justfile
git commit -m "feat(local-ai): add isolated Apple MLX worker"
```

---

### Task 13: Add frontend mode, pack, provenance, and evidence types

**Files:**

- Create: `frontend/src/types/local-ai.ts`
- Modify: `frontend/src/types/api.ts`
- Modify: `frontend/src/types/upload.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/extraction-progress.ts`
- Modify: `frontend/src/lib/extraction-progress.unit.spec.ts`
- Modify: `frontend/src/stores/useExtractionStore.ts`
- Modify: `frontend/src/stores/useExtractionStore.unit.spec.ts`

**Interfaces:**

- `ProcessingMode`, `LocalModelRole`, `PackState`
- `LocalPackStatus`, `LocalPackOperation`
- `LocalProcessingFailure`, `LocalRunInfo`
- `EvidenceReference`, `RecordExtractionProvenance`
- Pack/evidence API helpers

- [ ] **Step 1: Write failing TypeScript unit tests**

Extend the progress unit test:

```typescript
it("formats local OCR page and immutable model role", () => {
  expect(
    formatStageDetail("ocr", {
      page_index: 2,
      page_total: 9,
      model_role: "ocr",
      repository: "sahilchachra/ovisocr2-int4-mlx",
      revision: "0123456789abcdef0123456789abcdef01234567",
    }),
  ).toContain("page 2 of 9");
});
```

Extend the store test to prove a progress update containing `local_run_info` merges into an active batch without replacing other tracked files.

- [ ] **Step 2: Run and verify type/test failures**

Run:

```bash
cd frontend
npx playwright test --config playwright.unit.config.ts \
  extraction-progress useExtractionStore
npx tsc --noEmit
```

Expected: tests/typecheck fail because the local fields/types are absent.

- [ ] **Step 3: Add shared exact types**

```typescript
// frontend/src/types/local-ai.ts
export type ProcessingMode =
  | "validated_strict_local"
  | "custom_local"
  | "cloud_assisted"
  | "prompt_only";

export type LocalModelRole = "ocr" | "extraction" | "summary";
export type PackState =
  | "not_installed"
  | "downloading"
  | "verifying"
  | "ready"
  | "update_available"
  | "failed";

export interface LocalModelArtifact {
  role: LocalModelRole;
  repository: string;
  revision: string;
  quantization: string;
  runtime: string;
  license: string;
  download_bytes: number;
  expected_memory_bytes: number;
  installed: boolean;
  validated: boolean;
}

export interface LocalPackOperation {
  id: string;
  action: "install" | "update" | "rollback" | "remove";
  state: "queued" | "running" | "paused" | "failed" | "completed";
  current_role: LocalModelRole | null;
  bytes_done: number;
  bytes_total: number;
  message: string | null;
  retryable: boolean;
}

export interface LocalPackStatus {
  platform: "apple_silicon" | "linux_cpu" | "linux_cuda" | "linux_rocm" | "unsupported";
  compatible: boolean;
  state: PackState;
  active_revision: string | null;
  available_revision: string;
  models: LocalModelArtifact[];
  operation: LocalPackOperation | null;
}
```

Add the failure/run/evidence types specified by the API. Extend `ProgressDetail` with page/model fields while preserving `section_index` and `section_total`.

- [ ] **Step 4: Add typed client methods**

```typescript
getLocalPackStatus(): Promise<LocalPackStatus>
installLocalPack(): Promise<{ operation_id: string; state: string }>
updateLocalPack(): Promise<{ operation_id: string; state: string }>
resumeLocalPackOperation(id: string): Promise<LocalPackOperation>
retryLocalPackOperation(id: string): Promise<LocalPackOperation>
rollbackLocalPack(): Promise<{ operation_id: string; state: string }>
removeLocalModel(role: LocalModelRole): Promise<void>
removeLocalPack(): Promise<void>
getRecordEvidence(recordId: string): Promise<RecordExtractionProvenance>
```

Add `processing_mode` and provenance to summary request/response types. Add the already-used optional `provider` field to `GenerateSummaryRequest`.

- [ ] **Step 5: Run unit/type checks**

Run:

```bash
cd frontend
npx playwright test --config playwright.unit.config.ts \
  extraction-progress useExtractionStore
npx tsc --noEmit
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/local-ai.ts frontend/src/types/api.ts \
  frontend/src/types/upload.ts frontend/src/lib/api.ts \
  frontend/src/lib/extraction-progress.ts \
  frontend/src/lib/extraction-progress.unit.spec.ts \
  frontend/src/stores/useExtractionStore.ts \
  frontend/src/stores/useExtractionStore.unit.spec.ts
git commit -m "feat(local-ai): add frontend processing and provenance contracts"
```

---

### Task 14: Build the validated-pack settings UI inside Admin → System

**Files:**

- Create: `frontend/src/components/admin/AiSettingsCard.tsx`
- Create: `frontend/src/components/admin/ValidatedLocalPackCard.tsx`
- Create: `frontend/src/hooks/useLocalPackOperation.ts`
- Modify: `frontend/src/app/(dashboard)/admin/page.tsx`
- Create: `frontend/e2e/local-model-pack-settings.spec.ts`
- Modify: `frontend/e2e/llm-settings.spec.ts`
- Modify: `frontend/e2e/admin-consolidation.spec.ts`

**UI behavior:**

- Exactly four Admin tabs remain.
- Validated pack appears above custom/cloud providers inside System.
- Install/resume/retry/update/rollback/remove states are explicit.
- Ollama/LM Studio are labelled `Custom local (unverified)`.

- [ ] **Step 1: Write failing Playwright tests**

```typescript
// frontend/e2e/local-model-pack-settings.spec.ts
import { expect, test } from "./fixtures/console-gate";

test("installs and verifies the platform pack without adding an Admin tab", async ({ page }) => {
  await mockLocalPackApi(page, {
    platform: "apple_silicon",
    compatible: true,
    state: "not_installed",
    active_revision: null,
    available_revision: "apple-m4-16gb-v1",
    models: MODEL_FIXTURES,
    operation: null,
  });
  await page.goto("/admin?tab=system");
  await expect(page.getByRole("tab")).toHaveCount(4);
  await expect(page.getByRole("heading", { name: "Validated local pack" })).toBeVisible();
  await page.getByRole("button", { name: "Install local pack" }).click();
  await expect(page.getByText("Verifying downloaded models")).toBeVisible();
});

test("labels Ollama and LM Studio as unverified custom local", async ({ page }) => {
  await page.goto("/admin?tab=system");
  await expect(page.getByText("Custom local (unverified)")).toBeVisible();
});
```

- [ ] **Step 2: Run and verify UI failures**

Run:

```bash
cd frontend
npx playwright test local-model-pack-settings llm-settings \
  admin-consolidation --workers=1 --reporter=list
```

Expected: tests fail because the pack card and copy do not exist.

- [ ] **Step 3: Extract and compose the settings card**

Move the existing `LlmProvidersCard` implementation from `admin/page.tsx` to `AiSettingsCard.tsx` without behavioral changes. Render:

```tsx
<section aria-labelledby="ai-settings-heading" className="space-y-6">
  <ValidatedLocalPackCard />
  <CustomAndCloudProviderSettings />
</section>
```

`ValidatedLocalPackCard` displays platform, 16 GB requirement, each role/repository/revision/license/download/expected memory/validation state, operation progress, and exact available actions. The hook polls only while a pack operation is non-terminal and stops on unmount.

Replace the old `local | hybrid | gemini` extraction selector with a processing-mode selector. Disable `validated_strict_local` until `state === "ready"`. Show mode-specific privacy text. Keep provider routing subordinate to custom/cloud modes.

- [ ] **Step 4: Run focused e2e and lint**

Run:

```bash
cd frontend
npx playwright test local-model-pack-settings llm-settings \
  admin-consolidation --workers=1 --reporter=list
npm run lint
```

Expected: all focused tests pass, Admin still has four tabs, and lint passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/admin frontend/src/hooks/useLocalPackOperation.ts \
  'frontend/src/app/(dashboard)/admin/page.tsx' \
  frontend/e2e/local-model-pack-settings.spec.ts \
  frontend/e2e/llm-settings.spec.ts frontend/e2e/admin-consolidation.spec.ts
git commit -m "feat(local-ai): manage the validated pack in Admin System"
```

---

### Task 15: Add strict-local upload progress, failure details, and evidence UI

**Files:**

- Create: `frontend/src/components/retro/LocalProcessingDetails.tsx`
- Create: `frontend/src/components/retro/ExtractionEvidencePanel.tsx`
- Modify: `frontend/src/app/(dashboard)/upload/page.tsx`
- Modify: `frontend/src/components/retro/GlobalExtractionStatusBar.tsx`
- Modify: `frontend/src/components/retro/RecordDetailSheet.tsx`
- Modify: `frontend/src/app/(dashboard)/records/[id]/page.tsx`
- Modify: `frontend/src/app/(dashboard)/admin/page.tsx`
- Create: `frontend/e2e/strict-local-upload-progress.spec.ts`
- Create: `frontend/e2e/record-extraction-evidence.spec.ts`
- Modify: `frontend/e2e/upload-row-status-polling.spec.ts`
- Modify: `frontend/e2e/extraction-terminal-state.spec.ts`

- [ ] **Step 1: Write failing upload/evidence tests**

```typescript
// frontend/e2e/strict-local-upload-progress.spec.ts
import { expect, test } from "./fixtures/console-gate";

test("stamps strict-local mode and shows fail-closed model progress", async ({ page }) => {
  const request = await mockStrictLocalUpload(page);
  await page.goto("/upload");
  await selectFixture(page, "record.pdf");
  await page.getByRole("button", { name: "Upload" }).click();
  expect((await request).postData()).toContain("validated_strict_local");
  await expect(page.getByText("OCR page 2 of 9")).toBeVisible();
  await expect(page.getByText("OvisOCR2")).toBeVisible();
  await expect(page.getByText("Cloud fallback was not attempted")).toBeVisible();
});
```

```typescript
// frontend/e2e/record-extraction-evidence.spec.ts
import { expect, test } from "./fixtures/console-gate";

test("loads evidence only from an AI-extracted record detail", async ({ page }) => {
  await mockAiRecordWithEvidence(page);
  await page.goto("/records/record-1");
  await expect(page.getByRole("heading", { name: "Extraction evidence" })).toBeVisible();
  await expect(page.getByText("Page 3")).toBeVisible();
  await expect(page.getByText("Metformin 500 mg twice daily")).toBeVisible();
  await expect(page.getByText(/Qwen3.5-9B/)).toHaveCount(0);
});
```

- [ ] **Step 2: Run and verify failures**

Run:

```bash
cd frontend
npx playwright test strict-local-upload-progress record-extraction-evidence \
  upload-row-status-polling extraction-terminal-state \
  --workers=1 --reporter=list
```

Expected: tests fail because mode, local details, and evidence panel are absent.

- [ ] **Step 3: Stamp the selected mode on every unstructured upload**

Append `processing_mode` to the `FormData` used by single and batch unstructured uploads before sending. Capture the selected mode once at `handleUploadAll` start so a later settings refresh cannot alter in-flight requests.

Do not add another upload polling loop. Keep `GlobalExtractionStatusBar` as the sole two-second extraction poller and extend its mapping into the Zustand store with the compatible local fields.

- [ ] **Step 4: Render bounded local status and failure details**

`LocalProcessingDetails` displays mode, current role/repository/short revision, failure stage, retryability, checkpoint preservation, and the exact `Cloud fallback was not attempted` assurance only when the API's literal false field is present. It never renders evidence excerpts.

Reuse shared `ExtractionFileStatus` in Admin Extractions instead of its local narrower duplicate.

- [ ] **Step 5: Add evidence to both record-detail surfaces**

`ExtractionEvidencePanel` fetches only when `ai_extracted === true`. Render page/section, bounded excerpt, supported field paths, unresolved/rejected field names, and extraction model revision. Keep status colors operationally neutral and do not render summary-model provenance in ingestion evidence.

- [ ] **Step 6: Run focused frontend tests**

Run:

```bash
cd frontend
npx playwright test strict-local-upload-progress upload-extraction-ux \
  upload-row-status-polling extraction-terminal-state upload-ocr-notices \
  --workers=1 --reporter=list
npx playwright test record-extraction-evidence record-ai-metadata \
  record-detail-sheet record-detail-page --workers=1 --reporter=list
npm run lint
```

Expected: all pass; existing OCR notices remain; only one extraction polling loop runs.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/retro/LocalProcessingDetails.tsx \
  frontend/src/components/retro/ExtractionEvidencePanel.tsx \
  'frontend/src/app/(dashboard)/upload/page.tsx' \
  frontend/src/components/retro/GlobalExtractionStatusBar.tsx \
  frontend/src/components/retro/RecordDetailSheet.tsx \
  'frontend/src/app/(dashboard)/records/[id]/page.tsx' \
  'frontend/src/app/(dashboard)/admin/page.tsx' \
  frontend/e2e/strict-local-upload-progress.spec.ts \
  frontend/e2e/record-extraction-evidence.spec.ts \
  frontend/e2e/upload-row-status-polling.spec.ts \
  frontend/e2e/extraction-terminal-state.spec.ts
git commit -m "feat(local-ai): show local progress failures and evidence"
```

---

### Task 16: Replace provider-first summaries with explicit execution modes

**Files:**

- Create: `frontend/src/components/retro/AiExecutionModeControl.tsx`
- Modify: `frontend/src/app/(dashboard)/summaries/page.tsx`
- Create: `frontend/e2e/summary-processing-modes.spec.ts`
- Modify: `frontend/e2e/multi-provider-summary.spec.ts`

- [ ] **Step 1: Write failing mode-routing tests**

```typescript
// frontend/e2e/summary-processing-modes.spec.ts
import { expect, test } from "./fixtures/console-gate";

test("validated local summary hides providers and sends the local mode", async ({ page }) => {
  const generated = await mockSummaryApis(page);
  await page.goto("/summaries");
  await page.getByLabel("AI execution mode").selectOption("validated_strict_local");
  await expect(page.getByLabel("Provider")).toHaveCount(0);
  await page.getByRole("button", { name: "Generate summary" }).click();
  expect((await generated).postDataJSON()).toMatchObject({
    processing_mode: "validated_strict_local",
  });
});

test("prompt-only builds copyable payload and never calls generate", async ({ page }) => {
  const calls = await mockPromptOnly(page);
  await page.goto("/summaries");
  await page.getByLabel("AI execution mode").selectOption("prompt_only");
  await page.getByRole("button", { name: "Build prompt" }).click();
  await expect(page.getByRole("button", { name: "Copy prompt payload" })).toBeVisible();
  expect(calls.generate).toBe(0);
  expect(calls.buildPrompt).toBe(1);
});
```

- [ ] **Step 2: Run and verify failures**

Run:

```bash
cd frontend
npx playwright test summary-processing-modes multi-provider-summary \
  --workers=1 --reporter=list
```

Expected: tests fail because the page is provider-first and prompt-only is absent.

- [ ] **Step 3: Implement mode-specific controls and calls**

`AiExecutionModeControl` behavior:

- Validated strict local: show exact Qwen repository/revision, no provider selector, call `/summary/generate` with the mode.
- Custom local: show only enabled loopback Ollama/LM Studio providers, label unverified, include provider choice.
- Cloud assisted: show configured cloud providers plus an explicit egress notice.
- Prompt only: call `/summary/build-prompt`, render/copy `copyable_payload`, never call `/summary/generate`.

Rewrite blanket privacy copy so each mode states its actual boundary. Preserve the no-medical-advice disclaimer for generated output and prompt payloads.

- [ ] **Step 4: Run summary e2e and build**

Run:

```bash
cd frontend
npx playwright test summary-processing-modes multi-provider-summary summaries \
  --workers=1 --reporter=list
npm run lint
npm run build
```

Expected: all pass; the production build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/retro/AiExecutionModeControl.tsx \
  'frontend/src/app/(dashboard)/summaries/page.tsx' \
  frontend/e2e/summary-processing-modes.spec.ts \
  frontend/e2e/multi-provider-summary.spec.ts
git commit -m "feat(local-ai): add explicit summary execution modes"
```

---

### Task 17: Add privacy, fidelity, and 16 GB M4 release gates

**Files:**

- Create: `backend/tests/fidelity/local_ai/fixtures/synthetic/`
- Create: `backend/tests/test_local_ai_fidelity.py`
- Create: `backend/tests/test_local_ai_resources.py`
- Create: `backend/tests/test_local_ai_log_privacy.py`
- Create: `backend/scripts/benchmark_local_ai.py`
- Modify: `backend/pyproject.toml`
- Create: `.github/workflows/backend-ci.yml`
- Create: `.github/workflows/local-ai-contract-ci.yml`

**Hard gates:**

- Zero non-loopback processing egress.
- Zero cloud provider construction/fallback.
- 100% accepted-output schema validity.
- Every accepted fact and summary claim has valid evidence.
- Zero unsupported summary facts.
- Zero PHI canaries in logs/telemetry/crash output.
- Scratch cleanup on success/cancel/crash.
- Exact manifest hashes.
- Critical numeric-token OCR exactness at least 99%.
- Critical extraction precision at least 98% and recall at least 95%.
- No memory-pressure termination or sustained swap thrash on the 16 GB M4.

- [ ] **Step 1: Add failing synthetic scoring and privacy tests**

Use committed synthetic PDFs/TIFFs for dosages, lab decimals, units, dates, negation, tables, skew, repeated headers, poor illumination, and malformed output. Tests compute exact metrics rather than subjective assertions:

```python
assert metrics.critical_numeric_exact >= 0.99
assert metrics.critical_precision >= 0.98
assert metrics.critical_recall >= 0.95
assert metrics.unsupported_summary_facts == 0
assert metrics.accepted_facts_without_evidence == 0
```

Register `local_model`, `fidelity`, and `hardware` markers. Private fixtures are loaded only from `REAL_MEDICAL_FIXTURES_DIR`.

- [ ] **Step 2: Run fake-worker CI gates**

Run:

```bash
cd backend
uv run pytest tests/test_strict_local_egress.py tests/test_local_ai_log_privacy.py \
  tests/test_local_ai_cleanup.py tests/test_strict_local_pipeline.py -v
```

Expected: passes without models/network.

- [ ] **Step 3: Implement the benchmark report**

`benchmark_local_ai.py` records machine model/RAM/macOS, manifest hash, runtime version, three cold runs per role, peak RSS, MLX active/peak memory, system memory pressure, swap delta, throughput, and post-process reclamation. It writes content-free JSON to `artifacts/local-ai-benchmark.json`.

Acceptance assertions:

```python
assert report.physical_memory_gib >= 16
assert report.processes.max_live_models == 1
assert report.system.memory_pressure_termination is False
assert report.system.sustained_swap_thrashing is False
assert report.reclamation.final_active_memory_ratio <= 0.20
```

- [ ] **Step 4: Run real model and private-fixture gates on the 16 GB M4**

Run:

```bash
cd backend
LOCAL_AI_ENABLED=true uv run pytest tests/test_local_ai_fidelity.py \
  -m "local_model and fidelity" -v -rs
LOCAL_AI_ENABLED=true uv run pytest tests/test_local_ai_resources.py \
  -m "local_model and hardware" -v -rs
uv run python scripts/benchmark_local_ai.py \
  --manifest app/model_manifests/apple-m4-16gb-v1.lock.json \
  --runs 3 --output artifacts/local-ai-benchmark.json
```

Expected: all hard gates pass. The benchmark records no model overlap, no memory-pressure termination, no sustained swap thrash, and reclamation within the configured threshold. Do not mark the manifest `validated` if any gate fails.

- [ ] **Step 5: Promote the tested lock to the shipped validated manifest**

Copy the exact candidate lock produced in Task 2 to `backend/app/model_manifests/apple-m4-16gb-v1.lock.json`, add the fixture-suite version and benchmark-report SHA-256, and rerun `local-ai-pack-verify`. This is the only step that changes UI status from preview/unavailable to validated.

- [ ] **Step 6: Add CI workflows**

`backend-ci.yml` runs ordinary backend tests and Ruff on Python 3.11. `local-ai-contract-ci.yml` runs manifest/artifact/policy/manager/fake-pipeline/egress/log/cleanup tests without model downloads. Pin GitHub Action commits and the `uv` installer version; do not use mutable `latest`.

- [ ] **Step 7: Commit gates and the validated manifest**

```bash
git add backend/tests/fidelity/local_ai backend/tests/test_local_ai_fidelity.py \
  backend/tests/test_local_ai_resources.py backend/tests/test_local_ai_log_privacy.py \
  backend/scripts/benchmark_local_ai.py backend/pyproject.toml \
  backend/app/model_manifests/apple-m4-16gb-v1.lock.json \
  .github/workflows/backend-ci.yml .github/workflows/local-ai-contract-ci.yml
git commit -m "test(local-ai): gate privacy fidelity and M4 resources"
```

---

### Task 18: Document the validated boundary and run the complete verification

**Files:**

- Create: `docs/operations-strict-local-ai.md`
- Create: `docs/third-party-local-model-pack-notices.md`
- Modify: `docs/backend-handoff.md`
- Modify: `docs/operations-backup-restore.md`
- Modify: `README.md`
- Modify: `backend/.env.example`
- Modify: `AGENTS.md` symlink target

- [ ] **Step 1: Update operational documentation**

Document:

- 16 GB minimum and recommended Apple baseline.
- Install/download/verify/update/rollback/remove commands.
- Model/cache/scratch paths and permissions.
- Model download network boundary versus processing no-egress boundary.
- Exact validated model roles and revisions.
- Native macOS support. State that Docker Desktop/host companion is not validated until separately gated.
- Custom-local unverified versus cloud-assisted versus prompt-only.
- Failure, cancellation, retry, checkpoint, and recovery behavior.
- Model artifacts are redownloadable/excluded from mandatory backups; encrypted PHI checkpoints/evidence remain in database backups.
- What the release gates prove and do not prove.

Generate notices from the locked manifest so base and quantization repository licenses, attribution, revision, and hashes stay synchronized.

- [ ] **Step 2: Reconcile agent guidance with raw local OCR**

Update the private AGENTS overlay through the existing symlink so the rule becomes:

```markdown
- Raw PHI may be passed only to a validated, contained local worker for the
  strict-local OCR/extraction job that owns it. It must never reach cloud,
  custom external, logs, telemetry, or model-download code.
- De-identification remains mandatory before every cloud-provider call.
- Strict-local code must branch before constructing any cloud provider and
  must fail closed without fallback.
```

Preserve all other medical-safety and verification rules.

- [ ] **Step 3: Humanize user-facing prose**

Invoke the `humanizer` skill on README/UI-facing operational prose. Preserve exact model names, mode names, privacy qualifications, medical disclaimer, commands, and release thresholds.

- [ ] **Step 4: Run complete backend verification**

Run:

```bash
cd backend
uv run ruff check app tests scripts
uv run pytest -m "not slow and not local_model and not hardware" -x -q
uv run alembic current
```

Expected: Ruff passes, non-hardware suite passes, and migration head is `f6a7b8c9d0e1`.

- [ ] **Step 5: Run complete frontend verification**

Run:

```bash
cd frontend
npm run lint
npm run build
npx playwright test --workers=1 --reporter=list
```

Expected: lint/build/all Playwright tests pass with the console gate.

- [ ] **Step 6: Run release-only M4 gates once more**

Run:

```bash
cd backend
LOCAL_AI_ENABLED=true uv run pytest -m "local_model or hardware" -v -rs
uv run python scripts/benchmark_local_ai.py \
  --manifest app/model_manifests/apple-m4-16gb-v1.lock.json \
  --runs 3 --output artifacts/local-ai-benchmark.json
```

Expected: all privacy/fidelity/resource gates pass against the exact shipped manifest.

- [ ] **Step 7: Inspect final scope and commit**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only strict-local feature/docs/test files are changed; no downloaded weights, private fixtures, plaintext scratch, keys, or unrelated user files are staged.

```bash
git add README.md AGENTS.md docs/operations-strict-local-ai.md \
  docs/third-party-local-model-pack-notices.md docs/backend-handoff.md \
  docs/operations-backup-restore.md backend/.env.example
git commit -m "docs(local-ai): document the validated strict-local boundary"
```

---

## Completion Criteria

- The validated Apple pack installs only from immutable, hash-verified, license-recorded artifacts and loads with repository code disabled.
- A strict-local PDF/TIFF job branches before all cloud-capable configuration and completes OCR → NuExtract → deterministic validation/FHIR mapping with zero external attempts.
- A strict-local summary loads Qwen3.5-9B only after ingestion, accepts only validated fact/evidence ids, and rejects unsupported claims.
- Missing, incompatible, crashed, timed-out, or malformed local workers expose a local error with checkpoint state and never fall back.
- Exactly one model process is resident at a time, and the 16 GB M4 passes the measured memory/reclamation gates.
- Operational logs, analytics, crash output, progress, notices, and generic audit details contain no PHI canaries.
- Scratch cleanup passes on success, cancellation, exception, SIGKILL recovery, and startup sweep.
- Admin remains four tabs; the System pane distinguishes validated local, custom local unverified, cloud assisted, and prompt only.
- Uploads and summaries snapshot mode/manifest; settings changes cannot mutate queued/running jobs.
- Every accepted clinical fact and summary claim exposes user-scoped evidence and exact model provenance.
- Full backend/frontend verification and the physical-M4 release gates pass against the exact shipped manifest.
