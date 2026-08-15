# Strict-local AI operations

## Release status

The strict-local path is implemented as an optional, fail-closed processing
mode. This Track D state prepares the Apple M4 16 GB v2 candidate through
[`catalog-v2.json`](../backend/app/model_manifests/catalog-v2.json) and
[`apple-m4-16gb-v2.lock.json`](../backend/app/model_manifests/apple-m4-16gb-v2.lock.json).
It does not contain or claim a verified v2 release. The
`apple-m4-16gb-v2.release.json` file is generated only after the separately
authorized model-backed benchmark, fidelity, promotion, and post-promotion
verification gates pass. It is not present or verified in this Track D state.

The v1 catalog, lock, and release evidence remain immutable diagnostic history.
They are not defaults for new strict-local jobs and must not be relabelled as
v2 evidence.

`LOCAL_AI_ENABLED=false` remains the default so model installation and local
processing are opt-in. After an operator enables the feature, the UI reports
the pack as ready only when the installed files match the shipped lock and the
runtime-validation receipt matches the same manifest.

## Supported machine

The candidate profile is native Apple Silicon macOS with 16 GB of unified
memory. Sixteen GB is both the minimum and the recommended baseline for this
profile. The setup check also requires:

- `uv`;
- 2 GiB of free disk for the isolated runtime; and
- 25 GiB of free disk before downloading the 9.02 GiB model pack.

The larger disk check preserves room for encrypted uploads, OCR scratch, the
database, and macOS swap. Native MLX is the supported path. Docker Desktop
cannot use Metal, and a Docker-to-host companion has not been validated.

Linux CPU, CUDA, and ROCm workers are a follow-on. Their shared contracts are
being kept platform-neutral, but none of those profiles should be advertised
as validated until each exact runtime and model pack passes its own gates.

## The four processing modes

| Mode | Where inference runs | Privacy boundary |
| --- | --- | --- |
| Validated strict local | Self-hosted backend plus a network-denied native model worker using the locked pack | Raw document content stays on the machine. The path branches before cloud configuration and never falls back. |
| Custom local | User-managed loopback Ollama or LM Studio endpoint | Available for summaries, not unstructured ingestion. Routing stays on loopback, but Strand does not verify the model, server, or output quality. |
| Cloud assisted | Configured external provider | PDF and TIFF OCR sends the original document or pages to the selected vision provider before text is available to scrub. Downstream extraction and summary calls receive best-effort de-identified content. |
| Prompt only | No model call by Strand | Strand builds a de-identified payload for the user to copy elsewhere. The destination is outside Strand's privacy boundary. |

In validated strict-local mode, the self-hosted backend receives and decrypts
the upload, then locally rasterizes PDF/TIFF pages or decodes RTF text. Those
steps run in the network-capable web process, but their strict-local code path
does not construct or call a model provider. Model inference runs in a separate
worker with an OS-enforced network deny. Raw PHI must not enter model-download
code, provider routing, logs, telemetry, audit details, or progress responses.
This is an application-enforced no-cloud route with an extra OS boundary around
inference, not a machine-wide data-loss-prevention boundary around the backend.

Cloud-assisted scanned-document ingestion crosses a different boundary. The
selected vision provider receives the unredacted PDF or TIFF content before
OCR because Strand has no text to scrub yet. Returned OCR text is scrubbed
before downstream external extraction calls. If a selected cloud provider
refuses or fails, Strand does not resend the document to a second cloud
provider. A loopback vision provider that cannot read the document may fall
back once to Gemini, which then receives the original document.

## Model roles

The v2 candidate lock pins these immutable repository revisions:

| Role | Locked model | Revision | Quantization |
| --- | --- | --- | --- |
| OCR ingestion | `sahilchachra/ovisocr2-int4-mlx` | `1e9cea98871c19b2349a5d2df36fb6c4c38a1237` | int4 |
| Clinical extraction | `numind/NuExtract3-mlx-4bits` | `29c38269f94054282bf9ea97a20dfc6bb8bbefea` | 4-bit |
| Final summary | `mlx-community/Qwen3.5-9B-MLX-4bit` | `938d8919941c6e7efd3c7150eff7fe9d12afa631` | 4-bit |

OvisOCR2 handles page OCR. NuExtract3 receives the OCR Markdown and the
clinical schema, validates evidence, and maps accepted facts to FHIR. Qwen
does not ingest documents. It receives only the already validated
fact-and-evidence projection used for the final summary.

The lock lists every allowed file, byte size, SHA-256 digest, license record,
runtime version, and fixture-suite version. After promotion, release evidence
binds that lock to the exact benchmark and fidelity report bytes. It also
records the fidelity corpus identity and promotion thresholds.

### Worker runtime identity

Schema-v2 manifests bind the model pack to a portable worker identity using
the `local-ai-worker-bundle.v1` scheme. The digest covers only these inputs:

- the scheme name;
- the fixed console entry point
  `local-ai-mlx-worker=local_ai_mlx_worker.__main__:main`;
- `pyproject.toml`;
- `uv.lock`; and
- the effective `local_ai_mlx_worker` Python source tree, including every
  regular `.py` file that can be imported from that package.

The resolver also validates the worker command, launcher, virtual-environment
layout, CPython compatibility, import precedence, and startup customization
surface. Those topology checks fail closed but are not hashed, so absolute
installation paths and supported Python patch releases do not change the
portable digest. The digest does not attest the operating-system owner or root
of trust. File ownership, host integrity, process isolation, and administrator
control remain outside this boundary.

## Install and manage the pack

Install the isolated runtime without downloading models:

```bash
just local-ai-runtime-install
```

The pre-promotion command binds the v2 lock to the retained artifact tree and
runs the offline synthetic runtime fixture against those bytes. It does not
download, relabel, activate, or require release evidence:

```bash
just local-ai-candidate-verify
```

Its deterministic contract is covered by tests, but the command itself was not
run in Track D. It must not be substituted with the normal lifecycle check.

Run the three-cold-run, content-free memory benchmark only as part of the
separately authorized release gates:

```bash
just local-ai-benchmark
```

Run the fidelity gate against the same v2-bound retained artifact tree with
private fixtures removed from the environment:

```bash
cd backend
env -u REAL_MEDICAL_FIXTURES_DIR uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-fidelity.json
```

The command renders the committed synthetic PDF and TIFF fixtures, runs
OvisOCR2, NuExtract3, and Qwen3.5-9B in that order, and writes an owner-only
JSON report containing aggregate metrics only. It exits nonzero if any release
threshold fails. The runner checks the committed corpus SHA-256 before loading
the models. The checked-in release report must contain no private run or
private metrics. The normal test suite uses a fake worker and does not load the
models.

To run the same fidelity gate through pytest:

```bash
cd backend
LOCAL_AI_ENABLED=true uv run pytest \
  tests/test_local_ai_fidelity.py \
  -m "local_model and fidelity" -v -rs
```

After the benchmark and synthetic fidelity gates pass, generate the v2 release
evidence:

```bash
just local-ai-release-promote
```

Promotion fails unless both reports match the exact manifest and meet their
current hard gates. The promoted fidelity report must cover all six committed
synthetic documents and no private documents. Readiness revalidates both report
hashes, the fidelity corpus identity, document count, and recorded thresholds.
Older benchmark-only release evidence is not accepted. This Track D state did
not run these separately authorized gates or generate the v2 release file.

Promotion does not change the retained v1 activation metadata. If v1 is still
active, normal v2 installation fails closed. Do not remove or rewrite that
state as part of candidate verification. A separately reviewed and authorized
operator procedure must retire the v1 activation before the normal v2 install
can run. Track D did not define or run that mutation.

After that prerequisite is complete and promotion has written
`apple-m4-16gb-v2.release.json`, the normal install may download, verify, and
atomically activate the configured pack:

```bash
just local-ai-pack-download
```

`local-ai-pack-download` is post-promotion because normal installation requires
the v2 release file. It is not a candidate-preparation command.

Re-run the normal offline runtime and fixture verification after installation:

```bash
just local-ai-pack-verify
```

`local-ai-pack-verify` is the post-promotion command contract and requires
release evidence. It fails before the v2 release file exists.

### v1-to-v2 deployment preflight

Before applying the v2 database guard or changing the release default, an
operator must prove that no validated strict-local job is queued or processing.
This is an operational command contract, not permission to query a production
database during development:

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
        raise SystemExit(
            "refusing v2 deployment: active strict-local jobs exist"
        )


asyncio.run(main())
PY
cd ..
```

If the count is nonzero, stop. Let those jobs finish or fail them under the v1
behavior before deployment. Do not rewrite, relabel, or promote their captured
manifests. This preflight was not run in Track D and remains an operational
deployment gate.

## Run browser tests with local-only enforcement

Create a dedicated PostgreSQL database once, then run the complete Playwright
directory through the local-only profile:

```bash
createdb medtimeline_e2e_local
cd frontend
E2E_LOCAL_ONLY=1 \
E2E_DATABASE_URL=postgresql+asyncpg://localhost:5432/medtimeline_e2e_local \
  npx playwright test --workers=1
```

The profile accepts only a `postgresql+asyncpg` URL whose host is loopback and
whose database name contains a `test` or `e2e` segment. It rejects the same URL
as any `DATABASE_URL` inherited from the launching shell. Migrations and the
backend run inside an in-process socket guard that blocks DNS and connections
outside loopback. On macOS, the Next.js process runs under an OS profile that
permits loopback traffic and denies other outbound sockets. Next telemetry is
disabled, and the app uses system font stacks instead of fetching Google fonts.
Every Chromium context uses a closed proxy with loopback bypass, and service
workers are disabled. Cloud credentials are cleared and Hugging Face offline
flags remain set.

These controls cover database traffic, the backend, browser requests, and the
already sandboxed model worker. The local-only profile fails closed before
starting Next.js when its OS network profile is unavailable. The server binds
to `127.0.0.1` and does not proxy raw uploads; the browser sends them to the
backend. The real upload, encryption, job, polling, summary, grounded
extraction, and persistence paths still run. The E2E worker returns
deterministic validator-compatible output so the browser suite does not
repeatedly load the model pack. Use the fidelity gate above for real-model
quality checks.

Remove model artifacts:

```bash
just local-ai-pack-remove
```

Removal does not delete uploads, records, summaries, encrypted checkpoints, or
evidence. Pack update and rollback are available in Admin > System. Their API
routes are `POST /api/v1/local-ai/update` and
`POST /api/v1/local-ai/rollback`. Both are asynchronous lifecycle operations;
poll `GET /api/v1/local-ai/operations/{operation_id}` until a terminal state.

## Configuration and paths

The defaults assume the backend process starts from `backend/`:

```dotenv
LOCAL_AI_ENABLED=false
LOCAL_AI_MODEL_DIR=./data/local-ai/models
LOCAL_AI_SCRATCH_DIR=./data/local-ai/scratch
LOCAL_AI_MANIFEST_PATH=./app/model_manifests/apple-m4-16gb-v2.lock.json
LOCAL_AI_RELEASE_EVIDENCE_PATH=./app/model_manifests/apple-m4-16gb-v2.release.json
LOCAL_AI_BENCHMARK_PATH=./artifacts/local-ai-benchmark.json
LOCAL_AI_FIDELITY_PATH=./artifacts/local-ai-fidelity.json
LOCAL_AI_WORKER_COMMAND=../workers/local_ai/apple_mlx/.venv/bin/local-ai-mlx-worker
LOCAL_AI_WORKER_TIMEOUT_SECONDS=1800
LOCAL_AI_WORKER_HARD_TIMEOUT_SECONDS=7200
```

Keep the model directory separate from uploads and scratch. Model files contain
no medical data. The application creates the model-store control files and
scratch directories with owner-only permissions. Per-job scratch is plaintext
only while the owning job runs, uses owner-only directories and files, and is
removed after success, cancellation, failure, and startup recovery.

`LOCAL_AI_WORKER_TIMEOUT_SECONDS` is the idle deadline. A validated stage,
completed-page counter, or internal worker-activity heartbeat resets it.
Activity-only heartbeats refresh the durable job lease without changing
user-facing progress.
`LOCAL_AI_WORKER_HARD_TIMEOUT_SECONDS` caps the total request duration. The
30-minute idle timeout accounts for slower NuExtract decoding on the 16 GB
profile.

Do not place `LOCAL_AI_MODEL_DIR` or `LOCAL_AI_SCRATCH_DIR` on a
world-readable shared volume. FileVault remains recommended because plaintext
must exist in memory and briefly in owner-only scratch during inference.

## Network boundary

Model installation and document processing have different boundaries.

During installation, the document-free downloader may connect to Hugging Face
to fetch only the exact files in the lock. It receives no uploads, prompts,
records, evidence, database connection, or clinical scratch. Files are written
to staging, checked for path and type safety, hashed, runtime-tested offline,
fsynced, and only then activated.

During a validated strict-local job:

- the backend selects the captured processing mode and manifest snapshot;
- no cloud-capable provider is constructed;
- the backend decrypts and prepares the document with local parsing libraries;
- one native worker process and one model role are resident at a time;
- macOS launches the worker through a fixed OS sandbox profile that denies all
  network access;
- an owner-only file lock is held across backend processes and inherited by
  the worker until it exits;
- a separate parent-death watchdog terminates the worker process group if the
  backend exits unexpectedly;
- the worker receives bounded local paths and serializable request data;
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and telemetry-disable settings are
  applied;
- no HTTP inference port is opened; and
- a missing, incompatible, crashed, timed-out, or malformed worker produces a
  visible local failure without fallback.

The fast contract suite starts a real subprocess inside the same sandbox and
confirms that a worker-side socket receives an operating-system permission
error. It also starts competing backend processes and confirms that their
workers cannot overlap. A separate crash test leaves the first worker blocked,
kills its backend, and confirms that the watchdog terminates the orphan before
the next backend acquires the lease. The historical v1 release passed offline
role loading with its exact locked artifacts on the target Mac. That result is
not v2 release evidence.

The current macOS boundary uses `/usr/bin/sandbox-exec`, which Apple has
deprecated. Strict-local startup fails closed if that system tool is missing or
not trusted; it never falls back to an unrestricted worker. A signed helper
with App Sandbox network entitlements omitted is the long-term packaging path.
Linux workers require a separately verified network namespace or container
boundary and remain unavailable until that profile ships.

## Progress, cancellation, and recovery

Uploads and summaries capture their mode, model manifest, schema version, and
prompt version before processing starts. A later settings change cannot alter
an existing job.

OCR progress is page-based. Durable encrypted checkpoints allow a retry to
reuse completed work when the captured versions still match. Cancellation sets
the job flag and terminates the active local worker. The API reports the failed
stage, whether retry is safe, and whether checkpoints were preserved. It never
includes OCR text, evidence excerpts, prompts, or model output in operational
logs or progress fields.

Model-pack installation is separate from clinical processing. Install,
verify, update, and rollback operations are persisted without document
content. Interrupted lifecycle work is reconciled on startup. A validated
active pack remains usable after a failed maintenance operation, but new
strict-local jobs are blocked while maintenance is active.

Schema-v1 strict-local snapshots cannot run under the v2 worker contract. A
queued or processing legacy ingestion job makes one non-retryable transition
to `failed` with the bounded code `runtime_identity_required`; its paired
upload is terminalized in the same transaction. A legacy summary job makes the
same job-only transition. The stored v1 manifest remains unchanged.

For schema-v2 work, a missing or changed worker bundle is rejected before
strict-local admission and checked again immediately before every worker
spawn. Responses and logs use the same content-free error boundary; they do
not include paths, source names, manifest bytes, or digests. Operators must
repair the supported worker runtime, revalidate the exact candidate or
promoted pack as appropriate, and submit a new job. Do not retry or rewrite a
job whose captured runtime identity no longer matches.

## Backup and restore

Back up the Postgres database and encrypted upload directory as described in
[Backup, restore, and data retention](operations-backup-restore.md).
Strict-local jobs, encrypted page checkpoints, evidence, typed summaries, and
their immutable provenance live in the database and are part of that backup.

The model directory is optional in backups. Its contents are immutable and
redownloadable from the lock, and it contains no PHI. Excluding it reduces
backup size. After a restore, reinstall and verify the exact pack before
enabling strict-local jobs.

Never back up scratch. Stop processing, confirm no job is active, and let the
startup sweeper remove abandoned job directories.

## Validation evidence and policy

The historical v1 M4 16 GB profile passed the committed six-document synthetic
fidelity suite. All six rate metrics were `1.0`: output-schema validity,
critical numeric exactness, critical precision, critical recall, summary fact
recall, and summary typed-field recall. Accepted facts without evidence,
forbidden extraction facts, and unsupported summary facts were all zero.

Its content-free benchmark ran each model role cold three times on a 16 GB
Apple M4 machine. Peak MLX allocations were approximately:

| Role | Peak MLX allocation |
| --- | ---: |
| OCR | 0.86 GB |
| Clinical extraction | 4.73 GB |
| Final summary | 7.10 GB |

That benchmark recorded one live model at a time, no memory-pressure
termination, and no sustained swap thrashing. Its final MLX active-memory
ratio was `3.7e-9`.

The v2 candidate can be called validated only after it passes all of these
checks against its exact release lock:

- every file hash, size, type, license record, and revision matches;
- all three roles load with repository code disabled and no network;
- accepted output is schema-valid and every accepted fact or summary claim has
  valid evidence;
- unsupported summary facts are zero;
- critical numeric OCR exactness is at least 99%;
- critical extraction precision is at least 98% and recall at least 95%;
- no PHI canary reaches logs, telemetry, crash output, or operational JSON;
- success, cancellation, crash, and startup cleanup leave no clinical scratch;
- only one model process is resident at a time; and
- three cold runs on the 16 GB Apple machine show no memory-pressure
  termination, no sustained swap thrash, and at most 20% final active-memory
  retention.

These gates do not certify HIPAA compliance, prove that OCR is clinically
correct for every document, or turn the application into medical advice or
clinical decision support. Users must review extracted records and summaries
against their source documents.
