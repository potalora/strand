# Strict-local AI operations

## Release status

The strict-local path is implemented as an optional, fail-closed processing
mode. The native Apple M4 16 GB profile is shipped and validated. The
repository includes the exact
[`apple-m4-16gb-v1` lock](../backend/app/model_manifests/apple-m4-16gb-v1.lock.json),
its [release evidence](../backend/app/model_manifests/apple-m4-16gb-v1.release.json),
and the content-free
[fidelity](../backend/artifacts/local-ai-fidelity.json) and
[benchmark](../backend/artifacts/local-ai-benchmark.json) reports.

`LOCAL_AI_ENABLED=false` remains the default so model installation and local
processing are opt-in. After an operator enables the feature, the UI reports
the pack as ready only when the installed files match the shipped lock and the
runtime-validation receipt matches the same manifest.

## Supported machine

The first release profile is native Apple Silicon macOS with 16 GB of unified
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

The shipped Apple lock pins these immutable repository revisions:

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
runtime version, and fixture-suite version. The release evidence binds that
lock to the exact benchmark and fidelity report bytes. It also records the
fidelity corpus identity and the thresholds used for promotion.

## Install and manage the pack

Install the isolated runtime without downloading models:

```bash
just local-ai-runtime-install
```

Download, verify, and atomically activate the shipped pack:

```bash
just local-ai-pack-download
```

Re-run the offline runtime and fixture verification:

```bash
just local-ai-pack-verify
```

Run the three-cold-run, content-free memory benchmark:

```bash
just local-ai-benchmark
```

Run the real fidelity gate against the active, receipt-validated pack:

```bash
cd backend
uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-fidelity.json
```

The command renders the committed synthetic PDF and TIFF fixtures, runs
OvisOCR2, NuExtract3, and Qwen3.5-9B in that order, and writes an owner-only
JSON report containing aggregate metrics only. It exits nonzero if any release
threshold fails. The runner checks the committed corpus SHA-256 before loading
the models. Private-corpus metrics are gated separately, so they cannot offset
a failure in the committed synthetic suite. The normal test suite uses a fake
worker and does not load the models.

After both gates pass, regenerate the release evidence:

```bash
just local-ai-release-promote
```

Promotion fails unless both reports match the exact manifest and meet their
current hard gates. The promoted fidelity report must cover all six committed
synthetic documents and no private documents. Readiness revalidates both report
hashes, the fidelity corpus identity, document count, and recorded thresholds.
Older benchmark-only release evidence is not accepted.

To run the same gate through pytest:

```bash
cd backend
LOCAL_AI_ENABLED=true uv run pytest \
  tests/test_local_ai_fidelity.py \
  -m "local_model and fidelity" -v -rs
```

An optional private corpus must stay outside the repository and include an
explicit `local-ai-corpus-v1.json` golden sidecar. Run it as an additional
operator-only check and write its report to an ignored path:

```bash
REAL_MEDICAL_FIXTURES_DIR=/owner/private/path \
  uv run python scripts/run_local_ai_fidelity.py \
  --output artifacts/local-ai-fidelity.private.json
```

The private metrics are gated separately, so they cannot rescue a failed
synthetic run. The report still contains no OCR text, facts, evidence excerpts,
or summary output, but it is not the checked-in promotion artifact.

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
LOCAL_AI_MANIFEST_PATH=./app/model_manifests/apple-m4-16gb-v1.lock.json
LOCAL_AI_RELEASE_EVIDENCE_PATH=./app/model_manifests/apple-m4-16gb-v1.release.json
LOCAL_AI_BENCHMARK_PATH=./artifacts/local-ai-benchmark.json
LOCAL_AI_FIDELITY_PATH=./artifacts/local-ai-fidelity.json
LOCAL_AI_WORKER_COMMAND=../workers/local_ai/apple_mlx/.venv/bin/local-ai-mlx-worker
LOCAL_AI_WORKER_TIMEOUT_SECONDS=900
```

Keep the model directory separate from uploads and scratch. Model files contain
no medical data. The application creates the model-store control files and
scratch directories with owner-only permissions. Per-job scratch is plaintext
only while the owning job runs, uses owner-only directories and files, and is
removed after success, cancellation, failure, and startup recovery.

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
the next backend acquires the lease. The shipped release passed offline role
loading with the exact locked artifacts on the target Mac.

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

The shipped M4 16 GB profile passed the committed six-document synthetic
fidelity suite. All six rate metrics were `1.0`: output-schema validity,
critical numeric exactness, critical precision, critical recall, summary fact
recall, and summary typed-field recall. Accepted facts without evidence,
forbidden extraction facts, and unsupported summary facts were all zero.

The content-free benchmark ran each model role cold three times on a 16 GB
Apple M4 machine. Peak MLX allocations were approximately:

| Role | Peak MLX allocation |
| --- | ---: |
| OCR | 0.86 GB |
| Clinical extraction | 4.73 GB |
| Final summary | 7.10 GB |

The benchmark recorded one live model at a time, no memory-pressure
termination, and no sustained swap thrashing. Its final MLX active-memory
ratio was `3.7e-9`.

Future pack revisions can be called validated only when all of these pass
against their exact release lock:

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
