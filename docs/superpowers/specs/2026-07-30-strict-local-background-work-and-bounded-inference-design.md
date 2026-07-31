# Strict-local background work and bounded inference design

## Context

The validated strict-local path is fail-closed and durable at the database
layer, but the user experience still behaves like an in-tab request. A manual
run on the 16 GB M4 found:

- a small RTF ingestion completed in 27.9 seconds after navigating away;
- reloading removed its live monitor and left upload history stale;
- a two-record summary spent about 51 seconds before a projection/evidence
  validation failure;
- a nineteen-record summary generated for 14 minutes 27 seconds, exhausted two
  unconstrained structured-output attempts, and failed with
  `invalid_structured_output`;
- closing the summary tab did not stop the backend, but reopening the app showed
  `0 records`, an unknown model, and no useful active-job state.

This design makes long local work honest and recoverable, and bounds the two
confirmed inference amplification paths. It extends the existing
`LocalAIJob`, extraction worker, strict summary implementation, and global
status bar. It does not introduce another queue or a second source of truth.

## Goals

1. A user can close a tab, navigate, reload, or sign back in without losing
   visibility into accepted strict-local work.
2. Strict-local summary creation returns immediately after its durable commit.
3. Job status, progress, cancellation, failure, and retry actions reflect
   server state rather than optimistic client state.
4. Planned backend shutdown requeues unfinished, uncancelled work without
   discarding encrypted checkpoints.
5. Projection/evidence failures are detected before Qwen is loaded.
6. Qwen structured selection and NuExtract retries have explicit, tested work
   bounds on the 16 GB profile.
7. New accounts cannot send health information to a cloud provider without an
   explicit mode change.

## Non-goals

- Changing the validated OvisOCR2, NuExtract3, or Qwen3.5-9B model revisions.
- Increasing concurrent model execution on a 16 GB machine.
- Adding cloud fallback to a strict-local failure.
- Allowing the summary model to author unsupported clinical prose.
- Replacing the 30-minute security timeout. Processing continues after sign-out
  and is rediscovered after sign-in.
- Providing a fabricated percentage or ETA when the backend cannot measure one.

## 1. Server-authoritative background jobs

`local_ai_jobs` remains the authoritative lifecycle record. The in-process task
registry exists only to wake queued work, route cancellation to a live child,
and drain tasks during planned shutdown.

### 1.1 Content-free job response

`GET /api/v1/local-ai/jobs` and `GET /api/v1/local-ai/jobs/{job_id}` add:

```json
{
  "id": "uuid",
  "kind": "ingestion",
  "upload_id": "uuid",
  "summary_prompt_id": null,
  "processing_mode": "validated_strict_local",
  "status": "processing",
  "stage": "extracting",
  "progress": {
    "model_role": "extraction",
    "page_index": 2,
    "page_total": 8,
    "worker_current": 1,
    "worker_total": 4,
    "attempt": 1,
    "input_tokens": 2034,
    "output_tokens": 221
  },
  "failure": null,
  "cancel_requested": false,
  "created_at": "2026-07-30T22:00:00Z",
  "updated_at": "2026-07-30T22:03:00Z",
  "started_at": "2026-07-30T22:00:02Z",
  "completed_at": null
}
```

All progress keys are allowlisted bounded integers or enums. Failure responses
contain only bounded safe codes, stage, role, `retryable`,
`checkpoint_preserved`, and `cloud_fallback_attempted`. The API never returns a
stored exception message, filename, path, prompt, evidence excerpt, clinical
value, model output, or patient identifier.

The target UUID lets the authenticated frontend correlate a job with an
existing owner-scoped upload or summary response. It is not content telemetry.

### 1.2 Asynchronous strict-local summaries

For `processing_mode=validated_strict_local`,
`POST /api/v1/summary/generate`:

1. validates ownership and strict-local admission;
2. creates and commits `AISummaryPrompt` plus `LocalAIJob(status=queued)`;
3. schedules the server-owned summary runner after the commit;
4. returns HTTP 202:

```json
{
  "id": "summary-prompt-uuid",
  "job_id": "local-job-uuid",
  "processing_mode": "validated_strict_local",
  "kind": "summary",
  "status": "queued",
  "stage": "queued",
  "created_at": "2026-07-30T22:00:00Z"
}
```

Custom-local and cloud-assisted summary behavior remains synchronous in this
slice. The UI must not describe those modes as durable background jobs.

The summary runner uses a fresh database session, claims one queued summary,
and calls the existing grounded summary service. Duplicate wakeups are safe:
only the task registry owner may run a job, and the database transition is
checked under a row lock. Startup enqueues every resumable summary through the
same runner.

### 1.3 Cancel, retry, and planned shutdown

Cancellation remains cooperative. UI copy is:

> Cancellation requested. Processing will stop after the current safe
> checkpoint.

`POST /local-ai/jobs/{job_id}/retry` is owner-scoped and accepts only a failed
job whose stored safe failure is retryable. It preserves the immutable manifest
snapshot and encrypted checkpoints, clears terminal lifecycle fields, and
requeues the paired upload or summary. Active, cancelled, completed, and
non-retryable jobs return 409. Missing and cross-owner jobs return 404.

During planned shutdown:

1. stop admitting new summary and extraction work;
2. cancel and await tracked runner tasks within a bounded drain;
3. requeue every unfinished job with `cancel_requested=false` as
   `status=queued, stage=recovery`;
4. preserve encrypted OCR/extraction checkpoints and scratch belonging to
   active job IDs;
5. terminalize explicit user cancellations;
6. stop the local model manager.

Crash recovery keeps using the existing startup reconciliation.

## 2. Background processing experience

The dashboard layout mounts one `BackgroundProcessingMonitor` after
authentication. It replaces extraction-session-only monitoring.

On every authenticated mount it queries active server state. It does not rely
on localStorage:

- active ingestion uploads and their latest strict-local jobs;
- active strict-local summary jobs.

The Zustand store normalizes cards by stable server job ID. A newly accepted
upload or summary is inserted immediately, but the next server response wins.
Polling is serialized and limited to active jobs. A terminal transition is kept
long enough to show one notification and can then be dismissed client-side.
Historical jobs do not emit notifications during initial hydration.

Collapsed copy:

> Processing safely in the background. You can leave this page.

Expanded rows show:

- Document or Summary;
- filename only when obtained from the owner-scoped upload API;
- queued/processing/terminal status;
- stable stage label;
- page or batch `current / total` when known;
- elapsed time from `started_at || created_at`;
- last update from `updated_at`;
- model role without model input/output;
- only server-valid actions.

An active single-file job uses an indeterminate progress treatment plus its
real stage and counters. It never displays a 0%-looking determinate ring for a
job that is making progress.

The idle-timeout warning states that local processing continues after sign-out
and will reappear after sign-in. Completion/failure notifications use the
existing local toaster and link to Upload or Summaries.

Upload history hides Extract/Retry for active rows. Failed rows offer retry only
when the server reports it as eligible. A failed cancel request clears the
client action state so the next server poll remains truthful.

## 3. Privacy-safe default

No saved preference means `prompt_only`, not `cloud_assisted`, in:

- resolved per-user LLM configuration;
- the settings API response;
- the summary request default;
- new processing preference rows and defensive database defaults.

Existing explicitly saved user choices remain unchanged. Changing to cloud
assisted remains an explicit user action that shows its raw-vision exception.
Installing a validated pack does not silently switch modes; the UI offers a
clear explicit switch to validated strict local.

## 4. Summary preflight and bounded structured generation

### 4.1 Evidence preflight

Before changing the job to model execution, the backend validates that every
projected safety qualifier has exact linked evidence. Safety qualifiers include
`/assertion`, `/relationship`, `/status`, and `/statuses/*`.

The backend builds the deterministic maximal valid reference document:

- every eligible fact appears once;
- each fact includes only evidence-supported field paths;
- evidence IDs are linked to that fact;
- each eligible uncertainty appears once.

The locked tokenizer measures its compact JSON. The requested output budget is:

```text
summary_output_tokens = maximal_valid_reference_tokens + 128
```

with a floor of 256 and a ceiling of the lower of the manifest cap and 4096. If
the maximal valid document cannot fit, preflight fails before model loading.

Worker input validation and this output-fit calculation run before
`load_role_from_payload`.

### 4.2 One constrained attempt

The Apple MLX runtime uses its installed JSON-schema logits processor for the
existing reference-only schema. Qwen receives one deterministic constrained
generation attempt. The existing post-generation identifier, field-path,
evidence-link, uncertainty, and heading validation remains mandatory.

There is no second blind 4096-token generation. A structured-generation failure
is terminal with a safe category and measured token/attempt diagnostics. The
worker never returns partial or repaired clinical content.

## 5. Bounded NuExtract work

NuExtract retains deterministic batching, one under-cap syntax retry, and
runtime splitting, but one request has four independent limits:

- at most 16,384 generated tokens;
- at most 12 generation attempts;
- at most 7 runtime splits;
- fragment depth at most 3, yielding at most 8 fragments for one source page.

Each call receives:

```text
min(4096, remaining_generated_token_budget)
```

and charges exact generated tokens when the runtime supplies them. If exact
tokens are unavailable, it conservatively charges the tokenizer count or the
requested cap. Output-limit failures split immediately. Under-cap malformed
JSON receives one syntax-only retry. Exhausting any bound fails with a distinct
content-free category and preserves completed checkpoints.

These limits retain the current successful shallow-split behavior while
removing the 256-attempt/255-split amplification path.

## 6. Operational telemetry

Only local, content-free counters are persisted:

- stage and model role;
- page/batch current and total;
- generation attempt;
- bounded input and output token counts;
- runtime split count and fragment depth;
- elapsed duration derived from timestamps;
- safe terminal category.

No document text, prompt text, evidence, summary, clinical value, filename,
patient identity, or raw exception is added to job progress or logs.

## 7. Walkthrough defects included in this slice

- Normalize FastAPI validation-detail arrays/objects into readable client error
  messages instead of `[object Object]`.
- Use the existing calendar-date formatter for clinical `effective_date`
  displays so `2019-01-01` never becomes December 31 in a negative offset.
- Show the owner-visible patient name in the summary selector, with
  `Record subject N` as the fallback. Do not display an internal UUID.
- Always synchronize model-pack operation state from the server so a prior
  failure cannot coexist with a current ready state.
- Add an accessible description to the record detail sheet.

## 8. Verification

The implementation is accepted only when:

1. Backend route tests prove 202 summary admission, owner scoping, safe job
   projection, cancellation, retry eligibility, and restart recovery.
2. A service test proves projection failure never invokes the model manager.
3. Worker tests prove validation-before-load, one constrained summary attempt,
   exact output-fit bounds, and the four NuExtract work limits.
4. Privacy tests prove job APIs/logs exclude seeded PHI and model content.
5. Frontend tests prove reload/login rehydration, serialized polling, truthful
   action states, one terminal notification, and indeterminate long-file UI.
6. Existing strict-local egress, grounding, checkpoint, extraction, lint, build,
   and fast backend suites pass.
7. A final local Browser walkthrough verifies upload and summary jobs through
   navigation and reload with no cloud request.
