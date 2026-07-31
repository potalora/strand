# Durable local background jobs implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make accepted strict-local uploads and summaries discoverable,
actionable, and truthful after navigation, reload, sign-out, and planned backend
restart.

**Architecture:** Extend the existing owner-scoped `LocalAIJob` API and use it
as the server authority. Add an in-process summary runner only for wake,
cancellation, and graceful drain; the database remains the queue. Replace the
session-only extraction status bar with one server-hydrated background monitor.

**Tech Stack:** FastAPI, SQLAlchemy async, PostgreSQL, Pydantic v2, Next.js,
TypeScript, Zustand, Playwright.

## Global constraints

- Strict-local jobs never construct a cloud provider or fall back to one.
- Job progress and failure responses contain allowlisted operational metadata
  only; no filenames, prompts, evidence, model output, clinical values, patient
  identifiers, paths, or raw exception messages.
- Custom-local and cloud-assisted summaries remain synchronous and are not
  advertised as durable background work.
- The 30-minute idle timeout remains enabled; its copy states that processing
  continues and is rediscovered after sign-in.
- The server is authoritative after every client hydration or poll.
- New accounts default to `prompt_only`; cloud-assisted processing is explicit
  opt-in.
- Subagents must not commit. The root agent stages and commits each reviewed
  task.

---

### Task 1: Complete the content-free job API and retry transition

**Files:**
- Modify: `backend/app/schemas/local_ai.py`
- Modify: `backend/app/api/local_ai.py`
- Modify: `backend/app/api/upload.py`
- Modify: `backend/tests/test_local_ai_api.py`
- Modify: `backend/tests/test_unstructured_failure_recovery.py`
- Modify: `backend/tests/test_local_ai_log_privacy.py`
- Modify: `docs/backend-handoff.md`

**Interfaces:**
- Produces: `LocalAIJobResponse` with target UUIDs, mode, bounded progress,
  bounded failure, and `updated_at`.
- Produces: `POST /api/v1/local-ai/jobs/{job_id}/retry` for ingestion jobs;
  Task 2 adds summary wake-up after the summary runner exists.
- Consumes: existing `LocalAIJob`, `UploadedFile`, strict checkpoints, and the
  upload worker wake path.

- [ ] **Step 1: Write failing API projection and retry tests**

Add assertions equivalent to:

```python
payload = owner_list.json()[0]
assert payload["upload_id"] == str(upload.id)
assert payload["summary_prompt_id"] is None
assert payload["processing_mode"] == "validated_strict_local"
assert payload["progress"] == {
    "model_role": "extraction",
    "page_index": 2,
    "page_total": 8,
    "attempt": 1,
}
assert payload["failure"] == {
    "stage": "extracting",
    "code": "local_worker_error",
    "retryable": True,
    "checkpoint_preserved": True,
    "cloud_fallback_attempted": False,
}
assert "message" not in json.dumps(payload)
assert "sensitive patient content" not in json.dumps(payload)
```

Test retry success for a retryable failed ingestion job and 409 responses for
active, completed, cancelled, and non-retryable jobs. Test cross-owner retry as
404. Assert the upload and job requeue in one transaction and retained
`LocalAIPage` rows are unchanged.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_local_ai_api.py \
  tests/test_unstructured_failure_recovery.py \
  tests/test_local_ai_log_privacy.py -q
```

Expected: new response fields and retry route assertions fail.

- [ ] **Step 3: Add bounded response models**

Implement strict Pydantic models with this public shape:

```python
class LocalAIJobProgress(_StrictResponse):
    model_role: Literal["ocr", "extraction", "summary"] | None = None
    page_index: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    page_total: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    worker_current: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    worker_total: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    attempt: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    input_tokens: StrictInt | None = Field(default=None, ge=0, le=10_000_000)
    output_tokens: StrictInt | None = Field(default=None, ge=0, le=10_000_000)
    splits_used: StrictInt | None = Field(default=None, ge=0, le=1_000_000)


class LocalAIJobFailure(_StrictResponse):
    stage: StrictStr = Field(min_length=1, max_length=32)
    code: StrictStr = Field(min_length=1, max_length=64)
    model_role: Literal["ocr", "extraction", "summary"] | None = None
    retryable: bool
    checkpoint_preserved: bool = False
    cloud_fallback_attempted: bool = False
```

`LocalAIJobResponse` adds `upload_id`, `summary_prompt_id`,
`processing_mode`, `progress`, `failure`, and `updated_at`. `_job_response`
copies only keys accepted by these models and never returns the stored
`failure.message`.

- [ ] **Step 4: Implement one shared retry transition**

Add an owner-scoped row-locked helper:

```python
async def _retry_local_ai_job(
    db: AsyncSession,
    *,
    job: LocalAIJob,
) -> None:
    failure = job.failure if isinstance(job.failure, dict) else {}
    if job.status != "failed" or failure.get("retryable") is not True:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")
    if job.cancel_requested:
        raise HTTPException(status_code=409, detail="This job cannot be retried.")
    if job.kind != "ingestion":
        raise HTTPException(status_code=409, detail="This job cannot be retried.")
    job.status = "queued"
    job.stage = "queued"
    job.progress = {}
    job.failure = None
    job.started_at = None
    job.completed_at = None
```

Atomically reset the paired owner-scoped `UploadedFile` to
`pending_extraction`, clear terminal timing/errors, and preserve checkpoints.
Make the legacy strict upload retry path call this helper instead of carrying a
second transition. Task 2 removes the `kind != "ingestion"` guard and wakes the
new summary runner after committing a summary retry.

- [ ] **Step 5: Verify GREEN and privacy**

Run the command from Step 2. Expected: all selected tests pass and seeded
sensitive strings are absent from API/log captures.

- [ ] **Step 6: Root review and commit**

```bash
git add backend/app/schemas/local_ai.py backend/app/api/local_ai.py \
  backend/app/api/upload.py backend/tests/test_local_ai_api.py \
  backend/tests/test_unstructured_failure_recovery.py \
  backend/tests/test_local_ai_log_privacy.py docs/backend-handoff.md
git commit -m "feat(local-ai): expose durable background job state"
```

---

### Task 2: Queue strict-local summaries and drain them safely

**Files:**
- Create: `backend/app/services/local_ai/summary_runner.py`
- Modify: `backend/app/api/summary.py`
- Modify: `backend/app/api/local_ai.py`
- Modify: `backend/app/schemas/summary.py`
- Modify: `backend/app/services/ai/summarizer.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/api/upload.py`
- Test: `backend/tests/test_local_ai_summary_runner.py`
- Modify: `backend/tests/test_summarization.py`
- Modify: `backend/tests/test_local_ai_lifecycle_audit_regressions.py`

**Interfaces:**
- Produces: `StrictLocalSummaryAccepted`.
- Produces: `local_summary_runner.enqueue(job_id)`,
  `local_summary_runner.start(job_ids)`, and
  `local_summary_runner.stop_and_requeue()`.
- Extends: Task 1 retry endpoint so eligible summary retries are enqueued after
  their durability commit.
- Produces: `stop_extraction_worker()` that drains tracked file tasks before the
  model manager stops.
- Consumes: Task 1 job projection and retry wake.

- [ ] **Step 1: Write failing admission, claim, and shutdown tests**

The endpoint test must patch the runner and prove inference is not awaited:

```python
enqueue = Mock()
monkeypatch.setattr(
    "app.api.summary.local_summary_runner.enqueue",
    enqueue,
)
response = await client.post("/api/v1/summary/generate", headers=headers, json=body)
assert response.status_code == 202
assert response.json()["status"] == "queued"
enqueue.assert_called_once()
model_run.assert_not_awaited()
```

Runner tests create two wakeups for one queued job and assert only one grounded
summary call. Shutdown tests cancel a live task, then assert uncancelled work is
`queued/recovery` while explicit user cancellation remains `cancelled`.
Extraction shutdown tests prove its tracked child task is awaited and its
processing upload/job pair is requeued before `local_model_manager.stop`.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
cd backend
uv run pytest tests/test_local_ai_summary_runner.py \
  tests/test_summarization.py \
  tests/test_local_ai_lifecycle_audit_regressions.py -q
```

Expected: missing runner/202/drain behavior fails.

- [ ] **Step 3: Implement the summary runner**

The runner owns no clinical state:

```python
class LocalSummaryRunner:
    def __init__(self) -> None:
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._draining = False

    def enqueue(self, job_id: UUID) -> None:
        if self._draining or job_id in self._tasks:
            return
        task = asyncio.create_task(self._run_one(job_id))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(job_id, None))

    async def _run_one(self, job_id: UUID) -> None:
        await resume_grounded_local_summary_jobs([job_id])

    async def stop_and_requeue(self) -> None:
        self._draining = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await requeue_interrupted_summary_jobs()
        self._tasks.clear()
```

The real implementation row-locks before running and requeues only jobs whose
`cancel_requested` is false. `start()` clears draining and enqueues recovered
IDs.

- [ ] **Step 4: Return HTTP 202 after the durability commit**

Add:

```python
class StrictLocalSummaryAccepted(BaseModel):
    id: UUID
    job_id: UUID
    processing_mode: Literal["validated_strict_local"]
    kind: Literal["summary"] = "summary"
    status: Literal["queued"] = "queued"
    stage: Literal["queued"] = "queued"
    created_at: datetime
```

After `db.commit()` and refresh, call `local_summary_runner.enqueue(job.id)` and
return this response with `status_code=202`. Keep all non-strict response
branches unchanged.

- [ ] **Step 5: Track and stop extraction child tasks**

Maintain a module set for `_process_and_release` tasks, stop the poller first,
cancel/await tracked children, and invoke the existing recovery transaction
before stopping the model manager. Planned task cancellation must be requeued,
not exposed as a user cancellation.

- [ ] **Step 6: Verify GREEN**

Run Step 2, then:

```bash
cd backend
uv run pytest tests/test_local_ai_api.py \
  tests/test_unstructured_failure_recovery.py \
  tests/test_strict_local_pipeline.py -q
```

- [ ] **Step 7: Root review and commit**

```bash
git add backend/app/services/local_ai/summary_runner.py \
  backend/app/api/summary.py backend/app/api/local_ai.py \
  backend/app/schemas/summary.py \
  backend/app/services/ai/summarizer.py backend/app/main.py \
  backend/app/api/upload.py backend/tests/test_local_ai_summary_runner.py \
  backend/tests/test_summarization.py \
  backend/tests/test_local_ai_lifecycle_audit_regressions.py
git commit -m "feat(local-ai): run strict summaries in background"
```

---

### Task 3: Replace the session monitor with server hydration

**Files:**
- Create: `frontend/src/stores/useBackgroundProcessingStore.ts`
- Create: `frontend/src/components/retro/BackgroundProcessingMonitor.tsx`
- Create: `frontend/src/lib/background-processing.ts`
- Modify: `frontend/src/app/(dashboard)/layout.tsx`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/types/local-ai.ts`
- Modify: `frontend/src/types/upload.ts`
- Modify: `frontend/src/app/(dashboard)/summaries/page.tsx`
- Modify: `frontend/src/app/(dashboard)/upload/page.tsx`
- Delete: `frontend/src/components/retro/GlobalExtractionStatusBar.tsx`
- Modify: `frontend/e2e/upload-extraction-ux.spec.ts`
- Create: `frontend/e2e/background-processing-summary.spec.ts`
- Create: `frontend/src/lib/background-processing.unit.spec.ts`

**Interfaces:**
- Consumes: Tasks 1-2 job API and strict summary 202 response.
- Produces: one normalized `BackgroundJobCard` store and global monitor.

- [ ] **Step 1: Write failing normalization and E2E tests**

Unit cases must prove:

```typescript
expect(mergeServerJobs(local, server)[job.id].status).toBe("processing");
expect(nextTerminalNotifications(hydratedHistory, hydratedHistory)).toEqual([]);
expect(nextTerminalNotifications(activeBefore, completedAfter)).toEqual([job.id]);
```

Playwright cases mock active job discovery on first dashboard load, reload the
page, and assert `Background processing`, the real stage, elapsed copy, and
“You can leave this page” return. A 202 summary response must clear the submit
spinner immediately and register a Summary card. A processing upload-history
row must not expose Extract.

- [ ] **Step 2: Run tests and verify RED**

```bash
cd frontend
npx playwright test e2e/upload-extraction-ux.spec.ts \
  e2e/background-processing-summary.spec.ts --workers=1
```

Expected: hydration, 202, and truthful-action assertions fail.

- [ ] **Step 3: Add typed API helpers and normalized store**

Add:

```typescript
export interface BackgroundJobCard {
  id: string;
  kind: "ingestion" | "summary";
  targetId: string;
  label: string;
  status: LocalJobStatus;
  stage: string;
  progress: LocalJobProgress;
  failure: LocalJobFailure | null;
  cancelRequested: boolean;
  createdAt: string;
  updatedAt: string;
  startedAt: string | null;
  completedAt: string | null;
}
```

API helpers list active local jobs, fetch one job, cancel one job, and retry one
job. The store exposes `hydrate`, `upsert`, `markActionPending`,
`clearActionPending`, `dismiss`, and a terminal-notification high-water set.
Hydration always replaces conflicting local lifecycle fields with server
fields.

- [ ] **Step 4: Implement the global monitor**

Mount after auth hydration. Poll sequentially every two seconds while any card
is active. Use an indeterminate bar for active jobs without a measurable
current/total ratio. Show server-valid Cancel/Retry only. On cancel request
failure, clear the pending action and show the API message. Use `sonner` once
per observed active-to-terminal transition, but never for terminal rows in the
initial hydration.

- [ ] **Step 5: Adapt Upload and Summaries**

Strict-local summary submission handles `202` as:

```typescript
const accepted = await api.generateSummary(body);
if ("job_id" in accepted) {
  backgroundStore.registerAcceptedSummary(accepted);
  setNotice("Summary is processing in the background.");
  setLoading(false);
  return;
}
setResult(accepted);
```

Upload acceptance inserts its new strict-local job/upload card, then lets
hydration win. Remove Extract/Retry from processing history rows. Preserve ZIP
children that genuinely need a manual trigger.

- [ ] **Step 6: Verify GREEN, lint, and build**

```bash
cd frontend
npx playwright test e2e/upload-extraction-ux.spec.ts \
  e2e/background-processing-summary.spec.ts --workers=1
npm run lint
npm run build
```

- [ ] **Step 7: Root review and commit**

```bash
git add frontend/src frontend/e2e
git commit -m "feat(ui): recover local jobs after reload"
```

---

### Task 4: Make cloud opt-in and close walkthrough defects

**Files:**
- Modify: `backend/app/services/ai/llm/config.py`
- Modify: `backend/app/api/llm_settings.py`
- Modify: `backend/app/schemas/summary.py`
- Modify: `backend/app/models/ai_summary.py`
- Modify: `backend/app/models/uploaded_file.py`
- Modify: `backend/app/models/llm_settings.py`
- Create: `backend/alembic/versions/b0c1d2e3f4a5_default_processing_prompt_only.py`
- Modify: `backend/tests/test_llm_settings_api.py`
- Modify: `backend/tests/test_llm_settings_models.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/format-date.ts`
- Modify: clinical date consumers under `frontend/src/app/(dashboard)` and
  `frontend/src/components/retro`
- Modify: `frontend/src/app/(dashboard)/summaries/page.tsx`
- Modify: `frontend/src/hooks/useLocalPackOperation.ts`
- Modify: `frontend/src/components/retro/RecordDetailSheet.tsx`
- Modify: relevant frontend unit and Playwright specs

**Interfaces:**
- Produces: no-preference processing mode `prompt_only`.
- Produces: `formatApiErrorDetail()` and consistent calendar-date rendering.

- [ ] **Step 1: Write failing privacy-default and UI regression tests**

Backend:

```python
config = await load_llm_config(db_session, user_id_without_preferences)
assert config.processing_mode is ProcessingMode.PROMPT_ONLY
assert settings_response.json()["routing"]["processing_mode"] == "prompt_only"
assert GenerateSummaryRequest(patient_id=uuid4()).processing_mode is ProcessingMode.PROMPT_ONLY
```

Frontend tests cover a FastAPI 422 `detail` list, `2019-01-01` rendering in
America/New_York, owner name in the summary selector, a ready pack with an old
failed operation, and a record sheet with an accessible description.

- [ ] **Step 2: Run tests and verify RED**

```bash
cd backend
uv run pytest tests/test_llm_settings_api.py \
  tests/test_llm_settings_models.py tests/test_processing_mode_snapshot.py -q
cd ../frontend
npm run lint
```

- [ ] **Step 3: Change defensive defaults without rewriting explicit choices**

Use `ProcessingMode.PROMPT_ONLY` for no-preference application/schema defaults.
Change new-row/server defaults in an Alembic migration, but do not update
existing non-null `processing_mode` values.

- [ ] **Step 4: Normalize API errors once**

```typescript
export function formatApiErrorDetail(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String(item.msg)
          : ""
      )
      .filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (detail && typeof detail === "object" && "msg" in detail) {
    return String(detail.msg);
  }
  return "Request failed";
}
```

`ApiClient.request` always passes this string to `ApiError`.

- [ ] **Step 5: Apply targeted UI corrections**

Use `fmtDay` only for clinical calendar dates; retain normal timestamp
formatting for `created_at`/`generated_at`. Summary subject labels use decrypted
name or `Record subject ${index + 1}`. Pack-operation state always synchronizes
from the newest server status. Add a visually hidden `SheetDescription`.

- [ ] **Step 6: Verify GREEN**

Run Step 2 plus:

```bash
cd frontend
npx playwright test e2e/strict-local-upload-progress.spec.ts \
  e2e/background-processing-summary.spec.ts --workers=1
npm run build
```

- [ ] **Step 7: Root review and commit**

```bash
git add backend/app backend/alembic/versions backend/tests frontend/src frontend/e2e
git commit -m "fix(privacy): require explicit cloud opt-in"
```
