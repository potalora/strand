# Durable jobs and upload identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make strict-local ingestion cancellation atomic across its paired upload/job rows, restore retryable failures after reload, and make every batch-upload result unambiguously identify or explain each file.

**Architecture:** Add one strict-ingestion lifecycle service which always locks the upload before its paired job, atomically claims both strict rows into their processing state, and returns the committed state needed for best-effort worker cancellation. Keep summary jobs job-only. Extend the existing owner-scoped job list with an opt-in retryable-failure hydration slice, then have the global monitor request that slice only at initial hydration. Make the server author accepted filenames and bounded stable-code rejections; the upload page consumes those values instead of correlating compacted response positions with the original browser `File[]`.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async/PostgreSQL, Pydantic v2, pytest/pytest-asyncio/httpx; Next.js 16, TypeScript, Zustand, Playwright.

## Global constraints

- This is Track C only and must remain independently executable in parallel with Track A (global pack authorization) and Track B (clinical extraction and bounded work).
- Do not change the worker, model-pack authorization endpoints, validation receipts, Alembic guards, provider selection, 12-attempt ceiling, or 32,768-generated-token ceiling.
- Validated strict-local work must branch before any cloud-capable provider is constructed and must never fall back to cloud.
- Job, upload, progress, failure, and batch rejection responses remain content-free: no document text, prompts, model output, patient identity, server paths, or raw exceptions.
- An ingestion job and its upload must transition atomically when they describe the same strict-local operation. Commit before worker IPC and audit follow-up; IPC and audit remain best effort after the durable state is written.
- Strict worker claiming is also a paired transition: it must atomically change only a verified `("queued", "pending_extraction")` pair to `("processing", "processing")` before a task is spawned. It must not commit an upload-only processing claim.
- For a strict-ingestion pair, only `("queued", "pending_extraction")` and `("processing", "processing")` are cancellable states. A mismatched pair is skipped by the bulk upload route and rejected by the job route; it is never silently repaired.
- A bulk cancellation locks strict pairs in ascending upload UUID order, and each pair locks its upload before its job. No route may retain an upload lock acquired in request order before entering the paired helper.
- Every paired strict read used for a transition is a fresh `FOR UPDATE` read with `populate_existing=True`; non-locking discovery reads only scalar identifiers/classification values and are never used as the transition state.
- Under the pair locks, canonicalize `UploadedFile.processing_manifest` and require its digest and exact snapshot to equal the ingestion job's immutable snapshot/digest; require the strict extraction schema version before either cancellation or claim mutates a row.
- Preserve owner scoping for all job and upload reads/mutations. Summary cancellation remains job-only.
- `GET /local-ai/jobs` remains backward compatible: `include_retryable_failed` is additive, defaults to `false`, and normal active polling remains active-only.
- The batch endpoint remains best effort: accepted entries stay in `uploads`; rejected entries are bounded and contain only submitted filename plus a stable local rejection code.
- Use deterministic synthetic fixtures only. Do not use private medical fixtures, provider calls, model downloads, live cloud tests, or live services.
- Existing unrelated worktrees and untracked files remain untouched. Subagents do not commit; the
  root agent reviews and verifies this track. Every commit block is a proposed checkpoint only;
  do not stage or commit unless Pedro explicitly authorizes it.
- Start this plan in a Codex-managed **Worktree** task based on
  `codex/pr62-pr63-remediation-planning`; retain it as
  `codex/durable-jobs-upload-identity` only when the implementation is ready for root review.

---

## File structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `backend/app/services/local_ai/ingestion_lifecycle.py` | Create | The single paired strict-ingestion cancellation helper, including fixed upload-then-job lock order and terminal state projection. |
| `backend/app/api/local_ai.py` | Modify | Delegate ingestion cancellation to the helper; keep summary cancellation job-only; add the additive retryable-failed list filter. |
| `backend/app/api/upload.py` | Modify | Delegate strict upload cancellation to the helper; keep non-strict cancellation behavior; populate server-authored names and bounded rejection entries. |
| `backend/app/schemas/upload.py` | Modify | Define accepted filename and bounded stable rejection response contracts. |
| `backend/tests/test_local_ai_api.py` | Modify | Cover job-route ingestion cancellation, list hydration filtering, owner isolation, and summary compatibility. |
| `backend/tests/test_upload_progress_cancel.py` | Modify | Cover upload-route paired cancellation for queued and processing ingestion plus post-commit IPC ordering. |
| `backend/tests/test_unstructured_upload.py` | Modify | Cover accepted filenames, compacted batches, stable rejections, size/signature rejection, and all-rejected response shape. |
| `frontend/src/types/api.ts` | Modify | Mirror the accepted/rejected batch contract with literal rejection codes. |
| `frontend/src/lib/api.ts` | Modify | Add the optional hydration flag to `getLocalAIJobs`. |
| `frontend/src/components/retro/BackgroundProcessingMonitor.tsx` | Modify | Hydrate active plus retryable-failed jobs once, silently; continue polling only active jobs. |
| `frontend/src/app/(dashboard)/upload/page.tsx` | Modify | Use `response.filename`, render fixed rejection copy, and present an actionable all-rejected result. |
| `frontend/e2e/upload-extraction-ux.spec.ts` | Modify | Prove reload/retry hydration and that compacted accepted responses/rejections cannot mislabel uploads. |
| `docs/backend-handoff.md` | Root integration only | After Track A merges, the root agent adds Track C's supplied upload/job contract text without editing Track A's operator-authorization section. It is not a Track C branch edit. |

### Task 1: Add one atomic strict-ingestion cancellation primitive and route both cancellation APIs through it

**Files:**
- Create: `backend/app/services/local_ai/ingestion_lifecycle.py`
- Modify: `backend/app/api/local_ai.py:870-927`
- Modify: `backend/app/api/upload.py:1125-1248`
- Test: `backend/tests/test_local_ai_api.py`
- Test: `backend/tests/test_upload_progress_cancel.py`
- Test: `backend/tests/test_unstructured_failure_recovery.py`

**Interfaces:**
- Consumes: `AsyncSession`, owner UUID, `LocalAIJob`, `UploadedFile`, and the existing active states `("queued", "processing")`.
- Produces: `StrictIngestionCancellation(job: LocalAIJob, upload: UploadedFile, worker_cancel_required: bool)` from `cancel_strict_ingestion_pair(db, *, user_id, upload_id, expected_job_id=None)`.
- Produces: A queued pair with both rows terminalized; a processing pair with both `cancel_requested` flags set but neither row terminalized until cooperative worker cleanup.
- Produces: `StrictIngestionClaim` only after one verified queued/pending pair is atomically advanced to the matching processing/processing state.
- Preserves: `POST /local-ai/jobs/{job_id}/cancel` response type `LocalAIJobResponse`, `POST /upload/cancel` response type `CancelExtractionResponse`, owner scoping, and summary job-only cancellation.

- [ ] **Step 1: Write failing route regressions for the paired-state contract**

  Add these tests beside the existing strict cancellation tests. Use the existing `_manifest_payload()`, `_mk_upload()`, `auth_headers()`, and `LocalAIJob` fixtures/helpers rather than a real worker or pack.

  ```python
  @pytest.mark.asyncio
  async def test_job_cancel_terminalizes_queued_strict_ingestion_pair(
      client: AsyncClient, db_session: AsyncSession
  ) -> None:
      headers, user_id = await auth_headers(client)
      snapshot, _digest = canonicalize_manifest_snapshot(_manifest_payload())
      upload = _mk_upload(
          user_id,
          "pending_extraction",
          processing_mode="validated_strict_local",
          processing_manifest=snapshot,
          processing_schema_version="clinical-document-extraction.v1",
      )
      db_session.add(upload)
      await db_session.flush()
      job = LocalAIJob(
          user_id=user_id,
          upload_id=upload.id,
          kind="ingestion",
          processing_mode="validated_strict_local",
          manifest_snapshot=snapshot,
          status="queued",
          stage="queued",
      )
      db_session.add(job)
      await db_session.commit()

      response = await client.post(
          f"/api/v1/local-ai/jobs/{job.id}/cancel", headers=headers
      )

      assert response.status_code == 200
      assert response.json()["status"] == "cancelled"
      await db_session.refresh(upload)
      await db_session.refresh(job)
      assert (upload.ingestion_status, upload.cancel_requested) == ("cancelled", True)
      assert upload.processing_completed_at is not None
      assert (job.status, job.stage, job.cancel_requested) == (
          "cancelled", "cancelled", True
      )
      assert job.completed_at == upload.processing_completed_at
  ```

  Add the mirror upload-route assertion to `test_cancel_finishes_queued_strict_job_without_reserving_worker_cancel`: it must assert both flags, both terminal statuses, cleared upload progress, equal completion timestamps, and no `cancel_registered`/`cancel` call. Assert the job's `private` progress key is absent but its valid `model_role`, `attempt`, `attempt_limit`, `output_tokens`, and `output_token_limit` counters remain. Add a processing-pair test for each route that asserts both flags are durable while `upload.ingestion_status == "processing"`, `job.status == "processing"`, and exactly one `cancel_registered(str(job.id))` is attempted only after the database commit.

  Add a deterministic two-session claim/cancel race in
  `test_unstructured_failure_recovery.py`: pause the strict claim after its
  upload lock but before its paired job lock; a concurrent cancellation must
  block, then observe either the wholly queued/pending pair before claim or the
  wholly processing/processing pair after claim, never queued/processing or
  processing/pending. Release both tasks and assert the final state is paired
  cancelled with no worker spawn. Add a second race which changes a row between
  non-locking discovery and the helper's lock; assert the helper uses the fresh
  locked value rather than the stale ORM identity-map value. Add identity
  mismatch regressions for matching lifecycle states but a different upload
  manifest, a noncanonical upload manifest, a different job digest, and a
  wrong upload extraction schema: job cancellation returns `409`, bulk returns
  `skipped`, and neither row changes.

  Retain the explicit incoherent-pair regression for pre-existing corrupted
  rows (`job.status == "processing"`, `upload.ingestion_status ==
  "pending_extraction"`): job cancellation returns `409`, and bulk puts that
  requested ID in `skipped` without changing either row. The normal worker
  claim path must no longer be able to create this state.

- [ ] **Step 2: Run the cancellation regressions to verify they fail**

  Run:

  ```bash
  cd backend && uv run pytest -q \
    tests/test_local_ai_api.py::test_job_cancel_terminalizes_queued_strict_ingestion_pair \
    tests/test_upload_progress_cancel.py::test_cancel_finishes_queued_strict_job_without_reserving_worker_cancel
  ```

  Expected: FAIL. The job route currently terminalizes only the queued `LocalAIJob`, leaving its upload in `pending_extraction`; the upload route owns duplicated paired logic instead of a shared helper.

- [ ] **Step 3: Create the fixed-lock-order service and make its transition rules explicit**

  Create `backend/app/services/local_ai/ingestion_lifecycle.py`. Both entry points must arrive here before acquiring either paired row. For an upload-originated request, resolve only owner-scoped candidate UUIDs without `FOR UPDATE`, sort distinct strict IDs by `str(upload_id)`, then call this helper once per ID. For a job-originated request, make a non-locking, owner-scoped projection to discover `upload_id`, then use the same helper and the same upload-then-job lock order. Never lock the job first in one route and never retain request-order upload locks before this helper; the sorted pair loop is required to prevent two reversed bulk requests from deadlocking.

  ```python
  """Atomic lifecycle transitions for one strict-local ingestion pair."""
  from __future__ import annotations

  from dataclasses import dataclass
  from datetime import datetime, timezone
  from uuid import UUID

  from fastapi import HTTPException
  from sqlalchemy import select
  from sqlalchemy.ext.asyncio import AsyncSession

  from app.models.local_ai import LocalAIJob
  from app.models.uploaded_file import UploadedFile
  from app.services.local_ai.manifest import canonicalize_manifest_snapshot

  _ACTIVE = ("queued", "processing")
  _CANCELLABLE_PAIRS = frozenset({
      ("queued", "pending_extraction"),
      ("processing", "processing"),
  })
  _STRICT_LOCAL_MODEL_ROLES = frozenset({"ocr", "extraction", "summary"})
  _STRICT_LOCAL_PROGRESS_COUNTERS = frozenset({
      "page_index", "page_total", "worker_current", "worker_total", "current",
      "total", "activity", "attempt", "attempt_limit", "input_tokens",
      "output_tokens", "output_token_limit", "splits_used", "split_limit",
      "active_memory_bytes", "peak_memory_bytes",
  })
  _STRICT_LOCAL_MAX_PROGRESS_COUNTER = 2**63 - 1
  _STRICT_INGESTION_SCHEMA = "clinical-document-extraction.v1"

  @dataclass(frozen=True)
  class StrictIngestionCancellation:
      job: LocalAIJob
      upload: UploadedFile
      worker_cancel_required: bool

  @dataclass(frozen=True)
  class StrictIngestionClaim:
      """One committed strict pair claimed by this worker attempt."""

      upload_id: UUID
      job_id: UUID
      storage_path: str
      user_id: UUID
      claimed_at: datetime

  def _cancelled_progress(prior: object) -> dict[str, object]:
      """Retain only content-free strict-local telemetry at terminal cancellation."""
      progress: dict[str, object] = {"stage": "cancelled"}
      if not isinstance(prior, dict):
          return progress
      model_role = prior.get("model_role")
      if model_role in _STRICT_LOCAL_MODEL_ROLES:
          progress["model_role"] = model_role
      for key in _STRICT_LOCAL_PROGRESS_COUNTERS:
          value = prior.get(key)
          if type(value) is int and 0 <= value <= _STRICT_LOCAL_MAX_PROGRESS_COUNTER:
              progress[key] = value
      return progress

  async def cancel_strict_ingestion_pair(
      db: AsyncSession,
      *,
      user_id: UUID,
      upload_id: UUID,
      expected_job_id: UUID | None = None,
  ) -> StrictIngestionCancellation:
      """Lock upload then job and atomically persist one valid cancellation state."""
      upload = (
          await db.execute(
              select(UploadedFile)
              .where(UploadedFile.id == upload_id, UploadedFile.user_id == user_id)
              .with_for_update()
              .execution_options(populate_existing=True)
          )
      ).scalar_one_or_none()
      if upload is None:
          raise HTTPException(status_code=404, detail="Local AI job not found.")
      job_query = select(LocalAIJob).where(
          LocalAIJob.upload_id == upload.id,
          LocalAIJob.user_id == user_id,
          LocalAIJob.kind == "ingestion",
          LocalAIJob.processing_mode == "validated_strict_local",
      )
      if expected_job_id is not None:
          job_query = job_query.where(LocalAIJob.id == expected_job_id)
      job = (
          await db.execute(
              job_query.with_for_update().execution_options(populate_existing=True)
          )
      ).scalar_one_or_none()
      if job is None or upload.processing_mode != "validated_strict_local":
          raise HTTPException(status_code=409, detail="This job cannot be cancelled.")
      try:
          _require_matching_strict_ingestion_identity(upload, job)
      except Exception:
          raise HTTPException(status_code=409, detail="This job cannot be cancelled.")
      pair_state = (job.status, upload.ingestion_status)
      if pair_state not in _CANCELLABLE_PAIRS:
          raise HTTPException(status_code=409, detail="This job cannot be cancelled.")

      worker_cancel_required = pair_state == ("processing", "processing")
      job.cancel_requested = True
      upload.cancel_requested = True
      if not worker_cancel_required:
          cancelled_at = datetime.now(timezone.utc)
          job.status = job.stage = "cancelled"
          job.progress = _cancelled_progress(job.progress)
          job.failure = None
          job.completed_at = cancelled_at
          upload.ingestion_status = "cancelled"
          upload.progress_stage = None
          upload.progress_detail = None
          upload.processing_completed_at = cancelled_at
      return StrictIngestionCancellation(job, upload, worker_cancel_required)

  async def claim_next_strict_ingestion_pair(
      db: AsyncSession,
  ) -> StrictIngestionClaim | None:
      """Atomically claim one verified strict pair before task creation."""
      upload = (
          await db.execute(
              select(UploadedFile)
              .where(
                  UploadedFile.ingestion_status == "pending_extraction",
                  UploadedFile.file_category == "unstructured",
                  UploadedFile.manual_extraction_required.is_(False),
                  UploadedFile.processing_mode == "validated_strict_local",
              )
              .order_by(UploadedFile.created_at, UploadedFile.id)
              .limit(1)
              .with_for_update(skip_locked=True)
              .execution_options(populate_existing=True)
          )
      ).scalar_one_or_none()
      if upload is None:
          return None
      job = (
          await db.execute(
              select(LocalAIJob)
              .where(
                  LocalAIJob.upload_id == upload.id,
                  LocalAIJob.user_id == upload.user_id,
                  LocalAIJob.kind == "ingestion",
                  LocalAIJob.processing_mode == "validated_strict_local",
              )
              .with_for_update()
              .execution_options(populate_existing=True)
          )
      ).scalar_one_or_none()
      if job is None:
          return None
      _require_matching_strict_ingestion_identity(upload, job)
      if (job.status, upload.ingestion_status) != ("queued", "pending_extraction"):
          return None
      claimed_at = datetime.now(timezone.utc)
      upload.ingestion_status = "processing"
      upload.processing_started_at = claimed_at
      upload.progress_stage = "local_preflight"
      upload.progress_detail = None
      job.status = "processing"
      job.stage = "preflight"
      job.started_at = claimed_at
      job.failure = None
      return StrictIngestionClaim(
          upload_id=upload.id,
          job_id=job.id,
          storage_path=upload.storage_path,
          user_id=upload.user_id,
          claimed_at=claimed_at,
      )
  ```

  Factor the duplicated snapshot checks above into
  `_require_matching_strict_ingestion_identity(upload, job)`. It canonicalizes
  the upload snapshot, requires the immutable canonical snapshot/digest to
  equal the job snapshot/digest, and requires
  `upload.processing_schema_version == "clinical-document-extraction.v1"`.
  It raises one content-free `409` boundary for cancellation and one safe
  claim rejection for the worker loop; it never mutates either row.

  Preserve the repository's existing allowlisted progress counters when replacing `job.progress`; move the current strict cancellation projection into this service rather than importing `app.api.upload` from a service. The final helper must not accept a summary job, must never change an immutable processing snapshot, and must not call IPC, audit logging, or `commit()`. The caller commits the claim before creating a task, so no worker can observe an upload-only processing state.

- [ ] **Step 4: Delegate each route, commit once, then do best-effort side effects**

  In `backend/app/api/upload.py`, make strict claiming use the same lifecycle
  service rather than the current upload-only SQL update. When a strict slot is
  available, call `claim_next_strict_ingestion_pair()` in its own transaction,
  commit it before `asyncio.create_task`, and return the claim's upload ID,
  job ID, immutable `claimed_at`, storage path, and owner ID to the child.
  The non-strict `SELECT ... FOR UPDATE SKIP LOCKED` path remains separate and
  explicitly excludes validated strict-local rows. The worker loop must not
  first mark a strict upload `processing` and later rely on
  `_run_strict_local_ingestion_for_upload()` to claim the job.

  Thread `StrictIngestionClaim` into `_process_and_release()` and
  `_process_unstructured()`. For that strict claim, re-lock upload then the
  expected job with `populate_existing=True`, require the same snapshot/schema/
  digest parity and the exact `(processing, processing)` state with the
  returned `job.started_at == claimed_at`, then run the pipeline. Remove the
  current second `queued -> processing` job update in
  `_run_strict_local_ingestion_for_upload()`; it must treat the committed claim
  as authoritative and fail/return safely if stale or cancelled. A cancellation
  between commit and task start sees a coherent processing pair, records both
  cancellation flags, and causes the child to terminate without worker spawn.

  In `backend/app/api/local_ai.py`, retain the owner-scoped preliminary **non-locking** job projection so a summary route remains distinguishable. For an ingestion job, call `cancel_strict_ingestion_pair(...)`, commit and refresh before invoking `local_model_manager.cancel_registered()`/fallback `cancel()`, then audit. For a summary job, retain its existing locked job-only state transition and IPC behavior.

  Preserve non-strict file cancellation, but replace the current bulk `UploadedFile.with_for_update()` plus `strict_jobs` block. First make a non-locking owner-scoped scalar lookup of requested IDs and classification values, never ORM entities. Process unique strict candidate IDs in `sorted(strict_ids, key=str)` through `cancel_strict_ingestion_pair`; treat its `404`/`409` as `skipped` for this bulk response. Only then lock and mark eligible non-strict rows. Build `cancelled` and `skipped` from the original `body.upload_ids` order after those outcomes are known, so duplicate input IDs retain the current response shape while each strict pair is mutated once. This keeps every retained lock in one ascending pair order and prevents a reversed two-file request from deadlocking.

  The route shape should be structurally equivalent to this sequence:

  ```python
  result = await cancel_strict_ingestion_pair(
      db, user_id=user_id, upload_id=upload_id, expected_job_id=job_id
  )
  await db.commit()
  await db.refresh(result.job)
  if result.worker_cancel_required:
      try:
          registered = await local_model_manager.cancel_registered(str(result.job.id))
          if not registered:
              await local_model_manager.cancel(str(result.job.id))
      except LocalWorkerError:
          logger.warning("Local worker cancellation could not be confirmed", extra={"job_id": str(result.job.id)})
  await log_audit_event(...)
  ```

  Keep the upload endpoint's request-order `cancelled`/`skipped` response arrays. A terminal, foreign, malformed, non-strict, missing-pair, incoherent, or already-closed target is skipped for the bulk endpoint; it must not widen ownership or raise a raw database error. Do not start/restart the extraction worker as part of cancellation.

- [ ] **Step 5: Run the focused lifecycle suite and verify the complete paired invariant**

  Run:

  ```bash
  cd backend && uv run pytest -q \
    tests/test_local_ai_api.py \
    tests/test_upload_progress_cancel.py \
    tests/test_unstructured_failure_recovery.py
  ```

  Expected: PASS. Strict claim atomically advances a verified queued/pending pair
  to processing/processing before task creation; queued cancellation through
  either route leaves the upload and job cancelled/unclaimable in one committed
  state while retaining only allowlisted job telemetry; processing cancellation
  sets both flags before best-effort IPC; stale, identity-mismatched, and
  pre-existing incoherent strict pairs are never repaired; summary cancellation
  stays job-only; owner and terminal-state regressions stay green.

- [ ] **Step 6: Root-agent review and prepare the proposed backend checkpoint**

  The implementing subagent does not commit. The root agent inspects the diff, confirms both routes
  acquire pair locks only through `ingestion_lifecycle.py`, and reruns Step 5. Run the commands
  below only if Pedro separately authorizes a commit:

  ```bash
  git add backend/app/services/local_ai/ingestion_lifecycle.py backend/app/api/local_ai.py backend/app/api/upload.py backend/tests/test_local_ai_api.py backend/tests/test_upload_progress_cancel.py backend/tests/test_unstructured_failure_recovery.py
  git commit -m "fix: atomically cancel strict ingestion pairs"
  ```

### Task 2: Rehydrate retryable failed jobs without changing steady-state polling

**Files:**
- Modify: `backend/app/api/local_ai.py:706-739`
- Modify: `frontend/src/lib/api.ts:324-333`
- Modify: `frontend/src/components/retro/BackgroundProcessingMonitor.tsx:87-155`
- Test: `backend/tests/test_local_ai_api.py`
- Test: `frontend/e2e/upload-extraction-ux.spec.ts`
- Test: `frontend/src/lib/background-processing.unit.spec.ts`

**Interfaces:**
- Consumes: `GET /local-ai/jobs?active_only=<bool>&include_retryable_failed=<bool>`.
- Produces: When both query values are true, up to 50 owner-scoped active jobs plus failed jobs whose bounded `failure.retryable` is exactly `true`; completed, cancelled, non-retryable, and cross-owner jobs remain absent.
- Produces: `api.getLocalAIJobs(activeOnly = false, includeRetryableFailed = false)`.
- Preserves: `getLocalAIJobs(true)` polling behavior, 50-row limit, silent initial hydration, sequential per-job polling, and existing Retry UI action.

- [ ] **Step 1: Write failing backend filter and reload/retry UI tests**

  Add a single deterministic backend fixture matrix: one active summary, one retryable failed summary, one retryable failed manual ZIP child, one non-retryable failed ingestion, one cancelled job, one completed job, and one retryable failed job owned by another user. Assert only the active and retryable failed owner jobs are returned for the additive query; the manual child must remain absent. Assert `active_only=true` without the new flag returns only the active job.

  ```python
  response = await client.get(
      "/api/v1/local-ai/jobs?active_only=true&include_retryable_failed=true",
      headers=owner_headers,
  )

  assert response.status_code == 200
  assert {item["id"] for item in response.json()} == {
      str(active_summary.id), str(retryable_failure.id)
  }
  assert all("message" not in json.dumps(item) for item in response.json())
  ```

  In `frontend/e2e/upload-extraction-ux.spec.ts`, extend `mockBackend` so its `/local-ai/jobs` handler reads `include_retryable_failed`. Add a test that serves a retryable failed job only when that flag is `true`, loads `/upload`, reloads, opens the Background processing monitor, finds the retryable job, and clicks Retry. Assert the initial hydration did not render a completion/failure toast, the Retry button is visible, and the retry request changes it to queued.

- [ ] **Step 2: Run the focused tests to verify they fail**

  Run:

  ```bash
  cd backend && uv run pytest -q tests/test_local_ai_api.py -k "retryable_failed or summary_job_status"
  cd ../frontend && npx playwright test e2e/upload-extraction-ux.spec.ts --grep "retryable failure reload"
  ```

  Expected: FAIL. The current list query filters strictly to `queued`/`processing`, and the monitor only calls `/local-ai/jobs?active_only=true`, so a failed retryable job is not hydrated after reload.

- [ ] **Step 3: Add the additive server filter before ordering and limit**

  In `list_local_ai_jobs`, add `include_retryable_failed: bool = False`. Apply the failed branch only when `active_only and include_retryable_failed` are both true. Use SQLAlchemy's JSON boolean accessor, not Python-side filtering, so owner filtering, exclusion, ordering, and `limit(50)` all occur in the database.

  ```python
  from sqlalchemy import and_, exists, func, or_, select

  retryable_failed = and_(
      LocalAIJob.status == "failed",
      LocalAIJob.failure["retryable"].as_boolean().is_(True),
      ~manual_upload_gate,
  )
  active_visible = and_(
      LocalAIJob.status.in_(_ACTIVE_JOB_STATES),
      ~manual_upload_gate,
  )
  if active_only:
      query = query.where(
          or_(active_visible, retryable_failed)
          if include_retryable_failed
          else active_visible
      )
  ```

  Build `manual_upload_gate` only inside the `active_only` branch, before both `retryable_failed` and `active_visible`. Do not expose a failed manual ZIP child merely because it has a retryable payload; it is still available through the existing history/manual workflow. Keep `kind` filtering, `user_id == user_id`, descending `created_at`/`id`, `_job_response()` projection, and the 50-row limit exactly as they are.

- [ ] **Step 4: Wire only initial monitor hydration to the expanded query**

  Extend the API client without changing existing callers:

  ```ts
  async getLocalAIJobs(
    activeOnly = false,
    includeRetryableFailed = false
  ): Promise<LocalAIJobResponse[]> {
    return this.get<LocalAIJobResponse[]>(
      `/local-ai/jobs?active_only=${activeOnly ? "true" : "false"}` +
        `&include_retryable_failed=${includeRetryableFailed ? "true" : "false"}`
    );
  }
  ```

  In `BackgroundProcessingMonitor`, replace only the mount-time call with `api.getLocalAIJobs(true, true)`. Leave the polling effect based on `activeIdsKey`, so a hydrated failed job is displayed with Retry but never receives `GET /local-ai/jobs/{id}` polling. Keep `hydrate(serverJobs)` as the silent initial state application; do not invoke `terminalTransitions()` during hydration.

- [ ] **Step 5: Run focused backend and frontend verification**

  Run:

  ```bash
  cd backend && uv run pytest -q tests/test_local_ai_api.py
  cd ../frontend && npx playwright test e2e/upload-extraction-ux.spec.ts
  npx playwright test --config playwright.unit.config.ts src/lib/background-processing.unit.spec.ts
  ```

  Expected: PASS. Reload hydrates retryable strict-local failures with Retry and no initial toast, a retry turns it back into an actively polled queued job, and normal active polling remains serial and excludes terminal jobs.

- [ ] **Step 6: Root-agent review and prepare the proposed hydration checkpoint**

  The implementing subagent does not commit. The root agent verifies that the new query defaults to
  false and that no terminal response includes failure text. Run the commands below only if Pedro
  separately authorizes a commit:

  ```bash
  git add backend/app/api/local_ai.py backend/tests/test_local_ai_api.py frontend/src/lib/api.ts frontend/src/components/retro/BackgroundProcessingMonitor.tsx frontend/e2e/upload-extraction-ux.spec.ts frontend/src/lib/background-processing.unit.spec.ts
  git commit -m "fix: restore retryable local jobs after reload"
  ```

### Task 3: Make unstructured batch responses self-identifying and expose bounded rejections

**Files:**
- Modify: `backend/app/schemas/upload.py:91-119`
- Modify: `backend/app/api/upload.py:3296-3524,3527-3684`
- Modify: `backend/tests/test_unstructured_upload.py`
- Modify: `frontend/src/types/api.ts:219-224`
- Modify: `frontend/src/app/(dashboard)/upload/page.tsx:155,475-520,980-1093`
- Modify: `frontend/e2e/upload-extraction-ux.spec.ts`
- Root integration only: `docs/backend-handoff.md` after Track A's authorization documentation has merged

**Interfaces:**
- Produces: `UnstructuredUploadResponse(filename: str, upload_id: str, status: str, file_type: str, manual_extraction_required: bool = False)` for `/upload/unstructured`, `/upload/unstructured-batch`, and reprocess responses.
- Produces: `BatchUploadResponse(uploads: list[UnstructuredUploadResponse], rejected: list[RejectedUnstructuredUpload], total: int)` where `rejected` has at most 50 entries.
- Produces: `RejectedUnstructuredUpload(filename: str, code: Literal["missing_filename", "unsupported_type", "file_too_large", "invalid_signature"])`.
- Consumes: the server response's `filename`, never the array index of browser-selected files.
- Preserves: accepted `uploads`, `total == len(uploads)`, HTTP 202 for best-effort batches, no raw failure detail in accepted/rejected payloads, and the batch-wide immutable processing snapshot. Invalid files are classified before snapshot admission; an all-rejected batch returns 202 even if the requested processing mode would reject a valid upload.

- [ ] **Step 1: Write failing API contract tests for compacted and all-rejected batches**

  Extend `test_batch_upload_skips_invalid_files` so the first submitted item is invalid and later valid files are accepted. Assert the server reports each accepted server filename exactly, rather than assuming it returns one response per submitted file. Add cases for all stable rejection codes and the all-rejected case.

  ```python
  assert response.status_code == 202
  payload = response.json()
  assert payload["total"] == 2
  assert [(item["filename"], item["file_type"]) for item in payload["uploads"]] == [
      ("later-valid.rtf", "rtf"), ("also-valid.pdf", "pdf")
  ]
  assert payload["rejected"] == [
      {"filename": "first-invalid.txt", "code": "unsupported_type"}
  ]
  ```

  Add a test posting only an unsupported filename, an over-limit file (patch `settings.max_file_size_mb` to a tiny deterministic value), and a PDF filename with non-PDF header. Patch `_resolve_ingestion_snapshot_or_409` to raise `HTTPException(status_code=409, detail="unavailable")`; the all-rejected batch must still return 202 because snapshot admission is never attempted. Assert `total == 0`, `uploads == []`, the exact ordered `(filename, code)` list, and that serializing the response contains neither `/tmp/`, an exception class, nor a raw stream-validation message. Add a companion valid-RTF case with the same patched resolver that returns 409 and leaves no staged upload file behind. Add a single-upload assertion that `data["filename"] == "note.rtf"` and update reprocess assertions to require the stored filename.

  In the Playwright upload-page test, mock a compacted response in which the first selected browser file is rejected and the two accepted responses carry `later-valid.rtf` and `final-valid.pdf`. Assert rows and tracked progress labels use those server names. Add an all-rejected response and assert the page shows `No files were accepted. Choose a supported PDF, RTF, or TIFF and try again.` plus each fixed rejection label, without displaying a server `detail` string.

- [ ] **Step 2: Run the batch identity regressions to verify they fail**

  Run:

  ```bash
  cd backend && uv run pytest -q tests/test_unstructured_upload.py -k "batch_upload or unstructured"
  cd ../frontend && npx playwright test e2e/upload-extraction-ux.spec.ts --grep "compacted batch|all rejected"
  ```

  Expected: FAIL. The batch endpoint silently skips bad files, accepted responses lack `filename`, and the page maps returned index zero to the first selected browser file even when that file was rejected.

- [ ] **Step 3: Define the bounded public Pydantic contract**

  In `backend/app/schemas/upload.py`, add a literal code type and use `Field(max_length=50)` at the response boundary. The maximum is intentional: it prevents a malformed or adversarial batch from causing an unbounded error payload while still giving a useful actionable sample.

  ```python
  BatchRejectionCode = Literal[
      "missing_filename",
      "unsupported_type",
      "file_too_large",
      "invalid_signature",
  ]

  class RejectedUnstructuredUpload(BaseModel):
      filename: str
      code: BatchRejectionCode

  class UnstructuredUploadResponse(BaseModel):
      upload_id: str
      filename: str
      status: str
      file_type: str
      manual_extraction_required: bool = False

  class BatchUploadResponse(BaseModel):
      uploads: list[UnstructuredUploadResponse]
      rejected: list[RejectedUnstructuredUpload] = Field(
          default_factory=list, max_length=50
      )
      total: int
  ```

  Do not add server error strings, file paths, validation detail, or a general-purpose free-text code. Keep this additive so clients that read only `uploads` and `total` continue to parse the response.

- [ ] **Step 4: Populate accepted names and stable rejections at the upload boundary**

  In `backend/app/api/upload.py`, define `MAX_BATCH_REJECTIONS = 50` and a local append helper that stores at most that many `RejectedUnstructuredUpload` instances. It must derive only the safe submitted `file.filename or ""` and a literal code. Remove the current eager `_resolve_ingestion_snapshot_or_409(...)` call before the batch loop: snapshot admission applies only after a file has passed filename, extension, size, and magic-byte validation.

  ```python
  def _append_batch_rejection(
      rejected: list[RejectedUnstructuredUpload], *, filename: str, code: BatchRejectionCode
  ) -> None:
      if len(rejected) < MAX_BATCH_REJECTIONS:
          rejected.append(RejectedUnstructuredUpload(filename=filename, code=code))
  ```

  Apply it in this exact order per batch file:

  ```python
  if not file.filename:
      _append_batch_rejection(rejected, filename="", code="missing_filename")
      continue
  ext = Path(file.filename).suffix.lower()
  if ext not in ALLOWED_UNSTRUCTURED:
      _append_batch_rejection(rejected, filename=file.filename, code="unsupported_type")
      continue
  try:
      file_size, header, file_hash = await _stream_upload_to_disk(...)
  except HTTPException as exc:
      if exc.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE:
          _append_batch_rejection(rejected, filename=file.filename, code="file_too_large")
          continue
      raise
  if not _validate_magic_bytes(header, ext):
      file_path.unlink(missing_ok=True)
      _append_batch_rejection(rejected, filename=file.filename, code="invalid_signature")
      continue
  if snapshot is None:
      try:
          snapshot = await _resolve_ingestion_snapshot_or_409(
              db, user_id, processing_mode
          )
  except Exception:
          file_path.unlink(missing_ok=True)
          raise
  ```

  Initialize `snapshot = None` before the loop and reuse the first successfully admitted snapshot for every accepted item; after the first valid file, check `snapshot is not None` before accessing `snapshot.mode`, `snapshot.manifest_snapshot`, or `snapshot.schema_version`. The exception cleanup above is required because stream validation created the encrypted temporary file before admission. Add `filename=upload_record.filename` to every `UnstructuredUploadResponse` constructor: direct unstructured upload, each batch accepted item, duplicate reprocess return, and newly reprocessed return. Return `BatchUploadResponse(uploads=results, rejected=rejected, total=len(results))`. Do not alter the existing per-file transaction, immutable snapshot admission, audit shape, accepted row ordering, or worker start behavior.

- [ ] **Step 5: Replace client-side positional association with typed response identity and fixed copy**

  In `frontend/src/types/api.ts`, mirror the literal union and the additive response object:

  ```ts
  export type UnstructuredBatchRejectionCode =
    | "missing_filename"
    | "unsupported_type"
    | "file_too_large"
    | "invalid_signature";

  export interface UnstructuredUploadResponse {
    upload_id: string;
    filename: string;
    status: string;
    file_type: string;
    manual_extraction_required: boolean;
  }

  export interface UnstructuredBatchResponse {
    uploads: UnstructuredUploadResponse[];
    rejected: { filename: string; code: UnstructuredBatchRejectionCode }[];
    total: number;
  }
  ```

  In the upload page, use a total code-to-copy mapping that never displays backend details:

  ```ts
  const BATCH_REJECTION_COPY: Record<UnstructuredBatchRejectionCode, string> = {
    missing_filename: "This file did not include a filename.",
    unsupported_type: "This file type is not supported.",
    file_too_large: "This file is larger than the upload limit.",
    invalid_signature: "This file does not match its claimed format.",
  };
  ```

  Replace the positional loop with response-authored identity:

  ```ts
  const response = await api.postForm<UnstructuredBatchResponse>(
    "/upload/unstructured-batch", formData
  );
  for (const upload of response.uploads) {
    results.push({ type: "unstructured", filename: upload.filename, response: upload });
    batchInputs.push({
      upload_id: upload.upload_id,
      filename: upload.filename,
      status: upload.status || "pending_extraction",
      needsTrigger: upload.manual_extraction_required === true,
    });
  }
  for (const rejection of response.rejected) {
    results.push({
      type: "unstructured",
      filename: rejection.filename || "Unnamed file",
      error: BATCH_REJECTION_COPY[rejection.code],
    });
  }
  if (response.uploads.length === 0 && response.rejected.length > 0) {
    setUploadError("No files were accepted. Choose a supported PDF, RTF, or TIFF and try again.");
  }
  ```

  Use `resp.filename` for the single-upload result and `batchInputs` too. Render the result card heading as `Upload results` whenever every displayed unstructured item is a rejection, rather than showing a success-heading for an all-rejected request. Do not start a progress batch when `batchInputs.length === 0`.

- [ ] **Step 6: Hand the exact documentation amendment to the root integrator**

  Do not edit `docs/backend-handoff.md` in the Track C branch; Track A owns the adjacent validated-local authorization section. Supply the root integrator this exact amendment after Track A merges: update the single and batch unstructured response examples to include `filename`; state that `uploads` is best effort, `total` is the accepted count, rejection entries are capped at 50, and the only rejection codes are `missing_filename`, `unsupported_type`, `file_too_large`, and `invalid_signature`; replace the current “silently skipped” statement; and append to the Local processing jobs section that `include_retryable_failed=false` is additive and, only with `active_only=true`, includes owner-scoped failed jobs whose bounded `failure.retryable` is true while excluding manual ZIP children, completed, cancelled, non-retryable, and foreign jobs before the 50-row limit. Retain the existing no-content privacy language.

- [ ] **Step 7: Run focused identity and UX verification**

  Run:

  ```bash
  cd backend && uv run pytest -q tests/test_unstructured_upload.py tests/test_upload_audit_privacy.py tests/test_processing_mode_snapshot.py
  cd ../frontend && npx playwright test e2e/upload-extraction-ux.spec.ts
  npx tsc --noEmit
  ```

  Expected: PASS. A rejected first file cannot shift either accepted ID onto another filename, each public rejection uses a stable code and no sensitive detail, and an all-rejected response has actionable UI without an extraction batch.

- [ ] **Step 8: Root-agent review and prepare the proposed response/UI checkpoint**

  The implementing subagent does not commit. The root agent checks the rendered response examples
  and verifies no raw exception/path is surfaced. Run the commands below only if Pedro separately
  authorizes a commit:

  ```bash
  git add backend/app/schemas/upload.py backend/app/api/upload.py backend/tests/test_unstructured_upload.py frontend/src/types/api.ts frontend/src/app/\(dashboard\)/upload/page.tsx frontend/e2e/upload-extraction-ux.spec.ts
  git commit -m "fix: identify unstructured batch upload results"
  ```

### Task 4: Perform Track C integration verification and hand off to the root agent

**Files:**
- Modify only if verification exposes a Track C defect: the exact Track C files listed above.
- Test: `backend/tests/test_local_ai_api.py`
- Test: `backend/tests/test_upload_progress_cancel.py`
- Test: `backend/tests/test_unstructured_upload.py`
- Test: `backend/tests/test_unstructured_failure_recovery.py`
- Test: `frontend/e2e/upload-extraction-ux.spec.ts`
- Test: `frontend/src/lib/background-processing.unit.spec.ts`

**Interfaces:**
- Consumes: all three completed Track C slices.
- Produces: evidence that queue/processing cancellation, retryable hydration/retry, compact batch identities, and bounded rejection UI work together without changing Tracks A/B contracts.
- Preserves: no commit from subagents and no cross-track code edits.

- [ ] **Step 1: Run the complete focused Track C backend regression group**

  Run:

  ```bash
  cd backend && uv run pytest -q \
    tests/test_local_ai_api.py \
    tests/test_upload_progress_cancel.py \
    tests/test_unstructured_failure_recovery.py \
    tests/test_unstructured_upload.py \
    tests/test_upload_audit_privacy.py \
    tests/test_processing_mode_snapshot.py
  ```

  Expected: PASS. If test collection is blocked because this worktree lacks `pypdfium2`, report that exact collection dependency instead of claiming backend verification; do not install, upgrade, or use a provider as a workaround without authorization.

- [ ] **Step 2: Run the complete focused Track C frontend regression group**

  Run:

  ```bash
  cd frontend && npx playwright test e2e/upload-extraction-ux.spec.ts
  npx playwright test --config playwright.unit.config.ts src/lib/background-processing.unit.spec.ts
  npx tsc --noEmit
  npm run lint
  ```

  Expected: PASS. The monitor silently hydrates retryable failures, Retry moves the item back to queued, no terminal job is polled, and accepted/rejected batch rows retain their server-authored filename.

- [ ] **Step 3: Run the ordinary repository gates after focused tests pass**

  Run:

  ```bash
  cd backend && uv run pytest -m "not slow and not fidelity and not local_model and not hardware" -q
  cd ../frontend && npm run build
  ```

  Expected: PASS. Record the actual pass/skip/deselection counts and any pre-existing environment blocker in the handoff; never manufacture green counts.

- [ ] **Step 4: Root-agent integration review and parallel handoff**

  Subagents do not commit. The root agent reviews the final diff for only Track C paths, confirms no
  conflict with Track A's adjacent `backend/app/api/local_ai.py` endpoint groups or Track B's worker
  changes, and reruns the applicable commands above. Run the commands below only if Pedro
  explicitly authorizes a Track C commit and earlier checkpoints were intentionally deferred:

  ```bash
  git status --short
  git diff --check
  git add backend/app/services/local_ai/ingestion_lifecycle.py backend/app/api/local_ai.py backend/app/api/upload.py backend/app/schemas/upload.py backend/tests/test_local_ai_api.py backend/tests/test_upload_progress_cancel.py backend/tests/test_unstructured_failure_recovery.py backend/tests/test_unstructured_upload.py frontend/src/lib/api.ts frontend/src/types/api.ts frontend/src/components/retro/BackgroundProcessingMonitor.tsx frontend/src/app/\(dashboard\)/upload/page.tsx frontend/src/lib/background-processing.unit.spec.ts frontend/e2e/upload-extraction-ux.spec.ts
  git commit -m "fix: make local jobs and uploads durable"
  ```

  Hand off the commit SHA, focused/full command outputs, any exact environment blocker, and the explicit integration note: Track C is independent of Tracks A/B but the root integrator must resolve any nearby `local_ai.py` conflict before the final combined verification gate.

## Plan self-review

- **Spec coverage:** Task 1 covers fresh upload-then-job locks, immutable upload/job
  snapshot/schema/digest parity, atomic strict claim, deterministic claim/cancel races,
  both cancellation routes, queued terminalization, processing cooperative cancellation,
  commit-before-IPC/audit, and summary job-only compatibility. Task 2 covers the additive retryable-failed filter, 50-row SQL
  ordering/limit, owner isolation, reload hydration, silent monitor behavior, and Retry. Task 3
  covers server-authored accepted filenames, compact responses, bounded stable rejection codes,
  all-rejected UI, compatibility, and documentation. Task 4 provides focused and full verification;
  root-only commits remain separately authorized.
- **Placeholder scan:** The plan contains concrete paths, signatures, response fields, code snippets, commands, expected outcomes, and commit scopes; it contains no deferred implementation marker.
- **Type consistency:** `StrictIngestionCancellation`, `StrictIngestionClaim`,
  `cancel_strict_ingestion_pair`, `claim_next_strict_ingestion_pair`,
  `include_retryable_failed`, `RejectedUnstructuredUpload`,
  `UnstructuredBatchRejectionCode`, and `UnstructuredBatchResponse` use the same names and fields throughout the plan.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-durable-jobs-and-upload-identity.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, and retain root-agent-only commits.

2. **Inline Execution** - Execute tasks in this session using executing-plans, in reviewable batches.

Which approach?
