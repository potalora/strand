import { expect, test } from "@playwright/test";

import {
  mergeServerLifecycle,
  modelRoleLabel,
  progressRatio,
  progressCounterLabel,
  terminalTransitions,
  toBackgroundJobCard,
  updatedLabel,
} from "./background-processing";
import type { LocalAIJobResponse } from "@/types/local-ai";
import { useBackgroundProcessingStore } from "@/stores/useBackgroundProcessingStore";

function job(
  overrides: Partial<LocalAIJobResponse> = {}
): LocalAIJobResponse {
  return {
    id: "job-1",
    upload_id: "upload-1",
    summary_prompt_id: null,
    kind: "ingestion",
    processing_mode: "validated_strict_local",
    status: "queued",
    stage: "queued",
    progress: null,
    failure: null,
    cancel_requested: false,
    created_at: "2026-07-31T00:00:00Z",
    updated_at: "2026-07-31T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  };
}

test("server lifecycle replaces optimistic accepted state", () => {
  const optimistic = toBackgroundJobCard(job(), { label: "Health summary" });
  const server = job({
    status: "processing",
    stage: "summarizing",
    progress: { worker_current: 2, worker_total: 5 },
    updated_at: "2026-07-31T00:00:02Z",
  });

  const merged = mergeServerLifecycle(
    { [optimistic.id]: optimistic },
    [server]
  );

  expect(merged["job-1"]).toMatchObject({
    status: "processing",
    stage: "summarizing",
    progress: { worker_current: 2, worker_total: 5 },
    label: "Health summary",
  });
});

test("initial terminal hydration never produces a notification", () => {
  const completed = {
    "job-1": toBackgroundJobCard(
      job({ status: "completed", stage: "completed" })
    ),
  };

  expect(terminalTransitions({}, completed, { initialHydration: true })).toEqual(
    []
  );
});

test("active to completed produces one notification and repeated terminal none", () => {
  const active = {
    "job-1": toBackgroundJobCard(
      job({ status: "processing", stage: "ocr" })
    ),
  };
  const completed = {
    "job-1": toBackgroundJobCard(
      job({ status: "completed", stage: "completed" })
    ),
  };

  expect(terminalTransitions(active, completed)).toEqual(["job-1"]);
  expect(terminalTransitions(completed, completed)).toEqual([]);
});

test("progress ratio is bounded and falls back to indeterminate", () => {
  expect(progressRatio({ page_index: 3, page_total: 8 })).toBe(3 / 8);
  expect(progressRatio({ worker_current: 2, worker_total: 4 })).toBe(0.5);
  expect(
    progressRatio({
      page_index: 9,
      page_total: 8,
      worker_current: 1,
      worker_total: 5,
    })
  ).toBe(0.2);
  expect(progressRatio({ page_index: 1, page_total: 0 })).toBeNull();
  expect(progressRatio(null)).toBeNull();
});

test("content-free progress metadata uses bounded counters, role, and update time", () => {
  expect(
    progressCounterLabel({
      page_index: 3,
      page_total: 8,
      model_role: "ocr",
    })
  ).toBe("Page 3 of 8");
  expect(
    progressCounterLabel({ worker_current: 2, worker_total: 5 })
  ).toBe("Batch 2 of 5");
  expect(
    progressCounterLabel({ worker_current: 6, worker_total: 5 })
  ).toBeNull();
  expect(modelRoleLabel({ model_role: "extraction" })).toBe(
    "Extraction model"
  );
  expect(updatedLabel("not-a-date")).toBe("Updated time unavailable");
  expect(updatedLabel("2026-07-31T00:00:00Z")).toMatch(/^Updated /);
});

test("late initial hydration cannot erase a concurrently accepted summary", () => {
  const store = useBackgroundProcessingStore.getState();
  store.reset();
  store.registerAcceptedSummary({
    id: "prompt-race",
    job_id: "job-race",
    processing_mode: "validated_strict_local",
    kind: "summary",
    status: "queued",
    stage: "queued",
    created_at: "2026-07-31T00:00:00Z",
  });

  useBackgroundProcessingStore.getState().hydrate([]);

  expect(useBackgroundProcessingStore.getState().jobs["job-race"]).toMatchObject(
    {
      id: "job-race",
      summary_prompt_id: "prompt-race",
      status: "queued",
    }
  );
  useBackgroundProcessingStore.getState().reset();
});

test("retrying a terminal job clears its notification high-water mark", () => {
  const store = useBackgroundProcessingStore.getState();
  store.reset();
  store.hydrate([job({ status: "failed", stage: "failed" })]);
  useBackgroundProcessingStore.getState().markTerminalNotified("job-1");

  useBackgroundProcessingStore
    .getState()
    .applyServerJobs([job({ status: "queued", stage: "queued" })]);

  expect(
    useBackgroundProcessingStore.getState().notifiedTerminalIds["job-1"]
  ).toBeUndefined();
  useBackgroundProcessingStore.getState().reset();
});
