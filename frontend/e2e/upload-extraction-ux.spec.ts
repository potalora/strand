import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const json = (body: unknown) => ({
  contentType: "application/json",
  body: JSON.stringify(body),
});

function localJob(
  id: string,
  uploadId: string,
  overrides: Record<string, unknown> = {}
) {
  return {
    id,
    upload_id: uploadId,
    summary_prompt_id: null,
    kind: "ingestion",
    processing_mode: "validated_strict_local",
    status: "processing",
    stage: "ocr",
    progress: { page_index: 1, page_total: 4 },
    failure: null,
    cancel_requested: false,
    created_at: new Date(Date.now() - 5_000).toISOString(),
    updated_at: new Date().toISOString(),
    started_at: new Date(Date.now() - 4_000).toISOString(),
    completed_at: null,
    ...overrides,
  };
}

function historyItem(
  id: string,
  filename: string,
  status: string,
  overrides: Record<string, unknown> = {}
) {
  return {
    id,
    filename,
    ingestion_status: status,
    record_count: 0,
    file_size_bytes: 1024,
    created_at: "2026-07-31T00:00:00Z",
    ingestion_progress: {},
    ingestion_errors: [],
    manual_extraction_required: false,
    local_run: {
      privacy_mode: "validated_strict_local",
      models: [],
    },
    local_failure: null,
    ...overrides,
  };
}

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

interface MockState {
  jobs: ReturnType<typeof localJob>[];
  history: ReturnType<typeof historyItem>[];
  pollInFlight: number;
  maxPollInFlight: number;
  retryFailures: number;
  cancelFailures: number;
  triggerCalls: string[][];
  processingMode?: "validated_strict_local" | "cloud_assisted";
  directUploadId?: string;
  directUploadFilename?: string;
  batchUploadResponse?: {
    uploads: {
      upload_id: string;
      filename: string;
      status: string;
      file_type: string;
      manual_extraction_required: boolean;
    }[];
    rejected: {
      filename: string;
      code:
        | "missing_filename"
        | "unsupported_type"
        | "file_too_large"
        | "invalid_signature";
    }[];
    total: number;
    detail?: string;
  };
  batchUploadCalls?: number;
  extractionProgressCalls?: number;
  jobAfterUpload?: ReturnType<typeof localJob>;
  extractionProgress?: {
    total: number;
    completed: number;
    processing: number;
    failed: number;
    pending: number;
    records_created: number;
  };
  extractionFiles?: {
    id: string;
    filename: string;
    ingestion_status: string;
  }[];
  jobsDelayMs?: number;
  historyDelayMs?: number;
  historyGate?: Promise<void>;
  jobListQueries?: string[];
}

async function mockBackend(page: Page, state: MockState): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;

    // Explicitly handle local jobs before this file's broad API fallback.
    if (path === "/api/v1/local-ai/jobs") {
      const activeOnly = url.searchParams.get("active_only") !== "false";
      const includeRetryableFailed =
        url.searchParams.get("include_retryable_failed") === "true";
      state.jobListQueries?.push(url.search);
      if (state.jobsDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, state.jobsDelayMs));
      }
      return route.fulfill(
        json(
          activeOnly
            ? state.jobs.filter(
                (job) =>
                  ((job.status === "queued" || job.status === "processing") ||
                    (includeRetryableFailed &&
                      job.status === "failed" &&
                      (
                        job.failure as { retryable?: boolean } | null
                      )?.retryable === true)) &&
                  !state.history.some(
                    (upload) =>
                      upload.id === job.upload_id &&
                      upload.manual_extraction_required === true
                  )
              )
            : state.jobs
        )
      );
    }
    const jobMatch = path.match(/^\/api\/v1\/local-ai\/jobs\/([^/]+)$/);
    if (jobMatch && request.method() === "GET") {
      state.pollInFlight += 1;
      state.maxPollInFlight = Math.max(
        state.maxPollInFlight,
        state.pollInFlight
      );
      await new Promise((resolve) => setTimeout(resolve, 100));
      state.pollInFlight -= 1;
      return route.fulfill(
        json(state.jobs.find((job) => job.id === jobMatch[1]))
      );
    }
    const retryMatch = path.match(
      /^\/api\/v1\/local-ai\/jobs\/([^/]+)\/retry$/
    );
    if (retryMatch) {
      if (state.retryFailures > 0) {
        state.retryFailures -= 1;
        return route.fulfill({
          status: 503,
          ...json({ detail: "Retry is temporarily unavailable." }),
        });
      }
      const job = state.jobs.find((item) => item.id === retryMatch[1])!;
      Object.assign(job, {
        status: "queued",
        stage: "queued",
        failure: null,
      });
      return route.fulfill(json(job));
    }
    const cancelMatch = path.match(
      /^\/api\/v1\/local-ai\/jobs\/([^/]+)\/cancel$/
    );
    if (cancelMatch) {
      if (state.cancelFailures > 0) {
        state.cancelFailures -= 1;
        return route.fulfill({
          status: 503,
          ...json({ detail: "Cancel is temporarily unavailable." }),
        });
      }
      const job = state.jobs.find((item) => item.id === cancelMatch[1])!;
      Object.assign(job, {
        status: "cancelled",
        stage: "cancelled",
        cancel_requested: true,
      });
      return route.fulfill(json(job));
    }
    if (path === "/api/v1/upload/history") {
      if (state.historyGate) await state.historyGate;
      if (state.historyDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, state.historyDelayMs));
      }
      return route.fulfill(
        json({ items: state.history, total: state.history.length })
      );
    }
    if (path === "/api/v1/upload/trigger-extraction") {
      const body = request.postDataJSON() as { upload_ids: string[] };
      state.triggerCalls.push(body.upload_ids);
      for (const upload of state.history) {
        if (body.upload_ids.includes(upload.id)) {
          upload.manual_extraction_required = false;
        }
      }
      return route.fulfill(
        json({
          triggered: body.upload_ids.length,
          failed: 0,
          results: body.upload_ids.map((upload_id) => ({
            upload_id,
            status: "pending_extraction",
          })),
        })
      );
    }
    if (
      path === "/api/v1/upload/unstructured" &&
      request.method() === "POST"
    ) {
      if (
        state.jobAfterUpload &&
        !state.jobs.some((job) => job.id === state.jobAfterUpload?.id)
      ) {
        state.jobs.push(state.jobAfterUpload);
      }
      return route.fulfill(
        json({
          upload_id: state.directUploadId ?? "direct-upload",
          filename: state.directUploadFilename ?? "direct.pdf",
          status: "pending_extraction",
          file_type: "pdf",
          manual_extraction_required: false,
        })
      );
    }
    if (
      path === "/api/v1/upload/unstructured-batch" &&
      request.method() === "POST"
    ) {
      state.batchUploadCalls = (state.batchUploadCalls ?? 0) + 1;
      return route.fulfill(
        json(
          state.batchUploadResponse ?? {
            uploads: [],
            rejected: [],
            total: 0,
          }
        )
      );
    }
    if (path === "/api/v1/upload/extraction-progress") {
      state.extractionProgressCalls = (state.extractionProgressCalls ?? 0) + 1;
      return route.fulfill(
        json(
          state.extractionProgress ?? {
            total: 0,
            completed: 0,
            processing: 0,
            failed: 0,
            pending: 0,
            records_created: 0,
          }
        )
      );
    }
    if (path === "/api/v1/upload/pending-extraction") {
      return route.fulfill(
        json({
          files: (state.extractionFiles ?? []).map((file) => ({
            ...file,
            mime_type: "application/pdf",
            file_category: "unstructured",
            file_size_bytes: 1024,
            created_at: null,
            manual_extraction_required: false,
          })),
          total: state.extractionFiles?.length ?? 0,
        })
      );
    }
    if (path === "/api/v1/auth/me") {
      return route.fulfill(
        json({
          id: "user-1",
          email: "pedro@example.com",
          display_name: "Pedro",
          is_active: true,
          created_at: "2026-01-01T00:00:00Z",
        })
      );
    }
    if (path === "/api/v1/settings/llm") {
      return route.fulfill(
        json({
          providers: [],
          routing: {
            default: "gemini",
            summary: "gemini",
            section: "gemini",
            dedup: "gemini",
            extraction: "gemini",
            vision: "gemini",
            extraction_engine: "local",
            processing_mode:
              state.processingMode ?? "validated_strict_local",
          },
        })
      );
    }
    return route.fulfill(json({}));
  });
}

test.describe("server-hydrated background processing", () => {
  for (const resolutionOrder of ["history-first", "jobs-first"] as const) {
    test(`retryable failure reload stays silent and keeps labels when ${resolutionOrder}`, async ({
      page,
    }) => {
      await injectAuth(page);
      const failedJob = localJob("retryable-job", "retryable-upload", {
        status: "failed",
        stage: "failed",
        progress: null,
        failure: {
          stage: "ocr",
          code: "worker_timeout",
          retryable: true,
          checkpoint_preserved: true,
          cloud_fallback_attempted: false,
        },
      });
      const state: MockState = {
        jobs: [failedJob],
        history: [
          historyItem("retryable-upload", "server-retryable.pdf", "failed"),
        ],
        pollInFlight: 0,
        maxPollInFlight: 0,
        retryFailures: 0,
        cancelFailures: 0,
        triggerCalls: [],
        jobsDelayMs: resolutionOrder === "history-first" ? 150 : 0,
        historyDelayMs: resolutionOrder === "jobs-first" ? 150 : 0,
        jobListQueries: [],
      };
      await mockBackend(page, state);

      await page.goto("/upload");
      await page.reload();
      const monitor = page.getByRole("region", {
        name: "Background processing",
      });
      await expect(monitor).toBeVisible();
      await monitor
        .getByRole("button", { name: /background processing/i })
        .click();
      await expect(monitor).toContainText("server-retryable.pdf");
      await expect(monitor.getByRole("button", { name: /^retry$/i })).toBeVisible();
      await expect(page.getByText(/could not be processed\.$/)).toHaveCount(0);
      expect(
        state.jobListQueries?.some((query) =>
          query.includes("include_retryable_failed=true")
        )
      ).toBe(true);
      expect(state.maxPollInFlight).toBe(0);

      await monitor.getByRole("button", { name: /^retry$/i }).click();
      await expect
        .poll(() => failedJob.status)
        .toBe("queued");
      await expect(monitor.getByRole("button", { name: /^cancel$/i })).toBeVisible();
      await expect
        .poll(() => state.maxPollInFlight, { timeout: 8_000 })
        .toBe(1);
    });
  }

  test("retryable job hydration does not wait for hung upload history", async ({
    page,
  }) => {
    await injectAuth(page);
    let releaseHistory!: () => void;
    const historyGate = new Promise<void>((resolve) => {
      releaseHistory = resolve;
    });
    const state: MockState = {
      jobs: [
        localJob("hung-history-job", "hung-history-upload", {
          status: "failed",
          stage: "failed",
          progress: null,
          failure: { code: "worker_timeout", retryable: true },
        }),
      ],
      history: [historyItem("hung-history-upload", "late-label.pdf", "failed")],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 0,
      triggerCalls: [],
      historyGate,
    };
    await mockBackend(page, state);

    await page.goto("/upload");
    const monitor = page.getByRole("region", {
      name: "Background processing",
    });
    await expect(monitor).toBeVisible({ timeout: 2_000 });
    await monitor
      .getByRole("button", { name: /background processing/i })
      .click();
    await expect(monitor.getByRole("button", { name: /^retry$/i })).toBeVisible();

    releaseHistory();
    await expect(monitor).toContainText("late-label.pdf");
  });

  test("discovers active jobs on initial load and reload, stays visible off-page, and polls serially", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [
        localJob("job-1", "upload-1"),
        localJob("job-2", "upload-2", {
          stage: "extracting_entities",
          progress: { worker_current: 2, worker_total: 8 },
        }),
      ],
      history: [
        historyItem("upload-1", "scan.pdf", "processing"),
        historyItem("upload-2", "long-note.pdf", "processing"),
      ],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 1,
      triggerCalls: [],
    };
    await mockBackend(page, state);

    await page.goto("/upload");
    const monitor = page.getByRole("region", {
      name: "Background processing",
    });
    await expect(monitor).toBeVisible();
    await monitor.getByRole("button", { name: /background processing/i }).click();
    await expect(monitor).toContainText("scan.pdf");
    await expect(monitor).toContainText("Extracting Entities");
    await expect(monitor).toContainText("Page 1 of 4");
    await expect(monitor).toContainText(/Updated /);
    await expect(monitor).toContainText(/elapsed/);
    await expect(monitor).toContainText(
      "Processing safely in the background. You can leave this page."
    );
    const cancel = monitor.getByRole("button", { name: /^cancel$/i }).first();
    await cancel.click();
    await expect(
      page.getByText("Cancel is temporarily unavailable.")
    ).toBeVisible();
    await expect(cancel).toBeEnabled();
    await cancel.click();
    await expect(page.getByText(/was cancelled\.$/)).toBeVisible();
    await expect
      .poll(() => state.maxPollInFlight, { timeout: 8_000 })
      .toBe(1);

    await page.goto("/timeline");
    await expect(monitor).toBeVisible();
    await page.reload();
    await expect(monitor).toBeVisible();
  });

  test("history hides Extract for processing/direct pending, gates retry, and preserves manual ZIP Extract", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [
        localJob("failed-job", "failed-upload", {
          status: "failed",
          stage: "failed",
          progress: null,
          failure: {
            stage: "ocr",
            code: "worker_timeout",
            retryable: true,
            checkpoint_preserved: true,
            cloud_fallback_attempted: false,
          },
        }),
        localJob("manual-job", "manual-upload", {
          status: "queued",
          stage: "queued",
          progress: null,
        }),
      ],
      history: [
        historyItem("processing-upload", "processing.pdf", "processing"),
        historyItem("failed-upload", "retryable.pdf", "failed", {
          local_job_id: "failed-job",
          local_failure: {
            stage: "ocr",
            code: "worker_timeout",
            message: "Strict-local processing did not complete.",
            retryable: true,
            checkpoint_preserved: true,
            cloud_fallback_attempted: false,
          },
        }),
        historyItem("manual-upload", "zip-child.pdf", "pending_extraction", {
          manual_extraction_required: true,
        }),
        historyItem("direct-upload", "direct.pdf", "pending_extraction"),
      ],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 1,
      cancelFailures: 0,
      triggerCalls: [],
    };
    await mockBackend(page, state);

    await page.goto("/upload");
    const hydratedMonitor = page.getByRole("region", {
      name: "Background processing",
    });
    await expect(hydratedMonitor).toBeVisible();
    await hydratedMonitor
      .getByRole("button", { name: /background processing/i })
      .click();
    await expect(hydratedMonitor).toContainText("retryable.pdf");
    await page.getByRole("button", { name: /upload history/i }).click();

    const processingRow = page.locator("tr", { hasText: "processing.pdf" });
    const directRow = page.locator("tr", { hasText: "direct.pdf" });
    const manualRow = page.locator("tr", { hasText: "zip-child.pdf" });
    const retryRow = page.locator("tr", { hasText: "retryable.pdf" });
    await expect(processingRow.getByRole("button", { name: /extract/i })).toHaveCount(0);
    await expect(directRow.getByRole("button", { name: /extract/i })).toHaveCount(0);
    await expect(manualRow.getByRole("button", { name: /extract/i })).toBeVisible();
    await expect(
      retryRow.getByRole("button", { name: /^retry$/i })
    ).toBeVisible();

    await retryRow.getByRole("button", { name: /^retry$/i }).click();
    await expect(page.getByText("Retry is temporarily unavailable.")).toBeVisible();
    await expect(
      retryRow.getByRole("button", { name: /^retry$/i })
    ).toBeEnabled();

    await manualRow.getByRole("button", { name: /extract/i }).click();
    await expect.poll(() => state.triggerCalls).toContainEqual(["manual-upload"]);
    const monitor = page.getByRole("region", {
      name: "Background processing",
    });
    await expect(monitor).toBeVisible();
    await expect(monitor).toContainText("zip-child.pdf");
    await page.goto("/timeline");
    await expect(monitor).toBeVisible();
  });

  test("cloud direct upload keeps inline progress through terminal dismissal", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [],
      history: [],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 0,
      triggerCalls: [],
      processingMode: "cloud_assisted",
      directUploadId: "cloud-upload",
      directUploadFilename: "cloud.pdf",
      extractionProgress: {
        total: 1,
        completed: 0,
        processing: 1,
        failed: 0,
        pending: 0,
        records_created: 0,
      },
      extractionFiles: [
        {
          id: "cloud-upload",
          filename: "cloud.pdf",
          ingestion_status: "processing",
        },
      ],
    };
    await mockBackend(page, state);
    await page.goto("/upload");

    await page.locator('input[type="file"]').first().setInputFiles({
      name: "cloud.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("%PDF-1.4 cloud mock"),
    });
    await page.getByRole("button", { name: /upload all/i }).click();
    await expect(page.getByText("Extracting clinical entities")).toBeVisible();

    state.extractionProgress = {
      total: 1,
      completed: 1,
      processing: 0,
      failed: 0,
      pending: 0,
      records_created: 4,
    };
    state.extractionFiles = [
      {
        id: "cloud-upload",
        filename: "cloud.pdf",
        ingestion_status: "completed",
      },
    ];
    await expect(page.getByText("Extraction complete")).toBeVisible({
      timeout: 8_000,
    });
    await expect(page.getByText("4 records created")).toBeVisible();
    await page.getByRole("button", { name: /^dismiss$/i }).click();
    await expect(page.getByText("Extraction complete")).toHaveCount(0);
  });

  test("direct strict upload correlates the durable job by upload_id", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [],
      history: [],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 0,
      triggerCalls: [],
      directUploadId: "strict-upload",
      directUploadFilename: "strict.pdf",
      jobAfterUpload: localJob("strict-job", "strict-upload"),
    };
    await mockBackend(page, state);
    await page.goto("/upload");

    await page.locator('input[type="file"]').first().setInputFiles({
      name: "strict.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("%PDF-1.4 strict mock"),
    });
    await page.getByRole("button", { name: /upload all/i }).click();
    const monitor = page.getByRole("region", {
      name: "Background processing",
    });
    await expect(monitor).toContainText("strict.pdf");
    expect(state.jobs).toHaveLength(1);
    expect(state.jobs[0].upload_id).toBe("strict-upload");
  });

  test("compacted batch uses accepted server filenames after a rejection", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [],
      history: [],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 0,
      triggerCalls: [],
      processingMode: "cloud_assisted",
      batchUploadResponse: {
        uploads: [
          {
            upload_id: "later-valid-id",
            filename: "later-valid.rtf",
            status: "pending_extraction",
            file_type: "rtf",
            manual_extraction_required: false,
          },
          {
            upload_id: "final-valid-id",
            filename: "final-valid.pdf",
            status: "pending_extraction",
            file_type: "pdf",
            manual_extraction_required: false,
          },
        ],
        rejected: [
          { filename: "first-rejected.rtf", code: "invalid_signature" },
        ],
        total: 2,
      },
    };
    await mockBackend(page, state);
    await page.goto("/upload");

    await page.locator('input[type="file"]').first().setInputFiles([
      {
        name: "first-rejected.rtf",
        mimeType: "application/rtf",
        buffer: Buffer.from("not an rtf"),
      },
      {
        name: "browser-second.rtf",
        mimeType: "application/rtf",
        buffer: Buffer.from("{\\rtf1 second}"),
      },
      {
        name: "browser-third.pdf",
        mimeType: "application/pdf",
        buffer: Buffer.from("%PDF-1.4 third"),
      },
    ]);
    await page.getByRole("button", { name: /upload all/i }).click();

    await expect(page.getByText("later-valid.rtf")).toHaveCount(2);
    await expect(page.getByText("final-valid.pdf")).toHaveCount(2);
    await expect(page.getByText("first-rejected.rtf")).toBeVisible();
    await expect(
      page.getByText("This file does not match its claimed format.")
    ).toBeVisible();
    await expect(page.getByText("browser-second.rtf")).toHaveCount(0);
    await expect(page.getByText("browser-third.pdf")).toHaveCount(0);
    expect(state.batchUploadCalls).toBe(1);
  });

  test("all rejected batch shows fixed guidance without starting progress", async ({
    page,
  }) => {
    await injectAuth(page);
    const state: MockState = {
      jobs: [],
      history: [],
      pollInFlight: 0,
      maxPollInFlight: 0,
      retryFailures: 0,
      cancelFailures: 0,
      triggerCalls: [],
      processingMode: "cloud_assisted",
      batchUploadResponse: {
        uploads: [],
        rejected: [
          { filename: "bad-name.rtf", code: "invalid_signature" },
          { filename: "too-large.pdf", code: "file_too_large" },
        ],
        total: 0,
        detail: "private resolver failure at /tmp/secret",
      },
    };
    await mockBackend(page, state);
    await page.goto("/upload");

    await page.locator('input[type="file"]').first().setInputFiles([
      {
        name: "bad-name.rtf",
        mimeType: "application/rtf",
        buffer: Buffer.from("not an rtf"),
      },
      {
        name: "too-large.pdf",
        mimeType: "application/pdf",
        buffer: Buffer.from("%PDF-1.4 mock"),
      },
    ]);
    await page.getByRole("button", { name: /upload all/i }).click();

    await expect(
      page.getByText(
        "No files were accepted. Choose a supported PDF, RTF, or TIFF and try again."
      )
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Upload results" })
    ).toBeVisible();
    await expect(
      page.getByText("This file does not match its claimed format.")
    ).toBeVisible();
    await expect(
      page.getByText("This file is larger than the upload limit.")
    ).toBeVisible();
    await expect(page.getByText("private resolver failure")).toHaveCount(0);
    await expect(page.getByText("/tmp/secret")).toHaveCount(0);
    await expect(
      page.getByRole("region", { name: "Extraction status" })
    ).toHaveCount(0);
    expect(state.batchUploadCalls).toBe(1);
    expect(state.extractionProgressCalls ?? 0).toBe(0);
  });
});
