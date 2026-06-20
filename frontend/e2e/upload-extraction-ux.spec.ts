import { test, expect, type Page } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Upload / extraction UX — session §2a (i)–(iv).
 *
 * Pattern mirrors `upload-row-status-polling.spec.ts`: a REAL login (so auth/me,
 * nav data, and other pages hit the real backend) with the upload + extraction
 * endpoints deterministically MOCKED via `page.route`, so we control the batch's
 * progress timeline without waiting on a live LLM extract.
 *
 *   (i)   terminal "all done" + a working Dismiss
 *   (ii)  a new upload while the prior is in-flight ACCUMULATES rows (WS-U; the
 *         prior in-flight batch is not dropped)
 *   (iii) progress is scoped to the current batch (`?ids=…`), not user-global
 *   (iv)  a global, cross-page status bar with section detail + Cancel
 *
 * NOTE: written for the lead's live integration pass; not run here (no server).
 */

const json = (body: unknown) => ({
  contentType: "application/json",
  body: JSON.stringify(body),
});

/** Mutable backend the route handlers read each (re)fetch, so a test can flip it. */
interface MockBackend {
  unstructuredIds: string[]; // returned in order, one per /upload/unstructured POST
  nextIdx: number;
  progress: {
    total: number;
    completed: number;
    processing: number;
    failed: number;
    pending: number;
    records_created: number;
  };
  pendingFiles: {
    id: string;
    filename: string;
    ingestion_status: string;
    progress_stage?: string | null;
    progress_detail?: { section_index: number; section_total: number } | null;
  }[];
  progressUrls: string[];
  cancelCalls: string[][];
}

async function mockUpload(page: Page, be: MockBackend): Promise<void> {
  // Single direct unstructured upload → returns the next pre-seeded id.
  await page.route(
    (url) => url.pathname === "/api/v1/upload/unstructured",
    (route) => {
      const id = be.unstructuredIds[be.nextIdx] ?? `u${be.nextIdx}`;
      be.nextIdx += 1;
      return route.fulfill(
        json({ upload_id: id, status: "pending_extraction", file_type: "pdf" })
      );
    }
  );

  // Scoped aggregate progress. Record the URL so a test can assert `?ids=…`.
  await page.route(
    (url) => url.pathname === "/api/v1/upload/extraction-progress",
    (route) => {
      be.progressUrls.push(route.request().url());
      return route.fulfill(json(be.progress));
    }
  );

  // Per-file status (carries optional section progress).
  await page.route(
    (url) => url.pathname === "/api/v1/upload/pending-extraction",
    (route) =>
      route.fulfill(
        json({
          files: be.pendingFiles.map((f) => ({
            id: f.id,
            filename: f.filename,
            mime_type: "application/pdf",
            file_category: "unstructured",
            file_size_bytes: 2048,
            created_at: null,
            ingestion_status: f.ingestion_status,
            progress_stage: f.progress_stage ?? null,
            progress_detail: f.progress_detail ?? null,
          })),
          total: be.pendingFiles.length,
        })
      )
  );

  // Cancel — record the ids, then the test flips the file to `cancelled`.
  await page.route(
    (url) => url.pathname === "/api/v1/upload/cancel",
    (route) => {
      const body = route.request().postDataJSON() as { upload_ids: string[] };
      be.cancelCalls.push(body.upload_ids);
      return route.fulfill(json({ cancelled: body.upload_ids, skipped: [] }));
    }
  );

  await page.route(
    (url) => url.pathname === "/api/v1/upload/history",
    (route) => route.fulfill(json({ items: [], total: 0 }))
  );
}

async function selectAndUpload(page: Page, filename: string): Promise<void> {
  const fileInput = page.locator('input[type="file"]').first();
  await fileInput.setInputFiles({
    name: filename,
    mimeType: "application/pdf",
    buffer: Buffer.from(`%PDF-1.4 mock ${filename}`),
  });
  await expect(page.getByText(filename)).toBeVisible();
  await page.getByRole("button", { name: /upload all/i }).click();
}

test.describe("Upload/extraction UX (§2a)", () => {
  test.setTimeout(90_000);

  test("(i,iii) progress is scoped to the batch, reaches a terminal state, and dismisses", async ({
    page,
  }) => {
    const api = new ApiClient();
    const email = testEmail("ux-terminal");
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    const be: MockBackend = {
      unstructuredIds: ["u1"],
      nextIdx: 0,
      progress: {
        total: 1,
        completed: 0,
        processing: 1,
        failed: 0,
        pending: 0,
        records_created: 0,
      },
      pendingFiles: [{ id: "u1", filename: "scan.pdf", ingestion_status: "processing" }],
      progressUrls: [],
      cancelCalls: [],
    };
    await mockUpload(page, be);

    await page.goto("/upload");
    await selectAndUpload(page, "scan.pdf");

    // Card shows the in-flight batch — "0 of 1", NOT a user-global count.
    await expect(page.getByText("Extracting clinical entities")).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText(/0 of 1 file processed/i)).toBeVisible();

    // (iii) the aggregate request was scoped to this batch's id.
    await expect
      .poll(() => be.progressUrls.some((u) => /[?&]ids=u1\b/.test(u)), {
        timeout: 10_000,
      })
      .toBe(true);

    // Worker finishes → terminal "all done".
    be.progress = {
      total: 1,
      completed: 1,
      processing: 0,
      failed: 0,
      pending: 0,
      records_created: 5,
    };
    be.pendingFiles = [{ id: "u1", filename: "scan.pdf", ingestion_status: "completed" }];

    // Scope to the card's title — the global bar's collapsed pill also reads
    // "Extraction complete", so an unscoped getByText would match two nodes.
    const cardComplete = page.locator("h3.sec-title", {
      hasText: "Extraction complete",
    });
    await expect(cardComplete).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText("5 records created")).toBeVisible();

    // (i) Dismiss actually clears the panel.
    await page.getByRole("button", { name: /^dismiss$/i }).first().click();
    await expect(cardComplete).toHaveCount(0);
  });

  test("(ii) a new upload while the prior is in-flight accumulates rows", async ({ page }) => {
    const api = new ApiClient();
    const email = testEmail("ux-reset");
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    const be: MockBackend = {
      unstructuredIds: ["uA", "uB"],
      nextIdx: 0,
      progress: {
        total: 1,
        completed: 0,
        processing: 1,
        failed: 0,
        pending: 0,
        records_created: 0,
      },
      pendingFiles: [{ id: "uA", filename: "first.pdf", ingestion_status: "processing" }],
      progressUrls: [],
      cancelCalls: [],
    };
    await mockUpload(page, be);

    await page.goto("/upload");
    await selectAndUpload(page, "first.pdf");
    await expect(page.getByText("Extracting clinical entities")).toBeVisible({
      timeout: 15_000,
    });
    // first.pdf appears in the per-file rows of the progress card.
    await expect(
      page.locator("tr", { hasText: "first.pdf" }).first()
    ).toBeVisible();

    // A second upload while first.pdf is still processing: WS-U accumulates the
    // in-flight batch — the prior row is NOT dropped (the fix for the bug where the
    // status pane under-counted concurrent uploads vs the Admin extractions pane).
    be.progress = { total: 2, completed: 0, processing: 2, failed: 0, pending: 0, records_created: 0 };
    be.pendingFiles = [
      { id: "uA", filename: "first.pdf", ingestion_status: "processing" },
      { id: "uB", filename: "second.pdf", ingestion_status: "processing" },
    ];
    await selectAndUpload(page, "second.pdf");

    await expect(
      page.locator("tr", { hasText: "second.pdf" }).first()
    ).toBeVisible({ timeout: 15_000 });
    // The prior batch's row is STILL shown — accumulated, not replaced (WS-U).
    await expect(page.locator("tr", { hasText: "first.pdf" }).first()).toBeVisible();
  });

  test("(iv) global status bar shows section detail, persists across pages, and cancels", async ({
    page,
  }) => {
    const api = new ApiClient();
    const email = testEmail("ux-globalbar");
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    const be: MockBackend = {
      unstructuredIds: ["u1"],
      nextIdx: 0,
      progress: {
        total: 1,
        completed: 0,
        processing: 1,
        failed: 0,
        pending: 0,
        records_created: 0,
      },
      pendingFiles: [
        {
          id: "u1",
          filename: "long-note.pdf",
          ingestion_status: "processing",
          progress_stage: "extracting_entities",
          progress_detail: { section_index: 3, section_total: 8 },
        },
      ],
      progressUrls: [],
      cancelCalls: [],
    };
    await mockUpload(page, be);

    await page.goto("/upload");
    await selectAndUpload(page, "long-note.pdf");

    const bar = page.locator('section[aria-label="Extraction status"]');
    await expect(bar).toBeVisible({ timeout: 15_000 });

    // Hover to expand → section-level detail for the long LLM extract.
    await bar.hover();
    await expect(bar.getByText("Extracting entities — section 3 of 8")).toBeVisible({
      timeout: 10_000,
    });

    // It is GLOBAL: a client-side nav (FloatingDock → Summarize) keeps it mounted.
    await page.getByRole("button", { name: /summarize/i }).click();
    await expect(page).toHaveURL(/\/summaries/);
    await expect(bar).toBeVisible();

    // Cancel from the bar → POST /upload/cancel, file lands `cancelled`.
    be.pendingFiles = [
      { id: "u1", filename: "long-note.pdf", ingestion_status: "cancelled" },
    ];
    be.progress = {
      total: 1,
      completed: 0,
      processing: 0,
      failed: 0,
      pending: 0,
      records_created: 0,
    };
    await bar.hover();
    await bar.getByRole("button", { name: /cancel all/i }).click();

    await expect.poll(() => be.cancelCalls.length, { timeout: 10_000 }).toBeGreaterThan(0);
    expect(be.cancelCalls[0]).toContain("u1");
    await expect(bar.getByText("Extraction cancelled")).toBeVisible({ timeout: 15_000 });
  });
});
