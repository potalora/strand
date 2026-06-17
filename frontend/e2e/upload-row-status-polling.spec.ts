import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Bug #3 — frozen per-row extraction labels.
 *
 * The "N unstructured files detected" table renders each row's status from
 * `extractionStatuses[upload_id] || file.status`. Before the fix the poll loop
 * only refreshed the aggregate progress bar (`/upload/extraction-progress`) and
 * never the per-file statuses, so a row stayed frozen at its trigger-time status
 * (e.g. "processing") and never advanced to "completed".
 *
 * This drives the upload page with mocked backend responses so that across
 * successive poll ticks a file transitions processing -> completed, and asserts
 * the ROW's status text follows. Fails before the fix (row frozen), passes after.
 */
test.describe("Per-row extraction status tracks polling", () => {
  test.setTimeout(90_000);

  test("row status advances processing -> completed across poll ticks", async ({
    page,
  }) => {
    const api = new ApiClient();
    const email = testEmail("row-status-poll");
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    // Mutable state the route handlers read on each (re)fetch so we can flip the
    // backend's reported status mid-test, mimicking the worker finishing.
    let pendingStatus = "processing";
    let progressProcessing = 1;
    let progressCompleted = 0;

    const json = (body: unknown) => ({
      contentType: "application/json",
      body: JSON.stringify(body),
    });

    // Structured upload (.zip) returns an unstructured child that needs extraction.
    await page.route(
      (url) => url.pathname === "/api/v1/upload",
      (route) =>
        route.fulfill(
          json({
            upload_id: "mock-zip",
            status: "completed",
            records_inserted: 0,
            errors: [],
            unstructured_uploads: [
              {
                upload_id: "mock-u1",
                filename: "scan-note.pdf",
                status: "pending_extraction",
              },
            ],
          })
        )
    );

    // Triggering extraction reports the file as now processing.
    await page.route(
      (url) => url.pathname === "/api/v1/upload/trigger-extraction",
      (route) =>
        route.fulfill(
          json({
            triggered: 1,
            failed: 0,
            results: [{ upload_id: "mock-u1", status: "processing" }],
          })
        )
    );

    // Aggregate progress bar — reads current mutable counts.
    await page.route(
      (url) => url.pathname === "/api/v1/upload/extraction-progress",
      (route) =>
        route.fulfill(
          json({
            total: 1,
            completed: progressCompleted,
            processing: progressProcessing,
            failed: 0,
            pending: 0,
            records_created: progressCompleted > 0 ? 4 : 0,
          })
        )
    );

    // Per-file status list — reads current mutable status.
    await page.route(
      (url) => url.pathname === "/api/v1/upload/pending-extraction",
      (route) =>
        route.fulfill(
          json({
            files: [
              {
                id: "mock-u1",
                filename: "scan-note.pdf",
                mime_type: "application/pdf",
                file_category: "unstructured",
                file_size_bytes: 1234,
                created_at: null,
                ingestion_status: pendingStatus,
              },
            ],
            total: 1,
          })
        )
    );

    // History fetch (only on demand) — keep it inert.
    await page.route(
      (url) => url.pathname === "/api/v1/upload/history",
      (route) => route.fulfill(json({ items: [], total: 0 }))
    );

    await page.goto("/upload");
    await expect(
      page.getByRole("button", { name: /browse files/i })
    ).toBeVisible();

    // Select a structured .zip via the dropzone input (drag-only, but the hidden
    // input still accepts setInputFiles and fires react-dropzone's onDrop).
    const fileInput = page.locator('input[type="file"]').first();
    await fileInput.setInputFiles({
      name: "export.zip",
      mimeType: "application/zip",
      buffer: Buffer.from("PK mock zip payload"),
    });
    await expect(page.getByText("export.zip")).toBeVisible();

    await page.getByRole("button", { name: /upload all/i }).click();

    // The detected-files panel appears with the child doc, pending extraction.
    await expect(
      page.getByText(/unstructured file.*detected/i)
    ).toBeVisible({ timeout: 30_000 });
    const row = page.locator("tr", { hasText: "scan-note.pdf" });
    await expect(row).toBeVisible();

    // Select the row and trigger extraction -> row should show "processing".
    await row.locator('input[type="checkbox"]').check();
    await page.getByRole("button", { name: /extract 1 file/i }).click();
    await expect(row.getByText("processing", { exact: true })).toBeVisible({
      timeout: 10_000,
    });

    // Backend now reports the file finished. A poll tick must propagate this to
    // the ROW (not just the aggregate bar).
    pendingStatus = "completed";
    progressProcessing = 0;
    progressCompleted = 1;

    await expect(row.getByText("completed", { exact: true })).toBeVisible({
      timeout: 20_000,
    });
  });
});
