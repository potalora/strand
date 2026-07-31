import { test, expect, type Page } from "./fixtures/console-gate";
import type { TriggerExtractionResponse } from "../src/types/api";
import type { LocalAIJobResponse } from "../src/types/local-ai";

/**
 * Extraction terminal-state invariants (bugs #2 / #3).
 *
 * The stuck-PDF bug left a failing document in `processing` forever (the poisoned
 * session was never rolled back, so the `failed` write never landed). The suite
 * never asserted the invariant that a *failing* document must reach `failed`, and
 * that `processing` is transient — "stuck in processing" was indistinguishable
 * from "slow". These tests drive Admin → Extractions with mocked status endpoints
 * to assert:
 *   • a failed document surfaces as `Failed` (and `failed` is terminal);
 *   • `processing` is NOT terminal — an erroring doc advances processing → failed;
 *   • a completing doc leaves the pending/processing/failed queue entirely.
 *
 * Fully mocked + auth injected → parallel-safe, no rate limiter.
 */

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const ME_OK = {
  id: "11111111-2222-3333-4444-555555555555",
  email: "pedro@example.com",
  display_name: "Pedro",
  is_active: true,
  created_at: "2024-01-01T00:00:00Z",
};

interface MockFile {
  id: string;
  filename: string;
  ingestion_status: string;
  manual_extraction_required?: boolean;
}

interface ExtractionMockOptions {
  jobsAfterTrigger?: LocalAIJobResponse[];
  triggerResponse?: TriggerExtractionResponse;
}

function file(id: string, filename: string, status: string): MockFile {
  return { id, filename, ingestion_status: status };
}

function localJob(id: string, uploadId: string): LocalAIJobResponse {
  return {
    id,
    upload_id: uploadId,
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
  };
}

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

/**
 * Mock the Extractions tab's data. `getFiles` is read on every fetch so a test
 * can flip the backend's reported state and re-fetch via the Refresh button.
 */
async function mockExtractions(
  page: Page,
  getFiles: () => MockFile[],
  options: ExtractionMockOptions = {}
): Promise<void> {
  let triggered = false;
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    const path = new URL(url).pathname;
    const json = (body: unknown) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    // Keep explicit local-job handling ahead of this broad mock's fallback.
    if (path === "/api/v1/local-ai/jobs")
      return json(triggered ? (options.jobsAfterTrigger ?? []) : []);
    const localJobMatch = path.match(/^\/api\/v1\/local-ai\/jobs\/([^/]+)$/);
    if (localJobMatch) {
      return json(
        options.jobsAfterTrigger?.find(
          (job) => job.id === decodeURIComponent(localJobMatch[1])
        )
      );
    }
    if (path === "/api/v1/upload/history")
      return json({ items: [], total: 0 });
    if (url.includes("/auth/me")) return json(ME_OK);
    if (url.includes("/auth/refresh"))
      return json({ access_token: "fresh", refresh_token: "fresh" });
    if (url.includes("/upload/pending-extraction")) {
      const files = getFiles().map((f) => ({
        id: f.id,
        filename: f.filename,
        mime_type: "application/pdf",
        file_category: "unstructured",
        file_size_bytes: 2048,
        created_at: "2024-01-01T00:00:00Z",
        ingestion_status: f.ingestion_status,
        manual_extraction_required: f.manual_extraction_required ?? false,
      }));
      return json({ files, total: files.length });
    }
    if (path === "/api/v1/upload/trigger-extraction") {
      const body = route.request().postDataJSON() as { upload_ids: string[] };
      triggered = true;
      return json(
        options.triggerResponse ?? {
          triggered: body.upload_ids.length,
          failed: 0,
          results: body.upload_ids.map((upload_id) => ({
            upload_id,
            status: "pending_extraction",
          })),
        }
      );
    }
    return json({});
  });
}

test.describe("Extraction terminal-state (Admin → Extractions)", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
  });

  test("failed and processing both surface in the queue", async ({ page }) => {
    await mockExtractions(page, () => [
      file("u1", "slow-scan.pdf", "processing"),
      file("u2", "broken-scan.pdf", "failed"),
    ]);
    await page.goto("/admin?tab=extractions");

    const procRow = page.locator("tr", { hasText: "slow-scan.pdf" });
    const failRow = page.locator("tr", { hasText: "broken-scan.pdf" });
    await expect(procRow.getByText("Processing")).toBeVisible({ timeout: 10_000 });
    await expect(failRow.getByText("Failed")).toBeVisible();

    // Toolbar count chips reflect both states.
    await expect(page.getByText("1 processing")).toBeVisible();
    await expect(page.getByText("1 failed")).toBeVisible();
    await expect(
      procRow.getByRole("checkbox", { name: /select slow-scan/i })
    ).toBeDisabled();
    await expect(
      failRow.getByRole("checkbox", { name: /select broken-scan/i })
    ).toBeDisabled();
  });

  test("only durable manual ZIP children are selectable", async ({ page }) => {
    await mockExtractions(
      page,
      () => [
        file("u1", "direct.pdf", "pending_extraction"),
        {
          ...file("u2", "zip-child.pdf", "pending_extraction"),
          manual_extraction_required: true,
        },
      ],
      { jobsAfterTrigger: [localJob("job-u2", "u2")] }
    );
    await page.goto("/admin?tab=extractions");

    const direct = page.locator("tr", { hasText: "direct.pdf" });
    const manual = page.locator("tr", { hasText: "zip-child.pdf" });
    await expect(
      direct.getByRole("checkbox", { name: /select direct/i })
    ).toBeDisabled();
    await expect(
      manual.getByRole("checkbox", { name: /select zip-child/i })
    ).toBeEnabled();
    await page.getByRole("checkbox", { name: "Select all files" }).check();
    await expect(
      manual.getByRole("checkbox", { name: /select zip-child/i })
    ).toBeChecked();
    await expect(page.getByRole("button", { name: "Extract 1" })).toBeEnabled();
    await page.getByRole("button", { name: "Extract 1" }).click();
    const monitor = page.getByRole("region", {
      name: "Background processing",
    });
    await monitor
      .getByRole("button", { name: /background processing/i })
      .click();
    await expect(monitor).toContainText("zip-child.pdf");
    await page.goto("/timeline");
    await expect(monitor).toBeVisible();
  });

  test("manual extraction application failures keep the selection actionable", async ({
    page,
  }) => {
    await mockExtractions(
      page,
      () => [
        {
          ...file("u1", "zip-child.pdf", "pending_extraction"),
          manual_extraction_required: true,
        },
      ],
      {
        triggerResponse: {
          triggered: 0,
          failed: 1,
          results: [{ upload_id: "u1", status: "manual_extraction_required" }],
        },
      }
    );
    await page.goto("/admin?tab=extractions");

    await page
      .getByRole("checkbox", { name: /select zip-child/i })
      .check();
    await page.getByRole("button", { name: "Extract 1" }).click();

    await expect(
      page.getByRole("alert").filter({
        hasText: "1 selected file could not be started.",
      })
    ).toBeVisible();
    await expect(
      page.getByRole("checkbox", { name: /select zip-child/i })
    ).toBeChecked();
    await expect(page.getByRole("button", { name: "Extract 1" })).toBeEnabled();
  });

  test("an erroring document advances processing → failed (not stuck)", async ({
    page,
  }) => {
    let files = [file("u1", "poison-date.pdf", "processing")];
    await mockExtractions(page, () => files);
    await page.goto("/admin?tab=extractions");

    const row = page.locator("tr", { hasText: "poison-date.pdf" });
    await expect(row.getByText("Processing")).toBeVisible({ timeout: 10_000 });

    // Backend rolls back + marks the poisoned file failed (the #2 fix). Refresh.
    files = [file("u1", "poison-date.pdf", "failed")];
    await page.getByRole("button", { name: "Refresh" }).click();

    await expect(row.getByText("Failed")).toBeVisible({ timeout: 10_000 });
    // It is no longer stuck in processing.
    await expect(page.getByText("1 processing")).toHaveCount(0);
    await expect(page.getByText("1 failed")).toBeVisible();
  });

  test("a completing document leaves the pending/processing/failed queue", async ({
    page,
  }) => {
    let files = [file("u1", "good-scan.pdf", "processing")];
    await mockExtractions(page, () => files);
    await page.goto("/admin?tab=extractions");

    const row = page.locator("tr", { hasText: "good-scan.pdf" });
    await expect(row.getByText("Processing")).toBeVisible({ timeout: 10_000 });

    // It completes → drops out of the (pending/processing/failed) queue.
    files = [];
    await page.getByRole("button", { name: "Refresh" }).click();

    await expect(page.getByText(/Nothing waiting/i)).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("tr", { hasText: "good-scan.pdf" })).toHaveCount(0);
  });
});
