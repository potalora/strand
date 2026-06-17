import { test, expect, type Page } from "./fixtures/console-gate";

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
}

function file(id: string, filename: string, status: string): MockFile {
  return { id, filename, ingestion_status: status };
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
async function mockExtractions(page: Page, getFiles: () => MockFile[]): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    const json = (body: unknown) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

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
      }));
      return json({ files, total: files.length });
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
