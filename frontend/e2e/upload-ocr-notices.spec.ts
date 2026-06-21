import { test, expect, type Page } from "./fixtures/console-gate";

/**
 * OCR provider notices in the upload history (fully mocked — no real backend,
 * parallel-safe). The backend adds a per-file `notices: Notice[]` to each
 * uploaded-file status; the frontend renders one compact line per notice under
 * the file row (info = muted, warning = warning style). See
 * docs/superpowers/specs/2026-06-21-ocr-provider-notices-design.md.
 *
 * Auth is injected straight into localStorage (the persisted zustand shape) so
 * the dashboard layout authenticates without hitting the login rate limiter,
 * and every API call is stubbed with page.route for determinism.
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

const FALLBACK_MSG =
  "Read by Anthropic — Gemini declined this document (content policy).";
const UNREADABLE_MSG =
  "This document couldn't be read by any configured provider.";

// Upload history with a completed file carrying both notice levels, plus a
// clean file (no notices) to prove the null-safe path renders nothing.
const HISTORY = {
  items: [
    {
      id: "aaaa1111-2222-3333-4444-555555555555",
      filename: "scan-note.pdf",
      ingestion_status: "completed",
      records_inserted: 4,
      created_at: "2024-02-01T00:00:00Z",
      file_category: "unstructured",
      record_count: 4,
      ingestion_progress: { records_inserted: 4 },
      notices: [
        {
          type: "ocr_fallback",
          level: "info",
          message: FALLBACK_MSG,
          detail: {
            used: "anthropic",
            refused: ["gemini"],
            attempts: [
              { provider: "gemini", status: "refused" },
              { provider: "anthropic", status: "ok" },
            ],
          },
        },
        {
          type: "ocr_unreadable",
          level: "warning",
          message: UNREADABLE_MSG,
        },
      ],
    },
    {
      id: "bbbb1111-2222-3333-4444-555555555555",
      filename: "labs.json",
      ingestion_status: "completed",
      records_inserted: 9,
      created_at: "2024-02-02T00:00:00Z",
      file_category: "structured",
      record_count: 9,
      ingestion_progress: { records_inserted: 9 },
      notices: [],
    },
  ],
  total: 2,
};

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

async function mockBackend(page: Page): Promise<void> {
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.includes("/auth/me")) return json(ME_OK);
    if (url.includes("/auth/refresh"))
      return json({
        access_token: "fresh.access.token",
        refresh_token: "fresh.refresh.token",
      });
    if (url.includes("/upload/history")) return json(HISTORY);
    // The upload page's batch progress bar is inert (no active batch) but be safe.
    if (url.includes("/upload/extraction-progress"))
      return json({
        total: 0,
        completed: 0,
        processing: 0,
        failed: 0,
        pending: 0,
        records_created: 0,
      });
    if (url.includes("/upload/pending-extraction"))
      return json({ files: [], total: 0 });
    return json({});
  });
}

test.describe("OCR provider notices in upload history", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
    await mockBackend(page);
  });

  test("renders an info fallback notice and a warning unreadable notice", async ({
    page,
  }) => {
    await page.goto("/upload");

    // Open the collapsible Upload history section (lazy-loads on open).
    await page.getByRole("button", { name: "Upload history" }).click();

    const row = page.locator("tr", { hasText: "scan-note.pdf" });
    await expect(row).toBeVisible({ timeout: 10_000 });

    // Both notices render their message under the file row.
    await expect(row.getByText(FALLBACK_MSG)).toBeVisible();
    await expect(row.getByText(UNREADABLE_MSG)).toBeVisible();

    // Levels map to the right notice type (info=fallback, warning=unreadable).
    await expect(
      row.locator('[data-ocr-notice="ocr_fallback"]')
    ).toHaveCount(1);
    await expect(
      row.locator('[data-ocr-notice="ocr_unreadable"]')
    ).toHaveCount(1);

    // The clean file (notices: []) renders no notice lines — null-safe path.
    const cleanRow = page.locator("tr", { hasText: "labs.json" });
    await expect(cleanRow).toBeVisible();
    await expect(cleanRow.locator("[data-ocr-notice]")).toHaveCount(0);
  });
});
