import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

/**
 * Repaired for the "Reimagined" Timeline. Current IA:
 *  - Filter pills read "All", "Conditions", "Labs & vitals", "Medications", …
 *    (not "ALL"/"COND"/"OBS"); the active one carries aria-pressed.
 *  - Month group headers are Title-case "Jan 2024" (not "JAN 2024"); each group
 *    shows its own count (there is no global "N events" header).
 *  - Empty state text is "No events."; the detail drawer header is "Record detail".
 *  - Event cards are `button.tl-card`.
 */
test.describe("Timeline page", () => {
  const email = testEmail("timeline");

  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(
      PATHS.fhirBundle,
      "sample_fhir_bundle.json"
    );
    await api.pollUploadStatus(result.upload_id, 60_000);
  });

  test("events render grouped by month", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");
    const monthHeading = page.locator(".tl-grp-label").filter({
      hasText: /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$/,
    });
    await expect(monthHeading.first()).toBeVisible({ timeout: 15_000 });
  });

  test("filter pills render", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");
    await expect(
      page.getByRole("button", { name: "All", exact: true })
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByRole("button", { name: "Conditions", exact: true })
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Labs & vitals", exact: true })
    ).toBeVisible();
  });

  test("click filter narrows events and sets aria-pressed", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");

    const cards = page.locator("button.tl-card");
    await expect(cards.first()).toBeVisible({ timeout: 15_000 });
    const allCount = await cards.count();

    const conditions = page.getByRole("button", {
      name: "Conditions",
      exact: true,
    });
    await conditions.click();
    await expect(conditions).toHaveAttribute("aria-pressed", "true");

    // The filtered set is a subset of the full set.
    await expect(async () => {
      const filtered = await cards.count();
      expect(filtered).toBeLessThanOrEqual(allCount);
    }).toPass({ timeout: 10_000 });
  });

  test("click All restores the full set", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");

    const cards = page.locator("button.tl-card");
    await expect(cards.first()).toBeVisible({ timeout: 15_000 });
    const originalCount = await cards.count();

    await page.getByRole("button", { name: "Conditions", exact: true }).click();
    await page.waitForTimeout(1000);

    const all = page.getByRole("button", { name: "All", exact: true });
    await all.click();
    await expect(all).toHaveAttribute("aria-pressed", "true");
    await expect(async () => {
      expect(await cards.count()).toBe(originalCount);
    }).toPass({ timeout: 10_000 });
  });

  test("click event opens detail drawer", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");
    const card = page.locator("button.tl-card").first();
    await expect(card).toBeVisible({ timeout: 15_000 });
    await card.click();
    await expect(page.getByText("Record detail")).toBeVisible({
      timeout: 10_000,
    });
  });

  test("empty timeline shows no events message", async ({ page }) => {
    await page.route("**/api/v1/timeline**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ events: [], total: 0 }),
      })
    );

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");

    await expect(page.getByText("No events.")).toBeVisible({ timeout: 10_000 });
  });

  test("rows render the inline metric strip from preview, and omit it when null", async ({
    page,
  }) => {
    await page.route("**/api/v1/timeline**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          total: 2,
          events: [
            {
              id: "11111111-1111-1111-1111-111111111111",
              record_type: "observation",
              display_text: "Vitamin D",
              effective_date: "2026-02-28T00:00:00Z",
              code_display: "Vitamin D",
              category: ["laboratory"],
              provider: null,
              preview: {
                value: "17",
                unit: "ng/mL",
                flag: "LOW",
                emphasis: "notable",
                gauge: { value: 17, low: 30, high: 100 },
                facets: [],
              },
            },
            {
              id: "22222222-2222-2222-2222-222222222222",
              record_type: "document",
              display_text: "Visit note",
              effective_date: "2026-02-27T00:00:00Z",
              code_display: null,
              category: ["document"],
              provider: null,
              preview: null,
            },
          ],
        }),
      })
    );

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");

    // Lab row: metric strip with mono value, neutral notable flag, and gauge.
    const labCard = page.locator("button.tl-card", { hasText: "Vitamin D" });
    await expect(labCard).toBeVisible({ timeout: 10_000 });
    await expect(labCard.locator(".tl-ms-val")).toContainText("17");
    await expect(
      labCard.locator('.tl-ms-flag[data-emphasis="notable"]')
    ).toHaveText("LOW");
    await expect(labCard.locator(".tl-ms-gauge .gauge")).toBeVisible();

    // Document row (preview null): no metric strip at all.
    const docCard = page.locator("button.tl-card", { hasText: "Visit note" });
    await expect(docCard).toBeVisible();
    await expect(docCard.locator(".tl-ms")).toHaveCount(0);
  });
});
