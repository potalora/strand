import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Graceful-degradation under backend failure. Repaired for the "Reimagined" IA:
 * the Overview (home) page no longer calls /dashboard/overview — it loads
 * /observations/by-code, /records, /records/recent, /records/stats,
 * /dashboard/patients and /auth/me — so we fail those instead. The masthead
 * ("Personal Health Record") always renders, even when every data call fails.
 *
 * The console-error gate is also active here: intentional 500s / aborts surface
 * as browser "Failed to load resource" / "net::ERR_" noise (allowlisted), so any
 * real uncaught error or app-level console.error would still fail the test.
 */
test.describe("Error handling — graceful degradation", () => {
  const email = testEmail("errors");

  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
  });

  test("500 on home data endpoints renders gracefully", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    // Fail every data endpoint the Overview page loads.
    for (const ep of [
      "observations/by-code",
      "records",
      "records/recent",
      "records/stats",
      "dashboard/patients",
    ]) {
      await page.route(`**/api/v1/${ep}*`, (route) =>
        route.fulfill({ status: 500, body: "Internal Server Error" })
      );
    }

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // The masthead + navigation still render despite the failed data calls.
    await expect(page.locator("nav")).toBeVisible();
    await expect(page.getByText("Personal Health Record")).toBeVisible();
  });

  test("500 on timeline shows empty state", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    await page.route("**/api/v1/timeline*", (route) =>
      route.fulfill({ status: 500, body: "Internal Server Error" })
    );

    await page.goto("/timeline");
    await page.waitForLoadState("networkidle");

    // Page loads without crashing — nav is present and the empty state shows.
    await expect(page.locator("nav")).toBeVisible();
    await expect(page.getByText("No events.")).toBeVisible({ timeout: 10_000 });

    // No stack traces leak into the DOM.
    const bodyText = await page.locator("body").textContent();
    expect(bodyText).not.toContain("Traceback");
    expect(bodyText).not.toContain("at Object.");
    expect(bodyText).not.toContain("TypeError");
  });

  test("network error on summary generate shows page gracefully", async ({
    page,
  }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    // Aborting the (on-demand) generate call must not crash page load.
    await page.route("**/api/v1/summary/generate", (route) => route.abort());

    await page.goto("/summaries");
    await page.waitForLoadState("networkidle");

    await expect(page.locator("nav")).toBeVisible();
  });

  test("API error does not expose stack traces / raw detail", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    // Return a JSON error with a raw detail on every home data endpoint.
    for (const ep of ["records/stats", "observations/by-code", "records"]) {
      await page.route(`**/api/v1/${ep}*`, (route) =>
        route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Internal Server Error" }),
        })
      );
    }

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // The raw error detail must NOT be surfaced to the user.
    const bodyText = await page.locator("body").textContent();
    expect(bodyText).not.toContain("Internal Server Error");

    await expect(page.locator("nav")).toBeVisible();
  });
});
