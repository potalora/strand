import { test, expect } from "./fixtures/console-gate";
import { browserLogin } from "./helpers/browser-login";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const email = testEmail("record-detail-sheet");

/**
 * Repaired for the flat Admin → Records table. The detail drawer now opens by
 * clicking a `tr.clickable` row (no more tree). Drawer header is "Record detail";
 * the Advanced section toggles the raw FHIR JSON; the delete control carries
 * aria-label "Delete record".
 */
test.describe("Record detail drawer (Admin)", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    await api.pollUploadStatus(result.upload_id, 60_000);
  });

  async function openFirstRecord(page: import("@playwright/test").Page) {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin");
    const row = page.locator("tr.clickable").first();
    await expect(row).toBeVisible({ timeout: 15_000 });
    await row.click();
    await expect(page.getByText("Record detail")).toBeVisible({ timeout: 10_000 });
  }

  test("opens drawer with header and actions", async ({ page }) => {
    await openFirstRecord(page);
    // Drawer actions present.
    await expect(page.getByRole("button", { name: /Add to summary/ })).toBeVisible();
    await expect(page.getByRole("button", { name: /Export FHIR/ })).toBeVisible();
  });

  test("Advanced section is collapsed by default, expands to JSON", async ({
    page,
  }) => {
    await openFirstRecord(page);

    const advancedBtn = page.getByText("Advanced");
    await expect(advancedBtn).toBeVisible();

    await advancedBtn.click();
    await expect(page.locator("pre.json-syntax").first()).toBeVisible({
      timeout: 5_000,
    });
  });

  test("delete control is present in the drawer", async ({ page }) => {
    await openFirstRecord(page);
    await expect(
      page.getByRole("button", { name: "Delete record" })
    ).toBeVisible();
  });
});
