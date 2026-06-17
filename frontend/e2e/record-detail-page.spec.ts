import { test, expect } from "./fixtures/console-gate";
import { browserLogin } from "./helpers/browser-login";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const email = testEmail("record-detail-page");

test.describe("Record Detail Page (/records/[id])", () => {
  const api = new ApiClient();
  let recordsByType: Record<string, string[]> = {};

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    await api.pollUploadStatus(result.upload_id, 60_000);

    // Fetch all records grouped by type
    const data = await api.getRecords({ page: 1 });
    for (const item of data.items) {
      const type = item.record_type;
      if (!recordsByType[type]) recordsByType[type] = [];
      recordsByType[type].push(item.id);
    }
  });

  test("renders record with breadcrumb and icon", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    const types = Object.keys(recordsByType);
    const recordId = recordsByType[types[0]][0];

    await page.goto(`/records/${recordId}`);
    await page.waitForSelector("h1, [data-testid='record-title']", { timeout: 10_000 });

    // Breadcrumb
    await expect(page.getByText("Back to records")).toBeVisible();
  });

  test("has Advanced section that toggles JSON", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    const types = Object.keys(recordsByType);
    const recordId = recordsByType[types[0]][0];

    await page.goto(`/records/${recordId}`);
    await page.waitForSelector("h1, [data-testid='record-title']", { timeout: 10_000 });

    // Advanced button should exist
    const advancedBtn = page.getByText("Advanced");
    await expect(advancedBtn).toBeVisible();

    // JSON not visible initially
    const jsonPre = page.locator("pre.json-syntax");
    await expect(jsonPre).not.toBeVisible();

    // Click to expand
    await advancedBtn.click();
    await expect(jsonPre).toBeVisible();
  });
});
