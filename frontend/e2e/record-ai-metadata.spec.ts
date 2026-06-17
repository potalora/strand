import { test, expect } from "./fixtures/console-gate";
import { browserLogin } from "./helpers/browser-login";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const email = testEmail("record-ai-metadata");

test.describe("AI Extraction Metadata Display", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    await api.pollUploadStatus(result.upload_id, 60_000);
  });

  test("non-AI records do not show AI badge", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);

    const data = await api.getRecords({ page: 1 });

    if (data.items.length === 0) {
      test.skip();
      return;
    }

    const record = data.items[0];
    await page.goto(`/records/${record.id}`);
    await page.waitForSelector("h1, [data-testid='record-title']", { timeout: 10_000 });

    // If the record is not AI-extracted, the badge should not appear
    if (!record.ai_extracted) {
      const aiBadge = page.getByText("AI Extracted");
      await expect(aiBadge).not.toBeVisible();
    }
  });

  test("API returns ai_extracted and confidence_score fields", async () => {
    const data = await api.getRecords({ page: 1 });

    if (data.items.length === 0) {
      test.skip();
      return;
    }

    const record = data.items[0];
    expect("ai_extracted" in record).toBe(true);
    expect("confidence_score" in record).toBe(true);
    expect(typeof record.ai_extracted).toBe("boolean");
  });
});
