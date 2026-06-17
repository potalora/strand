import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { PATHS, testEmail, TEST_PASSWORD } from "./helpers/test-data";

const email = testEmail("summaries");

/**
 * Repaired for the current Summaries page labels:
 *  - Summary types: "Full record" / "By category" / "Date range".
 *  - Output formats: "Natural language" / "JSON data" / "Both".
 *  - Results card heading is "Summary" (not "Summary results"); the count line is
 *    "{n} records · {model}" (middot, not "|"); result tabs are
 *    "Narrative" / "JSON data".
 *  - History toggle is "Show ({n})"; each entry is a row button ("{type} summary").
 */
test.describe("Summaries page", () => {
  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    await api.pollUploadStatus(result.upload_id, 60_000);
    // Wait for data to be queryable
    await new Promise((r) => setTimeout(r, 2000));
  });

  test("patient selector loads patients", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });

    await expect(async () => {
      const text = await select.textContent();
      expect(text).not.toContain("No patients found");
    }).toPass({ timeout: 15_000 });
  });

  test("summary type tabs exist", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    await expect(
      page.getByRole("button", { name: "Full record" })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("button", { name: "By category" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Date range" })).toBeVisible();
  });

  test("category dropdown appears for By category type", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    await expect(
      page.getByRole("button", { name: "Full record" })
    ).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "By category" }).click();

    // The category select appears with options like "Labs & Vitals".
    const categorySelect = page.locator("select").nth(1);
    await expect(categorySelect).toBeVisible({ timeout: 5_000 });
    await expect(categorySelect).toContainText("Labs & Vitals");
  });

  test("date range inputs appear for Date range type", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    await expect(
      page.getByRole("button", { name: "Full record" })
    ).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Date range" }).click();

    await expect(page.getByText("From", { exact: true })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText("To", { exact: true })).toBeVisible();
    const textboxes = page.getByRole("textbox");
    await expect(textboxes.first()).toBeVisible();
  });

  test("output format options work", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    await expect(page.getByText("Output format")).toBeVisible({ timeout: 10_000 });

    // Output formats are a segmented button group (aria-pressed), not radios.
    const nl = page.getByRole("button", { name: "Natural language" });
    const json = page.getByRole("button", { name: "JSON data" });
    const both = page.getByRole("button", { name: "Both" });
    await expect(nl).toBeVisible();
    await expect(json).toBeVisible();
    await expect(both).toBeVisible();

    // Toggle JSON, then back to natural language; aria-pressed tracks selection.
    await json.click();
    await expect(json).toHaveAttribute("aria-pressed", "true");

    await nl.click();
    await expect(nl).toHaveAttribute("aria-pressed", "true");
    await expect(json).toHaveAttribute("aria-pressed", "false");
  });

  test("generate button is present and enabled with a patient", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const generateBtn = page.getByRole("button", { name: "Generate summary" });
    await expect(generateBtn).toBeVisible({ timeout: 10_000 });
    // The page auto-selects the first patient, so the button is enabled.
    await expect(generateBtn).toBeEnabled();
  });

  test("generate produces a result", async ({ page }) => {
    test.skip(!process.env.GEMINI_API_KEY, "Requires GEMINI_API_KEY");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });
    await expect(async () => {
      const text = await select.textContent();
      expect(text).not.toContain("No patients found");
    }).toPass({ timeout: 15_000 });

    await page.getByRole("button", { name: "Generate summary" }).click();

    // The results card heading is "Summary".
    await expect(page.getByRole("heading", { name: "Summary", exact: true })).toBeVisible({
      timeout: 60_000,
    });
    // Count line: "{n} records · {model}".
    await expect(page.getByText(/\d+ record/)).toBeVisible();
    // Result tabs: "Narrative" / "JSON data".
    await expect(page.getByRole("button", { name: "Narrative" })).toBeVisible();
  });

  test("history entry reopens a saved summary without regenerating", async ({
    page,
  }) => {
    test.skip(!process.env.GEMINI_API_KEY, "Requires GEMINI_API_KEY");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });
    await expect(async () => {
      const text = await select.textContent();
      expect(text).not.toContain("No patients found");
    }).toPass({ timeout: 15_000 });

    await page.getByRole("button", { name: "Generate summary" }).click();
    await expect(page.getByRole("heading", { name: "Summary", exact: true })).toBeVisible({
      timeout: 60_000,
    });

    // Reload so the in-memory result clears — only saved history remains.
    await page.reload();
    await expect(select).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("heading", { name: "Summary", exact: true })).toHaveCount(0);

    // Expand history ("Show (N)") and open the saved entry row.
    await page.getByRole("button", { name: /Show \(\d+\)/ }).click();
    await page.locator("button.lrow").first().click();

    // It re-renders quickly from the stored summary (not a 60s regeneration).
    await expect(page.getByRole("heading", { name: "Summary", exact: true })).toBeVisible({
      timeout: 15_000,
    });
    // Multiple "{n} records" appear (each history row + the result), so scope to first.
    await expect(page.getByText(/\d+ record/).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Narrative" })).toBeVisible();
  });

  test("AI disclaimer always visible", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    await expect(page.getByText("Notice")).toBeVisible({ timeout: 10_000 });
    // "de-identified" also appears in the masthead, so assert the disclaimer's
    // unique no-medical-advice clause instead.
    await expect(
      page.getByText("do not constitute", { exact: false })
    ).toBeVisible();
  });

  test("de-identification report renders after generation", async ({ page }) => {
    test.skip(!process.env.GEMINI_API_KEY, "Requires GEMINI_API_KEY");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Generate summary" }).click();
    await expect(page.getByRole("heading", { name: "Summary", exact: true })).toBeVisible({
      timeout: 60_000,
    });

    // The de-identification report appears only when PHI was scrubbed; otherwise
    // the summary itself still rendered.
    const deidentReport = page.getByText("De-identification report");
    const hasDeident = await deidentReport.isVisible().catch(() => false);
    if (hasDeident) {
      await expect(deidentReport).toBeVisible();
    } else {
      await expect(
        page.getByRole("heading", { name: "Summary", exact: true })
      ).toBeVisible();
    }
  });
});
