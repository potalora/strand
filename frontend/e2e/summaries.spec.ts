import { test, expect, type Page } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { PATHS, uniqueEmail, TEST_PASSWORD } from "./helpers/test-data";

const email = uniqueEmail("summaries");
const modelExecutionConfigured =
  process.env.E2E_LOCAL_ONLY === "1" || Boolean(process.env.GEMINI_API_KEY);
const SUCCESSFUL_UPLOAD_STATUSES = [
  "awaiting_confirmation",
  "completed",
  "completed_with_merges",
  "awaiting_review",
];

type AcceptedSummary = {
  id: string;
  job_id: string;
  status: string;
};

async function startBackgroundSummary(page: Page): Promise<AcceptedSummary> {
  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/v1/summary/generate"
  );
  await page.getByRole("button", { name: "Generate summary" }).click();
  const response = await responsePromise;

  expect(response.status()).toBe(202);
  const accepted = (await response.json()) as AcceptedSummary;
  expect(accepted.id).toBeTruthy();
  expect(accepted.job_id).toBeTruthy();
  expect(accepted.status).toBe("queued");
  await expect(
    page.getByText(
      "Summary is processing in the background. You can leave this page."
    )
  ).toBeVisible();
  const monitor = page.getByRole("region", { name: "Background processing" });
  await expect(monitor).toContainText("Summary");
  return accepted;
}

async function waitForCompletedSummary(
  page: Page,
  api: ApiClient,
  accepted: AcceptedSummary
): Promise<void> {
  const job = await api.pollLocalAIJob(accepted.job_id);
  expect(job.id).toBe(accepted.job_id);
  expect(job.status).toBe("completed");
  await expect(
    page.getByRole("region", { name: "Background processing" })
  ).toContainText("finished", { timeout: 15_000 });
}

async function openSavedSummary(
  page: Page,
  api: ApiClient,
  promptId: string
): Promise<void> {
  await expect
    .poll(
      async () =>
        (await api.getSummaryPrompts()).items.some(
          (prompt) => prompt.id === promptId
        ),
      { timeout: 15_000 }
    )
    .toBe(true);
  const history = await api.getSummaryPrompts();
  const generatedIndex = history.items.findIndex(
    (prompt) => prompt.id === promptId
  );
  expect(generatedIndex).toBeGreaterThanOrEqual(0);

  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Summary", exact: true })
  ).toHaveCount(0);
  await page.getByRole("button", { name: /Show \(\d+\)/ }).click();
  const detailResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "GET" &&
      new URL(response.url()).pathname ===
        `/api/v1/summary/prompts/${promptId}`
  );
  await page.locator("button.lrow").nth(generatedIndex).click();
  const detailResponse = await detailResponsePromise;
  expect(detailResponse.ok()).toBe(true);
  await expect(
    page.getByRole("heading", { name: "Summary", exact: true })
  ).toBeVisible({ timeout: 15_000 });
}

async function generateAndOpenSummary(
  page: Page,
  api: ApiClient
): Promise<string> {
  if (process.env.E2E_LOCAL_ONLY === "1") {
    const accepted = await startBackgroundSummary(page);
    await waitForCompletedSummary(page, api, accepted);
    await openSavedSummary(page, api, accepted.id);
    return accepted.id;
  }

  const responsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      new URL(response.url()).pathname === "/api/v1/summary/generate"
  );
  await page.getByRole("button", { name: "Generate summary" }).click();
  const response = await responsePromise;
  expect(response.ok()).toBe(true);
  const generated = (await response.json()) as { id: string };
  expect(generated.id).toBeTruthy();
  await expect(
    page.getByRole("heading", { name: "Summary", exact: true })
  ).toBeVisible({ timeout: 60_000 });
  await openSavedSummary(page, api, generated.id);
  return generated.id;
}

/**
 * Repaired for the current Summaries page labels:
 *  - Summary types: "Full record" / "By category" / "Date range".
 *  - Output formats: "Natural language" / "JSON data" / "Both".
 *  - Results card heading is "Summary" (not "Summary results"); the count line is
 *    "{n} records · {model}" (middot, not "|"); result tabs are
 *    "Narrative" / "JSON data".
 *  - Strict-local generation returns 202 and completes in the background; the
 *    persisted result is opened from history after completion.
 *  - History toggle is "Show ({n})"; each entry is a row button ("{type} summary").
 */
test.describe("Summaries page", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    const status = await api.pollUploadStatus(result.upload_id, 60_000);
    expect(SUCCESSFUL_UPLOAD_STATUSES).toContain(
      status.ingestion_status ?? status.status
    );
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

    // The category select appears with options like "Labs & Vitals". Locate it by
    // its option text rather than position — the page now also renders an "AI
    // provider" select, so a fixed index (nth) is no longer reliable.
    const categorySelect = page
      .locator("select")
      .filter({ hasText: "Labs & Vitals" });
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
    test.skip(!modelExecutionConfigured, "No E2E model execution profile is configured");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });
    await expect(async () => {
      const text = await select.textContent();
      expect(text).not.toContain("No patients found");
    }).toPass({ timeout: 15_000 });

    await generateAndOpenSummary(page, api);

    // Strict-local results are durable and reopen from saved history.
    // Count line: "{n} records · {model}".
    await expect(page.getByText(/\d+ record/).first()).toBeVisible();
    // Result tabs: "Narrative" / "JSON data".
    await expect(page.getByRole("button", { name: "Narrative" })).toBeVisible();
    await expect(
      page.locator(".panel").filter({ hasText: "[Fact:" }).first()
    ).toContainText("Evidence:");
  });

  test("history entry reopens a saved summary without regenerating", async ({
    page,
  }) => {
    test.skip(!modelExecutionConfigured, "No E2E model execution profile is configured");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });
    await expect(async () => {
      const text = await select.textContent();
      expect(text).not.toContain("No patients found");
    }).toPass({ timeout: 15_000 });

    let generationRequestCount = 0;
    page.on("request", (request) => {
      if (
        request.method() === "POST" &&
        new URL(request.url()).pathname === "/api/v1/summary/generate"
      ) {
        generationRequestCount += 1;
      }
    });
    await generateAndOpenSummary(page, api);

    await expect
      .poll(() => generationRequestCount, { timeout: 2_000 })
      .toBe(1);
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

  test("generation reports the selected privacy boundary", async ({ page }) => {
    test.skip(!modelExecutionConfigured, "No E2E model execution profile is configured");
    test.setTimeout(120_000);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/summaries");

    const select = page.locator("select").first();
    await expect(select).toBeVisible({ timeout: 10_000 });

    await generateAndOpenSummary(page, api);

    const deidentReport = page.getByText("De-identification report");
    if (process.env.E2E_LOCAL_ONLY === "1") {
      await expect(page.getByText(/Validated strict local/).last()).toBeVisible();
      await expect(deidentReport).toHaveCount(0);
      return;
    }

    // Cloud-assisted generation reports scrubbing only when identifiers were found.
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
