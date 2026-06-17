import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

/**
 * Repaired for the "Reimagined" Overview (home) page. The old "Dashboard"
 * masthead, "Records by category" badges, "Recent activity" TerminalLog and
 * "Go to Upload"/"Create summary" links no longer exist. The current page is an
 * editorial masthead + "Your most recent results" markers + a
 * "Conditions & medications on file" / "Recently added" bento, with an empty
 * state that offers an "Upload records" button.
 */
test.describe("Overview (home) with seeded data", () => {
  const email = testEmail("dashboard");

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

  test("masthead renders with secure chip", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(page.getByText("Personal Health Record")).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.getByText("End-to-end encrypted")).toBeVisible();
  });

  test("most recent results section renders", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Your most recent results" })
    ).toBeVisible({ timeout: 15_000 });
  });

  test("conditions & medications and recently added sections render", async ({
    page,
  }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Conditions & medications on file" })
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.getByRole("heading", { name: "Recently added" })
    ).toBeVisible();
  });

  test("All labs & vitals navigates to /labs", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Your most recent results" })
    ).toBeVisible({ timeout: 15_000 });
    await page.getByRole("button", { name: /labs & vitals/i }).click();
    await expect(page).toHaveURL(/\/labs/);
  });

  test("Recently added Timeline button navigates to /timeline", async ({
    page,
  }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Recently added" })
    ).toBeVisible({ timeout: 15_000 });
    await page.getByRole("button", { name: "Timeline" }).click();
    await expect(page).toHaveURL(/\/timeline/);
  });
});

test.describe("Overview (home) empty state", () => {
  const email = testEmail("dashboard-empty");

  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
  });

  test("shows No records yet for fresh user", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(page.getByText("No records yet")).toBeVisible({
      timeout: 15_000,
    });
  });

  test("shows Upload records button in empty state", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: /Upload records/ })
    ).toBeVisible({ timeout: 15_000 });
  });

  test("Upload records button navigates to /upload", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/");
    await page.getByRole("button", { name: /Upload records/ }).click();
    await expect(page).toHaveURL(/\/upload/);
  });
});
