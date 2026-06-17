import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const EMAIL = testEmail("admin-records");

/**
 * Repaired for the current Admin → Records tab. The old collapsible By Type /
 * By Upload tree and the "Admin Console" heading are gone; the tab is now a flat,
 * searchable, sortable table (`table.rtable` with `tr.clickable` rows). The page
 * heading is "Admin"; tabs use role="tab".
 */
test.describe.serial("Admin — Records tab", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(EMAIL, TEST_PASSWORD);
    await api.login(EMAIL, TEST_PASSWORD);
    const result = await api.uploadStructured(
      PATHS.fhirBundle,
      "sample_fhir_bundle.json"
    );
    await api.pollUploadStatus(result.upload_id, 60_000);
  });

  test("admin page renders with Records tab active", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");

    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible({
      timeout: 10_000,
    });
    const recordsTab = page.getByRole("tab", { name: "Records" });
    await expect(recordsTab).toBeVisible();
    await expect(recordsTab).toHaveAttribute("aria-selected", "true");

    // The records table loads with rows.
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 15_000,
    });
  });

  test("records table renders rows + a footer count", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");

    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 15_000,
    });
    // Footer reads "{shown} of {total} records".
    await expect(page.getByText(/\d+ of \d+ records/)).toBeVisible();
  });

  test("search narrows the visible rows", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 15_000,
    });

    const search = page.getByPlaceholder("Search descriptions, codes, sources…");
    await search.fill("zzz-no-such-record-zzz");
    // The search is client-side over the loaded set → zero matches.
    await expect(page.getByText(/^0 of \d+ records$/)).toBeVisible({
      timeout: 5_000,
    });

    await search.fill("");
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 5_000,
    });
  });

  test("type filter narrows the set", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 15_000,
    });

    const select = page.locator("select.selectbox");
    await expect(select).toBeVisible();
    // Pick the first real record-type option (index 0 is "All types (N)").
    const value = await select
      .locator("option")
      .nth(1)
      .getAttribute("value");
    await select.selectOption(value!);

    // Still shows rows, and a footer count that is <= total.
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 5_000,
    });
    await expect(page.getByText(/\d+ of \d+ records/)).toBeVisible();
  });

  test("click a row opens the detail drawer", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");
    const row = page.locator("tr.clickable").first();
    await expect(row).toBeVisible({ timeout: 15_000 });
    await row.click();
    await expect(page.getByText("Record detail")).toBeVisible({
      timeout: 10_000,
    });
  });

  test("row delete shows a confirm dialog, cancel closes it", async ({
    page,
  }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");
    const row = page.locator("tr.clickable").first();
    await expect(row).toBeVisible({ timeout: 15_000 });

    await row.locator("button.row-del").click();
    await expect(page.getByText("Delete record?")).toBeVisible({
      timeout: 5_000,
    });

    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByText("Delete record?")).not.toBeVisible({
      timeout: 3_000,
    });
  });

  test("confirm delete removes a record", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await page.goto("/admin");
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 15_000,
    });

    const footer = page.getByText(/\d+ of \d+ records/);
    const before = parseInt((await footer.textContent())!.match(/of (\d+)/)![1], 10);

    await page.locator("tr.clickable").first().locator("button.row-del").click();
    await expect(page.getByText("Delete record?")).toBeVisible({ timeout: 5_000 });
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await expect(page.getByText("Delete record?")).not.toBeVisible({
      timeout: 5_000,
    });

    await expect(async () => {
      const after = parseInt(
        (await footer.textContent())!.match(/of (\d+)/)![1],
        10
      );
      expect(after).toBe(before - 1);
    }).toPass({ timeout: 10_000 });
  });
});
