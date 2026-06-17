import { test, expect } from "./fixtures/console-gate";
import { browserLogin } from "./helpers/browser-login";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

const email = testEmail("admin-extractions");

/**
 * Repaired for the current Admin → Extractions tab. Tabs use role="tab"; the
 * empty-state copy is "Nothing waiting — no files pending extraction, processing,
 * or failed." (The populated table — failed/processing rows, Extract button — is
 * covered by extraction-terminal-state.spec.ts with mocked endpoints.)
 */
test.beforeAll(async () => {
  const api = new ApiClient();
  await api.register(email, TEST_PASSWORD);
});

test.describe("Admin — Extractions tab", () => {
  test("empty state renders when nothing is pending", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=extractions");
    await expect(page.getByText(/Nothing waiting/i)).toBeVisible({
      timeout: 10_000,
    });
    await expect(
      page.getByText(/no files pending extraction, processing, or failed/i)
    ).toBeVisible();
  });

  test("all four admin tabs are present", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin");
    await expect(page.getByRole("tab")).toHaveCount(4);
    await expect(page.getByRole("tab", { name: "Records" })).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByRole("tab", { name: "Extractions" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Deduplication" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "System" })).toBeVisible();
  });

  test("Extractions tab is selectable from the admin page", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin");
    const tab = page.getByRole("tab", { name: "Extractions" });
    await tab.click();
    await expect(tab).toHaveAttribute("aria-selected", "true");
    await expect(page.getByText(/Nothing waiting/i)).toBeVisible({
      timeout: 10_000,
    });
  });
});
