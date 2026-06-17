import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

const EMAIL = testEmail("navigation");

/**
 * Repaired for the "Reimagined" IA (RetroNav):
 *  - Top nav links are exactly: Overview (/) · Timeline · Summaries · Admin.
 *  - Upload + Summarize are hero actions in the floating dock, NOT top-nav links;
 *    the old "Home"/"Summarize"/"Upload" nav links and the notifications bell are gone.
 *  - The theme toggle's aria-label is "Toggle light/dark".
 *  - Sign out lives behind the avatar button (aria-label "Sign out").
 */
test.describe("Navigation", () => {
  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(EMAIL, TEST_PASSWORD);
  });

  test("all nav links navigate correctly", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    const navLinks: [string, RegExp][] = [
      ["Timeline", /\/timeline/],
      ["Summaries", /\/summaries/],
      ["Admin", /\/admin/],
      ["Overview", /\/$/],
    ];

    for (const [name, urlPattern] of navLinks) {
      await page.getByRole("link", { name, exact: true }).click();
      await expect(page).toHaveURL(urlPattern, { timeout: 10_000 });
    }
  });

  test("Upload is NOT a top-nav link (moved to the floating dock)", async ({
    page,
  }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    // The old standalone nav entries are gone from the top nav.
    const nav = page.locator("nav").first();
    await expect(nav.getByRole("link", { name: "Upload", exact: true })).toHaveCount(0);
    await expect(nav.getByRole("link", { name: "Home", exact: true })).toHaveCount(0);
    await expect(nav.getByRole("link", { name: "Summarize", exact: true })).toHaveCount(0);
  });

  test("active link has aria-current indicator", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    await page.locator('nav a[href="/timeline"]').click();
    await expect(page).toHaveURL(/\/timeline/, { timeout: 10_000 });

    // RetroNav marks the active link with aria-current="page".
    await expect(page.locator('nav a[href="/timeline"]')).toHaveAttribute(
      "aria-current",
      "page"
    );
  });

  test("logo links to home", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    await page.goto("/timeline");
    await expect(page).toHaveURL(/\/timeline/, { timeout: 10_000 });

    // The MedTimeline brand is the first link in nav (href="/").
    await page.locator('nav a[href="/"]').first().click();
    await expect(page).toHaveURL(/\/$/, { timeout: 10_000 });
  });

  test("theme toggle switches light/dark", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    const html = page.locator("html");
    const initialClass = await html.getAttribute("class");

    const themeToggle = page.locator('button[aria-label="Toggle light/dark"]');
    await expect(themeToggle).toBeVisible();
    await themeToggle.click();

    await expect(async () => {
      const newClass = await html.getAttribute("class");
      expect(newClass).not.toBe(initialClass);
    }).toPass({ timeout: 5_000 });
  });

  test("sign out clears auth and redirects to login", async ({ page }) => {
    await browserLogin(page, EMAIL, TEST_PASSWORD);
    await expect(page.locator("nav")).toBeVisible({ timeout: 10_000 });

    await page.locator('nav [aria-label="Sign out"]').click();
    await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });

    // Verify the auth guard redirects back to login on a protected route.
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });
  });
});
