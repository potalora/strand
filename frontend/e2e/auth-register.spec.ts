import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import {
  testIdentifier,
  uniqueIdentifier,
  TEST_PASSWORD,
} from "./helpers/test-data";

test.describe("Register page", () => {
  test("successful registration redirects to login", async ({ page }) => {
    const loginIdentifier = uniqueIdentifier("register");

    await page.goto("/register");
    await page.locator("#displayName").fill("Test User");
    const accountName = page.getByLabel("Account name");
    await expect(accountName).toHaveAttribute("type", "text");
    await expect(accountName).toHaveAttribute("autocomplete", "username");
    await expect(accountName).toHaveAttribute("autocapitalize", "none");
    await expect(accountName).toHaveAttribute("spellcheck", "false");
    await expect(
      page.getByText(/do not need an email address/i)
    ).toBeVisible();
    await expect(
      page.getByText(/visually similar Unicode text may still differ/i)
    ).toBeVisible();
    await page.locator("#loginIdentifier").fill(loginIdentifier);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    await page.waitForURL(/\/login/, { timeout: 15_000 });
    expect(page.url()).toContain("/login");
  });

  test("duplicate account name shows error", async ({ page }) => {
    const loginIdentifier = uniqueIdentifier("register duplicate");

    // Pre-register via API
    const api = new ApiClient();
    await api.register(loginIdentifier, TEST_PASSWORD);

    // Try the same account name in the browser.
    await page.goto("/register");
    await page.locator("#loginIdentifier").fill(loginIdentifier);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // Error div should appear
    const errorDiv = page
      .locator("div")
      .filter({ hasText: /unavailable|already|exists|registered|error/i })
      .first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });
  });

  test("short password rejected", async ({ page }) => {
    await page.goto("/register");
    await page.locator("#loginIdentifier").fill(testIdentifier("register short"));
    await page.locator("#password").fill("Ab1!");
    await page.locator('button[type="submit"]').click();

    // Browser minLength validation prevents submit, URL stays on /register
    expect(page.url()).toContain("/register");
  });

  test("empty account name prevents submit", async ({ page }) => {
    await page.goto("/register");
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // Required field validation prevents submit, URL stays on /register
    expect(page.url()).toContain("/register");
  });

  test("loading state shows during submit", async ({ page }) => {
    const loginIdentifier = uniqueIdentifier("register loading");

    await page.goto("/register");
    await page.locator("#displayName").fill("Loading Test");
    await page.locator("#loginIdentifier").fill(loginIdentifier);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // Immediately check for loading text on the button (uses an ellipsis char).
    await expect(
      page.locator('button[type="submit"]')
    ).toHaveText(/Creating account/, { timeout: 5_000 });
  });

  test("sign in link navigates to login", async ({ page }) => {
    await page.goto("/register");
    await page.getByRole("link", { name: "Sign in" }).click();

    await page.waitForURL(/\/login/, { timeout: 10_000 });
    expect(page.url()).toContain("/login");
  });
});
