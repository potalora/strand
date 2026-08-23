import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { testEmail, testIdentifier, TEST_PASSWORD } from "./helpers/test-data";

const LOGIN_IDENTIFIER = testIdentifier("login");
const LEGACY_EMAIL = testEmail("login-legacy");

test.describe("Login page", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(LOGIN_IDENTIFIER, TEST_PASSWORD);
    await api.register(LEGACY_EMAIL, TEST_PASSWORD);
  });

  test("account-name login redirects to dashboard", async ({ page }) => {
    await page.goto("/login");
    const accountName = page.getByLabel("Account name or existing email");
    await expect(accountName).toHaveAttribute("type", "text");
    await expect(accountName).toHaveAttribute("autocomplete", "username");
    await expect(accountName).toHaveAttribute("autocapitalize", "none");
    await expect(accountName).toHaveAttribute("spellcheck", "false");
    await expect(
      page.getByText(/visually similar Unicode text may still differ/i)
    ).toBeVisible();
    await page.locator("#loginIdentifier").fill(LOGIN_IDENTIFIER);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();
    await page.waitForURL(/\/$/, { timeout: 30_000 });
  });

  test("existing email-shaped account login redirects to dashboard", async ({
    page,
  }) => {
    await page.goto("/login");
    await page.locator("#loginIdentifier").fill(LEGACY_EMAIL);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();
    await page.waitForURL(/\/$/, { timeout: 30_000 });
  });

  test("wrong password shows error", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#loginIdentifier").fill(LOGIN_IDENTIFIER);
    await page.locator("#password").fill("WrongPass1!");
    await page.locator('button[type="submit"]').click();

    const errorDiv = page.locator("div").filter({ hasText: /failed|invalid|incorrect/i }).first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });

    const borderColor = await errorDiv.evaluate(
      (el) => getComputedStyle(el).borderColor
    );
    expect(borderColor).toBeTruthy();
  });

  test("nonexistent account name shows error", async ({ page }) => {
    await page.goto("/login");
    await page
      .locator("#loginIdentifier")
      .fill(`nonexistent account ${Date.now()}`);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    const errorDiv = page.locator("div").filter({ hasText: /failed|invalid|incorrect/i }).first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });
  });

  test("empty account name prevents submit", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();
    expect(page.url()).toContain("/login");
  });

  test("empty password prevents submit", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#loginIdentifier").fill(LOGIN_IDENTIFIER);
    await page.locator('button[type="submit"]').click();
    expect(page.url()).toContain("/login");
  });

  test("register link navigates to /register", async ({ page }) => {
    await page.goto("/login");
    // The link reads "Create one" ("No account? Create one") and points at /register.
    await page.locator('a[href="/register"]').click();
    await page.waitForURL(/\/register/, { timeout: 10_000 });
  });

  test("loading state shows during submit", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#loginIdentifier").fill(LOGIN_IDENTIFIER);
    await page.locator("#password").fill(TEST_PASSWORD);

    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();
    await expect(submitBtn).toContainText("Signing in");
  });
});
