import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

const EMAIL = testEmail("login");

test.describe("Login page", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(EMAIL, TEST_PASSWORD);
  });

  test("successful login redirects to dashboard", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#email").fill(EMAIL);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();
    await page.waitForURL(/\/$/, { timeout: 30_000 });
  });

  test("wrong password shows error", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#email").fill(EMAIL);
    await page.locator("#password").fill("WrongPass1!");
    await page.locator('button[type="submit"]').click();

    const errorDiv = page.locator("div").filter({ hasText: /failed|invalid|incorrect/i }).first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });

    const borderColor = await errorDiv.evaluate(
      (el) => getComputedStyle(el).borderColor
    );
    expect(borderColor).toBeTruthy();
  });

  test("nonexistent email shows error", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#email").fill(`nonexistent-${Date.now()}@test.com`);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    const errorDiv = page.locator("div").filter({ hasText: /failed|invalid|incorrect/i }).first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });
  });

  test("empty email prevents submit", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();
    expect(page.url()).toContain("/login");
  });

  test("empty password prevents submit", async ({ page }) => {
    await page.goto("/login");
    await page.locator("#email").fill(EMAIL);
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
    await page.locator("#email").fill(EMAIL);
    await page.locator("#password").fill(TEST_PASSWORD);

    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();
    await expect(submitBtn).toContainText("Signing in");
  });
});
