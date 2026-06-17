import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

test.describe("Register page", () => {
  test("successful registration redirects to login", async ({ page }) => {
    const uniqueEmail = `e2e-register-${Date.now()}@test.com`;

    await page.goto("/register");
    await page.locator("#displayName").fill("Test User");
    await page.locator("#email").fill(uniqueEmail);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    await page.waitForURL(/\/login/, { timeout: 15_000 });
    expect(page.url()).toContain("/login");
  });

  test("duplicate email shows error", async ({ page }) => {
    const email = `e2e-register-dup-${Date.now()}@test.com`;

    // Pre-register via API
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);

    // Try same email in browser
    await page.goto("/register");
    await page.locator("#email").fill(email);
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // Error div should appear
    const errorDiv = page.locator("div").filter({ hasText: /already|exists|registered|error/i }).first();
    await expect(errorDiv).toBeVisible({ timeout: 10_000 });
  });

  test("short password rejected", async ({ page }) => {
    await page.goto("/register");
    await page.locator("#email").fill(testEmail("register"));
    await page.locator("#password").fill("Ab1!");
    await page.locator('button[type="submit"]').click();

    // Browser minLength validation prevents submit, URL stays on /register
    expect(page.url()).toContain("/register");
  });

  test("empty email prevents submit", async ({ page }) => {
    await page.goto("/register");
    await page.locator("#password").fill(TEST_PASSWORD);
    await page.locator('button[type="submit"]').click();

    // Required field validation prevents submit, URL stays on /register
    expect(page.url()).toContain("/register");
  });

  test("loading state shows during submit", async ({ page }) => {
    const uniqueEmail = `e2e-register-loading-${Date.now()}@test.com`;

    await page.goto("/register");
    await page.locator("#displayName").fill("Loading Test");
    await page.locator("#email").fill(uniqueEmail);
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
