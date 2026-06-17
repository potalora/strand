import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

let testIndex = 0;

function uniqueEmail(): string {
  return testEmail(`guard-${++testIndex}-${Date.now()}`);
}

test.describe("Authentication guards", () => {
  test("unauthenticated user on / redirected to /login", async ({ page }) => {
    await page.goto("/");
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    expect(page.url()).toContain("/login");
  });

  test("unauthenticated user on /timeline redirected to /login", async ({
    page,
  }) => {
    await page.goto("/timeline");
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    expect(page.url()).toContain("/login");
  });

  test("unauthenticated user on /admin redirected to /login", async ({
    page,
  }) => {
    await page.goto("/admin");
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    expect(page.url()).toContain("/login");
  });

  test("logout clears tokens and redirects", async ({ page }) => {
    const api = new ApiClient();
    const email = uniqueEmail();
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    // Click Sign out in the nav
    await page.getByRole("button", { name: "Sign out" }).click();

    // Should redirect to /login
    await page.waitForURL(/\/login/, { timeout: 10_000 });
    expect(page.url()).toContain("/login");
  });

  test("localStorage cleared after logout", async ({ page }) => {
    const api = new ApiClient();
    const email = uniqueEmail();
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    // Click Sign out in the nav
    await page.getByRole("button", { name: "Sign out" }).click();
    await page.waitForURL(/\/login/, { timeout: 10_000 });

    // Verify localStorage auth state is cleared. The persisted store flushes
    // asynchronously after logout, so poll until it settles to a cleared state
    // (key removed, or isAuthenticated false) rather than reading once.
    await expect
      .poll(
        async () =>
          page.evaluate(() => {
            const raw = localStorage.getItem("medtimeline-auth");
            if (!raw) return "cleared";
            try {
              const parsed = JSON.parse(raw);
              return parsed.state?.isAuthenticated === false ? "cleared" : "authed";
            } catch {
              return "cleared";
            }
          }),
        { timeout: 5_000 }
      )
      .toBe("cleared");
  });
});
