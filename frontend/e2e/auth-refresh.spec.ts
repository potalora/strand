import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { uniqueEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Regression for the missing token-refresh: the frontend stored a refreshToken
 * but never called /auth/refresh, so once the 15-min access token expired every
 * API call 401'd and pages silently rendered empty. The api client should now
 * transparently refresh on a 401 and retry the original request.
 */
test.describe("Access token auto-refresh", () => {
  // Two UI logins back-to-back; run serially so they don't race the backend
  // login rate limiter (5/60s per IP). Real-backend refresh is sensitive to that
  // limiter under heavy parallel auth load, so retry a transient contention
  // failure rather than letting it flake the suite (coverage is unchanged).
  test.describe.configure({ mode: "serial", retries: 2 });

  test("expired access token is transparently refreshed on a 401", async ({
    page,
  }) => {
    const email = uniqueEmail("authrefresh");
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);

    // Log in via the UI so real access + refresh tokens land in localStorage.
    await browserLogin(page, email, TEST_PASSWORD);

    // Wait for zustand-persist to actually flush the refresh token to
    // localStorage before we mutate it. Under parallel load the flush can lag the
    // post-login navigation; corrupting too early would drop the refresh token and
    // wrongly bounce us to /login.
    await expect
      .poll(
        () =>
          page.evaluate(() => {
            try {
              const p = JSON.parse(localStorage.getItem("medtimeline-auth") || "{}");
              return Boolean(p?.state?.refreshToken);
            } catch {
              return false;
            }
          }),
        { timeout: 10_000 }
      )
      .toBe(true);

    // Corrupt ONLY the access token (keep the valid refresh token) to simulate
    // a 15-min expiry. The next API call must 401 → refresh → retry.
    await page.evaluate(() => {
      const raw = localStorage.getItem("medtimeline-auth");
      const parsed = JSON.parse(raw as string);
      parsed.state.accessToken = "invalid.expired.token";
      localStorage.setItem("medtimeline-auth", JSON.stringify(parsed));
    });

    // Observe refresh attempts (the api client refreshes + retries on a 401).
    const refreshStatuses: number[] = [];
    page.on("response", (r) => {
      if (r.url().includes("/auth/refresh")) refreshStatuses.push(r.status());
    });

    await page.goto("/").catch(() => {});

    // End state: we stay on the home page (NOT bounced to /login), the masthead
    // renders, and the stored access token has been rotated away from the bad one.
    // Asserting the end state (rather than racing a specific intermediate
    // response) keeps the test robust under heavy parallel backend load.
    await expect(page).toHaveURL(/\/$/, { timeout: 30_000 });
    await expect(page.getByText("Personal Health Record")).toBeVisible({
      timeout: 15_000,
    });

    await expect
      .poll(
        () =>
          page.evaluate(() => {
            try {
              return JSON.parse(localStorage.getItem("medtimeline-auth") || "{}")
                ?.state?.accessToken as string | undefined;
            } catch {
              return undefined;
            }
          }),
        { timeout: 10_000 }
      )
      .not.toBe("invalid.expired.token");

    // A transparent refresh actually happened and succeeded.
    expect(refreshStatuses).toContain(200);
  });

  test("a 401 with no usable refresh token redirects to login", async ({
    page,
  }) => {
    const email = uniqueEmail("authrefresh-fail");
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
    await browserLogin(page, email, TEST_PASSWORD);

    // Invalidate BOTH tokens — refresh cannot succeed, so the app must bounce
    // the user to /login rather than silently showing empty data.
    await page.evaluate(() => {
      const parsed = JSON.parse(
        localStorage.getItem("medtimeline-auth") as string
      );
      parsed.state.accessToken = "invalid.expired.token";
      parsed.state.refreshToken = "invalid.refresh.token";
      localStorage.setItem("medtimeline-auth", JSON.stringify(parsed));
    });

    // The redirect to /login can fire mid-navigation and interrupt goto(),
    // which is itself proof the guard kicked in — tolerate it and assert the
    // end state.
    await page.goto("/").catch(() => {});
    await expect(page).toHaveURL(/\/login/, { timeout: 20_000 });
  });
});
