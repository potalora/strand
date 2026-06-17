import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { uniqueEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Regression for a logout/refresh race surfaced by the hardened e2e suite.
 *
 * Making /auth/me participate in transparent 401→refresh means a refresh can be
 * in-flight when the user logs out. If that refresh completes AFTER the session
 * was cleared and unconditionally calls setTokens(), it RESURRECTS the just-
 * killed session (localStorage repopulated with valid tokens). The single-flight
 * refresh must discard its result when the session was cleared while in-flight.
 *
 * Deterministic design (drives the REAL guard path without UI/render timing):
 *  - GATE /auth/refresh so the test controls exactly when it resolves.
 *  - Corrupt ONLY the access token, then trigger the refresh with a CLIENT-SIDE
 *    nav (clicking the Timeline link). A full page.goto/reload is NOT used: with
 *    /auth/refresh gated, a full document load makes the app bail to /login and
 *    abort the in-flight refresh (net::ERR_ABORTED) before we can release it. A
 *    client-side nav keeps the document mounted and reliably fires one refresh.
 *  - Log out for real WHILE the refresh is held (click Sign out → clearTokens),
 *    which clears the in-memory zustand store AND its localStorage mirror.
 *    Removing only the localStorage key is unfaithful: zustand re-persists the
 *    still-authed in-memory state under parallel load and defeats the guard
 *    (false "resurrected"). A real logout is what the guard must withstand.
 *  - Release the refresh and assert the rotated tokens were NOT applied.
 */
test.describe("Logout during an in-flight token refresh", () => {
  test.describe.configure({ retries: 2 });

  test("a refresh that resolves after logout must not resurrect the session", async ({
    page,
  }) => {
    const email = uniqueEmail("logout-race");
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);

    // Real tokens in localStorage; dashboard + top nav fully loaded.
    await browserLogin(page, email, TEST_PASSWORD);
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
    await expect(page.locator('nav [aria-label="Sign out"]')).toBeVisible({
      timeout: 10_000,
    });

    // Hold /auth/refresh open until the test releases it, then return a valid
    // new pair (a "successful" rotation that would resurrect the session unless
    // the client guards against the intervening logout).
    let releaseRefresh: () => void = () => {};
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve;
    });
    await page.route("**/auth/refresh", async (route) => {
      await refreshGate;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          access_token: "resurrected.access.token",
          refresh_token: "resurrected.refresh.token",
        }),
      });
    });
    // Mock the logout round-trip so the Sign-out click resolves immediately and
    // deterministically (no real-backend latency under heavy parallel load).
    await page.route("**/auth/logout", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "{}" })
    );

    // Corrupt ONLY the access token (keep the valid refresh token) so the next
    // authed call 401s → triggers the now-gated single-flight refresh.
    await page.evaluate(() => {
      const raw = localStorage.getItem("medtimeline-auth");
      const parsed = JSON.parse(raw as string);
      parsed.state.accessToken = "invalid.expired.token";
      localStorage.setItem("medtimeline-auth", JSON.stringify(parsed));
    });

    // Client-side nav to Timeline → its authed fetch 401s → gated refresh. The
    // (dashboard) layout stays mounted (no full document load). Register the
    // waiter first so we never miss the single (single-flight) refresh request.
    // `exact: true` — otherwise the "MedTimeline" brand link also matches.
    const refreshIssued = page.waitForRequest("**/auth/refresh", { timeout: 15_000 });
    await page.getByRole("link", { name: "Timeline", exact: true }).click();
    await refreshIssued;

    // Log out for REAL while the refresh is held: the Sign-out click runs
    // clearTokens(), clearing BOTH the in-memory zustand store and its
    // localStorage mirror, then redirects to /login. getRefreshToken() then
    // reads null — exactly as in a genuine logout. (The nav/layout stays mounted
    // because the trigger was a client-side nav, not a full document load.)
    await page.locator('nav [aria-label="Sign out"]').click();
    await expect(page).toHaveURL(/\/login/, { timeout: 15_000 });

    // Now let the refresh resolve — AFTER the session was cleared. Arm the
    // response waiter before releasing so the fulfilled response isn't missed.
    const refreshResponded = page.waitForResponse("**/auth/refresh", { timeout: 10_000 });
    releaseRefresh();
    await refreshResponded;

    // The session must stay dead: the rotated tokens must NOT be applied.
    await expect
      .poll(
        () =>
          page.evaluate(() => {
            try {
              const raw = localStorage.getItem("medtimeline-auth");
              if (!raw) return "cleared";
              const s = JSON.parse(raw).state ?? {};
              if (s.accessToken === "resurrected.access.token") return "resurrected";
              if (s.isAuthenticated === true) return "authed";
              return "cleared";
            } catch {
              return "cleared";
            }
          }),
        { timeout: 5_000 }
      )
      .toBe("cleared");
  });
});
