import { test, expect, type Page } from "./fixtures/console-gate";

/**
 * Consolidation changes #4/#5/#6 (fully mocked — no real backend, parallel-safe):
 *  - #4 shared user store: a transient 401 on /auth/me must NOT blank the name.
 *  - #5 Duplicates + Merges collapse into one "Deduplication" tab with two sub-tabs.
 *  - #6 the nav bell is gone; Settings (Preferences + Session) folds into System;
 *       /settings redirects to /admin?tab=sys.
 *
 * Auth is injected straight into localStorage (the persisted zustand shape) so the
 * dashboard layout authenticates without hitting the login rate limiter, and every
 * API call is stubbed with page.route for determinism.
 */

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const ME_OK = {
  id: "11111111-2222-3333-4444-555555555555",
  email: "pedro@example.com",
  display_name: "Pedro",
  is_active: true,
  created_at: "2024-01-01T00:00:00Z",
};

const OVERVIEW = {
  total_records: 12,
  total_uploads: 3,
  records_by_type: { condition: 5, observation: 7 },
  date_range_start: "2020-01-01T00:00:00Z",
  date_range_end: "2024-01-01T00:00:00Z",
};

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

/**
 * Stub every backend call the admin page + nav make. `meSequence` lets a test
 * drive the /auth/me status per call (e.g. [401, 200] to simulate a token race).
 */
async function mockBackend(
  page: Page,
  opts: { meSequence?: number[] } = {}
): Promise<void> {
  let meCall = 0;
  await page.route("**/api/v1/**", async (route) => {
    const url = route.request().url();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.includes("/auth/me")) {
      const seq = opts.meSequence;
      const status = seq ? seq[Math.min(meCall, seq.length - 1)] : 200;
      meCall++;
      if (status === 401) return json({ detail: "Token has expired" }, 401);
      return json(ME_OK);
    }
    if (url.includes("/auth/refresh")) {
      return json({
        access_token: "fresh.access.token",
        refresh_token: "fresh.refresh.token",
      });
    }
    if (url.includes("/auth/logout")) return json({});
    if (url.includes("/dashboard/overview")) return json(OVERVIEW);
    if (url.includes("/audit-log")) return json({ items: [], total: 0 });
    if (url.includes("/dedup/candidates/summary"))
      return json({ bands: [], total: 0 });
    if (url.includes("/dedup/candidates")) return json({ items: [], total: 0 });
    if (url.includes("/dedup/merges"))
      return json({ items: [], total: 0, counts: { auto: 0, manual: 0 } });
    if (url.includes("/records"))
      return json({ items: [], total: 0, page: 1, page_size: 100 });
    return json({});
  });
}

test.describe("Admin consolidation (#4/#5/#6)", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
  });

  test("(a) top nav has no bell/notifications button", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    const nav = page.locator("nav").first();
    await expect(nav).toBeVisible();

    // The bell linked to /settings with aria-label "Settings & notifications" — gone.
    // (Scope to the nav: the body-level toast region legitimately uses
    // aria-label "Notifications".)
    await expect(nav.locator('a[href="/settings"]')).toHaveCount(0);
    await expect(nav.locator('[aria-label*="otification"]')).toHaveCount(0);

    // Theme toggle + avatar (sign out) remain.
    await expect(nav.locator('[aria-label="Toggle light/dark"]')).toBeVisible();
    await expect(nav.locator('[aria-label="Sign out"]')).toBeVisible();
  });

  test("(b) admin shows exactly Records / Extractions / Deduplication / System", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("tab")).toHaveCount(4);
    await expect(page.getByRole("tab", { name: "Records" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Extractions" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Deduplication" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "System" })).toBeVisible();

    // The old standalone Duplicates / Merges top tabs are gone.
    await expect(page.getByRole("tab", { name: "Duplicates" })).toHaveCount(0);
    await expect(page.getByRole("tab", { name: "Merges" })).toHaveCount(0);
  });

  test("(c) Deduplication tab exposes both sub-tabs and renders each pane", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=dedup");

    const pending = page.getByRole("button", { name: "Pending review" });
    const ledger = page.getByRole("button", { name: "Merge ledger" });
    await expect(pending).toBeVisible();
    await expect(ledger).toBeVisible();

    // Default sub-tab = Pending review → the DedupTab scan control is present.
    await expect(
      page.getByRole("button", { name: "Scan for duplicates" })
    ).toBeVisible();

    // Switch to the Merge ledger sub-tab → the merges pane renders.
    await ledger.click();
    await expect(page.getByPlaceholder("Search merged records…")).toBeVisible();

    // ...and back to Pending review.
    await pending.click();
    await expect(
      page.getByRole("button", { name: "Scan for duplicates" })
    ).toBeVisible();
  });

  test("(d) System tab shows Account, Preferences, and Session", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    // Account
    await expect(page.getByRole("heading", { name: "Account" })).toBeVisible();
    await expect(page.getByText("Pedro")).toBeVisible({ timeout: 15_000 });

    // Preferences (folded in from /settings): theme toggle + delete confirmation.
    await expect(
      page.getByRole("heading", { name: "Preferences" })
    ).toBeVisible();
    await expect(page.getByText("Appearance")).toBeVisible();
    await expect(page.getByText("Delete confirmation")).toBeVisible();

    // Session: sign out is reachable here (scope to the pane — the nav avatar
    // also carries an aria-label "Sign out").
    await expect(
      page.locator("main").getByRole("button", { name: "Sign out" })
    ).toBeVisible();
  });

  test("(e) account name does not blank when /auth/me 401s once then 200s", async ({
    page,
  }) => {
    await mockBackend(page, { meSequence: [401, 200] });
    await page.goto("/admin?tab=sys");

    // The real name resolves after the transient 401 — never a permanent blank.
    await expect(page.getByText("Pedro")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Not set")).toHaveCount(0);

    // The shared store also feeds the nav avatar (initials of "Pedro" → "P").
    await expect(page.locator('nav [aria-label="Sign out"]')).toHaveText("P");
  });

  test("(f) old /settings route redirects into Admin → System", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/settings");

    await expect(page).toHaveURL(/\/admin\?tab=sys/, { timeout: 10_000 });
    await expect(page.getByRole("heading", { name: "Account" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "System" })).toBeVisible();
  });
});
