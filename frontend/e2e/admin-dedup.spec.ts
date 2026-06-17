import { test, expect, type Page } from "./fixtures/console-gate";

/**
 * Admin → Deduplication (Pending review sub-tab). Repaired + strengthened.
 *
 * The old spec uploaded the same FHIR bundle twice to manufacture duplicates,
 * but stable-id idempotency now makes an identical re-upload a no-op (0 new
 * records → 0 candidates), so that approach can no longer populate the queue.
 * Instead we mock the dedup API with stateful candidates so the full grouped-
 * review flow — confidence bands, expand, per-row Merge / Keep both, and the
 * Merge-ledger sub-tab — is exercised deterministically (parallel-safe, no rate
 * limiter). The console-error gate is active throughout, so a render error in any
 * candidate row fails the test.
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

function mkCandidate(id: string, score: number) {
  return {
    id,
    similarity_score: score,
    status: "pending",
    match_reasons: { code_value: true, display_text: true, effective_date: true },
    record_a: {
      id: `${id}-a`,
      display_text: "Hemoglobin A1c",
      record_type: "observation",
      source_format: "fhir",
      effective_date: "2024-01-01T00:00:00Z",
    },
    record_b: {
      id: `${id}-b`,
      display_text: "Hemoglobin A1c",
      record_type: "observation",
      source_format: "epic",
      effective_date: "2024-01-01T00:00:00Z",
    },
  };
}

const band10 = (score: number) => Math.floor(Math.round(score * 100) / 10) * 10;

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

/** Stateful dedup mock — merges/dismisses actually remove candidates. */
async function mockDedup(page: Page): Promise<void> {
  let candidates = [mkCandidate("c1", 0.92), mkCandidate("c2", 0.91)];

  await page.route("**/api/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.includes("/auth/me")) return json(ME_OK);
    if (url.includes("/auth/refresh"))
      return json({ access_token: "fresh", refresh_token: "fresh" });

    if (url.includes("/dedup/candidates/summary")) {
      const counts = new Map<number, number>();
      for (const c of candidates) {
        const b = band10(c.similarity_score);
        counts.set(b, (counts.get(b) ?? 0) + 1);
      }
      const bands = [...counts.entries()]
        .map(([band, count]) => ({ band, count }))
        .sort((a, b) => b.band - a.band);
      return json({ bands, total: candidates.length });
    }

    if (url.includes("/dedup/candidates")) {
      const u = new URL(url);
      const min = Math.round(parseFloat(u.searchParams.get("score_min") ?? "0") * 100);
      const max = Math.round(parseFloat(u.searchParams.get("score_max") ?? "1") * 100);
      const items = candidates.filter((c) => {
        const pct = Math.round(c.similarity_score * 100);
        return pct >= min && pct < max;
      });
      return json({ items, total: items.length });
    }

    if (url.includes("/dedup/merge") || url.includes("/dedup/dismiss")) {
      const body = req.postDataJSON?.() ?? {};
      candidates = candidates.filter((c) => c.id !== body.candidate_id);
      return json({ ok: true });
    }

    if (url.includes("/dedup/scan")) return json({ auto_merged: 5, candidates_found: 2 });
    if (url.includes("/dedup/merges"))
      return json({ items: [], total: 0, counts: { auto: 0, manual: 0 } });
    if (url.includes("/dashboard/overview"))
      return json({ total_records: 0, total_uploads: 0, records_by_type: {} });
    if (url.includes("/records")) return json({ items: [], total: 0, page: 1, page_size: 100 });
    return json({});
  });
}

test.describe("Admin — Deduplication (Pending review)", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
    await mockDedup(page);
  });

  test("pending sub-tab shows scan button and confidence bands", async ({ page }) => {
    await page.goto("/admin?tab=dedup");
    await expect(
      page.getByRole("button", { name: "Scan for duplicates" })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("button", { name: /90% match/ })).toBeVisible();
    await expect(page.getByText("2 pairs")).toBeVisible();
  });

  test("expanding a band reveals candidate rows with Merge / Keep both", async ({
    page,
  }) => {
    await page.goto("/admin?tab=dedup");
    await page.getByRole("button", { name: /90% match/ }).click();

    await expect(
      page.getByRole("button", { name: "Merge", exact: true }).first()
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByRole("button", { name: "Keep both", exact: true }).first()
    ).toBeVisible();
    // RecordMini renders the record title on both sides.
    await expect(page.getByText("Hemoglobin A1c").first()).toBeVisible();
  });

  test("merging a row removes it from the band", async ({ page }) => {
    await page.goto("/admin?tab=dedup");
    await page.getByRole("button", { name: /90% match/ }).click();

    const keepBoth = page.getByRole("button", { name: "Keep both", exact: true });
    await expect(keepBoth).toHaveCount(2, { timeout: 10_000 });

    await page.getByRole("button", { name: "Merge", exact: true }).first().click();
    await expect(keepBoth).toHaveCount(1, { timeout: 10_000 });
  });

  test("scan reports a result summary", async ({ page }) => {
    await page.goto("/admin?tab=dedup");
    await page.getByRole("button", { name: "Scan for duplicates" }).click();
    await expect(page.getByText(/sent to review/)).toBeVisible({ timeout: 10_000 });
  });

  test("Merge ledger sub-tab renders the ledger pane", async ({ page }) => {
    await page.goto("/admin?tab=dedup");
    await page.getByRole("button", { name: "Merge ledger" }).click();
    await expect(
      page.getByPlaceholder("Search merged records…")
    ).toBeVisible({ timeout: 10_000 });
  });
});
