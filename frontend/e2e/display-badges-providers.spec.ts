import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

// Covers two display refinements (session summary §2b / extraction-remediation C1):
//   1. Observation badges show the SUB-TYPE — Lab / Vital / Social — derived from
//      the category each record already carries, instead of a generic "Observation".
//   2. Providers/performers are surfaced on the record detail (Encounter.participant,
//      Observation.performer, Procedure.performer).
//
// The seed bundle (sample_fhir_bundle.json) contains exactly one observation of
// each sub-type (laboratory / vital-signs / social-history), an Encounter whose
// participant is "Dr. Smith", and a Procedure performed by "Dr. Johnson".

const API_BASE = "http://localhost:8000/api/v1";
const email = testEmail("display-badges-providers");

// One login (with 429 backoff) → tokens injected into localStorage per test,
// matching the rate-limiter-friendly pattern used across the suite.
async function loginTokens(): Promise<{ accessToken: string; refreshToken: string }> {
  for (let i = 0; i < 6; i++) {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password: TEST_PASSWORD }),
    });
    if (res.status === 429) {
      await new Promise((r) => setTimeout(r, Math.min(2000 * 2 ** i, 20_000)));
      continue;
    }
    if (!res.ok) throw new Error(`login failed: ${res.status}`);
    const d = await res.json();
    return { accessToken: d.access_token, refreshToken: d.refresh_token };
  }
  throw new Error("login rate-limited");
}

/** Classify an observation record by its flat `category` codes. */
function subTypeOf(category: string[] | null): "lab" | "vital" | "social" | "other" {
  for (const raw of category ?? []) {
    const c = String(raw).toLowerCase();
    if (c.includes("vital")) return "vital";
    if (c.includes("social")) return "social";
    if (c.includes("lab")) return "lab";
  }
  return "other";
}

test.describe("Observation sub-type badges + provider surfacing", () => {
  const api = new ApiClient();
  const obsBySubtype: Record<string, string> = {};
  let encounterId = "";
  let procedureId = "";
  let auth: { accessToken: string; refreshToken: string };

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    await api.pollUploadStatus(result.upload_id, 60_000);

    const data = await api.getRecords({ page: 1, page_size: 100 });
    for (const item of data.items) {
      if (item.record_type === "observation") {
        const sub = subTypeOf(item.category);
        if (sub !== "other" && !obsBySubtype[sub]) obsBySubtype[sub] = item.id;
      } else if (item.record_type === "encounter" && !encounterId) {
        encounterId = item.id;
      } else if (item.record_type === "procedure" && !procedureId) {
        procedureId = item.id;
      }
    }

    auth = await loginTokens();
  });

  test.beforeEach(async ({ page }) => {
    await page.addInitScript((a) => {
      localStorage.setItem(
        "medtimeline-auth",
        JSON.stringify({
          state: {
            accessToken: a.accessToken,
            refreshToken: a.refreshToken,
            isAuthenticated: true,
          },
          version: 0,
        })
      );
    }, auth);
  });

  // The header badge on a record-detail page reflects the sub-type.
  for (const [sub, label] of [
    ["lab", "Lab"],
    ["vital", "Vital"],
    ["social", "Social"],
  ] as const) {
    test(`${sub} observation badge reads "${label}", not "Observation"`, async ({ page }) => {
      const id = obsBySubtype[sub];
      if (!id) {
        test.skip();
        return;
      }
      await page.goto(`/records/${id}`);
      await page.waitForSelector("h1", { timeout: 10_000 });

      const badge = page.locator(".badge").first();
      await expect(badge).toBeVisible({ timeout: 5_000 });
      await expect(badge).toHaveText(new RegExp(`^\\s*${label}\\s*$`, "i"));
      await expect(badge).not.toHaveText(/observation/i);
    });
  }

  test("records list shows Lab / Vital / Social badges for observations", async ({ page }) => {
    await page.goto("/records");
    await page.waitForSelector("table.rtable", { timeout: 10_000 });

    // Narrow to observations so the sub-type badges are easy to read.
    await page.selectOption("select.selectbox", "observation");
    await page.waitForLoadState("networkidle");

    const badges = page.locator("table.rtable .badge");
    await expect(badges.first()).toBeVisible({ timeout: 5_000 });
    const texts = (await badges.allTextContents()).map((t) => t.trim().toLowerCase());

    // No generic "observation" badge survives; at least one real sub-type shows.
    expect(texts).not.toContain("observation");
    expect(texts.some((t) => t === "lab" || t === "vital" || t === "social")).toBe(true);
  });

  test("encounter detail surfaces the provider (Dr. Smith)", async ({ page }) => {
    if (!encounterId) {
      test.skip();
      return;
    }
    await page.goto(`/records/${encounterId}`);
    await page.waitForSelector("h1", { timeout: 10_000 });
    const body = await page.textContent("body");
    expect(body).toMatch(/provider/i);
    expect(body).toMatch(/Dr\.\s*Smith/i);
  });

  test("procedure detail surfaces the performer (Dr. Johnson)", async ({ page }) => {
    if (!procedureId) {
      test.skip();
      return;
    }
    await page.goto(`/records/${procedureId}`);
    await page.waitForSelector("h1", { timeout: 10_000 });
    const body = await page.textContent("body");
    expect(body).toMatch(/performer/i);
    expect(body).toMatch(/Dr\.\s*Johnson/i);
  });

  test("timeline surfaces the provider on events that carry one", async ({ page }) => {
    // The encounter event carries provider "Dr. Smith" (server-derived). Filter
    // to Visits so the encounter card is on screen.
    await page.goto("/timeline");
    await page.waitForSelector(".tl-card", { timeout: 10_000 });
    await page.getByRole("button", { name: "Visits" }).click();
    await page.waitForLoadState("networkidle");

    const card = page.locator(".tl-card").filter({ hasText: /Dr\.\s*Smith/i }).first();
    await expect(card).toBeVisible({ timeout: 5_000 });
  });

  test("timeline renders events without a provider gracefully (no stray label)", async ({ page }) => {
    // Conditions have no provider; the cards must render cleanly (the global
    // console-error gate also fails this test on any render error).
    await page.goto("/timeline");
    await page.waitForSelector(".tl-card", { timeout: 10_000 });
    await page.getByRole("button", { name: "Conditions" }).click();
    await page.waitForLoadState("networkidle");
    await expect(page.locator(".tl-card").first()).toBeVisible({ timeout: 5_000 });
    // No provider chips are emitted when provider is null.
    await expect(page.locator(".tl-provider")).toHaveCount(0);
  });
});
