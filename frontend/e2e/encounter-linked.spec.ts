import { test, expect } from "./fixtures/console-gate";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { testEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * "From this visit" — the encounter detail surfaces records linked via
 * linked_encounter_id (GET /records/{id}/linked). Mocked so it's deterministic;
 * runs under the global console-error gate.
 */
test.describe("Encounter 'From this visit'", () => {
  const email = testEmail("encounter-linked");

  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
  });

  const ENC_ID = "11111111-1111-1111-1111-111111111111";

  async function mockEncounter(page: import("@playwright/test").Page, linked: unknown[]) {
    await page.route("**/api/v1/timeline**", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          total: 1,
          events: [
            {
              id: ENC_ID,
              record_type: "encounter",
              display_text: "Office visit",
              effective_date: "2026-03-01T00:00:00Z",
              code_display: "Visit",
              category: ["encounter"],
              provider: "NP Katie",
              preview: { value: null, unit: null, flag: "Ambulatory", emphasis: "normal", gauge: null, facets: [] },
            },
          ],
        }),
      })
    );
    await page.route(`**/api/v1/records/${ENC_ID}`, (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: ENC_ID,
          record_type: "encounter",
          fhir_resource_type: "Encounter",
          fhir_resource: { resourceType: "Encounter", class: { code: "AMB" } },
          display_text: "Office visit",
          effective_date: "2026-03-01T00:00:00Z",
          source_format: "ai_extracted",
          code_value: null,
          code_system: null,
          category: ["encounter"],
          ai_extracted: true,
          confidence_score: 0.9,
          created_at: "2026-03-01T00:00:00Z",
        }),
      })
    );
    await page.route(`**/api/v1/records/${ENC_ID}/linked`, (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(linked) })
    );
  }

  test("shows linked records from the visit", async ({ page }) => {
    await mockEncounter(page, [
      {
        id: "22222222-2222-2222-2222-222222222222",
        record_type: "medication",
        display_text: "omeprazole",
        effective_date: "2026-03-01T00:00:00Z",
        code_display: "omeprazole",
        category: ["medication"],
        provider: null,
        preview: { value: "20 mg", unit: null, flag: "ACTIVE", emphasis: "normal", gauge: null, facets: ["oral"] },
      },
    ]);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");
    await page.locator("button.tl-card", { hasText: "Office visit" }).first().click();

    await expect(page.getByText(/From this visit/)).toBeVisible({ timeout: 10_000 });
    await expect(page.locator(".dv-linked-row", { hasText: "omeprazole" })).toBeVisible();
  });

  test("omits the section when the encounter has no linked records", async ({ page }) => {
    await mockEncounter(page, []);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/timeline");
    await page.locator("button.tl-card", { hasText: "Office visit" }).first().click();

    // Detail opens (title visible) but no "From this visit" section.
    await expect(page.getByRole("heading", { name: "Office visit" })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/From this visit/)).toHaveCount(0);
    await expect(page.locator(".dv-linked-row")).toHaveCount(0);
  });
});
