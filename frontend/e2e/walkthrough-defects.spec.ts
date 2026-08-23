import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const RECORD = {
  id: "record-1",
  patient_id: "patient-1",
  user_id: "user-1",
  record_type: "condition",
  fhir_resource: {
    resourceType: "Condition",
    code: { text: "Calendar-day regression" },
  },
  effective_date: "2019-01-01T00:00:00Z",
  code_system: null,
  code_value: null,
  code_display: null,
  display_text: "Calendar-day regression",
  status: "active",
  source_format: "fhir",
  source_file_id: null,
  ai_extracted: false,
  confidence_score: null,
  is_duplicate: false,
  created_at: "2019-01-01T02:00:00Z",
  updated_at: "2019-01-01T02:00:00Z",
};

async function setupRecordSheet(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const json = (body: unknown) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (path === "/api/v1/local-ai/jobs") return json([]);
    if (path === "/api/v1/upload/history")
      return json({ items: [], total: 0 });
    if (path === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        login_identifier: "pedro@example.com",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2026-01-01T00:00:00Z",
      });
    }
    if (path === "/api/v1/dashboard/overview") {
      return json({
        total_records: 1,
        total_patients: 1,
        total_uploads: 0,
        records_by_type: { condition: 1 },
        recent_records: [],
        date_range_start: RECORD.effective_date,
        date_range_end: RECORD.effective_date,
      });
    }
    if (path === "/api/v1/records/record-1") return json(RECORD);
    if (path === "/api/v1/records") {
      return json({
        items: [RECORD],
        total: 1,
        page: 1,
        page_size: 100,
      });
    }
    return json({});
  });
}

test.use({ timezoneId: "America/New_York" });

test("record sheet dialog has an accessible description", async ({
  page,
}) => {
  await setupRecordSheet(page);
  await page.goto("/admin?tab=records");
  await page.locator("tr.clickable").click();

  const dialog = page.getByRole("dialog");
  await expect(dialog).toHaveAccessibleDescription(
    "Clinical record details and actions."
  );
});

test("record sheet preserves clinical days while timestamps remain local", async ({
  page,
}) => {
  await setupRecordSheet(page);
  await page.goto("/admin?tab=records");
  await page.locator("tr.clickable").click();

  const dialog = page.getByRole("dialog");
  await expect(
    dialog.locator(".field").filter({ hasText: "Date" })
  ).toContainText("Jan 1, 2019");
  await expect(
    dialog.locator(".field").filter({ hasText: "Added" })
  ).toContainText("Dec 31, 2018");
});
