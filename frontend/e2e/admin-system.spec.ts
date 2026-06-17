import { test, expect } from "./fixtures/console-gate";
import { browserLogin } from "./helpers/browser-login";
import { ApiClient } from "./helpers/api-client";
import { testEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const email = testEmail("admin-system");

/**
 * Repaired for the consolidated Admin → System tab (Settings folded in). Tabs use
 * role="tab"; the pane now shows an Account card, a "This record" stats card, a
 * Preferences card (Appearance + Delete confirmation), data-export + Sign out,
 * and an audit log. The old "Patients/Uploads/Active" stats and the raw user-UUID
 * field no longer exist.
 */
test.describe("Admin — System tab", () => {
  test.beforeAll(async () => {
    const api = new ApiClient();
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
    await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
  });

  test("account info renders", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("heading", { name: "Account" })).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByText("E2E Test User")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/@test\.com/)).toBeVisible();
  });

  test("this-record statistics render", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("heading", { name: "This record" })).toBeVisible({
      timeout: 10_000,
    });
    // Scope to the field labels in the "This record" card ("Records" also names
    // the admin tab and appears in audit-log rows).
    await expect(page.locator(".field-l").filter({ hasText: /^Records$/ })).toBeVisible();
    await expect(page.locator(".field-l").filter({ hasText: /^Sources$/ })).toBeVisible();
    await expect(page.locator(".field-l").filter({ hasText: /^Span$/ })).toBeVisible();
  });

  test("preferences section renders (folded in from /settings)", async ({
    page,
  }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=sys");

    await expect(
      page.getByRole("heading", { name: "Preferences" })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Appearance")).toBeVisible();
    await expect(page.getByText("Delete confirmation")).toBeVisible();
  });

  test("export and sign-out controls render", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=sys");

    await expect(
      page.getByRole("button", { name: /Export all \(FHIR\)/ })
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.locator("main").getByRole("button", { name: "Sign out" })
    ).toBeVisible();
  });

  test("sign out from System redirects to /login", async ({ page }) => {
    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin?tab=sys");

    const signOut = page.locator("main").getByRole("button", { name: "Sign out" });
    await expect(signOut).toBeVisible({ timeout: 10_000 });
    await signOut.click();

    await expect(page).toHaveURL(/\/login/, { timeout: 10_000 });
  });
});
