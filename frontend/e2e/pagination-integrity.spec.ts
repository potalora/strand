import { test, expect } from "./fixtures/console-gate";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { uniqueEmail, TEST_PASSWORD } from "./helpers/test-data";

/**
 * Large-dataset pagination integrity (bug #1).
 *
 * Backend `GET /records` used to order only by effective_date (+ type) with no
 * stable `id` tiebreaker. With >100 records (the page-size cap) and large tie
 * groups straddling a page boundary, a row could be returned on two pages or
 * dropped between them. Admin → Records pages the whole set at page_size=100 and
 * accumulates, so a duplicated row produced two React children with the same
 * `key` — a `console.error` ("Encountered two children with the same key").
 *
 * This seeds 250 records engineered for that failure mode: unique titles (so no
 * dedup merges them) but effective_dates drawn from a tiny set plus many nulls,
 * making giant tie groups that cross the 100/200 page boundaries. Then it opens
 * Admin → Records and asserts:
 *   • zero console errors / React duplicate-key warnings (the console gate);
 *   • zero duplicate rows (unique row titles == rendered rows);
 *   • the rendered row count equals the reported total (no dropped rows).
 *
 * On the pre-fix ordering this fails (duplicate-key warning + a missing/duplicated
 * row); with the `id` tiebreaker it passes.
 */

const SEED_COUNT = 250;

function buildBundle(): object {
  const entries: object[] = [
    {
      resource: {
        resourceType: "Patient",
        id: "e2e-pagination-patient",
        name: [{ family: "Pagey", given: ["Tessa"] }],
        gender: "female",
        birthDate: "1985-05-05",
      },
    },
  ];

  // Tie shape: 120 share date A (crosses the 100→200 boundary), 80 share date B,
  // 50 are dateless (null effective_date). All sorted desc → huge tie groups.
  const DATE_A = "2021-04-10";
  const DATE_B = "2018-09-01";

  for (let i = 0; i < SEED_COUNT; i++) {
    const n = String(i + 1).padStart(3, "0");
    let onset: string | undefined;
    if (i < 120) onset = DATE_A;
    else if (i < 200) onset = DATE_B;
    else onset = undefined; // null effective_date

    const resource: Record<string, unknown> = {
      resourceType: "Condition",
      id: `e2e-pag-cond-${n}`,
      subject: { reference: "Patient/e2e-pagination-patient" },
      code: {
        coding: [
          {
            system: "http://e2e.medtimeline.test/codes",
            code: `E2EP-${n}`,
            display: `E2E Pagination Condition ${n}`,
          },
        ],
        text: `E2E Pagination Condition ${n}`,
      },
      clinicalStatus: {
        coding: [
          {
            system:
              "http://terminology.hl7.org/CodeSystem/condition-clinical",
            code: "active",
          },
        ],
      },
    };
    if (onset) resource.onsetDateTime = onset;
    entries.push({ resource });
  }

  return { resourceType: "Bundle", type: "searchset", entry: entries };
}

test.describe("Large-dataset pagination integrity (Admin → Records)", () => {
  test("250 records with tie groups page without duplicate or dropped rows", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    const api = new ApiClient();
    const email = uniqueEmail("pagination");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    // Seed via a temp bundle file.
    const bundlePath = path.join(
      os.tmpdir(),
      `e2e-pagination-${Date.now()}.json`
    );
    fs.writeFileSync(bundlePath, JSON.stringify(buildBundle()));
    try {
      const up = await api.uploadStructured(bundlePath, "pagination-seed.json");
      await api.pollUploadStatus(up.upload_id, 90_000);
    } finally {
      fs.existsSync(bundlePath) && fs.unlinkSync(bundlePath);
    }

    // Ground-truth total from the API.
    const meta = await api.getRecords({ page: 1, page_size: 1 });
    const expectedTotal: number = meta.total;
    expect(expectedTotal).toBeGreaterThanOrEqual(SEED_COUNT);

    await browserLogin(page, email, TEST_PASSWORD);
    await page.goto("/admin");

    // Wait for the Records tab to finish accumulating every page.
    await expect(page.locator("tr.clickable").first()).toBeVisible({
      timeout: 30_000,
    });
    await expect
      .poll(async () => page.locator("tr.clickable").count(), {
        timeout: 30_000,
      })
      .toBeGreaterThanOrEqual(expectedTotal);

    const renderedCount = await page.locator("tr.clickable").count();

    // Each row's delete control carries a unique aria-label ("Delete <title>").
    const labels = await page
      .locator("button.row-del")
      .evaluateAll((els) => els.map((e) => e.getAttribute("aria-label") || ""));
    const unique = new Set(labels);

    // No row rendered twice (the duplicate-key symptom of #1).
    expect(
      labels.length - unique.size,
      "duplicate rows detected in the paged table"
    ).toBe(0);

    // Rendered rows match the reported total — nothing dropped between pages.
    expect(renderedCount).toBe(expectedTotal);
    expect(unique.size).toBe(expectedTotal);

    // (The console-error gate also fails this test if React logged a
    // duplicate-key warning while accumulating pages.)
  });
});
