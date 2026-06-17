import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import JSZip from "jszip";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import { PATHS, hasTestData, testEmail, TEST_PASSWORD } from "./helpers/test-data";

let testIndex = 0;

function uniqueEmail(): string {
  return testEmail(`structured-${++testIndex}-${Date.now()}`);
}

/**
 * Register + login a user via API, then log into the browser session.
 */
async function setupBrowserUser(
  page: import("@playwright/test").Page,
  api: ApiClient
): Promise<string> {
  const email = uniqueEmail();
  await api.register(email, TEST_PASSWORD);
  await api.login(email, TEST_PASSWORD);

  // Log in via the browser so the frontend session is authenticated. Use the
  // shared helper, which retries on the login rate limiter (429) under parallel
  // load — a raw inline login would time out waiting for the redirect.
  await browserLogin(page, email, TEST_PASSWORD);
  return email;
}

test.describe("Structured file uploads", () => {
  test("upload FHIR JSON bundle via dropzone", async ({ page }) => {
    test.setTimeout(120_000);

    const api = new ApiClient();
    await setupBrowserUser(page, api);

    await page.goto("/upload");
    await page.waitForSelector("text=Drop files or a folder");

    const fileInput = page.locator('input[type="file"]').first();
    await fileInput.setInputFiles(PATHS.fhirBundle);

    await expect(page.getByText("sample_fhir_bundle.json")).toBeVisible();
    await page.getByRole("button", { name: "Upload All" }).click();

    // Wait for upload results
    await expect(page.getByText("records inserted")).toBeVisible({
      timeout: 60_000,
    });

    // Verify via API
    const records = await api.getRecords();
    expect(records.items.length).toBeGreaterThan(0);

    const history = await api.getUploadHistory();
    const fhirUpload = history.items.find(
      (item: any) =>
        item.filename === "sample_fhir_bundle.json" &&
        ["completed", "completed_with_merges", "awaiting_review"].includes(
          item.ingestion_status
        )
    );
    expect(fhirUpload).toBeTruthy();
    expect(fhirUpload.record_count).toBeGreaterThan(0);
  });

  test("upload XDM/CDA ZIP package", async ({ page }) => {
    test.skip(
      !hasTestData(PATHS.xdmDir),
      "XDM test data not available — skipping"
    );
    test.setTimeout(120_000);

    const api = new ApiClient();
    await setupBrowserUser(page, api);

    const zip = new JSZip();
    const xdmDir = PATHS.xdmDir;
    const parentDir = path.dirname(xdmDir);

    const parentMetadata = path.join(parentDir, "METADATA.XML");
    if (fs.existsSync(parentMetadata)) {
      zip.file("METADATA.XML", fs.readFileSync(parentMetadata));
    }

    const addDir = (dirPath: string, zipFolder: JSZip) => {
      const entries = fs.readdirSync(dirPath, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
          addDir(fullPath, zipFolder.folder(entry.name)!);
        } else {
          zipFolder.file(entry.name, fs.readFileSync(fullPath));
        }
      }
    };
    addDir(xdmDir, zip.folder("IHE_XDM")!);

    const zipBuffer = await zip.generateAsync({ type: "nodebuffer" });
    const tmpZipPath = path.join(
      process.env.TMPDIR || "/tmp",
      "e2e-xdm-test.zip"
    );
    fs.writeFileSync(tmpZipPath, zipBuffer);

    try {
      await page.goto("/upload");
      await page.waitForSelector("text=Drop files or a folder");

      const fileInput = page.locator('input[type="file"]').first();
      await fileInput.setInputFiles(tmpZipPath);

      await expect(page.getByText("e2e-xdm-test.zip")).toBeVisible();
      await page.getByRole("button", { name: "Upload All" }).click();

      await expect(page.getByText("records inserted")).toBeVisible({
        timeout: 90_000,
      });

      const records = await api.getRecords();
      expect(records.items.length).toBeGreaterThan(0);
    } finally {
      if (fs.existsSync(tmpZipPath)) fs.unlinkSync(tmpZipPath);
    }
  });

  test("upload standalone CDA XML", async () => {
    test.skip(
      !hasTestData(PATHS.cdaExport),
      "CDA export test data not available — skipping"
    );
    test.setTimeout(120_000);

    // API-only test — no browser login needed, avoids rate limiting
    const api = new ApiClient();
    const email = uniqueEmail();
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const cdaDir = PATHS.cdaExport;
    const xmlFiles = fs
      .readdirSync(cdaDir)
      .filter((f) => f.toLowerCase().endsWith(".xml"));
    test.skip(xmlFiles.length === 0, "No XML files found in CDA export dir");

    const xmlPath = path.join(cdaDir, xmlFiles[0]);

    const uploadData = await api.uploadStructured(xmlPath, xmlFiles[0]);
    expect(uploadData.upload_id).toBeTruthy();

    // Poll for completion
    const status = await api.pollUploadStatus(uploadData.upload_id, 90_000);
    expect(
      [
        "completed",
        "completed_with_merges",
        "completed_with_errors",
        "awaiting_review",
        "dedup_scanning",
      ].includes(status.ingestion_status ?? status.status)
    ).toBeTruthy();

    const records = await api.getRecords();
    expect(records.items.length).toBeGreaterThan(0);
  });
});
