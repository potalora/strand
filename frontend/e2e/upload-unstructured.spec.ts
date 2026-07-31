import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import { ApiClient } from "./helpers/api-client";
import { browserLogin } from "./helpers/browser-login";
import {
  PATHS,
  getRtfFiles,
  uniqueEmail,
  TEST_PASSWORD,
  TEST_DATA_DIR,
} from "./helpers/test-data";

/**
 * Find a PDF file in test_data directories for PDF upload test.
 */
function findPdf(): string | null {
  const searchDirs = [
    PATHS.rtfDir, // might have PDFs alongside RTFs
    PATHS.epicExport,
    TEST_DATA_DIR,
  ];
  for (const dir of searchDirs) {
    if (!fs.existsSync(dir)) continue;
    try {
      const entries = fs.readdirSync(dir, { recursive: true }) as string[];
      for (const entry of entries) {
        if (entry.toLowerCase().endsWith(".pdf")) {
          return path.join(dir, entry);
        }
      }
    } catch {
      // directory might not support recursive read
      const files = fs.readdirSync(dir);
      for (const f of files) {
        if (f.toLowerCase().endsWith(".pdf")) {
          return path.join(dir, f);
        }
      }
    }
  }
  return null;
}

function mimeForExt(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".rtf":
      return "application/rtf";
    case ".pdf":
      return "application/pdf";
    case ".tif":
    case ".tiff":
      return "image/tiff";
    default:
      return "application/octet-stream";
  }
}

function expectLocalOnlyStatus(
  status: Awaited<ReturnType<ApiClient["pollUploadStatus"]>>
): void {
  if (process.env.E2E_LOCAL_ONLY !== "1") return;
  expect(status.local_run?.privacy_mode).toBe("validated_strict_local");
  expect(status.local_failure?.cloud_fallback_attempted ?? false).toBe(false);
}

test.describe("Unstructured Upload", () => {
  let api: ApiClient;
  let email: string;

  // Fresh user per test: the backend now skips re-ingesting identical file content
  // for the same user (duplicate_file). A unique user per test ensures each upload is
  // genuinely extracted rather than detected as a duplicate of an earlier test/run.
  test.beforeEach(async () => {
    api = new ApiClient();
    email = uniqueEmail("unstructured");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
  });

  test("upload single RTF via API", async () => {
    const allRtf = getRtfFiles(40);
    test.skip(allRtf.length === 0, "RTF test data not available");
    test.setTimeout(300_000);

    // Pick a small RTF (not the large multi-section one) for faster, more reliable testing
    const filePath = allRtf.length > 1 ? allRtf[1] : allRtf[0];
    const fileName = path.basename(filePath);

    // Upload via API
    const result = await api.uploadUnstructuredBatch([
      { path: filePath, name: fileName, mime: "application/rtf" },
    ]);
    expect(result.uploads).toHaveLength(1);
    const uploadId = result.uploads[0].upload_id;
    expect(uploadId).toBeTruthy();

    // Poll until extraction completes.
    const status = await api.pollUploadStatus(uploadId, 270_000);
    const st = status.ingestion_status ?? status.status;
    expect(["awaiting_confirmation", "completed", "completed_with_merges", "awaiting_review"]).toContain(st);
    expectLocalOnlyStatus(status);

    // Verify entities were extracted
    const progress = await api.getExtractionProgress();
    expect(progress.total).toBeGreaterThanOrEqual(1);
  });

  test("upload batch RTFs via UI", async ({ page }) => {
    const rtfFiles = getRtfFiles(3);
    test.skip(rtfFiles.length === 0, "RTF test data not available");
    test.setTimeout(300_000);

    // Login via UI (with rate-limit retry)
    await browserLogin(page, email, TEST_PASSWORD);

    // Navigate to upload page
    await page.goto("/upload");
    await page.waitForLoadState("networkidle");

    // Set files on the hidden file input (dropzone input)
    const fileInput = page.locator('input[type="file"]:not([webkitdirectory])');
    await fileInput.setInputFiles(rtfFiles);

    // Wait for file list to appear, then click Upload All
    await page.waitForSelector("text=Upload All", { timeout: 10_000 });
    const batchResponsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname.endsWith(
          "/api/v1/upload/unstructured-batch"
        )
    );
    await page.getByRole("button", { name: "Upload all" }).click();
    const batchResponse = await batchResponsePromise;
    expect(batchResponse.ok()).toBeTruthy();
    const batchResult = (await batchResponse.json()) as {
      uploads: { upload_id: string }[];
    };
    expect(batchResult.uploads).toHaveLength(rtfFiles.length);
    const uploadIds = batchResult.uploads.map((upload) => upload.upload_id);

    // The response is returned only after every batch row is committed. Verify
    // those exact IDs rather than racing a wall-clock history window.
    const history = await api.getUploadHistory();
    const historyIds = new Set(history.items.map((upload) => upload.id));
    expect(uploadIds.every((uploadId) => historyIds.has(uploadId))).toBe(true);

    // Poll all uploads concurrently.
    const statuses = await Promise.all(
      uploadIds.map((uploadId) => api.pollUploadStatus(uploadId, 240_000))
    );
    for (const status of statuses) {
      const st = status.ingestion_status ?? status.status;
      expect(["awaiting_confirmation", "completed", "completed_with_merges", "awaiting_review"]).toContain(st);
      expectLocalOnlyStatus(status);
    }
  });

  test("upload PDF via API", async () => {
    const pdfPath = findPdf();
    test.skip(!pdfPath, "No PDF test data found");
    test.setTimeout(300_000);

    const fileName = path.basename(pdfPath!);
    const result = await api.uploadUnstructuredBatch([
      { path: pdfPath!, name: fileName, mime: "application/pdf" },
    ]);
    expect(result.uploads).toHaveLength(1);
    const uploadId = result.uploads[0].upload_id;

    // PDF extraction includes local rasterization and one sandboxed OCR job per page.
    const status = await api.pollUploadStatus(uploadId, 270_000);
    const st = status.ingestion_status ?? status.status;
    expect(["awaiting_confirmation", "completed", "completed_with_merges", "awaiting_review"]).toContain(st);
    expectLocalOnlyStatus(status);
  });

  test("extraction progress tracking via API", async () => {
    // Use smaller RTFs to avoid the large multi-section file that's slow under parallel contention
    const allRtf = getRtfFiles(40);
    test.skip(allRtf.length === 0, "RTF test data not available");
    test.setTimeout(300_000);

    // Pick 3 smaller RTFs (skip the first one which is the largest)
    const rtfFiles = allRtf.length > 3 ? allRtf.slice(1, 4) : allRtf.slice(0, 3);
    const files = rtfFiles.map((f) => ({
      path: f,
      name: path.basename(f),
      mime: mimeForExt(f),
    }));
    const result = await api.uploadUnstructuredBatch(files);
    expect(result.uploads).toHaveLength(files.length);

    const uploadIds = result.uploads.map((u) => u.upload_id);

    // Poll all uploads concurrently for completion
    const statuses = await Promise.all(
      uploadIds.map((id) => api.pollUploadStatus(id, 270_000))
    );

    for (const status of statuses) {
      const st = status.ingestion_status ?? status.status;
      expect(["awaiting_confirmation", "completed", "completed_with_merges", "awaiting_review"]).toContain(st);
      expectLocalOnlyStatus(status);
    }

    // Verify extraction progress reflects the completed files
    const progress = await api.getExtractionProgress();
    expect(progress.total).toBeGreaterThanOrEqual(files.length);
    expect(progress.completed).toBeGreaterThanOrEqual(files.length);
  });
});
