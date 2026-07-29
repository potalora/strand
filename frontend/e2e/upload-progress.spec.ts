import { test, expect } from "@playwright/test";
import { ApiClient } from "./helpers/api-client";
import {
  PATHS,
  hasTestData,
  getRtfFiles,
  uniqueEmail,
  TEST_PASSWORD,
} from "./helpers/test-data";

const SUCCESSFUL_UPLOAD_STATUSES = [
  "awaiting_confirmation",
  "completed",
  "completed_with_merges",
  "awaiting_review",
];

test.describe("Extraction progress tracking", () => {
  test.setTimeout(180_000);

  test("extraction progress counts are accurate for batch upload", async () => {
    test.skip(
      !hasTestData(PATHS.rtfDir) || getRtfFiles(3).length < 3,
      "Need at least 3 RTF files in test data"
    );

    const api = new ApiClient();
    const email = uniqueEmail("progress");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const rtfFiles = getRtfFiles(3);
    const files = rtfFiles.map((p) => ({
      path: p,
      name: p.split("/").pop()!,
      mime: "application/rtf",
    }));

    const batch = await api.uploadUnstructuredBatch(files);
    expect(batch.uploads.length).toBeGreaterThanOrEqual(3);
    const uploadIds = batch.uploads.map((upload) => upload.upload_id);

    // Poll extraction progress until all complete or timeout
    const start = Date.now();
    let lastProgress: Awaited<ReturnType<typeof api.getExtractionProgress>> | null = null;

    while (Date.now() - start < 120_000) {
      const progress = await api.getExtractionProgress(uploadIds);
      lastProgress = progress;

      // All done when nothing is pending or processing
      if (
        progress.total === uploadIds.length &&
        progress.pending === 0 &&
        progress.processing === 0
      ) {
        break;
      }

      await new Promise((r) => setTimeout(r, 2000));
    }

    expect(lastProgress).toBeTruthy();
    expect(lastProgress!.total).toBe(uploadIds.length);
    expect(lastProgress!.completed).toBe(uploadIds.length);
    expect(lastProgress!.failed).toBe(0);
    expect(lastProgress!.processing).toBe(0);
    expect(lastProgress!.pending).toBe(0);

    for (const upload of batch.uploads) {
      const status = await api.pollUploadStatus(upload.upload_id, 30_000);
      expect(SUCCESSFUL_UPLOAD_STATUSES).toContain(
        status.ingestion_status ?? status.status
      );
      if (process.env.E2E_LOCAL_ONLY === "1") {
        expect(status.local_run?.privacy_mode).toBe("validated_strict_local");
        expect(status.local_failure?.cloud_fallback_attempted ?? false).toBe(false);
      }
    }
  });
});

test.describe("Mixed content upload classification", () => {
  test.setTimeout(120_000);

  test("structured upload inserts records", async () => {
    const api = new ApiClient();
    const email = uniqueEmail("progress-mixed");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const result = await api.uploadStructured(
      PATHS.fhirBundle,
      "sample_fhir_bundle.json"
    );
    expect(result.upload_id).toBeTruthy();

    const status = await api.pollUploadStatus(result.upload_id, 60_000);
    expect(SUCCESSFUL_UPLOAD_STATUSES).toContain(
      status.ingestion_status ?? status.status
    );
    // Structured upload should insert records directly
    const records = await api.getRecords();
    expect(records.items.length).toBeGreaterThan(0);
  });

  test("unstructured upload goes to extraction pipeline", async () => {
    test.skip(
      !hasTestData(PATHS.rtfDir) || getRtfFiles(1).length < 1,
      "Need at least 1 RTF file in test data"
    );

    const api = new ApiClient();
    const email = uniqueEmail("progress-unstruct");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const rtfFiles = getRtfFiles(1);
    const files = rtfFiles.map((p) => ({
      path: p,
      name: p.split("/").pop()!,
      mime: "application/rtf",
    }));

    const batch = await api.uploadUnstructuredBatch(files);
    const uploadIds = batch.uploads.map((upload) => upload.upload_id);
    expect(uploadIds).toHaveLength(files.length);

    const status = await api.pollUploadStatus(uploadIds[0], 90_000);
    expect(SUCCESSFUL_UPLOAD_STATUSES).toContain(
      status.ingestion_status ?? status.status
    );
    if (process.env.E2E_LOCAL_ONLY === "1") {
      expect(status.local_run?.privacy_mode).toBe("validated_strict_local");
      expect(status.local_failure?.cloud_fallback_attempted ?? false).toBe(false);
    }

    const progress = await api.getExtractionProgress(uploadIds);
    expect(progress).toMatchObject({
      total: uploadIds.length,
      completed: uploadIds.length,
      processing: 0,
      failed: 0,
      pending: 0,
    });
  });
});
