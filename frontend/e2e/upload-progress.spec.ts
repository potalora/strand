import { test, expect } from "@playwright/test";
import { ApiClient } from "./helpers/api-client";
import {
  PATHS,
  hasTestData,
  getRtfFiles,
  testEmail,
  TEST_PASSWORD,
} from "./helpers/test-data";

test.describe("Extraction progress tracking", () => {
  test.setTimeout(180_000);

  test("extraction progress counts are accurate for batch upload", async () => {
    test.skip(
      !hasTestData(PATHS.rtfDir) || getRtfFiles(3).length < 3,
      "Need at least 3 RTF files in test data"
    );

    const api = new ApiClient();
    const email = testEmail("progress");
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

    // Poll extraction progress until all complete or timeout
    const start = Date.now();
    let lastProgress: Awaited<ReturnType<typeof api.getExtractionProgress>> | null = null;
    let completedIncreased = false;
    let initialCompleted = 0;

    while (Date.now() - start < 120_000) {
      const progress = await api.getExtractionProgress();
      lastProgress = progress;

      if (!completedIncreased && progress.completed > initialCompleted) {
        completedIncreased = true;
        initialCompleted = progress.completed;
      }

      // All done when nothing is pending or processing
      if (
        progress.total >= 3 &&
        progress.pending === 0 &&
        progress.processing === 0
      ) {
        break;
      }

      await new Promise((r) => setTimeout(r, 2000));
    }

    expect(lastProgress).toBeTruthy();
    expect(lastProgress!.total).toBeGreaterThanOrEqual(3);
    // All should have completed or failed (no stuck files)
    // processing may be > 0 if other tests' files are still being extracted
    expect(lastProgress!.completed + lastProgress!.failed).toBeGreaterThanOrEqual(3);
    expect(lastProgress!.pending).toBe(0);
  });
});

test.describe("Mixed content upload classification", () => {
  test.setTimeout(120_000);

  test("structured upload inserts records", async () => {
    const api = new ApiClient();
    const email = testEmail("progress-mixed");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const result = await api.uploadStructured(
      PATHS.fhirBundle,
      "sample_fhir_bundle.json"
    );
    expect(result.upload_id).toBeTruthy();

    await api.pollUploadStatus(result.upload_id, 60_000);
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
    const email = testEmail("progress-unstruct");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    const rtfFiles = getRtfFiles(1);
    const files = rtfFiles.map((p) => ({
      path: p,
      name: p.split("/").pop()!,
      mime: "application/rtf",
    }));

    await api.uploadUnstructuredBatch(files);

    // Check that extraction progress shows the file
    // Give it a moment to register
    await new Promise((r) => setTimeout(r, 2000));
    const progress = await api.getExtractionProgress();
    expect(progress.total).toBeGreaterThanOrEqual(1);
  });
});
