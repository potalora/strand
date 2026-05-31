import { test, expect } from "@playwright/test";
import * as path from "path";
import { ApiClient } from "./helpers/api-client";
import { getRtfFiles, uniqueEmail, TEST_PASSWORD } from "./helpers/test-data";

test.describe("Duplicate file upload (idempotency)", () => {
  test("re-uploading identical unstructured file returns duplicate_file", async () => {
    const rtf = getRtfFiles(40);
    test.skip(rtf.length === 0, "RTF test data not available");
    test.setTimeout(300_000);

    // Pick a small RTF (skip the first/largest) for a faster, more reliable extraction.
    const filePath = rtf.length > 1 ? rtf[1] : rtf[0];
    const fileName = path.basename(filePath);

    const api = new ApiClient();
    const email = uniqueEmail("dupfile");
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);

    // First upload of file X — extract to a produced-records terminal status.
    const first = await api.uploadUnstructuredBatch([
      { path: filePath, name: fileName, mime: "application/rtf" },
    ]);
    expect(first.uploads).toHaveLength(1);
    const firstId = first.uploads[0].upload_id;
    expect(firstId).toBeTruthy();

    const firstStatus = await api.pollUploadStatus(firstId, 270_000);
    const firstSt = firstStatus.ingestion_status ?? firstStatus.status;
    // The first upload should genuinely process, not be flagged as a duplicate.
    expect(
      ["awaiting_confirmation", "completed", "completed_with_merges", "awaiting_review"]
    ).toContain(firstSt);

    // Second upload of the SAME file X by the same user — backend idempotency
    // should short-circuit re-ingestion and mark it as a duplicate.
    const second = await api.uploadUnstructuredBatch([
      { path: filePath, name: fileName, mime: "application/rtf" },
    ]);
    expect(second.uploads).toHaveLength(1);
    const secondId = second.uploads[0].upload_id;
    expect(secondId).toBeTruthy();

    const secondStatus = await api.pollUploadStatus(secondId, 270_000);
    const secondSt = secondStatus.ingestion_status ?? secondStatus.status;
    expect(secondSt).toBe("duplicate_file");
  });
});
