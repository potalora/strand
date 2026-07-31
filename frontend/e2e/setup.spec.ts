import { test, expect } from "@playwright/test";
import { ApiClient } from "./helpers/api-client";
import { uniqueEmail, TEST_PASSWORD, PATHS } from "./helpers/test-data";

const email = uniqueEmail("setup");
const SUCCESSFUL_UPLOAD_STATUSES = [
  "awaiting_confirmation",
  "completed",
  "completed_with_merges",
  "awaiting_review",
];

test.describe("E2E Setup", () => {
  const api = new ApiClient();

  test.beforeAll(async () => {
    await api.register(email, TEST_PASSWORD);
    await api.login(email, TEST_PASSWORD);
  });

  test("test account is authenticated", async () => {
    const me = await api.getMe();
    expect(me.email).toBeTruthy();
  });

  test("local-only profile persists validated strict-local routing", async () => {
    test.skip(process.env.E2E_LOCAL_ONLY !== "1", "Local-only profile is not active");
    const settings = await api.getLlmSettings();
    expect(settings.routing.processing_mode).toBe("validated_strict_local");
  });

  test("fixture data can be uploaded and ingested", async () => {
    const result = await api.uploadStructured(PATHS.fhirBundle, "sample_fhir_bundle.json");
    expect(result.upload_id).toBeTruthy();
    const status = await api.pollUploadStatus(result.upload_id, 60_000);
    expect(SUCCESSFUL_UPLOAD_STATUSES).toContain(
      status.ingestion_status ?? status.status
    );
  });

  test("multiple record types were created", async () => {
    const records = await api.getRecords({ page: 1 });
    const types = new Set(records.items.map((record) => record.record_type));
    // We expect at least conditions, observations, medications, encounters
    expect(types.size).toBeGreaterThanOrEqual(4);
  });
});
