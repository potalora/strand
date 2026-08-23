import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

test("loads ingestion evidence only for an AI-extracted record", async ({
  page,
}) => {
  await injectAuth(page);
  let evidenceCalls = 0;

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/records/record-1") {
      return json({
        id: "record-1",
        patient_id: "patient-1",
        record_type: "medication",
        fhir_resource_type: "MedicationRequest",
        fhir_resource: {
          resourceType: "MedicationRequest",
          medicationCodeableConcept: { text: "Metformin" },
        },
        source_format: "local_ai",
        effective_date: "2025-01-10T00:00:00Z",
        status: "active",
        category: [],
        code_system: null,
        code_value: null,
        code_display: "Metformin",
        display_text: "Metformin 500 mg twice daily",
        ai_extracted: true,
        confidence_score: 0.98,
        created_at: "2025-01-10T00:00:00Z",
      });
    }
    if (url.pathname === "/api/v1/records/record-1/evidence") {
      evidenceCalls += 1;
      return json({
        record_id: "record-1",
        processing_mode: "validated_strict_local",
        schema_version: "clinical-document-extraction.v1",
        evidence: [
          {
            id: "evidence-1",
            page_number: 3,
            section: "Medications",
            excerpt: "Metformin 500 mg twice daily",
            start_offset: 120,
            end_offset: 149,
            field_paths: [
              "$.medications[0].name",
              "$.medications[0].dose",
            ],
          },
        ],
        unresolved_fields: ["$.medications[0].start_date"],
        rejected_fields: [],
        models: [
          {
            role: "ocr",
            repository: "sahilchachra/ovisocr2-int4-mlx",
            revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
            quantization: "int4",
            runtime: "mlx-vlm 0.5.0",
          },
          {
            role: "extraction",
            repository: "numind/NuExtract3-mlx-4bits",
            revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
            quantization: "4bit",
            runtime: "mlx-vlm 0.5.0",
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        login_identifier: "pedro@example.com",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/records/record-1");
  await expect(
    page.getByRole("heading", { name: "Extraction evidence" })
  ).toBeVisible();
  await expect(page.getByText("Page 3")).toBeVisible();
  await expect(
    page
      .getByLabel("Extraction evidence")
      .getByText("Metformin 500 mg twice daily")
  ).toBeVisible();
  await expect(page.getByText("NuExtract3")).toBeVisible();
  await expect(page.getByText(/Qwen3\.5-9B/)).toHaveCount(0);
  await expect.poll(() => evidenceCalls).toBeGreaterThan(0);
});

test("does not request extraction evidence for a structured record", async ({
  page,
}) => {
  await injectAuth(page);
  let evidenceCalls = 0;

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith("/evidence")) evidenceCalls += 1;
    const body = url.pathname === "/api/v1/auth/me"
      ? {
          id: "user-1",
          login_identifier: "pedro@example.com",
          email: "pedro@example.com",
          display_name: "Pedro",
          is_active: true,
          created_at: "2024-01-01T00:00:00Z",
        }
      : url.pathname === "/api/v1/records/record-2"
        ? {
            id: "record-2",
            patient_id: "patient-1",
            record_type: "condition",
            fhir_resource_type: "Condition",
            fhir_resource: { resourceType: "Condition", code: { text: "Asthma" } },
            source_format: "fhir",
            effective_date: null,
            status: "active",
            category: [],
            code_system: null,
            code_value: null,
            code_display: "Asthma",
            display_text: "Asthma",
            ai_extracted: false,
            confidence_score: null,
            created_at: "2025-01-10T00:00:00Z",
          }
        : {};
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });

  await page.goto("/records/record-2");
  await expect(page.getByText("Asthma").first()).toBeVisible();
  await expect.poll(() => evidenceCalls).toBe(0);
});
