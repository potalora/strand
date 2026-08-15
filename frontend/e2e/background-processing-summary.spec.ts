import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const json = (body: unknown) => ({
  contentType: "application/json",
  body: JSON.stringify(body),
});

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

test("strict summary 202 registers a durable background card and clears loading", async ({
  page,
}) => {
  await injectAuth(page);
  let accepted = false;
  const acceptedJob = {
    id: "summary-job-1",
    upload_id: null,
    summary_prompt_id: "prompt-1",
    kind: "summary",
    processing_mode: "validated_strict_local",
    status: "queued",
    stage: "queued",
    progress: null,
    failure: null,
    cancel_requested: false,
    created_at: "2026-07-31T00:00:00Z",
    updated_at: "2026-07-31T00:00:00Z",
    started_at: null,
    completed_at: null,
  };
  const acceptedSummary = {
    id: "prompt-1",
    job_id: "summary-job-1",
    processing_mode: "validated_strict_local",
    kind: "summary",
    status: "queued",
    stage: "queued",
    created_at: "2026-07-31T00:00:00Z",
  };

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;

    // Background job routes must precede the broad API fallback.
    if (path === "/api/v1/local-ai/jobs") {
      return route.fulfill(json(accepted ? [acceptedJob] : []));
    }
    if (path === "/api/v1/local-ai/jobs/summary-job-1") {
      return route.fulfill(json(acceptedJob));
    }
    if (path === "/api/v1/auth/me") {
      return route.fulfill(
        json({
          id: "user-1",
          email: "pedro@example.com",
          display_name: "Pedro",
          is_active: true,
          created_at: "2026-01-01T00:00:00Z",
        })
      );
    }
    if (path === "/api/v1/dashboard/patients") {
      return route.fulfill(
        json({
          items: [
            {
              id: "patient-1",
              fhir_id: "patient-1",
              gender: null,
              name: "Ada Lovelace",
              birth_date: null,
            },
            {
              id: "patient-2",
              fhir_id: "internal-fhir-id",
              gender: null,
              name: null,
              birth_date: null,
            },
          ],
        })
      );
    }
    if (path === "/api/v1/settings/llm") {
      return route.fulfill(
        json({
          providers: [],
          routing: {
            default: "gemini",
            summary: "gemini",
            section: "gemini",
            dedup: "gemini",
            extraction: "gemini",
            vision: "gemini",
            extraction_engine: "local",
            processing_mode: "validated_strict_local",
          },
        })
      );
    }
    if (path === "/api/v1/local-ai/status") {
      return route.fulfill(
        json({
          platform: "apple_silicon",
          compatible: true,
          enabled: true,
          can_manage_pack: false,
          state: "ready",
          status_reason: null,
          active_revision: "v1",
          available_revision: "v1",
          models: [
            {
              role: "summary",
              repository: "Qwen/Qwen3.5-9B",
              revision: "fixed",
              quantization: "4bit",
              runtime: "mlx",
              license: "apache-2.0",
              download_bytes: 1,
              expected_memory_bytes: 1,
              installed: true,
              validated: true,
            },
          ],
          operation: null,
        })
      );
    }
    if (path === "/api/v1/summary/prompts") {
      return route.fulfill(json({ items: [], total: 0 }));
    }
    if (
      path === "/api/v1/summary/generate" &&
      request.method() === "POST"
    ) {
      accepted = true;
      return route.fulfill({ status: 202, ...json(acceptedSummary) });
    }
    return route.fulfill(json({}));
  });

  await page.goto("/summaries");
  const subject = page.getByLabel("Record subject");
  await expect(subject.locator("option")).toHaveText([
    "Ada Lovelace",
    "Record subject 2",
  ]);
  await expect(subject).not.toContainText("patient-1");
  await expect(subject).not.toContainText("internal-fhir-id");
  const generate = page.getByRole("button", { name: "Generate summary" });
  await expect(generate).toBeEnabled();
  await generate.click();

  await expect(generate).toBeEnabled();
  await expect(
    page.getByText("Summary is processing in the background. You can leave this page.")
  ).toBeVisible();
  const monitor = page.getByRole("region", { name: "Background processing" });
  await expect(monitor).toContainText("Summary");
  await expect(page.getByText("Generating…")).toHaveCount(0);
});
