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

test("stamps strict-local mode and shows fail-closed model progress", async ({
  page,
}) => {
  await injectAuth(page);
  let uploadedMode: string | null = null;
  let statusPolls = 0;

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/settings/llm") {
      return json({
        providers: [],
        routing: {
          default: "gemini",
          summary: "gemini",
          section: "gemini",
          dedup: "gemini",
          extraction: "gemini",
          vision: "gemini",
          extraction_engine: "hybrid",
          processing_mode: "validated_strict_local",
        },
      });
    }
    if (
      url.pathname === "/api/v1/upload/unstructured" &&
      request.method() === "POST"
    ) {
      uploadedMode = request.postData()?.includes("validated_strict_local")
        ? "validated_strict_local"
        : null;
      return json({
        upload_id: "upload-1",
        status: "pending_extraction",
        file_type: "pdf",
      });
    }
    if (url.pathname === "/api/v1/upload/extraction-progress") {
      return json({
        total: 1,
        completed: 0,
        processing: 1,
        failed: 0,
        pending: 0,
        records_created: 0,
      });
    }
    if (url.pathname === "/api/v1/upload/pending-extraction") {
      statusPolls += 1;
      return json({
        files: [
          {
            id: "upload-1",
            filename: "record.pdf",
            ingestion_status: "processing",
            progress_stage: "ocr",
            progress_detail: {
              page_index: 2,
              page_total: 9,
              model_role: "ocr",
              repository: "sahilchachra/ovisocr2-int4-mlx",
              revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
            },
            local_run: {
              privacy_mode: "validated_strict_local",
              models: [
                {
                  role: "ocr",
                  repository: "sahilchachra/ovisocr2-int4-mlx",
                  revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
                },
              ],
            },
            local_failure: {
              stage: "ocr",
              code: "local_worker_failed",
              message: "Local OCR stopped before this page completed.",
              model_role: "ocr",
              repository: "sahilchachra/ovisocr2-int4-mlx",
              revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
              retryable: true,
              checkpoint_preserved: true,
              cloud_fallback_attempted: false,
            },
          },
        ],
        total: 1,
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/upload");

  await expect(page.getByText("Application-layer encrypted at rest")).toBeVisible();
  await expect(page.getByText("End-to-end encrypted")).toHaveCount(0);
  const fileInput = page.locator('input[type="file"]').first();
  await fileInput.setInputFiles({
    name: "record.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 strict local fixture"),
  });
  await page.getByRole("button", { name: /upload all/i }).click();

  await expect.poll(() => uploadedMode).toBe("validated_strict_local");
  await expect.poll(() => statusPolls).toBeGreaterThan(0);
  await expect(page.getByText(/Local OCR.*page 2 of 9/)).toBeVisible();
  await expect(page.getByText("OvisOCR2")).toBeVisible();
  await expect(page.getByText("Cloud fallback was not attempted")).toBeVisible();
  await expect(page.getByText("Checkpoint preserved")).toBeVisible();
});

test("blocks every upload until privacy settings load and makes failure retryable", async ({
  page,
}) => {
  await injectAuth(page);
  let settingsUnavailable = true;
  let uploadRequests = 0;

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/settings/llm") {
      if (settingsUnavailable) {
        return json({ detail: "settings unavailable" }, 503);
      }
      return json({
        providers: [],
        routing: {
          default: "gemini",
          summary: "gemini",
          section: "gemini",
          dedup: "gemini",
          extraction: "gemini",
          vision: "gemini",
          extraction_engine: "hybrid",
          processing_mode: "cloud_assisted",
        },
      });
    }
    if (
      url.pathname === "/api/v1/upload/unstructured" &&
      request.method() === "POST"
    ) {
      uploadRequests += 1;
      return json({
        upload_id: "upload-after-retry",
        status: "pending_extraction",
        file_type: "pdf",
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/upload");
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "blocked.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 blocked until mode resolves"),
  });

  const uploadAll = page.getByRole("button", { name: /upload all/i });
  await expect(uploadAll).toBeDisabled();
  await expect(
    page.getByRole("alert").filter({ hasText: "Privacy settings are unavailable" })
  ).toBeVisible();
  expect(uploadRequests).toBe(0);

  settingsUnavailable = false;
  await page.getByRole("button", { name: "Retry privacy settings" }).click();
  await expect(uploadAll).toBeEnabled();
  await expect(
    page.getByRole("note", { name: "Cloud-assisted OCR privacy" })
  ).toContainText(
    "Scanned PDF and TIFF pages are sent in their original, unredacted form"
  );
  await uploadAll.click();
  await expect.poll(() => uploadRequests).toBe(1);
});

for (const mode of ["custom_local", "prompt_only"] as const) {
  test(`blocks structured and unstructured uploads when the saved mode is ${mode}`, async ({
    page,
  }) => {
    await injectAuth(page);
    const uploadRequests: string[] = [];

    await page.route("**/api/v1/**", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const json = (body: unknown) =>
        route.fulfill({
          contentType: "application/json",
          body: JSON.stringify(body),
        });

      if (url.pathname === "/api/v1/settings/llm") {
        return json({
          providers: [],
          routing: {
            default: "gemini",
            summary: "gemini",
            section: "gemini",
            dedup: "gemini",
            extraction: "gemini",
            vision: "gemini",
            extraction_engine: "hybrid",
            processing_mode: mode,
          },
        });
      }
      if (
        request.method() === "POST" &&
        (url.pathname === "/api/v1/upload" ||
          url.pathname === "/api/v1/upload/unstructured")
      ) {
        uploadRequests.push(url.pathname);
        return json({ upload_id: "must-not-upload", status: "pending" });
      }
      if (url.pathname === "/api/v1/auth/me") {
        return json({
          id: "user-1",
          email: "pedro@example.com",
          display_name: "Pedro",
          is_active: true,
          created_at: "2024-01-01T00:00:00Z",
        });
      }
      return json({});
    });

    await page.goto("/upload");
    await page.locator('input[type="file"]').first().setInputFiles([
      {
        name: "structured.json",
        mimeType: "application/json",
        buffer: Buffer.from('{"resourceType":"Bundle","entry":[]}'),
      },
      {
        name: "unstructured.pdf",
        mimeType: "application/pdf",
        buffer: Buffer.from("%PDF-1.4 unsupported ingestion mode"),
      },
    ]);

    await expect(
      page.getByRole("alert").filter({ hasText: "Uploads do not support" })
    ).toContainText(mode === "custom_local" ? "Custom local" : "Prompt only");
    const uploadAll = page.getByRole("button", { name: /upload all/i });
    await expect(uploadAll).toBeDisabled();
    await uploadAll.evaluate((button) => (button as HTMLButtonElement).click());
    await page.waitForTimeout(100);
    expect(uploadRequests).toEqual([]);
  });
}

test("serializes delayed extraction polls and never applies an older response last", async ({
  page,
}) => {
  await injectAuth(page);
  const probe = { active: 0, maxActive: 0, polls: 0 };

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/settings/llm") {
      return json({
        providers: [],
        routing: {
          default: "gemini",
          summary: "gemini",
          section: "gemini",
          dedup: "gemini",
          extraction: "gemini",
          vision: "gemini",
          extraction_engine: "hybrid",
          processing_mode: "cloud_assisted",
        },
      });
    }
    if (
      url.pathname === "/api/v1/upload/unstructured" &&
      request.method() === "POST"
    ) {
      return json({
        upload_id: "serialized-poll",
        status: "pending_extraction",
        file_type: "pdf",
      });
    }
    if (url.pathname === "/api/v1/upload/extraction-progress") {
      probe.polls += 1;
      const poll = probe.polls;
      probe.active += 1;
      probe.maxActive = Math.max(probe.maxActive, probe.active);
      await new Promise((resolve) => setTimeout(resolve, poll === 1 ? 2_500 : 50));
      probe.active -= 1;
      return json(
        poll === 1
          ? {
              total: 1,
              completed: 0,
              processing: 1,
              failed: 0,
              pending: 0,
              records_created: 0,
            }
          : {
              total: 2,
              completed: 1,
              processing: 1,
              failed: 0,
              pending: 0,
              records_created: 1,
            }
      );
    }
    if (url.pathname === "/api/v1/upload/pending-extraction") {
      return json({
        files: [
          {
            id: "serialized-poll",
            filename: "serialized.pdf",
            ingestion_status: "processing",
          },
        ],
        total: 1,
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/upload");
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "serialized.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 serialized polling fixture"),
  });
  await page.getByRole("button", { name: /upload all/i }).click();

  const bar = page.getByRole("region", { name: "Extraction status" });
  await expect(bar.getByText("1 of 2")).toBeVisible({ timeout: 8_000 });
  expect(probe.maxActive).toBe(1);
  await page.waitForTimeout(700);
  await expect(bar.getByText("1 of 2")).toBeVisible();
});

test("renders strict-local failure provenance in upload history", async ({
  page,
}) => {
  await injectAuth(page);

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/settings/llm") {
      return json({
        providers: [],
        routing: {
          default: "gemini",
          summary: "gemini",
          section: "gemini",
          dedup: "gemini",
          extraction: "gemini",
          vision: "gemini",
          extraction_engine: "hybrid",
          processing_mode: "cloud_assisted",
        },
      });
    }
    if (url.pathname === "/api/v1/upload/history") {
      return json({
        items: [
          {
            id: "history-local-1",
            filename: "failed-local.pdf",
            ingestion_status: "failed",
            record_count: 0,
            file_size_bytes: 2048,
            created_at: "2025-01-01T00:00:00Z",
            ingestion_progress: {},
            ingestion_errors: [],
            notices: [],
            local_run: {
              privacy_mode: "validated_strict_local",
              models: [
                {
                  role: "extraction",
                  repository: "numind/NuExtract3-mlx-4bits",
                  revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
                },
              ],
            },
            local_failure: {
              stage: "extraction",
              code: "local_validation_failed",
              message: "Local extraction could not validate this section.",
              model_role: "extraction",
              repository: "numind/NuExtract3-mlx-4bits",
              revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
              retryable: true,
              checkpoint_preserved: true,
              cloud_fallback_attempted: false,
            },
          },
        ],
        total: 1,
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/upload");
  await page.getByRole("button", { name: "Upload history" }).click();

  await expect(page.getByText("Validated strict local")).toBeVisible();
  await expect(page.getByText("Failure stage: Local extraction")).toBeVisible();
  await expect(page.getByText("NuExtract3")).toBeVisible();
  await expect(page.getByText("Cloud fallback was not attempted")).toBeVisible();
});

test("renders immutable model identities for a completed strict-local upload", async ({
  page,
}) => {
  await injectAuth(page);

  const models = [
    {
      role: "ocr",
      repository: "sahilchachra/ovisocr2-int4-mlx",
      revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
    },
    {
      role: "extraction",
      repository: "numind/NuExtract3-mlx-4bits",
      revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
    },
    {
      role: "summary",
      repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
      revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
    },
  ];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/settings/llm") {
      return json({
        providers: [],
        routing: {
          default: "gemini",
          summary: "gemini",
          section: "gemini",
          dedup: "gemini",
          extraction: "gemini",
          vision: "gemini",
          extraction_engine: "hybrid",
          processing_mode: "cloud_assisted",
        },
      });
    }
    if (url.pathname === "/api/v1/upload/history") {
      return json({
        items: [
          {
            id: "history-local-complete",
            filename: "completed-local.pdf",
            ingestion_status: "completed",
            record_count: 4,
            created_at: "2025-01-01T00:00:00Z",
            ingestion_progress: {},
            ingestion_errors: [],
            notices: [],
            local_run: {
              privacy_mode: "validated_strict_local",
              models,
            },
            local_failure: null,
          },
        ],
        total: 1,
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/upload");
  await page.getByRole("button", { name: "Upload history" }).click();

  const row = page.locator("tr", { hasText: "completed-local.pdf" });
  for (const model of models) {
    await expect(row).toContainText(model.role);
    await expect(row).toContainText(model.repository);
    await expect(row).toContainText(model.revision);
  }
});

test("renders strict-local failure stage and provenance in Admin extractions", async ({
  page,
}) => {
  await injectAuth(page);

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.pathname === "/api/v1/upload/pending-extraction") {
      return json({
        files: [
          {
            id: "admin-local-1",
            filename: "admin-local.pdf",
            mime_type: "application/pdf",
            file_category: "unstructured",
            file_size_bytes: 4096,
            created_at: "2025-01-01T00:00:00Z",
            ingestion_status: "failed",
            progress_stage: "extraction",
            progress_detail: {
              model_role: "extraction",
              repository: "numind/NuExtract3-mlx-4bits",
              revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
            },
            notices: [],
            local_run: {
              privacy_mode: "validated_strict_local",
              models: [
                {
                  role: "extraction",
                  repository: "numind/NuExtract3-mlx-4bits",
                  revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
                },
              ],
            },
            local_failure: {
              stage: "extraction",
              code: "local_worker_failed",
              message: "The local extraction worker stopped.",
              model_role: "extraction",
              repository: "numind/NuExtract3-mlx-4bits",
              revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
              retryable: true,
              checkpoint_preserved: true,
              cloud_fallback_attempted: false,
            },
          },
        ],
        total: 1,
      });
    }
    if (url.pathname === "/api/v1/auth/me") {
      return json({
        id: "user-1",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    return json({});
  });

  await page.goto("/admin?tab=extractions");

  await expect(page.getByText("Validated strict local")).toBeVisible();
  await expect(page.getByText("Failure stage: Local extraction")).toBeVisible();
  await expect(page.getByText("NuExtract3")).toBeVisible();
  await expect(page.getByText("Cloud fallback was not attempted")).toBeVisible();
});
