import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const MODELS = [
  {
    role: "ocr",
    repository: "sahilchachra/ovisocr2-int4-mlx",
    revision: "1e9cea98871c19b2349a5d2df36fb6c4c38a1237",
    quantization: "int4",
    runtime: "mlx-vlm 0.5.0",
    license: "apache-2.0",
    download_bytes: 652031947,
    expected_memory_bytes: null,
    installed: false,
    validated: false,
  },
  {
    role: "extraction",
    repository: "numind/NuExtract3-mlx-4bits",
    revision: "29c38269f94054282bf9ea97a20dfc6bb8bbefea",
    quantization: "4bit",
    runtime: "mlx-vlm 0.5.0",
    license: "apache-2.0",
    download_bytes: 3054403529,
    expected_memory_bytes: null,
    installed: false,
    validated: false,
  },
  {
    role: "summary",
    repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
    revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
    quantization: "4bit",
    runtime: "mlx-vlm 0.5.0",
    license: "apache-2.0",
    download_bytes: 5977073021,
    expected_memory_bytes: null,
    installed: false,
    validated: false,
  },
];

const LLM_SETTINGS = {
  providers: [
    {
      name: "ollama",
      is_local: true,
      supports_vision: false,
      configured: true,
      has_key: false,
      key_masked: null,
      base_url: "http://localhost:11434/v1",
      model: "qwen3.5:9b",
      enabled: true,
      source: "default",
    },
    {
      name: "lmstudio",
      is_local: true,
      supports_vision: false,
      configured: true,
      has_key: false,
      key_masked: null,
      base_url: "http://localhost:1234/v1",
      model: null,
      enabled: true,
      source: "default",
    },
  ],
  routing: {
    default: "ollama",
    summary: "ollama",
    extraction: "ollama",
    vision: "ollama",
    dedup: "ollama",
    section: "ollama",
    extraction_engine: "local",
    processing_mode: "custom_local",
  },
};

interface PollProbe {
  active: number;
  maxActive: number;
  polls: number;
}

interface StatusOperation {
  id: string;
  action: "install";
  state: "queued" | "running" | "completed" | "failed" | "paused";
  current_role: string | null;
  bytes_done: number;
  bytes_total: number;
  message: string;
  retryable: boolean;
}

async function setup(
  page: Page,
  initialState: "not_installed" | "preview" | "ready" | "failed",
  pollProbe?: PollProbe,
  operationPollFailures = 0,
  modelStates?: Partial<Record<string, { installed: boolean; validated: boolean }>>,
  removalProbe?: { calls: number },
  modelMemory?: Partial<Record<string, number | null>>,
  previewReason: "feature_disabled" | "release_evidence_missing" =
    "feature_disabled",
  statusOperation?: StatusOperation,
  terminalOutcome: "completed" | "failed" = "completed"
) {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);

  let state = initialState;
  let operationPoll = 0;
  let statusOperationSnapshot = statusOperation ?? null;
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = request.url();
    const method = request.method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.endsWith("/local-ai/status")) {
      return json({
        platform: "apple_silicon",
        compatible: true,
        enabled:
          state === "ready" ||
          (state === "preview" && previewReason !== "feature_disabled"),
        state,
        status_reason: state === "preview" ? previewReason : null,
        active_revision:
          state === "ready" || state === "preview" ? "apple-m4-16gb-v1" : null,
        available_revision: "apple-m4-16gb-v1",
        models: MODELS.map((model) => ({
          ...model,
          installed:
            modelStates?.[model.role]?.installed ?? state === "ready",
          validated:
            modelStates?.[model.role]?.validated ?? state === "ready",
          expected_memory_bytes:
            modelMemory?.[model.role] ?? model.expected_memory_bytes,
        })),
        operation: statusOperationSnapshot,
      });
    }
    if (url.endsWith("/local-ai") && method === "DELETE") {
      if (removalProbe) removalProbe.calls += 1;
      state = "not_installed";
      return json({ removed: true });
    }
    if (url.endsWith("/local-ai/install") && method === "POST") {
      return json({ operation_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", state: "queued" }, 202);
    }
    if (url.includes("/local-ai/operations/")) {
      if (operationPollFailures > 0) {
        operationPollFailures -= 1;
        return json({ detail: "temporary status failure" }, 503);
      }
      if (pollProbe) {
        pollProbe.polls += 1;
        pollProbe.active += 1;
        pollProbe.maxActive = Math.max(
          pollProbe.maxActive,
          pollProbe.active
        );
        await new Promise((resolve) =>
          setTimeout(resolve, pollProbe.polls === 1 ? 1_200 : 50)
        );
        pollProbe.active -= 1;
      }
      operationPoll += 1;
      if (operationPoll === 1) {
        return json({
          id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
          action: "install",
          state: "running",
          current_role: "ocr",
          bytes_done: 200,
          bytes_total: 1000,
          message: "Downloading verified model files.",
          retryable: false,
        });
      }
      if (operationPoll === 2) {
        return json({
          id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
          action: "install",
          state: "running",
          current_role: null,
          bytes_done: 1000,
          bytes_total: 1000,
          message: "Running local validation fixtures.",
          retryable: false,
        });
      }
      const terminalOperation: StatusOperation = {
        id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        action: "install",
        state: terminalOutcome,
        current_role: null,
        bytes_done: 1000,
        bytes_total: 1000,
        message:
          terminalOutcome === "completed"
            ? "Model pack is ready."
            : "Old install failed.",
        retryable: terminalOutcome === "failed",
      };
      state = "ready";
      statusOperationSnapshot = terminalOperation;
      return json(terminalOperation);
    }
    if (url.includes("/settings/llm/routing") && method === "PUT") {
      Object.assign(LLM_SETTINGS.routing, request.postDataJSON() ?? {});
      return json({ ok: true });
    }
    if (url.includes("/settings/llm")) return json(LLM_SETTINGS);
    if (url.includes("/auth/me")) {
      return json({
        id: "11111111-2222-3333-4444-555555555555",
        email: "pedro@example.com",
        display_name: "Pedro",
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
      });
    }
    if (url.includes("/dashboard/overview")) {
      return json({
        total_records: 0,
        total_uploads: 0,
        records_by_type: {},
        date_range_start: null,
        date_range_end: null,
      });
    }
    if (url.includes("/audit-log")) return json({ items: [], total: 0 });
    if (url.includes("/records")) {
      return json({ items: [], total: 0, page: 1, page_size: 100 });
    }
    return json({});
  });
}

test("installs and verifies the platform pack without adding an Admin tab", async ({
  page,
}) => {
  await setup(page, "not_installed");
  await page.goto("/admin?tab=sys");

  await expect(page.getByRole("tab")).toHaveCount(4);
  await expect(
    page.getByRole("heading", { name: "Validated local pack" })
  ).toBeVisible();
  await expect(page.getByText("16 GB unified memory minimum")).toBeVisible();
  await expect(page.getByText("numind/NuExtract3-mlx-4bits")).toBeVisible();
  await expect(
    page.getByText("mlx-community/Qwen3.5-9B-MLX-4bit")
  ).toBeVisible();

  const mode = page.getByLabel("Processing mode");
  await expect(mode.locator('option[value="validated_strict_local"]')).toBeDisabled();

  await page.getByRole("button", { name: "Install local pack" }).click();
  await expect(page.getByText("Verifying downloaded models")).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("Validated and ready")).toBeVisible({
    timeout: 10_000,
  });
  await expect(
    mode.locator('option[value="validated_strict_local"]')
  ).toBeEnabled();
});

test("ready pack enables strict local and custom servers stay explicitly unverified", async ({
  page,
}) => {
  await setup(page, "ready");
  await page.goto("/admin?tab=sys");

  const mode = page.getByLabel("Processing mode");
  await expect(mode.locator('option[value="validated_strict_local"]')).toBeEnabled();
  await expect(
    page.locator("span.tag").filter({ hasText: "Custom local (unverified)" })
  ).toHaveCount(2);
  await expect(page.getByText(/loopback only/i).first()).toBeVisible();
});

test("newest ready status clears an older failed operation", async ({ page }) => {
  await setup(
    page,
    "not_installed",
    undefined,
    0,
    undefined,
    undefined,
    undefined,
    "feature_disabled",
    {
      id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      action: "install",
      state: "running",
      current_role: "ocr",
      bytes_done: 100,
      bytes_total: 1000,
      message: "Downloading verified model files.",
      retryable: false,
    },
    "failed"
  );
  await page.goto("/admin?tab=sys");

  await expect(page.getByText("Validated and ready")).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("Old install failed.")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Retry operation" })
  ).toHaveCount(0);
});

test("disabled strict-local processing gives the enable-and-restart instruction", async ({
  page,
}) => {
  await setup(page, "preview");
  await page.goto("/admin?tab=sys");

  await expect(page.getByText("Runtime-verified preview")).toBeVisible();
  await expect(
    page.getByText(
      "The pack is installed and verified, but strict-local processing is disabled. Set LOCAL_AI_ENABLED=true and restart Strand."
    )
  ).toBeVisible();
  await expect(page.getByText(/· Runtime verified$/)).toHaveCount(3);
  await expect(page.getByText("Not runtime verified", { exact: true })).toHaveCount(0);
  await expect(page.getByText("memory gate pending").first()).toBeVisible();
  await expect(
    page.getByLabel("Processing mode").locator('option[value="validated_strict_local"]')
  ).toBeDisabled();
  await expect(page.getByRole("button", { name: "Verify again" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Remove pack" })).toBeEnabled();
});

test("missing release evidence stays distinct from the disabled feature", async ({
  page,
}) => {
  await setup(
    page,
    "preview",
    undefined,
    0,
    undefined,
    undefined,
    undefined,
    "release_evidence_missing"
  );
  await page.goto("/admin?tab=sys");

  await expect(
    page.getByText(
      "This pack cannot process health records because its release evidence is missing or invalid."
    )
  ).toBeVisible();
  await expect(page.getByText(/LOCAL_AI_ENABLED=true/)).toHaveCount(0);
});

test("ready pack shows the measured resident memory from release evidence", async ({
  page,
}) => {
  await setup(
    page,
    "ready",
    undefined,
    0,
    undefined,
    undefined,
    { summary: 8 * 1024 ** 3 }
  );
  await page.goto("/admin?tab=sys");

  const summary = page
    .getByLabel("Validated local models")
    .locator(".field")
    .filter({ hasText: "Summary" });
  await expect(summary).toContainText("8.0 GiB measured");
});

test("serializes operation polling and stops after the terminal response", async ({
  page,
}) => {
  const probe: PollProbe = { active: 0, maxActive: 0, polls: 0 };
  await setup(page, "not_installed", probe);
  await page.goto("/admin?tab=sys");

  await page.getByRole("button", { name: "Install local pack" }).click();
  await expect(page.getByText("Validated and ready")).toBeVisible({
    timeout: 12_000,
  });

  expect(probe.maxActive).toBe(1);
  const terminalPollCount = probe.polls;
  await page.waitForTimeout(1_700);
  expect(probe.polls).toBe(terminalPollCount);
});

test("a failed operation poll becomes non-busy and can retry status safely", async ({
  page,
}) => {
  await setup(page, "not_installed", undefined, 1);
  await page.goto("/admin?tab=sys");

  await page.getByRole("button", { name: "Install local pack" }).click();
  await expect(
    page.getByRole("alert").filter({
      hasText: "Local model pack operation could not be refreshed",
    })
  ).toBeVisible();

  const retry = page.getByRole("button", { name: "Retry status check" });
  await expect(retry).toBeEnabled();
  await retry.click();
  await expect(page.getByText("Validated and ready")).toBeVisible({
    timeout: 10_000,
  });
});

test("shows installed and validated state per model and permits failed-pack cleanup", async ({
  page,
}) => {
  const removalProbe = { calls: 0 };
  await setup(
    page,
    "failed",
    undefined,
    0,
    {
      ocr: { installed: true, validated: true },
      extraction: { installed: true, validated: false },
      summary: { installed: false, validated: false },
    },
    removalProbe
  );
  await page.goto("/admin?tab=sys");

  const models = page.getByLabel("Validated local models");
  const ocr = models.locator(".field").filter({ hasText: "OCR" });
  await expect(ocr).toContainText("Installed");
  await expect(ocr).toContainText("Validated");

  const extraction = models
    .locator(".field")
    .filter({ hasText: "Grounded extraction" });
  await expect(extraction).toContainText("Installed");
  await expect(extraction).toContainText("Not validated");

  const summary = models.locator(".field").filter({ hasText: "Summary" });
  await expect(summary).toContainText("Not installed");
  await expect(summary).toContainText("Not validated");

  const remove = page.getByRole("button", { name: "Remove pack" });
  await expect(remove).toBeEnabled();
  await remove.click();
  await expect.poll(() => removalProbe.calls).toBe(1);
});
