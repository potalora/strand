import { expect, test, type Page } from "./fixtures/console-gate";

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const GROUNDED_TYPED_RESPONSE = {
  sections: [
    {
      heading: "Observations",
      claims: [
        {
          fact_id: "fact_saved",
          field_paths: ["/name", "/value"],
          evidence_ids: ["evidence_saved"],
        },
      ],
    },
  ],
  uncertainties: [],
};

interface Calls {
  generate: number;
  buildPrompt: number;
  buildPromptBody: Record<string, unknown> | null;
  generateBody: Record<string, unknown> | null;
  routingBodies: Record<string, unknown>[];
  events: string[];
}

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

async function mockBackend(
  page: Page,
  options: {
    settingsFailures?: number;
    routingDelaysMs?: number[];
    routingFailures?: number;
    generateDelayMs?: number;
    historyScenario?:
      | "default"
      | "prompt_only"
      | "strict_json"
      | "strict_both"
      | "cloud_json"
      | "custom_both";
  } = {}
): Promise<Calls> {
  const calls: Calls = {
    generate: 0,
    buildPrompt: 0,
    buildPromptBody: null,
    generateBody: null,
    routingBodies: [],
    events: [],
  };
  const routing = {
    default: "gemini",
    summary: "gemini",
    section: "gemini",
    dedup: "gemini",
    extraction: "gemini",
    vision: "gemini",
    extraction_engine: "hybrid",
    processing_mode: "cloud_assisted",
    future_route_policy: "preserve-me",
  };
  let settingsFailures = options.settingsFailures ?? 0;
  let routingFailures = options.routingFailures ?? 0;
  let routingRequest = 0;

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const json = (body: unknown) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify(body),
      });

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
    if (url.pathname === "/api/v1/dashboard/patients") {
      return json({
        items: [
          {
            id: "patient-1",
            fhir_id: "PT-1",
            gender: "female",
            name: null,
            birth_date: null,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/summary/providers") {
      return json({
        default: "legacy-global",
        providers: [
          {
            name: "legacy-global",
            model: "stale-global-model",
            configured: true,
            supports_vision: true,
          },
          {
            name: "ollama",
            model: "stale-global-local-model",
            configured: true,
            supports_vision: false,
          },
        ],
      });
    }
    if (
      url.pathname === "/api/v1/settings/llm/routing" &&
      request.method() === "PUT"
    ) {
      const body = request.postDataJSON() as Record<string, unknown>;
      calls.routingBodies.push(body);
      const delayMs = options.routingDelaysMs?.[routingRequest] ?? 0;
      routingRequest += 1;
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      if (routingFailures > 0) {
        routingFailures -= 1;
        return route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "routing unavailable" }),
        });
      }
      Object.assign(routing, body);
      calls.events.push(`routing:${String(body.processing_mode)}`);
      return json({ ok: true });
    }
    if (url.pathname === "/api/v1/settings/llm") {
      if (settingsFailures > 0) {
        settingsFailures -= 1;
        return route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "settings unavailable" }),
        });
      }
      return json({
        providers: [
          {
            name: "gemini",
            is_local: false,
            supports_vision: true,
            configured: true,
            has_key: true,
            key_masked: "gem…test",
            base_url: null,
            model: "gemini-user-route",
            enabled: true,
            source: "user",
          },
          {
            name: "anthropic",
            is_local: false,
            supports_vision: true,
            configured: true,
            has_key: true,
            key_masked: "ant…test",
            base_url: null,
            model: "claude-user-route",
            enabled: true,
            source: "user",
          },
          {
            name: "ollama",
            is_local: true,
            supports_vision: false,
            configured: true,
            has_key: false,
            key_masked: null,
            base_url: "http://[::1]:11434/v1",
            model: "qwen-user:9b",
            enabled: true,
            source: "user",
          },
          {
            name: "lmstudio",
            is_local: true,
            supports_vision: false,
            configured: true,
            has_key: false,
            key_masked: null,
            base_url: "http://127.models.example.test/v1",
            model: "remote-lookalike",
            enabled: true,
            source: "user",
          },
        ],
        routing,
      });
    }
    if (url.pathname === "/api/v1/local-ai/status") {
      return json({
        platform: "apple_silicon",
        compatible: true,
        can_manage_pack: false,
        state: "ready",
        active_revision: "apple-m4-16gb-v1",
        available_revision: "apple-m4-16gb-v1",
        operation: null,
        models: [
          {
            role: "summary",
            repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            quantization: "4bit",
            runtime: "mlx-vlm",
            license: "apache-2.0",
            download_bytes: 5_977_073_021,
            expected_memory_bytes: null,
            installed: true,
            validated: true,
          },
        ],
      });
    }
    if (
      url.pathname === "/api/v1/summary/generate" &&
      request.method() === "POST"
    ) {
      calls.generate += 1;
      calls.events.push(
        `generate:${String(
          (request.postDataJSON() as Record<string, unknown>).processing_mode
        )}`
      );
      calls.generateBody = request.postDataJSON() as Record<string, unknown>;
      if (options.generateDelayMs) {
        await new Promise((resolve) =>
          setTimeout(resolve, options.generateDelayMs)
        );
      }
      return json({
        id: "summary-1",
        processing_mode: calls.generateBody.processing_mode,
        model_provenance: null,
        typed_response: GROUNDED_TYPED_RESPONSE,
        natural_language: "Grounded summary.",
        json_data: null,
        record_count: 4,
        duplicate_warning: null,
        de_identification_report: {},
        model_used: "Qwen3.5-9B",
        generated_at: "2025-01-01T00:00:00Z",
      });
    }
    if (
      url.pathname === "/api/v1/summary/build-prompt" &&
      request.method() === "POST"
    ) {
      calls.buildPrompt += 1;
      calls.buildPromptBody = request.postDataJSON() as Record<string, unknown>;
      return json({
        id: "prompt-1",
        summary_type: "full",
        system_prompt: "No medical advice.",
        user_prompt: "De-identified records.",
        target_model: "user-selected",
        suggested_config: {},
        record_count: 4,
        de_identification_report: {},
        copyable_payload: "SYSTEM\\nNo medical advice.\\nUSER\\nDe-identified records.",
        processing_mode: "prompt_only",
        model_provenance: null,
        generated_at: "2025-01-01T00:00:00Z",
      });
    }
    if (url.pathname === "/api/v1/summary/prompts/saved-prompt-only") {
      return json({
        id: "saved-prompt-only",
        summary_type: "full",
        system_prompt: "No medical advice.",
        user_prompt: "De-identified records.",
        target_model: "gemini-3.5-flash",
        suggested_config: {},
        record_count: 4,
        de_identification_report: {},
        copyable_payload: "PROMPT PAYLOAD FROM HISTORY",
        response_text: null,
        response_format: null,
        typed_response: null,
        processing_mode: "prompt_only",
        model_provenance: null,
        generated_at: "2025-01-03T00:00:00Z",
      });
    }
    const savedGeneratedMatch = url.pathname.match(
      /\/summary\/prompts\/saved-(strict|cloud|custom)-(json|both)$/
    );
    if (savedGeneratedMatch) {
      const [, modeName, responseFormat] = savedGeneratedMatch;
      const processingMode =
        modeName === "strict"
          ? "validated_strict_local"
          : modeName === "cloud"
            ? "cloud_assisted"
            : "custom_local";
      return json({
        id: `saved-${modeName}-${responseFormat}`,
        summary_type: "full",
        system_prompt: "No medical advice.",
        user_prompt: "Server-selected evidence-grounded record facts only.",
        target_model:
          processingMode === "validated_strict_local"
            ? "locked-local-summary"
            : "routed-summary",
        suggested_config: {},
        record_count: 1,
        de_identification_report: null,
        copyable_payload: "",
        // Persisted text is only a display projection. The deliberately stale
        // JSON proves typed_response remains authoritative when present.
        response_text:
          responseFormat === "both"
            ? '## Observations\n\n- Name: Potassium.\n\n---JSON---\n{"legacy":true}'
            : processingMode === "validated_strict_local"
              ? "## Observations\n\n- Name: Potassium."
              : '{"legacy":true}',
        response_format: responseFormat,
        typed_response: GROUNDED_TYPED_RESPONSE,
        processing_mode: processingMode,
        model_provenance: null,
        generated_at: "2025-01-03T00:00:00Z",
      });
    }
    if (url.pathname === "/api/v1/summary/prompts/saved-strict") {
      return json({
        id: "saved-strict",
        summary_type: "full",
        system_prompt: "No medical advice.",
        user_prompt: "De-identified records.",
        target_model: "legacy-target-must-not-render",
        suggested_config: {},
        record_count: 3,
        de_identification_report: {},
        copyable_payload: "",
        response_text: "Saved strict-local summary.",
        response_format: "natural_language",
        processing_mode: "validated_strict_local",
        model_provenance: {
          processing_mode: "validated_strict_local",
          manifest_sha256: "a".repeat(64),
          pack_revision: "apple-m4-16gb-v1",
          model: {
            role: "summary",
            repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            quantization: "4bit",
            runtime: { name: "mlx-vlm", version: "0.5.0" },
          },
        },
        generated_at: "2025-01-02T00:00:00Z",
      });
    }
    if (url.pathname === "/api/v1/summary/prompts") {
      if (options.historyScenario === "prompt_only") {
        return json({
          items: [
            {
              id: "saved-prompt-only",
              summary_type: "full",
              system_prompt: "",
              user_prompt: "",
              target_model: "gemini-3.5-flash",
              suggested_config: {},
              record_count: 4,
              de_identification_report: {},
              copyable_payload: "PROMPT PAYLOAD FROM HISTORY",
              processing_mode: "prompt_only",
              model_provenance: null,
              generated_at: "2025-01-03T00:00:00Z",
            },
          ],
        });
      }
      if (
        options.historyScenario === "strict_json" ||
        options.historyScenario === "strict_both" ||
        options.historyScenario === "cloud_json" ||
        options.historyScenario === "custom_both"
      ) {
        const [modeName, responseFormat] = options.historyScenario.split("_");
        const processingMode =
          modeName === "strict"
            ? "validated_strict_local"
            : modeName === "cloud"
              ? "cloud_assisted"
              : "custom_local";
        return json({
          items: [
            {
              id: `saved-${modeName}-${responseFormat}`,
              summary_type: "full",
              system_prompt: "",
              user_prompt: "",
              target_model:
                processingMode === "validated_strict_local"
                  ? "locked-local-summary"
                  : "routed-summary",
              suggested_config: {},
              record_count: 1,
              de_identification_report: null,
              copyable_payload: "",
              processing_mode: processingMode,
              model_provenance: null,
              generated_at: "2025-01-03T00:00:00Z",
            },
          ],
        });
      }
      return json({
        items: [
          {
            id: "saved-strict",
            summary_type: "full",
            system_prompt: "",
            user_prompt: "",
            target_model: "legacy-target-must-not-render",
            suggested_config: {},
            record_count: 3,
            de_identification_report: {},
            copyable_payload: "",
            processing_mode: "validated_strict_local",
            model_provenance: {
              processing_mode: "validated_strict_local",
              manifest_sha256: "a".repeat(64),
              pack_revision: "apple-m4-16gb-v1",
              model: {
                role: "summary",
                repository: "mlx-community/Qwen3.5-9B-MLX-4bit",
                revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
                quantization: "4bit",
                runtime: { name: "mlx-vlm", version: "0.5.0" },
              },
            },
            generated_at: "2025-01-02T00:00:00Z",
          },
          {
            id: "saved-unknown",
            summary_type: "category",
            system_prompt: "",
            user_prompt: "",
            target_model: "must-not-be-used-as-provenance",
            suggested_config: {},
            record_count: 2,
            de_identification_report: {},
            copyable_payload: "",
            generated_at: "2024-12-31T00:00:00Z",
          },
        ],
      });
    }
    return json({});
  });

  return calls;
}

test.beforeEach(async ({ page }) => {
  await injectAuth(page);
});

test("validated local summary hides providers and sends the local mode", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/summaries");

  await page
    .getByLabel("AI execution mode")
    .selectOption("validated_strict_local");
  await expect(page.getByLabel("Provider")).toHaveCount(0);
  await expect(
    page.getByText("mlx-community/Qwen3.5-9B-MLX-4bit")
  ).toBeVisible();
  await page.getByRole("button", { name: "Generate summary" }).click();

  await expect.poll(() => calls.generate).toBe(1);
  expect(calls.generateBody).toMatchObject({
    processing_mode: "validated_strict_local",
  });
  expect(calls.generateBody).not.toHaveProperty("provider");
});

test("prompt-only builds a copyable payload and never calls generate", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/summaries");

  await page.getByLabel("AI execution mode").selectOption("prompt_only");
  await expect(page.getByText("Output format", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Build prompt" }).click();

  await expect(
    page.getByRole("button", { name: "Copy prompt payload" })
  ).toBeVisible();
  expect(calls.generate).toBe(0);
  expect(calls.buildPrompt).toBe(1);
  expect(calls.buildPromptBody).toMatchObject({
    output_format: "both",
  });
  await expect(
    page.getByText("No health data was sent by MedTimeline").first()
  ).toBeVisible();
});

test("saved prompt-only history reopens the copyable payload without requiring a response", async ({
  page,
}) => {
  await mockBackend(page, { historyScenario: "prompt_only" });
  await page.goto("/summaries");

  await page.getByRole("button", { name: "Show (1)" }).click();
  await page
    .locator("button.lrow")
    .filter({ hasText: "Prompt only" })
    .click();

  await expect(
    page.getByRole("button", { name: "Copy prompt payload" })
  ).toBeVisible();
  await expect(page.getByText("PROMPT PAYLOAD FROM HISTORY")).toBeVisible();
  await expect(
    page.getByText("This saved summary has no stored response text.")
  ).toHaveCount(0);
});

for (const savedCase of [
  {
    modeName: "strict-local",
    modeLabel: "Validated strict local",
    responseFormat: "json",
    historyScenario: "strict_json",
  },
  {
    modeName: "strict-local",
    modeLabel: "Validated strict local",
    responseFormat: "both",
    historyScenario: "strict_both",
  },
  {
    modeName: "cloud-assisted",
    modeLabel: "Cloud assisted",
    responseFormat: "json",
    historyScenario: "cloud_json",
  },
  {
    modeName: "custom-local",
    modeLabel: "Custom local",
    responseFormat: "both",
    historyScenario: "custom_both",
  },
] as const) {
  test(`saved ${savedCase.modeName} ${savedCase.responseFormat} history uses typed_response for JSON display`, async ({
    page,
  }) => {
    await mockBackend(page, {
      historyScenario: savedCase.historyScenario,
    });
    await page.goto("/summaries");

    await page.getByRole("button", { name: "Show (1)" }).click();
    await page
      .locator("button.lrow")
      .filter({ hasText: savedCase.modeLabel })
      .click();

    const resultCard = page.locator(".card-surface").filter({
      has: page.getByRole("heading", { name: "Summary", exact: true }),
    });
    await expect(
      resultCard.getByRole("button", { name: "JSON data" })
    ).toBeVisible();
    await resultCard.getByRole("button", { name: "JSON data" }).click();
    await expect(resultCard.getByText(/fact_saved/)).toBeVisible();
    await expect(resultCard.getByText(/"legacy": true/)).toHaveCount(0);
    await expect(
      resultCard.getByRole("button", { name: "Narrative" })
    ).toHaveCount(savedCase.responseFormat === "both" ? 1 : 0);
    if (savedCase.modeName === "cloud-assisted") {
      await expect(
        page.getByText(
          /best-effort de-identification but cannot guarantee every identifier was removed.*saved cloud provider/
        )
      ).toBeVisible();
    }
  });
}

test("uses per-user providers and persists a complete loopback route before custom generation", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/summaries");

  await expect(page.getByLabel("Provider")).toHaveValue("gemini");
  await expect(
    page.getByLabel("Provider").locator('option[value="gemini"]')
  ).toHaveText(/gemini-user-route/);
  await expect(page.getByText("stale-global-model")).toHaveCount(0);

  await page.getByLabel("AI execution mode").selectOption("custom_local");
  await expect(page.getByLabel("Provider")).toHaveValue("ollama");
  await expect(
    page.getByLabel("Provider").locator('option[value="ollama"]')
  ).toHaveText(/qwen-user:9b/);
  await expect(
    page.getByLabel("Provider").locator('option[value="lmstudio"]')
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Generate summary" }).click();
  await expect.poll(() => calls.generate).toBe(1);

  expect(calls.events.at(-2)).toBe("routing:custom_local");
  expect(calls.events.at(-1)).toBe("generate:custom_local");
  expect(calls.routingBodies.at(-1)).toEqual({
    default: "ollama",
    summary: "ollama",
    section: "ollama",
    dedup: "ollama",
    extraction: "ollama",
    vision: "ollama",
    extraction_engine: "hybrid",
    processing_mode: "custom_local",
  });
  expect(calls.routingBodies.at(-1)).not.toHaveProperty("future_route_policy");
  expect(calls.generateBody).toMatchObject({
    processing_mode: "custom_local",
  });
  expect(calls.generateBody).not.toHaveProperty("provider");
  expect(calls.generateBody).not.toHaveProperty("model");
});

test("cloud transitions and provider changes persist complete cloud-only routing", async ({
  page,
}) => {
  const calls = await mockBackend(page);
  await page.goto("/summaries");

  await page.getByLabel("AI execution mode").selectOption("custom_local");
  await expect.poll(() => calls.routingBodies.length).toBe(1);

  await page.getByLabel("AI execution mode").selectOption("cloud_assisted");
  await expect.poll(() => calls.routingBodies.length).toBe(2);
  expect(calls.routingBodies.at(-1)).toEqual({
    default: "gemini",
    summary: "gemini",
    section: "gemini",
    dedup: "gemini",
    extraction: "gemini",
    vision: "gemini",
    extraction_engine: "hybrid",
    processing_mode: "cloud_assisted",
  });

  await page.getByLabel("Provider").selectOption("anthropic");
  await expect.poll(() => calls.routingBodies.length).toBe(3);
  expect(calls.routingBodies.at(-1)).toEqual({
    default: "anthropic",
    summary: "anthropic",
    section: "anthropic",
    dedup: "anthropic",
    extraction: "anthropic",
    vision: "anthropic",
    extraction_engine: "hybrid",
    processing_mode: "cloud_assisted",
  });
});

test("serializes route commits and waits for the latest snapshot before generation", async ({
  page,
}) => {
  const calls = await mockBackend(page, { routingDelaysMs: [700, 0] });
  await page.goto("/summaries");

  await page.getByLabel("Provider").selectOption("anthropic");
  await page.getByLabel("AI execution mode").selectOption("custom_local");
  await page.getByRole("button", { name: "Generate summary" }).click();

  await expect.poll(() => calls.events.length).toBe(3);
  expect(calls.events).toEqual([
    "routing:cloud_assisted",
    "routing:custom_local",
    "generate:custom_local",
  ]);
  expect(calls.generateBody).toMatchObject({
    processing_mode: "custom_local",
  });
});

test("locks execution routing for the full duration of summary generation", async ({
  page,
}) => {
  const calls = await mockBackend(page, { generateDelayMs: 1_000 });
  await page.goto("/summaries");

  const mode = page.getByLabel("AI execution mode");
  await mode.selectOption("custom_local");
  await expect.poll(() => calls.routingBodies.length).toBe(1);

  await page.getByRole("button", { name: "Generate summary" }).click();
  await expect.poll(() => calls.generate).toBe(1);

  const provider = page.getByLabel("Provider");
  await expect(mode).toBeDisabled();
  await expect(provider).toBeDisabled();

  const routingCount = calls.routingBodies.length;
  await mode.evaluate((select) => {
    const element = select as HTMLSelectElement;
    element.value = "cloud_assisted";
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForTimeout(100);
  expect(calls.routingBodies).toHaveLength(routingCount);

  await expect(page.getByText("Grounded summary.")).toBeVisible();
  await expect(mode).toBeEnabled();
  await expect(provider).toBeEnabled();
  await expect(mode).toHaveValue("custom_local");
});

test("a routing save failure clears optimistic settings and blocks generation until retry", async ({
  page,
}) => {
  const calls = await mockBackend(page, { routingFailures: 1 });
  await page.goto("/summaries");

  await page.getByLabel("AI execution mode").selectOption("custom_local");

  await expect(
    page.getByRole("alert").filter({
      hasText: "AI privacy settings are unavailable",
    })
  ).toBeVisible();
  await expect(page.getByLabel("AI execution mode")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Generate summary" })).toBeDisabled();
  expect(calls.generate).toBe(0);

  await page.getByRole("button", { name: "Retry privacy settings" }).click();
  await expect(page.getByLabel("AI execution mode")).toHaveValue("cloud_assisted");
  await expect(page.getByRole("button", { name: "Generate summary" })).toBeEnabled();

  await page.getByRole("button", { name: "Generate summary" }).click();
  await expect.poll(() => calls.generate).toBe(1);
  expect(calls.generateBody).toMatchObject({
    processing_mode: "cloud_assisted",
  });
});

test("settings failure leaves summary execution unresolved until an explicit retry", async ({
  page,
}) => {
  const calls = await mockBackend(page, { settingsFailures: 1 });
  await page.goto("/summaries");

  await expect(
    page.getByRole("alert").filter({ hasText: "AI privacy settings are unavailable" })
  ).toBeVisible();
  await expect(page.getByLabel("AI execution mode")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Generate summary" })).toBeDisabled();
  expect(calls.generate).toBe(0);

  await page.getByRole("button", { name: "Retry privacy settings" }).click();
  await expect(page.getByLabel("AI execution mode")).toHaveValue("cloud_assisted");
  await expect(
    page.getByRole("alert").filter({ hasText: "AI privacy settings are unavailable" })
  ).toHaveCount(0);
});

test("saved summaries render their own mode and provenance, never the current selector", async ({
  page,
}) => {
  await mockBackend(page);
  await page.goto("/summaries");

  await expect(page.getByLabel("AI execution mode")).toHaveValue(
    "cloud_assisted"
  );
  await page.getByRole("button", { name: "Show (2)" }).click();

  const strictHistory = page
    .locator("button.lrow")
    .filter({ hasText: "Validated strict local" });
  await expect(strictHistory).toBeVisible();
  await expect(strictHistory).toContainText(
    "mlx-community/Qwen3.5-9B-MLX-4bit"
  );
  const unknownHistory = page
    .locator("button.lrow")
    .filter({ hasText: "Execution mode unknown" });
  await expect(unknownHistory).toBeVisible();
  await expect(unknownHistory).toContainText("Model identity unknown");
  await expect(page.getByText("legacy-target-must-not-render")).toHaveCount(0);
  await expect(page.getByText("must-not-be-used-as-provenance")).toHaveCount(0);

  await page
    .getByRole("button", { name: /full summary/i })
    .click();
  await expect(page.getByText("Saved strict-local summary.")).toBeVisible();
  await expect(
    page.getByText(
      "Validated facts are summarized on this machine with no cloud fallback."
    )
  ).toBeVisible();
  await expect(
    page.getByText(/De-identified record content is sent to the selected cloud provider/)
  ).toHaveCount(0);
});
