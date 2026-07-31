import { test, expect, type Page } from "./fixtures/console-gate";

/**
 * Multi-LLM provider selector on the Summarize page (fully mocked — no real
 * backend, parallel-safe).
 *
 * The Summarize page (`/summaries`) now renders an "AI provider" <select>
 * populated from the current user's GET /settings/llm response. Cloud providers
 * (gemini, anthropic)
 * show a "records are sent to <provider>" note; local providers (ollama,
 * lmstudio) show a "Runs locally with <provider>" note. The chosen provider is
 * carried on the POST /summary/generate request body as `provider`.
 *
 * Auth is injected straight into localStorage (the persisted zustand shape) so
 * the dashboard layout authenticates without a UI login, and every API call is
 * stubbed with page.route for determinism.
 */

const AUTH_STATE = {
  state: {
    accessToken: "test.access.token",
    refreshToken: "test.refresh.token",
    isAuthenticated: true,
  },
  version: 0,
};

const ME_OK = {
  id: "11111111-2222-3333-4444-555555555555",
  email: "pedro@example.com",
  display_name: "Pedro",
  is_active: true,
  created_at: "2024-01-01T00:00:00Z",
};

const PATIENTS = {
  items: [
    {
      id: "p1",
      fhir_id: "PT-1",
      gender: "female",
      name: null,
      birth_date: null,
    },
  ],
};

const LLM_SETTINGS = {
  providers: [
    {
      name: "gemini",
      model: "gemini-3.5-flash",
      supports_vision: true,
      configured: true,
      is_local: false,
      enabled: true,
      has_key: true,
      key_masked: "gem…test",
      base_url: null,
      source: "user",
    },
    {
      name: "anthropic",
      model: "claude-haiku-4-5-20251001",
      supports_vision: true,
      configured: true,
      is_local: false,
      enabled: true,
      has_key: true,
      key_masked: "ant…test",
      base_url: null,
      source: "user",
    },
    {
      name: "ollama",
      model: "llama3.2:1b",
      supports_vision: false,
      configured: true,
      is_local: true,
      enabled: true,
      has_key: false,
      key_masked: null,
      base_url: "http://127.0.0.1:11434/v1",
      source: "user",
    },
  ],
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
};

const MODEL_BY_PROVIDER: Record<string, string> = Object.fromEntries(
  LLM_SETTINGS.providers.map((p) => [p.name, p.model])
);

type Captured = { generateBody: Record<string, unknown> | null };

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

/**
 * Stub every backend call the Summarize page + nav make. Returns a `Captured`
 * handle whose `generateBody` is filled from the POST /summary/generate request
 * body so the test can assert the chosen provider was sent.
 */
async function mockBackend(page: Page): Promise<Captured> {
  const captured: Captured = { generateBody: null };

  await page.route("**/api/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (url.includes("/auth/me")) return json(ME_OK);
    if (url.includes("/auth/refresh"))
      return json({
        access_token: "fresh.access.token",
        refresh_token: "fresh.refresh.token",
      });
    if (url.includes("/auth/logout")) return json({});
    if (url.includes("/dashboard/patients")) return json(PATIENTS);
    if (url.includes("/settings/llm/routing") && req.method() === "PUT") {
      return json({ ok: true });
    }
    if (url.includes("/settings/llm")) return json(LLM_SETTINGS);
    if (url.includes("/summary/generate") && req.method() === "POST") {
      const body = req.postDataJSON() as Record<string, unknown>;
      captured.generateBody = body;
      const model = MODEL_BY_PROVIDER[String(body.provider)] ?? "gemini-3.5-flash";
      return json({
        id: "sum-1",
        processing_mode: "cloud_assisted",
        model_provenance: {
          processing_mode: "cloud_assisted",
          provider: String(body.provider),
          model,
        },
        typed_response: { sections: [], uncertainties: [] },
        natural_language: "## Summary\nDe-identified overview.",
        json_data: null,
        record_count: 7,
        duplicate_warning: null,
        de_identification_report: {},
        model_used: model,
        generated_at: "2024-01-01T00:00:00Z",
      });
    }
    if (url.includes("/summary/prompts")) return json({ items: [] });
    return json({});
  });

  return captured;
}

/**
 * The provider <select> is NOT the first select on the page (the patient
 * "Record subject" select is). Scope to the card whose label is exactly
 * "AI provider" — the disclaimer's "the selected AI provider" copy is a
 * different (non-exact) string, so it never matches.
 */
function providerSelect(page: Page) {
  return page.getByLabel("Provider");
}

test.describe("Summarize — AI provider selector", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
  });

  test("(1) provider select renders the configured providers", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/summaries");

    const select = providerSelect(page);
    await expect(select).toBeVisible({ timeout: 10_000 });

    // Options are labeled `${name} · ${model}` and populate after the async fetch.
    await expect(select).toContainText("gemini · gemini-3.5-flash", { timeout: 10_000 });
    await expect(select).toContainText("anthropic · claude-haiku-4-5-20251001");
    await expect(select).not.toContainText("ollama · llama3.2:1b");
    await page.getByLabel("AI execution mode").selectOption("custom_local");
    await expect(page.getByLabel("Provider")).toContainText(
      "ollama · llama3.2:1b"
    );
  });

  test("(2) cloud vs local provider shows the right privacy note", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/summaries");

    const select = providerSelect(page);
    await expect(select).toContainText("anthropic · claude", { timeout: 10_000 });

    // Cloud provider → "sent to <provider>".
    await select.selectOption("anthropic");
    await expect(
      page.getByText(
        "Automated scrubbing provides best-effort de-identification but cannot guarantee every identifier is removed. The resulting record content is sent to anthropic."
      )
    ).toBeVisible();

    // Custom local is an explicit execution mode and stays visibly unverified.
    await page.getByLabel("AI execution mode").selectOption("custom_local");
    await page.getByLabel("Provider").selectOption("ollama");
    await expect(
      page.getByText(/selected loopback server.*not the model or server/i)
    ).toBeVisible();
  });

  test("(3) generate carries the selected provider and result reflects its model", async ({
    page,
  }) => {
    const captured = await mockBackend(page);
    await page.goto("/summaries");

    const select = providerSelect(page);
    await expect(select).toContainText("anthropic · claude", { timeout: 10_000 });
    await select.selectOption("anthropic");

    await page.getByRole("button", { name: "Generate summary" }).click();

    // The result uses saved execution provenance rather than the current selector.
    await expect(
      page.getByRole("heading", { name: "Summary", exact: true })
    ).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByText(
        "7 records · Cloud assisted · anthropic · claude-haiku-4-5-20251001"
      )
    ).toBeVisible();
    await expect(
      page.getByText(
        /best-effort de-identification but cannot guarantee every identifier was removed/
      )
    ).toBeVisible();

    // (3) The intercepted POST body carried the selected provider.
    expect(captured.generateBody).not.toBeNull();
    expect(captured.generateBody?.provider).toBe("anthropic");
    expect(captured.generateBody?.processing_mode).toBe("cloud_assisted");
    expect(captured.generateBody?.patient_id).toBe("p1");
  });
});
