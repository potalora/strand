import { test, expect, type Page } from "./fixtures/console-gate";

/**
 * AI providers card (Admin → System), fully mocked — no real backend.
 *
 * Mirrors the admin-consolidation pattern: auth is injected straight into
 * localStorage (the persisted zustand shape) so the dashboard authenticates
 * without hitting the login rate limiter, and every `/api/v1/**` call is stubbed
 * with page.route for determinism. The `/settings/llm` GET reflects a tiny piece
 * of mutable state so a saved key shows masked on the component's reload.
 *
 * Asserts: the card renders provider rows + a routing select; saving a key
 * issues `PUT /settings/llm/providers/openai` carrying the key in the body
 * (captured via the request's postDataJSON); changing the default select issues
 * `PUT /settings/llm/routing`; Test surfaces a result.
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
  login_identifier: "pedro@example.com",
  email: "pedro@example.com",
  display_name: "Pedro",
  is_active: true,
  created_at: "2024-01-01T00:00:00Z",
};

const OVERVIEW = {
  total_records: 12,
  total_uploads: 3,
  records_by_type: { condition: 5, observation: 7 },
  date_range_start: "2020-01-01T00:00:00Z",
  date_range_end: "2024-01-01T00:00:00Z",
};

interface MockProvider {
  name: string;
  is_local: boolean;
  supports_vision: boolean;
  configured: boolean;
  has_key: boolean;
  key_masked: string | null;
  base_url: string | null;
  model: string | null;
  enabled: boolean;
  source: string;
}

interface MockState {
  providers: MockProvider[];
  routing: Record<string, string>;
  routingCommits: string[];
}

function freshState(): MockState {
  return {
    providers: [
      {
        name: "gemini",
        is_local: false,
        supports_vision: true,
        configured: true,
        has_key: false,
        key_masked: null,
        base_url: null,
        model: "gemini-3.5-flash",
        enabled: true,
        source: "env",
      },
      {
        name: "openai",
        is_local: false,
        supports_vision: true,
        configured: false,
        has_key: false,
        key_masked: null,
        base_url: null,
        model: null,
        enabled: true,
        source: "default",
      },
      {
        name: "anthropic",
        is_local: false,
        supports_vision: true,
        configured: false,
        has_key: false,
        key_masked: null,
        base_url: null,
        model: null,
        enabled: true,
        source: "default",
      },
      {
        name: "openrouter",
        is_local: false,
        supports_vision: true,
        configured: false,
        has_key: false,
        key_masked: null,
        base_url: "https://openrouter.ai/api/v1",
        model: null,
        enabled: true,
        source: "default",
      },
      {
        name: "ollama",
        is_local: true,
        supports_vision: false,
        configured: true,
        has_key: false,
        key_masked: null,
        base_url: "http://localhost:11434/v1",
        model: "llama3.1",
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
      default: "gemini",
      summary: "gemini",
      section: "gemini",
      dedup: "gemini",
      extraction: "gemini",
      vision: "gemini",
      extraction_engine: "hybrid",
      processing_mode: "cloud_assisted",
    },
    routingCommits: [],
  };
}

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

async function mockBackend(
  page: Page,
  options: {
    routingDelaysMs?: number[];
    routingFailures?: number;
    settingsFailureCalls?: number[];
  } = {}
): Promise<MockState> {
  const state = freshState();
  let routingRequest = 0;
  let routingFailures = options.routingFailures ?? 0;
  let settingsGet = 0;

  await page.route("**/api/v1/**", async (route) => {
    const req = route.request();
    const url = req.url();
    const method = req.method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    // --- LLM settings (most specific first) ---
    if (url.includes("/settings/llm/routing")) {
      if (method === "PUT") {
        const body = (req.postDataJSON() ?? {}) as Record<string, unknown>;
        const delayMs = options.routingDelaysMs?.[routingRequest] ?? 0;
        routingRequest += 1;
        if (delayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
        if (routingFailures > 0) {
          routingFailures -= 1;
          return json({ detail: "routing unavailable" }, 503);
        }
        Object.assign(state.routing, body);
        state.routingCommits.push(String(body.processing_mode));
      }
      return json({ ok: true });
    }
    if (url.includes("/settings/llm/providers/")) {
      const name = url.split("/settings/llm/providers/")[1].split(/[/?]/)[0];
      if (url.endsWith("/test")) return json({ ok: true, model: "gpt-4o" });
      const prov = state.providers.find((p) => p.name === name);
      if (method === "PUT" && prov) {
        const body = (req.postDataJSON() ?? {}) as Record<string, unknown>;
        if (typeof body.api_key === "string" && body.api_key) {
          prov.has_key = true;
          prov.configured = true;
          prov.key_masked = `${body.api_key.slice(0, 3)}…${body.api_key.slice(-4)}`;
        }
        if (typeof body.enabled === "boolean") prov.enabled = body.enabled;
        if (typeof body.base_url === "string") prov.base_url = body.base_url;
        if (typeof body.model === "string") prov.model = body.model;
      }
      if (method === "DELETE" && prov) {
        prov.has_key = false;
        prov.configured = prov.is_local;
        prov.key_masked = null;
      }
      return json({ ok: true });
    }
    if (url.includes("/settings/llm")) {
      settingsGet += 1;
      if (options.settingsFailureCalls?.includes(settingsGet)) {
        return json({ detail: "settings unavailable" }, 503);
      }
      return json({ providers: state.providers, routing: state.routing });
    }

    // --- everything else the page + nav touch ---
    if (url.includes("/auth/me")) return json(ME_OK);
    if (url.includes("/auth/refresh"))
      return json({ access_token: "fresh.access.token", refresh_token: "fresh.refresh.token" });
    if (url.includes("/auth/logout")) return json({});
    if (url.includes("/dashboard/overview")) return json(OVERVIEW);
    if (url.includes("/audit-log")) return json({ items: [], total: 0 });
    if (url.includes("/records")) return json({ items: [], total: 0, page: 1, page_size: 100 });
    return json({});
  });

  return state;
}

test.describe("AI providers card (Admin → System)", () => {
  test.beforeEach(async ({ page }) => {
    await injectAuth(page);
  });

  test("renders provider rows and a routing select", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("heading", { name: "AI providers" })).toBeVisible();
    await expect(page.getByLabel("Default AI provider")).toBeVisible();

    // A cloud row and a local row both render their key inputs.
    await expect(page.getByLabel("openai API key")).toBeVisible();
    await expect(page.getByLabel("ollama API key")).toBeVisible();

    // The Advanced disclosure exposes a per-operation select.
    await page.getByText("Advanced — route each operation").click();
    await expect(page.getByLabel("summary provider")).toBeVisible();
  });

  test("renders the contextual intro and a 'Get an API key' link", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    // The muted intro paragraph at the top of the card body.
    await expect(
      page.getByText(/Choose how medical content is processed first/i)
    ).toBeVisible();

    // Cloud providers (openai/anthropic/gemini/openrouter) expose a key link
    // that opens the provider's key page in a new tab.
    const keyLinks = page.getByRole("link", { name: /Get an API key/i });
    await expect(keyLinks.first()).toBeVisible();
    await expect(keyLinks.first()).toHaveAttribute("target", "_blank");
    await expect(keyLinks.first()).toHaveAttribute("rel", /noreferrer/);
  });

  test("saving a key PUTs the provider with the key in the body", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByLabel("openai API key")).toBeVisible();

    const putPromise = page.waitForRequest(
      (r) =>
        r.method() === "PUT" &&
        r.url().includes("/settings/llm/providers/openai") &&
        !r.url().endsWith("/test")
    );

    await page.getByLabel("openai API key").fill("sk-openai-secret-9999");
    await page.getByRole("button", { name: "Save openai key" }).click();

    const putReq = await putPromise;
    expect(putReq.postDataJSON()).toMatchObject({ api_key: "sk-openai-secret-9999" });

    // The component reloads → masked preview shows (never the full key).
    await expect(page.getByText(/9999/)).toBeVisible();
    await expect(page.getByText("sk-openai-secret-9999")).toHaveCount(0);
  });

  test("processing-mode switches persist complete provider-safe routing", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByLabel("Default AI provider")).toBeVisible();
    await expect(
      page.getByText(
        /Scanned PDF and TIFF pages are sent to the vision provider before text can be de-identified/
      ).first()
    ).toBeVisible();
    const privacyCard = page
      .getByRole("heading", { name: "Your data, your control" })
      .locator("..");
    await expect(privacyCard).toContainText(
      "clinical records, source files, and stored AI payloads use application-layer encryption"
    );
    await expect(privacyCard).toContainText(
      "PDF and TIFF OCR sends the original unredacted document or pages"
    );
    await expect(privacyCard).toContainText(
      "Validated strict-local processing does not construct or call a cloud provider"
    );

    const mode = page.getByLabel("Processing mode");
    await expect(mode).toBeVisible();
    // Defaults to the mocked routing value.
    await expect(mode).toHaveValue("cloud_assisted");

    const routingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );

    await mode.selectOption("custom_local");

    const routingReq = await routingPromise;
    expect(routingReq.postDataJSON()).toEqual({
      default: "ollama",
      summary: "ollama",
      section: "ollama",
      dedup: "ollama",
      extraction: "ollama",
      vision: "ollama",
      extraction_engine: "hybrid",
      processing_mode: "custom_local",
    });

    const cloudRoutingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );
    await mode.selectOption("cloud_assisted");
    expect((await cloudRoutingPromise).postDataJSON()).toEqual({
      default: "gemini",
      summary: "gemini",
      section: "gemini",
      dedup: "gemini",
      extraction: "gemini",
      vision: "gemini",
      extraction_engine: "hybrid",
      processing_mode: "cloud_assisted",
    });
  });

  test("serializes processing-mode saves so the newest snapshot commits last", async ({
    page,
  }) => {
    const state = await mockBackend(page, { routingDelaysMs: [700, 0] });
    await page.goto("/admin?tab=sys");

    const mode = page.getByLabel("Processing mode");
    await expect(mode).toHaveValue("cloud_assisted");
    await mode.selectOption("custom_local");
    await mode.selectOption("cloud_assisted");

    await expect.poll(() => state.routingCommits.length).toBe(2);
    expect(state.routingCommits).toEqual([
      "custom_local",
      "cloud_assisted",
    ]);
    expect(state.routing.processing_mode).toBe("cloud_assisted");
    await expect(mode).toHaveValue("cloud_assisted");
  });

  test("a failed routing refresh clears optimistic settings and offers retry", async ({
    page,
  }) => {
    await mockBackend(page, {
      routingFailures: 1,
    });
    await page.goto("/admin?tab=sys");

    await page.getByLabel("Processing mode").selectOption("custom_local");

    await expect(
      page.getByText("AI provider settings are unavailable right now.")
    ).toBeVisible();
    await expect(page.getByLabel("Processing mode")).toHaveCount(0);
    const retry = page.getByRole("button", { name: "Retry AI provider settings" });
    await expect(retry).toBeEnabled();

    await retry.click();
    await expect(page.getByLabel("Processing mode")).toHaveValue(
      "cloud_assisted"
    );
  });

  test("an initial settings failure stays unresolved until retry succeeds", async ({
    page,
  }) => {
    await mockBackend(page, { settingsFailureCalls: [1] });
    await page.goto("/admin?tab=sys");

    await expect(
      page.getByText("AI provider settings are unavailable right now.")
    ).toBeVisible();
    await expect(page.getByLabel("Processing mode")).toHaveCount(0);

    await page
      .getByRole("button", { name: "Retry AI provider settings" })
      .click();
    await expect(page.getByLabel("Processing mode")).toHaveValue(
      "cloud_assisted"
    );
  });

  test("selecting a configured cloud default persists every effective route", async ({
    page,
  }) => {
    const state = await mockBackend(page);
    const openai = state.providers.find((provider) => provider.name === "openai");
    if (!openai) throw new Error("openai fixture missing");
    openai.configured = true;
    openai.has_key = true;
    openai.model = "gpt-user-route";
    await page.goto("/admin?tab=sys");

    const routingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );
    await page.getByLabel("Default AI provider").selectOption("openai");
    expect((await routingPromise).postDataJSON()).toEqual({
      default: "openai",
      summary: "openai",
      section: "openai",
      dedup: "openai",
      extraction: "openai",
      vision: "openai",
      extraction_engine: "hybrid",
      processing_mode: "cloud_assisted",
    });
  });

  test("changing one advanced route PUTs the complete effective routing", async ({
    page,
  }) => {
    const state = await mockBackend(page);
    const openai = state.providers.find((provider) => provider.name === "openai");
    if (!openai) throw new Error("openai fixture missing");
    openai.configured = true;
    openai.has_key = true;
    await page.goto("/admin?tab=sys");

    await expect(page.getByLabel("Default AI provider")).toBeVisible();
    await page.getByText("Advanced — route each operation").click();

    const routingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );

    await page.getByLabel("summary provider").selectOption("openai");

    const routingReq = await routingPromise;
    expect(routingReq.postDataJSON()).toEqual({
      default: "gemini",
      summary: "openai",
      section: "gemini",
      dedup: "gemini",
      extraction: "gemini",
      vision: "gemini",
      extraction_engine: "hybrid",
      processing_mode: "cloud_assisted",
    });
  });

  test("Test surfaces a connection result", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("button", { name: "Test openai" })).toBeVisible();
    await page.getByRole("button", { name: "Test openai" }).click();

    await expect(page.getByText(/OK .* gpt-4o/)).toBeVisible();
  });
});
