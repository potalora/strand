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
    },
  };
}

async function injectAuth(page: Page): Promise<void> {
  await page.addInitScript((auth) => {
    localStorage.setItem("medtimeline-auth", JSON.stringify(auth));
  }, AUTH_STATE);
}

async function mockBackend(page: Page): Promise<MockState> {
  const state = freshState();

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
      if (method === "PUT") Object.assign(state.routing, req.postDataJSON() ?? {});
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
      page.getByText(/Strand can use different AI providers/i)
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

  test("Extraction-engine select renders and PUTs routing with extraction_engine", async ({
    page,
  }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByLabel("Default AI provider")).toBeVisible();

    // The engine selector lives under the Advanced disclosure, beside the
    // per-operation extraction provider select.
    await page.getByText("Advanced — route each operation").click();

    const engine = page.getByLabel("Extraction engine");
    await expect(engine).toBeVisible();
    // Defaults to the mocked routing value.
    await expect(engine).toHaveValue("hybrid");

    const routingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );

    await engine.selectOption("local");

    const routingReq = await routingPromise;
    expect(routingReq.postDataJSON()).toMatchObject({ extraction_engine: "local" });
  });

  test("changing the default select PUTs routing", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByLabel("Default AI provider")).toBeVisible();

    const routingPromise = page.waitForRequest(
      (r) => r.method() === "PUT" && r.url().includes("/settings/llm/routing")
    );

    await page.getByLabel("Default AI provider").selectOption("ollama");

    const routingReq = await routingPromise;
    expect(routingReq.postDataJSON()).toMatchObject({ default: "ollama" });
  });

  test("Test surfaces a connection result", async ({ page }) => {
    await mockBackend(page);
    await page.goto("/admin?tab=sys");

    await expect(page.getByRole("button", { name: "Test openai" })).toBeVisible();
    await page.getByRole("button", { name: "Test openai" }).click();

    await expect(page.getByText(/OK .* gpt-4o/)).toBeVisible();
  });
});
