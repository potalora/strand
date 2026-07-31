import { defineConfig } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";
import {
  isLocalOnlyProfileReentry,
  LOCAL_ONLY_PROFILE_INITIALIZED,
  requireDedicatedLoopbackDatabase,
} from "./e2e/helpers/local-only-profile";
import { localOnlyNextServerCommand } from "./e2e/helpers/local-only-next";

const repoRoot = path.resolve(__dirname, "..");
const localOnly = process.env.E2E_LOCAL_ONLY === "1";
const externalAi = process.env.E2E_EXTERNAL_AI === "1";
const configuredDevelopmentDatabase = process.env.DATABASE_URL;

if (localOnly === externalAi) {
  throw new Error(
    "Set exactly one of E2E_LOCAL_ONLY=1 or E2E_EXTERNAL_AI=1 before running E2E tests."
  );
}

function loadEnvFile(envFile: string, allowlist?: ReadonlySet<string>): void {
  if (!fs.existsSync(envFile)) return;
  for (const line of fs.readFileSync(envFile, "utf-8").split("\n")) {
    const trimmed = line.replace(/\r$/, "").trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq > 0) {
      const key = trimmed.slice(0, eq).trim();
      const val = trimmed.slice(eq + 1).trim();
      if (allowlist && !allowlist.has(key)) continue;
      if (!process.env[key]) process.env[key] = val;
    }
  }
}

if (localOnly) {
  loadEnvFile(
    path.resolve(repoRoot, ".env.test.local"),
    new Set(["REAL_MEDICAL_FIXTURES_DIR"])
  );
  const e2eDatabaseUrl = process.env.E2E_DATABASE_URL;
  if (!e2eDatabaseUrl) {
    throw new Error(
      "Set E2E_DATABASE_URL to a dedicated local test database for E2E_LOCAL_ONLY."
    );
  }
  const localOnlyProfileReentry = isLocalOnlyProfileReentry(
    e2eDatabaseUrl,
    configuredDevelopmentDatabase,
    process.env[LOCAL_ONLY_PROFILE_INITIALIZED]
  );
  process.env.DATABASE_URL = requireDedicatedLoopbackDatabase(
    e2eDatabaseUrl,
    configuredDevelopmentDatabase,
    localOnlyProfileReentry
  );
  Object.assign(process.env, {
    [LOCAL_ONLY_PROFILE_INITIALIZED]: "1",
    APP_ENV: "test",
    LOCAL_AI_ENABLED: "true",
    LOCAL_AI_MODEL_DIR: path.resolve(repoRoot, "backend/data/local-ai/models"),
    LOCAL_AI_SCRATCH_DIR: path.resolve(
      repoRoot,
      "backend/data/local-ai/e2e-scratch"
    ),
    LOCAL_AI_MANIFEST_PATH: path.resolve(
      repoRoot,
      "backend/app/model_manifests/apple-m4-16gb-v1.lock.json"
    ),
    LOCAL_AI_RELEASE_EVIDENCE_PATH: path.resolve(
      repoRoot,
      "backend/app/model_manifests/apple-m4-16gb-v1.release.json"
    ),
    LOCAL_AI_BENCHMARK_PATH: path.resolve(
      repoRoot,
      "backend/artifacts/local-ai-benchmark.json"
    ),
    LOCAL_AI_FIDELITY_PATH: path.resolve(
      repoRoot,
      "backend/artifacts/local-ai-fidelity.json"
    ),
    LOCAL_AI_WORKER_COMMAND: `${path.resolve(
      repoRoot,
      "backend/.venv/bin/python"
    )} ${path.resolve(repoRoot, "backend/tests/e2e_local_ai_worker.py")}`,
    EXTRACTION_ENGINE: "local",
    GEMINI_API_KEY: "",
    GOOGLE_API_KEY: "",
    OPENAI_API_KEY: "",
    OPENROUTER_API_KEY: "",
    ANTHROPIC_API_KEY: "",
    VERTEX_PROJECT: "",
    GOOGLE_CLOUD_PROJECT: "",
    GOOGLE_APPLICATION_CREDENTIALS: "",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    HF_HUB_DISABLE_TELEMETRY: "1",
    NEXT_TELEMETRY_DISABLED: "1",
    DO_NOT_TRACK: "1",
    HTTP_PROXY: "http://127.0.0.1:9",
    HTTPS_PROXY: "http://127.0.0.1:9",
    ALL_PROXY: "http://127.0.0.1:9",
    NO_PROXY: "localhost,127.0.0.1",
    http_proxy: "http://127.0.0.1:9",
    https_proxy: "http://127.0.0.1:9",
    all_proxy: "http://127.0.0.1:9",
    no_proxy: "localhost,127.0.0.1",
    LOGIN_RATE_LIMIT: "1000",
    REGISTER_RATE_LIMIT: "1000",
  });
} else {
  loadEnvFile(path.resolve(repoRoot, "backend", ".env"));
  loadEnvFile(path.resolve(repoRoot, ".env"));
  loadEnvFile(path.resolve(repoRoot, ".env.test.local"));
}

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  // External runs share the backend login limiter across three parallel workers,
  // so a UI login can transiently 429 under burst load.
  retries: localOnly ? 0 : 2,
  workers: localOnly ? 1 : 3,
  expect: {
    timeout: 30_000,
  },
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
    screenshot: "only-on-failure",
    serviceWorkers: localOnly ? "block" : "allow",
    proxy: localOnly
      ? {
          server: "http://127.0.0.1:9",
          bypass: "localhost,127.0.0.1,[::1]",
        }
      : undefined,
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: [
    {
      command:
        localOnly
          ? "cd ../backend && .venv/bin/python tests/e2e_local_backend.py"
          : "cd ../backend && source .venv/bin/activate && alembic upgrade head && uvicorn app.main:app --port 8000",
      port: 8000,
      reuseExistingServer: !localOnly,
      timeout: 30_000,
    },
    {
      command: localOnly ? localOnlyNextServerCommand() : "npm run dev",
      port: 3000,
      reuseExistingServer: !localOnly,
      timeout: 30_000,
    },
  ],
});
