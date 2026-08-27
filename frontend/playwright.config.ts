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

type DirectoryIdentity = {
  realPath: string;
  dev: number;
  ino: number;
  uid: number;
  mode: number;
};

function ownedDirectoryIdentity(
  rawPath: string,
  label: string,
  taskCreated: boolean
): DirectoryIdentity {
  if (!path.isAbsolute(rawPath)) {
    throw new Error(`${label} must be absolute.`);
  }
  const lexicalPath = path.resolve(rawPath);
  const linkStats = fs.lstatSync(lexicalPath);
  if (linkStats.isSymbolicLink() || !linkStats.isDirectory()) {
    throw new Error(`${label} must be a non-symlink directory.`);
  }
  const realPath = fs.realpathSync.native(lexicalPath);
  const stats = fs.statSync(realPath);
  if (!stats.isDirectory() || typeof process.getuid !== "function") {
    throw new Error(`${label} must be an owned directory.`);
  }
  const mode = stats.mode & 0o777;
  const unsafeExistingMode = !taskCreated && (mode & 0o022) !== 0;
  if (
    stats.uid !== process.getuid() ||
    (taskCreated && mode !== 0o700) ||
    unsafeExistingMode
  ) {
    throw new Error(`${label} has unsafe ownership or mode.`);
  }
  return {
    realPath,
    dev: stats.dev,
    ino: stats.ino,
    uid: stats.uid,
    mode,
  };
}

let localOnlyOutputDir: string | undefined;

if (localOnly) {
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
  const configuredRuntimeRoot = process.env.E2E_RUNTIME_ROOT;
  const configuredOutputRoot = process.env.E2E_OUTPUT_ROOT;
  if (!configuredRuntimeRoot || !configuredOutputRoot) {
    throw new Error("E2E runtime and output roots are required.");
  }
  const repoIdentity = ownedDirectoryIdentity(repoRoot, "repository root", false);
  if (repoIdentity.realPath !== repoRoot) {
    throw new Error("Repository root must not traverse a symlink.");
  }
  const evidenceIdentity = ownedDirectoryIdentity(
    path.resolve(__dirname, "test-results"),
    "Playwright evidence parent",
    false
  );
  if (
    path.dirname(evidenceIdentity.realPath) !==
    path.join(repoIdentity.realPath, "frontend")
  ) {
    throw new Error("Playwright evidence parent escaped the worktree.");
  }
  const runtimeParentIdentity = ownedDirectoryIdentity(
    path.join(evidenceIdentity.realPath, "runtime"),
    "runtime parent",
    false
  );
  const outputParentIdentity = ownedDirectoryIdentity(
    path.join(evidenceIdentity.realPath, "executions"),
    "execution parent",
    false
  );
  if (
    path.dirname(runtimeParentIdentity.realPath) !== evidenceIdentity.realPath ||
    path.dirname(outputParentIdentity.realPath) !== evidenceIdentity.realPath
  ) {
    throw new Error("E2E root parent escaped the evidence directory.");
  }
  const rootIdentity = ownedDirectoryIdentity(
    configuredRuntimeRoot,
    "E2E_RUNTIME_ROOT",
    true
  );
  const outputIdentity = ownedDirectoryIdentity(
    configuredOutputRoot,
    "E2E_OUTPUT_ROOT",
    true
  );
  if (path.dirname(rootIdentity.realPath) !== runtimeParentIdentity.realPath) {
    throw new Error("E2E_RUNTIME_ROOT is not an exact runtime-parent child.");
  }
  if (path.dirname(outputIdentity.realPath) !== outputParentIdentity.realPath) {
    throw new Error("E2E_OUTPUT_ROOT is not an exact execution-parent child.");
  }
  if (
    rootIdentity.dev === outputIdentity.dev &&
    rootIdentity.ino === outputIdentity.ino
  ) {
    throw new Error("E2E runtime and output roots must be distinct.");
  }
  const runtimePaths = {
    uploads: path.join(rootIdentity.realPath, "uploads"),
    tempExtract: path.join(rootIdentity.realPath, "temp-extract"),
    scratch: path.join(rootIdentity.realPath, "scratch"),
    models: path.join(rootIdentity.realPath, "models"),
    nonWorkerProject: path.join(rootIdentity.realPath, "non-worker-project"),
  };
  for (const directory of Object.values(runtimePaths)) {
    if (!fs.existsSync(directory)) fs.mkdirSync(directory, { mode: 0o700 });
    const childIdentity = ownedDirectoryIdentity(directory, "runtime child", true);
    if (path.dirname(childIdentity.realPath) !== rootIdentity.realPath) {
      throw new Error("Runtime child escaped E2E_RUNTIME_ROOT.");
    }
  }
  const outputArtifacts = path.join(outputIdentity.realPath, "artifacts");
  if (!fs.existsSync(outputArtifacts)) {
    fs.mkdirSync(outputArtifacts, { mode: 0o700 });
  }
  const outputArtifactsIdentity = ownedDirectoryIdentity(
    outputArtifacts,
    "Playwright output directory",
    true
  );
  if (path.dirname(outputArtifactsIdentity.realPath) !== outputIdentity.realPath) {
    throw new Error("Playwright output directory escaped E2E_OUTPUT_ROOT.");
  }
  localOnlyOutputDir = outputArtifactsIdentity.realPath;
  Object.assign(process.env, {
    [LOCAL_ONLY_PROFILE_INITIALIZED]: "1",
    APP_ENV: "test",
    DATABASE_ENCRYPTION_KEY: "00".repeat(32),
    REAL_MEDICAL_FIXTURES_DIR: "",
    MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR: "",
    E2E_ATTESTED_STRICT_PACK: "",
    LOCAL_AI_ENABLED: "true",
    UPLOAD_DIR: runtimePaths.uploads,
    TEMP_EXTRACT_DIR: runtimePaths.tempExtract,
    LOCAL_AI_MODEL_DIR: runtimePaths.models,
    LOCAL_AI_SCRATCH_DIR: runtimePaths.scratch,
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
    // This sentinel only lets lifespan startup normalize a command. Legacy-v1
    // admission rejects before runtime identity validation or process spawn.
    LOCAL_AI_WORKER_COMMAND: "/usr/bin/false",
    LOCAL_AI_WORKER_PROJECT_DIR: runtimePaths.nonWorkerProject,
    EXTRACTION_ENGINE: "local",
    GEMINI_API_KEY: "",
    GOOGLE_API_KEY: "",
    OPENAI_API_KEY: "",
    OPENROUTER_API_KEY: "",
    ANTHROPIC_API_KEY: "",
    VERTEX_PROJECT: "",
    GOOGLE_CLOUD_PROJECT: "",
    GOOGLE_APPLICATION_CREDENTIALS: "",
    LLM_PROVIDER: "gemini",
    LLM_SUMMARY_PROVIDER: "",
    LLM_SECTION_PROVIDER: "",
    LLM_DEDUP_PROVIDER: "",
    LLM_EXTRACTION_PROVIDER: "",
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
  outputDir: localOnlyOutputDir,
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
