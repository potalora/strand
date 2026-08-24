import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { test, expect } from "@playwright/test";
import { ApiClient } from "./helpers/api-client";
import { TEST_PASSWORD, uniqueEmail } from "./helpers/test-data";

test("local-only non-worker profile starts backend without attesting a worker", async ({}, testInfo) => {
  test.skip(
    process.env.E2E_LOCAL_ONLY !== "1",
    "startup sentinel contract requires the local-only profile"
  );
  const runtimeRoot = process.env.E2E_RUNTIME_ROOT;
  const outputRoot = process.env.E2E_OUTPUT_ROOT;
  if (!runtimeRoot || !outputRoot) {
    throw new Error("Local-only task roots must be configured");
  }
  const runtimeReal = fs.realpathSync.native(runtimeRoot);
  const outputReal = fs.realpathSync.native(outputRoot);
  const runtimeStats = fs.statSync(runtimeReal);
  const expectedOutputDir = path.join(outputReal, "artifacts");

  expect(testInfo.project.outputDir).toBe(expectedOutputDir);
  expect(fs.lstatSync(testInfo.project.outputDir).isSymbolicLink()).toBe(false);
  expect(fs.realpathSync.native(testInfo.project.outputDir)).toBe(expectedOutputDir);

  expect(process.env.LOCAL_AI_WORKER_COMMAND).toBe("/usr/bin/false");
  expect(process.env.E2E_ATTESTED_STRICT_PACK).toBe("");
  expect(process.env.LOCAL_AI_MANIFEST_PATH).toContain("apple-m4-16gb-v1.lock.json");
  expect(process.env.LOCAL_AI_RELEASE_EVIDENCE_PATH).toContain(
    "apple-m4-16gb-v1.release.json"
  );
  expect(process.env.LOCAL_AI_BENCHMARK_PATH).toContain(
    "backend/artifacts/local-ai-benchmark.json"
  );
  expect(process.env.LOCAL_AI_FIDELITY_PATH).toContain(
    "backend/artifacts/local-ai-fidelity.json"
  );
  for (const root of [runtimeRoot, outputRoot]) {
    const linkStats = fs.lstatSync(root);
    const stats = fs.statSync(root);
    expect(linkStats.isSymbolicLink()).toBe(false);
    expect(stats.isDirectory()).toBe(true);
    expect(stats.uid).toBe(process.getuid?.());
    expect(stats.mode & 0o777).toBe(0o700);
  }
  expect(runtimeReal).not.toBe(outputReal);

  const runtimeChildren = [
    ["UPLOAD_DIR", "uploads"],
    ["TEMP_EXTRACT_DIR", "temp-extract"],
    ["LOCAL_AI_SCRATCH_DIR", "scratch"],
    ["LOCAL_AI_MODEL_DIR", "models"],
    ["LOCAL_AI_WORKER_PROJECT_DIR", "non-worker-project"],
  ] as const;
  const childIdentities = new Set<string>();
  for (const [environmentName, childName] of runtimeChildren) {
    const configuredChild = process.env[environmentName];
    if (!configuredChild) {
      throw new Error(`${environmentName} must be configured`);
    }
    const expectedChild = path.join(runtimeReal, childName);
    const linkStats = fs.lstatSync(configuredChild);
    const childReal = fs.realpathSync.native(configuredChild);
    const stats = fs.statSync(childReal);

    expect(path.isAbsolute(configuredChild)).toBe(true);
    expect(configuredChild).toBe(expectedChild);
    expect(childReal).toBe(expectedChild);
    expect(path.dirname(childReal)).toBe(runtimeReal);
    expect(linkStats.isSymbolicLink()).toBe(false);
    expect(linkStats.isDirectory()).toBe(true);
    expect(stats.isDirectory()).toBe(true);
    expect(stats.uid).toBe(process.getuid?.());
    expect(stats.mode & 0o777).toBe(0o700);
    expect(stats.dev).toBe(runtimeStats.dev);
    expect(stats.ino).not.toBe(runtimeStats.ino);
    childIdentities.add(`${stats.dev}:${stats.ino}`);
  }
  expect(childIdentities.size).toBe(runtimeChildren.length);

  const response = await fetch("http://127.0.0.1:8000/api/v1/health");
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ status: "healthy", version: "0.1.0" });
});

type StorageSnapshot = {
  rootExists: boolean;
  entries: { name: string; type: string; size: number; sha256?: string }[];
};

function effectiveUploadRoot(): string {
  const runtimeRoot = process.env.E2E_RUNTIME_ROOT;
  const uploadRoot = process.env.UPLOAD_DIR;
  if (
    !runtimeRoot ||
    !uploadRoot ||
    !path.isAbsolute(runtimeRoot) ||
    !path.isAbsolute(uploadRoot)
  ) {
    throw new Error("Local-only runtime and upload roots must be absolute");
  }
  const runtimeLinkStats = fs.lstatSync(runtimeRoot);
  const uploadLinkStats = fs.lstatSync(uploadRoot);
  const runtimeReal = fs.realpathSync.native(runtimeRoot);
  const uploadReal = fs.realpathSync.native(uploadRoot);
  const runtimeStats = fs.statSync(runtimeReal);
  const uploadStats = fs.statSync(uploadReal);
  if (
    runtimeLinkStats.isSymbolicLink() ||
    uploadLinkStats.isSymbolicLink() ||
    !runtimeStats.isDirectory() ||
    !uploadStats.isDirectory() ||
    runtimeStats.uid !== process.getuid?.() ||
    uploadStats.uid !== process.getuid?.() ||
    (runtimeStats.mode & 0o777) !== 0o700 ||
    (uploadStats.mode & 0o777) !== 0o700 ||
    path.dirname(uploadReal) !== runtimeReal
  ) {
    throw new Error("UPLOAD_DIR must be an owned real child of E2E_RUNTIME_ROOT");
  }
  return uploadReal;
}

function ownerStorageSnapshot(userId: string): StorageSnapshot {
  const uploadRoot = effectiveUploadRoot();
  if (!fs.existsSync(uploadRoot)) return { rootExists: false, entries: [] };
  const entries = fs
    .readdirSync(uploadRoot)
    .filter((name) => name.startsWith(`${userId}_`))
    .sort()
    .map((name) => {
      const fullPath = path.join(uploadRoot, name);
      const metadata = fs.lstatSync(fullPath);
      const entry: {
        name: string;
        type: string;
        size: number;
        sha256?: string;
      } = {
        name,
        type: metadata.isFile()
          ? "file"
          : metadata.isDirectory()
            ? "directory"
            : "other",
        size: metadata.size,
      };
      if (metadata.isFile()) {
        entry.sha256 = createHash("sha256")
          .update(fs.readFileSync(fullPath))
          .digest("hex");
      }
      return entry;
    });
  return { rootExists: true, entries };
}

async function ownerIngestionSnapshot(api: ApiClient) {
  const [history, jobs, records, patients, overview] = await Promise.all([
    api.getUploadHistory(),
    api.getLocalAIJobs(),
    api.getRecords({ page: 1, page_size: 100 }),
    api.getPatients(),
    api.getDashboardOverview(),
  ]);
  return {
    uploadIds: history.items.map((item) => item.id).sort(),
    jobIds: jobs.map((item) => item.id).sort(),
    recordIds: records.items.map((item) => item.id).sort(),
    recordTotal: records.total,
    patientIds: patients.items.map((item) => item.id).sort(),
    patientTotal: patients.total,
    overview,
  };
}

test("legacy strict-local upload fails before owner database or storage side effects", async () => {
  test.skip(
    process.env.E2E_LOCAL_ONLY !== "1",
    "strict-local negative admission requires the local-only profile"
  );
  const api = new ApiClient();
  const loginIdentifier = uniqueEmail("strict-local-admission");
  await api.register(loginIdentifier, TEST_PASSWORD);
  await api.login(loginIdentifier, TEST_PASSWORD);
  expect((await api.getLlmSettings()).routing.processing_mode).toBe(
    "validated_strict_local"
  );
  const user = await api.getMe();
  const beforeState = await ownerIngestionSnapshot(api);
  const beforeStorage = ownerStorageSnapshot(user.id);

  const response = await api.attemptTrackedSyntheticFhirUsingStoredPreference();

  expect(response).toEqual({
    status: 409,
    body: { detail: "Strict-local worker runtime identity is required." },
  });
  expect(await ownerIngestionSnapshot(api)).toEqual(beforeState);
  expect(ownerStorageSnapshot(user.id)).toEqual(beforeStorage);
});
