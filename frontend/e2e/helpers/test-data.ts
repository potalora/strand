import * as fs from "fs";
import * as path from "path";

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
export const TEST_DATA_DIR = path.join(REPO_ROOT, "test_data");
export const FIXTURES_DIR = path.join(REPO_ROOT, "backend", "tests", "fixtures");

export const PATHS = {
  fhirBundle: path.join(FIXTURES_DIR, "sample_fhir_bundle.json"),
  epicExport: path.join(TEST_DATA_DIR, "Requested Record"),
  epicTsvDir: path.join(TEST_DATA_DIR, "Requested Record", "EHITables"),
  rtfDir: path.join(TEST_DATA_DIR, "Requested Record", "Rich Text"),
  healthSummary: path.join(TEST_DATA_DIR, "HealthSummary_Apr_05_2026"),
  xdmDir: path.join(TEST_DATA_DIR, "HealthSummary_Apr_05_2026", "IHE_XDM"),
  cdaExport: path.join(TEST_DATA_DIR, "EhiExport-22259"),
};

export function hasTestData(dataPath: string): boolean {
  return fs.existsSync(dataPath);
}

export function getRtfFiles(count: number = 3): string[] {
  if (!hasTestData(PATHS.rtfDir)) return [];
  const files = fs
    .readdirSync(PATHS.rtfDir)
    .filter((f) => f.toUpperCase().endsWith(".RTF"))
    .slice(0, count)
    .map((f) => path.join(PATHS.rtfDir, f));
  return files;
}

export function testEmail(specName: string): string {
  return `e2e-${specName}@test.com`;
}

// Per-run-unique counter so uploads aren't treated as idempotent re-uploads of a
// prior run's identical content (the backend now skips duplicate file_hash / stable-id
// re-ingestion — see Phase 2a/1). Use for specs that must genuinely ingest each run.
let _uniqueCounter = 0;
export function uniqueEmail(specName: string): string {
  return `e2e-${specName}-${Date.now()}-${_uniqueCounter++}@test.com`;
}

export const TEST_PASSWORD = "E2eTest1!";
