import * as fs from "fs";
import * as path from "path";

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

// These legacy developer E2E specs predate signed fixturectl releases. Their
// off-repo corpus uses MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR and originals under
// <root>/raw/. The protected fixture-release variable is reserved for the
// backend's receipt-guarded resolver and must never be read here.
// playwright.config.ts loads .env.test.local for external-AI E2E runs. When the
// legacy variable is unset, TEST_DATA_DIR points at a path that does not exist,
// so data-dependent specs skip cleanly; real PHI has no in-repo fallback.
function privateFixtureRaw(): string {
  const root = process.env.MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR;
  if (!root) {
    return path.join(__dirname, "__MEDTIMELINE_LEGACY_DEV_FIXTURES_DIR_unset__");
  }
  const expanded = root.replace(/^~(?=$|\/)/, process.env.HOME ?? "");
  return path.join(expanded, "raw");
}
export const TEST_DATA_DIR = privateFixtureRaw();
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

export function testIdentifier(specName: string): string {
  return `e2e ${specName}`;
}

// Per-run-unique counter so uploads aren't treated as idempotent re-uploads of a
// prior run's identical content (the backend now skips duplicate file_hash / stable-id
// re-ingestion — see Phase 2a/1). Use for specs that must genuinely ingest each run.
let _uniqueCounter = 0;
export function uniqueEmail(specName: string): string {
  return `e2e-${specName}-${Date.now()}-${_uniqueCounter++}@test.com`;
}

export function uniqueIdentifier(specName: string): string {
  return `e2e ${specName} ${Date.now()} ${_uniqueCounter++}`;
}

export const TEST_PASSWORD = "E2eTest1!";
