import { defineConfig } from "@playwright/test";

/**
 * Server-free unit config for pure helpers (no browser, no webServer).
 *
 * The main `playwright.config.ts` boots uvicorn + `next dev` for browser e2e.
 * Pure logic (e.g. `src/lib/extraction-progress.ts`) needs neither, so this
 * config runs `*.unit.spec.ts` files under `src/` in isolation:
 *
 *   npx playwright test --config playwright.unit.config.ts
 *
 * testDir is `./src`, so the main e2e suite (testDir `./e2e`) never picks these
 * up — they stay out of the browser run entirely.
 */
export default defineConfig({
  testDir: "./src",
  testMatch: "**/*.unit.spec.ts",
  timeout: 15_000,
  workers: 1,
  reporter: "list",
});
