import { defineConfig } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

// Load backend .env so tests can check GEMINI_API_KEY, etc.
const backendEnv = path.resolve(__dirname, "../backend/.env");
if (fs.existsSync(backendEnv)) {
  for (const line of fs.readFileSync(backendEnv, "utf-8").split("\n")) {
    const trimmed = line.replace(/\r$/, "").trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq > 0) {
      const key = trimmed.slice(0, eq).trim();
      const val = trimmed.slice(eq + 1).trim();
      if (!process.env[key]) process.env[key] = val;
    }
  }
}

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  // The backend login rate limiter (5/60s per IP) is shared across the 3 parallel
  // workers, so a UI login can transiently 429 under burst load. Retries absorb
  // that contention; a genuinely broken test still fails on every attempt.
  retries: 2,
  workers: 3,
  expect: {
    timeout: 30_000,
  },
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
    screenshot: "only-on-failure",
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
        "cd ../backend && source .venv/bin/activate && uvicorn app.main:app --port 8000",
      port: 8000,
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      port: 3000,
      reuseExistingServer: true,
      timeout: 30_000,
    },
  ],
});
