import { test as base, expect } from "@playwright/test";

/**
 * Global console-error gate (e2e hardening — Task 2a).
 *
 * Fails ANY test whose page emits a `console.error`, a React key / duplicate-key
 * warning, or an uncaught `pageerror`. This is the highest-leverage guard in the
 * suite: the React "Encountered two children with the same key" warning — the
 * visible symptom of the pagination instability bug (#1) — is a `console.error`
 * that printed to the console but failed nothing before this gate existed.
 *
 * Wiring: spec files import `{ test, expect }` from this module instead of from
 * `@playwright/test`. The gate is an auto, test-scoped fixture, so it arms for
 * every test in those files with no per-test boilerplate.
 *
 * Allowlist — benign output only, each justified:
 *  - `Failed to load resource` / `the server responded with a status of` — the
 *    browser logs EVERY non-2xx HTTP response as a console error. Several specs
 *    deliberately drive 401s (token-refresh races), 500s and aborted requests
 *    (error-handling), and mocked error responses; the app surfaces/handles
 *    these and they are not regressions.
 *  - `net::ERR_` — aborted/failed network requests from `route.abort()` mocks.
 *  - `ResizeObserver loop …` — a benign browser notice from the layout observer
 *    used by the Overview marker grid; not an app error.
 *
 * The Next.js dev "Download the React DevTools" notice is `console.info`, not
 * `error`, so it never reaches this gate.
 *
 * Opt-out: a single test that intentionally provokes an app-level console.error
 * can annotate itself with `{ type: "allow-console-errors" }` (see the gate's own
 * proof test). Use sparingly and document why.
 */

const ALLOWLIST: RegExp[] = [
  /Failed to load resource/i,
  /the server responded with a status of/i,
  /net::ERR_/i,
  /ResizeObserver loop (limit exceeded|completed with undelivered notifications)/i,
];

export function isAllowedConsoleText(text: string): boolean {
  return ALLOWLIST.some((re) => re.test(text));
}

export const test = base.extend<{ consoleGate: void }>({
  consoleGate: [
    async ({ page }, use, testInfo) => {
      const violations: string[] = [];

      page.on("console", (msg) => {
        if (msg.type() !== "error") return;
        const text = msg.text();
        if (isAllowedConsoleText(text)) return;
        violations.push(`[console.error] ${text}`);
      });

      page.on("pageerror", (err) => {
        const text = err.message || String(err);
        if (isAllowedConsoleText(text)) return;
        violations.push(`[pageerror] ${text}`);
      });

      await use();

      const optedOut = testInfo.annotations.some(
        (a) => a.type === "allow-console-errors"
      );
      if (!optedOut && violations.length > 0) {
        throw new Error(
          `Console-error gate: page emitted ${violations.length} disallowed ` +
            `console error(s)/pageerror(s):\n` +
            violations.map((v) => `  • ${v}`).join("\n")
        );
      }
    },
    { auto: true },
  ],
});

export { expect };
// Re-export Playwright types so specs can import them from the gate alongside
// { test, expect } (keeps a single import line per spec).
export type { Page } from "@playwright/test";
