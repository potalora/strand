import { test, expect, isAllowedConsoleText } from "./fixtures/console-gate";

/**
 * Proof that the global console-error gate (fixtures/console-gate.ts) works.
 *
 *  1. It does NOT false-positive on a real, clean page (this test is gated and
 *     would itself fail if /login emitted any disallowed console error).
 *  2. Its detection predicate flags the exact symptoms we care about — a plain
 *     console.error and the React duplicate-key warning (bug #1's symptom) — and
 *     correctly allowlists benign HTTP-status "Failed to load resource" noise,
 *     verified end-to-end through the real browser console pipe.
 */
test.describe("Console-error gate", () => {
  test("passes on a clean page (no false positives)", async ({ page }) => {
    await page.goto("/login");
    // The login form renders; if the gate had tripped on benign noise this test
    // would have failed in teardown.
    await expect(page.locator("#email")).toBeVisible();
    await expect(page.locator("#password")).toBeVisible();
  });

  test("detects console.error + React duplicate-key warning, allowlists HTTP noise", async ({
    page,
  }, testInfo) => {
    // This test intentionally provokes console.errors, so opt out of the gate's
    // own assertion (we assert the detection ourselves below).
    testInfo.annotations.push({ type: "allow-console-errors" });

    // Mirror exactly what the gate listens to.
    const captured: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") captured.push(msg.text());
    });

    await page.goto("/login");

    // The React message format for the duplicate-key warning that the pagination
    // bug (#1) produced, plus a plain app-style error and a benign HTTP-noise
    // line that must be allowlisted.
    const DUP_KEY =
      "Warning: Encountered two children with the same key, `dup-id`. Keys should be unique so that components maintain their identity across updates.";
    const PLAIN = "[gate-probe] synthetic application console error";
    const BENIGN =
      "Failed to load resource: the server responded with a status of 404 (Not Found)";

    await page.evaluate(
      ([dup, plain, benign]) => {
        console.error(dup);
        console.error(plain);
        console.error(benign);
      },
      [DUP_KEY, PLAIN, BENIGN]
    );

    // Wait until all three reach our mirror listener.
    await expect.poll(() => captured.length, { timeout: 5_000 }).toBeGreaterThanOrEqual(3);

    // The two real violations are NOT allowlisted (the gate would fail on them)...
    expect(isAllowedConsoleText(DUP_KEY)).toBe(false);
    expect(isAllowedConsoleText(PLAIN)).toBe(false);
    // ...and the HTTP-status noise IS allowlisted (the gate ignores it).
    expect(isAllowedConsoleText(BENIGN)).toBe(true);

    // End-to-end: the duplicate-key warning actually travelled through the
    // browser console as an `error` and was captured.
    expect(captured.some((t) => t.includes("two children with the same key"))).toBe(
      true
    );
  });
});
