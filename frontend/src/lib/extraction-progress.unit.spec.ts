import { test, expect } from "@playwright/test";
import {
  isTerminalStatus,
  formatStage,
  deriveBatch,
} from "./extraction-progress";

/**
 * Pure-logic units behind the upload/extraction UX (session §2a i–iv).
 * No browser, no server — run with:
 *   npx playwright test --config playwright.unit.config.ts
 */

test.describe("isTerminalStatus", () => {
  test("completed / failed / cancelled are terminal", () => {
    expect(isTerminalStatus("completed")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
    expect(isTerminalStatus("cancelled")).toBe(true);
    expect(isTerminalStatus("awaiting_confirmation")).toBe(true);
    expect(isTerminalStatus("duplicate_file")).toBe(true);
  });

  test("in-flight statuses are NOT terminal", () => {
    expect(isTerminalStatus("processing")).toBe(false);
    expect(isTerminalStatus("pending_extraction")).toBe(false);
    expect(isTerminalStatus("pending")).toBe(false);
  });

  test("nullish is not terminal", () => {
    expect(isTerminalStatus(null)).toBe(false);
    expect(isTerminalStatus(undefined)).toBe(false);
    expect(isTerminalStatus("")).toBe(false);
  });
});

test.describe("formatStage", () => {
  test("known stage with section detail renders 'label — section X of Y'", () => {
    expect(
      formatStage("extracting_entities", { section_index: 3, section_total: 8 })
    ).toBe("Extracting entities — section 3 of 8");
  });

  test("known stage without detail renders just the label", () => {
    expect(formatStage("extracting_text", null)).toBe("Extracting text");
    expect(formatStage("scrubbing_phi", undefined)).toBe("De-identifying");
    expect(formatStage("mapping_fhir", null)).toBe("Mapping to records");
  });

  test("unknown stage is humanized, never raw snake_case", () => {
    expect(formatStage("loading_model", null)).toBe("Loading model");
  });

  test("missing stage returns null (render nothing)", () => {
    expect(formatStage(null, null)).toBeNull();
    expect(formatStage(undefined, undefined)).toBeNull();
  });

  test("a zero-section detail collapses to the bare label", () => {
    expect(
      formatStage("extracting_entities", { section_index: 0, section_total: 0 })
    ).toBe("Extracting entities");
  });
});

test.describe("deriveBatch", () => {
  test("null progress yields an empty, non-active batch", () => {
    const b = deriveBatch(null);
    expect(b.total).toBe(0);
    expect(b.percent).toBe(0);
    expect(b.allTerminal).toBe(false);
    expect(b.anyActive).toBe(false);
  });

  test("mid-run percent counts completed+failed+cancelled as done", () => {
    const b = deriveBatch(
      { total: 8, completed: 3, processing: 2, failed: 1, pending: 2, records_created: 12 },
    );
    // done = total - processing - pending = 8 - 2 - 2 = 4 → 50%
    expect(b.done).toBe(4);
    expect(b.percent).toBe(50);
    expect(b.anyActive).toBe(true);
    expect(b.allTerminal).toBe(false);
    expect(b.recordsCreated).toBe(12);
  });

  test("all terminal when nothing is processing or pending", () => {
    const b = deriveBatch(
      { total: 3, completed: 2, processing: 0, failed: 1, pending: 0, records_created: 7 },
    );
    expect(b.allTerminal).toBe(true);
    expect(b.anyActive).toBe(false);
    expect(b.percent).toBe(100);
  });

  test("cancelled count is derived from per-file statuses", () => {
    const b = deriveBatch(
      { total: 2, completed: 1, processing: 0, failed: 0, pending: 0, records_created: 4 },
      { a: "completed", b: "cancelled" }
    );
    expect(b.cancelled).toBe(1);
    expect(b.allTerminal).toBe(true);
  });
});
