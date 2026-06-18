import { test, expect } from "@playwright/test";
import {
  statusMapFromFiles,
  batchIsPollable,
  type TrackedFile,
} from "./useExtractionStore";

function f(over: Partial<TrackedFile>): TrackedFile {
  return {
    upload_id: over.upload_id ?? "x",
    filename: over.filename ?? "x.pdf",
    status: over.status ?? "pending_extraction",
    progress_stage: null,
    progress_detail: null,
    needsTrigger: over.needsTrigger ?? false,
    triggered: over.triggered ?? false,
  };
}

test.describe("statusMapFromFiles", () => {
  test("flattens tracked files to id → status", () => {
    const map = statusMapFromFiles({
      a: f({ upload_id: "a", status: "completed" }),
      b: f({ upload_id: "b", status: "cancelled" }),
    });
    expect(map).toEqual({ a: "completed", b: "cancelled" });
  });
});

test.describe("batchIsPollable", () => {
  test("a direct (auto-claimed) in-flight file is pollable", () => {
    expect(
      batchIsPollable({ a: f({ status: "processing", needsTrigger: false }) })
    ).toBe(true);
  });

  test("an untriggered ZIP child is NOT pollable (waits for Extract)", () => {
    expect(
      batchIsPollable({
        a: f({ status: "pending_extraction", needsTrigger: true, triggered: false }),
      })
    ).toBe(false);
  });

  test("a triggered ZIP child becomes pollable", () => {
    expect(
      batchIsPollable({
        a: f({ status: "processing", needsTrigger: true, triggered: true }),
      })
    ).toBe(true);
  });

  test("an all-terminal batch is not pollable", () => {
    expect(
      batchIsPollable({
        a: f({ status: "completed" }),
        b: f({ status: "failed" }),
        c: f({ status: "cancelled" }),
      })
    ).toBe(false);
  });
});
