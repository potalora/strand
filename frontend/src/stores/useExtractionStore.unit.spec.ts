import { test, expect } from "@playwright/test";
import {
  statusMapFromFiles,
  batchIsPollable,
  useExtractionStore,
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

test.describe("startBatch merge-or-replace", () => {
  test.beforeEach(() => useExtractionStore.getState().reset());

  test("merges a new upload into an in-flight batch", () => {
    const s = useExtractionStore.getState();
    s.startBatch([{ upload_id: "A", filename: "a.pdf", status: "processing" }]);
    s.startBatch([{ upload_id: "B", filename: "b.pdf", status: "pending_extraction" }]);
    const st = useExtractionStore.getState();
    expect(st.batchIds).toEqual(["A", "B"]);
    expect(Object.keys(st.files).sort()).toEqual(["A", "B"]);
    expect(st.dismissed).toBe(false);
  });

  test("replaces when the prior batch is fully terminal", () => {
    useExtractionStore.getState().startBatch([
      { upload_id: "A", filename: "a.pdf", status: "processing" },
    ]);
    useExtractionStore.getState().mergeFileStatuses([
      { id: "A", ingestion_status: "completed" },
    ]);
    useExtractionStore.getState().startBatch([
      { upload_id: "B", filename: "b.pdf", status: "pending_extraction" },
    ]);
    const st = useExtractionStore.getState();
    expect(st.batchIds).toEqual(["B"]);
    expect(Object.keys(st.files)).toEqual(["B"]);
  });

  test("does not duplicate an already-tracked id", () => {
    const s = useExtractionStore.getState();
    s.startBatch([{ upload_id: "A", filename: "a.pdf", status: "processing" }]);
    s.startBatch([{ upload_id: "A", filename: "a.pdf", status: "processing" }]);
    expect(useExtractionStore.getState().batchIds).toEqual(["A"]);
  });
});
