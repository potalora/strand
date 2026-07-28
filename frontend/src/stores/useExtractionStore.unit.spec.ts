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
    local_run: null,
    local_failure: null,
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

test.describe("mergeFileStatuses local provenance", () => {
  test.beforeEach(() => useExtractionStore.getState().reset());

  test("merges local run data without replacing other tracked files", () => {
    useExtractionStore.getState().startBatch([
      { upload_id: "A", filename: "a.pdf", status: "processing" },
      { upload_id: "B", filename: "b.pdf", status: "pending_extraction" },
    ]);

    useExtractionStore.getState().mergeFileStatuses([
      {
        id: "A",
        ingestion_status: "processing",
        local_run: {
          privacy_mode: "validated_strict_local",
          models: [
            {
              role: "ocr",
              repository: "sahilchachra/ovisocr2-int4-mlx",
              revision: "0123456789abcdef0123456789abcdef01234567",
            },
          ],
        },
      },
    ]);

    const state = useExtractionStore.getState();
    expect(state.files.A.local_run?.models[0]?.role).toBe("ocr");
    expect(state.files.B.filename).toBe("b.pdf");
    expect(state.batchIds).toEqual(["A", "B"]);
  });

  test("merges a fail-closed local failure into the tracked file", () => {
    useExtractionStore.getState().startBatch([
      { upload_id: "A", filename: "a.pdf", status: "processing" },
    ]);

    useExtractionStore.getState().mergeFileStatuses([
      {
        id: "A",
        ingestion_status: "failed",
        local_failure: {
          stage: "ocr",
          code: "worker_failed",
          message: "Local OCR did not complete.",
          model_role: "ocr",
          repository: "sahilchachra/ovisocr2-int4-mlx",
          revision: "0123456789abcdef0123456789abcdef01234567",
          retryable: true,
          checkpoint_preserved: true,
          cloud_fallback_attempted: false,
        },
      },
    ]);

    const tracked = useExtractionStore.getState().files.A;
    expect(tracked.status).toBe("failed");
    expect(tracked.local_failure?.code).toBe("worker_failed");
    expect(tracked.local_failure?.cloud_fallback_attempted).toBe(false);
  });
});
