"use client";

import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import {
  FolderUp,
  FileText,
  FileArchive,
  ChevronDown,
  ChevronUp,
  Lock,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import { useDirectoryUpload } from "@/hooks/useDirectoryUpload";
import { getFilesFromDrop } from "@/lib/getFilesFromDrop";
import { api } from "@/lib/api";
import type {
  UploadResponse,
  UnstructuredUploadResponse,
  TriggerExtractionResponse,
} from "@/types/api";
import {
  deriveBatch,
  formatStage,
  isTerminalStatus,
} from "@/lib/extraction-progress";
import {
  statusMapFromFiles,
  useExtractionStore,
  type TrackedFileInput,
} from "@/stores/useExtractionStore";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { ConfirmDialog } from "@/components/retro/ConfirmDialog";

/* ==========================================
   FILE CLASSIFICATION HELPERS
   ========================================== */

const STRUCTURED_EXTENSIONS = new Set([".json", ".zip", ".tsv"]);
const UNSTRUCTURED_EXTENSIONS = new Set([".pdf", ".rtf", ".tif", ".tiff"]);

function getExtension(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
}

function isStructured(file: File): boolean {
  return STRUCTURED_EXTENSIONS.has(getExtension(file.name));
}

function isUnstructured(file: File): boolean {
  return UNSTRUCTURED_EXTENSIONS.has(getExtension(file.name));
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ==========================================
   STATUS PILL HELPERS — neutral editorial hues
   ========================================== */

// Map a status to a tdot color drawn from the design tokens. We keep the pill
// neutral (.tag) and only color the dot, matching the Overview/admin treatment.
function statusDotColor(status: string): string {
  switch (status) {
    case "completed":
    case "completed_with_merges":
    case "awaiting_confirmation":
    case "awaiting_review":
    case "parsed":
      return "var(--success)";
    case "processing":
    case "pending_extraction":
    case "dedup_scanning":
    case "pending":
      return "var(--primary)";
    case "failed":
      return "var(--danger)";
    case "cancelled":
    case "duplicate_file":
    case "duplicate":
      return "var(--text-muted)";
    default:
      return "var(--text-muted)";
  }
}

/* ==========================================
   UPLOAD HISTORY TYPES
   ========================================== */

interface UploadHistoryItem {
  id: string;
  filename: string;
  ingestion_status: string;
  records_inserted?: number;
  created_at: string;
  file_category?: string;
  record_count?: number;
  ingestion_progress?: {
    records_inserted?: number;
    records_updated?: number;
    records_unchanged?: number;
    records_skipped?: number;
    duplicate_of?: string;
    record_count?: number;
    total_entries?: number;
  };
  ingestion_errors?: Array<Record<string, unknown>>;
}

function statusLabel(status: string): string {
  if (status === "duplicate_file") return "Duplicate";
  if (status === "pending_extraction") return "pending";
  return status;
}

function recordSummary(u: UploadHistoryItem): string {
  const p = u.ingestion_progress ?? {};
  const parts: string[] = [];
  if (p.records_inserted) parts.push(`${p.records_inserted} new`);
  if (p.records_updated) parts.push(`${p.records_updated} updated`);
  if (p.records_unchanged) parts.push(`${p.records_unchanged} unchanged`);
  if (parts.length) return parts.join(" · ");
  const n = u.record_count ?? u.records_inserted;
  return n != null ? String(n) : "--";
}

function formatUploadError(err: unknown): string {
  if (typeof err === "string") return err;
  if (err && typeof err === "object") {
    const e = err as Record<string, unknown>;
    if (e.stage === "entity_extraction") {
      return `Entity extraction: ${e.failed_chunks} of ${e.total_chunks} sections failed — some records may be incomplete`;
    }
    if (typeof e.error === "string") return e.error;
  }
  return JSON.stringify(err);
}

/* ==========================================
   UPLOAD RESULT TRACKING
   ========================================== */

interface UploadResult {
  type: "structured" | "unstructured";
  filename: string;
  response?: UploadResponse | UnstructuredUploadResponse;
  error?: string;
}

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

/* ==========================================
   MAIN UPLOAD PAGE
   ========================================== */

export default function UploadPage() {
  // --- Mount transition ---
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  // --- File selection state ---
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  // --- Upload state ---
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // --- Upload history ---
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<UploadHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  // --- Upload deletion state ---
  const [deleteTarget, setDeleteTarget] = useState<UploadHistoryItem | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // --- Current extraction batch (shared with the global status bar) ---
  // The store is the single source of truth: starting a batch RESETS the prior
  // one (so a new upload never carries stale rows), progress is polled scoped to
  // THIS batch by the always-mounted GlobalExtractionStatusBar, and `dismissed`
  // drives the terminal "all done" + Dismiss.
  const batchIds = useExtractionStore((s) => s.batchIds);
  const files = useExtractionStore((s) => s.files);
  const progress = useExtractionStore((s) => s.progress);
  const dismissed = useExtractionStore((s) => s.dismissed);
  const cancelling = useExtractionStore((s) => s.cancelling);
  const startBatch = useExtractionStore((s) => s.startBatch);
  const mergeFileStatuses = useExtractionStore((s) => s.mergeFileStatuses);
  const markTriggered = useExtractionStore((s) => s.markTriggered);
  const markCancelling = useExtractionStore((s) => s.markCancelling);
  const dismissBatch = useExtractionStore((s) => s.dismiss);

  // ZIP-extracted children that still need a manual Extract click.
  const [selectedForExtraction, setSelectedForExtraction] = useState<Set<string>>(
    new Set()
  );

  // --- Directory upload (client-side zipping) ---
  const {
    folderInputRef,
    isZipping,
    zipProgress,
    folderInfo,
    selectFolder,
    handleFolderSelect,
    createZipFromFiles,
    clearFolderInfo,
  } = useDirectoryUpload({
    onZipReady: (file) => {
      setSelectedFiles((prev) => [...prev, file]);
      setUploadResults([]);
      setUploadError(null);
    },
    onError: (message) => setUploadError(message),
  });

  // --- Load history (reusable so delete can refetch immediately) ---
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const data = await api.get<{ items: UploadHistoryItem[]; total: number }>(
        "/upload/history"
      );
      setHistory(data.items || []);
      setHistoryLoaded(true);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // --- Load history when section is opened ---
  useEffect(() => {
    if (!historyOpen || historyLoaded) return;
    void fetchHistory();
  }, [historyOpen, historyLoaded, fetchHistory]);

  // When the batch reaches a terminal state, refresh history so rows show final
  // status/record counts instead of the stale pending snapshot.
  const batch = deriveBatch(progress, statusMapFromFiles(files));
  useEffect(() => {
    if (batch.allTerminal) setHistoryLoaded(false);
  }, [batch.allTerminal]);

  // --- Delete an upload (cascade soft-deletes its records server-side) ---
  const performDeleteUpload = useCallback(
    async (id: string) => {
      setDeletingId(id);
      setDeleteError(null);
      try {
        await api.delete<void>(`/upload/${id}`);
        // Refetch so the cascade-deleted upload disappears and record counts
        // reflect the server state.
        await fetchHistory();
      } catch (err) {
        setDeleteError(
          err instanceof Error ? err.message : "Failed to delete upload"
        );
      } finally {
        setDeletingId(null);
        setDeleteTarget(null);
      }
    },
    [fetchHistory]
  );

  // --- Drop handler ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFiles((prev) => [...prev, ...acceptedFiles]);
      setUploadResults([]);
      setUploadError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      "application/json": [".json"],
      "application/zip": [".zip"],
      "application/x-zip-compressed": [".zip"],
      "text/tab-separated-values": [".tsv"],
      "application/pdf": [".pdf"],
      "application/rtf": [".rtf"],
      "text/rtf": [".rtf"],
      "image/tiff": [".tif", ".tiff"],
    },
    multiple: true,
    // Disable react-dropzone's built-in click-to-open. Its root click handler
    // fires the FILE picker, which raced with our "Select folder" button: a
    // single trusted click opened TWO dialogs (file + directory) because the
    // button's stopPropagation does not reliably suppress the ancestor root
    // handler, and the file picker could win — leaving the user unable to pick
    // a folder. With noClick the body is drag-only; the explicit "Browse files"
    // (open()) and "Select folder" (folder input) buttons each open exactly one
    // picker per gesture. Drag-and-drop is unaffected by noClick.
    noClick: true,
  });

  // --- Clear all selected files ---
  const clearFiles = useCallback(() => {
    setSelectedFiles([]);
    setUploadResults([]);
    setUploadError(null);
    clearFolderInfo();
  }, [clearFolderInfo]);

  // --- Upload all files ---
  const handleUploadAll = useCallback(async () => {
    if (selectedFiles.length === 0) return;
    setUploading(true);
    setUploadError(null);
    setUploadResults([]);

    const results: UploadResult[] = [];
    // Every unstructured upload ID produced by THIS action — direct files,
    // batch files, and ZIP-extracted children — becomes the new batch.
    const batchInputs: TrackedFileInput[] = [];

    // Separate structured vs unstructured
    const structured = selectedFiles.filter(isStructured);
    const unstructured = selectedFiles.filter(isUnstructured);

    // Upload structured files one at a time
    for (const file of structured) {
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await api.postForm<UploadResponse>("/upload", formData);
        results.push({ type: "structured", filename: file.name, response: resp });
        if (resp.unstructured_uploads && resp.unstructured_uploads.length > 0) {
          for (const u of resp.unstructured_uploads) {
            // ZIP children are NOT auto-claimed by the worker — they need a
            // manual Extract, so mark needsTrigger.
            batchInputs.push({
              upload_id: u.upload_id,
              filename: u.filename,
              status: u.status || "pending_extraction",
              needsTrigger: true,
            });
          }
        }
      } catch (err) {
        results.push({
          type: "structured",
          filename: file.name,
          error: err instanceof Error ? err.message : "Upload failed",
        });
      }
    }

    // Upload unstructured files (auto-claimed by the extraction worker).
    if (unstructured.length === 1) {
      const file = unstructured[0];
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await api.postForm<UnstructuredUploadResponse>(
          "/upload/unstructured",
          formData
        );
        results.push({ type: "unstructured", filename: file.name, response: resp });
        batchInputs.push({
          upload_id: resp.upload_id,
          filename: file.name,
          status: resp.status || "pending_extraction",
          needsTrigger: false,
        });
      } catch (err) {
        results.push({
          type: "unstructured",
          filename: file.name,
          error: err instanceof Error ? err.message : "Upload failed",
        });
      }
    } else if (unstructured.length > 1) {
      try {
        const formData = new FormData();
        for (const file of unstructured) {
          formData.append("files", file);
        }
        const resp = await api.postForm<{
          uploads: UnstructuredUploadResponse[];
        }>("/upload/unstructured-batch", formData);
        for (let i = 0; i < resp.uploads.length; i++) {
          const upload = resp.uploads[i];
          const filename = unstructured[i]?.name || `file-${i}`;
          results.push({ type: "unstructured", filename, response: upload });
          batchInputs.push({
            upload_id: upload.upload_id,
            filename,
            status: upload.status || "pending_extraction",
            needsTrigger: false,
          });
        }
      } catch (err) {
        for (const file of unstructured) {
          results.push({
            type: "unstructured",
            filename: file.name,
            error: err instanceof Error ? err.message : "Batch upload failed",
          });
        }
      }
    }

    // Replace any prior batch with this one (resets stale rows — §2a ii). The
    // global status bar picks it up and starts polling scoped to these IDs.
    if (batchInputs.length > 0) {
      startBatch(batchInputs);
      setSelectedForExtraction(new Set());
    }

    setUploadResults(results);
    setSelectedFiles([]);
    setUploading(false);
    setHistoryLoaded(false);
  }, [selectedFiles, startBatch]);

  // --- Trigger extraction for the selected ZIP children ---
  const handleTriggerExtraction = useCallback(async () => {
    if (selectedForExtraction.size === 0) return;
    const ids = Array.from(selectedForExtraction);
    // Optimistically reflect the trigger; polling refines from here.
    markTriggered(ids);
    mergeFileStatuses(ids.map((id) => ({ id, ingestion_status: "processing" })));
    try {
      const resp = await api.post<TriggerExtractionResponse>(
        "/upload/trigger-extraction",
        { upload_ids: ids }
      );
      mergeFileStatuses(
        resp.results.map((r) => ({ id: r.upload_id, ingestion_status: r.status }))
      );
    } catch {
      /* the file stays tracked; the next poll reflects reality */
    }
  }, [selectedForExtraction, markTriggered, mergeFileStatuses]);

  // --- Cancel in-flight extractions ---
  const handleCancel = useCallback(
    async (ids: string[]) => {
      if (ids.length === 0) return;
      markCancelling(ids);
      try {
        await api.cancelExtraction(ids);
      } catch {
        /* the file stays in flight; next poll reflects reality */
      }
    },
    [markCancelling]
  );

  // --- Retry failed files (re-batches them so the bar tracks the retry) ---
  const handleRetryFailed = useCallback(async () => {
    try {
      const resp = await api.get<{
        files: { id: string; filename: string }[];
      }>("/upload/pending-extraction?statuses=failed");
      if (resp.files.length === 0) return;
      const inputs: TrackedFileInput[] = resp.files.map((f) => ({
        upload_id: f.id,
        filename: f.filename,
        status: "processing",
        needsTrigger: false,
      }));
      await api.post<TriggerExtractionResponse>("/upload/trigger-extraction", {
        upload_ids: inputs.map((f) => f.upload_id),
      });
      startBatch(inputs);
    } catch {
      /* silently fail */
    }
  }, [startBatch]);

  // --- Retry / extract a single history row ---
  const handleHistoryExtract = useCallback(
    async (upload: UploadHistoryItem) => {
      try {
        await api.post<TriggerExtractionResponse>("/upload/trigger-extraction", {
          upload_ids: [upload.id],
        });
        startBatch([
          {
            upload_id: upload.id,
            filename: upload.filename,
            status: "processing",
            needsTrigger: false,
          },
        ]);
        setHistoryLoaded(false);
      } catch {
        /* silently fail */
      }
    },
    [startBatch]
  );

  const toggleExtractionSelection = useCallback((uploadId: string) => {
    setSelectedForExtraction((prev) => {
      const next = new Set(prev);
      if (next.has(uploadId)) next.delete(uploadId);
      else next.add(uploadId);
      return next;
    });
  }, []);

  // --- Directory drop handler (intercepts before react-dropzone) ---
  const handleDropCapture = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      if (!e.dataTransfer) return;
      const result = await getFilesFromDrop(e.dataTransfer);
      if (result) {
        e.stopPropagation();
        e.preventDefault();
        createZipFromFiles(result.files, result.folderName);
      }
    },
    [createZipFromFiles]
  );

  // --- Derived counts ---
  const structuredCount = selectedFiles.filter(isStructured).length;
  const unstructuredCount = selectedFiles.filter(isUnstructured).length;

  // --- Batch-derived view state ---
  const fileList = batchIds.map((id) => files[id]).filter(Boolean);
  // ZIP children still awaiting a manual Extract.
  const triggerFiles = fileList.filter(
    (f) => f.needsTrigger && !f.triggered && !isTerminalStatus(f.status)
  );
  // Files already in (or done with) the extraction pipeline.
  const pipelineFiles = fileList.filter((f) => !f.needsTrigger || f.triggered);
  const showTriggerPanel = !dismissed && triggerFiles.length > 0;
  const showProgressCard =
    !dismissed &&
    batchIds.length > 0 &&
    (progress !== null || pipelineFiles.length > 0);
  const inFlight = pipelineFiles.filter(
    (f) => !isTerminalStatus(f.status) && !cancelling.includes(f.upload_id)
  );

  const allTriggerSelected =
    triggerFiles.length > 0 &&
    triggerFiles.every((f) => selectedForExtraction.has(f.upload_id));
  const toggleSelectAllExtraction = useCallback(() => {
    if (allTriggerSelected) {
      setSelectedForExtraction(new Set());
    } else {
      setSelectedForExtraction(new Set(triggerFiles.map((f) => f.upload_id)));
    }
  }, [allTriggerSelected, triggerFiles]);

  // --- Render ---
  return (
    <div className={`screen s24 ${shown ? "on" : ""}`}>
      {/* ==========================================
          HEADER
          ========================================== */}
      <div className="page-top">
        <div>
          <p className="kicker">Add to your record</p>
          <h1 className="h1 display">Uploads</h1>
        </div>
        <SecureChip />
      </div>

      {/* ==========================================
          HERO DROPZONE
          ========================================== */}
      <div
        {...getRootProps()}
        onDropCapture={handleDropCapture}
        className={`dropzone ${isDragActive ? "drag" : ""}`}
        style={{ cursor: "default" }}
      >
        <input {...getInputProps()} />
        <div className="dz-ic">
          <FolderUp size={24} strokeWidth={1.8} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", margin: 0 }}>
            Drop files or a folder to add to the record
          </p>
          <p className="dim" style={{ fontSize: 13, margin: "4px 0 0" }}>
            FHIR R4 · Epic EHI (.zip) · C-CDA (.xml) · PDF / RTF / TIFF — encrypted on arrival.
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, flexShrink: 0 }}>
          <button
            className="btn ghost"
            onClick={(e) => {
              // Defensive: keep the click off the dropzone root (it is
              // drag-only now, but this guards against a future body handler).
              e.stopPropagation();
              open();
            }}
          >
            <FileText size={14} />
            Browse files
          </button>
          <button
            className="btn"
            onClick={(e) => {
              e.stopPropagation();
              selectFolder();
            }}
            disabled={isZipping}
          >
            <FileArchive size={14} />
            Select folder
          </button>
          <input
            ref={folderInputRef}
            type="file"
            webkitdirectory=""
            directory=""
            multiple
            onChange={handleFolderSelect}
            style={{ display: "none" }}
          />
        </div>
      </div>

      {/* ==========================================
          ZIPPING PROGRESS
          ========================================== */}
      {isZipping && folderInfo && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Preparing upload</h3>
            <span className="muted mono" style={{ fontSize: 12 }}>
              {zipProgress.toFixed(0)}%
            </span>
          </div>
          <p className="dim" style={{ fontSize: 13, marginBottom: 12 }}>
            {folderInfo.name} &mdash; {folderInfo.fileCount.toLocaleString()} files (
            {formatFileSize(folderInfo.totalSize)})
          </p>
          <div className="bar">
            <i style={{ width: `${zipProgress}%` }} />
          </div>
        </div>
      )}

      {/* ==========================================
          SELECTED FILES DISPLAY
          ========================================== */}
      {selectedFiles.length > 0 && (
        <div className="card-surface pad">
          <div className="card-h">
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <h3 className="sec-title">
                {selectedFiles.length} file{selectedFiles.length !== 1 ? "s" : ""} ready
              </h3>
              {structuredCount > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--success)" }} />
                  {structuredCount} structured · import
                </span>
              )}
              {unstructuredCount > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--primary)" }} />
                  {unstructuredCount} document{unstructuredCount !== 1 ? "s" : ""} · AI extraction
                </span>
              )}
            </div>
            <button className="btn ghost sm" onClick={clearFiles}>
              Clear
            </button>
          </div>

          {/* File list */}
          <div
            style={{
              maxHeight: 200,
              overflowY: "auto",
              borderRadius: "var(--radius-sm)",
              background: "var(--card-2)",
              border: "1px solid var(--border)",
            }}
          >
            {selectedFiles.map((file, i) => (
              <div
                key={`${file.name}-${i}`}
                className="between"
                style={{
                  padding: "10px 14px",
                  borderBottom:
                    i < selectedFiles.length - 1
                      ? "1px solid var(--border)"
                      : "none",
                }}
              >
                <span style={{ fontSize: 13.5, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {file.name}
                </span>
                <span className="muted mono" style={{ fontSize: 12, flexShrink: 0 }}>
                  {formatFileSize(file.size)}
                </span>
              </div>
            ))}
          </div>

          {/* Upload All button */}
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 16 }}>
            <button className="btn" onClick={handleUploadAll} disabled={uploading}>
              {uploading ? "Uploading…" : "Upload all"}
            </button>
          </div>
        </div>
      )}

      {/* ==========================================
          UPLOAD PROGRESS / RESULTS
          ========================================== */}
      {uploading && (
        <div className="card-surface pad">
          <span className="mono dim" style={{ fontSize: 13 }}>
            Uploading files…
          </span>
        </div>
      )}

      {uploadError && (
        <div className="card-surface pad">
          <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
            <span className="tag" style={{ color: "var(--danger)" }}>
              <span className="tdot" style={{ background: "var(--danger)" }} />
              Error
            </span>
            <p className="dim" style={{ fontSize: 13, margin: 0 }}>
              {uploadError}
            </p>
          </div>
        </div>
      )}

      {uploadResults.length > 0 && !uploading && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Upload complete</h3>
          </div>
          <div>
            {uploadResults.map((result, i) => (
              <div
                key={i}
                className="between"
                style={{
                  padding: "10px 0",
                  borderBottom:
                    i < uploadResults.length - 1
                      ? "1px solid var(--border)"
                      : "none",
                }}
              >
                <span style={{ fontSize: 13.5, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {result.filename}
                </span>
                {result.error ? (
                  <span className="tag" style={{ color: "var(--danger)", flexShrink: 0 }}>
                    <span className="tdot" style={{ background: "var(--danger)" }} />
                    {result.error}
                  </span>
                ) : result.type === "structured" &&
                  result.response &&
                  "records_inserted" in result.response ? (
                  <span className="tag" style={{ flexShrink: 0 }}>
                    <span className="tdot" style={{ background: "var(--success)" }} />
                    {(result.response as UploadResponse).records_inserted} records inserted
                  </span>
                ) : (
                  // Upload accepted; live extraction status is shown in the
                  // dedicated progress card below (this label is static, so
                  // don't imply ongoing progress here).
                  <span className="tag" style={{ flexShrink: 0 }}>
                    <span className="tdot" style={{ background: "var(--success)" }} />
                    Uploaded
                  </span>
                )}
              </div>
            ))}

            {/* Show structured upload errors if any */}
            {uploadResults
              .filter(
                (r) =>
                  r.type === "structured" &&
                  r.response &&
                  "errors" in r.response &&
                  Array.isArray((r.response as UploadResponse).errors) &&
                  (r.response as UploadResponse).errors.length > 0
              )
              .map((r, i) => (
                <div key={`errs-${i}`} style={{ marginTop: 12 }}>
                  <p className="field-l" style={{ color: "var(--danger)", marginBottom: 6 }}>
                    Errors in {r.filename}
                  </p>
                  <div
                    style={{
                      maxHeight: 140,
                      overflowY: "auto",
                      display: "flex",
                      flexDirection: "column",
                      gap: 4,
                    }}
                  >
                    {(r.response as UploadResponse).errors.map((err, j) => (
                      <p
                        key={j}
                        className="mono dim"
                        style={{
                          fontSize: 12,
                          padding: "6px 8px",
                          background: "var(--card-2)",
                          border: "1px solid var(--border)",
                          borderRadius: "var(--radius-sm)",
                          margin: 0,
                        }}
                      >
                        {formatUploadError(err)}
                      </p>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* ==========================================
          EXTRACTION PROGRESS / SUMMARY (scoped to the current batch)
          ========================================== */}
      {showProgressCard && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">
              {batch.allTerminal
                ? batch.failed > 0
                  ? "Extraction finished with errors"
                  : "Extraction complete"
                : "Extracting clinical entities"}
            </h3>
            <span className="muted mono" style={{ fontSize: 12 }}>
              {batch.recordsCreated} record{batch.recordsCreated === 1 ? "" : "s"} created
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Progress text */}
            <div className="between">
              <span
                className="mono"
                style={{
                  fontSize: 13,
                  color: batch.allTerminal
                    ? batch.failed > 0
                      ? "var(--danger)"
                      : "var(--success)"
                    : "var(--primary)",
                }}
              >
                {progress === null
                  ? "Preparing extraction…"
                  : `${batch.done} of ${batch.total} file${batch.total === 1 ? "" : "s"} processed`}
              </span>
            </div>

            {/* Progress bar */}
            <div className="bar">
              <i
                style={{
                  width: `${batch.percent}%`,
                  background:
                    batch.allTerminal && batch.failed === 0
                      ? "var(--success)"
                      : batch.failed > 0 && batch.allTerminal
                        ? "var(--danger)"
                        : "var(--primary)",
                }}
              />
            </div>

            {/* Status breakdown */}
            <div className="reasons">
              {batch.completed > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--success)" }} />
                  {batch.completed} completed
                </span>
              )}
              {batch.processing > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--primary)" }} />
                  {batch.processing} processing
                </span>
              )}
              {batch.failed > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--danger)" }} />
                  {batch.failed} failed
                </span>
              )}
              {batch.cancelled > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--text-muted)" }} />
                  {batch.cancelled} cancelled
                </span>
              )}
              {batch.pending > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--text-muted)" }} />
                  {batch.pending} pending
                </span>
              )}
            </div>

            {/* Per-file rows with live status + section progress + cancel */}
            {pipelineFiles.length > 0 && (
              <div className="tablewrap">
                <table className="rtable">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Status</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {pipelineFiles.map((f) => {
                      const isCancelling = cancelling.includes(f.upload_id);
                      const terminal = isTerminalStatus(f.status);
                      const stage = formatStage(f.progress_stage, f.progress_detail);
                      const shownStatus = isCancelling ? "cancelled" : f.status;
                      return (
                        <tr key={f.upload_id}>
                          <td className="desc">
                            <span
                              style={{
                                maxWidth: 280,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                                display: "inline-block",
                              }}
                              title={f.filename}
                            >
                              {f.filename}
                            </span>
                            {stage && !terminal && !isCancelling && (
                              <div className="muted mono" style={{ fontSize: 11, marginTop: 3 }}>
                                {stage}
                              </div>
                            )}
                          </td>
                          <td>
                            <span className="tag">
                              <span
                                className="tdot"
                                style={{ background: statusDotColor(shownStatus) }}
                              />
                              {isCancelling ? "cancelling…" : statusLabel(f.status)}
                            </span>
                          </td>
                          <td>
                            <div style={{ display: "flex", justifyContent: "flex-end" }}>
                              {!terminal && !isCancelling && (
                                <button
                                  className="row-del"
                                  title={`Cancel ${f.filename}`}
                                  aria-label={`Cancel ${f.filename}`}
                                  onClick={() => handleCancel([f.upload_id])}
                                >
                                  <X size={15} />
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Actions */}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              {inFlight.length > 0 && (
                <button
                  className="btn ghost sm"
                  onClick={() => handleCancel(inFlight.map((f) => f.upload_id))}
                >
                  Cancel all
                </button>
              )}
              {batch.allTerminal && (
                <button className="btn ghost sm" onClick={dismissBatch}>
                  Dismiss
                </button>
              )}
              {batch.allTerminal && batch.failed > 0 && (
                <button className="btn sm" onClick={handleRetryFailed}>
                  Retry {batch.failed} failed
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ==========================================
          EXTRACTION TRIGGER PANEL (ZIP children awaiting Extract)
          ========================================== */}
      {showTriggerPanel && (
        <div className="card-surface pad">
          <div style={{ marginBottom: 16 }}>
            <h3 className="sec-title">
              {triggerFiles.length} unstructured file
              {triggerFiles.length !== 1 ? "s" : ""} detected
            </h3>
            <p className="muted" style={{ fontSize: 12.5, margin: "4px 0 0" }}>
              Text extraction is required before clinical entities can be added to the record.
            </p>
          </div>

          <div className="tablewrap">
            <table className="rtable">
              <thead>
                <tr>
                  <th style={{ width: 32 }}>
                    <input
                      type="checkbox"
                      checked={allTriggerSelected}
                      onChange={toggleSelectAllExtraction}
                      style={{ accentColor: "var(--primary)" }}
                    />
                  </th>
                  <th>File</th>
                  <th>Format</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {triggerFiles.map((file) => {
                  const ext = file.filename.split(".").pop()?.toLowerCase() || "";
                  return (
                    <tr key={file.upload_id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedForExtraction.has(file.upload_id)}
                          onChange={() => toggleExtractionSelection(file.upload_id)}
                          style={{ accentColor: "var(--primary)" }}
                        />
                      </td>
                      <td className="desc">
                        <span
                          style={{
                            maxWidth: 280,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            display: "inline-block",
                          }}
                        >
                          {file.filename}
                        </span>
                      </td>
                      <td>
                        <span className="tag" style={{ textTransform: "uppercase" }}>
                          {ext}
                        </span>
                      </td>
                      <td>
                        <span className="tag">
                          <span
                            className="tdot"
                            style={{ background: statusDotColor(file.status) }}
                          />
                          {statusLabel(file.status)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div
            className="between"
            style={{
              padding: "14px 0 0",
              borderTop: "1px solid var(--border)",
              marginTop: 12,
            }}
          >
            <span className="muted" style={{ fontSize: 12.5 }}>
              {selectedForExtraction.size} of {triggerFiles.length} selected
            </span>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn ghost sm" onClick={dismissBatch}>
                Do later
              </button>
              <button
                className="btn sm"
                onClick={handleTriggerExtraction}
                disabled={selectedForExtraction.size === 0}
              >
                Extract {selectedForExtraction.size} file
                {selectedForExtraction.size !== 1 ? "s" : ""}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ==========================================
          UPLOAD HISTORY (COLLAPSIBLE)
          ========================================== */}
      <div className="card-surface">
        <button
          onClick={() => setHistoryOpen((prev) => !prev)}
          className="between"
          style={{
            width: "100%",
            padding: "16px 22px",
            cursor: "pointer",
            background: "transparent",
            border: 0,
            textAlign: "left",
          }}
        >
          <span className="sec-title">Upload history</span>
          {historyOpen ? (
            <ChevronUp size={16} style={{ color: "var(--text-muted)" }} />
          ) : (
            <ChevronDown size={16} style={{ color: "var(--text-muted)" }} />
          )}
        </button>

        {historyOpen && (
          <div style={{ padding: "0 22px 22px" }}>
            {deleteError && (
              <div
                className="tag"
                style={{ color: "var(--danger)", marginBottom: 12 }}
              >
                <span className="tdot" style={{ background: "var(--danger)" }} />
                {deleteError}
              </div>
            )}
            {historyLoading ? (
              <RetroLoadingState text="Loading upload history" />
            ) : history.length === 0 ? (
              <p className="muted" style={{ textAlign: "center", padding: "16px 0", fontSize: 13 }}>
                No uploads yet.
              </p>
            ) : (
              <div className="tablewrap">
                <table className="rtable">
                  <thead>
                    <tr>
                      <th>File</th>
                      <th>Format</th>
                      <th>Status</th>
                      <th>Records</th>
                      <th>Uploaded</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((upload) => {
                      const ext =
                        upload.filename.split(".").pop()?.toLowerCase() || "—";
                      return (
                        <tr key={upload.id}>
                          <td className="desc">
                            <span
                              style={{
                                maxWidth: 220,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                                display: "inline-block",
                              }}
                            >
                              {upload.filename}
                            </span>
                            {upload.ingestion_status === "duplicate_file" && (
                              <div className="muted" style={{ fontSize: 11.5, fontWeight: 400, marginTop: 3 }}>
                                Already added
                                {upload.ingestion_progress?.record_count != null
                                  ? ` · ${upload.ingestion_progress.record_count} records`
                                  : ""}{" "}
                                (duplicate of an earlier upload)
                              </div>
                            )}
                          </td>
                          <td>
                            <span className="tag" style={{ textTransform: "uppercase" }}>
                              {ext}
                            </span>
                          </td>
                          <td>
                            <span className="tag">
                              <span
                                className="tdot"
                                style={{ background: statusDotColor(upload.ingestion_status) }}
                              />
                              {statusLabel(upload.ingestion_status)}
                            </span>
                          </td>
                          <td className="num">{recordSummary(upload)}</td>
                          <td className="num">
                            {upload.created_at
                              ? new Date(upload.created_at).toLocaleDateString()
                              : "--"}
                          </td>
                          <td>
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "flex-end",
                                gap: 8,
                              }}
                            >
                              {(upload.ingestion_status === "pending_extraction" ||
                                upload.ingestion_status === "failed" ||
                                upload.ingestion_status === "processing") && (
                                <button
                                  className="btn ghost sm"
                                  onClick={(e: React.MouseEvent) => {
                                    e.stopPropagation();
                                    void handleHistoryExtract(upload);
                                  }}
                                >
                                  <RotateCcw size={13} />
                                  {upload.ingestion_status === "failed" ? "Retry" : "Extract"}
                                </button>
                              )}
                              <button
                                className="row-del"
                                title="Delete upload"
                                aria-label={`Delete ${upload.filename}`}
                                disabled={deletingId === upload.id}
                                style={
                                  deletingId === upload.id
                                    ? { opacity: 0.5, cursor: "wait" }
                                    : undefined
                                }
                                onClick={(e: React.MouseEvent) => {
                                  e.stopPropagation();
                                  setDeleteError(null);
                                  setDeleteTarget(upload);
                                }}
                              >
                                <Trash2 size={16} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ==========================================
          DELETE UPLOAD CONFIRMATION
          ========================================== */}
      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete upload?"
        description={
          deleteTarget
            ? `"${deleteTarget.filename}" and all records ingested from it will be soft-deleted. This cannot be undone from here.`
            : ""
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={() => {
          if (deleteTarget) void performDeleteUpload(deleteTarget.id);
        }}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
