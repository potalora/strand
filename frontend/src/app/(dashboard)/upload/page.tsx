"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
} from "lucide-react";
import { useDirectoryUpload } from "@/hooks/useDirectoryUpload";
import { getFilesFromDrop } from "@/lib/getFilesFromDrop";
import { api } from "@/lib/api";
import type {
  UploadResponse,
  UnstructuredUploadResponse,
  TriggerExtractionResponse,
  ExtractionProgressResponse,
} from "@/types/api";
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
    case "parsed":
      return "var(--success)";
    case "processing":
    case "pending_extraction":
    case "awaiting_confirmation":
    case "pending":
      return "var(--primary)";
    case "failed":
      return "var(--danger)";
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

  // --- Extraction trigger state (for unstructured files from ZIP uploads) ---
  const [pendingExtractions, setPendingExtractions] = useState<
    { upload_id: string; filename: string; status: string }[]
  >([]);
  const [selectedForExtraction, setSelectedForExtraction] = useState<Set<string>>(
    new Set()
  );
  const [extractionTriggered, setExtractionTriggered] = useState(false);
  const [extractionStatuses, setExtractionStatuses] = useState<
    Record<string, string>
  >({});

  // --- Extraction progress (replaces entity review) ---
  const [extractionProgress, setExtractionProgress] = useState<ExtractionProgressResponse | null>(null);
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  // --- Poll for extraction progress ---
  const startProgressPolling = useCallback(() => {
    // Clear any existing poll
    if (progressPollRef.current) clearInterval(progressPollRef.current);

    const poll = async () => {
      try {
        const data = await api.get<ExtractionProgressResponse>(
          "/upload/extraction-progress"
        );
        setExtractionProgress(data);

        // Stop polling when nothing is processing or pending
        if (data.processing === 0 && data.pending === 0) {
          if (progressPollRef.current) {
            clearInterval(progressPollRef.current);
            progressPollRef.current = null;
          }
          // Extraction finished — refresh upload history so rows show their
          // final status/record counts instead of the stale pending snapshot
          // captured right after upload.
          setHistoryLoaded(false);
        }
      } catch {
        /* continue polling */
      }
    };

    // Fetch immediately, then poll
    poll();
    progressPollRef.current = setInterval(poll, 2000);
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (progressPollRef.current) clearInterval(progressPollRef.current);
    };
  }, []);

  // --- Drop handler ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFiles((prev) => [...prev, ...acceptedFiles]);
      setUploadResults([]);
      setUploadError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
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
          setPendingExtractions((prev) => {
            const existing = new Set(prev.map((p) => p.upload_id));
            const newFiles = resp.unstructured_uploads!.filter(
              (f) => !existing.has(f.upload_id)
            );
            return [...prev, ...newFiles];
          });
        }
      } catch (err) {
        results.push({
          type: "structured",
          filename: file.name,
          error: err instanceof Error ? err.message : "Upload failed",
        });
      }
    }

    // Upload unstructured files
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

    // Start progress polling if any unstructured files were uploaded
    if (unstructured.length > 0) {
      startProgressPolling();
    }

    setUploadResults(results);
    setSelectedFiles([]);
    setUploading(false);
    setHistoryLoaded(false);
  }, [selectedFiles, startProgressPolling]);

  // --- Extraction trigger handlers ---
  const handleTriggerExtraction = useCallback(async () => {
    if (selectedForExtraction.size === 0) return;
    setExtractionTriggered(true);

    const ids = Array.from(selectedForExtraction);

    try {
      const resp = await api.post<TriggerExtractionResponse>(
        "/upload/trigger-extraction",
        { upload_ids: ids }
      );

      const statuses: Record<string, string> = {};
      for (const r of resp.results) {
        statuses[r.upload_id] = r.status;
      }
      setExtractionStatuses(statuses);

      // Start polling for progress
      startProgressPolling();
    } catch {
      setExtractionTriggered(false);
    }
  }, [selectedForExtraction, startProgressPolling]);

  const handleRetryFailed = useCallback(async () => {
    try {
      const resp = await api.get<{
        files: { id: string; filename: string }[];
      }>("/upload/pending-extraction?statuses=failed");

      if (resp.files.length === 0) return;

      const ids = resp.files.map((f) => f.id);
      await api.post<TriggerExtractionResponse>(
        "/upload/trigger-extraction",
        { upload_ids: ids }
      );
      startProgressPolling();
    } catch {
      /* silently fail */
    }
  }, [startProgressPolling]);

  const handleDismissExtractionPanel = useCallback(() => {
    setPendingExtractions([]);
    setSelectedForExtraction(new Set());
    setExtractionTriggered(false);
    setExtractionStatuses({});
  }, []);

  const toggleExtractionSelection = useCallback((uploadId: string) => {
    setSelectedForExtraction((prev) => {
      const next = new Set(prev);
      if (next.has(uploadId)) next.delete(uploadId);
      else next.add(uploadId);
      return next;
    });
  }, []);

  const toggleSelectAllExtraction = useCallback(() => {
    if (selectedForExtraction.size === pendingExtractions.length) {
      setSelectedForExtraction(new Set());
    } else {
      setSelectedForExtraction(
        new Set(pendingExtractions.map((p) => p.upload_id))
      );
    }
  }, [pendingExtractions, selectedForExtraction]);

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

  // --- Progress bar helpers ---
  const progressPercent = extractionProgress && extractionProgress.total > 0
    ? Math.round(((extractionProgress.completed + extractionProgress.failed) / extractionProgress.total) * 100)
    : 0;
  // Show the progress card whenever we have extraction data with files, and
  // keep it visible through completion (processing 0 → "Extraction complete")
  // until the user dismisses it (Dismiss nulls extractionProgress). Gating on
  // `processing > 0` alone made the card vanish on completion for directly
  // uploaded files, so the "Extraction complete · N records created" summary
  // was never shown.
  const showProgress = !!extractionProgress && extractionProgress.total > 0;
  const allDone = extractionProgress && extractionProgress.processing === 0 && extractionProgress.total > 0;

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
        style={{ cursor: "pointer" }}
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
              e.stopPropagation();
              const input = document.querySelector(
                'input[type="file"]:not([webkitdirectory])'
              ) as HTMLInputElement | null;
              if (input) {
                // Reset value so re-selecting the SAME file still fires a
                // change event (browsers suppress it when the value is
                // unchanged) — otherwise re-picking a just-uploaded file is
                // a silent no-op with no preview.
                input.value = "";
                input.click();
              }
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
          EXTRACTION PROGRESS BAR
          ========================================== */}
      {showProgress && extractionProgress && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">
              {extractionProgress.processing > 0
                ? "Extracting clinical entities"
                : "Extraction complete"}
            </h3>
            <span className="muted mono" style={{ fontSize: 12 }}>
              {extractionProgress.records_created} records created
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Progress text */}
            <div className="between">
              <span
                className="mono"
                style={{
                  fontSize: 13,
                  color:
                    extractionProgress.processing > 0
                      ? "var(--primary)"
                      : "var(--success)",
                }}
              >
                {extractionProgress.completed + extractionProgress.failed} of{" "}
                {extractionProgress.total} files processed
              </span>
            </div>

            {/* Progress bar */}
            <div className="bar">
              <i
                style={{
                  width: `${progressPercent}%`,
                  background:
                    allDone && extractionProgress.failed === 0
                      ? "var(--success)"
                      : "var(--primary)",
                }}
              />
            </div>

            {/* Status breakdown */}
            <div className="reasons">
              {extractionProgress.completed > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--success)" }} />
                  {extractionProgress.completed} completed
                </span>
              )}
              {extractionProgress.processing > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--primary)" }} />
                  {extractionProgress.processing} processing
                </span>
              )}
              {extractionProgress.failed > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--danger)" }} />
                  {extractionProgress.failed} failed
                </span>
              )}
              {extractionProgress.pending > 0 && (
                <span className="tag">
                  <span className="tdot" style={{ background: "var(--text-muted)" }} />
                  {extractionProgress.pending} pending
                </span>
              )}
            </div>

            {/* Retry button when there are failed files and nothing processing */}
            {allDone && extractionProgress.failed > 0 && (
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
                <button
                  className="btn ghost sm"
                  onClick={() => {
                    setExtractionProgress(null);
                    setExtractionTriggered(false);
                  }}
                >
                  Dismiss
                </button>
                <button className="btn sm" onClick={handleRetryFailed}>
                  Retry {extractionProgress.failed} failed
                </button>
              </div>
            )}

            {/* Dismiss when all done with no failures */}
            {allDone && extractionProgress.failed === 0 && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button
                  className="btn ghost sm"
                  onClick={() => {
                    setExtractionProgress(null);
                    setExtractionTriggered(false);
                  }}
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ==========================================
          EXTRACTION TRIGGER PANEL
          ========================================== */}
      {pendingExtractions.length > 0 && (
        <div className="card-surface pad">
          <div style={{ marginBottom: 16 }}>
            <h3 className="sec-title">
              {pendingExtractions.length} unstructured file
              {pendingExtractions.length !== 1 ? "s" : ""} detected
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
                      checked={
                        selectedForExtraction.size === pendingExtractions.length &&
                        pendingExtractions.length > 0
                      }
                      onChange={toggleSelectAllExtraction}
                      disabled={extractionTriggered}
                      style={{ accentColor: "var(--primary)" }}
                    />
                  </th>
                  <th>File</th>
                  <th>Format</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {pendingExtractions.map((file) => {
                  const ext = file.filename.split(".").pop()?.toLowerCase() || "";
                  const currentStatus =
                    extractionStatuses[file.upload_id] || file.status;
                  return (
                    <tr key={file.upload_id}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedForExtraction.has(file.upload_id)}
                          onChange={() => toggleExtractionSelection(file.upload_id)}
                          disabled={extractionTriggered}
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
                            style={{ background: statusDotColor(currentStatus) }}
                          />
                          {currentStatus}
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
              {selectedForExtraction.size} of {pendingExtractions.length} selected
            </span>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn ghost sm" onClick={handleDismissExtractionPanel}>
                Do later
              </button>
              <button
                className="btn sm"
                onClick={handleTriggerExtraction}
                disabled={selectedForExtraction.size === 0 || extractionTriggered}
              >
                {extractionTriggered
                  ? "Extracting…"
                  : `Extract ${selectedForExtraction.size} file${selectedForExtraction.size !== 1 ? "s" : ""}`}
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
                                  onClick={async (e: React.MouseEvent) => {
                                    e.stopPropagation();
                                    try {
                                      await api.post<TriggerExtractionResponse>(
                                        "/upload/trigger-extraction",
                                        { upload_ids: [upload.id] }
                                      );
                                      setHistoryLoaded(false);
                                    } catch {
                                      // Silently fail
                                    }
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
