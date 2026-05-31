"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { FolderUp, FileText, FileArchive, ChevronDown, ChevronUp } from "lucide-react";
import { useDirectoryUpload } from "@/hooks/useDirectoryUpload";
import { getFilesFromDrop } from "@/lib/getFilesFromDrop";
import { api } from "@/lib/api";
import type {
  UploadResponse,
  UnstructuredUploadResponse,
  TriggerExtractionResponse,
  ExtractionProgressResponse,
} from "@/types/api";
import { GlowText } from "@/components/retro/GlowText";
import { RetroCard, RetroCardHeader, RetroCardContent } from "@/components/retro/RetroCard";
import { RetroButton } from "@/components/retro/RetroButton";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import {
  RetroTable,
  RetroTableHeader,
  RetroTableHead,
  RetroTableBody,
  RetroTableRow,
  RetroTableCell,
} from "@/components/retro/RetroTable";

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
   UPLOAD HISTORY TYPES
   ========================================== */

interface UploadHistoryItem {
  id: string;
  original_filename: string;
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

/* ==========================================
   MAIN UPLOAD PAGE
   ========================================== */

export default function UploadPage() {
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

  // --- Load history when section is opened ---
  useEffect(() => {
    if (!historyOpen || historyLoaded) return;
    setHistoryLoading(true);
    api
      .get<{ items: UploadHistoryItem[]; total: number }>("/upload/history")
      .then((data) => {
        setHistory(data.items || []);
        setHistoryLoaded(true);
      })
      .catch(() => setHistory([]))
      .finally(() => setHistoryLoading(false));
  }, [historyOpen, historyLoaded]);

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
  const showProgress = extractionProgress && extractionProgress.total > 0 && (
    extractionProgress.processing > 0 || extractionTriggered
  );
  const allDone = extractionProgress && extractionProgress.processing === 0 && extractionProgress.total > 0;

  // --- Render ---
  return (
    <div className="space-y-6 retro-stagger">
      <GlowText as="h1">Upload</GlowText>

      {/* ==========================================
          HERO DROPZONE
          ========================================== */}
      <RetroCard accentTop>
        <RetroCardContent>
          <div
            {...getRootProps()}
            onDropCapture={handleDropCapture}
            className="border-2 border-dashed transition-all duration-200 cursor-pointer"
            style={{
              borderColor: isDragActive
                ? "var(--theme-amber)"
                : "var(--theme-border)",
              backgroundColor: isDragActive
                ? "var(--theme-bg-card-hover)"
                : "transparent",
              borderRadius: "6px",
              minHeight: "200px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "2rem",
            }}
          >
            <input {...getInputProps()} />
            <FolderUp
              size={48}
              style={{ color: "var(--theme-text-muted)", marginBottom: "1rem" }}
            />
            <p
              style={{
                color: "var(--theme-text)",
                fontFamily: "var(--font-body)",
                fontWeight: 600,
                fontSize: "1rem",
                marginBottom: "0.25rem",
              }}
            >
              Drop files or folders
            </p>
            <p
              style={{
                color: "var(--theme-text-dim)",
                fontFamily: "var(--font-body)",
                fontSize: "0.8rem",
              }}
            >
              JSON, ZIP, PDF, RTF, TIFF, or Epic export directories
            </p>
          </div>

          {/* Buttons below dropzone */}
          <div
            style={{
              display: "flex",
              gap: "0.75rem",
              justifyContent: "center",
              marginTop: "1rem",
            }}
          >
            <RetroButton
              variant="ghost"
              onClick={(e) => {
                e.stopPropagation();
                const input = document.querySelector(
                  'input[type="file"]:not([webkitdirectory])'
                ) as HTMLInputElement | null;
                input?.click();
              }}
            >
              <FileText size={14} style={{ marginRight: "0.5rem" }} />
              Select Files
            </RetroButton>
            <RetroButton
              onClick={(e) => {
                e.stopPropagation();
                selectFolder();
              }}
              disabled={isZipping}
            >
              <FileArchive size={14} style={{ marginRight: "0.5rem" }} />
              Select Folder
            </RetroButton>
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
        </RetroCardContent>
      </RetroCard>

      {/* ==========================================
          ZIPPING PROGRESS
          ========================================== */}
      {isZipping && folderInfo && (
        <RetroCard>
          <RetroCardContent>
            <span
              className="animate-pulse"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1rem",
                color: "var(--theme-amber)",
                display: "block",
                marginBottom: "0.5rem",
              }}
            >
              Preparing upload...
            </span>
            <p
              style={{
                fontSize: "0.8rem",
                color: "var(--theme-text-dim)",
                fontFamily: "var(--font-body)",
                marginBottom: "0.75rem",
              }}
            >
              {folderInfo.name} &mdash; {folderInfo.fileCount.toLocaleString()} files ({formatFileSize(folderInfo.totalSize)})
            </p>
            <div
              style={{
                height: "6px",
                backgroundColor: "var(--theme-bg-deep)",
                borderRadius: "3px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${zipProgress}%`,
                  backgroundColor: "var(--theme-amber)",
                  borderRadius: "3px",
                  transition: "width 0.2s ease",
                }}
              />
            </div>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.95rem",
                color: "var(--theme-amber)",
                display: "block",
                textAlign: "right",
                marginTop: "0.25rem",
              }}
            >
              {zipProgress.toFixed(0)}%
            </span>
          </RetroCardContent>
        </RetroCard>
      )}

      {/* ==========================================
          SELECTED FILES DISPLAY
          ========================================== */}
      {selectedFiles.length > 0 && (
        <RetroCard>
          <RetroCardContent>
            {/* Status bar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "1rem",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "1rem",
                    color: "var(--theme-amber)",
                  }}
                >
                  {selectedFiles.length} file{selectedFiles.length !== 1 ? "s" : ""}{" "}
                  selected
                </span>

                {structuredCount > 0 && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.25rem",
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      backgroundColor: "var(--theme-sage)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {structuredCount} structured &rarr; Import
                  </span>
                )}

                {unstructuredCount > 0 && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.25rem",
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      backgroundColor: "var(--theme-amber)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {unstructuredCount} document{unstructuredCount !== 1 ? "s" : ""}{" "}
                    &rarr; AI extraction
                  </span>
                )}
              </div>

              <RetroButton variant="ghost" onClick={clearFiles}>
                Clear
              </RetroButton>
            </div>

            {/* File list */}
            <div
              style={{
                maxHeight: "180px",
                overflowY: "auto",
                borderRadius: "4px",
                backgroundColor: "var(--theme-bg-deep)",
                padding: "0.5rem",
              }}
            >
              {selectedFiles.map((file, i) => (
                <div
                  key={`${file.name}-${i}`}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0.35rem 0.5rem",
                    borderBottom:
                      i < selectedFiles.length - 1
                        ? "1px solid var(--theme-border)"
                        : "none",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.95rem",
                      color: "var(--theme-text)",
                    }}
                  >
                    {file.name}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.85rem",
                      color: "var(--theme-text-muted)",
                    }}
                  >
                    {formatFileSize(file.size)}
                  </span>
                </div>
              ))}
            </div>

            {/* Upload All button */}
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "1rem" }}>
              <RetroButton onClick={handleUploadAll} disabled={uploading}>
                {uploading ? "Uploading..." : "Upload All"}
              </RetroButton>
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {/* ==========================================
          UPLOAD PROGRESS / RESULTS
          ========================================== */}
      {uploading && (
        <RetroCard>
          <RetroCardContent>
            <span
              className="animate-pulse"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1rem",
                color: "var(--theme-amber)",
              }}
            >
              Uploading files...
            </span>
          </RetroCardContent>
        </RetroCard>
      )}

      {uploadError && (
        <RetroCard>
          <RetroCardContent>
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.75rem" }}>
              <span
                style={{
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  padding: "0.15rem 0.5rem",
                  borderRadius: "4px",
                  backgroundColor: "var(--theme-terracotta)",
                  color: "var(--theme-text)",
                  flexShrink: 0,
                  fontFamily: "var(--font-body)",
                }}
              >
                ERROR
              </span>
              <p
                style={{
                  fontSize: "0.8rem",
                  color: "var(--theme-text-dim)",
                  fontFamily: "var(--font-body)",
                }}
              >
                {uploadError}
              </p>
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {uploadResults.length > 0 && !uploading && (
        <RetroCard accentTop>
          <RetroCardHeader>
            <GlowText as="h4" glow={false}>
              Upload complete
            </GlowText>
          </RetroCardHeader>
          <RetroCardContent>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              {uploadResults.map((result, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "0.5rem 0",
                    borderBottom:
                      i < uploadResults.length - 1
                        ? "1px solid var(--theme-border)"
                        : "none",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.95rem",
                      color: "var(--theme-text)",
                    }}
                  >
                    {result.filename}
                  </span>
                  {result.error ? (
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        color: "var(--theme-terracotta)",
                        fontFamily: "var(--font-body)",
                      }}
                    >
                      {result.error}
                    </span>
                  ) : result.type === "structured" &&
                    result.response &&
                    "records_inserted" in result.response ? (
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        color: "var(--theme-sage)",
                        fontFamily: "var(--font-body)",
                      }}
                    >
                      {(result.response as UploadResponse).records_inserted} records
                      inserted
                    </span>
                  ) : (
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        color: "var(--theme-amber)",
                        fontFamily: "var(--font-body)",
                      }}
                    >
                      Extracting...
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
                  <div key={`errs-${i}`} style={{ marginTop: "0.5rem" }}>
                    <p
                      style={{
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        color: "var(--theme-terracotta)",
                        fontFamily: "var(--font-body)",
                        marginBottom: "0.25rem",
                      }}
                    >
                      Errors in {r.filename}
                    </p>
                    <div
                      style={{
                        maxHeight: "120px",
                        overflowY: "auto",
                        display: "flex",
                        flexDirection: "column",
                        gap: "0.25rem",
                      }}
                    >
                      {(r.response as UploadResponse).errors.map((err, j) => (
                        <p
                          key={j}
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.85rem",
                            padding: "0.35rem 0.5rem",
                            backgroundColor: "var(--theme-bg-deep)",
                            color: "var(--theme-text-dim)",
                            borderRadius: "4px",
                          }}
                        >
                          {formatUploadError(err)}
                        </p>
                      ))}
                    </div>
                  </div>
                ))}
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {/* ==========================================
          EXTRACTION PROGRESS BAR
          ========================================== */}
      {showProgress && extractionProgress && (
        <RetroCard accentTop>
          <RetroCardHeader>
            <GlowText as="h4" glow={false}>
              {extractionProgress.processing > 0
                ? "Extracting clinical entities..."
                : "Extraction complete"}
            </GlowText>
          </RetroCardHeader>
          <RetroCardContent>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {/* Progress text */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "1rem",
                    color: extractionProgress.processing > 0
                      ? "var(--theme-amber)"
                      : "var(--theme-sage)",
                  }}
                  className={extractionProgress.processing > 0 ? "animate-pulse" : ""}
                >
                  {extractionProgress.completed + extractionProgress.failed} of {extractionProgress.total} files processed
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "1rem",
                    color: "var(--theme-sage)",
                  }}
                >
                  {extractionProgress.records_created} records created
                </span>
              </div>

              {/* Progress bar */}
              <div
                style={{
                  height: "8px",
                  backgroundColor: "var(--theme-bg-deep)",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${progressPercent}%`,
                    backgroundColor: allDone && extractionProgress.failed === 0
                      ? "var(--theme-sage)"
                      : "var(--theme-amber)",
                    borderRadius: "4px",
                    transition: "width 0.3s ease",
                  }}
                />
              </div>

              {/* Status breakdown */}
              <div
                style={{
                  display: "flex",
                  gap: "1rem",
                  flexWrap: "wrap",
                }}
              >
                {extractionProgress.completed > 0 && (
                  <span
                    style={{
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      backgroundColor: "var(--theme-sage)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {extractionProgress.completed} completed
                  </span>
                )}
                {extractionProgress.processing > 0 && (
                  <span
                    style={{
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      backgroundColor: "var(--theme-amber)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {extractionProgress.processing} processing
                  </span>
                )}
                {extractionProgress.failed > 0 && (
                  <span
                    style={{
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      backgroundColor: "var(--theme-terracotta)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {extractionProgress.failed} failed
                  </span>
                )}
                {extractionProgress.pending > 0 && (
                  <span
                    style={{
                      fontSize: "0.7rem",
                      fontFamily: "var(--font-body)",
                      fontWeight: 600,
                      padding: "0.15rem 0.5rem",
                      borderRadius: "4px",
                      backgroundColor: "var(--theme-text-muted)",
                      color: "var(--theme-bg-deep)",
                    }}
                  >
                    {extractionProgress.pending} pending
                  </span>
                )}
              </div>

              {/* Retry button when there are failed files and nothing processing */}
              {allDone && extractionProgress.failed > 0 && (
                <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>
                  <RetroButton
                    variant="ghost"
                    onClick={() => {
                      setExtractionProgress(null);
                      setExtractionTriggered(false);
                    }}
                  >
                    Dismiss
                  </RetroButton>
                  <RetroButton onClick={handleRetryFailed}>
                    Retry {extractionProgress.failed} Failed
                  </RetroButton>
                </div>
              )}

              {/* Dismiss when all done with no failures */}
              {allDone && extractionProgress.failed === 0 && (
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <RetroButton
                    variant="ghost"
                    onClick={() => {
                      setExtractionProgress(null);
                      setExtractionTriggered(false);
                    }}
                  >
                    Dismiss
                  </RetroButton>
                </div>
              )}
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {/* ==========================================
          EXTRACTION TRIGGER PANEL
          ========================================== */}
      {pendingExtractions.length > 0 && (
        <RetroCard>
          <RetroCardHeader>
            <GlowText as="h3" className="text-base">
              {pendingExtractions.length} Unstructured File
              {pendingExtractions.length !== 1 ? "s" : ""} Detected
            </GlowText>
            <p
              style={{
                fontSize: "0.7rem",
                color: "var(--theme-text-muted)",
                fontFamily: "var(--font-body)",
                marginTop: "0.25rem",
              }}
            >
              Text extraction required for clinical entity recognition
            </p>
          </RetroCardHeader>
          <RetroCardContent>
            <RetroTable>
              <RetroTableHeader>
                <RetroTableHead className="w-8">
                  <input
                    type="checkbox"
                    checked={
                      selectedForExtraction.size === pendingExtractions.length &&
                      pendingExtractions.length > 0
                    }
                    onChange={toggleSelectAllExtraction}
                    disabled={extractionTriggered}
                    style={{ accentColor: "var(--theme-amber)" }}
                  />
                </RetroTableHead>
                <RetroTableHead>File</RetroTableHead>
                <RetroTableHead>Type</RetroTableHead>
                <RetroTableHead>Status</RetroTableHead>
              </RetroTableHeader>
              <RetroTableBody>
                {pendingExtractions.map((file) => {
                  const ext = file.filename.split(".").pop()?.toLowerCase() || "";
                  const badgeColor =
                    ext === "pdf"
                      ? "var(--theme-terracotta)"
                      : ext === "rtf"
                        ? "var(--theme-sage)"
                        : "var(--theme-ochre)";
                  const currentStatus =
                    extractionStatuses[file.upload_id] || file.status;
                  return (
                    <RetroTableRow key={file.upload_id}>
                      <RetroTableCell>
                        <input
                          type="checkbox"
                          checked={selectedForExtraction.has(file.upload_id)}
                          onChange={() => toggleExtractionSelection(file.upload_id)}
                          disabled={extractionTriggered}
                          style={{ accentColor: "var(--theme-amber)" }}
                        />
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontSize: "0.75rem",
                            fontFamily: "var(--font-body)",
                            color: "var(--theme-text)",
                            maxWidth: "250px",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            display: "inline-block",
                          }}
                        >
                          {file.filename}
                        </span>
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontSize: "0.6rem",
                            fontWeight: 700,
                            padding: "0.1rem 0.4rem",
                            borderRadius: "3px",
                            backgroundColor: badgeColor,
                            color: "var(--theme-bg-deep)",
                            fontFamily: "var(--font-body)",
                            textTransform: "uppercase",
                          }}
                        >
                          {ext}
                        </span>
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontSize: "0.65rem",
                            fontWeight: 600,
                            padding: "0.15rem 0.5rem",
                            borderRadius: "4px",
                            fontFamily: "var(--font-body)",
                            backgroundColor:
                              currentStatus === "processing"
                                ? "var(--theme-amber)"
                                : currentStatus === "completed"
                                  ? "var(--theme-sage)"
                                  : currentStatus === "failed"
                                    ? "var(--theme-terracotta)"
                                    : "var(--theme-text-muted)",
                            color: "var(--theme-bg-deep)",
                          }}
                        >
                          {currentStatus}
                        </span>
                      </RetroTableCell>
                    </RetroTableRow>
                  );
                })}
              </RetroTableBody>
            </RetroTable>

            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0.75rem 0 0",
                borderTop: "1px solid var(--theme-border)",
                marginTop: "0.5rem",
              }}
            >
              <span
                style={{
                  fontSize: "0.7rem",
                  color: "var(--theme-text-muted)",
                  fontFamily: "var(--font-body)",
                }}
              >
                {selectedForExtraction.size} of {pendingExtractions.length} selected
              </span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <RetroButton
                  variant="ghost"
                  onClick={handleDismissExtractionPanel}
                >
                  Do Later
                </RetroButton>
                <RetroButton
                  onClick={handleTriggerExtraction}
                  disabled={
                    selectedForExtraction.size === 0 || extractionTriggered
                  }
                >
                  {extractionTriggered
                    ? "Extracting..."
                    : `Extract ${selectedForExtraction.size} File${selectedForExtraction.size !== 1 ? "s" : ""}`}
                </RetroButton>
              </div>
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {/* ==========================================
          UPLOAD HISTORY (COLLAPSIBLE)
          ========================================== */}
      <RetroCard>
        <div
          onClick={() => setHistoryOpen((prev) => !prev)}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0.75rem 1rem",
            cursor: "pointer",
          }}
        >
          <span
            style={{
              fontSize: "0.8rem",
              fontWeight: 600,
              color: "var(--theme-text-dim)",
              fontFamily: "var(--font-body)",
            }}
          >
            Upload history
          </span>
          {historyOpen ? (
            <ChevronUp size={16} style={{ color: "var(--theme-text-muted)" }} />
          ) : (
            <ChevronDown size={16} style={{ color: "var(--theme-text-muted)" }} />
          )}
        </div>

        {historyOpen && (
          <RetroCardContent>
            {historyLoading ? (
              <RetroLoadingState text="Loading upload history" />
            ) : history.length === 0 ? (
              <p
                style={{
                  textAlign: "center",
                  padding: "1rem 0",
                  fontSize: "0.75rem",
                  color: "var(--theme-text-muted)",
                  fontFamily: "var(--font-body)",
                }}
              >
                No uploads yet
              </p>
            ) : (
              <RetroTable>
                <RetroTableHeader>
                  <RetroTableHead>Date</RetroTableHead>
                  <RetroTableHead>Source</RetroTableHead>
                  <RetroTableHead>Records</RetroTableHead>
                  <RetroTableHead>Status</RetroTableHead>
                  <RetroTableHead>Actions</RetroTableHead>
                </RetroTableHeader>
                <RetroTableBody>
                  {history.map((upload) => (
                    <RetroTableRow key={upload.id}>
                      <RetroTableCell>
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.9rem",
                            color: "var(--theme-text-dim)",
                          }}
                        >
                          {upload.created_at
                            ? new Date(upload.created_at).toLocaleDateString()
                            : "--"}
                        </span>
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "var(--theme-text)",
                            fontFamily: "var(--font-body)",
                            maxWidth: "200px",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            display: "inline-block",
                          }}
                        >
                          {upload.original_filename}
                        </span>
                        {upload.ingestion_status === "duplicate_file" && (
                          <div style={{ fontSize: "0.65rem", color: "var(--theme-ochre)", fontFamily: "var(--font-body)", marginTop: "0.15rem" }}>
                            Already extracted{upload.ingestion_progress?.record_count != null ? ` · ${upload.ingestion_progress.record_count} records` : ""} (duplicate of an earlier upload)
                          </div>
                        )}
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.9rem",
                            color: "var(--theme-text-dim)",
                          }}
                        >
                          {recordSummary(upload)}
                        </span>
                      </RetroTableCell>
                      <RetroTableCell>
                        <span
                          style={{
                            fontSize: "0.65rem",
                            fontWeight: 600,
                            padding: "0.15rem 0.5rem",
                            borderRadius: "4px",
                            fontFamily: "var(--font-body)",
                            backgroundColor:
                              upload.ingestion_status === "completed"
                                ? "var(--theme-sage)"
                                : upload.ingestion_status === "processing"
                                  ? "var(--theme-amber)"
                                  : upload.ingestion_status === "failed"
                                    ? "var(--theme-terracotta)"
                                    : upload.ingestion_status ===
                                        "awaiting_confirmation"
                                      ? "var(--record-procedure-text)"
                                      : upload.ingestion_status ===
                                          "pending_extraction"
                                        ? "var(--theme-ochre)"
                                        : upload.ingestion_status ===
                                            "duplicate_file"
                                          ? "var(--theme-ochre)"
                                          : "var(--theme-text-muted)",
                            color: "var(--theme-bg-deep)",
                          }}
                        >
                          {statusLabel(upload.ingestion_status)}
                        </span>
                      </RetroTableCell>
                      <RetroTableCell>
                        {(upload.ingestion_status === "pending_extraction" ||
                          upload.ingestion_status === "failed" ||
                          upload.ingestion_status === "processing") && (
                          <RetroButton
                            variant="ghost"
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
                            style={{ fontSize: "0.65rem", padding: "0.15rem 0.4rem" }}
                          >
                            {upload.ingestion_status === "failed" ? "Retry" : "Extract"}
                          </RetroButton>
                        )}
                      </RetroTableCell>
                    </RetroTableRow>
                  ))}
                </RetroTableBody>
              </RetroTable>
            )}
          </RetroCardContent>
        )}
      </RetroCard>
    </div>
  );
}
