/**
 * Upload/extraction types kept OUT of the shared `types/api.ts` barrel so the
 * upload-UX work (session §2a) doesn't collide with concurrent edits there.
 * Re-export the existing aggregate shape for convenience.
 */
import type { ProgressDetail } from "@/lib/extraction-progress";
import type { OcrNotice } from "@/lib/api";
import type {
  LocalProcessingFailure,
  LocalRunInfo,
} from "@/types/local-ai";

export type { ExtractionProgressResponse } from "@/types/api";

/** POST /upload/cancel → which in-flight files were stopped vs. already done. */
export interface CancelExtractionResponse {
  cancelled: string[];
  skipped: string[];
}

/**
 * Per-file status row from GET /upload/pending-extraction. `progress_stage` and
 * `progress_detail` are optional — older payloads omit them, so every consumer
 * must render gracefully when absent.
 */
export interface ExtractionFileStatus {
  id: string;
  filename: string;
  ingestion_status: string;
  manual_extraction_required?: boolean;
  progress_stage?: string | null;
  progress_detail?: ProgressDetail | null;
  // Per-file OCR provider notices (fallback/unreadable). Default [] — older
  // payloads omit it, so consumers must treat missing as no notices.
  notices?: OcrNotice[];
  local_run?: LocalRunInfo | null;
  local_failure?: LocalProcessingFailure | null;
  local_job_id?: string | null;
}

/** Detailed GET /upload/{id}/status payload. */
export interface UploadStatusResponse {
  upload_id: string;
  filename: string;
  ingestion_status: string;
  record_count: number;
  total_file_count: number;
  ingestion_progress: Record<string, unknown>;
  ingestion_errors: unknown[];
  manual_extraction_required: boolean;
  processing_started_at: string | null;
  processing_completed_at: string | null;
  progress_stage: string | null;
  progress_detail: ProgressDetail | null;
  notices: OcrNotice[];
  local_run: LocalRunInfo | null;
  local_failure: LocalProcessingFailure | null;
  local_job_id: string | null;
}

/** Local provenance/failure fields are intentionally content-free. */
export interface UploadHistoryItem {
  id: string;
  filename: string;
  ingestion_status: string;
  record_count: number;
  file_size_bytes: number | null;
  created_at: string | null;
  ingestion_progress: Record<string, unknown>;
  ingestion_errors: unknown[];
  manual_extraction_required: boolean;
  local_run: LocalRunInfo | null;
  local_failure: LocalProcessingFailure | null;
  local_job_id: string | null;
}

export interface UploadHistoryResponse {
  items: UploadHistoryItem[];
  total: number;
}
