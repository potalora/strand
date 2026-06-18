/**
 * Upload/extraction types kept OUT of the shared `types/api.ts` barrel so the
 * upload-UX work (session §2a) doesn't collide with concurrent edits there.
 * Re-export the existing aggregate shape for convenience.
 */
import type { ProgressDetail } from "@/lib/extraction-progress";

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
  progress_stage?: string | null;
  progress_detail?: ProgressDetail | null;
}
