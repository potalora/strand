/**
 * Pure helpers for the upload/extraction UX (session §2a i–iv).
 *
 * Dependency-free on purpose: the global status bar, the upload page, and the
 * server-free unit suite all import these, so they must not pull in React, the
 * API client, or app type barrels.
 */

/** Statuses at which a file is DONE — it will not advance on its own. */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "completed",
  "completed_with_merges",
  "completed_with_errors",
  "awaiting_confirmation",
  "awaiting_review",
  "failed",
  "cancelled",
  "duplicate_file",
]);

export function isTerminalStatus(status: string | null | undefined): boolean {
  return !!status && TERMINAL_STATUSES.has(status);
}

/** Section-level progress emitted per file while a long LLM extract runs. */
export interface ProgressDetail {
  section_index: number;
  section_total: number;
}

// Human labels for the worker's `progress_stage` values (contract §2a iv).
const STAGE_LABELS: Record<string, string> = {
  extracting_text: "Extracting text",
  scrubbing_phi: "De-identifying",
  extracting_entities: "Extracting entities",
  mapping_fhir: "Mapping to records",
};

function humanize(stage: string): string {
  const spaced = stage.replace(/_/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Render a stage + optional section detail as "Extracting entities — section 3
 * of 8". Returns null when there is no stage to show (older payloads omit it),
 * and collapses to the bare label when the section detail is absent or empty.
 */
export function formatStage(
  stage: string | null | undefined,
  detail: ProgressDetail | null | undefined
): string | null {
  if (!stage) return null;
  const label = STAGE_LABELS[stage] ?? humanize(stage);
  if (detail && detail.section_total > 0) {
    return `${label} — section ${detail.section_index} of ${detail.section_total}`;
  }
  return label;
}

/** Aggregate counts the progress endpoint returns (scoped to a batch of IDs). */
export interface RawProgress {
  total: number;
  completed: number;
  processing: number;
  failed: number;
  pending: number;
  records_created: number;
}

export interface BatchProgress {
  total: number;
  /** completed + failed + cancelled — everything no longer moving. */
  done: number;
  completed: number;
  processing: number;
  failed: number;
  pending: number;
  cancelled: number;
  recordsCreated: number;
  percent: number;
  allTerminal: boolean;
  anyActive: boolean;
}

const EMPTY_RAW: RawProgress = {
  total: 0,
  completed: 0,
  processing: 0,
  failed: 0,
  pending: 0,
  records_created: 0,
};

/**
 * Reduce the raw scoped-progress payload (+ per-file statuses) to the values the
 * UI renders. `done` is `total - processing - pending`, so cancelled files —
 * which the contract folds into "terminal/done" without a dedicated count —
 * advance the bar correctly. `cancelled` is recovered from the per-file map for
 * display only.
 */
export function deriveBatch(
  progress: RawProgress | null | undefined,
  fileStatuses: Record<string, string> = {}
): BatchProgress {
  const p = progress ?? EMPTY_RAW;
  const total = p.total;
  const processing = p.processing;
  const pending = p.pending;
  const done = Math.max(0, total - processing - pending);
  const cancelled = Object.values(fileStatuses).filter(
    (s) => s === "cancelled"
  ).length;
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  return {
    total,
    done,
    completed: p.completed,
    processing,
    failed: p.failed,
    pending,
    cancelled,
    recordsCreated: p.records_created,
    percent,
    allTerminal: total > 0 && processing === 0 && pending === 0,
    anyActive: processing > 0 || pending > 0,
  };
}
