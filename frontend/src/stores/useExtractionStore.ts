"use client";

import { create } from "zustand";
import type { ProgressDetail, RawProgress } from "@/lib/extraction-progress";
import { isTerminalStatus } from "@/lib/extraction-progress";

/**
 * Single source of truth for the CURRENT extraction batch.
 *
 * Both the upload page (inline panels) and the always-mounted
 * GlobalExtractionStatusBar (ambient, cross-page) render from this store, and
 * the status bar owns the one polling loop. Centralizing here is what makes the
 * UX correct per session §2a:
 *   (ii) `startBatch` REPLACES the prior batch, so a new upload never shows
 *        carried-over rows;
 *   (iii) progress is polled scoped to `batchIds`, never user-global;
 *   (i)  `dismissed` + the derived all-terminal state give a real done+dismiss.
 */

export interface TrackedFile {
  upload_id: string;
  filename: string;
  status: string;
  progress_stage?: string | null;
  progress_detail?: ProgressDetail | null;
  /** ZIP-extracted child — the worker does NOT auto-claim it; needs Extract. */
  needsTrigger: boolean;
  /** Extraction has been triggered for this (needsTrigger) file. */
  triggered: boolean;
}

export interface TrackedFileInput {
  upload_id: string;
  filename: string;
  status: string;
  needsTrigger?: boolean;
}

interface ExtractionState {
  batchIds: string[];
  files: Record<string, TrackedFile>;
  progress: RawProgress | null;
  dismissed: boolean;
  /** Upload IDs the user asked to cancel (optimistic, before the worker stops). */
  cancelling: string[];

  /** Reset to a brand-new batch (replaces any prior one). */
  startBatch: (files: TrackedFileInput[]) => void;
  setProgress: (p: RawProgress) => void;
  mergeFileStatuses: (
    rows: {
      id: string;
      ingestion_status: string;
      progress_stage?: string | null;
      progress_detail?: ProgressDetail | null;
    }[]
  ) => void;
  markTriggered: (ids: string[]) => void;
  markCancelling: (ids: string[]) => void;
  dismiss: () => void;
  reset: () => void;
}

function toTracked(f: TrackedFileInput): TrackedFile {
  return {
    upload_id: f.upload_id,
    filename: f.filename,
    status: f.status,
    progress_stage: null,
    progress_detail: null,
    needsTrigger: f.needsTrigger ?? false,
    triggered: false,
  };
}

const EMPTY: Pick<
  ExtractionState,
  "batchIds" | "files" | "progress" | "dismissed" | "cancelling"
> = {
  batchIds: [],
  files: {},
  progress: null,
  dismissed: false,
  cancelling: [],
};

export const useExtractionStore = create<ExtractionState>((set) => ({
  ...EMPTY,

  startBatch: (input) =>
    set((state) => {
      const priorInFlight = Object.values(state.files).some(
        (f) => !isTerminalStatus(f.status)
      );
      if (priorInFlight) {
        // A prior batch is still extracting — accumulate the new upload instead
        // of dropping it (concurrent uploads must all stay tracked). The poll
        // loop re-arms on the new batchIds and re-scopes progress to the union.
        const files = { ...state.files };
        const batchIds = [...state.batchIds];
        for (const f of input) {
          if (!files[f.upload_id]) batchIds.push(f.upload_id);
          files[f.upload_id] = toTracked(f);
        }
        return { batchIds, files, dismissed: false };
      }
      // Prior batch finished (or none) — fresh batch, dropping stale terminal
      // rows (the original anti-stale intent, §2a ii).
      const files: Record<string, TrackedFile> = {};
      for (const f of input) files[f.upload_id] = toTracked(f);
      return {
        batchIds: input.map((f) => f.upload_id),
        files,
        progress: null,
        dismissed: false,
        cancelling: [],
      };
    }),

  setProgress: (p) => set({ progress: p }),

  mergeFileStatuses: (rows) =>
    set((state) => {
      const files = { ...state.files };
      for (const row of rows) {
        const existing = files[row.id];
        if (!existing) continue; // only track files in the current batch
        files[row.id] = {
          ...existing,
          status: row.ingestion_status,
          progress_stage:
            row.progress_stage !== undefined
              ? row.progress_stage
              : existing.progress_stage,
          progress_detail:
            row.progress_detail !== undefined
              ? row.progress_detail
              : existing.progress_detail,
        };
      }
      return { files };
    }),

  markTriggered: (ids) =>
    set((state) => {
      const files = { ...state.files };
      for (const id of ids) {
        if (files[id]) files[id] = { ...files[id], triggered: true };
      }
      return { files };
    }),

  markCancelling: (ids) =>
    set((state) => ({
      cancelling: Array.from(new Set([...state.cancelling, ...ids])),
    })),

  dismiss: () => set({ dismissed: true }),

  reset: () => set({ ...EMPTY }),
}));

/** Map of upload_id → current status, for `deriveBatch`'s cancelled count. */
export function statusMapFromFiles(
  files: Record<string, TrackedFile>
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const id of Object.keys(files)) out[id] = files[id].status;
  return out;
}

/**
 * Should the polling loop run? Only when the batch has a file that will advance
 * on its own — i.e. non-terminal AND (not a ZIP child awaiting Extract, or one
 * that's already been triggered). Untriggered ZIP children sit idle until the
 * user clicks Extract, so we don't poll for them.
 */
export function batchIsPollable(files: Record<string, TrackedFile>): boolean {
  return Object.values(files).some(
    (f) => !isTerminalStatus(f.status) && (!f.needsTrigger || f.triggered)
  );
}
