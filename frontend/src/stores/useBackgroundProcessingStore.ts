"use client";

import { create } from "zustand";

import {
  isActiveJob,
  isTerminalJob,
  mergeServerLifecycle,
  toBackgroundJobCard,
  type BackgroundJobMap,
} from "@/lib/background-processing";
import type { StrictLocalSummaryAccepted } from "@/types/api";
import type { LocalAIJobResponse } from "@/types/local-ai";

type PendingAction = "cancel" | "retry";

interface BackgroundProcessingState {
  jobs: BackgroundJobMap;
  hydrated: boolean;
  pendingActions: Record<string, PendingAction>;
  notifiedTerminalIds: Record<string, true>;
  hydrate: (jobs: LocalAIJobResponse[]) => void;
  applyServerJobs: (jobs: LocalAIJobResponse[]) => void;
  registerUploadLabels: (labels: Record<string, string>) => void;
  registerAcceptedSummary: (accepted: StrictLocalSummaryAccepted) => void;
  setPendingAction: (id: string, action: PendingAction | null) => void;
  markTerminalNotified: (id: string) => void;
  dismiss: (id: string) => void;
  reset: () => void;
}

const EMPTY = {
  jobs: {},
  hydrated: false,
  pendingActions: {},
  notifiedTerminalIds: {},
} satisfies Pick<
  BackgroundProcessingState,
  "jobs" | "hydrated" | "pendingActions" | "notifiedTerminalIds"
>;

export const useBackgroundProcessingStore =
  create<BackgroundProcessingState>((set) => ({
    ...EMPTY,
    hydrate: (jobs) =>
      set((state) => ({
        jobs: mergeServerLifecycle(state.jobs, jobs),
        hydrated: true,
        pendingActions: {},
      })),
    applyServerJobs: (jobs) =>
      set((state) => {
        const next = mergeServerLifecycle(state.jobs, jobs);
        const notifiedTerminalIds = { ...state.notifiedTerminalIds };
        for (const job of jobs) {
          const previous = state.jobs[job.id];
          if (
            previous &&
            isTerminalJob(previous.status) &&
            isActiveJob(job.status)
          ) {
            delete notifiedTerminalIds[job.id];
          }
        }
        return { jobs: next, notifiedTerminalIds };
      }),
    registerUploadLabels: (labels) =>
      set((state) => ({
        jobs: Object.fromEntries(
          Object.entries(state.jobs).map(([id, job]) => [
            id,
            {
              ...job,
              label:
                (job.upload_id ? labels[job.upload_id] : undefined) ??
                job.label,
            },
          ])
        ),
      })),
    registerAcceptedSummary: (accepted) =>
      set((state) => {
        const job = toBackgroundJobCard(
          {
            id: accepted.job_id,
            upload_id: null,
            summary_prompt_id: accepted.id,
            kind: accepted.kind,
            processing_mode: accepted.processing_mode,
            status: accepted.status,
            stage: accepted.stage,
            progress: null,
            failure: null,
            cancel_requested: false,
            created_at: accepted.created_at,
            updated_at: accepted.created_at,
            started_at: null,
            completed_at: null,
          },
          { label: "Summary" }
        );
        return { jobs: { ...state.jobs, [job.id]: job } };
      }),
    setPendingAction: (id, action) =>
      set((state) => {
        const pendingActions = { ...state.pendingActions };
        if (action === null) delete pendingActions[id];
        else pendingActions[id] = action;
        return { pendingActions };
      }),
    markTerminalNotified: (id) =>
      set((state) => ({
        notifiedTerminalIds: {
          ...state.notifiedTerminalIds,
          [id]: true,
        },
      })),
    dismiss: (id) =>
      set((state) => {
        const jobs = { ...state.jobs };
        delete jobs[id];
        return { jobs };
      }),
    reset: () => set({ ...EMPTY }),
  }));
