import type {
  LocalAIJobProgress,
  LocalAIJobResponse,
  LocalJobStatus,
} from "@/types/local-ai";

const TERMINAL_STATUSES = new Set<LocalJobStatus>([
  "completed",
  "failed",
  "cancelled",
]);

export interface BackgroundJobCard extends LocalAIJobResponse {
  label: string;
}

export type BackgroundJobMap = Record<string, BackgroundJobCard>;

export function isActiveJob(status: LocalJobStatus): boolean {
  return status === "queued" || status === "processing";
}

export function isTerminalJob(status: LocalJobStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function toBackgroundJobCard(
  job: LocalAIJobResponse,
  options: { label?: string } = {}
): BackgroundJobCard {
  return {
    ...job,
    label: options.label ?? (job.kind === "summary" ? "Summary" : "Document"),
  };
}

export function mergeServerLifecycle(
  existing: BackgroundJobMap,
  serverJobs: LocalAIJobResponse[]
): BackgroundJobMap {
  const next = { ...existing };
  for (const serverJob of serverJobs) {
    next[serverJob.id] = toBackgroundJobCard(serverJob, {
      label: existing[serverJob.id]?.label,
    });
  }
  return next;
}

export function terminalTransitions(
  previous: BackgroundJobMap,
  next: BackgroundJobMap,
  options: { initialHydration?: boolean } = {}
): string[] {
  if (options.initialHydration) return [];
  return Object.values(next)
    .filter(
      (job) =>
        isTerminalJob(job.status) &&
        previous[job.id] !== undefined &&
        isActiveJob(previous[job.id].status)
    )
    .map((job) => job.id);
}

function validRatio(current: unknown, total: unknown): number | null {
  if (
    typeof current !== "number" ||
    typeof total !== "number" ||
    !Number.isFinite(current) ||
    !Number.isFinite(total) ||
    current < 0 ||
    total <= 0 ||
    current > total
  ) {
    return null;
  }
  return current / total;
}

export function progressRatio(
  progress: LocalAIJobProgress | null
): number | null {
  if (!progress) return null;
  return (
    validRatio(progress.page_index, progress.page_total) ??
    validRatio(progress.worker_current, progress.worker_total)
  );
}

export function progressCounterLabel(
  progress: LocalAIJobProgress | null
): string | null {
  if (!progress) return null;
  if (
    validRatio(progress.page_index, progress.page_total) !== null
  ) {
    return `Page ${progress.page_index} of ${progress.page_total}`;
  }
  if (
    validRatio(progress.worker_current, progress.worker_total) !== null
  ) {
    return `Batch ${progress.worker_current} of ${progress.worker_total}`;
  }
  return null;
}

export function modelRoleLabel(
  progress: LocalAIJobProgress | null
): string | null {
  if (!progress?.model_role) return null;
  return `${formatBackgroundStage(progress.model_role)} model`;
}

export function updatedLabel(updatedAt: string): string {
  const timestamp = Date.parse(updatedAt);
  if (!Number.isFinite(timestamp)) return "Updated time unavailable";
  return `Updated ${new Date(timestamp).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  })}`;
}

export function formatBackgroundStage(stage: string): string {
  return stage
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function elapsedLabel(
  job: Pick<LocalAIJobResponse, "created_at" | "started_at" | "completed_at">,
  now = Date.now()
): string {
  const start = Date.parse(job.started_at ?? job.created_at);
  const end = job.completed_at ? Date.parse(job.completed_at) : now;
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "Elapsed time unavailable";
  const seconds = Math.max(0, Math.floor((end - start) / 1000));
  if (seconds < 60) return `${seconds}s elapsed`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s elapsed`;
}
