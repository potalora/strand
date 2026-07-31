"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronUp,
  CircleX,
  Loader2,
  RotateCcw,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import {
  elapsedLabel,
  formatBackgroundStage,
  isActiveJob,
  mergeServerLifecycle,
  modelRoleLabel,
  progressCounterLabel,
  progressRatio,
  terminalTransitions,
  updatedLabel,
  type BackgroundJobCard,
} from "@/lib/background-processing";
import { useAuthStore } from "@/stores/useAuthStore";
import { useBackgroundProcessingStore } from "@/stores/useBackgroundProcessingStore";
import type { LocalAIJobResponse } from "@/types/local-ai";

let localJobPollTail: Promise<void> = Promise.resolve();

function pollJobsSequentially(ids: string[]): Promise<LocalAIJobResponse[]> {
  const request = localJobPollTail.then(async () => {
    const jobs: LocalAIJobResponse[] = [];
    for (const id of ids) jobs.push(await api.getLocalAIJob(id));
    return jobs;
  });
  localJobPollTail = request.then(
    () => undefined,
    () => undefined
  );
  return request;
}

function terminalToast(job: BackgroundJobCard): void {
  const subject = job.kind === "summary" ? "Summary" : job.label;
  if (job.status === "completed") toast.success(`${subject} finished.`);
  else if (job.status === "failed") toast.error(`${subject} could not be processed.`);
  else toast.info(`${subject} was cancelled.`);
}

function safeError(error: unknown): string {
  return error instanceof Error && error.message
    ? error.message
    : "The background action could not be completed.";
}

function notifyTerminalOnce(job: BackgroundJobCard): void {
  const store = useBackgroundProcessingStore.getState();
  if (store.notifiedTerminalIds[job.id]) return;
  store.markTerminalNotified(job.id);
  terminalToast(job);
}

function JobProgress({ job }: { job: BackgroundJobCard }) {
  const ratio = progressRatio(job.progress);
  return (
    <div
      className="medtl-bpm-progress"
      data-indeterminate={ratio === null && isActiveJob(job.status)}
      aria-label={
        ratio === null
          ? "Progress unavailable"
          : `${Math.round(ratio * 100)}% complete`
      }
    >
      <i style={ratio === null ? undefined : { width: `${ratio * 100}%` }} />
    </div>
  );
}

export function BackgroundProcessingMonitor() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const jobs = useBackgroundProcessingStore((state) => state.jobs);
  const hydrated = useBackgroundProcessingStore((state) => state.hydrated);
  const pendingActions = useBackgroundProcessingStore(
    (state) => state.pendingActions
  );
  const hydrate = useBackgroundProcessingStore((state) => state.hydrate);
  const applyServerJobs = useBackgroundProcessingStore(
    (state) => state.applyServerJobs
  );
  const setPendingAction = useBackgroundProcessingStore(
    (state) => state.setPendingAction
  );
  const dismiss = useBackgroundProcessingStore((state) => state.dismiss);
  const reset = useBackgroundProcessingStore((state) => state.reset);
  const registerUploadLabels = useBackgroundProcessingStore(
    (state) => state.registerUploadLabels
  );
  const generation = useRef(0);
  const [expanded, setExpanded] = useState(false);
  const [, setClock] = useState(0);

  const jobList = useMemo(
    () =>
      Object.values(jobs).sort((left, right) =>
        right.created_at.localeCompare(left.created_at)
      ),
    [jobs]
  );
  const activeIdsKey = jobList
    .filter((job) => isActiveJob(job.status))
    .map((job) => job.id)
    .sort()
    .join(",");

  useEffect(() => {
    const requestGeneration = ++generation.current;
    reset();
    if (!accessToken) return;
    let mounted = true;

    void api
      .getLocalAIJobs(true)
      .then((serverJobs) => {
        if (!mounted || generation.current !== requestGeneration) return;
        // Initial terminal state is intentionally silent.
        hydrate(serverJobs);
      })
      .catch(() => {
        if (!mounted || generation.current !== requestGeneration) return;
        hydrate([]);
      });
    void api
      .getUploadHistory()
      .then((uploadHistory) => {
        if (!mounted || generation.current !== requestGeneration) return;
        registerUploadLabels(
          Object.fromEntries(
            uploadHistory.items.map((item) => [item.id, item.filename])
          )
        );
      })
      .catch(() => {
        // Filenames are optional decoration; job hydration remains usable.
      });

    return () => {
      mounted = false;
      if (generation.current === requestGeneration) generation.current += 1;
      reset();
    };
  }, [accessToken, hydrate, registerUploadLabels, reset]);

  useEffect(() => {
    if (!accessToken || !hydrated || activeIdsKey === "") return;
    const requestGeneration = generation.current;
    let mounted = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const ids = activeIdsKey.split(",");

    const tick = async () => {
      try {
        // Deliberately sequential across jobs, ticks, and remounts. A slow
        // request must finish before a replacement monitor can start another.
        const serverJobs = await pollJobsSequentially(ids);
        if (!mounted || generation.current !== requestGeneration) return;
        const previous = useBackgroundProcessingStore.getState().jobs;
        const next = mergeServerLifecycle(previous, serverJobs);
        for (const id of terminalTransitions(previous, next)) {
          notifyTerminalOnce(next[id]);
        }
        applyServerJobs(serverJobs);
      } catch {
        // A transient read failure leaves the last authoritative state visible.
      } finally {
        if (mounted && generation.current === requestGeneration) {
          timer = setTimeout(() => void tick(), 2000);
        }
      }
    };

    void tick();
    return () => {
      mounted = false;
      if (timer !== null) clearTimeout(timer);
    };
  }, [accessToken, activeIdsKey, applyServerJobs, hydrated]);

  useEffect(() => {
    if (!jobList.some((job) => isActiveJob(job.status))) return;
    const timer = setInterval(() => setClock((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [jobList]);

  const runAction = async (
    job: BackgroundJobCard,
    action: "cancel" | "retry"
  ) => {
    const actionGeneration = generation.current;
    const actionToken = accessToken;
    setPendingAction(job.id, action);
    try {
      const response =
        action === "cancel"
          ? await api.cancelLocalAIJob(job.id)
          : await api.retryLocalAIJob(job.id);
      if (
        generation.current !== actionGeneration ||
        useAuthStore.getState().accessToken !== actionToken
      ) {
        return;
      }
      const previous = useBackgroundProcessingStore.getState().jobs;
      const next = mergeServerLifecycle(previous, [response]);
      for (const id of terminalTransitions(previous, next)) {
        notifyTerminalOnce(next[id]);
      }
      applyServerJobs([response]);
    } catch (error) {
      if (
        generation.current !== actionGeneration ||
        useAuthStore.getState().accessToken !== actionToken
      ) {
        return;
      }
      toast.error(safeError(error));
    } finally {
      if (
        generation.current === actionGeneration &&
        useAuthStore.getState().accessToken === actionToken
      ) {
        setPendingAction(job.id, null);
      }
    }
  };

  if (!hydrated || jobList.length === 0) return null;
  const activeCount = jobList.filter((job) => isActiveJob(job.status)).length;

  return (
    <>
      <BackgroundMonitorStyles />
      <section
        className="medtl-bpm"
        role="region"
        aria-label="Background processing"
      >
        {expanded && (
          <div className="medtl-bpm-panel">
            <header>
              <div>
                <strong>Background processing</strong>
                <p>Processing safely in the background. You can leave this page.</p>
              </div>
              <button
                className="medtl-bpm-icon"
                aria-label="Collapse background processing"
                onClick={() => setExpanded(false)}
              >
                <X size={15} />
              </button>
            </header>
            <ul>
              {jobList.map((job) => {
                const pending = pendingActions[job.id];
                const cancellable =
                  isActiveJob(job.status) && !job.cancel_requested;
                const retryable =
                  job.status === "failed" && job.failure?.retryable === true;
                const progressMeta = [
                  progressCounterLabel(job.progress),
                  modelRoleLabel(job.progress),
                  updatedLabel(job.updated_at),
                ]
                  .filter(Boolean)
                  .join(" · ");
                return (
                  <li key={job.id}>
                    <div className="medtl-bpm-row">
                      <div>
                        <strong>{job.label}</strong>
                        <span>
                          {formatBackgroundStage(job.stage)} · {elapsedLabel(job)}
                        </span>
                      </div>
                      <span className="tag">{job.status}</span>
                    </div>
                    <JobProgress job={job} />
                    <p className="medtl-bpm-meta">{progressMeta}</p>
                    <div className="medtl-bpm-actions">
                      {cancellable && (
                        <button
                          className="btn ghost sm"
                          disabled={pending !== undefined}
                          onClick={() => void runAction(job, "cancel")}
                        >
                          {pending === "cancel" ? "Cancelling…" : "Cancel"}
                        </button>
                      )}
                      {retryable && (
                        <button
                          className="btn ghost sm"
                          disabled={pending !== undefined}
                          onClick={() => void runAction(job, "retry")}
                        >
                          <RotateCcw size={13} />
                          {pending === "retry" ? "Retrying…" : "Retry"}
                        </button>
                      )}
                      {!isActiveJob(job.status) && (
                        <button
                          className="btn ghost sm"
                          onClick={() => dismiss(job.id)}
                        >
                          Dismiss
                        </button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        <button
          className="medtl-bpm-pill"
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          {activeCount > 0 ? (
            <Loader2 className="medtl-bpm-spin" size={16} />
          ) : jobList.some((job) => job.status === "failed") ? (
            <CircleX size={16} />
          ) : (
            <Check size={16} />
          )}
          <span>
            <strong>Background processing</strong>
            <small>
              {activeCount > 0
                ? `${activeCount} active · ${jobList[0].label}`
                : `${jobList.length} finished`}
            </small>
          </span>
          <ChevronUp data-open={expanded} size={15} />
        </button>
      </section>
    </>
  );
}

function BackgroundMonitorStyles() {
  return (
    <style>{`
      .medtl-bpm{position:fixed;right:22px;bottom:22px;z-index:60;width:min(390px,calc(100vw - 32px));color:var(--foreground)}
      .medtl-bpm-pill,.medtl-bpm-panel{border:1px solid var(--border);background:var(--background);box-shadow:0 12px 36px color-mix(in srgb,var(--foreground) 15%,transparent)}
      .medtl-bpm-pill{margin-left:auto;display:flex;align-items:center;gap:10px;border-radius:999px;padding:10px 14px;text-align:left}
      .medtl-bpm-pill span{display:grid}.medtl-bpm-pill strong{font-size:13px}.medtl-bpm-pill small{font:11px var(--font-mono);color:var(--text-muted)}
      .medtl-bpm-pill svg:last-child{transition:transform .2s}.medtl-bpm-pill svg[data-open="true"]{transform:rotate(180deg)}
      .medtl-bpm-panel{margin-bottom:8px;border-radius:16px;overflow:hidden}.medtl-bpm-panel header{display:flex;justify-content:space-between;gap:12px;padding:15px;border-bottom:1px solid var(--border)}
      .medtl-bpm-panel header p{margin:4px 0 0;font-size:12px;color:var(--text-muted)}.medtl-bpm-icon{border:0;background:none}
      .medtl-bpm-panel ul{list-style:none;padding:0;margin:0;max-height:330px;overflow:auto}.medtl-bpm-panel li{padding:14px 15px;border-bottom:1px solid var(--border)}
      .medtl-bpm-row{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.medtl-bpm-row>div{display:grid}.medtl-bpm-row strong{font-size:13px}.medtl-bpm-row span{font:11px var(--font-mono);color:var(--text-muted)}
      .medtl-bpm-progress{height:4px;margin-top:10px;overflow:hidden;border-radius:999px;background:var(--border)}.medtl-bpm-progress i{display:block;height:100%;background:var(--primary);transition:width .3s}
      .medtl-bpm-meta{margin:6px 0 0;font:10.5px var(--font-mono);color:var(--text-muted)}
      .medtl-bpm-progress[data-indeterminate="true"] i{width:35%;animation:medtl-bpm-slide 1.2s ease-in-out infinite}.medtl-bpm-actions{display:flex;justify-content:flex-end;gap:6px;margin-top:9px}
      .medtl-bpm-spin{animation:medtl-bpm-spin 1s linear infinite}@keyframes medtl-bpm-spin{to{transform:rotate(360deg)}}@keyframes medtl-bpm-slide{from{transform:translateX(-110%)}to{transform:translateX(300%)}}
    `}</style>
  );
}
