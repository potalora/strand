"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, ChevronUp, Loader2, TriangleAlert, X } from "lucide-react";
import { api } from "@/lib/api";
import { deriveBatch, formatStage, isTerminalStatus } from "@/lib/extraction-progress";
import {
  batchIsPollable,
  statusMapFromFiles,
  useExtractionStore,
} from "@/stores/useExtractionStore";
import { useUIStore } from "@/stores/useUIStore";

/* ==========================================================================
   GlobalExtractionStatusBar — ambient, cross-page extraction monitor.

   A sibling of the bottom-center FloatingDock (NOT an edit of it): a warm-paper
   capsule anchored bottom-right that surfaces whenever an extraction batch is in
   flight, hovers/expands to reveal per-file section progress, and offers Cancel.
   It owns the single polling loop, so progress keeps updating as the user moves
   between pages.

   Signature element: a determinate ring (the Reimagined "gauge" language) that
   breathes while extracting and settles to a check — the one bold mark; the rest
   stays quiet (mono counts, neutral tags, hairline borders).
   ========================================================================== */

// Statuses the per-file poll asks for (mirror of the upload page's set + cancelled).
const POLL_STATUSES = [
  "pending_extraction",
  "processing",
  "completed",
  "failed",
  "cancelled",
  "awaiting_confirmation",
  "awaiting_review",
  "completed_with_merges",
  "dedup_scanning",
];

function statusDotColor(status: string): string {
  switch (status) {
    case "completed":
    case "completed_with_merges":
    case "awaiting_confirmation":
    case "awaiting_review":
      return "var(--success)";
    case "processing":
    case "pending_extraction":
    case "pending":
    case "dedup_scanning":
      return "var(--primary)";
    case "failed":
      return "var(--danger)";
    case "cancelled":
      return "var(--text-muted)";
    default:
      return "var(--text-muted)";
  }
}

function ProgressRing({
  percent,
  tone,
  active,
}: {
  percent: number;
  tone: string;
  active: boolean;
}) {
  const r = 10;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - Math.max(0, Math.min(100, percent)) / 100);
  return (
    <span className="medtl-esb-ring" data-active={active ? "true" : "false"}>
      <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
        <circle cx="13" cy="13" r={r} fill="none" stroke="var(--border)" strokeWidth="3" />
        <circle
          cx="13"
          cy="13"
          r={r}
          fill="none"
          stroke={tone}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          transform="rotate(-90 13 13)"
          style={{ transition: "stroke-dashoffset 0.4s ease, stroke 0.4s ease" }}
        />
      </svg>
    </span>
  );
}

export function GlobalExtractionStatusBar() {
  const router = useRouter();
  const detailOpen = useUIStore((s) => s.detailOpen);

  const batchIds = useExtractionStore((s) => s.batchIds);
  const files = useExtractionStore((s) => s.files);
  const progress = useExtractionStore((s) => s.progress);
  const dismissed = useExtractionStore((s) => s.dismissed);
  const cancelling = useExtractionStore((s) => s.cancelling);
  const setProgress = useExtractionStore((s) => s.setProgress);
  const mergeFileStatuses = useExtractionStore((s) => s.mergeFileStatuses);
  const markCancelling = useExtractionStore((s) => s.markCancelling);
  const dismiss = useExtractionStore((s) => s.dismiss);

  const [expanded, setExpanded] = useState(false);

  const batchKey = batchIds.join(",");
  const pollable = batchIsPollable(files);

  // --- The one polling loop. `batchIds` identity changes only when a new batch
  //     starts (startBatch makes a fresh array), so the interval re-arms only on
  //     a batch change, dismissal, or a pollability flip — never every tick. ---
  useEffect(() => {
    if (batchIds.length === 0 || dismissed || !pollable) return;
    let active = true;

    const tick = async () => {
      try {
        const [prog, statusList] = await Promise.all([
          api.getExtractionProgress(batchIds),
          api.getExtractionFileStatuses(POLL_STATUSES),
        ]);
        if (!active) return;
        setProgress(prog);
        mergeFileStatuses(
          statusList.files.map((f) => ({
            id: f.id,
            ingestion_status: f.ingestion_status,
            progress_stage: f.progress_stage,
            progress_detail: f.progress_detail,
          }))
        );
      } catch {
        /* transient — keep polling */
      }
    };

    void tick();
    const timer = setInterval(tick, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [batchIds, dismissed, pollable, setProgress, mergeFileStatuses]);

  const fileList = batchIds.map((id) => files[id]).filter(Boolean);
  const batch = deriveBatch(progress, statusMapFromFiles(files));

  // In-flight = non-terminal files that haven't already been asked to cancel.
  const inFlight = fileList.filter(
    (f) => !isTerminalStatus(f.status) && !cancelling.includes(f.upload_id)
  );

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

  // Visible whenever a batch is alive (not dismissed) AND there is something to
  // show: real progress in hand, or a pollable batch just starting up.
  const visible = batchKey !== "" && !dismissed && (progress !== null || pollable);
  if (!visible) return null;

  const starting = progress === null;
  const tone = batch.failed > 0 && batch.allTerminal
    ? "var(--danger)"
    : batch.allTerminal
      ? "var(--success)"
      : "var(--primary)";

  // Collapsed headline.
  let headline: string;
  if (starting) headline = "Starting extraction";
  else if (!batch.allTerminal) headline = "Extracting documents";
  else if (batch.failed > 0) headline = "Extraction finished with errors";
  else if (batch.cancelled > 0 && batch.completed === 0) headline = "Extraction cancelled";
  else headline = "Extraction complete";

  const counts = starting
    ? "preparing…"
    : batch.allTerminal
      ? `${batch.recordsCreated} record${batch.recordsCreated === 1 ? "" : "s"} added`
      : `${batch.done} of ${batch.total}`;

  return (
    <>
      <StatusBarStyles />
      <section
        className="medtl-esb"
        data-hidden={detailOpen ? "true" : "false"}
        aria-hidden={detailOpen}
        aria-label="Extraction status"
        onMouseEnter={() => setExpanded(true)}
        onMouseLeave={() => setExpanded(false)}
      >
        {expanded && (
          <div className="medtl-esb-panel" role="region" aria-label="Extraction detail">
            <div className="medtl-esb-panel-h">
              <span className="medtl-esb-title">Extraction</span>
              <span className="mono medtl-esb-sub">
                {batch.allTerminal
                  ? `${batch.done} of ${batch.total} done`
                  : `${batch.done} of ${batch.total} files`}
              </span>
            </div>

            <div className="bar medtl-esb-bar">
              <i style={{ width: `${batch.percent}%`, background: tone }} />
            </div>

            <ul className="medtl-esb-files">
              {fileList.map((f) => {
                const isCancelling = cancelling.includes(f.upload_id);
                const stage = formatStage(f.progress_stage, f.progress_detail);
                const terminal = isTerminalStatus(f.status);
                return (
                  <li key={f.upload_id} className="medtl-esb-file">
                    <div className="medtl-esb-file-main">
                      <span className="medtl-esb-file-name" title={f.filename}>
                        {f.filename}
                      </span>
                      {!terminal && !isCancelling && (
                        <button
                          className="medtl-esb-x"
                          aria-label={`Cancel ${f.filename}`}
                          onClick={() => handleCancel([f.upload_id])}
                        >
                          <X size={13} />
                        </button>
                      )}
                    </div>
                    <div className="medtl-esb-file-meta">
                      <span className="tag medtl-esb-tag">
                        <span
                          className="tdot"
                          style={{ background: statusDotColor(isCancelling ? "cancelled" : f.status) }}
                        />
                        {isCancelling ? "cancelling…" : f.status}
                      </span>
                      {stage && !terminal && (
                        <span className="mono medtl-esb-stage">{stage}</span>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>

            <div className="medtl-esb-panel-f">
              {inFlight.length > 0 ? (
                <button
                  className="btn ghost sm"
                  onClick={() => handleCancel(inFlight.map((f) => f.upload_id))}
                >
                  Cancel all
                </button>
              ) : (
                <button className="btn ghost sm" onClick={() => router.push("/upload")}>
                  View uploads
                </button>
              )}
              {batch.allTerminal && (
                <button className="btn sm" onClick={dismiss}>
                  Dismiss
                </button>
              )}
            </div>
          </div>
        )}

        <button
          className="medtl-esb-pill"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          {starting || !batch.allTerminal ? (
            <ProgressRing percent={starting ? 0 : batch.percent} tone={tone} active={!batch.allTerminal} />
          ) : batch.failed > 0 ? (
            <span className="medtl-esb-icon" style={{ color: "var(--danger)" }}>
              <TriangleAlert size={16} />
            </span>
          ) : (
            <span className="medtl-esb-icon" style={{ color: "var(--success)" }}>
              <Check size={16} />
            </span>
          )}

          <span className="medtl-esb-pill-text">
            <span className="medtl-esb-headline">{headline}</span>
            <span className="mono medtl-esb-counts">{counts}</span>
          </span>

          {starting && (
            <Loader2 size={14} className="medtl-esb-spin" style={{ color: "var(--text-muted)" }} />
          )}
          <ChevronUp
            size={14}
            className="medtl-esb-chev"
            data-open={expanded ? "true" : "false"}
            style={{ color: "var(--text-muted)" }}
          />
        </button>
      </section>
    </>
  );
}

/* Component-scoped CSS, injected so globals.css stays untouched (Agent C owns
   the theme tokens). Namespaced `medtl-esb-*`; reuses theme variables only. */
function StatusBarStyles() {
  return (
    <style>{`
.medtl-esb {
  position: fixed; right: 24px; bottom: 24px; z-index: 45;
  display: flex; flex-direction: column; align-items: flex-end; gap: 10px;
  transition: opacity 0.28s ease, transform 0.28s ease;
}
.medtl-esb[data-hidden="true"] { opacity: 0; pointer-events: none; transform: translateY(14px); }
.medtl-esb-pill {
  appearance: none; cursor: pointer; display: inline-flex; align-items: center; gap: 11px;
  padding: 9px 14px 9px 11px; border-radius: 999px;
  background: var(--card); border: 1px solid var(--border-strong); box-shadow: var(--shadow-lg);
  color: var(--text); transition: transform 0.18s ease, border-color 0.18s ease;
}
.medtl-esb-pill:hover { transform: translateY(-1px); border-color: var(--primary); }
.medtl-esb-ring { display: inline-flex; }
.medtl-esb-ring[data-active="true"] svg { animation: medtl-esb-breathe 2.4s ease-in-out infinite; }
.medtl-esb-icon { display: inline-flex; }
.medtl-esb-pill-text { display: flex; flex-direction: column; align-items: flex-start; line-height: 1.2; }
.medtl-esb-headline { font-family: var(--font-body), sans-serif; font-weight: 700; font-size: 13px; color: var(--text); }
.medtl-esb-counts { font-size: 11px; color: var(--text-muted); letter-spacing: 0.02em; }
.medtl-esb-spin { animation: medtl-esb-spin 1s linear infinite; }
.medtl-esb-chev { transition: transform 0.2s ease; }
.medtl-esb-chev[data-open="true"] { transform: rotate(180deg); }

.medtl-esb-panel {
  width: 340px; max-width: calc(100vw - 32px);
  background: var(--card); border: 1px solid var(--border-strong);
  border-radius: var(--radius-lg); box-shadow: var(--shadow-lg);
  padding: 16px; display: flex; flex-direction: column; gap: 12px;
  animation: medtl-esb-rise 0.22s ease;
}
.medtl-esb-panel-h { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
.medtl-esb-title { font-family: var(--font-display), serif; font-style: italic; font-size: 16px; color: var(--text); }
.medtl-esb-sub { font-size: 11px; color: var(--text-muted); }
.medtl-esb-bar { height: 6px; }
.medtl-esb-files { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; max-height: 240px; overflow-y: auto; }
.medtl-esb-file { display: flex; flex-direction: column; gap: 4px; }
.medtl-esb-file-main { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.medtl-esb-file-name { font-size: 12.5px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.medtl-esb-file-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.medtl-esb-tag { font-size: 10.5px; }
.medtl-esb-stage { font-size: 11px; color: var(--text-dim); }
.medtl-esb-x {
  appearance: none; cursor: pointer; flex-shrink: 0; display: inline-flex; padding: 2px;
  background: transparent; border: 0; border-radius: 6px; color: var(--text-muted);
}
.medtl-esb-x:hover { background: var(--card-2); color: var(--danger); }
.medtl-esb-panel-f { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }

@keyframes medtl-esb-breathe { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
@keyframes medtl-esb-spin { to { transform: rotate(360deg); } }
@keyframes medtl-esb-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 560px) {
  .medtl-esb { right: 16px; bottom: 84px; }
  .medtl-esb-panel { width: calc(100vw - 32px); }
}
@media (prefers-reduced-motion: reduce) {
  .medtl-esb, .medtl-esb-pill, .medtl-esb-chev,
  .medtl-esb-ring[data-active="true"] svg, .medtl-esb-spin, .medtl-esb-panel { animation: none !important; transition: none !important; }
}
`}</style>
  );
}
