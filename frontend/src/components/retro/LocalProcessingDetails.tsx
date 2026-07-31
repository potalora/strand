"use client";

import {
  formatStage,
  type ProgressDetail,
} from "@/lib/extraction-progress";
import type {
  LocalProcessingFailure,
  LocalRunInfo,
} from "@/types/local-ai";

function modelName(repository: string): string {
  const leaf = repository.split("/").pop() ?? repository;
  if (/ovisocr2/i.test(leaf)) return "OvisOCR2";
  if (/nuextract3/i.test(leaf)) return "NuExtract3";
  return leaf;
}

function shortRevision(revision: string): string {
  return revision.length > 10 ? revision.slice(0, 10) : revision;
}

interface LocalProcessingDetailsProps {
  localRun?: LocalRunInfo | null;
  failure?: LocalProcessingFailure | null;
  progressDetail?: ProgressDetail | null;
  compact?: boolean;
}

export function LocalProcessingDetails({
  localRun,
  failure,
  progressDetail,
  compact = false,
}: LocalProcessingDetailsProps) {
  if (!localRun && !failure && !progressDetail?.repository) return null;

  const currentRepository =
    failure?.repository ?? progressDetail?.repository ?? null;
  const currentRevision =
    failure?.revision ?? progressDetail?.revision ?? null;
  const currentRole =
    failure?.model_role ?? progressDetail?.model_role ?? null;
  const currentModelIsSnapshotted =
    currentRepository !== null &&
    localRun?.models.some(
      (model) =>
        model.role === currentRole &&
        model.repository === currentRepository &&
        (!currentRevision || model.revision === currentRevision)
    );
  const failureStage = failure
    ? formatStage(failure.stage, null) ?? failure.stage
    : null;

  return (
    <div
      className={compact ? "mt-2 space-y-1" : "panel space-y-2"}
      aria-label="Local processing details"
    >
      {localRun?.privacy_mode === "validated_strict_local" && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="tag">Validated strict local</span>
          <span className="muted text-xs">Health data stays on this machine.</span>
        </div>
      )}

      {localRun && localRun.models.length > 0 && (
        <div aria-label="Models used for this local run" className="space-y-1">
          {localRun.models.map((model) => (
            <p
              className="mono text-xs"
              style={{ color: "var(--text-dim)", overflowWrap: "anywhere" }}
              key={`${model.role}:${model.repository}:${model.revision}`}
            >
              {model.role} · {model.repository} · {model.revision}
            </p>
          ))}
        </div>
      )}

      {currentRepository && !currentModelIsSnapshotted && (
        <p className="mono text-xs" style={{ color: "var(--text-dim)" }}>
          {currentRole ? `${currentRole} · ` : ""}
          <span title={currentRepository}>{modelName(currentRepository)}</span>
          {currentRevision ? ` · ${shortRevision(currentRevision)}` : ""}
        </p>
      )}

      {failure && (
        <div className="space-y-1">
          {failureStage && (
            <p className="mono text-xs" style={{ color: "var(--text-dim)" }}>
              Failure stage: {failureStage}
            </p>
          )}
          <p className="text-xs" style={{ color: "var(--text)" }}>
            {failure.message}
          </p>
          <div className="flex flex-wrap gap-2">
            <span className="tag">{failure.retryable ? "Retry available" : "Not retryable"}</span>
            {failure.checkpoint_preserved && (
              <span className="tag">Checkpoint preserved</span>
            )}
            {failure.cloud_fallback_attempted === false && (
              <span className="tag">Cloud fallback was not attempted</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
