"use client";

import { Check, Cpu, Download, RefreshCw, RotateCcw, Trash2 } from "lucide-react";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { useLocalPackOperation } from "@/hooks/useLocalPackOperation";
import type {
  LocalModelArtifact,
  LocalPackOperation,
  PackState,
} from "@/types/local-ai";

const STATE_LABELS: Record<PackState, string> = {
  not_installed: "Optional download",
  downloading: "Downloading",
  verifying: "Verifying",
  preview: "Runtime-verified preview",
  ready: "Validated and ready",
  update_available: "Update available",
  failed: "Needs attention",
};

function bytes(value: number): string {
  return `${(value / 1024 ** 3).toFixed(value >= 1024 ** 3 ? 1 : 2)} GiB`;
}

function modelLabel(model: LocalModelArtifact): string {
  if (model.role === "ocr") return "OCR";
  if (model.role === "extraction") return "Grounded extraction";
  return "Summary";
}

function operationLabel(operation: LocalPackOperation): string {
  if (operation.message === "Running local validation fixtures.") {
    return "Verifying downloaded models";
  }
  if (operation.message === "Downloading verified model files.") {
    return operation.current_role
      ? `Downloading ${modelLabel({ role: operation.current_role } as LocalModelArtifact)} model`
      : "Downloading verified models";
  }
  return operation.message || "Preparing local model pack";
}

export function ValidatedLocalPackCard() {
  const {
    status,
    operation,
    loading,
    busy,
    pollingInterrupted,
    error,
    start,
    restart,
    retryPolling,
    remove,
  } = useLocalPackOperation();
  const percent =
    operation && operation.bytes_total > 0
      ? Math.min(100, Math.round((operation.bytes_done / operation.bytes_total) * 100))
      : 0;
  const artifactsMayBeInstalled =
    status !== null &&
    (status.state !== "not_installed" ||
      status.models.some((model) => model.installed));

  return (
    <div className="card-surface pad" style={{ marginBottom: 18 }}>
      <div className="card-h" style={{ alignItems: "flex-start" }}>
        <div>
          <p className="kicker" style={{ marginBottom: 4 }}>
            Optional, on-device AI
          </p>
          <h3 className="sec-title" style={{ marginBottom: 5 }}>
            Validated local pack
          </h3>
          <p className="muted" style={{ fontSize: 13, lineHeight: 1.55, margin: 0 }}>
            OvisOCR2 reads pages, NuExtract3 creates evidence-linked facts, and
            Qwen3.5-9B writes summaries. Processing fails locally if the verified
            pack is unavailable; it never falls back to a cloud provider.
          </p>
        </div>
        {status && (
          <span className="tag">
            {status.state === "ready" && <Check size={12} />}{" "}
            {STATE_LABELS[status.state]}
          </span>
        )}
      </div>

      {loading ? (
        <RetroLoadingState text="Checking local model pack" />
      ) : !status ? (
        <p className="muted" style={{ margin: "12px 0 0" }}>
          Local model pack status is unavailable.
        </p>
      ) : (
        <>
          <div
            className="field"
            style={{ display: "flex", justifyContent: "space-between", gap: 18 }}
          >
            <div>
              <div className="field-l">Compatibility</div>
              <div className="field-v" style={{ padding: "5px 0 0" }}>
                <Cpu size={14} style={{ marginRight: 6, verticalAlign: -2 }} />
                16 GB unified memory minimum · Apple silicon
              </div>
            </div>
            <span className="tag">
              {status.compatible ? "Compatible" : "Not compatible"}
            </span>
          </div>

          <div aria-label="Validated local models">
            {status.models.map((model) => (
              <div
                className="field"
                key={model.role}
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(120px, .55fr) minmax(220px, 1.45fr) auto",
                  gap: 14,
                  alignItems: "center",
                }}
              >
                <div>
                  <div className="field-l">{modelLabel(model)}</div>
                  <div className="field-v" style={{ padding: "4px 0 0" }}>
                    {model.quantization}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 13.5, fontWeight: 650 }}>
                    {model.repository}
                  </div>
                  <div
                    className="muted"
                    style={{ fontFamily: "var(--font-mono), monospace", fontSize: 11.5 }}
                  >
                    {model.revision.slice(0, 12)} · {model.runtime} · {model.license}
                  </div>
                </div>
                <div className="num" style={{ textAlign: "right", fontSize: 12 }}>
                  {bytes(model.download_bytes)}
                  <div className="muted" style={{ fontSize: 11 }}>
                    {model.expected_memory_bytes
                      ? `${bytes(model.expected_memory_bytes)} measured`
                      : "memory gate pending"}
                  </div>
                  <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                    {model.installed ? "Installed" : "Not installed"} ·{" "}
                    {status.state === "preview"
                      ? "Runtime verified"
                      : model.validated
                        ? "Validated"
                        : "Not validated"}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {status.state === "preview" && (
            <p className="muted" style={{ fontSize: 13, lineHeight: 1.55, margin: "12px 0 0" }}>
              {status.status_reason === "feature_disabled"
                ? "The pack is installed and verified, but strict-local processing is disabled. Set LOCAL_AI_ENABLED=true and restart Strand."
                : status.status_reason === "release_evidence_missing"
                  ? "This pack cannot process health records because its release evidence is missing or invalid."
                  : "This pack cannot process health records until its validation status is resolved."}
            </p>
          )}

          {operation && ["queued", "running", "paused", "failed"].includes(operation.state) && (
            <div
              aria-live="polite"
              style={{
                background: "var(--bg-2)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: 12,
                marginTop: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <span style={{ fontSize: 13.5, fontWeight: 650 }}>
                  {operationLabel(operation)}
                </span>
                <span className="num" style={{ fontSize: 12 }}>
                  {operation.bytes_total > 0 ? `${percent}%` : operation.state}
                </span>
              </div>
              {operation.bytes_total > 0 && (
                <div
                  aria-label="Model pack operation progress"
                  style={{
                    height: 5,
                    background: "var(--border)",
                    borderRadius: 999,
                    marginTop: 9,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${percent}%`,
                      height: "100%",
                      background: "var(--primary)",
                    }}
                  />
                </div>
              )}
            </div>
          )}

          {error && (
            <p role="alert" style={{ color: "var(--danger)", fontSize: 13 }}>
              {error}
            </p>
          )}

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
            {status.state === "not_installed" && (
              <button
                type="button"
                className="btn"
                disabled={!status.compatible || busy}
                onClick={() => void start("install")}
              >
                <Download size={14} /> Install local pack
              </button>
            )}
            {status.state === "ready" && (
              <>
                <button
                  type="button"
                  className="btn ghost sm"
                  disabled={busy}
                  onClick={() => void start("verify")}
                >
                  <RefreshCw size={14} /> Verify again
                </button>
              </>
            )}
            {status.state === "update_available" && (
              <>
                <button
                  type="button"
                  className="btn"
                  disabled={busy}
                  onClick={() => void start("update")}
                >
                  <Download size={14} /> Install verified update
                </button>
                <button
                  type="button"
                  className="btn ghost sm"
                  disabled={busy}
                  onClick={() => void start("rollback")}
                >
                  <RotateCcw size={14} /> Roll back
                </button>
              </>
            )}
            {operation?.state === "paused" && (
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() => void restart("resume")}
              >
                Resume operation
              </button>
            )}
            {operation?.state === "failed" && operation.retryable && (
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() => void restart("retry")}
              >
                Retry operation
              </button>
            )}
            {pollingInterrupted &&
              operation &&
              ["queued", "running"].includes(operation.state) && (
                <button
                  type="button"
                  className="btn"
                  disabled={busy}
                  onClick={retryPolling}
                >
                  Retry status check
                </button>
              )}
            {artifactsMayBeInstalled && !busy && (
              <button
                type="button"
                className="btn ghost sm"
                onClick={() => void remove()}
              >
                <Trash2 size={14} /> Remove pack
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
