"use client";

import type {
  LocalModelArtifact,
  ProcessingMode,
} from "@/types/local-ai";

export interface SummaryProviderOption {
  name: string;
  model: string;
  configured: boolean;
}

interface AiExecutionModeControlProps {
  mode: ProcessingMode;
  onModeChange: (mode: ProcessingMode) => void;
  providers: SummaryProviderOption[];
  provider: string;
  onProviderChange: (provider: string) => void;
  strictModel: LocalModelArtifact | null;
  strictAvailable: boolean;
  disabled?: boolean;
}

const LOCAL_PROVIDERS = new Set(["ollama", "lmstudio"]);

export function AiExecutionModeControl({
  mode,
  onModeChange,
  providers,
  provider,
  onProviderChange,
  strictModel,
  strictAvailable,
  disabled = false,
}: AiExecutionModeControlProps) {
  const customProviders = providers.filter(
    (item) => LOCAL_PROVIDERS.has(item.name) && item.configured
  );
  const cloudProviders = providers.filter(
    (item) => !LOCAL_PROVIDERS.has(item.name)
  );
  const selectableProviders =
    mode === "custom_local" ? customProviders : cloudProviders;

  return (
    <section className="card-surface pad" aria-labelledby="execution-mode-heading">
      <div className="card-h">
        <h2 id="execution-mode-heading" className="sec-title">
          AI execution
        </h2>
      </div>

      <label className="field-l" htmlFor="summary-execution-mode">
        Execution mode
      </label>
      <select
        id="summary-execution-mode"
        aria-label="AI execution mode"
        className="selectbox"
        style={{ width: "100%", marginTop: 8 }}
        value={mode}
        disabled={disabled}
        onChange={(event) =>
          onModeChange(event.target.value as ProcessingMode)
        }
      >
        <option value="validated_strict_local" disabled={!strictAvailable}>
          Validated strict local{strictAvailable ? "" : " (pack not ready)"}
        </option>
        <option value="custom_local">Custom local (unverified)</option>
        <option value="cloud_assisted">Cloud assisted</option>
        <option value="prompt_only">Prompt only</option>
      </select>

      {mode === "validated_strict_local" && (
        <div className="panel" style={{ marginTop: 12 }}>
          <p className="text-sm" style={{ margin: 0, color: "var(--text)" }}>
            Qwen3.5-9B summarizes validated facts locally. No provider route or
            cloud fallback is used.
          </p>
          {strictModel && (
            <p
              className="mono muted"
              style={{
                margin: "8px 0 0",
                fontSize: 11,
                overflowWrap: "anywhere",
              }}
            >
              {strictModel.repository} · {strictModel.revision}
            </p>
          )}
        </div>
      )}

      {(mode === "custom_local" || mode === "cloud_assisted") && (
        <div style={{ marginTop: 14 }}>
          <label className="field-l" htmlFor="summary-provider">
            AI provider
          </label>
          <select
            id="summary-provider"
            aria-label="Provider"
            className="selectbox"
            style={{ width: "100%", marginTop: 8 }}
            value={provider}
            disabled={disabled}
            onChange={(event) => onProviderChange(event.target.value)}
          >
            {selectableProviders.length === 0 && (
              <option value="">No configured provider</option>
            )}
            {selectableProviders.map((item) => (
              <option
                key={item.name}
                value={item.name}
                disabled={mode === "cloud_assisted" && !item.configured}
              >
                {item.name} · {item.model}
                {!item.configured ? " (no key)" : ""}
              </option>
            ))}
          </select>
          <p className="muted text-xs" style={{ marginTop: 8 }}>
            {mode === "custom_local"
              ? "Uses the selected loopback server. MedTimeline verifies loopback routing, but not the model or server."
              : `Automated scrubbing provides best-effort de-identification but cannot guarantee every identifier is removed. The resulting record content is sent to ${provider || "the selected cloud provider"}.`}
          </p>
        </div>
      )}

      {mode === "prompt_only" && (
        <div className="panel" style={{ marginTop: 12 }}>
          <p className="text-sm" style={{ margin: 0, color: "var(--text)" }}>
            MedTimeline builds a de-identified prompt for you to copy and use
            elsewhere. No health data was sent by MedTimeline.
          </p>
        </div>
      )}
    </section>
  );
}
