"use client";

import { useCallback, useEffect, useState } from "react";
import { Lock, Copy, ChevronRight } from "lucide-react";
import {
  api,
  getLlmSettings,
  saveRouting,
  type LlmRouting,
  type LlmSettings,
} from "@/lib/api";
import {
  completeProviderRouting,
  createSerializedRoutingSaver,
  isConfiguredCloudProvider as isCloudProvider,
  isCustomLocalProvider as isCustomProvider,
  isLoopbackUrl,
  selectProviderForMode,
} from "@/lib/llm-routing";
import type {
  GenerateSummaryRequest,
  PatientInfo,
  GenerateSummaryResponse,
  OutputFormat,
  PromptDetailResponse,
  PromptResponse,
} from "@/types/api";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import {
  AiExecutionModeControl,
  type SummaryProviderOption,
} from "@/components/retro/AiExecutionModeControl";
import type {
  LocalModelArtifact,
  ProcessingMode,
  SummaryModelProvenance,
} from "@/types/local-ai";

const SUMMARY_TYPES = [
  { key: "full", label: "Full record" },
  { key: "category", label: "By category" },
  { key: "date_range", label: "Date range" },
];

const CATEGORIES = [
  { value: "observation", label: "Labs & Vitals" },
  { value: "medication", label: "Medications" },
  { value: "condition", label: "Conditions" },
  { value: "encounter", label: "Encounters" },
  { value: "immunization", label: "Immunizations" },
  { value: "procedure", label: "Procedures" },
];

const OUTPUT_FORMATS: { value: OutputFormat; label: string }[] = [
  { value: "natural_language", label: "Natural language" },
  { value: "json", label: "JSON data" },
  { value: "both", label: "Both" },
];

const RESULT_TABS = [
  { key: "nl", label: "Narrative" },
  { key: "json", label: "JSON data" },
];

type ProviderInfo = SummaryProviderOption & {
  supports_vision: boolean;
  is_local: boolean;
  base_url: string | null;
  enabled: boolean;
};

function providersFromSettings(settings: LlmSettings): ProviderInfo[] {
  return settings.providers
    .filter(
      (item) =>
        item.enabled &&
        (!item.is_local ||
          (["ollama", "lmstudio"].includes(item.name) &&
            isLoopbackUrl(item.base_url)))
    )
    .map((item) => ({
      name: item.name,
      model: item.model || "model not configured",
      configured: item.configured && item.enabled,
      supports_vision: item.supports_vision,
      is_local: item.is_local,
      base_url: item.base_url,
      enabled: item.enabled,
    }));
}

function modeLabel(mode: ProcessingMode | null | undefined): string {
  switch (mode) {
    case "validated_strict_local":
      return "Validated strict local";
    case "custom_local":
      return "Custom local (unverified)";
    case "cloud_assisted":
      return "Cloud assisted";
    case "prompt_only":
      return "Prompt only";
    default:
      return "Execution mode unknown";
  }
}

function provenanceLabel(
  provenance: SummaryModelProvenance | null | undefined,
  mode: ProcessingMode | null | undefined
): string {
  if (!provenance || !mode || provenance.processing_mode !== mode) {
    return "Model identity unknown";
  }
  if (provenance.processing_mode === "validated_strict_local") {
    return provenance.model?.repository || "Model identity unknown";
  }
  return provenance.provider && provenance.model
    ? `${provenance.provider} · ${provenance.model}`
    : "Model identity unknown";
}

function privacyBoundary(mode: ProcessingMode | null | undefined): string {
  switch (mode) {
    case "validated_strict_local":
      return "Validated facts are summarized on this machine with no cloud fallback.";
    case "custom_local":
      return "The configured loopback server received record content; its model is not validated by MedTimeline.";
    case "cloud_assisted":
      return "Automated scrubbing provides best-effort de-identification but cannot guarantee every identifier was removed; the resulting record content was sent to the saved cloud provider.";
    case "prompt_only":
      return "No health data was sent by MedTimeline; you decide where to paste the generated prompt.";
    default:
      return "This saved summary does not include an execution-mode record.";
  }
}

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> Privacy boundary shown below
    </span>
  );
}

export default function SummariesPage() {
  // Entrance
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  // Patient selector
  const [patients, setPatients] = useState<PatientInfo[]>([]);
  const [selectedPatient, setSelectedPatient] = useState("");

  // AI provider selector
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [provider, setProvider] = useState<string>("");
  const [llmSettings, setLlmSettings] = useState<LlmSettings | null>(null);
  const [executionMode, setExecutionMode] =
    useState<ProcessingMode | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(true);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [strictModel, setStrictModel] =
    useState<LocalModelArtifact | null>(null);
  const [strictAvailable, setStrictAvailable] = useState(false);
  const [routingSaves] = useState(() =>
    createSerializedRoutingSaver(saveRouting)
  );

  // Config
  const [summaryType, setSummaryType] = useState("full");
  const [category, setCategory] = useState("observation");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("both");
  const [showCustomize, setShowCustomize] = useState(false);
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");

  // Results
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateSummaryResponse | null>(null);
  const [promptResult, setPromptResult] = useState<PromptResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultTab, setResultTab] = useState("nl");
  const [copied, setCopied] = useState(false);

  // History
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<PromptResponse[]>([]);

  // Load patients
  useEffect(() => {
    (async () => {
      try {
        const data = await api.get<{ items: PatientInfo[] }>(
          "/dashboard/patients"
        );
        setPatients(data.items);
        if (data.items.length > 0) setSelectedPatient(data.items[0].id);
      } catch {
        // ignore
      }
    })();
  }, []);

  const loadAiSettings = useCallback(async () => {
    setSettingsLoading(true);
    setSettingsError(null);
    setError(null);
    setExecutionMode(null);
    setLlmSettings(null);
    setProviders([]);
    setProvider("");
    try {
      const settings = await getLlmSettings();
      const mode = settings?.routing?.processing_mode;
      if (
        !settings?.routing ||
        !Array.isArray(settings.providers) ||
        !mode ||
        ![
          "validated_strict_local",
          "custom_local",
          "cloud_assisted",
          "prompt_only",
        ].includes(mode)
      ) {
        throw new Error("Invalid AI privacy settings response.");
      }
      const effectiveProviders = providersFromSettings(settings);
      const routedProvider =
        mode === "custom_local" || mode === "cloud_assisted"
          ? selectProviderForMode(effectiveProviders, settings.routing, mode)
          : null;
      routingSaves.reset();
      setLlmSettings(settings);
      setProviders(effectiveProviders);
      setExecutionMode(mode);
      setProvider(routedProvider?.name ?? "");
    } catch {
      setSettingsError(
        "AI privacy settings are unavailable. Summary generation is blocked until they load."
      );
    } finally {
      setSettingsLoading(false);
    }
  }, [routingSaves]);

  useEffect(() => {
    void loadAiSettings();
  }, [loadAiSettings]);

  useEffect(() => {
    let active = true;
    void api
      .getLocalPackStatus()
      .then((status) => {
        if (!active) return;
        const model = Array.isArray(status?.models)
          ? status.models.find((item) => item.role === "summary") ?? null
          : null;
        setStrictModel(model);
        setStrictAvailable(
          status.state === "ready" && model?.validated === true
        );
      })
      .catch(() => {
        if (!active) return;
        setStrictModel(null);
        setStrictAvailable(false);
      });
    return () => {
      active = false;
    };
  }, []);

  // Load history
  const loadHistory = useCallback(async () => {
    try {
      const data = await api.get<{ items: PromptResponse[] }>(
        "/summary/prompts"
      );
      setHistory(data.items);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Re-open a previously generated summary from history (no re-generation).
  const handleViewSaved = async (id: string) => {
    setError(null);
    try {
      const d = await api.get<PromptDetailResponse>(`/summary/prompts/${id}`);

      if (!d.response_text) {
        if (d.processing_mode === "prompt_only" && d.copyable_payload) {
          setPromptResult(d);
          setResult(null);
          window.scrollTo({ top: 0, behavior: "smooth" });
          return;
        }
        setError("This saved summary has no stored response text.");
        return;
      }

      const marker = "\n\n---JSON---\n";
      let naturalLanguage: string | null = null;
      let jsonData: Record<string, unknown> | null = null;

      if (d.typed_response) {
        if (d.response_format !== "json") {
          naturalLanguage = d.response_text.split(marker, 1)[0];
        }
        if (d.response_format === "json" || d.response_format === "both") {
          jsonData = { ...d.typed_response };
        }
      } else if (d.response_text.includes(marker)) {
        // Older rows predate typed_response and need the stored projection.
        const [nlPart, jsonPart] = d.response_text.split(marker);
        naturalLanguage = nlPart;
        try {
          jsonData = JSON.parse(jsonPart);
        } catch {
          /* leave jsonData null if it can't be parsed */
        }
      } else if (d.response_format === "json") {
        try {
          jsonData = JSON.parse(d.response_text);
        } catch {
          naturalLanguage = d.response_text;
        }
      } else {
        naturalLanguage = d.response_text;
      }

      setResult({
        id: d.id,
        processing_mode: d.processing_mode,
        model_provenance: d.model_provenance,
        natural_language: naturalLanguage,
        json_data: jsonData,
        record_count: d.record_count,
        duplicate_warning: null,
        de_identification_report: d.de_identification_report,
        model_used: "",
        generated_at: d.generated_at,
      });
      setPromptResult(null);
      setResultTab(naturalLanguage ? "nl" : "json");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load saved summary"
      );
    }
  };

  const handleGenerate = async () => {
    const mode = executionMode;
    if (
      !selectedPatient ||
      !mode ||
      settingsLoading ||
      settingsError
    ) {
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setPromptResult(null);

    try {
      await routingSaves.latest();
      const base = {
        patient_id: selectedPatient,
        summary_type: summaryType,
      };
      const scoped = {
        ...base,
        ...(summaryType === "category" ? { category } : {}),
        ...(summaryType === "date_range" && dateFrom
          ? { date_from: dateFrom }
          : {}),
        ...(summaryType === "date_range" && dateTo
          ? { date_to: dateTo }
          : {}),
      };

      if (mode === "prompt_only") {
        setPromptResult(
          await api.post<PromptResponse>("/summary/build-prompt", {
            ...scoped,
            output_format: outputFormat,
          })
        );
      } else {
        const customPrompt = customSystemPrompt.trim()
          ? { custom_system_prompt: customSystemPrompt.trim() }
          : {};
        let body: GenerateSummaryRequest;
        if (mode === "validated_strict_local") {
          body = {
            ...scoped,
            output_format: outputFormat,
            processing_mode: mode,
          };
        } else if (mode === "custom_local") {
          if (!provider || !providers.some((item) => item.name === provider && isCustomProvider(item))) {
            throw new Error(
              "Choose a configured loopback Ollama or LM Studio route before generating."
            );
          }
          body = {
            ...scoped,
            ...customPrompt,
            output_format: outputFormat,
            processing_mode: mode,
          };
        } else {
          body = {
            ...scoped,
            ...customPrompt,
            ...(provider ? { provider } : {}),
            output_format: outputFormat,
            processing_mode: mode,
          };
        }
        setResult(await api.generateSummary(body));
      }
      loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Summary generation failed");
    } finally {
      setLoading(false);
    }
  };

  const persistRouting = (
    body: Parameters<typeof saveRouting>[0],
    failureMessage: string
  ): Promise<void> => {
    const ticket = routingSaves.enqueue(body);
    void ticket.promise
      .then(() => {
        if (!ticket.isLatest()) return;
        setLlmSettings((current) =>
          current
            ? {
                ...current,
                routing: { ...current.routing, ...body } as LlmRouting,
              }
            : current
        );
      })
      .catch(() => {
        if (!ticket.isLatest()) return;
        setExecutionMode(null);
        setLlmSettings(null);
        setProviders([]);
        setProvider("");
        setSettingsError(
          "AI privacy settings are unavailable. Summary generation is blocked until they load."
        );
        setError(failureMessage);
      });
    return ticket.promise;
  };

  const handleModeChange = (mode: ProcessingMode) => {
    if (loading) return;
    setExecutionMode(mode);
    setError(null);
    setResult(null);
    setPromptResult(null);
    if (mode === "custom_local" || mode === "cloud_assisted") {
      const next = llmSettings
        ? selectProviderForMode(providers, llmSettings.routing, mode)
        : null;
      setProvider(next?.name ?? "");
      if (!next || !llmSettings) {
        setError(
          mode === "custom_local"
            ? "Configure a loopback Ollama or LM Studio provider before using custom local."
            : "Configure and enable a cloud provider before using cloud assisted."
        );
        return;
      }
      const body = completeProviderRouting(
        llmSettings.routing,
        next.name,
        mode
      );
      void persistRouting(
        body,
        mode === "custom_local"
          ? "Could not save the selected loopback processing routes."
          : "Could not save the selected cloud processing routes."
      ).catch(() => {
        // The visible error is set by persistRouting for the newest save.
      });
    } else {
      void persistRouting(
        { processing_mode: mode },
        "Could not save the selected processing mode."
      ).catch(() => {
        // The visible error is set by persistRouting for the newest save.
      });
    }
  };

  const handleProviderChange = (next: string) => {
    if (loading) return;
    setProvider(next);
    setError(null);
    if (
      (executionMode === "custom_local" ||
        executionMode === "cloud_assisted") &&
      next
    ) {
      const selected = providers.find(
        (item) =>
          item.name === next &&
          (executionMode === "custom_local"
            ? isCustomProvider(item)
            : isCloudProvider(item))
      );
      if (!selected || !llmSettings) {
        setError(
          executionMode === "custom_local"
            ? "The selected custom provider is not a valid loopback route."
            : "The selected cloud provider is not configured and enabled."
        );
        return;
      }
      const body = completeProviderRouting(
        llmSettings.routing,
        next,
        executionMode
      );
      void persistRouting(
        body,
        executionMode === "custom_local"
          ? "Could not save the selected loopback processing routes."
          : "Could not save the selected cloud processing routes."
      ).catch(() => {
        // The visible error is set by persistRouting for the newest save.
      });
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className={`screen s24 ${shown ? "on" : ""}`}>
      {/* Header */}
      <div className="page-top">
        <div>
          <p className="kicker">Privacy controls</p>
          <h1 className="h1 display">Summaries</h1>
          <p className="h-sub">
            Choose exactly where summarization runs. Strict local uses the
            validated on-device pack; custom local uses your loopback server;
            cloud assisted applies best-effort automated scrubbing before
            sending content; prompt only makes a copyable payload.
          </p>
        </div>
        <SecureChip />
      </div>

      {/* Patient selector */}
      <div className="card-surface pad">
        <div className="field-l" style={{ marginBottom: 8 }}>
          Record subject
        </div>
        <select
          className="selectbox"
          style={{ width: "100%" }}
          value={selectedPatient}
          onChange={(e) => setSelectedPatient(e.target.value)}
        >
          {patients.length === 0 && <option value="">No record found</option>}
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.fhir_id || p.id.slice(0, 8)} ({p.gender || "unknown"})
            </option>
          ))}
        </select>
      </div>

      {settingsLoading ? (
        <div className="card-surface pad">
          <RetroLoadingState text="Loading AI privacy settings" />
        </div>
      ) : settingsError ? (
        <div className="card-surface pad" role="alert">
          <p className="dim" style={{ fontSize: 13, lineHeight: 1.5, margin: 0 }}>
            {settingsError}
          </p>
          <button
            type="button"
            className="btn ghost sm"
            style={{ marginTop: 12 }}
            onClick={() => void loadAiSettings()}
          >
            Retry privacy settings
          </button>
        </div>
      ) : executionMode ? (
        <AiExecutionModeControl
          mode={executionMode}
          onModeChange={handleModeChange}
          providers={providers}
          provider={provider}
          onProviderChange={handleProviderChange}
          strictModel={strictModel}
          strictAvailable={strictAvailable}
          disabled={loading}
        />
      ) : null}

      {/* Duplicate notice */}
      {result?.duplicate_warning &&
        result.duplicate_warning.duplicates_excluded > 0 && (
          <div className="card-surface pad">
            <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
              <span className="tag" style={{ flexShrink: 0 }}>
                <span
                  className="tdot"
                  style={{ background: "var(--theme-ochre)" }}
                />
                Deduped
              </span>
              <p className="dim" style={{ fontSize: 13, lineHeight: 1.5, margin: 0 }}>
                {result.duplicate_warning.message} Review in Admin &gt; Dedup tab.
              </p>
            </div>
          </div>
        )}

      {/* Configuration */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Configuration</h3>
        </div>

        {/* Summary type */}
        <div className="field-l" style={{ marginBottom: 10 }}>
          What to summarize
        </div>
        <div className="tabs" style={{ marginBottom: 20 }}>
          {SUMMARY_TYPES.map((t) => (
            <button
              key={t.key}
              type="button"
              className="tab"
              aria-pressed={summaryType === t.key}
              onClick={() => setSummaryType(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Category (conditional) */}
        {summaryType === "category" && (
          <div style={{ marginBottom: 20 }}>
            <div className="field-l" style={{ marginBottom: 8 }}>
              Category
            </div>
            <select
              className="selectbox"
              style={{ width: "100%" }}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Date range (conditional) */}
        {summaryType === "date_range" && (
          <div className="grid-2" style={{ marginBottom: 20 }}>
            <div>
              <div className="field-l" style={{ marginBottom: 8 }}>
                From
              </div>
              <input
                type="date"
                className="selectbox"
                style={{ width: "100%" }}
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
              />
            </div>
            <div>
              <div className="field-l" style={{ marginBottom: 8 }}>
                To
              </div>
              <input
                type="date"
                className="selectbox"
                style={{ width: "100%" }}
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="field-l" style={{ marginBottom: 10 }}>
          Output format
        </div>
        <div className="filters" style={{ marginBottom: 18 }}>
          {OUTPUT_FORMATS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              className="filt"
              aria-pressed={outputFormat === opt.value}
              onClick={() => setOutputFormat(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Customize prompt (expandable) */}
        {(executionMode === "custom_local" ||
          executionMode === "cloud_assisted") && (
          <>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => setShowCustomize(!showCustomize)}
            >
              {showCustomize ? "Hide prompt options" : "Customize prompt"}
            </button>
            {showCustomize && (
              <textarea
                value={customSystemPrompt}
                onChange={(e) => setCustomSystemPrompt(e.target.value)}
                maxLength={4096}
                placeholder="Add instructions; server safety rules remain in force."
                rows={6}
                className="search"
                style={{
                  display: "block",
                  width: "100%",
                  marginTop: 12,
                  fontFamily: "var(--font-mono), monospace",
                  fontSize: 13,
                  lineHeight: 1.5,
                  resize: "vertical",
                }}
              />
            )}
          </>
        )}
      </div>

      {/* Generate */}
      <div style={{ display: "flex", justifyContent: "center" }}>
        <button
          className="btn"
          onClick={handleGenerate}
          disabled={
            loading ||
            settingsLoading ||
            settingsError !== null ||
            executionMode === null ||
            !selectedPatient ||
            (executionMode === "validated_strict_local" && !strictAvailable) ||
            (executionMode === "custom_local" &&
              !providers.some(
                (item) =>
                  item.name === provider && isCustomProvider(item)
              )) ||
            (executionMode === "cloud_assisted" &&
              !providers.some(
                (item) =>
                  item.name === provider &&
                  isCloudProvider(item) &&
                  item.configured
              ))
          }
        >
          {loading
            ? executionMode === "prompt_only"
              ? "Building…"
              : "Generating…"
            : executionMode === "prompt_only"
              ? "Build prompt"
              : "Generate summary"}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <RetroLoadingState
          text={
            executionMode === "prompt_only"
              ? "Building prompt"
              : "Generating summary"
          }
        />
      )}

      {/* Error */}
      {error && (
        <div className="card-surface pad">
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <span className="tag" style={{ flexShrink: 0 }}>
              <span
                className="tdot"
                style={{ background: "var(--danger)" }}
              />
              Error
            </span>
            <p className="dim" style={{ fontSize: 13, lineHeight: 1.5, margin: 0 }}>
              {error}
            </p>
          </div>
        </div>
      )}

      {promptResult && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Prompt payload</h3>
            <button
              type="button"
              className="btn ghost sm"
              aria-label="Copy prompt payload"
              onClick={() => copyToClipboard(promptResult.copyable_payload)}
            >
              <Copy size={13} />{" "}
              {copied ? "Copied" : "Copy prompt payload"}
            </button>
          </div>
          <pre
            className="panel mono"
            style={{
              fontSize: 12.5,
              lineHeight: 1.5,
              maxHeight: 600,
              overflow: "auto",
              whiteSpace: "pre-wrap",
            }}
          >
            {promptResult.copyable_payload}
          </pre>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Summary</h3>
            <span className="muted mono" style={{ fontSize: 11 }}>
              {result.record_count} record{result.record_count === 1 ? "" : "s"}
              {` · ${modeLabel(result.processing_mode)}`}
              {` · ${provenanceLabel(
                result.model_provenance,
                result.processing_mode
              )}`}
            </span>
          </div>

          {/* Output tabs */}
          {(result.natural_language || result.json_data) && (
            <div className="tabs">
              {RESULT_TABS.map((t) => {
                const disabled =
                  (t.key === "nl" && !result.natural_language) ||
                  (t.key === "json" && !result.json_data);
                if (disabled) return null;
                return (
                  <button
                    key={t.key}
                    type="button"
                    className="tab"
                    aria-pressed={resultTab === t.key}
                    onClick={() => setResultTab(t.key)}
                  >
                    {t.label}
                  </button>
                );
              })}
            </div>
          )}

          {/* Narrative tab */}
          {resultTab === "nl" && result.natural_language && (
            <div
              className="panel"
              style={{
                whiteSpace: "pre-wrap",
                fontSize: 14.5,
                lineHeight: 1.6,
                color: "var(--text-dim)",
                maxHeight: 600,
                overflow: "auto",
              }}
            >
              {result.natural_language}
            </div>
          )}

          {/* JSON tab */}
          {resultTab === "json" && result.json_data && (
            <div style={{ position: "relative" }}>
              <button
                type="button"
                className="btn ghost sm"
                style={{ position: "absolute", top: 10, right: 10, zIndex: 10 }}
                onClick={() =>
                  copyToClipboard(JSON.stringify(result.json_data, null, 2))
                }
              >
                <Copy size={13} /> {copied ? "Copied" : "Copy"}
              </button>
              <pre
                className="panel mono"
                style={{
                  fontSize: 12.5,
                  lineHeight: 1.5,
                  color: "var(--success)",
                  maxHeight: 600,
                  overflow: "auto",
                  margin: 0,
                }}
              >
                {JSON.stringify(result.json_data, null, 2)}
              </pre>
            </div>
          )}

          {/* De-identification report */}
          {result.de_identification_report &&
            Object.keys(result.de_identification_report).length > 0 && (
              <div
                style={{
                  marginTop: 18,
                  paddingTop: 16,
                  borderTop: "1px solid var(--border)",
                }}
              >
                <div className="field-l" style={{ marginBottom: 10 }}>
                  De-identification report
                </div>
                <div className="reasons">
                  {Object.entries(result.de_identification_report).map(
                    ([key, val]) => (
                      <span key={key} className="reason">
                        {key.replace(/_/g, " ")} · {val}
                      </span>
                    )
                  )}
                </div>
              </div>
            )}
        </div>
      )}

      {/* AI disclaimer — legally required wherever AI prompts/responses are shown. */}
      <div className="card-surface pad">
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <span className="tag" style={{ flexShrink: 0 }}>
            <span className="tdot" style={{ background: "var(--theme-ochre)" }} />
            Notice
          </span>
          <p className="dim" style={{ fontSize: 13, lineHeight: 1.55, margin: 0 }}>
            AI summaries and prompts are for personal reference only and do not
            constitute medical advice, diagnoses, or treatment recommendations.{" "}
            {privacyBoundary(
              result
                ? result.processing_mode
                : promptResult
                  ? "prompt_only"
                  : executionMode
            )}{" "}
            Model output may contain inaccuracies; verify anything important
            against the original records.
          </p>
        </div>
      </div>

      {/* History */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Summary history</h3>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? "Hide" : "Show"} ({history.length})
          </button>
        </div>
        {showHistory &&
          (history.length === 0 ? (
            <p className="muted" style={{ fontSize: 13, margin: 0 }}>
              No saved summaries yet.
            </p>
          ) : (
            <div>
              {history.map((h) => (
                <button
                  key={h.id}
                  type="button"
                  className="lrow"
                  onClick={() => handleViewSaved(h.id)}
                  title="Open this saved summary"
                  style={{
                    width: "100%",
                    background: "transparent",
                    border: 0,
                    borderBottom: "1px solid var(--border)",
                    cursor: "pointer",
                    textAlign: "left",
                  }}
                >
                  <span className="lrow-main">
                    <span className="lrow-title" style={{ textTransform: "capitalize" }}>
                      {h.summary_type.replace(/_/g, " ")} summary
                    </span>
                    <span className="lrow-sub">
                      {h.record_count} record{h.record_count === 1 ? "" : "s"}
                    </span>
                    <span className="lrow-sub">
                      {modeLabel(h.processing_mode)} ·{" "}
                      {provenanceLabel(
                        h.model_provenance,
                        h.processing_mode
                      )}
                    </span>
                  </span>
                  <span className="lrow-meta tnum">
                    {h.generated_at
                      ? new Date(h.generated_at).toLocaleDateString()
                      : ""}
                  </span>
                  <ChevronRight size={15} style={{ color: "var(--text-muted)" }} />
                </button>
              ))}
            </div>
          ))}
      </div>
    </div>
  );
}
