"use client";

import { useCallback, useEffect, useState } from "react";
import { Lock, Copy, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type {
  PatientInfo,
  GenerateSummaryResponse,
  PromptResponse,
} from "@/types/api";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";

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

const OUTPUT_FORMATS = [
  { value: "natural_language", label: "Natural language" },
  { value: "json", label: "JSON data" },
  { value: "both", label: "Both" },
];

const RESULT_TABS = [
  { key: "nl", label: "Narrative" },
  { key: "json", label: "JSON data" },
];

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
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

  // Config
  const [summaryType, setSummaryType] = useState("full");
  const [category, setCategory] = useState("observation");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [outputFormat, setOutputFormat] = useState("both");
  const [showCustomize, setShowCustomize] = useState(false);
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");

  // Results
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateSummaryResponse | null>(null);
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
      const d = await api.get<
        PromptResponse & {
          response_text?: string | null;
          response_format?: string | null;
        }
      >(`/summary/prompts/${id}`);

      if (!d.response_text) {
        setError("This saved summary has no stored response text.");
        return;
      }

      // The backend stores "both" output as NL + "\n\n---JSON---\n" + JSON.
      const marker = "\n\n---JSON---\n";
      let naturalLanguage: string | null = null;
      let jsonData: Record<string, unknown> | null = null;

      if (d.response_text.includes(marker)) {
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
        natural_language: naturalLanguage,
        json_data: jsonData,
        record_count: d.record_count,
        duplicate_warning: null,
        de_identification_report: d.de_identification_report,
        model_used: d.target_model,
        generated_at: d.generated_at,
      });
      setResultTab(naturalLanguage ? "nl" : "json");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load saved summary"
      );
    }
  };

  const handleGenerate = async () => {
    if (!selectedPatient) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body: Record<string, unknown> = {
        patient_id: selectedPatient,
        summary_type: summaryType,
        output_format: outputFormat,
      };
      if (summaryType === "category") body.category = category;
      if (summaryType === "date_range" && dateFrom) body.date_from = dateFrom;
      if (summaryType === "date_range" && dateTo) body.date_to = dateTo;
      if (customSystemPrompt.trim())
        body.custom_system_prompt = customSystemPrompt;

      const resp = await api.post<GenerateSummaryResponse>(
        "/summary/generate",
        body
      );
      setResult(resp);
      loadHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Summary generation failed");
    } finally {
      setLoading(false);
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
          <p className="kicker">Private &amp; de-identified</p>
          <h1 className="h1 display">Summaries</h1>
          <p className="h-sub">
            Generate a private, PHI-scrubbed summary of this record to share with a
            provider, family member, advocate, or AI assistant. Personal identifiers
            are removed before anything leaves your device.
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
              aria-selected={summaryType === t.key}
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

        {/* Output format */}
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
            placeholder="Override the system prompt (leave empty for the default)…"
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
      </div>

      {/* Generate */}
      <div style={{ display: "flex", justifyContent: "center" }}>
        <button
          className="btn"
          onClick={handleGenerate}
          disabled={loading || !selectedPatient}
        >
          {loading ? "Generating…" : "Generate summary"}
        </button>
      </div>

      {/* Loading */}
      {loading && <RetroLoadingState text="Generating summary" />}

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

      {/* Results */}
      {result && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Summary</h3>
            <span className="muted mono" style={{ fontSize: 11 }}>
              {result.record_count} record{result.record_count === 1 ? "" : "s"}
              {result.model_used ? ` · ${result.model_used}` : ""}
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
                    aria-selected={resultTab === t.key}
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
            AI summaries are for personal reference only and do not constitute
            medical advice, diagnoses, or treatment recommendations. All health
            data is de-identified before being sent to the AI model. Summaries are
            generated by Gemini 3 Flash and may contain inaccuracies; verify
            anything important against your original records.
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
