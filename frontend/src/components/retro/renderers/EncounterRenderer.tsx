"use client";

import React from "react";
import { DetailRow, StatusBadge, str, obj, arr, nested, formatDate, performerNames } from "./shared";
import { narrativeText } from "./narrative";

const CLASS_BORDER_COLORS: Record<string, string> = {
  amb: "var(--theme-sage)",
  ambulatory: "var(--theme-sage)",
  imp: "var(--theme-ochre)",
  inpatient: "var(--theme-ochre)",
  emer: "var(--theme-terracotta)",
  emergency: "var(--theme-terracotta)",
  vr: "var(--record-procedure-text)",
  virtual: "var(--record-procedure-text)",
};

// Human-readable labels for HL7 ActEncounterCode class codes.
const CLASS_LABELS: Record<string, string> = {
  amb: "Ambulatory",
  imp: "Inpatient",
  emer: "Emergency",
  vr: "Virtual",
  hh: "Home Health",
  obsenc: "Observation",
  acute: "Acute",
  ss: "Short Stay",
  prenc: "Pre-admission",
  nonac: "Non-acute",
  ub: "Newborn",
};

/** Compact label for the encounter class. Maps ACT codes to readable text;
 * otherwise prefers a short display, falling back to the raw code. */
function classLabel(r: Record<string, unknown>): string {
  const code = str(nested(r, "class", "code"));
  const display = str(nested(r, "class", "display"));
  const mapped = CLASS_LABELS[code.toLowerCase()];
  if (mapped) return mapped;
  // Some sources put a CPT visit description in class.display — keep it short.
  if (display && display.length <= 48) return display;
  if (code && display) return code; // long display: show the code only
  return display || code;
}

/** Descriptive duration between two ISO instants, e.g. "1 hr 30 min". */
function durationLabel(start: unknown, end: unknown): string {
  const s = str(start);
  const e = str(end);
  if (!s || !e) return "";
  const ds = new Date(s);
  const de = new Date(e);
  if (isNaN(ds.getTime()) || isNaN(de.getTime())) return "";
  const mins = Math.round((de.getTime() - ds.getTime()) / 60000);
  if (mins <= 0) return "";
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h && m) return `${h} hr ${m} min`;
  if (h) return `${h} hr`;
  return `${m} min`;
}

/** Collect display text from an array of FHIR references, e.g. participant[].individual. */
function refDisplays(items: unknown[], ...keys: string[]): string[] {
  const out: string[] = [];
  for (const item of items) {
    const node = keys.length ? nested(obj(item), ...keys) : item;
    const display = str(obj(node).display);
    if (display && !out.includes(display)) out.push(display);
  }
  return out;
}

/** Collect human text from a codeableConcept[] (text → coding display → coding code). */
function conceptTexts(items: unknown[]): string[] {
  const out: string[] = [];
  for (const item of items) {
    const cc = obj(item);
    const text =
      str(cc.text) ||
      str(nested(cc, "coding", "0", "display")) ||
      str(nested(cc, "coding", "0", "code"));
    if (text && !out.includes(text)) out.push(text);
  }
  return out;
}

export function EncounterRenderer({ r }: { r: Record<string, unknown> }) {
  const encounterType =
    str(nested(r, "type", "0", "text")) ||
    str(nested(r, "type", "0", "coding", "0", "display")) ||
    "";
  const status = str(r.status);
  const periodStart = formatDate(nested(r, "period", "start"));
  const periodEnd = formatDate(nested(r, "period", "end"));
  const duration = durationLabel(nested(r, "period", "start"), nested(r, "period", "end"));

  const klass = classLabel(r);

  // Providers — list all, not just the first; decode a readable reference when
  // no display is present.
  const providers = performerNames(arr(r.participant), "individual");

  // Reasons — list all coded reasons.
  const reasons = conceptTexts(arr(r.reasonCode));

  const serviceProvider = str(nested(r, "serviceProvider", "display"));

  // Locations (rooms/departments), minus any entry that just repeats the
  // medical center promoted from a location.
  const locations = refDisplays(arr(r.location), "location").filter(
    (loc) => loc !== serviceProvider,
  );

  // Visit summary — AI-extracted encounters carry a synopsis in the FHIR
  // Narrative (text.div). Rendered as plain prose; nothing shown when absent.
  const summary = narrativeText(nested(r, "text", "div"));

  // Hospitalization details (inpatient stays).
  const admitSource =
    str(nested(r, "hospitalization", "admitSource", "text")) ||
    str(nested(r, "hospitalization", "admitSource", "coding", "0", "display"));
  const dischargeDisposition =
    str(nested(r, "hospitalization", "dischargeDisposition", "text")) ||
    str(nested(r, "hospitalization", "dischargeDisposition", "coding", "0", "display"));

  // Diagnoses — show what the source linked (use label + condition reference).
  const diagnoses = arr(r.diagnosis);
  const diagnosisItems: string[] = [];
  for (const d of diagnoses) {
    const node = obj(d);
    const use =
      str(nested(node, "use", "text")) ||
      str(nested(node, "use", "coding", "0", "display"));
    const condition =
      str(nested(node, "condition", "display")) ||
      str(nested(node, "condition", "reference"));
    const label = [use, condition].filter(Boolean).join(": ");
    if (label) diagnosisItems.push(label);
  }

  const encounterClass = str(nested(r, "class", "code")).toLowerCase() ||
    str(nested(r, "class", "display")).toLowerCase();
  const borderColor = CLASS_BORDER_COLORS[encounterClass] ?? "var(--record-encounter-dot)";

  // Fall back to the class label as a title when there is no type (common for
  // CDA-sourced encounters that only carry a CPT-coded class).
  const title = encounterType || klass;

  return (
    <div
      className="space-y-3 record-accent-left"
      style={{ "--accent-color": borderColor } as React.CSSProperties}
    >
      {title && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {title}
        </p>
      )}

      {/* Class + status chips */}
      {(status || (klass && klass !== title)) && (
        <div className="flex flex-wrap items-center gap-2">
          {status && <StatusBadge label={status} />}
          {klass && klass !== title && (
            <span
              className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md"
              style={{
                backgroundColor: "var(--record-encounter-bg)",
                color: "var(--record-encounter-text)",
              }}
            >
              {klass}
            </span>
          )}
        </div>
      )}

      {/* Date range with duration */}
      {periodStart && (
        <div
          className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          <span style={{ color: "var(--theme-text)" }}>{periodStart}</span>
          {periodEnd && periodEnd !== periodStart && (
            <>
              <span style={{ color: "var(--theme-text-muted)" }}>→</span>
              <span style={{ color: "var(--theme-text)" }}>{periodEnd}</span>
            </>
          )}
          {duration && (
            <span style={{ color: "var(--theme-text-muted)" }}>({duration})</span>
          )}
        </div>
      )}

      {/* Provider leads — the most-asked-for visit detail. */}
      <DetailRow label={providers.length > 1 ? "Providers" : "Provider"} value={providers.join(", ")} />
      <DetailRow label="Medical center" value={serviceProvider} />
      <DetailRow label={locations.length > 1 ? "Locations" : "Location"} value={locations.join(", ")} />
      <DetailRow label={reasons.length > 1 ? "Reasons" : "Reason"} value={reasons.join("; ")} />

      {summary && (
        <div className="flex flex-col gap-1 py-1">
          <span
            className="text-[11px] font-medium uppercase tracking-wide"
            style={{ color: "var(--theme-text-muted)" }}
          >
            Summary
          </span>
          <p
            className="text-sm px-3 py-2 rounded-md"
            style={{
              color: "var(--theme-text)",
              backgroundColor: "var(--theme-bg-deep)",
              lineHeight: 1.5,
            }}
          >
            {summary}
          </p>
        </div>
      )}

      <DetailRow label="Admit source" value={admitSource} />
      <DetailRow label="Discharge disposition" value={dischargeDisposition} />

      {diagnosisItems.length > 0 && (
        <DetailRow
          label={diagnosisItems.length > 1 ? "Diagnoses" : "Diagnosis"}
          value={diagnosisItems.join("; ")}
        />
      )}
    </div>
  );
}
