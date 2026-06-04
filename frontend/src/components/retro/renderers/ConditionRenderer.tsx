"use client";

import React from "react";
import { DetailRow, SectionDivider, str, nested, formatDate, arr, obj } from "./shared";

/** Pull a human-readable label from a CodeableConcept: text first, then first coding display. */
function conceptLabel(concept: unknown): string {
  const c = obj(concept);
  return str(c.text) || str(nested(c, "coding", "0", "display"));
}

export function ConditionRenderer({ r }: { r: Record<string, unknown> }) {
  // Name: most CDA/Epic conditions carry only code.coding[].display, not code.text.
  const name = conceptLabel(r.code);

  const clinicalStatus =
    str(nested(r, "clinicalStatus", "coding", "0", "code")) ||
    str(nested(r, "clinicalStatus", "coding", "0", "display")) ||
    str(nested(r, "clinicalStatus", "text")) ||
    "";
  const verificationStatus =
    str(nested(r, "verificationStatus", "coding", "0", "code")) ||
    str(nested(r, "verificationStatus", "coding", "0", "display")) ||
    str(nested(r, "verificationStatus", "text"));

  // Onset / abatement — handle DateTime, Period, and String variants.
  const onset =
    formatDate(r.onsetDateTime) ||
    formatDate(nested(r, "onsetPeriod", "start")) ||
    str(r.onsetString);
  const abatement =
    formatDate(r.abatementDateTime) ||
    formatDate(nested(r, "abatementPeriod", "end")) ||
    str(r.abatementString);
  const recordedDate = formatDate(r.recordedDate);

  const severity = conceptLabel(r.severity);
  const stage = conceptLabel(nested(r, "stage", "0", "summary"));

  // Body sites recorded.
  const bodySites: string[] = [];
  for (const bs of arr(r.bodySite)) {
    const label = conceptLabel(bs);
    if (label) bodySites.push(label);
  }

  // Notes.
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const t = str(obj(n).text);
    if (t) notes.push(t);
  }

  const statusLower = clinicalStatus.toLowerCase();
  const statusConfig: Record<string, { bg: string; text: string; pulse: boolean }> = {
    active: { bg: "var(--theme-sage)", text: "var(--theme-bg-deep)", pulse: true },
    resolved: { bg: "var(--theme-text-muted)", text: "var(--theme-bg-deep)", pulse: false },
    inactive: { bg: "var(--theme-bg-deep)", text: "var(--theme-text-dim)", pulse: false },
    recurrence: { bg: "var(--theme-ochre)", text: "var(--theme-bg-deep)", pulse: true },
    remission: { bg: "var(--theme-text-dim)", text: "var(--theme-bg-deep)", pulse: false },
  };
  const config = statusConfig[statusLower] ?? { bg: "var(--theme-bg-deep)", text: "var(--theme-text)", pulse: false };

  // Category chip labels.
  const categories = arr(r.category);
  const categoryChips: string[] = [];
  for (const cat of categories) {
    const text = str(obj(cat).text) || str(nested(obj(cat), "coding", "0", "display"));
    if (text) categoryChips.push(text);
  }

  return (
    <div className="space-y-3">
      {name && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {name}
        </p>
      )}

      {/* Clinical status pill + verification + severity (verbatim, no good/bad coloring) */}
      {(clinicalStatus || verificationStatus || severity) && (
        <div className="flex flex-wrap items-center gap-2">
          {clinicalStatus && (
            <span
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full"
              style={{ backgroundColor: config.bg, color: config.text }}
            >
              {config.pulse && (
                <span
                  className="w-1.5 h-1.5 rounded-full pulse-dot"
                  style={{ backgroundColor: config.text, "--dot-color": config.text } as React.CSSProperties}
                />
              )}
              {clinicalStatus}
            </span>
          )}
          {verificationStatus && verificationStatus.toLowerCase() !== "confirmed" && (
            <span
              className="text-[11px] px-1.5 py-0.5 rounded"
              style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-muted)" }}
            >
              {verificationStatus}
            </span>
          )}
          {severity && (
            <span
              className="text-[11px] px-1.5 py-0.5 rounded"
              style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-muted)" }}
            >
              {severity}
            </span>
          )}
        </div>
      )}

      {/* Timeline bar: onset → abatement */}
      {onset && (
        <div
          className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          <span style={{ color: "var(--theme-text-muted)" }}>Onset</span>
          <span style={{ color: "var(--theme-text)" }}>{onset}</span>
          {abatement && (
            <>
              <span style={{ color: "var(--theme-text-muted)" }}>→</span>
              <span style={{ color: "var(--theme-text-muted)" }}>Resolved</span>
              <span style={{ color: "var(--theme-text)" }}>{abatement}</span>
            </>
          )}
        </div>
      )}

      <DetailRow label="Recorded" value={recordedDate} />

      {/* Body site(s) */}
      {bodySites.length > 0 && (
        <DetailRow label="Body Site" value={bodySites.join(", ")} />
      )}

      {/* Stage */}
      <DetailRow label="Stage" value={stage} />

      {/* Category chips */}
      {categoryChips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {categoryChips.map((chip) => (
            <span
              key={chip}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-condition-bg)",
                color: "var(--record-condition-text)",
              }}
            >
              {chip}
            </span>
          ))}
        </div>
      )}

      {/* Notes */}
      {notes.length > 0 && (
        <>
          <SectionDivider />
          <div className="space-y-2">
            {notes.map((note, i) => (
              <div
                key={i}
                className="px-3 py-2 rounded-md text-xs"
                style={{
                  backgroundColor: "var(--theme-bg-deep)",
                  color: "var(--theme-text-dim)",
                  borderLeft: "2px solid var(--theme-border-active)",
                }}
              >
                {note}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
