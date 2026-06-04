"use client";

import React from "react";
import { DetailRow, BarRow, str, arr, obj, nested, formatDate } from "./shared";

const MODALITY_COLORS: Record<string, string> = {
  CT: "var(--record-imaging-text)",
  MRI: "var(--record-procedure-text)",
  "X-Ray": "var(--theme-ochre)",
  XR: "var(--theme-ochre)",
  US: "var(--theme-sage)",
  MG: "var(--theme-sienna)",
  NM: "var(--theme-terracotta)",
};

// Pull display/text out of a CodeableConcept (or its first coding).
function codeableText(val: unknown): string {
  const c = obj(val);
  return (
    str(c.text) ||
    str(nested(c, "coding", "0", "display")) ||
    str(nested(c, "coding", "0", "code"))
  );
}

export function ImagingRenderer({ r }: { r: Record<string, unknown> }) {
  const description = str(r.description) || str(nested(r, "code", "text")) || "";
  const startedDate = formatDate(r.started);

  // Modality badges — modality may be an array of Coding/CodeableConcept,
  // or a single object on the study level (R4 vs R4B variants).
  const modalityArr = arr(r.modality);
  const modalityCodes: string[] = [];
  for (const m of modalityArr) {
    const code = str(obj(m).code) || str(nested(obj(m), "coding", "0", "code"));
    const display = str(obj(m).display) || str(nested(obj(m), "coding", "0", "display"));
    if (code || display) modalityCodes.push(display || code);
  }
  if (modalityCodes.length === 0) {
    const singleModality = str(nested(r, "modality", "code")) || str(nested(r, "modality", "display"));
    if (singleModality) modalityCodes.push(singleModality);
  }

  const procedureNames = arr(r.procedureCode).map(codeableText).filter(Boolean);

  // reasonCode[] (R4) / reason[] (R4B) — why the study was performed
  const reasonNames = [...arr(r.reasonCode), ...arr(r.reason)].map(codeableText).filter(Boolean);

  const referrer = str(nested(r, "referrer", "display"));

  const noteTexts = arr(r.note)
    .map((n) => str(obj(n).text))
    .filter(Boolean);

  const series = arr(r.series);
  const seriesCount = str(r.numberOfSeries) || (series.length ? String(series.length) : "");

  // numberOfInstances — explicit study count, else sum across series
  let instanceCount = str(r.numberOfInstances);
  if (!instanceCount && series.length > 0) {
    const sum = series.reduce<number>((acc, s) => {
      const n = Number(str(obj(s).numberOfInstances));
      return acc + (Number.isFinite(n) ? n : 0);
    }, 0);
    if (sum > 0) instanceCount = String(sum);
  }

  // Per-series summary: modality · body site / description · N instances
  const seriesSummaries = series
    .map((s) => {
      const so = obj(s);
      const mod =
        str(so.modality && obj(so.modality).code) ||
        str(nested(so, "modality", "coding", "0", "code")) ||
        str(nested(so, "modality", "display"));
      const site = str(nested(so, "bodySite", "display")) || str(so.description);
      const inst = str(so.numberOfInstances);
      const parts = [mod, site].filter(Boolean);
      const head = parts.join(" · ");
      return inst ? `${head}${head ? " " : ""}(${inst})` : head;
    })
    .filter(Boolean);

  return (
    <div className="space-y-3">
      {description && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {description}
        </p>
      )}

      {/* Modality badges */}
      {modalityCodes.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {modalityCodes.map((mod) => (
            <span
              key={mod}
              className="px-2 py-0.5 text-xs font-semibold rounded"
              style={{
                backgroundColor: "var(--record-imaging-bg)",
                color: MODALITY_COLORS[mod] ?? "var(--record-imaging-text)",
              }}
            >
              {mod}
            </span>
          ))}
        </div>
      )}

      <DetailRow label="Study Date" value={startedDate} />

      {procedureNames.length > 0 && (
        <DetailRow label="Procedure" value={procedureNames.join(", ")} />
      )}

      {reasonNames.length > 0 && (
        <DetailRow label="Reason" value={reasonNames.join(", ")} />
      )}

      <DetailRow label="Referrer" value={referrer} />

      {/* Series / instance counts */}
      {(seriesCount || instanceCount) && (
        <BarRow
          items={[
            { label: "Series", value: seriesCount },
            { label: "Instances", value: instanceCount },
          ]}
        />
      )}

      {/* Per-series breakdown */}
      {seriesSummaries.length > 0 && (
        <div
          className="px-3 py-2 rounded-md text-xs space-y-1"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          <div className="text-[10px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
            Series
          </div>
          {seriesSummaries.map((s, i) => (
            <div key={i} style={{ color: "var(--theme-text)" }}>
              {s}
            </div>
          ))}
        </div>
      )}

      {/* Notes */}
      {noteTexts.length > 0 && (
        <div
          className="px-3 py-2 rounded-md text-xs space-y-1"
          style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-dim)" }}
        >
          <div className="text-[10px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
            Notes
          </div>
          {noteTexts.map((n, i) => (
            <div key={i}>{n}</div>
          ))}
        </div>
      )}
    </div>
  );
}
