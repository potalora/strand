"use client";

import React from "react";
import { DetailRow, StatusBadge, BarRow, str, obj, arr, nested, formatDate } from "./shared";

// Pull display/text out of a CodeableConcept (or its first coding).
function codeableText(val: unknown): string {
  const c = obj(val);
  return (
    str(c.text) ||
    str(nested(c, "coding", "0", "display")) ||
    str(nested(c, "coding", "0", "code"))
  );
}

export function DiagnosticReportRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    str(nested(r, "code", "coding", "0", "code")) ||
    "";
  const status = str(r.status);

  // category[] — descriptive labels (e.g. "Hematology")
  const categoryNames = arr(r.category).map(codeableText).filter(Boolean);

  const effectiveDate = formatDate(r.effectiveDateTime ?? nested(r, "effectivePeriod", "start"));
  const effectiveEnd = formatDate(nested(r, "effectivePeriod", "end"));
  const issued = formatDate(r.issued);

  const conclusion = str(r.conclusion);
  const conclusionCodes = arr(r.conclusionCode).map(codeableText).filter(Boolean);

  const results = arr(r.result);
  const performers = arr(r.performer)
    .map((p) => str(obj(p).display))
    .filter(Boolean);

  // presentedForm[] — rendered/attached report documents
  const presentedForms = arr(r.presentedForm)
    .map((f) => {
      const att = obj(f);
      const title = str(att.title);
      const ct = str(att.contentType);
      const ctLabel = ct ? (ct.split("/").pop() ?? "").toUpperCase() : "";
      return [title, ctLabel].filter(Boolean).join(" · ");
    })
    .filter(Boolean);

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

      {/* Status + category badges */}
      {(status || categoryNames.length > 0) && (
        <div className="flex flex-wrap items-center gap-1.5">
          {status && <StatusBadge label={status} />}
          {categoryNames.map((cat) => (
            <span
              key={cat}
              className="px-2 py-0.5 text-xs font-medium rounded"
              style={{
                backgroundColor: "var(--record-diagnostic_report-bg)",
                color: "var(--record-diagnostic_report-text)",
              }}
            >
              {cat}
            </span>
          ))}
        </div>
      )}

      <DetailRow
        label="Effective"
        value={effectiveEnd && effectiveEnd !== effectiveDate ? `${effectiveDate} - ${effectiveEnd}` : effectiveDate}
      />
      <DetailRow label="Issued" value={issued} />

      {performers.length > 0 && (
        <DetailRow label="Performer" value={performers.join(", ")} />
      )}

      {results.length > 0 && (
        <DetailRow
          label="Results"
          value={`${results.length} ${results.length === 1 ? "result" : "results"}`}
        />
      )}

      {/* Conclusion block */}
      {conclusion && (
        <div
          className="px-3 py-2 rounded-md text-xs"
          style={{
            backgroundColor: "var(--record-diagnostic_report-bg)",
            color: "var(--theme-text)",
            borderLeft: "2px solid var(--record-diagnostic_report-dot)",
          }}
        >
          <div className="text-[10px] font-medium mb-1" style={{ color: "var(--theme-text-muted)" }}>
            Conclusion
          </div>
          {conclusion}
        </div>
      )}

      {conclusionCodes.length > 0 && (
        <DetailRow label="Conclusion Codes" value={conclusionCodes.join(", ")} />
      )}

      {/* presentedForm attachments */}
      {presentedForms.length > 0 && (
        <BarRow items={presentedForms.map((f, i) => ({ label: `Report ${i + 1}`, value: f }))} />
      )}
    </div>
  );
}
