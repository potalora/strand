"use client";

import React from "react";
import { DetailRow, SectionDivider, StatusBadge, str, obj, arr, nested, formatDate, formatDateTime } from "./shared";

// ---------------------------------------------------------------------------
// ImmunizationRecommendation renderer.
//
// Shows the recommendation date / issuing authority, then the per-vaccine
// recommendation list (vaccine, target disease, forecast status, date
// criteria, dose/series, description). Descriptive only — it reports what the
// source forecast recorded, not advice.
// ---------------------------------------------------------------------------

/** First non-empty text from a CodeableConcept (text, then coding display/code). */
function codeableText(val: unknown): string {
  const c = obj(val);
  const text = str(c.text);
  if (text) return text;
  const codings = arr(c.coding);
  for (const coding of codings) {
    const display = str(obj(coding).display);
    if (display) return display;
  }
  for (const coding of codings) {
    const code = str(obj(coding).code);
    if (code) return code;
  }
  return "";
}

/** vaccineCode is 0..* CodeableConcept; join the readable names. */
function vaccineNames(rec: Record<string, unknown>): string {
  const codes = arr(rec.vaccineCode);
  if (codes.length > 0) {
    return codes.map(codeableText).filter(Boolean).join(", ");
  }
  // Some payloads inline a single CodeableConcept rather than an array.
  return codeableText(rec.vaccineCode);
}

interface RecItemProps {
  rec: Record<string, unknown>;
}

function RecItem({ rec }: RecItemProps) {
  const vaccine = vaccineNames(rec) || "Recommended immunization";
  const targetDisease = codeableText(rec.targetDisease);
  const forecast = codeableText(rec.forecastStatus);
  const description = str(rec.description);

  // doseNumber / seriesDoses can be positiveInt (R4) or string (R5-ish exports).
  const doseNumber = str(rec.doseNumber);
  const seriesDoses = str(rec.seriesDoses);
  const series = str(rec.series);

  // dateCriterion[]: each has code (CodeableConcept) + value (dateTime).
  const dateCriteria = arr(rec.dateCriterion)
    .map((dc) => {
      const d = obj(dc);
      const label = codeableText(d.code) || "Date";
      const when = formatDate(d.value);
      return when ? { label, when } : null;
    })
    .filter((x): x is { label: string; when: string } => x !== null);

  return (
    <div className="px-3 py-2.5 space-y-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold" style={{ color: "var(--theme-text)" }}>
          {vaccine}
        </span>
        {forecast && <StatusBadge label={forecast} />}
      </div>

      {targetDisease && (
        <div className="flex items-baseline gap-1.5">
          <span style={{ color: "var(--theme-text-muted)" }}>Target</span>
          <span style={{ color: "var(--theme-text)" }}>{targetDisease}</span>
        </div>
      )}

      {(doseNumber || seriesDoses || series) && (
        <div className="flex items-baseline gap-1.5">
          <span style={{ color: "var(--theme-text-muted)" }}>Dose</span>
          <span style={{ color: "var(--theme-text)" }}>
            {doseNumber || "?"}
            {seriesDoses ? ` of ${seriesDoses}` : ""}
            {series ? ` · ${series}` : ""}
          </span>
        </div>
      )}

      {/* Date criteria (e.g. "Earliest date", "Recommended date", "Due date"). */}
      {dateCriteria.map((dc, i) => (
        <div key={i} className="flex items-baseline gap-1.5">
          <span style={{ color: "var(--theme-text-muted)" }}>{dc.label}</span>
          <span style={{ color: "var(--theme-text)" }}>{dc.when}</span>
        </div>
      ))}

      {description && (
        <p style={{ color: "var(--theme-text-dim)" }}>{description}</p>
      )}
    </div>
  );
}

export function ImmunizationRecommendationRenderer({ r }: { r: Record<string, unknown> }) {
  const date = formatDateTime(r.date);
  const authority = str(nested(r, "authority", "display"));
  const recommendations = arr(r.recommendation);

  return (
    <div className="space-y-3">
      <p
        className="text-base font-semibold"
        style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
      >
        Immunization Recommendation
      </p>

      <DetailRow label="Date" value={date} />
      <DetailRow label="Authority" value={authority} />

      {recommendations.length > 0 ? (
        <>
          <SectionDivider />
          <div
            className="rounded-md overflow-hidden"
            style={{ border: "1px solid var(--theme-border)", backgroundColor: "var(--theme-bg-deep)" }}
          >
            {recommendations.map((rec, i) => (
              <div
                key={i}
                style={{
                  borderBottom: i < recommendations.length - 1 ? "1px solid var(--theme-border)" : "none",
                }}
              >
                <RecItem rec={obj(rec)} />
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
          No recommendations recorded
        </p>
      )}
    </div>
  );
}
