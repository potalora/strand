"use client";

import React from "react";
import { DetailRow, StatusBadge, str, obj, arr, nested, formatDateTime } from "./shared";
import { Gauge } from "@/components/retro/DataViz";

// ---------------------------------------------------------------------------
// Local helpers (kept here so shared.tsx stays untouched)
// ---------------------------------------------------------------------------

/** Map an HL7 v3 ObservationInterpretation code to a readable label. Purely a
 *  transliteration of what the source recorded — no judgement, no good/bad
 *  coloring (CLAUDE.md Rule #1: descriptive, not normative). */
function interpretationLabel(code: string): string {
  switch (code.toUpperCase()) {
    case "H":
      return "High";
    case "HH":
      return "Critical High";
    case "HX":
      return "Above measurable range";
    case "HU":
      return "Significantly High";
    case "L":
      return "Low";
    case "LL":
      return "Critical Low";
    case "LX":
      return "Below measurable range";
    case "LU":
      return "Significantly Low";
    case "N":
      return "Normal";
    case "A":
      return "Abnormal";
    case "AA":
      return "Critical Abnormal";
    default:
      return code;
  }
}

/** Resolve a single performer entry to a display string, preferring a human
 *  display, then a non-opaque reference (drops bare Type/UUID references). */
function performerLabel(p: unknown): string {
  const display = str(obj(p).display);
  if (display) return display;
  const reference = str(obj(p).reference);
  if (!reference) return "";
  // References here are typically "Organization/<uuid>" — opaque, not useful to
  // surface. Only show the type segment if there is no UUID-looking id.
  const [, id] = reference.split("/");
  const looksLikeUuid = /^[0-9a-f]{8}-[0-9a-f]{4}/i.test(id ?? "");
  return looksLikeUuid ? "" : reference;
}

export function ObservationLabRenderer({ r }: { r: Record<string, unknown> }) {
  const testName =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";

  // value[x]: quantity (number + unit), string, or codeable concept.
  const valueQuantity = obj(r.valueQuantity);
  const valueNum = str(valueQuantity.value);
  const valueUnit = str(valueQuantity.unit);
  const valueString = str(r.valueString);
  const valueCodeable =
    str(nested(r, "valueCodeableConcept", "text")) ||
    str(nested(r, "valueCodeableConcept", "coding", "0", "display")) ||
    str(nested(r, "valueCodeableConcept", "coding", "0", "code"));
  const displayValue = valueNum || valueString || valueCodeable;
  const displayUnit = valueNum ? valueUnit : "";

  // Effective time: datetime, else the start of a period.
  const effective = formatDateTime(r.effectiveDateTime ?? nested(r, "effectivePeriod", "start"));
  const status = str(r.status);

  // Reference range as recorded by the source (verbatim, "per source").
  const refRangeArr = arr(r.referenceRange);
  const refRange = obj(refRangeArr[0]);
  const refLow = str(nested(refRange, "low", "value"));
  const refHigh = str(nested(refRange, "high", "value"));
  const refUnit = str(nested(refRange, "low", "unit")) || str(nested(refRange, "high", "unit"));
  let refText = str(refRange.text);
  if (!refText && (refLow || refHigh)) {
    refText = `${refLow || "?"} - ${refHigh || "?"}${refUnit ? ` ${refUnit}` : ""}`;
  }

  // The source's own flag (e.g. "H", "HX", "LX"). Shown as neutral text only —
  // this app organizes records, it does not interpret values as good/bad.
  const interpCode =
    str(nested(r, "interpretation", "0", "coding", "0", "code")) ||
    str(nested(r, "interpretation", "0", "coding", "0", "display")) ||
    str(nested(r, "interpretation", "0", "text")) ||
    "";
  const interpLabel = interpCode ? interpretationLabel(interpCode) : "";

  // Performer(s) — preferring display, falling back to a readable reference.
  const performers = arr(r.performer).map(performerLabel).filter(Boolean);
  const performerText = performers.join(", ");

  // Specimen, if the source recorded one.
  const specimen =
    str(nested(r, "specimen", "display")) || str(nested(r, "specimen", "reference"));

  // Free-text notes, verbatim.
  const notes = arr(r.note)
    .map((n) => str(obj(n).text))
    .filter(Boolean);

  // Reference-range marker: only when the value sits within a numeric low/high
  // band recorded by the source. The shaded band is the recorded range (a
  // neutral marker), not a "good zone".
  const numVal = parseFloat(valueNum);
  const numLow = parseFloat(refLow);
  const numHigh = parseFloat(refHigh);
  const showGauge = !isNaN(numVal) && !isNaN(numLow) && !isNaN(numHigh) && numHigh > numLow;

  return (
    <div className="space-y-3">
      {testName && (
        <p className="text-[13px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
          {testName}
        </p>
      )}

      {displayValue && (
        <div className="flex items-baseline gap-2 flex-wrap">
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "24px",
              lineHeight: 1,
              color: "var(--theme-amber)",
            }}
          >
            {displayValue}
          </span>
          {displayUnit && (
            <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
              {displayUnit}
            </span>
          )}
          {interpLabel && (
            <span className="text-xs font-medium ml-1" style={{ color: "var(--theme-text-muted)" }}>
              {interpLabel}
            </span>
          )}
        </div>
      )}

      {/* Reference range: neutral marker (value relative to the source's range) */}
      {showGauge && <Gauge value={numVal} low={numLow} high={numHigh} />}

      {!showGauge && refText && <DetailRow label="Reference Range (per source)" value={refText} mono />}

      {/* Interpretation shown standalone when there is no value to inline it next to */}
      {!displayValue && interpLabel && <DetailRow label="Interpretation (per source)" value={interpLabel} />}

      <DetailRow label="Date" value={effective} />
      <DetailRow label="Performer" value={performerText} />
      <DetailRow label="Specimen" value={specimen} />

      {status && (
        <div className="pt-1">
          <StatusBadge label={status} />
        </div>
      )}

      {notes.length > 0 && (
        <div className="space-y-1">
          {notes.map((n, i) => (
            <p
              key={i}
              className="text-xs px-3 py-2 rounded-md"
              style={{
                backgroundColor: "var(--theme-bg-deep)",
                color: "var(--theme-text)",
                borderLeft: "2px solid var(--theme-border-active)",
              }}
            >
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
