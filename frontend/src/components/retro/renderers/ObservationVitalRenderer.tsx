"use client";

import React from "react";
import { DetailRow, StatusBadge, str, obj, arr, nested, formatDateTime, performerNames } from "./shared";

// ---------------------------------------------------------------------------
// Local helpers (kept here so shared.tsx stays untouched)
// ---------------------------------------------------------------------------

/** Map an HL7 v3 ObservationInterpretation code to a readable label. Verbatim
 *  transliteration only — no good/bad judgement (CLAUDE.md Rule #1). */
function interpretationLabel(code: string): string {
  switch (code.toUpperCase()) {
    case "H":
      return "High";
    case "HH":
      return "Critical High";
    case "HX":
      return "Above measurable range";
    case "L":
      return "Low";
    case "LL":
      return "Critical Low";
    case "LX":
      return "Below measurable range";
    case "N":
      return "Normal";
    case "A":
      return "Abnormal";
    default:
      return code;
  }
}

interface ComponentValue {
  name: string;
  value: string;
  unit: string;
}

function readComponent(comp: unknown): ComponentValue | null {
  const c = obj(comp);
  const name =
    str(nested(c, "code", "text")) || str(nested(c, "code", "coding", "0", "display")) || "";
  const value =
    str(nested(c, "valueQuantity", "value")) ||
    str(c.valueString) ||
    str(nested(c, "valueCodeableConcept", "text")) ||
    str(nested(c, "valueCodeableConcept", "coding", "0", "display"));
  const unit = str(nested(c, "valueQuantity", "unit"));
  if (!value) return null;
  return { name, value, unit };
}

export function ObservationVitalRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";

  const valueQuantity = obj(r.valueQuantity);
  const valueNum = str(valueQuantity.value);
  const valueUnit = str(valueQuantity.unit);
  const valueString = str(r.valueString);
  const displayValue = valueNum || valueString;

  const effective = formatDateTime(r.effectiveDateTime ?? nested(r, "effectivePeriod", "start"));
  const status = str(r.status);

  // Component values (e.g. blood pressure: Systolic / Diastolic). Critical for
  // panels that have no top-level value — render each component's name + value.
  const components = arr(r.component)
    .map(readComponent)
    .filter((c): c is ComponentValue => c !== null);
  const showComponents = components.length > 0 && !displayValue;

  // Source's own flag, neutral text only.
  const interpCode =
    str(nested(r, "interpretation", "0", "coding", "0", "code")) ||
    str(nested(r, "interpretation", "0", "coding", "0", "display")) ||
    str(nested(r, "interpretation", "0", "text")) ||
    "";
  const interpLabel = interpCode ? interpretationLabel(interpCode) : "";

  const bodySite =
    str(nested(r, "bodySite", "text")) || str(nested(r, "bodySite", "coding", "0", "display"));

  // Who recorded it — preferring a display name, falling back to a readable reference.
  const performers = performerNames(arr(r.performer));

  const notes = arr(r.note)
    .map((n) => str(obj(n).text))
    .filter(Boolean);

  return (
    <div className="space-y-3">
      {name && (
        <p className="text-[13px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
          {name}
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
          {valueNum && valueUnit && (
            <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
              {valueUnit}
            </span>
          )}
          {interpLabel && (
            <span className="text-xs font-medium ml-1" style={{ color: "var(--theme-text-muted)" }}>
              {interpLabel}
            </span>
          )}
        </div>
      )}

      {/* Component values (blood pressure, etc.) — each labelled with its name */}
      {showComponents && (
        <div className="space-y-1">
          {components.map((c, i) => (
            <div key={i} className="flex items-baseline gap-2">
              {c.name && (
                <span
                  className="text-[11px] font-medium uppercase tracking-wide w-20 shrink-0"
                  style={{ color: "var(--theme-text-muted)" }}
                >
                  {c.name}
                </span>
              )}
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "20px",
                  lineHeight: 1,
                  color: "var(--theme-amber)",
                }}
              >
                {c.value}
              </span>
              {c.unit && (
                <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
                  {c.unit}
                </span>
              )}
            </div>
          ))}
          {interpLabel && (
            <span className="text-xs font-medium" style={{ color: "var(--theme-text-muted)" }}>
              {interpLabel}
            </span>
          )}
        </div>
      )}

      <DetailRow label="Date" value={effective} />
      <DetailRow label="Body Site" value={bodySite} />
      <DetailRow label={performers.length > 1 ? "Performers" : "Performer"} value={performers.join(", ")} />

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
