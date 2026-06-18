"use client";

import React from "react";
import { DetailRow, StatusBadge, str, obj, arr, nested, formatDateTime, performerNames } from "./shared";

// ---------------------------------------------------------------------------
// Local helpers (kept here so shared.tsx stays untouched)
// ---------------------------------------------------------------------------

interface ComponentValue {
  name: string;
  value: string;
  unit: string;
}

/** Read a component (e.g. tobacco pack-years, drinks/week) into a row. */
function readComponent(comp: unknown): ComponentValue | null {
  const c = obj(comp);
  const name =
    str(nested(c, "code", "text")) || str(nested(c, "code", "coding", "0", "display")) || "";
  const qtyVal = str(nested(c, "valueQuantity", "value"));
  const qtyUnit = str(nested(c, "valueQuantity", "unit"));
  const value =
    qtyVal ||
    str(c.valueString) ||
    str(nested(c, "valueCodeableConcept", "text")) ||
    str(nested(c, "valueCodeableConcept", "coding", "0", "display")) ||
    str(nested(c, "valuePeriod", "start"));
  if (!value) return null;
  return { name, value, unit: qtyVal ? qtyUnit : "" };
}

export function ObservationSocialRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";

  // value[x]: codeable concept (smoking status etc.), string, or quantity.
  const valueQuantity = obj(r.valueQuantity);
  const valueNum = str(valueQuantity.value);
  const valueUnit = str(valueQuantity.unit);
  const valueString = str(r.valueString);
  const valueCodeable =
    str(nested(r, "valueCodeableConcept", "text")) ||
    str(nested(r, "valueCodeableConcept", "coding", "0", "display")) ||
    str(nested(r, "valueCodeableConcept", "coding", "0", "code"));
  const displayValue =
    valueCodeable || valueString || (valueNum ? `${valueNum}${valueUnit ? ` ${valueUnit}` : ""}` : "");

  const effective = formatDateTime(r.effectiveDateTime ?? nested(r, "effectivePeriod", "start"));
  const status = str(r.status);
  const performers = performerNames(arr(r.performer));

  // Components (e.g. pack-years, period of use) — verbatim from the source.
  const components = arr(r.component)
    .map(readComponent)
    .filter((c): c is ComponentValue => c !== null);

  const notes = arr(r.note)
    .map((n) => str(obj(n).text))
    .filter(Boolean);

  return (
    <div
      className="space-y-2 px-3 py-2 rounded-md"
      style={{
        backgroundColor: "var(--theme-bg-deep)",
        borderLeft: "2px solid var(--theme-border-active)",
      }}
    >
      {name && (
        <p className="text-[13px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
          {name}
        </p>
      )}

      {displayValue && (
        <p className="text-sm" style={{ color: "var(--theme-text)" }}>
          {displayValue}
        </p>
      )}

      {components.length > 0 && (
        <div className="space-y-1 pt-1">
          {components.map((c, i) => (
            <div key={i} className="flex items-baseline gap-2">
              {c.name && (
                <span
                  className="text-[11px] font-medium uppercase tracking-wide"
                  style={{ color: "var(--theme-text-muted)" }}
                >
                  {c.name}:
                </span>
              )}
              <span className="text-sm" style={{ color: "var(--theme-text)" }}>
                {c.value}
                {c.unit ? ` ${c.unit}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      <DetailRow label="Date" value={effective} />
      <DetailRow label={performers.length > 1 ? "Performers" : "Performer"} value={performers.join(", ")} />

      {status && (
        <div className="pt-1">
          <StatusBadge label={status} />
        </div>
      )}

      {notes.length > 0 && (
        <div className="space-y-1 pt-1">
          {notes.map((n, i) => (
            <p key={i} className="text-xs" style={{ color: "var(--theme-text)" }}>
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
