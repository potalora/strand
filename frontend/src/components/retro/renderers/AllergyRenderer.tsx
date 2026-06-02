"use client";

import React from "react";
import { DetailRow, str, obj, arr, nested, formatDate } from "./shared";

export function AllergyRenderer({ r }: { r: Record<string, unknown> }) {
  const allergen =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";
  const severity =
    str(nested(r, "reaction", "0", "severity")) ||
    str(r.criticality) ||
    "";
  const clinicalStatus =
    str(nested(r, "clinicalStatus", "coding", "0", "code")) ||
    str(nested(r, "clinicalStatus", "text")) ||
    "";
  const date = formatDate(r.onsetDateTime ?? r.recordedDate);

  // Collect all reaction manifestations as chips
  const reactions = arr(nested(r, "reaction") as unknown);
  const manifestationChips: string[] = [];
  for (const reaction of reactions) {
    const manifestations = arr(obj(reaction).manifestation);
    for (const m of manifestations) {
      const text = str(obj(m).text) || str(nested(obj(m), "coding", "0", "display"));
      if (text) manifestationChips.push(text);
    }
  }

  // Record-TYPE accent (allergy hue) — neutral, not a severity good/bad judgement.
  const accentColor = "var(--record-allergy-dot)";

  return (
    <div
      className="space-y-3 record-accent-left"
      style={{ "--accent-color": accentColor } as React.CSSProperties}
    >
      {allergen && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--record-allergy-text)", fontFamily: "var(--font-body)" }}
        >
          {allergen}
        </p>
      )}

      {/* Severity flag — presented as neutral text, not color-coded good/bad */}
      {severity && (
        <span
          className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            color: "var(--theme-text-muted)",
          }}
        >
          {severity}
        </span>
      )}

      {/* Reaction manifestation chips */}
      {manifestationChips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {manifestationChips.map((chip) => (
            <span
              key={chip}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-allergy-bg)",
                color: "var(--record-allergy-text)",
              }}
            >
              {chip}
            </span>
          ))}
        </div>
      )}

      {clinicalStatus && (
        <span
          className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            color: "var(--theme-text)",
          }}
        >
          {clinicalStatus}
        </span>
      )}

      <DetailRow label="Recorded" value={date} />
    </div>
  );
}
