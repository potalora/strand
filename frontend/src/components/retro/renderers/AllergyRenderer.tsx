"use client";

import React from "react";
import { DetailRow, SectionDivider, str, obj, arr, nested, formatDate } from "./shared";

/** Pull a human-readable label from a CodeableConcept: text first, then first coding display. */
function conceptLabel(concept: unknown): string {
  const c = obj(concept);
  return str(c.text) || str(nested(c, "coding", "0", "display"));
}

export function AllergyRenderer({ r }: { r: Record<string, unknown> }) {
  // Substance / allergen name — most records carry only code.text or code.coding[].display.
  const allergen = conceptLabel(r.code);

  const clinicalStatus =
    str(nested(r, "clinicalStatus", "coding", "0", "code")) ||
    str(nested(r, "clinicalStatus", "coding", "0", "display")) ||
    str(nested(r, "clinicalStatus", "text")) ||
    "";
  const verificationStatus =
    str(nested(r, "verificationStatus", "coding", "0", "code")) ||
    str(nested(r, "verificationStatus", "coding", "0", "display")) ||
    str(nested(r, "verificationStatus", "text"));

  // type = allergy | intolerance; criticality = low | high | unable-to-assess.
  const type = str(r.type);
  const criticality = str(r.criticality);

  // Categories: food | medication | environment | biologic.
  const categories: string[] = [];
  for (const c of arr(r.category)) {
    const v = str(c);
    if (v) categories.push(v);
  }

  const onset = formatDate(r.onsetDateTime);
  const recordedDate = formatDate(r.recordedDate);

  // Reactions: collect manifestation labels, plus per-reaction description / substance / severity.
  const reactions = arr(r.reaction);
  const manifestationChips: string[] = [];
  const reactionDescriptions: string[] = [];
  const reactionSubstances: string[] = [];
  let reactionSeverity = "";
  for (const reaction of reactions) {
    const ro = obj(reaction);
    for (const m of arr(ro.manifestation)) {
      const label = conceptLabel(m);
      if (label && !manifestationChips.includes(label)) manifestationChips.push(label);
    }
    const desc = str(ro.description);
    if (desc && !reactionDescriptions.includes(desc)) reactionDescriptions.push(desc);
    const subst = conceptLabel(ro.substance);
    if (subst && !reactionSubstances.includes(subst)) reactionSubstances.push(subst);
    if (!reactionSeverity) reactionSeverity = str(ro.severity);
  }

  // Notes.
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const t = str(obj(n).text);
    if (t) notes.push(t);
  }

  // Record-TYPE accent (allergy hue) — neutral, not a severity good/bad judgement.
  const accentColor = "var(--record-allergy-dot)";

  // Neutral status/qualifier chips presented as plain text.
  const chips: Array<{ key: string; label: string }> = [];
  if (clinicalStatus) chips.push({ key: "clinical", label: clinicalStatus });
  if (verificationStatus && verificationStatus.toLowerCase() !== "confirmed") {
    chips.push({ key: "verification", label: verificationStatus });
  }
  if (type) chips.push({ key: "type", label: type });
  if (criticality) chips.push({ key: "criticality", label: `criticality: ${criticality}` });
  if (reactionSeverity) chips.push({ key: "severity", label: reactionSeverity });

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

      {/* Status / type / criticality / severity — neutral text, not color-coded good/bad */}
      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {chips.map((chip) => (
            <span
              key={chip.key}
              className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md"
              style={{
                backgroundColor: "var(--theme-bg-deep)",
                color: "var(--theme-text-muted)",
              }}
            >
              {chip.label}
            </span>
          ))}
        </div>
      )}

      {/* Category chips (food / medication / environment / biologic) */}
      {categories.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {categories.map((cat) => (
            <span
              key={cat}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-allergy-bg)",
                color: "var(--record-allergy-text)",
              }}
            >
              {cat}
            </span>
          ))}
        </div>
      )}

      {/* Reaction manifestation chips */}
      {manifestationChips.length > 0 && (
        <div>
          <span
            className="text-[11px] font-medium uppercase tracking-wide"
            style={{ color: "var(--theme-text-muted)" }}
          >
            Reaction
          </span>
          <div className="flex flex-wrap gap-1.5 mt-1">
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
        </div>
      )}

      <DetailRow label="Reaction Substance" value={reactionSubstances.join(", ")} />
      <DetailRow label="Reaction Description" value={reactionDescriptions.join("; ")} />

      <DetailRow label="Onset" value={onset} />
      <DetailRow label="Recorded" value={recordedDate} />

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
