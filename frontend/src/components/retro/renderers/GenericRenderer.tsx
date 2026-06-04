"use client";

import React from "react";
import { DetailRow, SectionDivider, str, obj, arr, nested, formatDate, formatDateTime } from "./shared";

// ---------------------------------------------------------------------------
// Generic fallback renderer.
//
// Used for any FHIR resource type that has no dedicated renderer. It surfaces
// the common R4 fields when present, using safe accessors so it never crashes
// on an unexpected shape. Descriptive only — it shows what the source recorded,
// no interpretation.
// ---------------------------------------------------------------------------

/** First non-empty value from a CodeableConcept (text, then any coding display). */
function codeableText(val: unknown): string {
  const c = obj(val);
  const text = str(c.text);
  if (text) return text;
  const codings = arr(c.coding);
  for (const coding of codings) {
    const display = str(obj(coding).display);
    if (display) return display;
  }
  // Fall back to a bare coding code if that's all we have.
  for (const coding of codings) {
    const code = str(obj(coding).code);
    if (code) return code;
  }
  return "";
}

/** Resolve the first date-like field that's present, formatted for display. */
function firstDate(
  r: Record<string, unknown>,
): { label: string; value: string } | null {
  // Ordered so the most specific/clinically-relevant date wins.
  const dateFields: Array<[string, string]> = [
    ["effectiveDateTime", "Effective"],
    ["occurrenceDateTime", "Occurrence"],
    ["performedDateTime", "Performed"],
    ["authoredOn", "Authored"],
    ["recordedDate", "Recorded"],
    ["issued", "Issued"],
    ["created", "Created"],
    ["date", "Date"],
    ["sent", "Sent"],
    ["onsetDateTime", "Onset"],
  ];
  for (const [field, label] of dateFields) {
    const formatted = formatDateTime(r[field]);
    if (formatted) return { label, value: formatted };
  }
  // Period (start/end) — present on encounters, service requests, etc.
  const periodStart = formatDate(nested(r, "period", "start"));
  const periodEnd = formatDate(nested(r, "period", "end"));
  if (periodStart || periodEnd) {
    const value = periodEnd && periodEnd !== periodStart
      ? `${periodStart || "?"} → ${periodEnd}`
      : periodStart || periodEnd;
    return { label: "Period", value };
  }
  return null;
}

/** Best-effort "who" line: performer / requester / author / actor display. */
function firstActor(r: Record<string, unknown>): { label: string; value: string } | null {
  const candidates: Array<[unknown, string]> = [
    [r.performer, "Performer"],
    [r.requester, "Requester"],
    [r.author, "Author"],
    [r.recorder, "Recorder"],
    [r.asserter, "Asserter"],
  ];
  for (const [raw, label] of candidates) {
    // Field may be a single reference or an array of them (possibly wrapped in
    // {actor: {...}} as in Procedure.performer).
    const entries = Array.isArray(raw) ? raw : raw != null ? [raw] : [];
    const names: string[] = [];
    for (const entry of entries) {
      const e = obj(entry);
      const display =
        str(e.display) ||
        str(nested(e, "actor", "display")) ||
        str(nested(e, "individual", "display")) ||
        str(nested(e, "agent", "display"));
      if (display) names.push(display);
    }
    if (names.length > 0) return { label, value: names.join(", ") };
  }
  return null;
}

/** valueQuantity / valueString / valueCodeableConcept / valueBoolean / valueInteger. */
function valueLine(r: Record<string, unknown>): string {
  const vq = obj(r.valueQuantity);
  const qtyValue = str(vq.value);
  if (qtyValue) {
    const unit = str(vq.unit) || str(vq.code);
    return unit ? `${qtyValue} ${unit}` : qtyValue;
  }
  const vs = str(r.valueString);
  if (vs) return vs;
  const vcc = codeableText(r.valueCodeableConcept);
  if (vcc) return vcc;
  if (typeof r.valueBoolean === "boolean") return r.valueBoolean ? "Yes" : "No";
  const vi = str(r.valueInteger);
  if (vi) return vi;
  return "";
}

export function GenericRenderer({ r }: { r: Record<string, unknown> }) {
  const title =
    codeableText(r.code) || str(r.resourceType) || "Record";

  const status = str(r.status);
  const date = firstDate(r);
  const category = codeableText(arr(r.category)[0] ?? r.category);
  const actor = firstActor(r);
  const value = valueLine(r);

  // Notes: collect every note[].text.
  const notes = arr(r.note)
    .map((n) => str(obj(n).text))
    .filter(Boolean);

  return (
    <div className="space-y-3">
      <p
        className="text-base font-semibold"
        style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
      >
        {title}
      </p>

      {status && <DetailRow label="Status" value={status} />}
      {date && <DetailRow label={date.label} value={date.value} />}
      {value && <DetailRow label="Value" value={value} />}
      {category && <DetailRow label="Category" value={category} />}
      {actor && <DetailRow label={actor.label} value={actor.value} />}

      {notes.length > 0 && (
        <>
          <SectionDivider />
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
        </>
      )}
    </div>
  );
}
