"use client";

import React from "react";
import { DetailRow, SectionDivider, StatusBadge, BarRow, str, obj, arr, nested, formatDate, formatDateTime } from "./shared";

/** Pull a human-readable label from a CodeableConcept: text first, then first coding display. */
function conceptLabel(concept: unknown): string {
  const c = obj(concept);
  return str(c.text) || str(nested(c, "coding", "0", "display"));
}

/** Build a "value unit" string from a Quantity, omitting empty parts. */
function quantityText(q: unknown): string {
  const o = obj(q);
  const value = str(o.value);
  const unit = str(o.unit) || str(o.code);
  if (!value) return "";
  return unit ? `${value} ${unit}` : value;
}

export function MedicationRenderer({ r }: { r: Record<string, unknown> }) {
  // Name: works for both MedicationRequest and MedicationStatement.
  // Real CDA statements often carry only medicationReference (no inline name);
  // in that case there is no resolvable label and we render none rather than a broken ref.
  const name =
    conceptLabel(r.medicationCodeableConcept) ||
    conceptLabel(r.code) ||
    "";

  // Dosage: MedicationRequest -> dosageInstruction[], MedicationStatement -> dosage[].
  const dosageArr = arr(r.dosageInstruction).length ? arr(r.dosageInstruction) : arr(r.dosage);
  const firstDosage = obj(dosageArr[0]);
  const doseQty = quantityText(nested(firstDosage, "doseAndRate", "0", "doseQuantity"));
  const dosageText = str(firstDosage.text) || doseQty;
  const route =
    str(nested(firstDosage, "route", "text")) ||
    str(nested(firstDosage, "route", "coding", "0", "display"));
  const timing =
    str(nested(firstDosage, "timing", "code", "text")) ||
    str(nested(firstDosage, "timing", "code", "coding", "0", "display")) ||
    str(nested(firstDosage, "timing", "repeat", "frequency"));

  const status = str(r.status);
  const intent = str(r.intent);
  const category =
    str(nested(r, "category", "0", "text")) ||
    str(nested(r, "category", "0", "coding", "0", "display"));

  // Prescriber: only show a human-readable display, never a bare URN reference.
  const prescriber = str(nested(r, "requester", "display"));

  // Dates.
  const authoredOn = formatDateTime(r.authoredOn);
  // MedicationStatement timing — read effectiveDateTime AND effectivePeriod.
  const effectiveSingle = formatDate(r.effectiveDateTime);
  const effectiveStart = formatDate(
    nested(r, "effectivePeriod", "start") ?? nested(r, "dispenseRequest", "validityPeriod", "start"),
  );
  const effectiveEnd = formatDate(
    nested(r, "effectivePeriod", "end") ?? nested(r, "dispenseRequest", "validityPeriod", "end"),
  );
  const dateAsserted = formatDate(r.dateAsserted);

  // Dispense info.
  const dispenseQuantity = quantityText(nested(r, "dispenseRequest", "quantity"));
  const refills = str(nested(r, "dispenseRequest", "numberOfRepeatsAllowed"));
  const pharmacy = str(nested(r, "dispenseRequest", "performer", "display"));

  // Reasons recorded for the medication (verbatim, no interpretation).
  const reasons: string[] = [];
  for (const rc of arr(r.reasonCode)) {
    const label = conceptLabel(rc);
    if (label) reasons.push(label);
  }

  // Notes.
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const t = str(obj(n).text);
    if (t) notes.push(t);
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

      {/* Dosage strip */}
      {(dosageText || route || timing) && (
        <div
          className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 rounded-md text-xs"
          style={{
            backgroundColor: "var(--record-medication-bg)",
            color: "var(--record-medication-text)",
          }}
        >
          {dosageText && <span className="font-semibold">{dosageText}</span>}
          {route && <span>{route}</span>}
          {timing && <span>{timing}</span>}
        </div>
      )}

      {/* Reason(s) recorded */}
      {reasons.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {reasons.map((reason) => (
            <span
              key={reason}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-medication-bg)",
                color: "var(--record-medication-text)",
              }}
            >
              {reason}
            </span>
          ))}
        </div>
      )}

      <DetailRow label="Prescriber" value={prescriber} />
      <DetailRow label="Category" value={category} />

      {/* Effective date range (MedicationStatement period / Request validity) */}
      {(effectiveStart || effectiveEnd) && (
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          {effectiveStart && (
            <>
              <span style={{ color: "var(--theme-text-muted)" }}>Start</span>
              <span style={{ color: "var(--theme-text)" }}>{effectiveStart}</span>
            </>
          )}
          {effectiveEnd && (
            <>
              <span style={{ color: "var(--theme-text-muted)" }}>→ End</span>
              <span style={{ color: "var(--theme-text)" }}>{effectiveEnd}</span>
            </>
          )}
        </div>
      )}

      {/* Single effective date (MedicationStatement effectiveDateTime) */}
      {effectiveSingle && !effectiveStart && !effectiveEnd && (
        <DetailRow label="Effective" value={effectiveSingle} />
      )}

      {authoredOn && !effectiveStart && !effectiveEnd && (
        <DetailRow label="Authored" value={authoredOn} />
      )}
      <DetailRow label="Asserted" value={dateAsserted} />

      {/* Dispense info */}
      {(dispenseQuantity || refills || pharmacy) && (
        <BarRow
          items={[
            { label: "Quantity", value: dispenseQuantity },
            { label: "Refills", value: refills },
            { label: "Pharmacy", value: pharmacy },
          ]}
        />
      )}

      {(status || intent) && (
        <div className="flex items-center gap-2 pt-1">
          {status && <StatusBadge label={status} />}
          {intent && (
            <span
              className="text-[11px] px-1.5 py-0.5 rounded"
              style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-muted)" }}
            >
              {intent}
            </span>
          )}
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
