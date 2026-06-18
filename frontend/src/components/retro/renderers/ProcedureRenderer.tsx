"use client";

import React from "react";
import {
  DetailRow,
  StatusBadge,
  SectionDivider,
  str,
  obj,
  arr,
  nested,
  formatDate,
  performerNames,
} from "./shared";

/** Collect human text from a codeableConcept[] (text → coding display → coding code). */
function conceptTexts(items: unknown[]): string[] {
  const out: string[] = [];
  for (const item of items) {
    const cc = obj(item);
    const text =
      str(cc.text) ||
      str(nested(cc, "coding", "0", "display")) ||
      str(nested(cc, "coding", "0", "code"));
    if (text && !out.includes(text)) out.push(text);
  }
  return out;
}

export function ProcedureRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";
  const status = str(r.status);

  // Date — performedDateTime, or a performedPeriod range.
  const performedDateTime = formatDate(r.performedDateTime);
  const periodStart = formatDate(nested(r, "performedPeriod", "start"));
  const periodEnd = formatDate(nested(r, "performedPeriod", "end"));
  let date = performedDateTime;
  if (!date && periodStart) {
    date = periodEnd && periodEnd !== periodStart ? `${periodStart} → ${periodEnd}` : periodStart;
  }

  // Performers — list all actors (display preferred, readable reference fallback).
  const performers = performerNames(arr(r.performer), "actor");
  const asserter = str(nested(r, "asserter", "display"));

  // Reasons / body sites / outcomes — descriptive, what the source recorded.
  const reasons = conceptTexts(arr(r.reasonCode));
  const bodySites = conceptTexts(arr(r.bodySite));
  const outcome =
    str(nested(r, "outcome", "text")) ||
    str(nested(r, "outcome", "coding", "0", "display"));
  const complications = conceptTexts(arr(r.complication));
  const followUps = conceptTexts(arr(r.followUp));
  const location = str(nested(r, "location", "display"));

  // Notes — free text the source attached.
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const text = str(obj(n).text);
    if (text) notes.push(text);
  }

  const codeValue = str(nested(r, "code", "coding", "0", "code"));
  const codeSystem = str(nested(r, "code", "coding", "0", "system"));

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
      {status && (
        <div>
          <StatusBadge label={status} />
        </div>
      )}

      <DetailRow label="Date" value={date} />
      <DetailRow label={performers.length > 1 ? "Performers" : "Performer"} value={performers.join(", ")} />
      <DetailRow label="Asserter" value={asserter} />
      <DetailRow label="Location" value={location} />

      {/* Body site chips */}
      {bodySites.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {bodySites.map((chip) => (
            <span
              key={chip}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-procedure-bg)",
                color: "var(--record-procedure-text)",
              }}
            >
              {chip}
            </span>
          ))}
        </div>
      )}

      <DetailRow label={reasons.length > 1 ? "Reasons" : "Reason"} value={reasons.join("; ")} />
      <DetailRow label="Outcome" value={outcome} />
      <DetailRow
        label={complications.length > 1 ? "Complications" : "Complication"}
        value={complications.join("; ")}
      />
      <DetailRow label="Follow-up" value={followUps.join("; ")} />

      {/* Notes — verbatim source text */}
      {notes.length > 0 && (
        <>
          <SectionDivider />
          <div className="space-y-1.5">
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

      {/* Code footer strip */}
      {codeValue && (
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-md text-xs"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            fontFamily: "var(--font-mono)",
          }}
        >
          <span style={{ color: "var(--theme-text-muted)" }}>Code</span>
          <span style={{ color: "var(--theme-amber)" }}>{codeValue}</span>
          {codeSystem && (
            <span className="text-[11px]" style={{ color: "var(--theme-text-dim)" }}>
              ({codeSystem})
            </span>
          )}
        </div>
      )}
    </div>
  );
}
