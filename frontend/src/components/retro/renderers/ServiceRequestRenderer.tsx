"use client";

import React from "react";
import { ArrowRight } from "lucide-react";
import { DetailRow, StatusBadge, SectionDivider, str, obj, arr, nested, formatDate } from "./shared";

export function ServiceRequestRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";

  const status = str(r.status);
  const intent = str(r.intent);
  const priority = str(r.priority);
  const authoredOn = formatDate(r.authoredOn);

  const requester = str(nested(r, "requester", "display"));
  const performer =
    str(nested(r, "performer", "0", "display")) ||
    str(nested(r, "performer", "display")) ||
    "";

  const occurrenceDateTime = formatDate(r.occurrenceDateTime);
  const periodStart = formatDate(nested(r, "occurrencePeriod", "start"));
  const periodEnd = formatDate(nested(r, "occurrencePeriod", "end"));

  // Categories (category[] — display/text).
  const categories: string[] = [];
  for (const c of arr(r.category)) {
    const text = str(obj(c).text) || str(nested(obj(c), "coding", "0", "display"));
    if (text) categories.push(text);
  }

  // Reasons (reasonCode[] — display/text).
  const reasons: string[] = [];
  for (const rc of arr(r.reasonCode)) {
    const text = str(obj(rc).text) || str(nested(obj(rc), "coding", "0", "display"));
    if (text) reasons.push(text);
  }

  // Body sites (bodySite[] — display/text).
  const bodySites: string[] = [];
  for (const b of arr(r.bodySite)) {
    const text = str(obj(b).text) || str(nested(obj(b), "coding", "0", "display"));
    if (text) bodySites.push(text);
  }

  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const text = str(obj(n).text);
    if (text) notes.push(text);
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

      {/* Category chips */}
      {categories.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {categories.map((cat) => (
            <span
              key={cat}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-service_request-bg)",
                color: "var(--record-service_request-text)",
              }}
            >
              {cat}
            </span>
          ))}
        </div>
      )}

      {/* Visual referral flow: requester → performer */}
      {(requester || performer) && (
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--record-service_request-bg)" }}
        >
          {requester && (
            <span style={{ color: "var(--theme-text)" }}>{requester}</span>
          )}
          {requester && performer && (
            <ArrowRight size={14} style={{ color: "var(--record-service_request-text)" }} />
          )}
          {performer && (
            <span className="font-semibold" style={{ color: "var(--theme-text)" }}>{performer}</span>
          )}
        </div>
      )}

      {/* status | intent | priority badges */}
      {(status || intent || priority) && (
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          {status && <StatusBadge label={status} />}
          {intent && <StatusBadge label={intent} />}
          {priority && <StatusBadge label={priority} />}
        </div>
      )}

      <DetailRow label="Authored" value={authoredOn} />
      <DetailRow label="Scheduled" value={occurrenceDateTime} />
      {(periodStart || periodEnd) && (
        <DetailRow
          label="Period"
          value={`${periodStart || "?"}${periodEnd ? ` - ${periodEnd}` : ""}`}
        />
      )}

      {/* Reasons */}
      {reasons.length > 0 && (
        <div className="flex flex-col gap-0.5 py-1">
          <span
            className="text-[11px] font-medium uppercase tracking-wide"
            style={{ color: "var(--theme-text-muted)" }}
          >
            Reason
          </span>
          <div className="flex flex-wrap gap-1.5">
            {reasons.map((reason) => (
              <span
                key={reason}
                className="px-2 py-0.5 text-[11px] font-medium rounded"
                style={{
                  backgroundColor: "var(--theme-bg-deep)",
                  color: "var(--theme-text)",
                }}
              >
                {reason}
              </span>
            ))}
          </div>
        </div>
      )}

      <DetailRow label="Body site" value={bodySites.join(", ")} />

      {/* Notes */}
      {notes.length > 0 && (
        <>
          <SectionDivider />
          <div className="space-y-1.5">
            {notes.map((note, i) => (
              <div
                key={i}
                className="px-3 py-2 rounded-md text-xs whitespace-pre-wrap"
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
