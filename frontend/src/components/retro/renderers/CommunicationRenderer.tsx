"use client";

import React from "react";
import { DetailRow, StatusBadge, str, arr, obj, nested, formatDateTime } from "./shared";

// CodeableConcept text: prefer .text, else first coding .display.
function codeableText(val: unknown): string {
  const c = obj(val);
  return str(c.text) || str(obj(arr(c.coding)[0]).display);
}

export function CommunicationRenderer({ r }: { r: Record<string, unknown> }) {
  const status = str(r.status);
  const sent = formatDateTime(r.sent);
  const received = formatDateTime(r.received);
  const sender = str(nested(r, "sender", "display"));
  const topic = codeableText(r.topic);

  // payload[] — contentString or contentAttachment.title
  const payloads = arr(r.payload);
  const payloadTexts: string[] = [];
  for (const p of payloads) {
    const pObj = obj(p);
    const text = str(pObj.contentString) || str(nested(pObj, "contentAttachment", "title"));
    if (text) payloadTexts.push(text);
  }

  // category[] — display/text
  const categoryTexts: string[] = [];
  for (const cat of arr(r.category)) {
    const text = codeableText(cat);
    if (text) categoryTexts.push(text);
  }

  // recipient[].display
  const recipients: string[] = [];
  for (const rec of arr(r.recipient)) {
    const label = str(obj(rec).display) || str(obj(rec).reference);
    if (label) recipients.push(label);
  }

  // medium[] — display/text
  const mediums: string[] = [];
  for (const m of arr(r.medium)) {
    const label = codeableText(m);
    if (label) mediums.push(label);
  }

  // note[].text
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const text = str(obj(n).text);
    if (text) notes.push(text);
  }

  return (
    <div className="space-y-3">
      {/* Category badges */}
      {categoryTexts.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {categoryTexts.map((cat, i) => (
            <span
              key={`${cat}-${i}`}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-communication-bg)",
                color: "var(--record-communication-text)",
              }}
            >
              {cat}
            </span>
          ))}
        </div>
      )}

      {topic && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {topic}
        </p>
      )}

      {/* Message bubbles */}
      {payloadTexts.map((text, i) => (
        <div
          key={i}
          className="px-3 py-2 rounded-lg text-xs"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            color: "var(--theme-text)",
            borderLeft: "2px solid var(--record-communication-dot)",
          }}
        >
          {text}
        </div>
      ))}

      {payloadTexts.length === 0 && (
        <p className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
          No message content
        </p>
      )}

      <div className="flex items-center gap-3">
        {status && <StatusBadge label={status} />}
      </div>

      <DetailRow label="From" value={sender} />
      {recipients.length > 0 && (
        <DetailRow label={recipients.length > 1 ? "Recipients" : "Recipient"} value={recipients.join(", ")} />
      )}
      <DetailRow label="Sent" value={sent} />
      <DetailRow label="Received" value={received} />
      {mediums.length > 0 && <DetailRow label="Medium" value={mediums.join(", ")} />}

      {/* Notes */}
      {notes.map((text, i) => (
        <p key={i} className="text-xs italic" style={{ color: "var(--theme-text-dim)" }}>
          {text}
        </p>
      ))}
    </div>
  );
}
