"use client";

import React from "react";
import { DetailRow, SectionDivider, str, obj, arr, nested, formatDate, formatDateTime } from "./shared";

// ---------------------------------------------------------------------------
// QuestionnaireResponse renderer.
//
// Surfaces the recorded questionnaire, who answered it and when, then the Q&A
// list. Items can nest (item[].item) — we recurse one level (and guard deeper)
// so grouped questions still render. Descriptive only: each question's recorded
// text with the recorded answer value(s); no interpretation.
// ---------------------------------------------------------------------------

/** Render a single answer entry's value, covering the common value[x] types. */
function answerText(answer: unknown): string {
  const a = obj(answer);
  const vs = str(a.valueString);
  if (vs) return vs;
  if (typeof a.valueBoolean === "boolean") return a.valueBoolean ? "Yes" : "No";
  const vi = str(a.valueInteger);
  if (vi) return vi;
  const vd = str(a.valueDecimal);
  if (vd) return vd;
  const coding = str(nested(a, "valueCoding", "display")) || str(nested(a, "valueCoding", "code"));
  if (coding) return coding;
  const dt = formatDateTime(a.valueDateTime);
  if (dt) return dt;
  const date = formatDate(a.valueDate);
  if (date) return date;
  const vt = str(a.valueTime);
  if (vt) return vt;
  const vq = obj(a.valueQuantity);
  const qty = str(vq.value);
  if (qty) {
    const unit = str(vq.unit) || str(vq.code);
    return unit ? `${qty} ${unit}` : qty;
  }
  const uri = str(a.valueUri);
  if (uri) return uri;
  const ref = str(nested(a, "valueReference", "display"));
  if (ref) return ref;
  return "";
}

/** Join all answer values on an item into a single display string. */
function answersForItem(item: Record<string, unknown>): string {
  return arr(item.answer)
    .map(answerText)
    .filter(Boolean)
    .join(", ");
}

interface QItemProps {
  item: Record<string, unknown>;
  depth: number;
}

function QItem({ item, depth }: QItemProps) {
  const question = str(item.text) || str(item.linkId) || "Question";
  const answer = answersForItem(item);
  const children = arr(item.item);

  return (
    <div
      className="text-xs"
      style={depth > 0 ? { marginLeft: 12, paddingLeft: 8, borderLeft: "1px solid var(--theme-border)" } : undefined}
    >
      <div className="flex flex-col gap-0.5 py-1">
        <span className="font-medium" style={{ color: "var(--theme-text-muted)" }}>
          {question}
        </span>
        <span style={{ color: "var(--theme-text)" }}>
          {answer || "—"}
        </span>
      </div>
      {/* Recurse for grouped/nested questions (guarded to a couple of levels). */}
      {depth < 3 &&
        children.map((child, i) => (
          <QItem key={i} item={obj(child)} depth={depth + 1} />
        ))}
    </div>
  );
}

export function QuestionnaireResponseRenderer({ r }: { r: Record<string, unknown> }) {
  // questionnaire may be a canonical URL string or a Reference object.
  const questionnaire =
    str(r.questionnaire) ||
    str(nested(r, "questionnaire", "reference")) ||
    str(nested(r, "questionnaire", "display"));
  const status = str(r.status);
  const authored = formatDateTime(r.authored);
  const author = str(nested(r, "author", "display"));
  const subject = str(nested(r, "subject", "display"));
  const source = str(nested(r, "source", "display"));

  const items = arr(r.item);

  return (
    <div className="space-y-3">
      <p
        className="text-base font-semibold"
        style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
      >
        Questionnaire Response
      </p>

      {status && <DetailRow label="Status" value={status} />}
      <DetailRow label="Questionnaire" value={questionnaire} mono />
      <DetailRow label="Authored" value={authored} />
      <DetailRow label="Author" value={author} />
      <DetailRow label="Subject" value={subject} />
      <DetailRow label="Source" value={source} />

      {items.length > 0 ? (
        <>
          <SectionDivider />
          <div
            className="rounded-md overflow-hidden"
            style={{ border: "1px solid var(--theme-border)", backgroundColor: "var(--theme-bg-deep)" }}
          >
            {items.map((item, i) => (
              <div
                key={i}
                className="px-3 py-1.5"
                style={{
                  borderBottom: i < items.length - 1 ? "1px solid var(--theme-border)" : "none",
                }}
              >
                <QItem item={obj(item)} depth={0} />
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
          No recorded answers
        </p>
      )}
    </div>
  );
}
