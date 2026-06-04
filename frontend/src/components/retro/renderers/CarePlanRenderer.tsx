"use client";

import React from "react";
import { CheckCircle2, Circle, XCircle } from "lucide-react";
import { DetailRow, StatusBadge, str, arr, obj, nested, formatDate } from "./shared";

const ACTIVITY_STATUS_ICONS: Record<string, React.ReactNode> = {
  completed: <CheckCircle2 size={12} style={{ color: "var(--theme-sage)" }} />,
  "in-progress": <Circle size={12} style={{ color: "var(--theme-ochre)" }} />,
  scheduled: <Circle size={12} style={{ color: "var(--theme-text-dim)" }} />,
  cancelled: <XCircle size={12} style={{ color: "var(--theme-terracotta)" }} />,
  "not-started": <Circle size={12} style={{ color: "var(--theme-text-muted)" }} />,
};

// CodeableConcept text: prefer .text, else first coding .display.
function codeableText(val: unknown): string {
  const c = obj(val);
  return str(c.text) || str(obj(arr(c.coding)[0]).display);
}

export function CarePlanRenderer({ r }: { r: Record<string, unknown> }) {
  const title = str(r.title) || codeableText(r.code);
  const description = str(r.description);
  const status = str(r.status);
  const intent = str(r.intent);
  const created = formatDate(r.created);
  const periodStart = formatDate(nested(r, "period", "start"));
  const periodEnd = formatDate(nested(r, "period", "end"));

  // category[] — display/text labels
  const categoryLabels: string[] = [];
  for (const cat of arr(r.category)) {
    const label = codeableText(cat);
    if (label) categoryLabels.push(label);
  }

  // addresses[] — conditions this plan addresses (display or text)
  const addresses: string[] = [];
  for (const a of arr(r.addresses)) {
    const aObj = obj(a);
    const label = str(aObj.display) || codeableText(a) || str(aObj.reference);
    if (label) addresses.push(label);
  }

  // goal[] — descriptive references / displays
  const goals: string[] = [];
  for (const g of arr(r.goal)) {
    const gObj = obj(g);
    const label = str(gObj.display) || str(gObj.reference);
    if (label) goals.push(label);
  }

  // note[].text
  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const text = str(obj(n).text);
    if (text) notes.push(text);
  }

  const activities = arr(r.activity);

  return (
    <div className="space-y-3">
      {title && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {title}
        </p>
      )}

      {/* Category badges */}
      {categoryLabels.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {categoryLabels.map((cat, i) => (
            <span
              key={`${cat}-${i}`}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-care_plan-bg)",
                color: "var(--record-care_plan-text)",
              }}
            >
              {cat}
            </span>
          ))}
        </div>
      )}

      {description && (
        <p className="text-xs" style={{ color: "var(--theme-text-dim)" }}>
          {description}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {status && <StatusBadge label={status} />}
        {intent && (
          <span
            className="text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded"
            style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-muted)" }}
          >
            {intent}
          </span>
        )}
        {created && (
          <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
            Created {created}
          </span>
        )}
      </div>

      {/* Period range */}
      {(periodStart || periodEnd) && (
        <DetailRow
          label="Period"
          value={`${periodStart || "?"}${periodEnd ? ` → ${periodEnd}` : ""}`}
        />
      )}

      {/* Conditions addressed */}
      {addresses.length > 0 && (
        <div className="space-y-1">
          <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--theme-text-muted)" }}>
            Addresses
          </span>
          <div className="flex flex-wrap gap-1.5">
            {addresses.map((a, i) => (
              <span
                key={`${a}-${i}`}
                className="px-2 py-0.5 text-[11px] rounded"
                style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text)" }}
              >
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Activity checklist */}
      {activities.length > 0 && (
        <div
          className="rounded-md overflow-hidden"
          style={{ border: "1px solid var(--theme-border)" }}
        >
          {activities.map((activity, i) => {
            const aObj = obj(activity);
            const detail = obj(aObj.detail);
            const activityStatus = str(detail.status);
            const activityDesc =
              codeableText(detail.code) ||
              str(detail.description) ||
              str(nested(aObj, "reference", "display")) ||
              str(nested(aObj, "reference", "reference")) ||
              `Activity ${i + 1}`;
            const icon =
              ACTIVITY_STATUS_ICONS[activityStatus] ?? (
                <Circle size={12} style={{ color: "var(--theme-text-muted)" }} />
              );

            return (
              <div
                key={i}
                className="flex items-center gap-2 px-3 py-2 text-xs"
                style={{
                  borderBottom: i < activities.length - 1 ? "1px solid var(--theme-border)" : "none",
                  backgroundColor: activityStatus === "completed" ? "var(--theme-bg-deep)" : "transparent",
                }}
              >
                {icon}
                <span
                  className="flex-1"
                  style={{
                    color: activityStatus === "completed" ? "var(--theme-text-muted)" : "var(--theme-text)",
                    textDecoration: activityStatus === "completed" ? "line-through" : "none",
                  }}
                >
                  {activityDesc}
                </span>
                {activityStatus && (
                  <span className="text-[10px]" style={{ color: "var(--theme-text-muted)" }}>
                    {activityStatus}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Goals */}
      {goals.length > 0 && (
        <div className="space-y-1">
          <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--theme-text-muted)" }}>
            Goals
          </span>
          {goals.map((g, i) => (
            <p key={`${g}-${i}`} className="text-xs" style={{ color: "var(--theme-text)" }}>
              {g}
            </p>
          ))}
        </div>
      )}

      {/* Notes */}
      {notes.map((text, i) => (
        <p key={i} className="text-xs italic" style={{ color: "var(--theme-text-dim)" }}>
          {text}
        </p>
      ))}
    </div>
  );
}
