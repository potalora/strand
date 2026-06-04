"use client";

import React from "react";
import { DetailRow, str, arr, obj, nested, formatDate, formatDateTime } from "./shared";

const STATUS_COLORS: Record<string, string> = {
  booked: "var(--theme-sage)",
  cancelled: "var(--theme-terracotta)",
  fulfilled: "var(--theme-ochre)",
  pending: "var(--theme-text-dim)",
  proposed: "var(--theme-text-dim)",
  noshow: "var(--theme-terracotta)",
};

// CodeableConcept text: prefer .text, else first coding .display.
function codeableText(val: unknown): string {
  const c = obj(val);
  return str(c.text) || str(obj(arr(c.coding)[0]).display);
}

export function AppointmentRenderer({ r }: { r: Record<string, unknown> }) {
  const description = str(r.description);
  const status = str(r.status);
  const start = formatDateTime(r.start);
  const end = formatDateTime(r.end);
  const minutesDuration = str(r.minutesDuration);
  const priority = str(r.priority);
  const comment = str(r.comment);
  const patientInstruction = str(r.patientInstruction);
  const created = formatDate(r.created);
  const appointmentType = codeableText(r.appointmentType);

  // serviceType[] / serviceCategory[] — display/text
  const serviceTypes: string[] = [];
  for (const s of arr(r.serviceType)) {
    const label = codeableText(s);
    if (label) serviceTypes.push(label);
  }
  const serviceCategories: string[] = [];
  for (const s of arr(r.serviceCategory)) {
    const label = codeableText(s);
    if (label) serviceCategories.push(label);
  }

  // reasonCode[] — display/text
  const reasons: string[] = [];
  for (const rc of arr(r.reasonCode)) {
    const label = codeableText(rc);
    if (label) reasons.push(label);
  }

  // participant[] — actor.display + participation status (type is array<CodeableConcept>)
  const participants = arr(r.participant);
  const participantEntries: Array<{ name: string; role: string; status: string }> = [];
  for (const p of participants) {
    const pObj = obj(p);
    const name = str(nested(pObj, "actor", "display"));
    const typeArr = arr(pObj.type);
    const role = codeableText(typeArr[0]) || codeableText(pObj.type) || "";
    const pStatus = str(pObj.status);
    if (name) participantEntries.push({ name, role, status: pStatus });
  }

  const statusColor = STATUS_COLORS[status.toLowerCase()] ?? "var(--theme-bg-deep)";

  return (
    <div className="space-y-3">
      {description && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {description}
        </p>
      )}

      {/* Status + appointment type badges */}
      <div className="flex flex-wrap items-center gap-2">
        {status && (
          <span
            className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md"
            style={{
              backgroundColor: statusColor,
              color: status.toLowerCase() === "booked" ? "var(--theme-bg-deep)" : "var(--theme-text)",
            }}
          >
            {status}
          </span>
        )}
        {appointmentType && (
          <span
            className="px-2 py-0.5 text-[11px] font-medium rounded"
            style={{
              backgroundColor: "var(--record-appointment-bg)",
              color: "var(--record-appointment-text)",
            }}
          >
            {appointmentType}
          </span>
        )}
        {priority && (
          <span
            className="text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded"
            style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text-muted)" }}
          >
            Priority {priority}
          </span>
        )}
      </div>

      {/* Time display */}
      {start && (
        <div
          className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--record-appointment-bg)" }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "16px",
              color: "var(--record-appointment-text)",
            }}
          >
            {start}
          </span>
          {end && (
            <>
              <span style={{ color: "var(--theme-text-muted)" }}>→</span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "16px",
                  color: "var(--record-appointment-text)",
                }}
              >
                {end}
              </span>
            </>
          )}
          {minutesDuration && (
            <span style={{ color: "var(--theme-text-muted)" }}>({minutesDuration} min)</span>
          )}
        </div>
      )}

      {/* Service type / category badges */}
      {(serviceTypes.length > 0 || serviceCategories.length > 0) && (
        <div className="flex flex-wrap gap-1.5">
          {[...serviceCategories, ...serviceTypes].map((s, i) => (
            <span
              key={`${s}-${i}`}
              className="px-2 py-0.5 text-[11px] rounded"
              style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text)" }}
            >
              {s}
            </span>
          ))}
        </div>
      )}

      {/* Reasons */}
      {reasons.length > 0 && (
        <div className="space-y-1">
          <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--theme-text-muted)" }}>
            Reason
          </span>
          <div className="flex flex-wrap gap-1.5">
            {reasons.map((reason, i) => (
              <span
                key={`${reason}-${i}`}
                className="px-2 py-0.5 text-[11px] rounded"
                style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text)" }}
              >
                {reason}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Participants */}
      {participantEntries.length > 0 && (
        <div className="space-y-1">
          <span className="text-[10px] font-medium uppercase tracking-wide" style={{ color: "var(--theme-text-muted)" }}>
            Participants
          </span>
          {participantEntries.map((p, i) => (
            <div key={i} className="flex items-baseline justify-between gap-2 py-0.5">
              <span className="text-xs" style={{ color: "var(--theme-text)" }}>
                {p.name}
                {p.status && (
                  <span className="ml-1.5 text-[10px]" style={{ color: "var(--theme-text-muted)" }}>
                    ({p.status})
                  </span>
                )}
              </span>
              {p.role && (
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap"
                  style={{
                    backgroundColor: "var(--theme-bg-deep)",
                    color: "var(--theme-text-muted)",
                  }}
                >
                  {p.role}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Patient instruction */}
      {patientInstruction && (
        <div
          className="px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--theme-bg-deep)", color: "var(--theme-text)" }}
        >
          {patientInstruction}
        </div>
      )}

      {/* Comment */}
      {comment && (
        <p className="text-xs italic" style={{ color: "var(--theme-text-dim)" }}>
          {comment}
        </p>
      )}

      <DetailRow label="Created" value={created} />
    </div>
  );
}
