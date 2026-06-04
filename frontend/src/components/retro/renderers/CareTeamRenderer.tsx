"use client";

import React from "react";
import { DetailRow, StatusBadge, str, arr, obj, nested, formatDate } from "./shared";

// CodeableConcept text: prefer .text, else first coding .display.
function codeableText(val: unknown): string {
  const c = obj(val);
  return str(c.text) || str(obj(arr(c.coding)[0]).display);
}

export function CareTeamRenderer({ r }: { r: Record<string, unknown> }) {
  const name = str(r.name);
  const status = str(r.status);
  const periodStart = formatDate(nested(r, "period", "start"));
  const periodEnd = formatDate(nested(r, "period", "end"));

  // participant[] — member.display + role label (role is an array of CodeableConcept)
  const members = arr(r.participant ?? r.member);
  const memberEntries: Array<{ name: string; role: string }> = [];
  for (const m of members) {
    const mObj = obj(m);
    const memberName =
      str(nested(mObj, "member", "display")) || str(nested(mObj, "actor", "display"));
    // role is array<CodeableConcept>; fall back to single-object shape just in case.
    const roleArr = arr(mObj.role);
    const role =
      codeableText(roleArr[0]) ||
      codeableText(mObj.role) ||
      "";
    if (memberName) memberEntries.push({ name: memberName, role });
  }

  // reasonCode[] — why the team exists (display/text)
  const reasons: string[] = [];
  for (const rc of arr(r.reasonCode)) {
    const label = codeableText(rc);
    if (label) reasons.push(label);
  }

  // managingOrganization — Reference or array of References (.display)
  const orgs: string[] = [];
  const mgmt = r.managingOrganization;
  const mgmtList = Array.isArray(mgmt) ? mgmt : mgmt ? [mgmt] : [];
  for (const o of mgmtList) {
    const label = str(obj(o).display) || str(obj(o).reference);
    if (label) orgs.push(label);
  }

  // note[].text
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

      <div className="flex flex-wrap items-center gap-2">
        {status && <StatusBadge label={status} />}
        {(periodStart || periodEnd) && (
          <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
            {periodStart && periodEnd
              ? `${periodStart} → ${periodEnd}`
              : periodStart
                ? `Since ${periodStart}`
                : `Until ${periodEnd}`}
          </span>
        )}
      </div>

      {/* Members list */}
      {memberEntries.length > 0 && (
        <div
          className="rounded-md overflow-hidden"
          style={{ border: "1px solid var(--theme-border)" }}
        >
          {memberEntries.map((member, i) => (
            <div
              key={i}
              className="flex items-center justify-between gap-2 px-3 py-2 text-xs"
              style={{
                borderBottom: i < memberEntries.length - 1 ? "1px solid var(--theme-border)" : "none",
              }}
            >
              <span style={{ color: "var(--theme-text)" }}>{member.name}</span>
              {member.role && (
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap"
                  style={{
                    backgroundColor: "var(--record-care_team-bg)",
                    color: "var(--record-care_team-text)",
                  }}
                >
                  {member.role}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Managing organization */}
      {orgs.length > 0 && (
        <DetailRow label={orgs.length > 1 ? "Organizations" : "Organization"} value={orgs.join(", ")} />
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

      {/* Notes */}
      {notes.map((text, i) => (
        <p key={i} className="text-xs italic" style={{ color: "var(--theme-text-dim)" }}>
          {text}
        </p>
      ))}
    </div>
  );
}
