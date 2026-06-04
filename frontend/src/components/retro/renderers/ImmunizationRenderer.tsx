"use client";

import React from "react";
import { DetailRow, StatusBadge, SectionDivider, str, obj, arr, nested, formatDate } from "./shared";

export function ImmunizationRenderer({ r }: { r: Record<string, unknown> }) {
  const name =
    str(nested(r, "vaccineCode", "text")) ||
    str(nested(r, "vaccineCode", "coding", "0", "display")) ||
    str(nested(r, "code", "text")) ||
    "";
  const date = formatDate(r.occurrenceDateTime ?? r.date);
  const dose = str(nested(r, "doseQuantity", "value"));
  const doseUnit =
    str(nested(r, "doseQuantity", "unit")) || str(nested(r, "doseQuantity", "code"));
  const route =
    str(nested(r, "route", "text")) || str(nested(r, "route", "coding", "0", "display"));
  const site =
    str(nested(r, "site", "text")) || str(nested(r, "site", "coding", "0", "display"));
  const manufacturer = str(nested(r, "manufacturer", "display"));
  const lotNumber = str(r.lotNumber);
  const expirationDate = formatDate(r.expirationDate);
  const status = str(r.status);

  // Dose / series (protocolApplied[]) — verbatim what the source recorded.
  const protocols = arr(r.protocolApplied);
  const doseSeries: string[] = [];
  for (const p of protocols) {
    const po = obj(p);
    const doseNum =
      str(po.doseNumberPositiveInt) ||
      str(po.doseNumberString) ||
      str(nested(po, "doseNumber", "value")) ||
      str(po.doseNumber);
    const series = str(po.series);
    const parts: string[] = [];
    if (doseNum) parts.push(`Dose ${doseNum}`);
    if (series) parts.push(series);
    if (parts.length) doseSeries.push(parts.join(" - "));
  }

  // Performers (performer[].actor.display). Real CDA/FHIR data often carries only a
  // reference, so display can be empty — chips render nothing when no display exists.
  const performers: string[] = [];
  for (const p of arr(r.performer)) {
    const actorDisplay = str(nested(obj(p), "actor", "display"));
    if (actorDisplay) performers.push(actorDisplay);
  }

  // Reasons (reasonCode[]).
  const reasons: string[] = [];
  for (const rc of arr(r.reasonCode)) {
    const text = str(obj(rc).text) || str(nested(obj(rc), "coding", "0", "display"));
    if (text) reasons.push(text);
  }

  // Source attribution.
  const primarySource =
    typeof r.primarySource === "boolean"
      ? r.primarySource
        ? "Primary source"
        : "Reported"
      : "";
  const reportOrigin =
    str(nested(r, "reportOrigin", "text")) ||
    str(nested(r, "reportOrigin", "coding", "0", "display"));

  const notes: string[] = [];
  for (const n of arr(r.note)) {
    const text = str(obj(n).text);
    if (text) notes.push(text);
  }

  return (
    <div className="space-y-3">
      {name && (
        <p
          className="text-sm font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {name}
        </p>
      )}

      {/* Dose number / series chips */}
      {doseSeries.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {doseSeries.map((d) => (
            <span
              key={d}
              className="px-2 py-0.5 text-[11px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-immunization-bg)",
                color: "var(--record-immunization-text)",
              }}
            >
              {d}
            </span>
          ))}
        </div>
      )}

      <DetailRow label="Date" value={date} />

      {/* Compact grid: dose | route | site */}
      {(dose || route || site) && (
        <div
          className="grid grid-cols-3 gap-2 px-3 py-2 rounded-md text-xs"
          style={{ backgroundColor: "var(--record-immunization-bg)" }}
        >
          {dose && (
            <div>
              <div className="text-[10px]" style={{ color: "var(--theme-text-muted)" }}>Dose</div>
              <div style={{ color: "var(--theme-text)" }}>{dose}{doseUnit ? ` ${doseUnit}` : ""}</div>
            </div>
          )}
          {route && (
            <div>
              <div className="text-[10px]" style={{ color: "var(--theme-text-muted)" }}>Route</div>
              <div style={{ color: "var(--theme-text)" }}>{route}</div>
            </div>
          )}
          {site && (
            <div>
              <div className="text-[10px]" style={{ color: "var(--theme-text-muted)" }}>Site</div>
              <div style={{ color: "var(--theme-text)" }}>{site}</div>
            </div>
          )}
        </div>
      )}

      {/* Manufacturer + lot + expiration in monospace */}
      {(manufacturer || lotNumber || expirationDate) && (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-xs">
          {manufacturer && <DetailRow label="Mfr" value={manufacturer} />}
          {lotNumber && (
            <div className="flex items-baseline gap-1.5">
              <span style={{ color: "var(--theme-text-muted)" }}>Lot</span>
              <span
                style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "var(--theme-text)" }}
              >
                {lotNumber}
              </span>
            </div>
          )}
          {expirationDate && (
            <div className="flex items-baseline gap-1.5">
              <span style={{ color: "var(--theme-text-muted)" }}>Expires</span>
              <span style={{ color: "var(--theme-text)" }}>{expirationDate}</span>
            </div>
          )}
        </div>
      )}

      <DetailRow label="Administered by" value={performers.join(", ")} />

      {/* Reason chips */}
      {reasons.length > 0 && (
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
      )}

      {(primarySource || reportOrigin) && (
        <DetailRow
          label="Source"
          value={[primarySource, reportOrigin].filter(Boolean).join(" - ")}
        />
      )}

      {status && (
        <div className="pt-1">
          <StatusBadge label={status} />
        </div>
      )}

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
