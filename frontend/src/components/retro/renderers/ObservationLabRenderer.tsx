"use client";

import React from "react";
import { DetailRow, str, obj, arr, nested, formatDate } from "./shared";
import { Gauge } from "@/components/retro/DataViz";

export function ObservationLabRenderer({ r }: { r: Record<string, unknown> }) {
  const testName =
    str(nested(r, "code", "text")) ||
    str(nested(r, "code", "coding", "0", "display")) ||
    "";
  const valueQuantity = obj(r.valueQuantity);
  const valueNum = str(valueQuantity.value);
  const valueUnit = str(valueQuantity.unit);
  const valueString = str(r.valueString);
  const displayValue = valueNum ? `${valueNum}` : valueString;
  const displayUnit = valueUnit || "";
  const effectiveDate = formatDate(r.effectiveDateTime);

  const refRangeArr = arr(r.referenceRange);
  const refRange = obj(refRangeArr[0]);
  const refLow = str(nested(refRange, "low", "value"));
  const refHigh = str(nested(refRange, "high", "value"));
  const refUnit = str(nested(refRange, "low", "unit")) || str(nested(refRange, "high", "unit"));
  const refText = str(refRange.text) || (refLow || refHigh ? `${refLow || "?"} - ${refHigh || "?"} ${refUnit}` : "");

  // The lab's own flag (e.g. "high", "low"). Presented as neutral text only —
  // this app organizes records, it does not interpret values as good/bad.
  const interpretationCode =
    str(nested(r, "interpretation", "0", "coding", "0", "code")) ||
    str(nested(r, "interpretation", "0", "text")) ||
    "";
  const interpUpper = interpretationCode.toUpperCase();
  let interpLabel = interpretationCode;
  if (interpUpper === "H" || interpUpper === "HH" || interpUpper === "HIGH") {
    interpLabel = interpUpper === "HH" ? "Critical High" : "High";
  } else if (interpUpper === "L" || interpUpper === "LL" || interpUpper === "LOW") {
    interpLabel = interpUpper === "LL" ? "Critical Low" : "Low";
  } else if (interpUpper === "N" || interpUpper === "NORMAL") {
    interpLabel = "Normal";
  }

  // Reference-range marker. The shaded band is the range as recorded by the
  // source (a neutral marker), not a "good zone".
  const numVal = parseFloat(valueNum);
  const numLow = parseFloat(refLow);
  const numHigh = parseFloat(refHigh);
  const showGauge = !isNaN(numVal) && !isNaN(numLow) && !isNaN(numHigh) && numHigh > numLow;

  return (
    <div className="space-y-3">
      {testName && (
        <p className="text-[13px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
          {testName}
        </p>
      )}

      {displayValue && (
        <div className="flex items-baseline gap-2">
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "24px",
              lineHeight: 1,
              color: "var(--theme-amber)",
            }}
          >
            {displayValue}
          </span>
          {displayUnit && (
            <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
              {displayUnit}
            </span>
          )}
          {interpretationCode && (
            <span className="text-xs font-medium ml-1" style={{ color: "var(--theme-text-muted)" }}>
              {interpLabel}
            </span>
          )}
        </div>
      )}

      {/* Reference range: neutral marker (value relative to the source's stated range) */}
      {showGauge && <Gauge value={numVal} low={numLow} high={numHigh} />}

      {!showGauge && refText && <DetailRow label="Reference" value={refText} mono />}

      {effectiveDate && <DetailRow label="Date" value={effectiveDate} />}
    </div>
  );
}
