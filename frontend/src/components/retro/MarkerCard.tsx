"use client";

// A single recorded observation, shown exactly as the source has it — no
// judgment, no target, no good/bad coloring. Descriptive only.
//
// Selection is mechanical (every observation code, newest first); the card just
// surfaces the latest value, how it moved (sparkline / prior reading), and the
// source's own reference range ("per source"). It NEVER synthesizes a range —
// when the source omits one, no gauge renders.

import { Sparkline, Gauge } from "./DataViz";
import { fmtShort } from "@/lib/format-date";
import type { ObservationByCode } from "@/types/api";

interface MarkerCardProps {
  m: ObservationByCode;
  onSelect: (recordId: string) => void;
}

// A reading carries a real value only when it isn't empty and isn't just an echo
// of the observation's own name (some coded/narrative observations have no
// measured value, so the source row falls back to the code's display text).
function hasRealValue(value: number | string | null, name: string): boolean {
  if (value == null) return false;
  const s = String(value).trim();
  return s !== "" && s !== name;
}

export function MarkerCard({ m, onSelect }: MarkerCardProps) {
  const { latest, prior } = m;
  const hasSeries = m.series.length >= 2;
  const isNumeric = typeof latest.value === "number";
  const showValue = hasRealValue(latest.value, m.display);
  const showPrior = prior != null && hasRealValue(prior.value, m.display);
  const hasGauge = !hasSeries && isNumeric && latest.ref_low != null && latest.ref_high != null;
  const refText =
    latest.ref_low != null && latest.ref_high != null
      ? `Ref ${latest.ref_low}–${latest.ref_high}${latest.unit ? ` ${latest.unit}` : ""} · per source`
      : null;

  return (
    <button
      className={"marker" + (hasSeries || hasGauge ? "" : " no-spark")}
      onClick={() => onSelect(latest.id)}
    >
      <div className="marker-top">
        <span className="marker-name">{m.display}</span>
        {latest.date && <span className="marker-asof mono">as of {fmtShort(latest.date)}</span>}
      </div>

      {showValue && (
        <div className="marker-val">
          <span className="marker-num tnum">
            {latest.value}
            {latest.unit && <span className="marker-unit">{latest.unit}</span>}
          </span>
        </div>
      )}

      {hasSeries ? (
        <div className="marker-spark">
          <Sparkline points={m.series} />
        </div>
      ) : hasGauge ? (
        <div style={{ margin: "15px 0 8px" }}>
          <Gauge value={latest.value as number} low={latest.ref_low as number} high={latest.ref_high as number} />
        </div>
      ) : null}

      <div className="marker-foot">
        {showPrior && prior ? (
          <span>
            Previously {prior.value}
            {prior.unit ? ` ${prior.unit}` : ""} · {fmtShort(prior.date)}
          </span>
        ) : prior ? (
          <span>Previously recorded · {fmtShort(prior.date)}</span>
        ) : (
          <span>
            {m.count} reading{m.count === 1 ? "" : "s"} on file
          </span>
        )}
        {hasSeries && refText && (
          <>
            <br />
            {refText}
          </>
        )}
        {!hasSeries && !hasGauge && latest.source && (
          <>
            <br />
            Per {latest.source}
          </>
        )}
      </div>
    </button>
  );
}
