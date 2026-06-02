"use client";

// Neutral data-viz atoms used across the redesign. They present recorded values
// factually — no good/bad coloring, no goal/target framing (the app organizes
// records, it does not interpret them).

interface GaugeProps {
  value: number;
  low: number;
  high: number;
}

/** Reference-range marker: shows where a value sits relative to the lab's own
 *  stated range. The shaded band is the reference range as recorded by the
 *  source — it is not a "good zone". */
export function Gauge({ value, low, high }: GaugeProps) {
  if (typeof value !== "number" || low == null || high == null) return null;
  const span = high - low || 1;
  const min = low - span * 0.7;
  const max = high + span * 0.7;
  const range = max - min;
  const pct = (v: number) => Math.max(2, Math.min(98, ((v - min) / range) * 100));
  return (
    <div>
      <div className="gauge">
        <div className="gauge-band" style={{ left: `${pct(low)}%`, width: `${pct(high) - pct(low)}%` }} />
        <div className="gauge-pin" style={{ left: `${pct(value)}%` }} />
      </div>
      <div className="gauge-sc">
        <span>{low}</span>
        <span>reference range (per source)</span>
        <span>{high}</span>
      </div>
    </div>
  );
}

interface SparklinePoint {
  value: number;
  date?: string | null;
}

interface SparklineProps {
  points: SparklinePoint[];
  height?: number;
}

/** Simple value-over-time line for a recurring observation. Descriptive only. */
export function Sparkline({ points, height = 70 }: SparklineProps) {
  if (!points || points.length < 2) return null;
  const W = 420;
  const H = height;
  const p = 10;
  const xs = points.map((_, i) => p + (i * (W - 2 * p)) / (points.length - 1));
  const vals = points.map((d) => d.value);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const y = (v: number) => H - p - ((v - lo) / (hi - lo || 1)) * (H - 2 * p);
  const line = xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y(vals[i]).toFixed(1)}`).join(" ");
  const stroke = "var(--primary)";
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="mt-spark" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={stroke} stopOpacity="0.18" />
          <stop offset="1" stopColor={stroke} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line} L ${xs[xs.length - 1]} ${H} L ${xs[0]} ${H} Z`} fill="url(#mt-spark)" />
      <path d={line} fill="none" stroke={stroke} strokeWidth="2.2" />
      {xs.map((x, i) => (
        <circle
          key={i}
          cx={x}
          cy={y(vals[i])}
          r={i === xs.length - 1 ? 4 : 2.6}
          fill={i === xs.length - 1 ? stroke : "var(--card)"}
          stroke={stroke}
          strokeWidth="1.6"
        />
      ))}
    </svg>
  );
}
