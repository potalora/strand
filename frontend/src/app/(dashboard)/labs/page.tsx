"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { LabItem } from "@/types/api";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";
import { Gauge } from "@/components/retro/DataViz";

const PAGE_SIZE = 20;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = (s: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

// The lab's OWN interpretation flag, surfaced as neutral text. This is the
// source's recorded annotation — never recolored as good/bad by this app.
const INTERPRETATION_LABELS: Record<string, string> = {
  H: "flagged high (per source)",
  HH: "flagged critical high (per source)",
  L: "flagged low (per source)",
  LL: "flagged critical low (per source)",
  A: "flagged abnormal (per source)",
  AA: "flagged critical abnormal (per source)",
  N: "within range (per source)",
};

function interpretationText(code: string): string | null {
  if (!code) return null;
  return INTERPRETATION_LABELS[code.toUpperCase()] || `flag: ${code} (per source)`;
}

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

interface LabResponse {
  items: LabItem[];
  total: number;
  page: number;
  page_size: number;
}

export default function LabsPage() {
  const router = useRouter();
  const [items, setItems] = useState<LabItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .get<LabResponse>(`/dashboard/labs?page=${page}&page_size=${PAGE_SIZE}`)
      .then((data) => {
        setItems(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => {
        setItems([]);
        setTotal(0);
      })
      .finally(() => setLoading(false));
  }, [page]);

  const dot = (RECORD_TYPE_COLORS.observation ?? DEFAULT_RECORD_COLOR).dot;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">Labs &amp; Vitals</p>
          <h1 className="h1 display">Lab results &amp; observations</h1>
        </div>
        <SecureChip />
      </div>

      <div className="card-surface pad">
        {loading ? (
          <RetroLoadingState text="Loading observations" />
        ) : items.length === 0 ? (
          <div className="py-8 text-center">
            <p className="muted text-sm mb-3">No observations on file.</p>
            <button className="btn" onClick={() => router.push("/upload")}>
              Upload records
            </button>
          </div>
        ) : (
          <>
            <div className="card-h">
              <h3 className="sec-title">{total} observation{total === 1 ? "" : "s"}</h3>
            </div>
            <div>
              {items.map((item) => {
                const hasValue = item.value !== null && item.value !== undefined && item.value !== "";
                const valueStr = hasValue
                  ? `${item.value}${item.unit ? ` ${item.unit}` : ""}`
                  : null;
                const flag = interpretationText(item.interpretation);
                const numericValue = typeof item.value === "number" ? item.value : null;
                const showGauge =
                  numericValue !== null &&
                  item.reference_low !== null &&
                  item.reference_high !== null;
                const subParts = [valueStr, flag].filter(Boolean).join(" · ");
                return (
                  <div
                    key={item.id}
                    style={{ borderBottom: "1px solid var(--border)" }}
                  >
                    <button
                      className="lrow"
                      onClick={() => setSelected(item.id)}
                      style={{
                        width: "100%",
                        background: "transparent",
                        border: 0,
                        borderBottom: 0,
                        cursor: "pointer",
                        textAlign: "left",
                        paddingBottom: showGauge ? 4 : undefined,
                      }}
                    >
                      <span className="dot" style={{ background: dot }} />
                      <span className="lrow-main">
                        <span className="lrow-title">{item.display_text}</span>
                        {subParts && <span className="lrow-sub muted">{subParts}</span>}
                      </span>
                      <span className="lrow-meta tnum">{fmtDate(item.effective_date)}</span>
                      <ChevronRight size={15} style={{ color: "var(--text-muted)" }} />
                    </button>
                    {showGauge && (
                      <div style={{ padding: "0 0 14px 25px", maxWidth: 460 }}>
                        <Gauge
                          value={numericValue as number}
                          low={item.reference_low as number}
                          high={item.reference_high as number}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {totalPages > 1 && (
              <div className="between" style={{ marginTop: 16 }}>
                <span className="muted mono" style={{ fontSize: 12 }}>
                  Page {page} of {totalPages}
                </span>
                <div className="chips" style={{ gap: 8 }}>
                  <button className="btn ghost sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                    Prev
                  </button>
                  <button className="btn ghost sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <RecordDetailSheet recordId={selected} open={!!selected} onClose={() => setSelected(null)} />
    </div>
  );
}
