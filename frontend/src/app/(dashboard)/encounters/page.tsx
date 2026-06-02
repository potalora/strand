"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { HealthRecord, RecordListResponse } from "@/types/api";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";

const RECORD_TYPE = "encounter";
const PAGE_SIZE = 25;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = (s: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

export default function EncountersPage() {
  const router = useRouter();
  const [records, setRecords] = useState<HealthRecord[]>([]);
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
      .get<RecordListResponse>(`/records?record_type=${RECORD_TYPE}&page=${page}&page_size=${PAGE_SIZE}`)
      .then((data) => {
        setRecords(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => {
        setRecords([]);
        setTotal(0);
      })
      .finally(() => setLoading(false));
  }, [page]);

  const dot = (RECORD_TYPE_COLORS[RECORD_TYPE] ?? DEFAULT_RECORD_COLOR).dot;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">Encounters</p>
          <h1 className="h1 display">Visits &amp; encounters</h1>
        </div>
        <SecureChip />
      </div>

      <div className="card-surface pad">
        {loading ? (
          <RetroLoadingState text="Loading encounters" />
        ) : records.length === 0 ? (
          <div className="py-8 text-center">
            <p className="muted text-sm mb-3">No encounters on file.</p>
            <button className="btn" onClick={() => router.push("/upload")}>
              Upload records
            </button>
          </div>
        ) : (
          <>
            <div className="card-h">
              <h3 className="sec-title">{total} record{total === 1 ? "" : "s"}</h3>
            </div>
            <div>
              {records.map((r) => (
                <button
                  key={r.id}
                  className="lrow"
                  onClick={() => setSelected(r.id)}
                  style={{ width: "100%", background: "transparent", border: 0, borderBottom: "1px solid var(--border)", cursor: "pointer", textAlign: "left" }}
                >
                  <span className="dot" style={{ background: dot }} />
                  <span className="lrow-main">
                    <span className="lrow-title">{r.display_text}</span>
                    {r.status && <span className="lrow-sub">{r.status}</span>}
                  </span>
                  <span className="lrow-meta tnum">{fmtDate(r.effective_date)}</span>
                  <ChevronRight size={15} style={{ color: "var(--text-muted)" }} />
                </button>
              ))}
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
