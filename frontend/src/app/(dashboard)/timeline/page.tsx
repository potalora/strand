"use client";

import { useEffect, useMemo, useState } from "react";
import { Lock } from "lucide-react";
import { api } from "@/lib/api";
import type { TimelineResponse, TimelineEvent, UserResponse } from "@/types/api";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const fmtShort = (s: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

const FILTERS: { v: string; label: string }[] = [
  { v: "", label: "All" },
  { v: "condition", label: "Conditions" },
  { v: "observation", label: "Labs & vitals" },
  { v: "medication", label: "Medications" },
  { v: "encounter", label: "Visits" },
  { v: "immunization", label: "Vaccines" },
  { v: "procedure", label: "Procedures" },
  { v: "imaging", label: "Imaging" },
  { v: "allergy", label: "Allergies" },
  { v: "document", label: "Documents" },
];

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

function groupByMonth(events: TimelineEvent[]): { label: string; events: TimelineEvent[] }[] {
  const groups = new Map<string, TimelineEvent[]>();
  for (const event of events) {
    const key = event.effective_date
      ? (() => {
          const d = new Date(event.effective_date);
          return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
        })()
      : "undated";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(event);
  }
  const sorted = Array.from(groups.entries()).sort((a, b) => {
    if (a[0] === "undated") return 1;
    if (b[0] === "undated") return -1;
    return b[0].localeCompare(a[0]);
  });
  return sorted.map(([key, evs]) => {
    if (key === "undated") return { label: "Undated", events: evs };
    const [y, m] = key.split("-");
    return { label: `${MONTHS[parseInt(m, 10) - 1]} ${y}`, events: evs };
  });
}

export default function TimelinePage() {
  const [data, setData] = useState<TimelineResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [selectedRecord, setSelectedRecord] = useState<string | null>(null);
  const [me, setMe] = useState<UserResponse | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    // Optional masthead kicker — falls back to "Your health story" when unavailable.
    api
      .get<UserResponse>("/auth/me")
      .then(setMe)
      .catch(() => setMe(null));
  }, []);

  useEffect(() => {
    setLoading(true);
    api
      .get<TimelineResponse>("/timeline?limit=200" + (filter ? `&record_type=${filter}` : ""))
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [filter]);

  const events: TimelineEvent[] = useMemo(() => data?.events ?? [], [data]);

  const sortedEvents = useMemo(
    () =>
      [...events].sort((a, b) => {
        if (!a.effective_date && !b.effective_date) return 0;
        if (!a.effective_date) return 1;
        if (!b.effective_date) return -1;
        return new Date(b.effective_date).getTime() - new Date(a.effective_date).getTime();
      }),
    [events],
  );

  const groups = useMemo(() => groupByMonth(sortedEvents), [sortedEvents]);

  const kicker = me?.display_name?.trim() || "Your health story";

  return (
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">{kicker}</p>
          <h1 className="h1 display">Timeline</h1>
        </div>
        <SecureChip />
      </div>

      <div className="filters">
        {FILTERS.map((f) => (
          <button
            key={f.v}
            className="filt"
            aria-pressed={filter === f.v}
            onClick={() => setFilter(f.v)}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading ? (
        <RetroLoadingState text="Loading timeline" />
      ) : sortedEvents.length === 0 ? (
        <div className="muted" style={{ padding: "60px 0", textAlign: "center" }}>
          No events.
        </div>
      ) : (
        <div className="tl">
          {groups.map((g) => (
            <div className="tl-grp" key={g.label}>
              <div className="tl-grp-h">
                <span className="tl-grp-label">{g.label}</span>
                <span className="tl-grp-rule" />
                <span className="tl-grp-n">{g.events.length}</span>
              </div>
              <div className="tl-list">
                {g.events.map((r) => {
                  const colors = RECORD_TYPE_COLORS[r.record_type] || DEFAULT_RECORD_COLOR;
                  const category = r.category?.length ? r.category.join(", ") : null;
                  return (
                    <button
                      key={r.id}
                      className="tl-card"
                      onClick={() => setSelectedRecord(r.id)}
                    >
                      <span className="tl-rail" style={{ background: colors.dot }} />
                      <span className="tl-main">
                        <span className="tl-top">
                          <RetroBadge recordType={r.record_type} />
                          <span className="tl-title">{r.display_text}</span>
                        </span>
                        {r.code_display && <span className="tl-sub">{r.code_display}</span>}
                      </span>
                      <span className="tl-date">
                        {fmtShort(r.effective_date)}
                        {category && (
                          <>
                            <br />
                            <span className="muted">{category}</span>
                          </>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      <RecordDetailSheet
        recordId={selectedRecord}
        open={!!selectedRecord}
        onClose={() => setSelectedRecord(null)}
      />
    </div>
  );
}
