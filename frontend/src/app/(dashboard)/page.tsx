"use client";

// Snapshot Overview — the committed home direction.
//
// Principle: MedTimeline surfaces what's in your record and how the recorded
// values have moved — and nothing more. It does NOT diagnose, judge, set
// targets, or recommend. Metric selection is mechanical (every observation
// code, newest first), and every value/range/flag is shown exactly as the
// source reported it.

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, ChevronRight, ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { recordTitle } from "@/lib/record-title";
import type {
  HealthRecord,
  RecordListResponse,
  UserResponse,
  PatientInfo,
  ObservationByCode,
  ObservationsByCodeResponse,
  RecentRecordItem,
  RecentRecordsResponse,
  RecordStats,
} from "@/types/api";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { fmtShort, yearOf } from "@/lib/format-date";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";
import { MarkerCard } from "@/components/retro/MarkerCard";

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

// A single recorded item in a list (recently-added feed). The type badge sits in
// a fixed-width column so every title lines up down the list.
function FeedRow({ r, onSelect }: { r: RecentRecordItem; onSelect: (id: string) => void }) {
  return (
    <button
      className="lrow"
      onClick={() => onSelect(r.id)}
      style={{ width: "100%", background: "transparent", border: 0, cursor: "pointer", textAlign: "left" }}
    >
      <RetroBadge recordType={r.record_type} />
      <span className="lrow-main">
        <span className="lrow-title">{recordTitle(r)}</span>
      </span>
      <span className="lrow-meta tnum">
        {r.source}
        {r.effective_date ? ` · ${fmtShort(r.effective_date)}` : ""}
      </span>
      <ChevronRight size={15} style={{ color: "var(--text-muted)" }} />
    </button>
  );
}

// A condition or medication row — neutral dot + title, conditions carry their
// recorded year and the source's own clinical status.
function ContextRow({
  r,
  type,
  onSelect,
}: {
  r: HealthRecord;
  type: "condition" | "medication";
  onSelect: (id: string) => void;
}) {
  const dot = (RECORD_TYPE_COLORS[type] ?? DEFAULT_RECORD_COLOR).dot;
  const year = yearOf(r.effective_date);
  return (
    <button
      className="lrow"
      onClick={() => onSelect(r.id)}
      style={{ width: "100%", background: "transparent", border: 0, cursor: "pointer", textAlign: "left" }}
    >
      <span className="dot" style={{ background: dot }} />
      <span className="lrow-main">
        <span className="lrow-title">{r.display_text}</span>
        {type === "condition" && (year || r.status) && (
          <span className="lrow-sub">
            {year ? `recorded ${year}` : ""}
            {year && r.status ? " · " : ""}
            {r.status ? `status: ${r.status}` : ""}
          </span>
        )}
      </span>
    </button>
  );
}

export default function OverviewPage() {
  const router = useRouter();
  const [byCode, setByCode] = useState<ObservationByCode[]>([]);
  const [conditions, setConditions] = useState<HealthRecord[]>([]);
  const [meds, setMeds] = useState<HealthRecord[]>([]);
  const [recent, setRecent] = useState<RecentRecordItem[]>([]);
  const [stats, setStats] = useState<RecordStats | null>(null);
  const [me, setMe] = useState<UserResponse | null>(null);
  const [patient, setPatient] = useState<PatientInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [shown, setShown] = useState(false);
  const markersRef = useRef<HTMLDivElement>(null);
  const [markerLimit, setMarkerLimit] = useState(6); // default desktop: 3 cols x 2 rows

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  // The marker grid shows your most recent results capped to two rows. Columns
  // are width-driven (CSS auto-fill); we read the rendered column count back and
  // show cols x 2, recomputing as the browser window resizes.
  useEffect(() => {
    const el = markersRef.current;
    if (!el) return;
    // ResizeObserver delivers an initial callback on observe() and again on every
    // resize, so the count stays in sync without a synchronous setState here.
    const ro = new ResizeObserver(() => {
      const cols = getComputedStyle(el).gridTemplateColumns.split(" ").filter(Boolean).length;
      if (cols > 0) setMarkerLimit(cols * 2);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [byCode.length]);

  function load() {
    // API → GET /observations/by-code (markers, recency-sorted server-side),
    //   GET /records?status=active (active conditions / current meds),
    //   GET /records/recent (ingestion-ordered feed), GET /records/stats (masthead),
    //   GET /dashboard/patients (subject identity), GET /auth/me (account fallback).
    return Promise.all([
      api.get<ObservationsByCodeResponse>("/observations/by-code").catch(() => null),
      api.get<RecordListResponse>("/records?record_type=condition&status=active&page_size=8").catch(() => null),
      api.get<RecordListResponse>("/records?record_type=medication&status=active&page_size=8").catch(() => null),
      api.get<RecentRecordsResponse>("/records/recent?limit=5").catch(() => null),
      api.get<RecordStats>("/records/stats").catch(() => null),
      api.get<{ items: PatientInfo[] }>("/dashboard/patients").catch(() => null),
      api.get<UserResponse>("/auth/me").catch(() => null),
    ]).then(([obs, cond, med, rec, st, patients, user]) => {
      setByCode(obs?.items ?? []);
      setConditions(cond?.items ?? []);
      setMeds(med?.items ?? []);
      setRecent(rec?.items ?? []);
      setStats(st);
      setPatient(patients?.items?.[0] ?? null);
      setMe(user);
    });
  }

  useEffect(() => {
    load().finally(() => setLoading(false));
  }, []);

  const subline = useMemo(() => {
    const start = yearOf(stats?.first_date);
    const end = yearOf(stats?.last_date);
    const span = start && end ? (start === end ? `${start}` : `${start}–${end}`) : null;
    const sex = patient?.gender ? patient.gender[0].toUpperCase() + patient.gender.slice(1) : null;
    const bornYear = yearOf(patient?.birth_date);
    return [
      sex,
      bornYear ? `born ${bornYear}` : null,
      `${stats?.total ?? 0} records`,
      span,
    ]
      .filter(Boolean)
      .join(" · ");
  }, [stats, patient]);

  if (loading) return <RetroLoadingState text="Loading your record" />;

  const subjectName = patient?.name?.trim() || me?.display_name?.trim() || "Your records";
  const isEmpty = byCode.length === 0 && conditions.length === 0 && meds.length === 0 && recent.length === 0;

  return (
    <div className={`screen s24 ${shown ? "on" : ""}`}>
      {/* Masthead — neutral, reads for any viewer (provider, family, advocate, AI). */}
      <div className="masthead">
        <div>
          <p className="kicker">Personal Health Record</p>
          <h1 className="masthead-name display">{subjectName}</h1>
          {subline && <p className="h-sub">{subline}</p>}
        </div>
        <div className="masthead-side">
          <SecureChip />
        </div>
      </div>

      {isEmpty ? (
        <div className="card-surface pad" style={{ textAlign: "center", padding: "64px 24px" }}>
          <p className="muted text-sm mb-3">No records yet.</p>
          <button className="btn" onClick={() => router.push("/upload")}>
            Upload records
          </button>
        </div>
      ) : (
        <>
          {/* Your most recent results — every observation type, latest value first. */}
          <div>
            <div className="ov-sec-head">
              <div className="lead">
                <h3 className="sec-title">Your most recent results</h3>
                <span className="ov-sec-note">
                  Every observation on file, latest value first. Values, ranges and flags appear exactly as the
                  source reported them — MedTimeline adds no interpretation.
                </span>
              </div>
              <button className="ov-link" onClick={() => router.push("/labs")}>
                {byCode.length > markerLimit ? `All ${byCode.length} labs & vitals` : "All labs & vitals"}{" "}
                <ArrowRight size={14} />
              </button>
            </div>
            {byCode.length === 0 ? (
              <p className="muted" style={{ fontSize: 13 }}>
                No labs or vitals recorded yet.
              </p>
            ) : (
              <div className="markers" ref={markersRef}>
                {byCode.slice(0, markerLimit).map((m) => (
                  <MarkerCard key={m.code} m={m} onSelect={setSelected} />
                ))}
              </div>
            )}
          </div>

          {/* Bento: conditions & meds on file + recently added */}
          <div className="grid-bento">
            <div className="card-surface pad">
              <div className="card-h">
                <h3 className="sec-title">Conditions &amp; medications on file</h3>
                <button className="btn ghost sm" onClick={() => router.push("/conditions")}>
                  Details
                </button>
              </div>
              <div className="grid-2" style={{ gap: 26 }}>
                <div>
                  <div className="field-l">Conditions · status active</div>
                  {conditions.length === 0 ? (
                    <p className="muted" style={{ fontSize: 13 }}>
                      None on file.
                    </p>
                  ) : (
                    conditions.map((r) => (
                      <ContextRow key={r.id} r={r} type="condition" onSelect={setSelected} />
                    ))
                  )}
                </div>
                <div>
                  <div className="field-l">Medications · status active</div>
                  {meds.length === 0 ? (
                    <p className="muted" style={{ fontSize: 13 }}>
                      None on file.
                    </p>
                  ) : (
                    meds.map((r) => <ContextRow key={r.id} r={r} type="medication" onSelect={setSelected} />)
                  )}
                </div>
              </div>
            </div>

            {/* Recently added — ordered by when records landed in MedTimeline. */}
            <div className="card-surface pad">
              <div className="card-h">
                <h3 className="sec-title">Recently added</h3>
                <button className="btn ghost sm" onClick={() => router.push("/timeline")}>
                  Timeline
                </button>
              </div>
              {recent.length === 0 ? (
                <p className="muted" style={{ fontSize: 13 }}>
                  Nothing added yet.
                </p>
              ) : (
                <div>
                  {recent.map((r) => (
                    <FeedRow key={r.id} r={r} onSelect={setSelected} />
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      <RecordDetailSheet
        recordId={selected}
        open={!!selected}
        onClose={() => setSelected(null)}
        onDelete={() => {
          setLoading(true);
          load().finally(() => setLoading(false));
        }}
      />
    </div>
  );
}
