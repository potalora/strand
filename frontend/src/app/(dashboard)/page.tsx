"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type {
  DashboardOverview,
  HealthRecord,
  RecordListResponse,
  UserResponse,
  PatientInfo,
  SourcesResponse,
  SourceBreakdown,
} from "@/types/api";
import { RECORD_TYPE_COLORS, RECORD_TYPE_LABELS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = (s: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

// Per-type browse destinations (category chips). Types without a dedicated page
// fall through to the full Records table.
const CATEGORY_ROUTES: Record<string, string> = {
  condition: "/conditions",
  observation: "/labs",
  medication: "/medications",
  encounter: "/encounters",
  imaging: "/imaging",
  immunization: "/immunizations",
};

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

export default function OverviewPage() {
  const router = useRouter();
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [conditions, setConditions] = useState<HealthRecord[]>([]);
  const [meds, setMeds] = useState<HealthRecord[]>([]);
  const [me, setMe] = useState<UserResponse | null>(null);
  const [patient, setPatient] = useState<PatientInfo | null>(null);
  const [sources, setSources] = useState<SourceBreakdown[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    // API → GET /dashboard/overview (stats · records_by_type · recent · date_range),
    //   GET /dashboard/sources (provenance breakdown + data-sources count),
    //   GET /records?status=active (active conditions / current meds),
    //   GET /dashboard/patients (subject name/DOB), GET /auth/me (account).
    Promise.all([
      api.get<DashboardOverview>("/dashboard/overview").catch(() => null),
      api.get<RecordListResponse>("/records?record_type=condition&status=active&page_size=6").catch(() => null),
      api.get<RecordListResponse>("/records?record_type=medication&status=active&page_size=6").catch(() => null),
      api.get<UserResponse>("/auth/me").catch(() => null),
      api.get<{ items: PatientInfo[] }>("/dashboard/patients").catch(() => null),
      api.get<SourcesResponse>("/dashboard/sources").catch(() => null),
    ])
      .then(([ov, cond, med, user, patients, src]) => {
        setOverview(ov);
        setConditions(cond?.items ?? []);
        setMeds(med?.items ?? []);
        setMe(user);
        setPatient(patients?.items?.[0] ?? null);
        setSources(src?.items ?? []);
      })
      .finally(() => setLoading(false));
  }, []);

  const data = useMemo(() => {
    const byType = overview?.records_by_type ?? {};
    const catList = Object.entries(byType).sort((a, b) => b[1] - a[1]);
    const total = catList.reduce((n, [, c]) => n + c, 0);
    const start = overview?.date_range_start ? new Date(overview.date_range_start).getFullYear() : null;
    const end = overview?.date_range_end ? new Date(overview.date_range_end).getFullYear() : null;
    const span = start && end ? (start === end ? `${start}` : `${start}–${end}`) : "—";
    return { byType, catList, total, span };
  }, [overview]);

  if (loading) return <RetroLoadingState text="Loading your record" />;

  const subjectName = patient?.name?.trim() || me?.display_name?.trim() || "Your records";
  const bornYear = patient?.birth_date ? new Date(patient.birth_date).getFullYear() : null;
  const subline = [
    patient?.gender ? patient.gender[0].toUpperCase() + patient.gender.slice(1) : null,
    bornYear ? `born ${bornYear}` : null,
    `${overview?.total_records ?? 0} records on file`,
    data.span !== "—" ? data.span : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const stats = [
    { val: overview?.total_records ?? 0, label: "Total records" },
    { val: data.catList.length, label: "Record types" },
    { val: sources.length, label: "Data sources" },
    { val: data.span, label: "Years covered" },
  ];

  return (
    <div className={`screen s24 ${shown ? "on" : ""}`}>
      {/* Masthead — generic, reads for any viewer (provider, family, advocate, AI). */}
      {/* TODO(backend): subject name/DOB aren't returned today (PII-encrypted). Using the
          account display name as the subject; add name/DOB to /dashboard/patients to show
          the true record subject. */}
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

      {/* Stats */}
      <div className="stats">
        {stats.map((s) => (
          <div className="stat" key={s.label}>
            <div className="stat-val tnum">{s.val}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Bento: conditions & meds + what's on file */}
      <div className="grid-bento">
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Conditions &amp; medications</h3>
            <button className="btn ghost sm" onClick={() => router.push("/conditions")}>
              Details
            </button>
          </div>
          <div className="grid-2" style={{ gap: 26 }}>
            <div>
              <div className="field-l">Conditions</div>
              {conditions.length === 0 ? (
                <p className="muted" style={{ fontSize: 13 }}>
                  None on file.
                </p>
              ) : (
                conditions.map((r) => (
                  <button
                    key={r.id}
                    className="lrow"
                    onClick={() => setSelected(r.id)}
                    style={{ width: "100%", background: "transparent", border: 0, cursor: "pointer", textAlign: "left" }}
                  >
                    <span
                      className="dot"
                      style={{ background: (RECORD_TYPE_COLORS.condition ?? DEFAULT_RECORD_COLOR).dot }}
                    />
                    <span className="lrow-main">
                      <span className="lrow-title">{r.display_text}</span>
                      {r.effective_date && (
                        <span className="lrow-sub">since {new Date(r.effective_date).getFullYear()}</span>
                      )}
                    </span>
                  </button>
                ))
              )}
            </div>
            <div>
              <div className="field-l">Medications</div>
              {meds.length === 0 ? (
                <p className="muted" style={{ fontSize: 13 }}>
                  None on file.
                </p>
              ) : (
                meds.map((r) => (
                  <button
                    key={r.id}
                    className="lrow"
                    onClick={() => setSelected(r.id)}
                    style={{ width: "100%", background: "transparent", border: 0, cursor: "pointer", textAlign: "left" }}
                  >
                    <span
                      className="dot"
                      style={{ background: (RECORD_TYPE_COLORS.medication ?? DEFAULT_RECORD_COLOR).dot }}
                    />
                    <span className="lrow-main">
                      <span className="lrow-title">{r.display_text}</span>
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Where records come from — real provenance breakdown (GET /dashboard/sources). */}
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Where records come from</h3>
            <span className="muted mono" style={{ fontSize: 11 }}>
              {sources.length} source{sources.length === 1 ? "" : "s"}
            </span>
          </div>
          {sources.length === 0 ? (
            <p className="muted" style={{ fontSize: 13 }}>
              No sources yet.
            </p>
          ) : (
            <div className="s16">
              {sources.slice(0, 6).map((s) => (
                <div key={s.source}>
                  <div className="between" style={{ marginBottom: 6 }}>
                    <span className="src-name">{s.source}</span>
                    <span className="muted mono" style={{ fontSize: 12, whiteSpace: "nowrap", flexShrink: 0 }}>
                      {s.count} record{s.count > 1 ? "s" : ""}
                    </span>
                  </div>
                  <div className="bar">
                    <i style={{ width: `${Math.max(6, (s.count / (sources[0]?.count || 1)) * 100)}%` }} />
                  </div>
                </div>
              ))}
              {sources.length > 6 && (
                <div className="muted" style={{ fontSize: 12.5 }}>
                  +{sources.length - 6} more {sources.length - 6 > 1 ? "sources" : "source"}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Browse by category */}
      {data.catList.length > 0 && (
        <div className="card-surface pad">
          <div className="card-h">
            <h3 className="sec-title">Browse by category</h3>
          </div>
          <div className="chips">
            {data.catList.map(([type, n]) => (
              <button
                key={type}
                className="chip"
                onClick={() => router.push(CATEGORY_ROUTES[type] || "/records")}
              >
                <span className="dot" style={{ background: (RECORD_TYPE_COLORS[type] ?? DEFAULT_RECORD_COLOR).dot }} />
                {RECORD_TYPE_LABELS[type] || type}
                <span className="n">{n}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Recent activity */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Recent activity</h3>
          <button className="btn ghost sm" onClick={() => router.push("/timeline")}>
            View all
          </button>
        </div>
        {(!overview || overview.recent_records.length === 0) ? (
          <div className="py-8 text-center">
            <p className="muted text-sm mb-3">No records yet.</p>
            <button className="btn" onClick={() => router.push("/upload")}>
              Upload records
            </button>
          </div>
        ) : (
          <div>
            {overview.recent_records.map((r) => (
              <button
                key={r.id}
                className="lrow"
                onClick={() => setSelected(r.id)}
                style={{ width: "100%", background: "transparent", border: 0, cursor: "pointer", textAlign: "left" }}
              >
                <RetroBadge recordType={r.record_type} />
                <span className="lrow-main">
                  <span className="lrow-title">{r.display_text}</span>
                </span>
                <span className="lrow-meta tnum">{fmtDate(r.effective_date || r.created_at)}</span>
                <ChevronRight size={15} style={{ color: "var(--text-muted)" }} />
              </button>
            ))}
          </div>
        )}
      </div>

      <RecordDetailSheet recordId={selected} open={!!selected} onClose={() => setSelected(null)} />
    </div>
  );
}
