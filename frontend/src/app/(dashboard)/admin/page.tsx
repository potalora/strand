"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Search, Trash2, Upload, FileStack, Copy, Settings, Check } from "lucide-react";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/useAuthStore";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import type {
  RecordListResponse,
  HealthRecord,
  DedupCandidate,
  UserResponse,
  DashboardOverview,
  AuditLogResponse,
  AuditLogEntry,
} from "@/types/api";
import { RECORD_TYPE_LABELS } from "@/lib/constants";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";
import { ConfirmDialog } from "@/components/retro/ConfirmDialog";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = (s: string | null | undefined) => {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return "—";
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

const TABS = [
  { key: "records", label: "Records", icon: FileStack },
  { key: "extractions", label: "Extractions", icon: Upload },
  { key: "dedup", label: "Duplicates", icon: Copy },
  { key: "sys", label: "System", icon: Settings },
];

export default function AdminPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialTab = searchParams.get("tab") || "records";
  const [activeTab, setActiveTab] = useState(initialTab);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    const tab = searchParams.get("tab") || "records";
    setActiveTab(tab);
  }, [searchParams]);

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    router.replace(`/admin?tab=${key}`, { scroll: false });
  };

  return (
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">Data &amp; settings</p>
          <h1 className="h1 display">Admin</h1>
        </div>
      </div>

      <div className="tabs" role="tablist">
        {TABS.map((t) => {
          const TabIcon = t.icon;
          return (
            <button
              key={t.key}
              className="tab"
              role="tab"
              aria-selected={activeTab === t.key}
              onClick={() => handleTabChange(t.key)}
            >
              <TabIcon size={16} />
              {t.label}
            </button>
          );
        })}
      </div>

      {activeTab === "records" && <RecordsTab />}
      {activeTab === "extractions" && <ExtractionsTab />}
      {activeTab === "dedup" && <DedupTab />}
      {activeTab === "sys" && <SystemTab />}
    </div>
  );
}

/* ==========================================
   EXTRACTIONS TAB — Pending extraction management
   ========================================== */

interface ExtractionFile {
  id: string;
  filename: string;
  mime_type: string;
  file_category: string;
  file_size_bytes: number | null;
  created_at: string | null;
  ingestion_status?: string;
}

const EXTRACTION_STATUS_HUE: Record<string, string> = {
  processing: "var(--primary)",
  failed: "var(--danger)",
  duplicate_file: "var(--ochre, var(--primary))",
  pending_extraction: "var(--text-muted)",
};

function fmtSize(bytes: number | null): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function ExtractionsTab() {
  const [files, setFiles] = useState<ExtractionFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [triggering, setTriggering] = useState(false);

  const fetchFiles = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await api.get<{
        files: ExtractionFile[];
        total: number;
      }>("/upload/pending-extraction?statuses=pending_extraction,processing,failed");
      setFiles(resp.files);
    } catch {
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleTrigger = async () => {
    if (selected.size === 0) return;
    setTriggering(true);
    try {
      await api.post("/upload/trigger-extraction", {
        upload_ids: Array.from(selected),
      });
      setSelected(new Set());
      setTimeout(fetchFiles, 1000);
    } catch {
      // Silently fail
    } finally {
      setTriggering(false);
    }
  };

  const toggleAll = () => {
    if (selected.size === files.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(files.map((f) => f.id)));
    }
  };

  if (loading) return <RetroLoadingState text="Loading extractions" />;

  if (files.length === 0) {
    return (
      <div>
        <p className="h-sub" style={{ margin: "0 0 18px" }}>
          Documents waiting for text + entity extraction. Structured formats parse instantly; PDF,
          RTF, and TIFF files are read with OCR + entity extraction once you trigger them.
        </p>
        <div className="card-surface pad" style={{ textAlign: "center", padding: "56px 24px" }}>
          <span className="dz-ic" style={{ margin: "0 auto 14px" }}>
            <Check size={22} />
          </span>
          <div className="muted" style={{ fontSize: 14.5 }}>
            Nothing waiting — no files pending extraction, processing, or failed.
          </div>
        </div>
      </div>
    );
  }

  const pendingCount = files.filter(
    (f) => f.ingestion_status === "pending_extraction" || !f.ingestion_status
  ).length;
  const processingCount = files.filter((f) => f.ingestion_status === "processing").length;
  const failedCount = files.filter((f) => f.ingestion_status === "failed").length;

  const allSelected = selected.size === files.length && files.length > 0;

  return (
    <div>
      <p className="h-sub" style={{ margin: "0 0 18px" }}>
        Documents waiting for text + entity extraction. Select files and extract — each is read with
        OCR + entity extraction, then confirmed into the record.
      </p>

      <div className="toolbar" style={{ justifyContent: "space-between" }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {pendingCount > 0 && (
            <span className="tag">
              <span className="tdot" style={{ background: "var(--text-muted)" }} />
              {pendingCount} pending
            </span>
          )}
          {processingCount > 0 && (
            <span className="tag">
              <span className="tdot" style={{ background: "var(--primary)" }} />
              {processingCount} processing
            </span>
          )}
          {failedCount > 0 && (
            <span className="tag">
              <span className="tdot" style={{ background: "var(--danger)" }} />
              {failedCount} failed
            </span>
          )}
        </div>
        <button
          className="btn"
          onClick={handleTrigger}
          disabled={selected.size === 0 || triggering}
        >
          {triggering ? "Triggering…" : `Extract ${selected.size || "selected"}`}
        </button>
      </div>

      <div className="tablewrap">
        <table className="rtable">
          <thead>
            <tr>
              <th style={{ width: 1 }}>
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleAll}
                  disabled={triggering}
                  style={{ accentColor: "var(--primary)" }}
                  aria-label="Select all files"
                />
              </th>
              <th>File</th>
              <th>Type</th>
              <th>Status</th>
              <th>Size</th>
              <th>Uploaded</th>
            </tr>
          </thead>
          <tbody>
            {files.map((file) => {
              const ext = file.filename.split(".").pop()?.toLowerCase() || "";
              const status = file.ingestion_status || "pending_extraction";
              const statusLabel =
                status === "duplicate_file"
                  ? "Duplicate"
                  : status === "pending_extraction"
                    ? "Pending"
                    : status.charAt(0).toUpperCase() + status.slice(1);
              return (
                <tr key={file.id} style={{ cursor: "default" }}>
                  <td style={{ width: 1 }}>
                    <input
                      type="checkbox"
                      checked={selected.has(file.id)}
                      onChange={() => {
                        setSelected((prev) => {
                          const next = new Set(prev);
                          if (next.has(file.id)) next.delete(file.id);
                          else next.add(file.id);
                          return next;
                        });
                      }}
                      disabled={triggering}
                      style={{ accentColor: "var(--primary)" }}
                      aria-label={`Select ${file.filename}`}
                    />
                  </td>
                  <td className="desc">{file.filename}</td>
                  <td className="num" style={{ textTransform: "uppercase" }}>
                    {ext || "—"}
                  </td>
                  <td>
                    <span className="tag">
                      <span
                        className="tdot"
                        style={{ background: EXTRACTION_STATUS_HUE[status] || "var(--text-muted)" }}
                      />
                      {statusLabel}
                    </span>
                  </td>
                  <td className="num">{fmtSize(file.file_size_bytes)}</td>
                  <td className="num">{fmtDate(file.created_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="between" style={{ marginTop: 14 }}>
        <span className="h-sub" style={{ margin: 0 }}>
          {selected.size} of {files.length} selected
        </span>
        <button className="btn ghost sm" onClick={fetchFiles}>
          Refresh
        </button>
      </div>
    </div>
  );
}

/* ==========================================
   RECORDS TAB — Flat searchable/sortable table
   ========================================== */

type SortKey = "type" | "desc" | "date";
interface SortState {
  key: SortKey;
  dir: 1 | -1;
}

// Map sortable header keys → the API's `sort` query values (server-sorted).
const SORT_API_KEY: Record<SortKey, string> = {
  type: "type",
  desc: "display_text",
  date: "date",
};

function RecordsTab() {
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);

  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [sort, setSort] = useState<SortState>({ key: "date", dir: -1 });

  const [selectedRecord, setSelectedRecord] = useState<string | null>(null);

  // Single delete state
  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();
  const [singleDeleteId, setSingleDeleteId] = useState<string | null>(null);
  const [singleConfirmOpen, setSingleConfirmOpen] = useState(false);
  const [singleDontAskChecked, setSingleDontAskChecked] = useState(false);

  const triggerRefresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  // API → GET /records (paginated, server-sorted). Search + type filter stay
  // client-side over the loaded page; column sort is sent to the server via
  // ?sort=&order= and refetches.
  useEffect(() => {
    setLoading(true);
    const order = sort.dir === 1 ? "asc" : "desc";
    api
      .get<RecordListResponse>(
        `/records?page=1&page_size=500&sort=${SORT_API_KEY[sort.key]}&order=${order}`
      )
      .then((data) => setRecords(data.items || []))
      .catch(() => setRecords([]))
      .finally(() => setLoading(false));
  }, [refreshKey, sort]);

  const types = useMemo(
    () => Array.from(new Set(records.map((r) => r.record_type))).sort((a, b) => a.localeCompare(b)),
    [records]
  );

  // Records arrive server-sorted; search + type filter narrow the loaded page.
  const rows = useMemo(() => {
    return records.filter((r) => {
      if (typeFilter && r.record_type !== typeFilter) return false;
      if (q) {
        const s = `${r.display_text} ${r.code_display || ""} ${r.source_format} ${
          r.code_value || ""
        }`.toLowerCase();
        if (!s.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [records, q, typeFilter]);

  const handleSingleDelete = (id: string) => {
    if (skipDeleteConfirm) {
      performSingleDelete(id);
    } else {
      setSingleDeleteId(id);
      setSingleDontAskChecked(false);
      setSingleConfirmOpen(true);
    }
  };

  const performSingleDelete = async (id: string) => {
    try {
      await api.delete(`/records/${id}`);
      if (singleDontAskChecked) setSkipDeleteConfirm(true);
      setSingleConfirmOpen(false);
      setSingleDeleteId(null);
      triggerRefresh();
    } catch {
      // silently fail
    }
  };

  const th = (key: SortKey, label: string) => (
    <th
      className="sortable"
      onClick={() =>
        setSort((s) => ({
          key,
          dir: s.key === key ? ((-s.dir) as 1 | -1) : key === "date" ? -1 : 1,
        }))
      }
    >
      {label}
      {sort.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );

  if (loading) return <RetroLoadingState text="Loading records" />;

  return (
    <div>
      <p className="h-sub" style={{ margin: "0 0 18px" }}>
        Every resource in the record, as structured data — search, sort, and open any row. (The
        Timeline is the same data, told as a story.)
      </p>

      <div className="toolbar">
        <div className="search">
          <Search size={16} />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search descriptions, codes, sources…"
          />
        </div>
        <select
          className="selectbox"
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
        >
          <option value="">All types ({records.length})</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {RECORD_TYPE_LABELS[t] || t}
            </option>
          ))}
        </select>
      </div>

      <div className="tablewrap">
        <table className="rtable">
          <thead>
            <tr>
              {th("type", "Type")}
              {th("desc", "Description")}
              {th("date", "Date")}
              <th>Source</th>
              <th>Code</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="clickable" onClick={() => setSelectedRecord(r.id)}>
                <td>
                  <RetroBadge recordType={r.record_type} short />
                </td>
                <td className="desc">{r.display_text}</td>
                <td className="num">{fmtDate(r.effective_date)}</td>
                <td className="num">{r.source_format}</td>
                <td className="num">
                  {[r.code_system, r.code_value].filter(Boolean).join(" ") || "—"}
                </td>
                <td style={{ width: 1 }}>
                  <button
                    className="row-del"
                    title="Delete record"
                    aria-label={`Delete ${r.display_text}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSingleDelete(r.id);
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="h-sub" style={{ marginTop: 14 }}>
        {rows.length} of {records.length} records
      </p>

      <RecordDetailSheet
        recordId={selectedRecord}
        open={!!selectedRecord}
        onClose={() => setSelectedRecord(null)}
        onDelete={triggerRefresh}
      />

      <ConfirmDialog
        open={singleConfirmOpen}
        title="Delete record?"
        description="This record will be soft-deleted and can no longer be viewed."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={() => {
          if (singleDeleteId) performSingleDelete(singleDeleteId);
        }}
        onCancel={() => {
          setSingleConfirmOpen(false);
          setSingleDeleteId(null);
        }}
        showDontAskAgain
        dontAskAgainChecked={singleDontAskChecked}
        onDontAskAgainChange={setSingleDontAskChecked}
      />
    </div>
  );
}

/* ==========================================
   DEDUP / DUPLICATES TAB
   ========================================== */

function DedupTab() {
  const [candidates, setCandidates] = useState<DedupCandidate[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<string | null>(null);
  const pageSize = 20;

  const fetchCandidates = useCallback(
    (p: number) => {
      setLoading(true);
      api
        .get<{ items: DedupCandidate[]; total: number }>(
          `/dedup/candidates?page=${p}&limit=${pageSize}`
        )
        .then((data) => {
          setCandidates(data.items || []);
          setTotal(data.total || 0);
        })
        .catch(() => {
          setCandidates([]);
          setTotal(0);
        })
        .finally(() => setLoading(false));
    },
    [pageSize]
  );

  useEffect(() => {
    fetchCandidates(page);
  }, [page, fetchCandidates]);

  const handleScan = async () => {
    setScanning(true);
    setError(null);
    setScanResult(null);
    try {
      const result = await api.post<{ candidates_found: number }>("/dedup/scan");
      setScanResult(`Scan complete. ${result.candidates_found} potential duplicates found.`);
      setPage(1);
      fetchCandidates(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setScanning(false);
    }
  };

  const handleMerge = async (candidateId: string) => {
    setActionLoading(candidateId);
    try {
      await api.post("/dedup/merge", { candidate_id: candidateId });
      setCandidates((prev) => prev.filter((c) => c.id !== candidateId));
      setTotal((t) => Math.max(0, t - 1));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Merge failed");
    } finally {
      setActionLoading(null);
    }
  };

  const handleDismiss = async (candidateId: string) => {
    setActionLoading(candidateId);
    try {
      await api.post("/dedup/dismiss", { candidate_id: candidateId });
      setCandidates((prev) => prev.filter((c) => c.id !== candidateId));
      setTotal((t) => Math.max(0, t - 1));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dismiss failed");
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div>
      <p className="h-sub" style={{ margin: "0 0 18px" }}>
        Potential duplicates found across sources. Each is scored, then auto-merged, auto-dismissed,
        or sent here for your review.
      </p>

      <div className="toolbar">
        <button className="btn" onClick={handleScan} disabled={scanning}>
          {scanning ? "Scanning…" : "Scan for duplicates"}
        </button>
        {scanResult && (
          <span className="h-sub" style={{ margin: 0 }}>
            {scanResult}
          </span>
        )}
      </div>

      {error && (
        <div className="card-surface pad" style={{ marginBottom: 14 }}>
          <div className="between" style={{ justifyContent: "flex-start", gap: 12 }}>
            <span className="tag" style={{ background: "var(--danger)", color: "var(--on-primary)" }}>
              ERROR
            </span>
            <p className="muted" style={{ fontSize: 13.5, margin: 0 }}>
              {error}
            </p>
          </div>
        </div>
      )}

      {loading ? (
        <RetroLoadingState text="Loading candidates" />
      ) : candidates.length === 0 ? (
        <div className="card-surface pad" style={{ textAlign: "center", padding: "56px 24px" }}>
          <span className="dz-ic" style={{ margin: "0 auto 14px" }}>
            <Check size={22} />
          </span>
          <div className="muted" style={{ fontSize: 14.5 }}>
            All clear — no duplicates waiting for review.
          </div>
          <p className="h-sub" style={{ marginTop: 6 }}>
            Run the scanner to check for potential duplicates.
          </p>
        </div>
      ) : (
        <div>
          <div className="between" style={{ marginBottom: 14 }}>
            <span className="h-sub" style={{ margin: 0 }}>
              {total.toLocaleString()} pending candidates — showing {(page - 1) * pageSize + 1}–
              {Math.min(page * pageSize, total)}
            </span>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="btn ghost sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </button>
              <button
                className="btn ghost sm"
                disabled={page * pageSize >= total}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>

          {candidates.map((candidate) => (
            <DedupCandidateCard
              key={candidate.id}
              candidate={candidate}
              busy={actionLoading === candidate.id}
              onMerge={() => handleMerge(candidate.id)}
              onKeepBoth={() => handleDismiss(candidate.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function DedupCandidateCard({
  candidate,
  busy,
  onMerge,
  onKeepBoth,
}: {
  candidate: DedupCandidate;
  busy: boolean;
  onMerge: () => void;
  onKeepBoth: () => void;
}) {
  const reasons = Object.entries(candidate.match_reasons)
    .filter(([, matched]) => matched)
    .map(([reason]) => reason);

  const records = [
    { label: "RECORD A", rec: candidate.record_a },
    { label: "RECORD B", rec: candidate.record_b },
  ];

  return (
    <div className="dedup-card">
      <div className="between" style={{ gap: 12 }}>
        <span
          className="display"
          style={{ fontSize: 22, color: "var(--primary)", whiteSpace: "nowrap" }}
        >
          {Math.round(candidate.similarity_score * 100)}% match
        </span>
        <div style={{ display: "flex", gap: 9 }}>
          <button className="btn sm" onClick={onMerge} disabled={busy}>
            {busy ? "…" : "Merge"}
          </button>
          <button className="btn ghost sm" onClick={onKeepBoth} disabled={busy}>
            Keep both
          </button>
        </div>
      </div>

      {reasons.length > 0 && (
        <div className="reasons" style={{ marginTop: 11 }}>
          {reasons.map((r) => (
            <span className="reason" key={r}>
              {r}
            </span>
          ))}
        </div>
      )}

      <div className="dedup-pair">
        {records.map(({ label, rec }, i) => (
          <DedupRecPair key={label} label={label} rec={rec} showVs={i === 1} />
        ))}
      </div>
    </div>
  );
}

function DedupRecPair({
  label,
  rec,
  showVs,
}: {
  label: string;
  rec: DedupCandidate["record_a"];
  showVs: boolean;
}) {
  return (
    <>
      {showVs && <div className="dedup-vs">VS</div>}
      <div className="dedup-rec">
        {rec ? (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <RetroBadge recordType={rec.record_type} short />
              <span className="num" style={{ fontSize: 11 }}>
                {label}
              </span>
            </div>
            <div style={{ fontWeight: 600, fontSize: 14.5 }}>{rec.display_text}</div>
            <div className="num" style={{ fontSize: 12, marginTop: 6 }}>
              {rec.source_format} · {fmtDate(rec.effective_date)}
            </div>
          </>
        ) : (
          <span className="muted" style={{ fontSize: 13 }}>
            {label} unavailable
          </span>
        )}
      </div>
    </>
  );
}

/* ==========================================
   SYSTEM TAB
   ========================================== */

function maskEmail(email: string | undefined | null): string {
  if (!email) return "—";
  const [local, domain] = email.split("@");
  if (!domain) return email;
  const head = local.slice(0, 1);
  return `${head}${"•".repeat(Math.max(3, local.length - 1))}@${domain}`;
}

function SystemTab() {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [auditLog, setAuditLog] = useState<AuditLogEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const { clearTokens } = useAuthStore();

  useEffect(() => {
    Promise.all([
      api.get<UserResponse>("/auth/me").catch(() => null),
      api.get<DashboardOverview>("/dashboard/overview").catch(() => null),
    ])
      .then(([userData, overviewData]) => {
        setUser(userData);
        setOverview(overviewData);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    setAuditLoading(true);
    api
      .get<AuditLogResponse>("/audit-log?limit=50")
      .then((data) => setAuditLog(data.items || []))
      .catch(() => setAuditLog([]))
      .finally(() => setAuditLoading(false));
  }, []);

  const handleExport = async () => {
    setExporting(true);
    try {
      const bundle = await api.get<unknown>("/records/export?format=fhir-bundle");
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: "application/fhir+json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "medtimeline-fhir-bundle.json";
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Best-effort: surface nothing on failure beyond resetting state.
    } finally {
      setExporting(false);
    }
  };

  const handleSignOut = async () => {
    // API → POST /auth/logout (revokes the refresh token server-side), then clear local tokens.
    try {
      await api.post("/auth/logout");
    } catch {
      // Best-effort: still clear locally even if the call fails.
    }
    clearTokens();
    window.location.href = "/login";
  };

  if (loading) return <RetroLoadingState text="Loading system info" />;

  const span =
    overview?.date_range_start && overview?.date_range_end
      ? (() => {
          const s = new Date(overview.date_range_start).getFullYear();
          const e = new Date(overview.date_range_end).getFullYear();
          return s === e ? `${s}` : `${s}–${e}`;
        })()
      : "—";

  const recordTypeCount = overview ? Object.keys(overview.records_by_type).length : 0;

  return (
    <div>
      <div className="grid-2" style={{ alignItems: "start", marginBottom: 18 }}>
        {/* Account */}
        <div className="card-surface pad">
          <h3 className="sec-title" style={{ marginBottom: 14 }}>
            Account
          </h3>
          <div className="s12">
            <Field l="Name" v={user?.display_name?.trim() || "Not set"} />
            <Field l="Email" v={maskEmail(user?.email)} />
            <Field l="Record owner" v="You" />
            {/* TODO(backend): created_at not on /auth/me yet */}
            <Field l="Member since" v="—" />
          </div>
        </div>

        {/* This record */}
        <div className="card-surface pad">
          <h3 className="sec-title" style={{ marginBottom: 14 }}>
            This record
          </h3>
          <div className="s12">
            <Field
              l="Records"
              v={`${overview?.total_records ?? 0} across ${recordTypeCount} ${
                recordTypeCount === 1 ? "type" : "types"
              }`}
            />
            <Field l="Sources" v={String(overview?.total_uploads ?? 0)} />
            <Field l="Span" v={span} />
          </div>
        </div>
      </div>

      {/* Your data, your control */}
      <div className="card-surface pad" style={{ marginBottom: 18 }}>
        <h3 className="sec-title" style={{ marginBottom: 6 }}>
          Your data, your control
        </h3>
        <p className="h-sub" style={{ margin: "0 0 16px" }}>
          Export everything in an open format, or sign out of this device. All health data is stored
          locally and encrypted at rest; it only leaves your device when you explicitly request an AI
          summary or document extraction, and only after de-identification.
        </p>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <button className="btn" onClick={handleExport} disabled={exporting}>
            <Upload size={15} style={{ transform: "rotate(180deg)" }} />{" "}
            {exporting ? "Exporting…" : "Export all (FHIR)"}
          </button>
          <button className="btn ghost" onClick={handleSignOut}>
            Sign out
          </button>
        </div>
      </div>

      {/* Audit log */}
      <div className="card-surface pad">
        <div className="card-h">
          <h3 className="sec-title">Audit log</h3>
          <span className="num" style={{ fontSize: 11 }}>
            recent events
          </span>
        </div>
        {auditLoading ? (
          <RetroLoadingState text="Loading audit log" />
        ) : auditLog.length === 0 ? (
          <div className="muted" style={{ fontSize: 13.5, padding: "20px 0" }}>
            No audit events recorded yet.
          </div>
        ) : (
          <div className="tablewrap" style={{ boxShadow: "none", border: 0, padding: 0 }}>
            <table className="rtable">
              <thead>
                <tr>
                  <th>Action</th>
                  <th>Resource</th>
                  <th>When</th>
                  <th>IP</th>
                </tr>
              </thead>
              <tbody>
                {auditLog.map((e) => {
                  const resource = e.resource_type
                    ? e.resource_id
                      ? `${e.resource_type} · ${e.resource_id.slice(0, 8)}`
                      : e.resource_type
                    : "—";
                  return (
                    <tr key={e.id} style={{ cursor: "default" }}>
                      <td className="desc">{e.action}</td>
                      <td className="num">{resource}</td>
                      <td className="num">
                        {e.created_at ? new Date(e.created_at).toLocaleString() : "—"}
                      </td>
                      <td className="num">{e.ip_address || "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ l, v }: { l: string; v: string }) {
  return (
    <div className="field">
      <div className="field-l">{l}</div>
      <div className="field-v">{v}</div>
    </div>
  );
}
