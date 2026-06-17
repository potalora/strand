"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import {
  Search,
  Trash2,
  Upload,
  FileStack,
  Copy,
  Settings,
  Check,
  ChevronDown,
  ArrowLeftRight,
  ArrowRight,
  GitMerge,
  Undo2,
  Sun,
  Moon,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/useAuthStore";
import { useUserStore } from "@/stores/useUserStore";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import type {
  RecordListResponse,
  HealthRecord,
  DedupCandidate,
  DedupSummary,
  ScanResponse,
  BulkResolveResponse,
  MergeRecord,
  MergesResponse,
  UndoBulkResponse,
  DashboardOverview,
  AuditLogResponse,
  AuditLogEntry,
} from "@/types/api";
import { RECORD_TYPE_LABELS } from "@/lib/constants";
import { sourceLabel } from "@/lib/source-label";
import { recordTitle } from "@/lib/record-title";
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
  { key: "dedup", label: "Deduplication", icon: Copy },
  { key: "sys", label: "System", icon: Settings },
];

type DedupSub = "pending" | "ledger";

// Map the `?tab=` (and `?sub=`) query into a concrete (tab, sub) pair. The old
// standalone "Merges" tab is gone, so a legacy `?tab=merges` deep-link resolves
// to the Deduplication tab's "Merge ledger" sub-tab (bookmarks keep working).
function resolveTabState(rawTab: string | null, rawSub: string | null): {
  tab: string;
  sub: DedupSub;
} {
  if (rawTab === "merges") return { tab: "dedup", sub: "ledger" };
  return { tab: rawTab || "records", sub: rawSub === "ledger" ? "ledger" : "pending" };
}

export default function AdminPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initial = resolveTabState(searchParams.get("tab"), searchParams.get("sub"));
  const [activeTab, setActiveTab] = useState(initial.tab);
  const [dedupSub, setDedupSub] = useState<DedupSub>(initial.sub);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    const next = resolveTabState(searchParams.get("tab"), searchParams.get("sub"));
    setActiveTab(next.tab);
    setDedupSub(next.sub);
  }, [searchParams]);

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    const url = key === "dedup" ? `/admin?tab=dedup&sub=${dedupSub}` : `/admin?tab=${key}`;
    router.replace(url, { scroll: false });
  };

  const handleSubChange = (sub: DedupSub) => {
    setDedupSub(sub);
    router.replace(`/admin?tab=dedup&sub=${sub}`, { scroll: false });
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
      {activeTab === "dedup" && (
        <DeduplicationTab sub={dedupSub} onSubChange={handleSubChange} />
      )}
      {activeTab === "sys" && <SystemTab />}
    </div>
  );
}

/* ==========================================
   DEDUPLICATION TAB — Pending review + Merge ledger sub-tabs
   ========================================== */

// Wraps the two dedup panes under one top tab. A segmented pill control (the
// established `.filt` vocabulary, also used by the ledger's Auto/Manual toggle)
// switches sub-views, reading as clearly secondary to the top `.tabs` underline.
function DeduplicationTab({
  sub,
  onSubChange,
}: {
  sub: DedupSub;
  onSubChange: (sub: DedupSub) => void;
}) {
  const SUBS: { key: DedupSub; label: string }[] = [
    { key: "pending", label: "Pending review" },
    { key: "ledger", label: "Merge ledger" },
  ];

  return (
    <div>
      <div
        role="group"
        aria-label="Deduplication views"
        style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}
      >
        {SUBS.map((s) => (
          <button
            key={s.key}
            className="filt"
            aria-pressed={sub === s.key}
            onClick={() => onSubChange(s.key)}
            style={{ textTransform: "none" }}
          >
            {s.label}
          </button>
        ))}
      </div>

      {sub === "pending" ? <DedupTab /> : <DedupMergesTab />}
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

  // API → GET /records (server-sorted). The tree/table needs every record, but
  // the endpoint caps page_size at 100, so we page through and accumulate
  // (a single page_size=500 request 422s and left the tab empty). Column sort
  // is sent to the server via ?sort=&order=; search + type filter stay
  // client-side over the loaded set.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const order = sort.dir === 1 ? "asc" : "desc";
    const sortKey = SORT_API_KEY[sort.key];
    const PAGE_SIZE = 100; // backend maximum

    (async () => {
      try {
        const all: HealthRecord[] = [];
        for (let page = 1; page <= 1000; page++) {
          const data = await api.get<RecordListResponse>(
            `/records?page=${page}&page_size=${PAGE_SIZE}&sort=${sortKey}&order=${order}`
          );
          const items = data.items || [];
          all.push(...items);
          if (items.length < PAGE_SIZE || all.length >= (data.total ?? all.length)) break;
        }
        if (!cancelled) setRecords(all);
      } catch {
        if (!cancelled) setRecords([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
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
                <td className="desc">{recordTitle(r)}</td>
                <td className="num">{fmtDate(r.effective_date)}</td>
                <td className="num">{sourceLabel(r.source_format)}</td>
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

const BAND_LIMIT = 20;

// Score band → the [score_min, score_max) fraction window the API expects.
function bandRange(band: number): { score_min: number; score_max: number } {
  return { score_min: band / 100, score_max: (band + 10) / 100 };
}

// A pending bulk action awaiting confirmation: either an entire score band
// (resolved server-side by score window) or an explicit set of checked rows.
type PendingBulk =
  | { kind: "band"; action: "merge" | "dismiss"; band: number; count: number }
  | { kind: "selection"; action: "merge" | "dismiss"; ids: string[] };

function bulkMeta(pb: PendingBulk): {
  title: string;
  description: string;
  confirmLabel: string;
} {
  const n = pb.kind === "band" ? pb.count : pb.ids.length;
  const noun = n === 1 ? "candidate" : "candidates";
  const scope = pb.kind === "band" ? `in the ${pb.band}% band` : "you selected";
  if (pb.action === "merge") {
    return {
      title: `Merge all ${n.toLocaleString()} ${noun}?`,
      description: `Merge the ${n.toLocaleString()} ${noun} ${scope}. This marks the lower-confidence duplicate of each pair as merged. Reversible — each merge can be undone individually.`,
      confirmLabel: "Merge all",
    };
  }
  return {
    title: `Dismiss all ${n.toLocaleString()} ${noun}?`,
    description: `Dismiss the ${n.toLocaleString()} ${noun} ${scope}. Both records are kept and the pair is cleared from the review queue.`,
    confirmLabel: "Dismiss all",
  };
}

function DedupTab() {
  // `summary === null` means "not loaded yet" → the only time we show the
  // full-tab spinner. Once loaded it stays non-null, so refreshes after an
  // action keep the band list on screen (the open band shows its own spinner).
  const [summary, setSummary] = useState<DedupSummary | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Open-band accordion + its lazily-loaded, paginated candidate page.
  const [openBand, setOpenBand] = useState<number | null>(null);
  const [bandItems, setBandItems] = useState<DedupCandidate[]>([]);
  const [bandTotal, setBandTotal] = useState(0);
  const [bandPage, setBandPage] = useState(1);
  const [bandLoading, setBandLoading] = useState(false);

  // Per-page row selection + in-flight flags.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rowBusy, setRowBusy] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  const [pendingBulk, setPendingBulk] = useState<PendingBulk | null>(null);

  // API → GET /dedup/candidates/summary (per-band counts, descending).
  const fetchSummary = useCallback(async (): Promise<DedupSummary | null> => {
    try {
      const data = await api.get<DedupSummary>("/dedup/candidates/summary");
      setSummary(data);
      return data;
    } catch {
      setSummary({ bands: [], total: 0 });
      return null;
    }
  }, []);

  // API → GET /dedup/candidates?score_min&score_max&page&limit. Clamps the
  // requested page to the last valid page so a bulk action that shrinks a band
  // never strands us on an empty page.
  const fetchBand = useCallback(async (band: number, requestedPage: number) => {
    const { score_min, score_max } = bandRange(band);
    setBandLoading(true);
    try {
      const url = (p: number) =>
        `/dedup/candidates?score_min=${score_min}&score_max=${score_max}&page=${p}&limit=${BAND_LIMIT}`;
      let page = Math.max(1, requestedPage);
      let data = await api.get<{ items: DedupCandidate[]; total: number }>(url(page));
      let total = data.total || 0;
      const lastPage = Math.max(1, Math.ceil(total / BAND_LIMIT));
      if (page > lastPage && total > 0) {
        page = lastPage;
        data = await api.get<{ items: DedupCandidate[]; total: number }>(url(page));
        total = data.total || 0;
      }
      setBandItems(data.items || []);
      setBandTotal(total);
      setBandPage(page);
    } catch {
      setBandItems([]);
      setBandTotal(0);
    } finally {
      setBandLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchSummary();
  }, [fetchSummary]);

  // After any resolution, re-pull the summary and, if the open band survives,
  // its current page; otherwise collapse it. Always clears selection.
  const refreshAfterAction = useCallback(async () => {
    setSelected(new Set());
    const next = await fetchSummary();
    const bands = next?.bands ?? [];
    if (openBand != null) {
      if (bands.some((b) => b.band === openBand)) {
        await fetchBand(openBand, bandPage);
      } else {
        setOpenBand(null);
        setBandItems([]);
        setBandTotal(0);
      }
    }
  }, [fetchSummary, fetchBand, openBand, bandPage]);

  const handleScan = async () => {
    setScanning(true);
    setError(null);
    setScanResult(null);
    try {
      const result = await api.post<ScanResponse>("/dedup/scan");
      setScanResult(result);
      setOpenBand(null);
      setBandItems([]);
      setBandTotal(0);
      setSelected(new Set());
      await fetchSummary();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setScanning(false);
    }
  };

  const toggleBand = (band: number) => {
    setError(null);
    setSelected(new Set());
    if (openBand === band) {
      setOpenBand(null);
      setBandItems([]);
      setBandTotal(0);
      return;
    }
    setOpenBand(band);
    setBandPage(1);
    void fetchBand(band, 1);
  };

  const goToPage = (p: number) => {
    if (openBand == null) return;
    setSelected(new Set());
    void fetchBand(openBand, p);
  };

  const handleRowAction = async (action: "merge" | "dismiss", candidateId: string) => {
    setRowBusy(candidateId);
    setError(null);
    try {
      await api.post(action === "merge" ? "/dedup/merge" : "/dedup/dismiss", {
        candidate_id: candidateId,
      });
      await refreshAfterAction();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Could not ${action} candidate`);
    } finally {
      setRowBusy(null);
    }
  };

  const confirmBulk = async () => {
    if (!pendingBulk) return;
    const pb = pendingBulk;
    setPendingBulk(null);
    setBulkBusy(true);
    setError(null);
    try {
      const body =
        pb.kind === "band"
          ? { action: pb.action, ...bandRange(pb.band) }
          : { action: pb.action, candidate_ids: pb.ids };
      await api.post<BulkResolveResponse>("/dedup/resolve-bulk", body);
      await refreshAfterAction();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Bulk action failed");
    } finally {
      setBulkBusy(false);
    }
  };

  const toggleRow = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const allOnPageSelected =
    bandItems.length > 0 && bandItems.every((c) => selected.has(c.id));

  const toggleSelectAllOnPage = () =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (bandItems.every((c) => next.has(c.id))) {
        bandItems.forEach((c) => next.delete(c.id));
      } else {
        bandItems.forEach((c) => next.add(c.id));
      }
      return next;
    });

  const lastPage = Math.max(1, Math.ceil(bandTotal / BAND_LIMIT));

  return (
    <div>
      <p className="h-sub" style={{ margin: "0 0 18px" }}>
        Potential duplicates found across sources. Each is scored, then auto-merged, auto-dismissed,
        or grouped by confidence below for your review.
      </p>

      <div className="toolbar">
        <button className="btn" onClick={handleScan} disabled={scanning}>
          {scanning ? "Scanning…" : "Scan for duplicates"}
        </button>
        {scanResult && (
          <span
            className="h-sub"
            style={{
              margin: 0,
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
            }}
          >
            {scanResult.auto_merged > 0 && (
              <>
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 6,
                    color: "var(--success)",
                    fontWeight: 600,
                  }}
                >
                  <Check size={15} />
                  {scanResult.auto_merged.toLocaleString()} exact{" "}
                  {scanResult.auto_merged === 1 ? "duplicate" : "duplicates"} auto-merged
                </span>
                <span style={{ color: "var(--text-muted)" }}>·</span>
              </>
            )}
            <span style={{ color: "var(--text-dim)" }}>
              {scanResult.candidates_found.toLocaleString()} sent to review
            </span>
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

      {summary === null ? (
        <RetroLoadingState text="Loading duplicates" />
      ) : summary.total === 0 ? (
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
              {summary.total.toLocaleString()} pending{" "}
              {summary.total === 1 ? "candidate" : "candidates"} across {summary.bands.length}{" "}
              {summary.bands.length === 1 ? "confidence band" : "confidence bands"}
            </span>
          </div>

          {summary.bands.map((b) => {
            const isOpen = openBand === b.band;
            return (
              <div key={b.band} className="card-surface" style={{ marginBottom: 12 }}>
                <div className="between" style={{ padding: "14px 18px", gap: 12 }}>
                  <button
                    onClick={() => toggleBand(b.band)}
                    aria-expanded={isOpen}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      background: "transparent",
                      border: 0,
                      padding: 0,
                      cursor: "pointer",
                      textAlign: "left",
                      flex: 1,
                      minWidth: 0,
                    }}
                  >
                    <ChevronDown
                      size={16}
                      style={{
                        color: "var(--text-muted)",
                        flexShrink: 0,
                        transition: "transform 0.2s ease",
                        transform: isOpen ? "rotate(180deg)" : "none",
                      }}
                    />
                    <span
                      className="display"
                      style={{ fontSize: 19, color: "var(--primary)", whiteSpace: "nowrap" }}
                    >
                      {b.band}% match
                    </span>
                    <span className="tag">
                      {b.count.toLocaleString()} {b.count === 1 ? "pair" : "pairs"}
                    </span>
                  </button>
                  <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
                    <button
                      className="btn ghost sm"
                      disabled={bulkBusy}
                      onClick={() =>
                        setPendingBulk({
                          kind: "band",
                          action: "merge",
                          band: b.band,
                          count: b.count,
                        })
                      }
                    >
                      Merge all
                    </button>
                    <button
                      className="btn ghost sm"
                      disabled={bulkBusy}
                      onClick={() =>
                        setPendingBulk({
                          kind: "band",
                          action: "dismiss",
                          band: b.band,
                          count: b.count,
                        })
                      }
                    >
                      Dismiss all
                    </button>
                  </div>
                </div>

                {isOpen && (
                  <div style={{ padding: "0 18px 18px" }}>
                    {bandLoading ? (
                      <RetroLoadingState text="Loading candidates" />
                    ) : bandItems.length === 0 ? (
                      <p className="muted" style={{ fontSize: 13.5, padding: "8px 0 4px" }}>
                        No candidates in this band.
                      </p>
                    ) : (
                      <>
                        <div className="between" style={{ marginBottom: 12 }}>
                          <label
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              cursor: "pointer",
                              fontSize: 13,
                              color: "var(--text-dim)",
                              fontWeight: 600,
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={allOnPageSelected}
                              onChange={toggleSelectAllOnPage}
                              style={{ accentColor: "var(--primary)" }}
                              aria-label="Select all candidates on this page"
                            />
                            Select page
                          </label>
                          <span className="num" style={{ fontSize: 12 }}>
                            {(bandPage - 1) * BAND_LIMIT + 1}–
                            {Math.min(bandPage * BAND_LIMIT, bandTotal)} of{" "}
                            {bandTotal.toLocaleString()}
                          </span>
                        </div>

                        {selected.size > 0 && (
                          <div
                            className="between"
                            style={{
                              background: "var(--primary-soft)",
                              border: "1px solid var(--border)",
                              borderRadius: "var(--radius-sm)",
                              padding: "10px 14px",
                              marginBottom: 12,
                              gap: 10,
                              flexWrap: "wrap",
                            }}
                          >
                            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
                              {selected.size} selected
                            </span>
                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                              <button
                                className="btn sm"
                                disabled={bulkBusy}
                                onClick={() =>
                                  setPendingBulk({
                                    kind: "selection",
                                    action: "merge",
                                    ids: Array.from(selected),
                                  })
                                }
                              >
                                Merge {selected.size} selected
                              </button>
                              <button
                                className="btn ghost sm"
                                disabled={bulkBusy}
                                onClick={() =>
                                  setPendingBulk({
                                    kind: "selection",
                                    action: "dismiss",
                                    ids: Array.from(selected),
                                  })
                                }
                              >
                                Dismiss {selected.size} selected
                              </button>
                              <button
                                className="btn ghost sm"
                                disabled={bulkBusy}
                                onClick={() => setSelected(new Set())}
                              >
                                Clear selection
                              </button>
                            </div>
                          </div>
                        )}

                        {bandItems.map((c) => (
                          <DedupRow
                            key={c.id}
                            candidate={c}
                            checked={selected.has(c.id)}
                            busy={rowBusy === c.id}
                            disabled={bulkBusy}
                            onToggle={() => toggleRow(c.id)}
                            onMerge={() => handleRowAction("merge", c.id)}
                            onKeepBoth={() => handleRowAction("dismiss", c.id)}
                          />
                        ))}

                        <div className="between" style={{ marginTop: 12 }}>
                          <button
                            className="btn ghost sm"
                            disabled={bandPage <= 1 || bandLoading}
                            onClick={() => goToPage(bandPage - 1)}
                          >
                            Prev
                          </button>
                          <span className="num" style={{ fontSize: 12 }}>
                            Page {bandPage} of {lastPage}
                          </span>
                          <button
                            className="btn ghost sm"
                            disabled={bandPage >= lastPage || bandLoading}
                            onClick={() => goToPage(bandPage + 1)}
                          >
                            Next
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <ConfirmDialog
        open={!!pendingBulk}
        title={pendingBulk ? bulkMeta(pendingBulk).title : ""}
        description={pendingBulk ? bulkMeta(pendingBulk).description : ""}
        confirmLabel={pendingBulk ? bulkMeta(pendingBulk).confirmLabel : "Confirm"}
        cancelLabel="Cancel"
        onConfirm={confirmBulk}
        onCancel={() => setPendingBulk(null)}
      />
    </div>
  );
}

// Legacy full-width candidate card — still used by the standalone /dedup page.
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
        <div className="dedup-rec">
          <RecordMini rec={candidate.record_a} />
        </div>
        <div className="dedup-vs">VS</div>
        <div className="dedup-rec">
          <RecordMini rec={candidate.record_b} />
        </div>
      </div>
    </div>
  );
}

// One candidate as a tight, checkbox-selectable row inside a confidence band.
// The A↔B comparison reuses RecordMini; actions mirror the legacy card.
function DedupRow({
  candidate,
  checked,
  busy,
  disabled,
  onToggle,
  onMerge,
  onKeepBoth,
}: {
  candidate: DedupCandidate;
  checked: boolean;
  busy: boolean;
  disabled: boolean;
  onToggle: () => void;
  onMerge: () => void;
  onKeepBoth: () => void;
}) {
  const reasons = Object.entries(candidate.match_reasons)
    .filter(([, matched]) => matched)
    .map(([reason]) => reason);

  return (
    <div
      style={{
        background: "var(--card-2)",
        border: `1px solid ${checked ? "var(--primary)" : "var(--border)"}`,
        borderRadius: "var(--radius-sm)",
        padding: "12px 14px",
        marginBottom: 8,
        transition: "border-color 0.16s",
      }}
    >
      <div
        className="between"
        style={{ gap: 12, marginBottom: 10, alignItems: "flex-start" }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            cursor: "pointer",
            minWidth: 0,
            flexWrap: "wrap",
          }}
        >
          <input
            type="checkbox"
            checked={checked}
            onChange={onToggle}
            disabled={disabled}
            style={{ accentColor: "var(--primary)", flexShrink: 0 }}
            aria-label="Select candidate"
          />
          <span
            className="num"
            style={{ fontSize: 12, color: "var(--primary)", fontWeight: 700 }}
          >
            {Math.round(candidate.similarity_score * 100)}%
          </span>
          {reasons.slice(0, 4).map((r) => (
            <span className="reason" key={r}>
              {r}
            </span>
          ))}
        </label>
        <div style={{ display: "flex", gap: 8, flexShrink: 0 }}>
          <button className="btn sm" onClick={onMerge} disabled={busy || disabled}>
            {busy ? "…" : "Merge"}
          </button>
          <button className="btn ghost sm" onClick={onKeepBoth} disabled={busy || disabled}>
            Keep both
          </button>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          gap: 12,
          alignItems: "center",
        }}
      >
        <RecordMini rec={candidate.record_a} />
        <ArrowLeftRight size={15} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
        <RecordMini rec={candidate.record_b} />
      </div>
    </div>
  );
}

// Compact single-record cell for the A↔B comparison (badge + title + meta).
export function RecordMini({ rec }: { rec: DedupCandidate["record_a"] }) {
  if (!rec) {
    return (
      <span className="muted" style={{ fontSize: 13 }}>
        Record unavailable
      </span>
    );
  }
  return (
    <div style={{ minWidth: 0 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 4,
          minWidth: 0,
        }}
      >
        <RetroBadge recordType={rec.record_type} short />
        <span
          style={{
            fontWeight: 600,
            fontSize: 13.5,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {recordTitle(rec)}
        </span>
      </div>
      <div className="num" style={{ fontSize: 11.5 }}>
        {sourceLabel(rec.source_format)} · {fmtDate(rec.effective_date)}
      </div>
    </div>
  );
}

/* ==========================================
   MERGES TAB — completed-merge history + unmerge
   ========================================== */

const MERGES_LIMIT = 20;

// Record types offered in the Merges type filter — the common clinical ones
// (the queue rarely produces merges outside this set; "All types" clears it).
const MERGE_TYPE_OPTIONS = [
  "condition",
  "observation",
  "medication",
  "procedure",
  "encounter",
  "immunization",
  "document",
  "allergy",
  "service_request",
  "diagnostic_report",
];

type MergeSource = "" | "auto" | "manual";

function DedupMergesTab() {
  // `data === null` only before the first load → the only full-tab spinner.
  // It stays non-null afterward so filter changes / refreshes keep the list on
  // screen (dimmed via `loading`) instead of flashing the spinner.
  const [data, setData] = useState<MergesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters. `source` is the segmented Auto/Manual toggle; "" = both.
  const [source, setSource] = useState<MergeSource>("");
  const [recordType, setRecordType] = useState("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(1);
  const [refreshKey, setRefreshKey] = useState(0);

  // Per-page selection + in-flight flags.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rowBusy, setRowBusy] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  // A pending bulk unmerge awaiting destructive confirmation (the candidate ids).
  const [pendingBulk, setPendingBulk] = useState<string[] | null>(null);

  // Debounce the search box (~300ms); settling the query resets to page 1.
  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [search]);

  // API → GET /dedup/merges?source=&record_type=&search=&page=&limit=.
  // Re-runs on any filter/page change and after each unmerge (refreshKey).
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const params = new URLSearchParams();
        if (source) params.set("source", source);
        if (recordType) params.set("record_type", recordType);
        const q = debouncedSearch.trim();
        if (q) params.set("search", q);
        params.set("page", String(page));
        params.set("limit", String(MERGES_LIMIT));
        const resp = await api.get<MergesResponse>(`/dedup/merges?${params.toString()}`);
        if (cancelled) return;
        // Clamp to the last valid page so an unmerge that empties the tail page
        // never strands us on a blank view.
        const lastPage = Math.max(1, Math.ceil((resp.total || 0) / MERGES_LIMIT));
        if (page > lastPage && resp.total > 0) {
          setPage(lastPage);
          return;
        }
        setData(resp);
        setError(null);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load merges");
          setData({ items: [], total: 0, counts: { auto: 0, manual: 0 } });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [source, recordType, debouncedSearch, page, refreshKey]);

  const refreshAfterUnmerge = useCallback(() => {
    setSelected(new Set());
    setRefreshKey((k) => k + 1);
  }, []);

  const changeSource = (next: MergeSource) => {
    setSource(next);
    setPage(1);
    setSelected(new Set());
  };

  const changeType = (next: string) => {
    setRecordType(next);
    setPage(1);
    setSelected(new Set());
  };

  const clearFilters = () => {
    setSource("");
    setRecordType("");
    setSearch("");
    setDebouncedSearch("");
    setPage(1);
    setSelected(new Set());
  };

  const handleUnmergeOne = async (candidateId: string) => {
    setRowBusy(candidateId);
    setError(null);
    try {
      await api.post("/dedup/undo-merge", { candidate_id: candidateId });
      refreshAfterUnmerge();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not unmerge");
    } finally {
      setRowBusy(null);
    }
  };

  const confirmBulk = async () => {
    if (!pendingBulk) return;
    const ids = pendingBulk;
    setPendingBulk(null);
    setBulkBusy(true);
    setError(null);
    try {
      await api.post<UndoBulkResponse>("/dedup/undo-bulk", { candidate_ids: ids });
      refreshAfterUnmerge();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Bulk unmerge failed");
    } finally {
      setBulkBusy(false);
    }
  };

  const toggleRow = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const autoCount = data?.counts.auto ?? 0;
  const manualCount = data?.counts.manual ?? 0;
  // The "All" chip counts both buckets so it stays stable while toggling source.
  const allCount = autoCount + manualCount;
  const lastPage = Math.max(1, Math.ceil(total / MERGES_LIMIT));
  const hasFilters = source !== "" || recordType !== "" || debouncedSearch.trim() !== "";

  const allOnPageSelected =
    items.length > 0 && items.every((m) => selected.has(m.candidate_id));
  const toggleSelectAllOnPage = () =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (items.every((m) => next.has(m.candidate_id))) {
        items.forEach((m) => next.delete(m.candidate_id));
      } else {
        items.forEach((m) => next.add(m.candidate_id));
      }
      return next;
    });

  return (
    <div>
      <p className="h-sub" style={{ margin: "0 0 18px" }}>
        Every merge the system made — auto-merged exact duplicates and the ones you merged. Unmerge
        any to keep the records separate.
      </p>

      <div className="toolbar">
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {(
            [
              { key: "" as MergeSource, label: "All", count: allCount },
              { key: "auto" as MergeSource, label: "Auto", count: autoCount },
              { key: "manual" as MergeSource, label: "Manual", count: manualCount },
            ]
          ).map((chip) => (
            <button
              key={chip.label}
              className="filt"
              aria-pressed={source === chip.key}
              onClick={() => changeSource(chip.key)}
            >
              {chip.label}{" "}
              <span
                style={{
                  fontFamily: "var(--font-mono), monospace",
                  fontSize: 11.5,
                  opacity: 0.7,
                }}
              >
                {chip.count.toLocaleString()}
              </span>
            </button>
          ))}
        </div>
        <select
          className="selectbox"
          value={recordType}
          onChange={(e) => changeType(e.target.value)}
          aria-label="Filter merges by record type"
        >
          <option value="">All types</option>
          {MERGE_TYPE_OPTIONS.map((t) => (
            <option key={t} value={t}>
              {RECORD_TYPE_LABELS[t] || t}
            </option>
          ))}
        </select>
        <div className="search">
          <Search size={16} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search merged records…"
          />
        </div>
      </div>

      {error && (
        <div className="card-surface pad" style={{ marginBottom: 14 }}>
          <div className="between" style={{ justifyContent: "flex-start", gap: 12 }}>
            <span
              className="tag"
              style={{ background: "var(--danger)", color: "var(--on-primary)" }}
            >
              ERROR
            </span>
            <p className="muted" style={{ fontSize: 13.5, margin: 0 }}>
              {error}
            </p>
          </div>
        </div>
      )}

      {data === null ? (
        <RetroLoadingState text="Loading merges" />
      ) : total === 0 ? (
        <div className="card-surface pad" style={{ textAlign: "center", padding: "56px 24px" }}>
          <span className="dz-ic" style={{ margin: "0 auto 14px" }}>
            <GitMerge size={22} />
          </span>
          {hasFilters ? (
            <>
              <div className="muted" style={{ fontSize: 14.5 }}>
                No merges match these filters.
              </div>
              <div style={{ marginTop: 14 }}>
                <button className="btn ghost sm" onClick={clearFilters}>
                  Clear filters
                </button>
              </div>
            </>
          ) : (
            <div className="muted" style={{ fontSize: 14.5 }}>
              No merges yet — run a scan from the Pending review tab.
            </div>
          )}
        </div>
      ) : (
        <div style={{ opacity: loading ? 0.55 : 1, transition: "opacity 0.15s ease" }}>
          {selected.size > 0 && (
            <div
              className="between"
              style={{
                background: "var(--primary-soft)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "10px 14px",
                marginBottom: 12,
                gap: 10,
                flexWrap: "wrap",
              }}
            >
              <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text)" }}>
                {selected.size} selected
              </span>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <button
                  className="btn sm"
                  disabled={bulkBusy}
                  onClick={() => setPendingBulk(Array.from(selected))}
                >
                  Unmerge {selected.size} selected
                </button>
                <button
                  className="btn ghost sm"
                  disabled={bulkBusy}
                  onClick={() => setSelected(new Set())}
                >
                  Clear selection
                </button>
              </div>
            </div>
          )}

          <div className="between" style={{ marginBottom: 12 }}>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
                fontSize: 13,
                color: "var(--text-dim)",
                fontWeight: 600,
              }}
            >
              <input
                type="checkbox"
                checked={allOnPageSelected}
                onChange={toggleSelectAllOnPage}
                style={{ accentColor: "var(--primary)" }}
                aria-label="Select all merges on this page"
              />
              Select page
            </label>
            <span className="num" style={{ fontSize: 12 }}>
              {(page - 1) * MERGES_LIMIT + 1}–{Math.min(page * MERGES_LIMIT, total)} of{" "}
              {total.toLocaleString()}
            </span>
          </div>

          {items.map((m) => (
            <MergeRow
              key={m.candidate_id}
              merge={m}
              checked={selected.has(m.candidate_id)}
              busy={rowBusy === m.candidate_id}
              disabled={bulkBusy}
              onToggle={() => toggleRow(m.candidate_id)}
              onUnmerge={() => handleUnmergeOne(m.candidate_id)}
            />
          ))}

          <div className="between" style={{ marginTop: 12 }}>
            <button
              className="btn ghost sm"
              disabled={page <= 1 || loading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Prev
            </button>
            <span className="num" style={{ fontSize: 12 }}>
              Page {page} of {lastPage}
            </span>
            <button
              className="btn ghost sm"
              disabled={page >= lastPage || loading}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </button>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={!!pendingBulk}
        title={`Unmerge ${pendingBulk?.length ?? 0} ${
          (pendingBulk?.length ?? 0) === 1 ? "record" : "records"
        }?`}
        description="They'll be restored as separate records and kept out of future merge suggestions."
        confirmLabel="Unmerge"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={confirmBulk}
        onCancel={() => setPendingBulk(null)}
      />
    </div>
  );
}

// One completed merge as a tight, checkbox-selectable row: an AUTO/YOU badge,
// score, match reasons, and a clear "archived → merged into → survivor" flow.
function MergeRow({
  merge,
  checked,
  busy,
  disabled,
  onToggle,
  onUnmerge,
}: {
  merge: MergeRecord;
  checked: boolean;
  busy: boolean;
  disabled: boolean;
  onToggle: () => void;
  onUnmerge: () => void;
}) {
  const reasons = Object.entries(merge.match_reasons)
    .filter(([, matched]) => Boolean(matched))
    .map(([reason]) => reason);

  return (
    <div
      style={{
        background: "var(--card-2)",
        border: `1px solid ${checked ? "var(--primary)" : "var(--border)"}`,
        borderRadius: "var(--radius-sm)",
        padding: "12px 14px",
        marginBottom: 8,
        transition: "border-color 0.16s",
      }}
    >
      <div className="between" style={{ gap: 12, marginBottom: 10, alignItems: "flex-start" }}>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            cursor: "pointer",
            minWidth: 0,
            flexWrap: "wrap",
          }}
        >
          <input
            type="checkbox"
            checked={checked}
            onChange={onToggle}
            disabled={disabled}
            style={{ accentColor: "var(--primary)", flexShrink: 0 }}
            aria-label="Select merge"
          />
          {merge.auto_resolved ? (
            <span className="tag">
              <span className="tdot" style={{ background: "var(--primary)" }} />
              AUTO
            </span>
          ) : (
            <span
              className="tag"
              style={{ background: "var(--primary)", color: "var(--on-primary)" }}
            >
              YOU
            </span>
          )}
          <span className="num" style={{ fontSize: 12, color: "var(--primary)", fontWeight: 700 }}>
            {Math.round(merge.similarity_score * 100)}%
          </span>
          {reasons.slice(0, 4).map((r) => (
            <span className="reason" key={r}>
              {r}
            </span>
          ))}
        </label>
        <button
          className="btn ghost sm"
          onClick={onUnmerge}
          disabled={busy || disabled}
          title="Restore both records as separate"
        >
          <Undo2 size={14} />
          {busy ? "…" : "Unmerge"}
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          gap: 12,
          alignItems: "center",
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div
            className="num"
            style={{
              fontSize: 9.5,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-muted)",
              marginBottom: 6,
            }}
          >
            Merged away
          </div>
          <RecordMini rec={merge.archived} />
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 3,
            color: "var(--text-muted)",
            flexShrink: 0,
          }}
        >
          <ArrowRight size={16} />
          <span style={{ fontSize: 9.5, fontFamily: "var(--font-mono), monospace" }}>
            merged into
          </span>
        </div>
        <div
          style={{
            minWidth: 0,
            paddingLeft: 12,
            borderLeft: "2px solid var(--success)",
          }}
        >
          <div
            className="num"
            style={{
              fontSize: 9.5,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--success)",
              marginBottom: 6,
            }}
          >
            Kept
          </div>
          <RecordMini rec={merge.survivor} />
        </div>
      </div>
    </div>
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
  // Account identity comes from the shared user store (fetched once, retried on
  // a transient 401) so the name never blanks. The overview gates this pane's
  // spinner; the name field shows its own skeleton until the user resolves.
  const { user, status: userStatus, fetchUser, clearUser } = useUserStore();
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [auditLog, setAuditLog] = useState<AuditLogEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [mounted, setMounted] = useState(false);
  const { clearTokens } = useAuthStore();
  const { theme, setTheme } = useTheme();
  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    void fetchUser();
  }, [fetchUser]);

  useEffect(() => {
    api
      .get<DashboardOverview>("/dashboard/overview")
      .catch(() => null)
      .then((overviewData) => setOverview(overviewData))
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
    clearUser();
    clearTokens();
    window.location.href = "/login";
  };

  if (loading) return <RetroLoadingState text="Loading system info" />;

  // Name: show "Not set" only once the user has genuinely loaded with an empty
  // display_name. While still loading (or after a transient error with no cached
  // user) show a skeleton — a transient 401 must NOT render a blank/"Not set".
  const nameNode: ReactNode = user
    ? user.display_name?.trim() || "Not set"
    : userStatus === "loaded"
      ? "Not set"
      : <FieldSkeleton width={108} />;
  const emailNode: ReactNode = user ? maskEmail(user.email) : <FieldSkeleton width={150} />;

  const isDark = mounted && theme === "dark";

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
            <Field l="Name" v={nameNode} />
            <Field l="Email" v={emailNode} />
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

      {/* Preferences (folded in from the former /settings page) */}
      <div className="card-surface pad" style={{ marginBottom: 18 }}>
        <h3 className="sec-title" style={{ marginBottom: 6 }}>
          Preferences
        </h3>

        {/* Appearance */}
        <div
          className="field"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div>
            <div className="field-l" style={{ marginBottom: 6 }}>
              Appearance
            </div>
            <div className="field-v" style={{ padding: 0 }}>
              {isDark ? "Dark" : "Light"} theme
            </div>
          </div>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setTheme(isDark ? "light" : "dark")}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun size={15} /> : <Moon size={15} />}
            Switch to {isDark ? "light" : "dark"}
          </button>
        </div>

        {/* Delete confirmation */}
        <div
          className="field"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div>
            <div className="field-l" style={{ marginBottom: 6 }}>
              Delete confirmation
            </div>
            <div className="field-v" style={{ padding: 0 }}>
              {skipDeleteConfirm
                ? "Off — records are removed without a prompt"
                : "On — confirm before removing a record"}
            </div>
          </div>
          <button
            type="button"
            className="btn ghost sm"
            onClick={() => setSkipDeleteConfirm(!skipDeleteConfirm)}
          >
            {skipDeleteConfirm ? "Re-enable prompt" : "Disable prompt"}
          </button>
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

function Field({ l, v }: { l: string; v: ReactNode }) {
  return (
    <div className="field">
      <div className="field-l">{l}</div>
      <div className="field-v">{v}</div>
    </div>
  );
}

// Loading placeholder for a field value — reuses the theme's `pulse-fade`
// keyframe so a still-loading account name shows a shimmer instead of "Not set".
function FieldSkeleton({ width = 120 }: { width?: number }) {
  return (
    <span
      aria-hidden
      style={{
        display: "inline-block",
        width,
        height: "0.9em",
        borderRadius: 4,
        background: "var(--bg-2)",
        animation: "pulse-fade 1.4s ease-in-out infinite",
        verticalAlign: "middle",
      }}
    />
  );
}
