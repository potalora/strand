"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { DashboardOverview, HealthRecord, RecordListResponse } from "@/types/api";
import { RECORD_TYPE_LABELS } from "@/lib/constants";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";

const PAGE_SIZE = 20;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Server-side sort keys accepted by GET /records?sort=...
type SortKey = "type" | "display_text" | "date";
type SortOrder = "asc" | "desc";

const fmtDate = (s: string | null) => {
  if (!s) return "--";
  const d = new Date(s);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
};

// ↑/↓ glyph shown only on the active sort column.
const sortIndicator = (col: SortKey, active: SortKey, order: SortOrder) =>
  col === active ? (order === "asc" ? " ↑" : " ↓") : "";

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

export default function RecordsPage() {
  const router = useRouter();
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [shown, setShown] = useState(false);

  // Filters
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [recordType, setRecordType] = useState("");
  const [typeOptions, setTypeOptions] = useState<string[]>([]);

  // Sort (server-side). Default: newest records first.
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  // Toggle a column header: same key flips direction, new key starts descending.
  // Reset to page 1 so the first page reflects the new ordering.
  const toggleSort = useCallback((key: SortKey) => {
    setSortKey((prevKey) => {
      if (prevKey === key) {
        setSortOrder((o) => (o === "asc" ? "desc" : "asc"));
      } else {
        setSortOrder("desc");
      }
      return key;
    });
    setPage(1);
  }, []);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  // Type filter options from the overview's records_by_type breakdown.
  useEffect(() => {
    api
      .get<DashboardOverview>("/dashboard/overview")
      .then((data) => {
        const types = Object.entries(data.records_by_type || {})
          .filter(([, count]) => count > 0)
          .map(([type]) => type)
          .sort((a, b) => a.localeCompare(b));
        setTypeOptions(types);
      })
      .catch(() => setTypeOptions([]));
  }, []);

  // Debounce the search input into the active search query.
  useEffect(() => {
    const t = setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 350);
    return () => clearTimeout(t);
  }, [searchInput]);

  useEffect(() => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(PAGE_SIZE),
      sort: sortKey,
      order: sortOrder,
    });
    if (recordType) params.set("record_type", recordType);
    if (search) params.set("search", search);
    api
      .get<RecordListResponse>(`/records?${params.toString()}`)
      .then((data) => {
        setRecords(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => {
        setRecords([]);
        setTotal(0);
      })
      .finally(() => setLoading(false));
  }, [page, search, recordType, sortKey, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const rangeStart = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const rangeEnd = Math.min(page * PAGE_SIZE, total);

  const showing = useMemo(
    () => (total === 0 ? "No records" : `${rangeStart}–${rangeEnd} of ${total.toLocaleString()}`),
    [total, rangeStart, rangeEnd]
  );

  return (
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">Records</p>
          <h1 className="h1 display">All records</h1>
        </div>
        <SecureChip />
      </div>

      {/* Toolbar: search + type filter */}
      <div className="toolbar">
        <div className="search">
          <Search size={15} />
          <input
            type="text"
            placeholder="Search records…"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
          />
        </div>
        <select
          className="selectbox"
          value={recordType}
          onChange={(e) => {
            setRecordType(e.target.value);
            setPage(1);
          }}
        >
          <option value="">All types</option>
          {typeOptions.map((type) => (
            <option key={type} value={type}>
              {RECORD_TYPE_LABELS[type] || type}
            </option>
          ))}
        </select>
      </div>

      {loading ? (
        <RetroLoadingState text="Loading records" />
      ) : records.length === 0 ? (
        <div className="card-surface pad">
          <div className="py-8 text-center">
            <p className="muted text-sm mb-3">
              {search || recordType ? "No records match your filters." : "No records on file."}
            </p>
            {!search && !recordType && (
              <button className="btn" onClick={() => router.push("/upload")}>
                Upload records
              </button>
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="tablewrap">
            <table className="rtable">
              <thead>
                <tr>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("type")}
                    aria-sort={
                      sortKey === "type"
                        ? sortOrder === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    Type{sortIndicator("type", sortKey, sortOrder)}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("display_text")}
                    aria-sort={
                      sortKey === "display_text"
                        ? sortOrder === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    Description{sortIndicator("display_text", sortKey, sortOrder)}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("date")}
                    aria-sort={
                      sortKey === "date"
                        ? sortOrder === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    Date{sortIndicator("date", sortKey, sortOrder)}
                  </th>
                  <th>Source</th>
                  <th>Code</th>
                </tr>
              </thead>
              <tbody>
                {records.map((r) => (
                  <tr key={r.id} className="clickable" onClick={() => setSelected(r.id)}>
                    <td>
                      <RetroBadge recordType={r.record_type} />
                    </td>
                    <td className="desc">{r.display_text}</td>
                    <td className="num">{fmtDate(r.effective_date)}</td>
                    <td className="muted">{r.source_format || "--"}</td>
                    <td className="num">{r.code_value || "--"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="between" style={{ marginTop: 16 }}>
            <span className="muted mono" style={{ fontSize: 12 }}>
              {showing}
            </span>
            <div className="chips" style={{ gap: 8 }}>
              <button className="btn ghost sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                Prev
              </button>
              <span className="muted mono" style={{ fontSize: 12, alignSelf: "center" }}>
                Page {page} of {totalPages}
              </span>
              <button className="btn ghost sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                Next
              </button>
            </div>
          </div>
        </>
      )}

      <RecordDetailSheet recordId={selected} open={!!selected} onClose={() => setSelected(null)} />
    </div>
  );
}
