"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { ChevronRight, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/useAuthStore";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import type {
  RecordListResponse,
  HealthRecord,
  DedupCandidate,
  UserResponse,
  DashboardOverview,
} from "@/types/api";
import { RECORD_TYPE_COLORS, RECORD_TYPE_LABELS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { GlowText } from "@/components/retro/GlowText";
import { RetroTabs } from "@/components/retro/RetroTabs";
import { RetroCard, RetroCardHeader, RetroCardContent } from "@/components/retro/RetroCard";
import { RetroButton } from "@/components/retro/RetroButton";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RecordDetailSheet } from "@/components/retro/RecordDetailSheet";
import { ConfirmDialog } from "@/components/retro/ConfirmDialog";
import {
  RetroTable,
  RetroTableHeader,
  RetroTableHead,
  RetroTableBody,
  RetroTableRow,
  RetroTableCell,
} from "@/components/retro/RetroTable";

const TABS = [
  { key: "records", label: "Records" },
  { key: "extractions", label: "Extractions" },
  { key: "dedup", label: "Dedup" },
  { key: "sys", label: "System" },
];

export default function AdminPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialTab = searchParams.get("tab") || "records";
  const [activeTab, setActiveTab] = useState(initialTab);

  useEffect(() => {
    const tab = searchParams.get("tab") || "records";
    setActiveTab(tab);
  }, [searchParams]);

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    router.replace(`/admin?tab=${key}`, { scroll: false });
  };

  return (
    <div className="space-y-6">
      <GlowText as="h1">Admin Console</GlowText>
      <RetroTabs tabs={TABS} active={activeTab} onChange={handleTabChange} />
      <div className="mt-4">
        {activeTab === "records" && <RecordsTab />}
        {activeTab === "extractions" && <ExtractionsTab />}
        {activeTab === "dedup" && <DedupTab />}
        {activeTab === "sys" && <SystemTab />}
      </div>
    </div>
  );
}

/* ==========================================
   EXTRACTIONS TAB — Pending extraction management
   ========================================== */

function ExtractionsTab() {
  interface ExtractionFile {
    id: string;
    filename: string;
    mime_type: string;
    file_category: string;
    file_size_bytes: number | null;
    created_at: string | null;
    ingestion_status?: string;
  }

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
      <RetroCard>
        <RetroCardContent>
          <p
            style={{
              textAlign: "center",
              padding: "2rem 0",
              fontSize: "0.8rem",
              color: "var(--theme-text-muted)",
              fontFamily: "var(--font-body)",
            }}
          >
            No files pending extraction, processing, or failed
          </p>
        </RetroCardContent>
      </RetroCard>
    );
  }

  const pendingCount = files.filter((f) => f.ingestion_status === "pending_extraction" || !f.ingestion_status).length;
  const processingCount = files.filter((f) => f.ingestion_status === "processing").length;
  const failedCount = files.filter((f) => f.ingestion_status === "failed").length;

  return (
    <RetroCard>
      <RetroCardHeader>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <GlowText as="h3" className="text-sm">
              {files.length} File{files.length !== 1 ? "s" : ""}
            </GlowText>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
              {pendingCount > 0 && (
                <span
                  style={{
                    fontSize: "0.6rem",
                    fontWeight: 600,
                    padding: "0.1rem 0.4rem",
                    borderRadius: "3px",
                    backgroundColor: "var(--theme-ochre)",
                    color: "var(--theme-bg-deep)",
                    fontFamily: "var(--font-body)",
                  }}
                >
                  {pendingCount} pending
                </span>
              )}
              {processingCount > 0 && (
                <span
                  style={{
                    fontSize: "0.6rem",
                    fontWeight: 600,
                    padding: "0.1rem 0.4rem",
                    borderRadius: "3px",
                    backgroundColor: "var(--theme-amber)",
                    color: "var(--theme-bg-deep)",
                    fontFamily: "var(--font-body)",
                  }}
                >
                  {processingCount} processing
                </span>
              )}
              {failedCount > 0 && (
                <span
                  style={{
                    fontSize: "0.6rem",
                    fontWeight: 600,
                    padding: "0.1rem 0.4rem",
                    borderRadius: "3px",
                    backgroundColor: "var(--theme-terracotta)",
                    color: "var(--theme-bg-deep)",
                    fontFamily: "var(--font-body)",
                  }}
                >
                  {failedCount} failed
                </span>
              )}
            </div>
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <RetroButton
              onClick={handleTrigger}
              disabled={selected.size === 0 || triggering}
            >
              {triggering
                ? "Triggering..."
                : `Extract ${selected.size || "Selected"}`}
            </RetroButton>
          </div>
        </div>
      </RetroCardHeader>
      <RetroCardContent>
        <RetroTable>
          <RetroTableHeader>
            <RetroTableHead className="w-8">
              <input
                type="checkbox"
                checked={selected.size === files.length && files.length > 0}
                onChange={toggleAll}
                disabled={triggering}
                style={{ accentColor: "var(--theme-amber)" }}
              />
            </RetroTableHead>
            <RetroTableHead>File</RetroTableHead>
            <RetroTableHead>Type</RetroTableHead>
            <RetroTableHead>Status</RetroTableHead>
            <RetroTableHead>Size</RetroTableHead>
            <RetroTableHead>Uploaded</RetroTableHead>
          </RetroTableHeader>
          <RetroTableBody>
            {files.map((file) => {
              const ext = file.filename.split(".").pop()?.toLowerCase() || "";
              const badgeColor =
                ext === "pdf"
                  ? "var(--theme-terracotta)"
                  : ext === "rtf"
                    ? "var(--theme-sage)"
                    : "var(--theme-ochre)";
              const status = file.ingestion_status || "pending_extraction";
              return (
                <RetroTableRow key={file.id}>
                  <RetroTableCell>
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
                      style={{ accentColor: "var(--theme-amber)" }}
                    />
                  </RetroTableCell>
                  <RetroTableCell>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        fontFamily: "var(--font-body)",
                        color: "var(--theme-text)",
                      }}
                    >
                      {file.filename}
                    </span>
                  </RetroTableCell>
                  <RetroTableCell>
                    <span
                      style={{
                        fontSize: "0.6rem",
                        fontWeight: 700,
                        padding: "0.1rem 0.4rem",
                        borderRadius: "3px",
                        backgroundColor: badgeColor,
                        color: "var(--theme-bg-deep)",
                        fontFamily: "var(--font-body)",
                        textTransform: "uppercase",
                      }}
                    >
                      {ext}
                    </span>
                  </RetroTableCell>
                  <RetroTableCell>
                    <span
                      style={{
                        fontSize: "0.6rem",
                        fontWeight: 600,
                        padding: "0.1rem 0.4rem",
                        borderRadius: "3px",
                        fontFamily: "var(--font-body)",
                        backgroundColor:
                          status === "processing"
                            ? "var(--theme-amber)"
                            : status === "failed"
                              ? "var(--theme-terracotta)"
                              : status === "duplicate_file"
                                ? "var(--theme-ochre)"
                                : "var(--theme-ochre)",
                        color: "var(--theme-bg-deep)",
                      }}
                    >
                      {status === "duplicate_file"
                        ? "Duplicate"
                        : status === "pending_extraction"
                          ? "pending"
                          : status}
                    </span>
                  </RetroTableCell>
                  <RetroTableCell>
                    <span
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.9rem",
                        color: "var(--theme-text-dim)",
                      }}
                    >
                      {file.file_size_bytes
                        ? file.file_size_bytes < 1024
                          ? `${file.file_size_bytes} B`
                          : file.file_size_bytes < 1024 * 1024
                            ? `${(file.file_size_bytes / 1024).toFixed(1)} KB`
                            : `${(file.file_size_bytes / (1024 * 1024)).toFixed(1)} MB`
                        : "--"}
                    </span>
                  </RetroTableCell>
                  <RetroTableCell>
                    <span
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.9rem",
                        color: "var(--theme-text-dim)",
                      }}
                    >
                      {file.created_at
                        ? new Date(file.created_at).toLocaleDateString()
                        : "--"}
                    </span>
                  </RetroTableCell>
                </RetroTableRow>
              );
            })}
          </RetroTableBody>
        </RetroTable>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "0.5rem 0 0",
            fontSize: "0.7rem",
            color: "var(--theme-text-muted)",
            fontFamily: "var(--font-body)",
          }}
        >
          <span>
            {selected.size} of {files.length} selected
          </span>
          <RetroButton variant="ghost" onClick={fetchFiles}>
            Refresh
          </RetroButton>
        </div>
      </RetroCardContent>
    </RetroCard>
  );
}

/* ==========================================
   RECORDS TAB — Tree-based records view
   ========================================== */

type ViewMode = "byType" | "byUpload";

interface TreeNodeData {
  key: string;
  label: string;
  count: number;
  recordType?: string;
  uploadId?: string;
  uploadDate?: string;
}

function RecordsTab() {
  const [viewMode, setViewMode] = useState<ViewMode>("byType");
  const [selectedRecord, setSelectedRecord] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [bulkConfirmOpen, setBulkConfirmOpen] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [dontAskChecked, setDontAskChecked] = useState(false);

  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();

  // Single delete state
  const [singleDeleteId, setSingleDeleteId] = useState<string | null>(null);
  const [singleConfirmOpen, setSingleConfirmOpen] = useState(false);
  const [singleDontAskChecked, setSingleDontAskChecked] = useState(false);

  // Refresh counter to trigger data re-fetches after deletes
  const [refreshKey, setRefreshKey] = useState(0);

  const triggerRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
    setSelectedIds(new Set());
  }, []);

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

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
      if (singleDontAskChecked) {
        setSkipDeleteConfirm(true);
      }
      setSingleConfirmOpen(false);
      setSingleDeleteId(null);
      triggerRefresh();
    } catch {
      // silently fail
    }
  };

  const handleBulkDelete = () => {
    if (selectedIds.size === 0) return;
    if (skipDeleteConfirm) {
      performBulkDelete();
    } else {
      setDontAskChecked(false);
      setBulkConfirmOpen(true);
    }
  };

  const performBulkDelete = async () => {
    setBulkDeleting(true);
    try {
      const promises = Array.from(selectedIds).map((id) =>
        api.delete(`/records/${id}`)
      );
      await Promise.all(promises);
      if (dontAskChecked) {
        setSkipDeleteConfirm(true);
      }
      setBulkConfirmOpen(false);
      triggerRefresh();
    } catch {
      // silently fail
    } finally {
      setBulkDeleting(false);
    }
  };

  return (
    <div style={{ position: "relative", minHeight: "200px" }}>
      {/* View toggle */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          marginBottom: "16px",
          gap: "0px",
        }}
      >
        <RetroButton
          variant="ghost"
          onClick={() => setViewMode("byType")}
          style={{
            color: viewMode === "byType" ? "var(--theme-amber)" : "var(--theme-text-dim)",
            borderBottom: viewMode === "byType" ? "2px solid var(--theme-amber)" : "2px solid transparent",
            borderRadius: 0,
            fontFamily: "var(--font-body)",
            fontSize: "0.75rem",
          }}
        >
          By Type
        </RetroButton>
        <RetroButton
          variant="ghost"
          onClick={() => setViewMode("byUpload")}
          style={{
            color: viewMode === "byUpload" ? "var(--theme-amber)" : "var(--theme-text-dim)",
            borderBottom: viewMode === "byUpload" ? "2px solid var(--theme-amber)" : "2px solid transparent",
            borderRadius: 0,
            fontFamily: "var(--font-body)",
            fontSize: "0.75rem",
          }}
        >
          By Upload
        </RetroButton>
      </div>

      {/* Tree content */}
      {viewMode === "byType" ? (
        <ByTypeTree
          refreshKey={refreshKey}
          selectedIds={selectedIds}
          onToggleSelection={toggleSelection}
          onDeleteRecord={handleSingleDelete}
          onSelectRecord={setSelectedRecord}
        />
      ) : (
        <ByUploadTree
          refreshKey={refreshKey}
          selectedIds={selectedIds}
          onToggleSelection={toggleSelection}
          onDeleteRecord={handleSingleDelete}
          onSelectRecord={setSelectedRecord}
        />
      )}

      {/* Floating bulk action bar */}
      {selectedIds.size > 0 && (
        <div
          style={{
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            height: "48px",
            backgroundColor: "var(--theme-bg-surface)",
            borderTop: "1px solid var(--theme-border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
            zIndex: 40,
            padding: "0 24px",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "1rem",
              color: "var(--theme-text)",
            }}
          >
            {selectedIds.size} selected
          </span>
          <RetroButton variant="destructive" onClick={handleBulkDelete}>
            Delete selected
          </RetroButton>
        </div>
      )}

      {/* Record detail sheet */}
      <RecordDetailSheet
        recordId={selectedRecord}
        open={!!selectedRecord}
        onClose={() => setSelectedRecord(null)}
        onDelete={triggerRefresh}
      />

      {/* Single delete confirm dialog */}
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

      {/* Bulk delete confirm dialog */}
      <ConfirmDialog
        open={bulkConfirmOpen}
        title={`Delete ${selectedIds.size} records?`}
        description={`This will soft-delete ${selectedIds.size} selected records. This action cannot be easily undone.`}
        confirmLabel="Delete all"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={performBulkDelete}
        onCancel={() => setBulkConfirmOpen(false)}
        showDontAskAgain
        dontAskAgainChecked={dontAskChecked}
        onDontAskAgainChange={setDontAskChecked}
      />
    </div>
  );
}

/* ------------------------------------------
   BY TYPE TREE
   ------------------------------------------ */

function ByTypeTree({
  refreshKey,
  selectedIds,
  onToggleSelection,
  onDeleteRecord,
  onSelectRecord,
}: {
  refreshKey: number;
  selectedIds: Set<string>;
  onToggleSelection: (id: string) => void;
  onDeleteRecord: (id: string) => void;
  onSelectRecord: (id: string) => void;
}) {
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .get<DashboardOverview>("/dashboard/overview")
      .then(setOverview)
      .catch(() => setOverview(null))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  if (loading) return <RetroLoadingState text="Loading records" />;

  if (!overview || Object.keys(overview.records_by_type).length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "48px 0" }}>
        <p
          style={{
            fontSize: "0.875rem",
            color: "var(--theme-text-muted)",
          }}
        >
          No records found
        </p>
      </div>
    );
  }

  const sortedTypes = Object.entries(overview.records_by_type)
    .filter(([, count]) => count > 0)
    .sort(([a], [b]) => a.localeCompare(b));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      {sortedTypes.map(([recordType, count]) => (
        <TypeTreeNode
          key={recordType}
          recordType={recordType}
          count={count}
          refreshKey={refreshKey}
          selectedIds={selectedIds}
          onToggleSelection={onToggleSelection}
          onDeleteRecord={onDeleteRecord}
          onSelectRecord={onSelectRecord}
        />
      ))}
    </div>
  );
}

function TypeTreeNode({
  recordType,
  count,
  refreshKey,
  selectedIds,
  onToggleSelection,
  onDeleteRecord,
  onSelectRecord,
}: {
  recordType: string;
  count: number;
  refreshKey: number;
  selectedIds: Set<string>;
  onToggleSelection: (id: string) => void;
  onDeleteRecord: (id: string) => void;
  onSelectRecord: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 25;

  const colors = RECORD_TYPE_COLORS[recordType] || DEFAULT_RECORD_COLOR;
  const label = RECORD_TYPE_LABELS[recordType] || recordType;

  useEffect(() => {
    if (!expanded) return;
    setLoadingRecords(true);
    api
      .get<RecordListResponse>(
        `/records?record_type=${recordType}&page=${page}&page_size=${pageSize}`
      )
      .then((data) => {
        setRecords(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => {
        setRecords([]);
        setTotal(0);
      })
      .finally(() => setLoadingRecords(false));
  }, [expanded, page, recordType, refreshKey]);

  const handleToggle = () => {
    if (!expanded) {
      setPage(1);
    }
    setExpanded(!expanded);
  };

  const hasMore = page * pageSize < total;

  return (
    <div>
      {/* Group header row */}
      <div
        onClick={handleToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "8px 12px",
          cursor: "pointer",
          transition: "background-color 150ms",
          borderRadius: "4px",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "var(--theme-bg-card-hover)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
        }}
      >
        <ChevronRight
          size={16}
          style={{
            color: "var(--theme-amber)",
            transition: "transform 200ms",
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            flexShrink: 0,
          }}
        />
        {/* Color dot */}
        <span
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            backgroundColor: colors.dot,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontFamily: "var(--font-body)",
            fontSize: "0.8125rem",
            color: "var(--theme-text)",
            flex: 1,
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.875rem",
            color: "var(--theme-text-muted)",
          }}
        >
          {count}
        </span>
      </div>

      {/* Expanded children */}
      {expanded && (
        <div
          style={{
            overflow: "hidden",
            transition: "max-height 300ms ease-in-out",
            paddingLeft: "36px",
          }}
        >
          {loadingRecords ? (
            <div style={{ padding: "12px 0" }}>
              <RetroLoadingState text="Loading" />
            </div>
          ) : records.length === 0 ? (
            <div style={{ padding: "12px 0" }}>
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--theme-text-muted)",
                }}
              >
                No records
              </span>
            </div>
          ) : (
            <>
              {records.map((record) => (
                <RecordLeafNode
                  key={record.id}
                  record={record}
                  isSelected={selectedIds.has(record.id)}
                  onToggleSelection={() => onToggleSelection(record.id)}
                  onDelete={() => onDeleteRecord(record.id)}
                  onSelect={() => onSelectRecord(record.id)}
                />
              ))}
              {hasMore && (
                <div style={{ padding: "8px 12px" }}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setPage((p) => p + 1);
                    }}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      fontSize: "0.75rem",
                      fontFamily: "var(--font-body)",
                      color: "var(--theme-amber)",
                      padding: 0,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.textDecoration = "underline";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.textDecoration = "none";
                    }}
                  >
                    Load more ({total - page * pageSize} remaining)
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------
   BY UPLOAD TREE
   ------------------------------------------ */

interface UploadHistoryItem {
  id: string;
  original_filename: string;
  ingestion_status: string;
  records_inserted?: number;
  created_at: string;
  file_category?: string;
}

function ByUploadTree({
  refreshKey,
  selectedIds,
  onToggleSelection,
  onDeleteRecord,
  onSelectRecord,
}: {
  refreshKey: number;
  selectedIds: Set<string>;
  onToggleSelection: (id: string) => void;
  onDeleteRecord: (id: string) => void;
  onSelectRecord: (id: string) => void;
}) {
  const [uploads, setUploads] = useState<UploadHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .get<{ items: UploadHistoryItem[]; total: number }>("/upload/history")
      .then((data) => setUploads(data.items || []))
      .catch(() => setUploads([]))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  if (loading) return <RetroLoadingState text="Loading uploads" />;

  if (uploads.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "48px 0" }}>
        <p
          style={{
            fontSize: "0.875rem",
            color: "var(--theme-text-muted)",
          }}
        >
          No uploads found
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      {uploads.map((upload) => (
        <UploadTreeNode
          key={upload.id}
          upload={upload}
          refreshKey={refreshKey}
          selectedIds={selectedIds}
          onToggleSelection={onToggleSelection}
          onDeleteRecord={onDeleteRecord}
          onSelectRecord={onSelectRecord}
        />
      ))}
    </div>
  );
}

function UploadTreeNode({
  upload,
  refreshKey,
  selectedIds,
  onToggleSelection,
  onDeleteRecord,
  onSelectRecord,
}: {
  upload: UploadHistoryItem;
  refreshKey: number;
  selectedIds: Set<string>;
  onToggleSelection: (id: string) => void;
  onDeleteRecord: (id: string) => void;
  onSelectRecord: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [records, setRecords] = useState<HealthRecord[]>([]);
  const [loadingRecords, setLoadingRecords] = useState(false);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const pageSize = 25;

  useEffect(() => {
    if (!expanded) return;
    setLoadingRecords(true);
    api
      .get<RecordListResponse>(
        `/records?source_upload_id=${upload.id}&page=${page}&page_size=${pageSize}`
      )
      .then((data) => {
        setRecords(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => {
        setRecords([]);
        setTotal(0);
      })
      .finally(() => setLoadingRecords(false));
  }, [expanded, page, upload.id, refreshKey]);

  const handleToggle = () => {
    if (!expanded) {
      setPage(1);
    }
    setExpanded(!expanded);
  };

  const hasMore = page * pageSize < total;
  const uploadDate = upload.created_at
    ? new Date(upload.created_at).toLocaleDateString()
    : "--";
  const recordCount = upload.records_inserted ?? 0;

  return (
    <div>
      {/* Upload header row */}
      <div
        onClick={handleToggle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "8px 12px",
          cursor: "pointer",
          transition: "background-color 150ms",
          borderRadius: "4px",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "var(--theme-bg-card-hover)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
        }}
      >
        <ChevronRight
          size={16}
          style={{
            color: "var(--theme-amber)",
            transition: "transform 200ms",
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            flexShrink: 0,
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <span
            style={{
              fontFamily: "var(--font-body)",
              fontSize: "0.8125rem",
              color: "var(--theme-text)",
              display: "block",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {upload.original_filename}
          </span>
          <span
            style={{
              fontFamily: "var(--font-body)",
              fontSize: "0.6875rem",
              color: "var(--theme-text-muted)",
            }}
          >
            {uploadDate}
          </span>
        </div>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.875rem",
            color: "var(--theme-text-muted)",
            flexShrink: 0,
          }}
        >
          {recordCount}
        </span>
      </div>

      {/* Expanded children */}
      {expanded && (
        <div
          style={{
            overflow: "hidden",
            transition: "max-height 300ms ease-in-out",
            paddingLeft: "36px",
          }}
        >
          {loadingRecords ? (
            <div style={{ padding: "12px 0" }}>
              <RetroLoadingState text="Loading" />
            </div>
          ) : records.length === 0 ? (
            <div style={{ padding: "12px 0" }}>
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--theme-text-muted)",
                }}
              >
                No records from this upload
              </span>
            </div>
          ) : (
            <>
              {records.map((record) => (
                <RecordLeafNode
                  key={record.id}
                  record={record}
                  isSelected={selectedIds.has(record.id)}
                  onToggleSelection={() => onToggleSelection(record.id)}
                  onDelete={() => onDeleteRecord(record.id)}
                  onSelect={() => onSelectRecord(record.id)}
                />
              ))}
              {hasMore && (
                <div style={{ padding: "8px 12px" }}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setPage((p) => p + 1);
                    }}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      fontSize: "0.75rem",
                      fontFamily: "var(--font-body)",
                      color: "var(--theme-amber)",
                      padding: 0,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.textDecoration = "underline";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.textDecoration = "none";
                    }}
                  >
                    Load more ({total - page * pageSize} remaining)
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------
   RECORD LEAF NODE (shared by both tree views)
   ------------------------------------------ */

function RecordLeafNode({
  record,
  isSelected,
  onToggleSelection,
  onDelete,
  onSelect,
}: {
  record: HealthRecord;
  isSelected: boolean;
  onToggleSelection: () => void;
  onDelete: () => void;
  onSelect: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "6px 12px",
        transition: "background-color 150ms",
        borderRadius: "4px",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = "var(--theme-bg-card-hover)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = "transparent";
      }}
    >
      {/* Checkbox */}
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => {
          e.stopPropagation();
          onToggleSelection();
        }}
        style={{
          accentColor: "var(--theme-amber)",
          cursor: "pointer",
          flexShrink: 0,
        }}
      />

      {/* Record text — clickable to open detail sheet */}
      <span
        onClick={onSelect}
        style={{
          flex: 1,
          fontFamily: "var(--font-body)",
          fontSize: "0.75rem",
          color: "var(--theme-text)",
          cursor: "pointer",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "var(--theme-amber)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "var(--theme-text)";
        }}
      >
        {record.display_text}
      </span>

      {/* Date */}
      <span
        style={{
          fontFamily: "var(--font-body)",
          fontSize: "0.6875rem",
          color: "var(--theme-text-muted)",
          flexShrink: 0,
          whiteSpace: "nowrap",
        }}
      >
        {record.effective_date
          ? new Date(record.effective_date).toLocaleDateString()
          : "--"}
      </span>

      {/* Trash icon */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "2px",
          display: "flex",
          alignItems: "center",
          color: "var(--theme-text-muted)",
          flexShrink: 0,
          transition: "color 150ms",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "var(--theme-terracotta)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "var(--theme-text-muted)";
        }}
      >
        <Trash2 size={14} />
      </button>
    </div>
  );
}

/* ==========================================
   DEDUP TAB
   ========================================== */

// TODO: Improve dedup UX — current detection works but resolution UI needs redesign

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

  const fetchCandidates = (p = page) => {
    setLoading(true);
    api
      .get<{ items: DedupCandidate[]; total: number }>(`/dedup/candidates?page=${p}&limit=${pageSize}`)
      .then((data) => {
        setCandidates(data.items || []);
        setTotal(data.total || 0);
      })
      .catch(() => { setCandidates([]); setTotal(0); })
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchCandidates(page); }, [page]);

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
    } catch (err) {
      setError(err instanceof Error ? err.message : "Dismiss failed");
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <RetroButton onClick={handleScan} disabled={scanning}>
          {scanning ? "Scanning..." : "Scan for duplicates"}
        </RetroButton>
        {scanResult && (
          <span className="text-xs" style={{ color: "var(--theme-text-dim)" }}>
            {scanResult}
          </span>
        )}
      </div>

      {error && (
        <RetroCard>
          <RetroCardContent>
            <div className="flex items-start gap-3">
              <span
                className="text-xs font-bold shrink-0 px-2 py-0.5"
                style={{
                  backgroundColor: "var(--theme-terracotta)",
                  color: "var(--theme-text)",
                  borderRadius: "4px",
                }}
              >
                ERROR
              </span>
              <p className="text-xs" style={{ color: "var(--theme-text-dim)" }}>{error}</p>
            </div>
          </RetroCardContent>
        </RetroCard>
      )}

      {loading ? (
        <RetroLoadingState text="Loading candidates" />
      ) : candidates.length === 0 && total === 0 ? (
        <div className="py-12 text-center">
          <p
            className="text-sm"
            style={{ color: "var(--theme-text-muted)" }}
          >
            No duplicate candidates
          </p>
          <p
            className="text-xs mt-1"
            style={{ color: "var(--theme-text-muted)" }}
          >
            Run the scanner to check for potential duplicates.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs" style={{ color: "var(--theme-text-dim)" }}>
              {total.toLocaleString()} pending candidates — showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)}
            </span>
            <div className="flex gap-2">
              <RetroButton variant="ghost" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Prev</RetroButton>
              <RetroButton variant="ghost" disabled={page * pageSize >= total} onClick={() => setPage(p => p + 1)}>Next</RetroButton>
            </div>
          </div>
          {candidates.map((candidate) => (
            <RetroCard key={candidate.id}>
              <RetroCardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span
                        className="text-sm font-medium"
                        style={{ color: "var(--theme-amber)" }}
                      >
                        {Math.round(candidate.similarity_score * 100)}% match
                      </span>
                      <div className="flex gap-1">
                        {Object.entries(candidate.match_reasons)
                          .filter(([, matched]) => matched)
                          .map(([reason]) => (
                            <span
                              key={reason}
                              className="px-2 py-0.5 text-xs rounded"
                              style={{
                                backgroundColor: "var(--theme-bg-surface)",
                                color: "var(--theme-text-dim)",
                                borderRadius: "4px",
                                border: "1px solid var(--theme-border)",
                              }}
                            >
                              {reason}
                            </span>
                          ))}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <RetroButton
                        onClick={() => handleMerge(candidate.id)}
                        disabled={actionLoading === candidate.id}
                      >
                        {actionLoading === candidate.id ? "..." : "Merge"}
                      </RetroButton>
                      <RetroButton
                        variant="ghost"
                        onClick={() => handleDismiss(candidate.id)}
                        disabled={actionLoading === candidate.id}
                      >
                        Dismiss
                      </RetroButton>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {candidate.record_a && (
                      <DedupRecordCard label="RECORD A" record={candidate.record_a} />
                    )}
                    {candidate.record_b && (
                      <DedupRecordCard label="RECORD B" record={candidate.record_b} />
                    )}
                  </div>
                </div>
              </RetroCardContent>
            </RetroCard>
          ))}
        </div>
      )}
    </div>
  );
}

function DedupRecordCard({
  label,
  record,
}: {
  label: string;
  record: NonNullable<DedupCandidate["record_a"]>;
}) {
  return (
    <div
      className="border p-3 space-y-2"
      style={{
        backgroundColor: "var(--theme-bg-surface)",
        borderColor: "var(--theme-border)",
        borderRadius: "4px",
      }}
    >
      <div className="flex items-center gap-2">
        <span className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
          {label}
        </span>
        <RetroBadge recordType={record.record_type} short />
      </div>
      <p className="text-sm" style={{ color: "var(--theme-text)" }}>
        {record.display_text}
      </p>
      <div className="flex items-center gap-3 text-xs" style={{ color: "var(--theme-text-muted)" }}>
        <span>{record.source_format}</span>
        <span>
          {record.effective_date
            ? new Date(record.effective_date).toLocaleDateString()
            : "--"}
        </span>
      </div>
    </div>
  );
}

/* ==========================================
   SYSTEM TAB
   ========================================== */

function interpretationStyle(interpretation: string): { color: string } {
  const code = interpretation?.toUpperCase();
  if (code === "H" || code === "HH") return { color: "var(--theme-terracotta)" };
  if (code === "L" || code === "LL") return { color: "var(--record-procedure-text)" };
  if (code === "A" || code === "AA") return { color: "var(--theme-ochre)" };
  return { color: "var(--theme-text-dim)" };
}

function interpretationLabel(interpretation: string): string {
  const code = interpretation?.toUpperCase();
  if (code === "H") return "HIGH";
  if (code === "HH") return "CRIT HIGH";
  if (code === "L") return "LOW";
  if (code === "LL") return "CRIT LOW";
  if (code === "A") return "ABNORMAL";
  if (code === "AA") return "CRIT ABNORM";
  if (code === "N") return "NORMAL";
  return interpretation || "--";
}

function statusColor(status: string | null): string {
  if (!status) return "var(--theme-text-dim)";
  const s = status.toLowerCase();
  if (s === "active" || s === "in-progress") return "var(--theme-ochre)";
  if (s === "completed" || s === "resolved" || s === "finished") return "var(--theme-sage)";
  if (s === "stopped" || s === "cancelled" || s === "not-done") return "var(--theme-terracotta)";
  return "var(--theme-text-dim)";
}

function SystemTab() {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState(true);
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

  if (loading) return <RetroLoadingState text="Loading system info" />;

  const dateRange =
    overview?.date_range_start && overview?.date_range_end
      ? `${new Date(overview.date_range_start).toLocaleDateString()} – ${new Date(overview.date_range_end).toLocaleDateString()}`
      : "N/A";

  return (
    <div className="space-y-4">
      {/* Account Info */}
      <RetroCard accentTop>
        <RetroCardHeader>
          <GlowText as="h4" glow={false}>Account information</GlowText>
        </RetroCardHeader>
        <RetroCardContent>
          {user ? (
            <div className="space-y-2">
              <SysRow label="Email" value={user.email} />
              <SysRow label="Display name" value={user.display_name || "Not set"} />
              <SysRow
                label="Status"
                value={user.is_active ? "Active" : "Inactive"}
                valueColor={user.is_active ? "var(--theme-sage)" : "var(--theme-terracotta)"}
              />
              <SysRow label="User ID" value={user.id} mono />
            </div>
          ) : (
            <p className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
              Unable to load account information.
            </p>
          )}
        </RetroCardContent>
      </RetroCard>

      {/* Data Stats */}
      <RetroCard>
        <RetroCardHeader>
          <GlowText as="h4" glow={false}>Data statistics</GlowText>
        </RetroCardHeader>
        <RetroCardContent>
          {overview ? (
            <div className="space-y-2">
              <SysRow label="Total records" value={String(overview.total_records)} />
              <SysRow label="Total patients" value={String(overview.total_patients)} />
              <SysRow label="Total uploads" value={String(overview.total_uploads)} />
              <SysRow label="Date range" value={dateRange} />
            </div>
          ) : (
            <p className="text-xs" style={{ color: "var(--theme-text-muted)" }}>
              No data available.
            </p>
          )}
        </RetroCardContent>
      </RetroCard>

      {/* Sign Out */}
      <RetroButton
        variant="destructive"
        onClick={() => {
          clearTokens();
          window.location.href = "/login";
        }}
      >
        Sign out
      </RetroButton>

      {/* Privacy Notice */}
      <RetroCard>
        <RetroCardContent>
          <div className="flex items-start gap-3">
            <span
              className="text-xs font-bold shrink-0 px-2 py-0.5"
              style={{
                backgroundColor: "var(--theme-sienna)",
                color: "var(--theme-text)",
                borderRadius: "4px",
              }}
            >
              NOTICE
            </span>
            <p
              className="text-xs leading-relaxed"
              style={{ color: "var(--theme-text-dim)" }}
            >
              All health data is stored locally and encrypted at rest. No data is
              transmitted to external services. AI summary prompts are constructed
              locally with de-identified data and are never sent automatically.
            </p>
          </div>
        </RetroCardContent>
      </RetroCard>
    </div>
  );
}

function SysRow({
  label,
  value,
  mono,
  valueColor,
}: {
  label: string;
  value: string;
  mono?: boolean;
  valueColor?: string;
}) {
  return (
    <div
      className="flex items-baseline justify-between py-1.5 border-b"
      style={{ borderColor: "var(--theme-border)" }}
    >
      <span
        className="text-xs font-medium"
        style={{ color: "var(--theme-text-muted)" }}
      >
        {label}
      </span>
      <span
        className={`text-xs ${mono ? "font-mono" : "font-medium"}`}
        style={{ color: valueColor || "var(--theme-text)" }}
      >
        {value}
      </span>
    </div>
  );
}
