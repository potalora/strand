"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ChevronDown, ChevronRight, ArrowLeft, CheckCircle, RotateCcw, Lock } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { DedupReviewCard, type ReviewCandidate } from "@/components/retro/DedupReviewCard";

/* ==========================================
   TYPES
   ========================================== */

interface DedupSummary {
  total_candidates: number;
  auto_merged: number;
  needs_review: number;
  dismissed: number;
  by_type: Record<string, number>;
}

interface UploadInfo {
  id: string;
  filename: string;
  uploaded_at: string;
  record_count: number;
  status: string;
  dedup_summary: DedupSummary;
}

interface AutoMergedEntry {
  candidate_id: string;
  primary: {
    id: string;
    display_text: string;
    record_type: string;
    fhir_resource: Record<string, unknown>;
  };
  secondary: {
    id: string;
    display_text: string;
    record_type: string;
    fhir_resource: Record<string, unknown>;
  };
  similarity_score: number;
  llm_classification: string;
  llm_confidence: number;
  llm_explanation: string;
  merged_at: string;
}

interface ReviewResponse {
  upload: UploadInfo;
  auto_merged: AutoMergedEntry[];
  needs_review: Record<string, ReviewCandidate[]>;
}

/* ==========================================
   HELPERS
   ========================================== */

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

function statusDotColor(status: string): string {
  switch (status) {
    case "complete":
      return "var(--success)";
    case "awaiting_review":
    case "processing":
      return "var(--primary)";
    case "failed":
      return "var(--danger)";
    default:
      return "var(--text-muted)";
  }
}

function StatusTag({ status }: { status: string }) {
  return (
    <span className="tag" style={{ textTransform: "capitalize" }}>
      <span className="tdot" style={{ background: statusDotColor(status) }} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

function SecureChip() {
  return (
    <span className="secure">
      <Lock size={13} strokeWidth={1.9} /> End-to-end encrypted
    </span>
  );
}

/* ==========================================
   AUTO-MERGED SECTION
   ========================================== */

function AutoMergedSection({
  entries,
  onUndo,
}: {
  entries: AutoMergedEntry[];
  onUndo: (candidateId: string) => void;
}) {
  const [open, setOpen] = useState(false);

  if (entries.length === 0) return null;

  return (
    <div className="card-surface">
      <button
        onClick={() => setOpen((v) => !v)}
        className="between"
        style={{
          width: "100%",
          padding: "16px 22px",
          cursor: "pointer",
          background: "transparent",
          border: 0,
          textAlign: "left",
        }}
      >
        <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
          {open ? (
            <ChevronDown size={16} style={{ color: "var(--text-muted)" }} />
          ) : (
            <ChevronRight size={16} style={{ color: "var(--text-muted)" }} />
          )}
          <span className="sec-title">Auto-merged</span>
          <span className="tag">
            <span className="tdot" style={{ background: "var(--success)" }} />
            {entries.length}
          </span>
        </span>
        <span className="muted mono" style={{ fontSize: 11 }}>
          {open ? "collapse" : "expand to undo"}
        </span>
      </button>

      {open && (
        <div style={{ padding: "0 18px 18px" }}>
          {entries.map((entry) => (
            <div key={entry.candidate_id} className="dedup-card">
              <div className="between">
                <span style={{ display: "inline-flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                  <span className="tag" style={{ textTransform: "capitalize" }}>
                    {entry.primary.record_type.replace(/_/g, " ")}
                  </span>
                  <span className="muted mono" style={{ fontSize: 11 }}>
                    merged {formatDate(entry.merged_at)}
                  </span>
                </span>
                <button
                  className="btn ghost sm"
                  onClick={() => onUndo(entry.candidate_id)}
                >
                  <RotateCcw size={13} />
                  Undo
                </button>
              </div>

              <div className="dedup-pair">
                <div className="dedup-rec">
                  <div className="field-l">Kept</div>
                  <div className="field-v" style={{ fontSize: 13.5 }}>
                    {entry.primary.display_text}
                  </div>
                </div>
                <div className="dedup-vs">+</div>
                <div className="dedup-rec">
                  <div className="field-l">Merged in</div>
                  <div className="field-v" style={{ fontSize: 13.5 }}>
                    {entry.secondary.display_text}
                  </div>
                </div>
              </div>

              <div className="reasons" style={{ marginTop: 12 }}>
                <span className="reason">similarity {Math.round(entry.similarity_score * 100)}%</span>
                {entry.llm_classification && (
                  <span className="reason">{entry.llm_classification}</span>
                )}
                {entry.llm_confidence != null && (
                  <span className="reason">confidence {Math.round(entry.llm_confidence * 100)}%</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ==========================================
   ALL RESOLVED STATE
   ========================================== */

function AllResolvedState({ onBack }: { onBack: () => void }) {
  return (
    <div className="screen on" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "80px 0", gap: 16 }}>
      <CheckCircle size={48} style={{ color: "var(--success)" }} />
      <h2 className="sec-title" style={{ fontSize: 22 }}>
        All candidates resolved
      </h2>
      <p className="dim" style={{ fontSize: 14 }}>
        No remaining deduplication decisions for this upload.
      </p>
      <button className="btn ghost" onClick={onBack}>
        <ArrowLeft size={15} />
        Back to Uploads
      </button>
    </div>
  );
}

/* ==========================================
   MAIN PAGE
   ========================================== */

export default function ReviewPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const uploadId = params.id;

  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ReviewResponse | null>(null);

  // Track resolved candidate IDs so we can hide them optimistically
  const [resolved, setResolved] = useState<Set<string>>(new Set());
  const [resolving, setResolving] = useState<Set<string>>(new Set());

  // Bulk selection
  const [selected, setSelected] = useState<Set<string>>(new Set());

  /* ---- Data fetching ---- */
  const fetchReview = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.get<ReviewResponse>(`/upload/${uploadId}/review`);
      setData(result);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Failed to load review data.");
      }
    } finally {
      setLoading(false);
    }
  }, [uploadId]);

  useEffect(() => {
    fetchReview();
  }, [fetchReview]);

  /* ---- Resolve handler ---- */
  const handleResolve = useCallback(
    async (candidateId: string, action: "accept" | "decline") => {
      if (resolving.has(candidateId) || resolved.has(candidateId)) return;

      setResolving((prev) => new Set(prev).add(candidateId));
      try {
        await api.post(`/upload/${uploadId}/review/resolve`, {
          candidate_id: candidateId,
          action,
        });
        setResolved((prev) => {
          const next = new Set(prev);
          next.add(candidateId);
          return next;
        });
        setSelected((prev) => {
          const next = new Set(prev);
          next.delete(candidateId);
          return next;
        });
      } catch (err) {
        console.error("Failed to resolve candidate:", err);
      } finally {
        setResolving((prev) => {
          const next = new Set(prev);
          next.delete(candidateId);
          return next;
        });
      }
    },
    [uploadId, resolved, resolving]
  );

  /* ---- Undo merge handler ---- */
  const handleUndoMerge = useCallback(
    async (candidateId: string) => {
      try {
        await api.post(`/upload/${uploadId}/review/undo-merge`, {
          candidate_id: candidateId,
        });
        // Refresh data after undo
        await fetchReview();
      } catch (err) {
        console.error("Failed to undo merge:", err);
      }
    },
    [uploadId, fetchReview]
  );

  /* ---- Toggle select ---- */
  const handleToggleSelect = useCallback((candidateId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(candidateId)) {
        next.delete(candidateId);
      } else {
        next.add(candidateId);
      }
      return next;
    });
  }, []);

  /* ---- Bulk resolve ---- */
  const handleBulkResolve = useCallback(
    async (action: "accept" | "decline") => {
      const ids = Array.from(selected).filter(
        (id) => !resolved.has(id) && !resolving.has(id)
      );
      for (const id of ids) {
        await handleResolve(id, action);
      }
    },
    [selected, resolved, resolving, handleResolve]
  );

  /* ---- Derived state ---- */
  const needsReviewByType = data?.needs_review ?? {};
  const totalNeedsReview = Object.values(needsReviewByType).reduce(
    (sum, arr) => sum + arr.filter((c) => !resolved.has(c.candidate_id)).length,
    0
  );
  const resolvedCount = resolved.size;
  const totalCandidates =
    (data?.upload.dedup_summary.needs_review ?? 0);
  const allResolved = totalNeedsReview === 0 && !loading && data !== null;

  /* ---- Render ---- */
  if (loading) {
    return (
      <div className="screen on">
        <RetroLoadingState text="Loading review data" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="screen on s24">
        <button className="btn ghost" onClick={() => router.push("/upload")}>
          <ArrowLeft size={15} />
          Back to Uploads
        </button>
        <div className="card-surface pad">
          <p style={{ fontSize: 13.5, color: "var(--danger)", margin: 0 }}>{error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  if (allResolved) {
    return <AllResolvedState onBack={() => router.push("/upload")} />;
  }

  const { upload, auto_merged } = data;

  const metaStats = [
    { label: "Uploaded", value: formatDate(upload.uploaded_at), color: "var(--text)" },
    { label: "Records", value: upload.record_count.toLocaleString(), color: "var(--text)" },
    { label: "Total candidates", value: String(upload.dedup_summary.total_candidates), color: "var(--text)" },
    { label: "Auto-merged", value: String(upload.dedup_summary.auto_merged), color: "var(--success)" },
    { label: "Needs review", value: String(upload.dedup_summary.needs_review), color: "var(--primary)" },
  ];

  return (
    <div className={`screen s24 ${shown ? "on" : ""}`} style={{ paddingBottom: 96 }}>
      {/* Back nav */}
      <button
        onClick={() => router.push("/upload")}
        className="btn ghost sm"
        style={{ alignSelf: "flex-start" }}
      >
        <ArrowLeft size={14} />
        Back to Uploads
      </button>

      {/* Header */}
      <div className="page-top" style={{ marginBottom: 0 }}>
        <div>
          <p className="kicker">Review duplicates</p>
          <h1 className="h1 display">Dedup review</h1>
          <p className="h-sub">{upload.filename}</p>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 12 }}>
          <SecureChip />
          <StatusTag status={upload.status} />
        </div>
      </div>

      {/* Upload metadata */}
      <div className="card-surface pad">
        <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
          {metaStats.map((s) => (
            <div key={s.label}>
              <div className="field-l">{s.label}</div>
              <div className="mono" style={{ fontSize: 16, fontWeight: 600, color: s.color }}>
                {s.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary bar */}
      <div
        className="between mono"
        style={{
          padding: "10px 16px",
          borderRadius: "var(--radius-sm)",
          background: "var(--card-2)",
          border: "1px solid var(--border)",
          fontSize: 12.5,
          color: "var(--text-dim)",
          justifyContent: "flex-start",
          gap: 16,
        }}
      >
        <span>
          <span style={{ color: "var(--success)" }}>{upload.dedup_summary.auto_merged}</span> auto-merged
        </span>
        <span style={{ color: "var(--border-strong)" }}>/</span>
        <span>
          <span style={{ color: "var(--primary)" }}>{totalNeedsReview}</span> remaining
        </span>
        <span style={{ color: "var(--border-strong)" }}>/</span>
        <span>
          <span style={{ color: "var(--text)" }}>{resolvedCount}</span> resolved this session
        </span>
      </div>

      {/* Auto-merged section */}
      <AutoMergedSection entries={auto_merged} onUndo={handleUndoMerge} />

      {/* Needs review section */}
      {Object.keys(needsReviewByType).length > 0 && (
        <div className="s16">
          <h2 className="sec-title">Needs review</h2>
          {Object.entries(needsReviewByType).map(([recordType, candidates]) => {
            const visible = candidates.filter((c) => !resolved.has(c.candidate_id));
            if (visible.length === 0) return null;
            return (
              <DedupReviewCard
                key={recordType}
                recordType={recordType}
                candidates={visible}
                onResolve={handleResolve}
                selected={selected}
                onToggleSelect={handleToggleSelect}
              />
            );
          })}
        </div>
      )}

      {/* Sticky bulk action bar */}
      {selected.size > 0 && (
        <div
          className="between"
          style={{
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            zIndex: 50,
            padding: "16px 24px",
            background: "var(--card)",
            borderTop: "1px solid var(--border-strong)",
            boxShadow: "var(--shadow-lg)",
          }}
        >
          <span className="dim" style={{ fontSize: 13.5 }}>
            <span className="mono" style={{ color: "var(--text)" }}>{selected.size}</span> selected ·{" "}
            <span style={{ color: "var(--text)" }}>
              {resolvedCount}/{totalCandidates}
            </span>{" "}
            resolved
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button className="btn ghost sm" onClick={() => setSelected(new Set())}>
              Clear
            </button>
            <button className="btn ghost sm" onClick={() => handleBulkResolve("decline")}>
              Keep both ({selected.size})
            </button>
            <button className="btn sm" onClick={() => handleBulkResolve("accept")}>
              Merge selected ({selected.size})
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
