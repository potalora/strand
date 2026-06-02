"use client";

import { useCallback, useEffect, useState } from "react";
import { Check } from "lucide-react";
import { api } from "@/lib/api";
import type { DedupCandidate } from "@/types/api";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { DedupCandidateCard } from "@/app/(dashboard)/admin/page";

export default function DedupPage() {
  const [candidates, setCandidates] = useState<DedupCandidate[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<string | null>(null);
  const [shown, setShown] = useState(false);
  const pageSize = 20;

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(id);
  }, []);

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
    <div className={`screen ${shown ? "on" : ""}`}>
      <div className="page-top">
        <div>
          <p className="kicker">Data &amp; settings</p>
          <h1 className="h1 display">Duplicates</h1>
        </div>
      </div>

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
