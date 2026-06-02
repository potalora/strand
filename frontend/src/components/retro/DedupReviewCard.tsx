"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { RetroCard, RetroCardHeader, RetroCardContent } from "@/components/retro/RetroCard";
import { RetroButton } from "@/components/retro/RetroButton";

/* ==========================================
   TYPES
   ========================================== */

export interface ReviewRecord {
  id: string;
  display_text: string;
  record_type: string;
  fhir_resource: Record<string, unknown>;
}

export interface ReviewCandidate {
  candidate_id: string;
  primary: ReviewRecord;
  secondary: ReviewRecord;
  similarity_score: number;
  llm_classification: string;
  llm_confidence: number;
  llm_explanation: string;
  field_diff?: Record<string, { old: string; new: string }>;
}

interface DedupReviewCardProps {
  recordType: string;
  candidates: ReviewCandidate[];
  onResolve: (candidateId: string, action: "accept" | "decline", fieldOverrides?: Record<string, string>) => void;
  selected: Set<string>;
  onToggleSelect: (candidateId: string) => void;
}

/* ==========================================
   CONFIDENCE BADGE
   ========================================== */

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  let bg = "var(--theme-terracotta)";
  let label = "Low";

  if (confidence >= 0.8) {
    bg = "var(--primary)";
    label = "High";
  } else if (confidence >= 0.5) {
    bg = "var(--theme-ochre)";
    label = "Med";
  }

  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-md"
      style={{ backgroundColor: bg, color: "#ffffff", fontFamily: "var(--font-mono)" }}
    >
      {label} {pct}%
    </span>
  );
}

/* ==========================================
   CLASSIFICATION BADGE
   ========================================== */

function ClassificationBadge({ classification }: { classification: string }) {
  const colorMap: Record<string, string> = {
    duplicate: "var(--primary)",
    update: "var(--theme-ochre)",
    related: "var(--success)",
    distinct: "var(--theme-terracotta)",
  };
  const bg = colorMap[classification] ?? "var(--text-dim)";

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md capitalize"
      style={{ backgroundColor: bg, color: "#ffffff" }}
    >
      {classification}
    </span>
  );
}

/* ==========================================
   CANDIDATE ROW
   ========================================== */

function CandidateRow({
  candidate,
  isSelected,
  onToggleSelect,
  onResolve,
}: {
  candidate: ReviewCandidate;
  isSelected: boolean;
  onToggleSelect: () => void;
  onResolve: (action: "accept" | "decline") => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const isUpdate = candidate.llm_classification === "update";
  const summary =
    isUpdate && candidate.primary.display_text !== candidate.secondary.display_text
      ? `${candidate.primary.display_text} → ${candidate.secondary.display_text}`
      : candidate.primary.display_text;

  const hasFieldDiff =
    candidate.field_diff && Object.keys(candidate.field_diff).length > 0;

  return (
    <div
      className="border-b last:border-b-0 transition-colors duration-150"
      style={{ borderColor: "var(--theme-border)" }}
    >
      {/* Main row */}
      <div
        className="flex items-center gap-3 px-4 py-3"
        style={{ backgroundColor: isSelected ? "var(--theme-bg-card-hover)" : undefined }}
      >
        {/* Checkbox */}
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggleSelect}
          className="h-4 w-4 cursor-pointer rounded"
          style={{ accentColor: "var(--theme-amber)" }}
        />

        {/* Expand toggle */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex-shrink-0 text-xs transition-colors duration-150"
          style={{ color: "var(--theme-text-dim)" }}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>

        {/* Summary text */}
        <span
          className="flex-1 text-sm truncate"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
          title={summary}
        >
          {summary}
        </span>

        {/* Badges */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <ConfidenceBadge confidence={candidate.llm_confidence} />
          <ClassificationBadge classification={candidate.llm_classification} />
        </div>

        {/* Per-row actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <RetroButton
            variant="primary"
            className="text-xs px-3 py-1"
            onClick={() => onResolve("accept")}
          >
            Accept
          </RetroButton>
          <RetroButton
            variant="ghost"
            className="text-xs px-3 py-1"
            onClick={() => onResolve("decline")}
          >
            Decline
          </RetroButton>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div
          className="px-11 pb-4 space-y-3"
          style={{ backgroundColor: "var(--theme-bg-card-hover)" }}
        >
          {/* LLM Explanation */}
          {candidate.llm_explanation && (
            <div>
              <p
                className="text-xs font-medium mb-1"
                style={{ color: "var(--theme-text-dim)" }}
              >
                AI Explanation
              </p>
              <p
                className="text-xs"
                style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
              >
                {candidate.llm_explanation}
              </p>
            </div>
          )}

          {/* Field diff */}
          {hasFieldDiff && (
            <div>
              <p
                className="text-xs font-medium mb-1"
                style={{ color: "var(--theme-text-dim)" }}
              >
                Field Differences
              </p>
              <div className="space-y-1">
                {Object.entries(candidate.field_diff!).map(([field, diff]) => (
                  <div
                    key={field}
                    className="flex items-center gap-2 text-xs rounded px-2 py-1"
                    style={{
                      backgroundColor: "var(--theme-bg-card)",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    <span
                      className="font-medium w-32 flex-shrink-0"
                      style={{ color: "var(--theme-text-dim)" }}
                    >
                      {field}
                    </span>
                    <span style={{ color: "var(--theme-terracotta)" }}>{diff.old}</span>
                    <ArrowRight className="h-3 w-3 flex-shrink-0" style={{ color: "var(--theme-text-dim)" }} />
                    <span style={{ color: "var(--success)" }}>{diff.new}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Similarity score */}
          <p
            className="text-xs"
            style={{ color: "var(--theme-text-dim)", fontFamily: "var(--font-mono)" }}
          >
            similarity_score: {(candidate.similarity_score * 100).toFixed(0)}%
          </p>
        </div>
      )}
    </div>
  );
}

/* ==========================================
   MAIN COMPONENT
   ========================================== */

export function DedupReviewCard({
  recordType,
  candidates,
  onResolve,
  selected,
  onToggleSelect,
}: DedupReviewCardProps) {
  const label =
    recordType.charAt(0).toUpperCase() + recordType.slice(1).replace(/_/g, " ");

  const allSelected = candidates.every((c) => selected.has(c.candidate_id));

  const handleAcceptAll = () => {
    candidates.forEach((c) => onResolve(c.candidate_id, "accept"));
  };

  const handleDeclineAll = () => {
    candidates.forEach((c) => onResolve(c.candidate_id, "decline"));
  };

  return (
    <RetroCard accentTop>
      {/* Group header */}
      <RetroCardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={() => {
                if (allSelected) {
                  // Deselect all in this group
                  candidates.forEach((c) => {
                    if (selected.has(c.candidate_id)) onToggleSelect(c.candidate_id);
                  });
                } else {
                  // Select all in this group
                  candidates.forEach((c) => {
                    if (!selected.has(c.candidate_id)) onToggleSelect(c.candidate_id);
                  });
                }
              }}
              className="h-4 w-4 cursor-pointer rounded"
              style={{ accentColor: "var(--theme-amber)" }}
              aria-label={`Select all ${label}`}
            />
            <h3
              className="text-sm font-medium"
              style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
            >
              {label}
            </h3>
            <span
              className="inline-flex items-center px-2 py-0.5 text-xs rounded-full"
              style={{
                backgroundColor: "var(--theme-bg-card-hover)",
                color: "var(--theme-text-dim)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {candidates.length}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <RetroButton
              variant="primary"
              className="text-xs px-3 py-1"
              onClick={handleAcceptAll}
            >
              Accept All
            </RetroButton>
            <RetroButton
              variant="ghost"
              className="text-xs px-3 py-1"
              onClick={handleDeclineAll}
            >
              Decline All
            </RetroButton>
          </div>
        </div>
      </RetroCardHeader>

      {/* Candidate rows */}
      <div>
        {candidates.map((candidate) => (
          <CandidateRow
            key={candidate.candidate_id}
            candidate={candidate}
            isSelected={selected.has(candidate.candidate_id)}
            onToggleSelect={() => onToggleSelect(candidate.candidate_id)}
            onResolve={(action) => onResolve(candidate.candidate_id, action)}
          />
        ))}
      </div>
    </RetroCard>
  );
}
