"use client";

import React, { useState } from "react";
import { ChevronDown, Sparkles, type LucideIcon } from "lucide-react";
import { RECORD_TYPE_ICONS, getObservationIcon } from "@/lib/record-icons";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR, RECORD_TYPE_LABELS } from "@/lib/constants";

// ---------------------------------------------------------------------------
// Safe accessors (extracted from original FhirResourceRenderer)
// ---------------------------------------------------------------------------

export function str(val: unknown): string {
  if (val === null || val === undefined) return "";
  if (typeof val === "string") return val;
  if (typeof val === "number" || typeof val === "boolean") return String(val);
  return "";
}

export function obj(val: unknown): Record<string, unknown> {
  if (val && typeof val === "object" && !Array.isArray(val))
    return val as Record<string, unknown>;
  return {};
}

export function arr(val: unknown): unknown[] {
  if (Array.isArray(val)) return val;
  return [];
}

export function nested(root: Record<string, unknown>, ...keys: string[]): unknown {
  let current: unknown = root;
  for (const key of keys) {
    if (current == null) return undefined;
    if (Array.isArray(current)) {
      // numeric key indexes into an array (e.g. nested(r, "coding", "0", "display"))
      const idx = Number(key);
      if (!Number.isInteger(idx)) return undefined;
      current = current[idx];
    } else if (typeof current === "object") {
      current = (current as Record<string, unknown>)[key];
    } else {
      return undefined;
    }
  }
  return current;
}

export function formatDate(val: unknown): string {
  const s = str(val);
  if (!s) return "";
  try {
    const d = new Date(s);
    if (isNaN(d.getTime())) return s;
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return s;
  }
}

export function formatDateTime(val: unknown): string {
  const s = str(val);
  if (!s) return "";
  try {
    const d = new Date(s);
    if (isNaN(d.getTime())) return s;
    return d.toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return s;
  }
}

// ---------------------------------------------------------------------------
// Shared UI primitives
// ---------------------------------------------------------------------------

export function DetailRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  if (!value) return null;
  return (
    <div className="flex flex-col gap-0.5 py-1">
      <span
        className="text-[11px] font-medium uppercase tracking-wide"
        style={{ color: "var(--theme-text-muted)" }}
      >
        {label}
      </span>
      <span
        className={`text-sm ${mono ? "font-mono" : ""}`}
        style={{ color: "var(--theme-text)" }}
      >
        {value}
      </span>
    </div>
  );
}

export function StatusBadge({
  label,
  color,
}: {
  label: string;
  color?: string;
}) {
  if (!label) return null;
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-md whitespace-nowrap"
      style={{
        backgroundColor: color ?? "var(--theme-bg-deep)",
        color: "var(--theme-text)",
      }}
    >
      {label}
    </span>
  );
}

export function SectionDivider() {
  return (
    <div
      className="border-t my-2"
      style={{ borderColor: "var(--theme-border)" }}
    />
  );
}

export function BarRow({ items }: { items: Array<{ label: string; value: string }> }) {
  const filtered = items.filter((i) => i.value);
  if (filtered.length === 0) return null;
  return (
    <div
      className="flex flex-wrap items-center gap-x-5 gap-y-1 px-3 py-2 rounded-md text-xs"
      style={{
        backgroundColor: "var(--theme-bg-deep)",
        borderColor: "var(--theme-border)",
      }}
    >
      {filtered.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5 min-w-0">
          <span className="min-w-0" style={{ color: "var(--theme-text-muted)" }}>
            {item.label}:
          </span>
          <span className="min-w-0" style={{ color: "var(--theme-text)" }}>{item.value}</span>
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// New shared components
// ---------------------------------------------------------------------------

export interface RecordHeaderProps {
  recordType: string;
  title: string;
  fhirResource?: Record<string, unknown>;
  status?: string;
  statusColor?: string;
}

export function RecordHeader({ recordType, title, fhirResource, status, statusColor }: RecordHeaderProps) {
  const type = recordType.toLowerCase();
  let IconComponent: LucideIcon | undefined = RECORD_TYPE_ICONS[type];
  if (type === "observation" && fhirResource) {
    IconComponent = getObservationIcon(fhirResource);
  }
  const colors = RECORD_TYPE_COLORS[type] ?? DEFAULT_RECORD_COLOR;
  const label = RECORD_TYPE_LABELS[type] ?? type;

  return (
    <div className="flex items-start gap-3 mb-3">
      {IconComponent && (
        <div
          className="flex items-center justify-center w-8 h-8 rounded-md shrink-0 mt-0.5"
          style={{ backgroundColor: colors.bg, color: colors.text }}
        >
          <IconComponent size={16} />
        </div>
      )}
      <div className="min-w-0 flex-1">
        <h3
          className="text-sm font-semibold leading-tight"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {title}
        </h3>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] font-medium uppercase tracking-wider" style={{ color: colors.text }}>
            {label}
          </span>
          {status && (
            <StatusBadge label={status} color={statusColor} />
          )}
        </div>
      </div>
    </div>
  );
}

export interface AIExtractionBadgeProps {
  aiExtracted: boolean;
  confidenceScore?: number | null;
}

export function AIExtractionBadge({ aiExtracted, confidenceScore }: AIExtractionBadgeProps) {
  if (!aiExtracted) return null;
  const pct = confidenceScore != null ? `${Math.round(confidenceScore * 100)}%` : null;

  return (
    <div
      className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium"
      style={{
        backgroundColor: "var(--theme-bg-deep)",
        color: "var(--theme-amber)",
      }}
    >
      <Sparkles size={12} />
      <span>AI Extracted</span>
      {pct && (
        <span
          className="ml-1 px-1.5 py-0 rounded text-[10px] font-mono"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          {pct}
        </span>
      )}
    </div>
  );
}

export interface AdvancedSectionProps {
  fhirResource: Record<string, unknown>;
}

function syntaxHighlight(json: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  // Simple regex-based JSON syntax highlighting
  const regex = /("(?:\\.|[^"\\])*")\s*:|("(?:\\.|[^"\\])*")|(\b(?:true|false)\b)|(null)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}[\],])/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = regex.exec(json)) !== null) {
    if (match.index > lastIndex) {
      parts.push(json.slice(lastIndex, match.index));
    }

    if (match[1]) {
      // Key
      parts.push(<span key={key++} className="json-key">{match[1]}</span>);
      parts.push(":");
    } else if (match[2]) {
      // String value
      parts.push(<span key={key++} className="json-string">{match[2]}</span>);
    } else if (match[3]) {
      // Boolean
      parts.push(<span key={key++} className="json-boolean">{match[3]}</span>);
    } else if (match[4]) {
      // Null
      parts.push(<span key={key++} className="json-null">{match[4]}</span>);
    } else if (match[5]) {
      // Number
      parts.push(<span key={key++} className="json-number">{match[5]}</span>);
    } else if (match[6]) {
      // Brace/bracket/comma
      parts.push(<span key={key++} className="json-brace">{match[6]}</span>);
    }

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < json.length) {
    parts.push(json.slice(lastIndex));
  }

  return parts;
}

export function AdvancedSection({ fhirResource }: AdvancedSectionProps) {
  const [open, setOpen] = useState(false);
  const jsonStr = JSON.stringify(fhirResource, null, 2);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs cursor-pointer transition-colors font-medium"
        style={{ color: "var(--theme-amber-dim)" }}
        onMouseEnter={(e) => (e.currentTarget.style.color = "var(--theme-amber)")}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--theme-amber-dim)")}
      >
        <ChevronDown
          size={14}
          className="advanced-chevron"
          data-open={open}
        />
        Advanced
      </button>
      {open && (
        <pre
          className="mt-2 p-3 text-xs overflow-auto max-h-80 json-syntax"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            color: "var(--theme-text-dim)",
            borderRadius: "var(--radius-lg)",
            border: "1px solid var(--theme-border)",
          }}
        >
          {syntaxHighlight(jsonStr)}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Observation category detection (shared across renderers)
// ---------------------------------------------------------------------------

export function getObservationCategory(r: Record<string, unknown>): string {
  const categories = arr(r.category);
  for (const cat of categories) {
    const codings = arr(obj(cat).coding);
    for (const coding of codings) {
      const code = str(obj(coding).code).toLowerCase();
      if (code === "vital-signs") return "vital-signs";
      if (code === "laboratory") return "laboratory";
      if (code === "social-history") return "social-history";
    }
    const text = str(obj(cat).text).toLowerCase();
    if (text.includes("vital")) return "vital-signs";
    if (text.includes("lab")) return "laboratory";
    if (text.includes("social")) return "social-history";
  }
  return "laboratory";
}
