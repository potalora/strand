"use client";

import { RECORD_TYPE_COLORS, RECORD_TYPE_SHORT, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface RetroBadgeProps {
  recordType: string;
  short?: boolean;
  className?: string;
}

/** Restrained record badge: a neutral chip with a single type-colored dot
 *  (the design's deliberately un-rainbow treatment). */
export function RetroBadge({ recordType, short = false, className }: RetroBadgeProps) {
  const colors = RECORD_TYPE_COLORS[recordType] || DEFAULT_RECORD_COLOR;
  const label = short
    ? RECORD_TYPE_SHORT[recordType] || recordType.toUpperCase().slice(0, 4)
    : recordType.replace(/_/g, " ");

  return (
    <span className={cn("badge", className)}>
      <span className="bd" style={{ background: colors.dot }} />
      {label}
    </span>
  );
}
