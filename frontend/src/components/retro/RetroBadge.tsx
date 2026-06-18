"use client";

import { RECORD_TYPE_COLORS, RECORD_TYPE_SHORT, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { getObservationSubTypeLabel, getObservationSubTypeShort } from "@/lib/record-icons";
import { cn } from "@/lib/utils";

interface RetroBadgeProps {
  recordType: string;
  short?: boolean;
  className?: string;
  /** Observation category codes (the flat `category: string[]` on a record /
   *  timeline event). When present on an observation, the badge shows the
   *  sub-type — Lab / Vital / Social — instead of the generic "Observation". */
  category?: readonly string[] | null;
  /** Full FHIR resource — an alternative source for the observation sub-type
   *  when the flat category list isn't to hand (e.g. the detail sheet). */
  fhirResource?: Record<string, unknown> | null;
}

/** Restrained record badge: a neutral chip with a single type-colored dot
 *  (the design's deliberately un-rainbow treatment). For observations, the
 *  label resolves to the sub-type (Lab / Vital / Social) so it agrees with the
 *  sub-type icon shown elsewhere. */
export function RetroBadge({ recordType, short = false, className, category, fhirResource }: RetroBadgeProps) {
  const colors = RECORD_TYPE_COLORS[recordType] || DEFAULT_RECORD_COLOR;

  let label = short
    ? RECORD_TYPE_SHORT[recordType] || recordType.toUpperCase().slice(0, 4)
    : recordType.replace(/_/g, " ");

  // Observations carry a sub-type (Lab / Vital / Social) on their category.
  // Surface it when we have either the flat list or the FHIR resource.
  if (recordType.toLowerCase() === "observation" && (category != null || fhirResource != null)) {
    const source = category ?? fhirResource;
    label = short ? getObservationSubTypeShort(source) : getObservationSubTypeLabel(source);
  }

  return (
    <span className={cn("badge", className)}>
      <span className="bd" style={{ background: colors.dot }} />
      {label}
    </span>
  );
}
