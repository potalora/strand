import {
  Stethoscope,
  TestTube,
  HeartPulse,
  Pill,
  Building2,
  Shield,
  AlertTriangle,
  Scissors,
  ArrowRightLeft,
  FileText,
  ClipboardList,
  ScanLine,
  ListChecks,
  MessageSquare,
  CalendarClock,
  Users,
  UserRound,
  FileQuestion,
  type LucideIcon,
} from "lucide-react";

export const RECORD_TYPE_ICONS: Record<string, LucideIcon> = {
  condition: Stethoscope,
  observation: TestTube,
  medication: Pill,
  encounter: Building2,
  immunization: Shield,
  allergy: AlertTriangle,
  procedure: Scissors,
  service_request: ArrowRightLeft,
  document: FileText,
  diagnostic_report: ClipboardList,
  imaging: ScanLine,
  care_plan: ListChecks,
  communication: MessageSquare,
  appointment: CalendarClock,
  care_team: Users,
  questionnaire_response: FileQuestion,
  immunization_recommendation: Shield,
};

// ---------------------------------------------------------------------------
// Observation sub-type: Lab / Vital / Social
// ---------------------------------------------------------------------------
// Labs ARE FHIR Observation/laboratory, so a generic "Observation" badge hides
// useful information. We derive the sub-type from the category the record
// already carries and use it for BOTH the icon and the badge label, so the two
// can never disagree. Mirrors the renderer routing in renderers/index.tsx
// (getObservationCategory) and defaults to "lab" (matching the lab renderer /
// TestTube icon) when no category is present.

export type ObservationSubType = "lab" | "vital" | "social";

const OBS_SUBTYPE_ICONS: Record<ObservationSubType, LucideIcon> = {
  lab: TestTube,
  vital: HeartPulse,
  social: UserRound,
};

const OBS_SUBTYPE_LABELS: Record<ObservationSubType, string> = {
  lab: "Lab",
  vital: "Vital",
  social: "Social",
};

const OBS_SUBTYPE_SHORT: Record<ObservationSubType, string> = {
  lab: "LAB",
  vital: "VITAL",
  social: "SOCIAL",
};

/** Derive the observation sub-type from a flat list of category codes/labels
 *  (e.g. the `category: string[]` on a record or timeline event). Checked
 *  vital → social → lab to match the renderer's precedence; defaults to "lab". */
export function observationSubTypeFromCategories(
  categories: readonly string[] | null | undefined,
): ObservationSubType {
  for (const raw of categories ?? []) {
    const c = String(raw).toLowerCase();
    if (c === "vital-signs" || c.includes("vital")) return "vital";
    if (c === "social-history" || c.includes("social")) return "social";
    if (c === "laboratory" || c.includes("lab")) return "lab";
  }
  return "lab";
}

/** Pull category code/display/text strings out of a FHIR resource's category[]. */
function categoryStringsFromResource(fhirResource: Record<string, unknown>): string[] {
  const categories = Array.isArray(fhirResource.category) ? fhirResource.category : [];
  const out: string[] = [];
  for (const cat of categories) {
    if (typeof cat === "string") {
      out.push(cat);
      continue;
    }
    const c = (cat ?? {}) as Record<string, unknown>;
    const codings = Array.isArray(c.coding) ? (c.coding as Record<string, unknown>[]) : [];
    for (const coding of codings) {
      if (coding?.code != null) out.push(String(coding.code));
      if (coding?.display != null) out.push(String(coding.display));
    }
    if (c.text != null) out.push(String(c.text));
  }
  return out;
}

/** Derive the observation sub-type from a FHIR resource. */
export function getObservationSubType(fhirResource: Record<string, unknown>): ObservationSubType {
  return observationSubTypeFromCategories(categoryStringsFromResource(fhirResource));
}

/** Human badge label (Lab / Vital / Social) from either a flat category list or
 *  a FHIR resource. */
export function getObservationSubTypeLabel(
  input: readonly string[] | Record<string, unknown> | null | undefined,
): string {
  const sub = Array.isArray(input)
    ? observationSubTypeFromCategories(input)
    : getObservationSubType((input as Record<string, unknown>) ?? {});
  return OBS_SUBTYPE_LABELS[sub];
}

/** Short uppercase badge label (LAB / VITAL / SOCIAL). */
export function getObservationSubTypeShort(
  input: readonly string[] | Record<string, unknown> | null | undefined,
): string {
  const sub = Array.isArray(input)
    ? observationSubTypeFromCategories(input)
    : getObservationSubType((input as Record<string, unknown>) ?? {});
  return OBS_SUBTYPE_SHORT[sub];
}

/** Get the observation-specific icon based on its sub-type (Lab / Vital /
 *  Social). Shares the sub-type derivation with the badge label so the icon and
 *  label always agree. */
export function getObservationIcon(fhirResource: Record<string, unknown>): LucideIcon {
  return OBS_SUBTYPE_ICONS[getObservationSubType(fhirResource)];
}
