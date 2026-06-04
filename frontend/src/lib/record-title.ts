// A human-readable title for a record.
//
// Some records arrive with `display_text` set to the bare FHIR resource type
// (e.g. "MedicationStatement", "ImagingStudy") because the source carried only a
// reference and no resolvable name. In those cases we derive a better title from
// the FHIR resource (when available), then fall back to a friendly singular label.

// FHIR resource types the parser supports — if display_text is exactly one of
// these, it isn't a real title.
const RESOURCE_TYPES = new Set([
  "Condition", "Observation", "MedicationRequest", "MedicationStatement",
  "AllergyIntolerance", "Procedure", "Encounter", "Immunization", "DiagnosticReport",
  "DocumentReference", "ImagingStudy", "ServiceRequest", "CarePlan", "Communication",
  "Appointment", "CareTeam", "ImmunizationRecommendation", "QuestionnaireResponse",
]);

// Friendly singular labels (RECORD_TYPE_LABELS in constants.ts is plural).
const SINGULAR: Record<string, string> = {
  condition: "Condition",
  observation: "Observation",
  medication: "Medication",
  encounter: "Encounter",
  immunization: "Immunization",
  procedure: "Procedure",
  allergy: "Allergy",
  imaging: "Imaging study",
  document: "Document",
  diagnostic_report: "Diagnostic report",
  service_request: "Service request",
  communication: "Communication",
  appointment: "Appointment",
  care_plan: "Care plan",
  care_team: "Care team",
  questionnaire_response: "Questionnaire response",
  immunization_recommendation: "Immunization recommendation",
};

function codeableText(cc: unknown): string {
  if (!cc || typeof cc !== "object") return "";
  const o = cc as Record<string, unknown>;
  if (typeof o.text === "string" && o.text.trim()) return o.text.trim();
  const coding = Array.isArray(o.coding) ? o.coding : [];
  for (const c of coding) {
    const cObj = c as Record<string, unknown>;
    const disp = (typeof cObj.display === "string" && cObj.display.trim()) || (typeof cObj.code === "string" && cObj.code.trim());
    if (disp) return String(disp);
  }
  return "";
}

interface TitleInput {
  display_text?: string;
  fhir_resource_type?: string;
  record_type?: string;
  fhir_resource?: Record<string, unknown>;
}

export function recordTitle(record: TitleInput): string {
  const dt = (record.display_text ?? "").trim();
  if (dt && !RESOURCE_TYPES.has(dt)) return dt;

  const r = record.fhir_resource ?? {};
  const derived =
    codeableText(r.code) ||
    codeableText(r.vaccineCode) ||
    codeableText(r.medicationCodeableConcept) ||
    codeableText(Array.isArray(r.type) ? r.type[0] : r.type) ||
    codeableText(r.medication);
  if (derived) return derived;
  if (typeof r.description === "string" && r.description.trim()) return r.description.trim();
  if (typeof r.title === "string" && r.title.trim()) return r.title.trim();

  const rtype = (record.record_type ?? "").toLowerCase();
  return SINGULAR[rtype] || dt || record.fhir_resource_type || "Record";
}
