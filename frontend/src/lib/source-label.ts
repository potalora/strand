// Human-readable label for a record's provenance. Mirrors the backend
// `source_label()` so the UI never shows raw machine tokens (`cda_r2`) or
// system identifiers (`urn:oid:2.16…`).

const SOURCE_LABELS: Record<string, string> = {
  fhir_r4: "FHIR R4",
  fhir: "FHIR R4",
  epic_ehi: "Epic EHI",
  epic_ehi_single: "Epic EHI",
  cda_r2: "CDA",
  cda: "CDA",
  ai_extracted: "AI extraction",
};

function isMachineIdentifier(value: string): boolean {
  const s = value.trim().toLowerCase();
  if (!s) return true;
  if (s.startsWith("urn:") || s.startsWith("http://") || s.startsWith("https://") || s.startsWith("oid:")) {
    return true;
  }
  // bare OID, e.g. "2.16.840.1.113883.19"
  if (s.includes(".") && /^[0-9.]+$/.test(s)) return true;
  return false;
}

export function sourceLabel(sourceFormat?: string | null, sourceSystem?: string | null): string {
  if (sourceSystem && !isMachineIdentifier(sourceSystem)) {
    const key = sourceSystem.trim().toLowerCase();
    return SOURCE_LABELS[key] ?? sourceSystem.trim();
  }
  const fmt = (sourceFormat ?? "").trim();
  if (!fmt) return "Unknown";
  return SOURCE_LABELS[fmt.toLowerCase()] ?? fmt.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
