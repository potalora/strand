import { test, expect } from "@playwright/test";
import { HeartPulse, TestTube, UserRound } from "lucide-react";
import {
  observationSubTypeFromCategories,
  getObservationSubType,
  getObservationSubTypeLabel,
  getObservationIcon,
} from "./record-icons";

/**
 * Observation sub-type derivation — Lab / Vital / Social (session §2b).
 * Pure logic, no browser/server — run with:
 *   npx playwright test --config playwright.unit.config.ts
 */

test.describe("observationSubTypeFromCategories", () => {
  test("maps the FHIR category codes", () => {
    expect(observationSubTypeFromCategories(["laboratory"])).toBe("lab");
    expect(observationSubTypeFromCategories(["vital-signs"])).toBe("vital");
    expect(observationSubTypeFromCategories(["social-history"])).toBe("social");
  });

  test("is case- and free-text-tolerant", () => {
    expect(observationSubTypeFromCategories(["Vital Signs"])).toBe("vital");
    expect(observationSubTypeFromCategories(["Social History"])).toBe("social");
    expect(observationSubTypeFromCategories(["Laboratory"])).toBe("lab");
  });

  test("defaults to lab when undetectable (matches the lab renderer / TestTube icon)", () => {
    expect(observationSubTypeFromCategories([])).toBe("lab");
    expect(observationSubTypeFromCategories(null)).toBe("lab");
    expect(observationSubTypeFromCategories(undefined)).toBe("lab");
    expect(observationSubTypeFromCategories(["imaging"])).toBe("lab");
  });
});

test.describe("getObservationSubType (from a FHIR resource)", () => {
  test("reads category[].coding[].code", () => {
    expect(getObservationSubType({ category: [{ coding: [{ code: "vital-signs" }] }] })).toBe("vital");
    expect(getObservationSubType({ category: [{ coding: [{ code: "social-history" }] }] })).toBe("social");
    expect(getObservationSubType({ category: [{ coding: [{ code: "laboratory" }] }] })).toBe("lab");
  });

  test("falls back to category[].text", () => {
    expect(getObservationSubType({ category: [{ text: "Vital Signs" }] })).toBe("vital");
  });

  test("defaults to lab when no category", () => {
    expect(getObservationSubType({})).toBe("lab");
  });
});

test.describe("badge label", () => {
  test("renders Lab / Vital / Social from a flat category list", () => {
    expect(getObservationSubTypeLabel(["laboratory"])).toBe("Lab");
    expect(getObservationSubTypeLabel(["vital-signs"])).toBe("Vital");
    expect(getObservationSubTypeLabel(["social-history"])).toBe("Social");
  });
});

test.describe("icon and label agreement", () => {
  // The badge label is derived from the SAME sub-type as the icon, so a record
  // that shows the "Vital" badge always shows the HeartPulse icon, etc.
  test("vital → HeartPulse + 'Vital'", () => {
    const r = { category: [{ coding: [{ code: "vital-signs" }] }] };
    expect(getObservationIcon(r)).toBe(HeartPulse);
    expect(getObservationSubTypeLabel(r)).toBe("Vital");
  });

  test("social → UserRound + 'Social'", () => {
    const r = { category: [{ coding: [{ code: "social-history" }] }] };
    expect(getObservationIcon(r)).toBe(UserRound);
    expect(getObservationSubTypeLabel(r)).toBe("Social");
  });

  test("lab → TestTube + 'Lab'", () => {
    const r = { category: [{ coding: [{ code: "laboratory" }] }] };
    expect(getObservationIcon(r)).toBe(TestTube);
    expect(getObservationSubTypeLabel(r)).toBe("Lab");
  });
});
