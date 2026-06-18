import { test, expect } from "@playwright/test";
import { narrativeText } from "./narrative";

/**
 * Plain-text extraction from a FHIR Narrative `text.div` — used to surface an
 * AI-extracted visit summary on the EncounterRenderer. Pure logic, no browser:
 *   npx playwright test --config playwright.unit.config.ts
 */

test.describe("narrativeText", () => {
  test("strips the XHTML wrapper to plain prose", () => {
    const div =
      '<div xmlns="http://www.w3.org/1999/xhtml">Follow-up for reflux; continue regimen.</div>';
    expect(narrativeText(div)).toBe("Follow-up for reflux; continue regimen.");
  });

  test("decodes the escaped XML entities the backend writes", () => {
    const div = '<div xmlns="http://www.w3.org/1999/xhtml">BP &lt;120 &amp; stable</div>';
    expect(narrativeText(div)).toBe("BP <120 & stable");
  });

  test("collapses whitespace across nested tags", () => {
    const div = "<div>\n  Line one.   <span>Line two.</span>\n</div>";
    expect(narrativeText(div)).toBe("Line one. Line two.");
  });

  test("returns empty string for missing or non-string input", () => {
    expect(narrativeText(undefined)).toBe("");
    expect(narrativeText(null)).toBe("");
    expect(narrativeText("")).toBe("");
    expect(narrativeText(42)).toBe("");
  });
});
