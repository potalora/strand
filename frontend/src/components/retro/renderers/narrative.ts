/**
 * Extract plain text from a FHIR Narrative `text.div` (XHTML).
 *
 * AI-extracted encounters store a visit summary in the resource's
 * `text.div` — the canonical FHIR home for a human-readable synopsis. The
 * backend XML-escapes the text when wrapping it, so here we strip the tags,
 * collapse whitespace, and decode the basic entities to render readable prose.
 *
 * Pure — no React/DOM — so it can be unit-tested in isolation.
 */
export function narrativeText(div: unknown): string {
  if (typeof div !== "string" || !div) return "";
  const stripped = div
    .replace(/<[^>]*>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  // Decode `&amp;` LAST so an escaped entity like `&amp;lt;` round-trips to the
  // literal `&lt;` rather than collapsing to `<`.
  return stripped
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}
