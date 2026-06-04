"use client";

import React from "react";
import { DetailRow, StatusBadge, BarRow, str, obj, arr, nested, formatDate } from "./shared";

// Pull display/text out of a CodeableConcept (or its first coding).
function codeableText(val: unknown): string {
  const c = obj(val);
  return (
    str(c.text) ||
    str(nested(c, "coding", "0", "display")) ||
    str(nested(c, "coding", "0", "code"))
  );
}

export function DocumentRenderer({ r }: { r: Record<string, unknown> }) {
  const docType =
    str(nested(r, "type", "text")) ||
    str(nested(r, "type", "coding", "0", "display")) ||
    str(nested(r, "type", "coding", "0", "code")) ||
    "";
  const description = str(r.description);
  const date = formatDate(r.date);
  const status = str(r.status);
  const docStatus = str(r.docStatus);
  const authors = arr(r.author)
    .map((a) => str(obj(a).display))
    .filter(Boolean);

  const categoryTexts = arr(r.category).map(codeableText).filter(Boolean);
  const isScanned = categoryTexts.some((t) => t.toLowerCase().includes("scanned"));

  // Content attachments — iterate all content[] entries.
  const contents = arr(r.content).map((c) => {
    const att = obj(obj(c).attachment);
    const contentType = str(att.contentType);
    return {
      contentType,
      label: contentType ? (contentType.split("/").pop() ?? "").toUpperCase() : "",
      title: str(att.title),
      url: str(att.url),
      creation: formatDate(att.creation),
    };
  });
  const contentLabels = Array.from(new Set(contents.map((c) => c.label).filter(Boolean)));

  // context — period, encounter, facility, practice setting (R4)
  const ctx = obj(r.context);
  const contextStart = formatDate(nested(ctx, "period", "start"));
  const contextEnd = formatDate(nested(ctx, "period", "end"));
  const encounter =
    str(nested(ctx, "encounter", "0", "display")) || str(nested(ctx, "encounter", "display"));
  const facilityType = codeableText(ctx.facilityType);
  const practiceSetting = codeableText(ctx.practiceSetting);

  return (
    <div className="space-y-3">
      {docType && (
        <p
          className="text-base font-semibold"
          style={{ color: "var(--theme-text)", fontFamily: "var(--font-body)" }}
        >
          {docType}
        </p>
      )}

      {/* Badge row: status, docStatus, scanned, category, content format */}
      {(status || docStatus || isScanned || categoryTexts.length > 0 || contentLabels.length > 0) && (
        <div className="flex flex-wrap items-center gap-2">
          {status && <StatusBadge label={status} />}
          {docStatus && <StatusBadge label={docStatus} color="var(--theme-bg-deep)" />}
          {isScanned && <StatusBadge label="Scanned" color="var(--theme-sienna)" />}
          {categoryTexts.map((cat) => (
            <span
              key={cat}
              className="px-2 py-0.5 text-xs font-medium rounded"
              style={{
                backgroundColor: "var(--record-document-bg)",
                color: "var(--record-document-text)",
              }}
            >
              {cat}
            </span>
          ))}
          {contentLabels.map((label) => (
            <span
              key={label}
              className="px-1.5 py-0.5 text-[10px] font-medium rounded"
              style={{
                backgroundColor: "var(--record-document-bg)",
                color: "var(--record-document-text)",
              }}
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Description text area */}
      {description && (
        <div
          className="px-3 py-2 rounded-md text-xs"
          style={{
            backgroundColor: "var(--theme-bg-deep)",
            color: "var(--theme-text-dim)",
          }}
        >
          {description}
        </div>
      )}

      {authors.length > 0 && <DetailRow label="Author" value={authors.join(", ")} />}
      <DetailRow label="Date" value={date} />

      {/* Encounter context */}
      {(contextStart || encounter || facilityType || practiceSetting) && (
        <BarRow
          items={[
            {
              label: "Period",
              value: contextStart
                ? `${contextStart}${contextEnd && contextEnd !== contextStart ? ` - ${contextEnd}` : ""}`
                : "",
            },
            { label: "Encounter", value: encounter },
            { label: "Facility", value: facilityType },
            { label: "Setting", value: practiceSetting },
          ]}
        />
      )}

      {/* Attachment details (title / url / creation) when present */}
      {contents.some((c) => c.title || c.url || c.creation) && (
        <div
          className="px-3 py-2 rounded-md text-xs space-y-1.5"
          style={{ backgroundColor: "var(--theme-bg-deep)" }}
        >
          <div className="text-[10px] font-medium" style={{ color: "var(--theme-text-muted)" }}>
            Attachments
          </div>
          {contents.map((c, i) => {
            const meta = [c.title, c.label, c.creation && `created ${c.creation}`]
              .filter(Boolean)
              .join(" · ");
            if (!meta && !c.url) return null;
            return (
              <div key={i} className="min-w-0">
                {meta && <span style={{ color: "var(--theme-text)" }}>{meta}</span>}
                {c.url && (
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block truncate font-mono text-[11px] underline"
                    style={{ color: "var(--record-document-text)" }}
                  >
                    {c.url}
                  </a>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
