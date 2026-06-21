"use client";

import { Info, AlertTriangle } from "lucide-react";
import type { OcrNotice } from "@/lib/api";

/**
 * Per-file OCR provider notices, rendered as compact one-line entries under a
 * file's status (upload history + Admin → Extractions). An `info` notice
 * (provider fallback succeeded) is muted/secondary; a `warning` notice
 * (document unreadable) uses the existing warning/error color. Only the
 * `message` is shown — `detail` is available on the type but not surfaced here.
 *
 * Null-safe: an empty/missing array renders nothing.
 */
export function OcrNotices({ notices }: { notices?: OcrNotice[] | null }) {
  if (!notices || notices.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
      {notices.map((notice, i) => {
        const isWarning = notice.level === "warning";
        const color = isWarning ? "var(--danger)" : "var(--text-muted)";
        const Icon = isWarning ? AlertTriangle : Info;
        return (
          <div
            key={i}
            data-ocr-notice={notice.type}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 6,
              fontSize: 11.5,
              lineHeight: 1.35,
              color,
            }}
          >
            <Icon
              size={12}
              strokeWidth={1.9}
              style={{ flexShrink: 0, marginTop: 1 }}
            />
            <span>{notice.message}</span>
          </div>
        );
      })}
    </div>
  );
}
