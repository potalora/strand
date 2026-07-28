"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { RecordExtractionProvenance } from "@/types/local-ai";

function modelName(repository: string): string {
  const leaf = repository.split("/").pop() ?? repository;
  if (/ovisocr2/i.test(leaf)) return "OvisOCR2";
  if (/nuextract3/i.test(leaf)) return "NuExtract3";
  return leaf;
}

function shortRevision(revision: string): string {
  return revision.length > 10 ? revision.slice(0, 10) : revision;
}

interface ExtractionEvidencePanelProps {
  recordId: string;
  enabled: boolean;
}

export function ExtractionEvidencePanel({
  recordId,
  enabled,
}: ExtractionEvidencePanelProps) {
  const [loaded, setLoaded] = useState<{
    recordId: string;
    provenance: RecordExtractionProvenance | null;
  } | null>(null);

  useEffect(() => {
    let active = true;
    if (!enabled) {
      return () => {
        active = false;
      };
    }

    api
      .getRecordEvidence(recordId)
      .then((data) => {
        if (active) setLoaded({ recordId, provenance: data });
      })
      .catch(() => {
        if (active) setLoaded({ recordId, provenance: null });
      });

    return () => {
      active = false;
    };
  }, [enabled, recordId]);

  if (!enabled) return null;
  const loading = loaded?.recordId !== recordId;
  const provenance =
    loaded?.recordId === recordId ? loaded.provenance : null;

  return (
    <section className="panel space-y-3" aria-labelledby={`evidence-${recordId}`}>
      <div>
        <h3
          id={`evidence-${recordId}`}
          className="sec-title"
          style={{ margin: 0 }}
        >
          Extraction evidence
        </h3>
        <p className="muted text-xs" style={{ marginTop: 4 }}>
          Source excerpts supporting fields extracted from this document.
        </p>
      </div>

      {loading ? (
        <p className="muted mono text-xs">Loading evidence…</p>
      ) : !provenance ? (
        <p className="muted text-xs">No grounded evidence is available for this record.</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            {provenance.models.map((model) => (
              <span className="tag" key={`${model.role}-${model.revision}`}>
                {modelName(model.repository)} · {shortRevision(model.revision)}
              </span>
            ))}
          </div>

          <div className="space-y-3">
            {provenance.evidence.map((item) => (
              <article
                key={item.id}
                className="field"
                style={{ display: "block", paddingBlock: 10 }}
              >
                <div className="flex flex-wrap gap-2">
                  {item.page_number !== null && (
                    <span className="tag">Page {item.page_number}</span>
                  )}
                  {item.section && <span className="tag">{item.section}</span>}
                </div>
                <blockquote
                  className="text-sm"
                  style={{
                    marginTop: 8,
                    color: "var(--text)",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {item.excerpt.slice(0, 500)}
                </blockquote>
                {item.field_paths.length > 0 && (
                  <p className="mono muted text-xs" style={{ marginTop: 8 }}>
                    {item.field_paths.join(" · ")}
                  </p>
                )}
              </article>
            ))}
          </div>

          {(provenance.unresolved_fields.length > 0 ||
            provenance.rejected_fields.length > 0) && (
            <div className="space-y-1">
              {provenance.unresolved_fields.length > 0 && (
                <p className="muted text-xs">
                  Unresolved: {provenance.unresolved_fields.join(", ")}
                </p>
              )}
              {provenance.rejected_fields.length > 0 && (
                <p className="muted text-xs">
                  Rejected: {provenance.rejected_fields.join(", ")}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
