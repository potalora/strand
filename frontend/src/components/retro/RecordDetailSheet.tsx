"use client";

import { useEffect, useState } from "react";
import { Trash2, Sparkles } from "lucide-react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { api } from "@/lib/api";
import type { HealthRecord, SeriesResponse, SeriesPoint } from "@/types/api";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import { useUIStore } from "@/stores/useUIStore";
import { RECORD_TYPE_ICONS, getObservationIcon } from "@/lib/record-icons";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroBadge } from "./RetroBadge";
import { RetroLoadingState } from "./RetroLoadingState";
import { FhirResourceRenderer } from "./FhirResourceRenderer";
import { Sparkline } from "./DataViz";
import { ConfirmDialog } from "./ConfirmDialog";
import { AIExtractionBadge, AdvancedSection } from "./renderers/shared";

interface RecordDetailSheetProps {
  recordId: string | null;
  open: boolean;
  onClose: () => void;
  onDelete?: () => void;
}

export function RecordDetailSheet({ recordId, open, onClose, onDelete }: RecordDetailSheetProps) {
  const [record, setRecord] = useState<HealthRecord | null>(null);
  const [trend, setTrend] = useState<SeriesPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [dontAskChecked, setDontAskChecked] = useState(false);

  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();
  const setDetailOpen = useUIStore((s) => s.setDetailOpen);

  // Tuck the floating dock away while the sheet is open.
  useEffect(() => {
    setDetailOpen(open);
    return () => setDetailOpen(false);
  }, [open, setDetailOpen]);

  useEffect(() => {
    if (!recordId || !open) {
      setRecord(null);
      return;
    }

    setLoading(true);
    setTrend([]);
    api
      .get<HealthRecord>(`/records/${recordId}`)
      .then((rec) => {
        setRecord(rec);
        // For recurring observations, pull the recorded series for a neutral trend line.
        if (rec.record_type === "observation" && rec.code_value) {
          api
            .get<SeriesResponse>(`/records/series?code_value=${encodeURIComponent(rec.code_value)}`)
            .then((s) => setTrend(s.items ?? []))
            .catch(() => setTrend([]));
        }
      })
      .catch(() => setRecord(null))
      .finally(() => setLoading(false));
  }, [recordId, open]);

  function handleDeleteClick() {
    if (skipDeleteConfirm) {
      performDelete();
    } else {
      setDontAskChecked(false);
      setConfirmOpen(true);
    }
  }

  function performDelete() {
    if (!record) return;
    setDeleting(true);
    api
      .delete(`/records/${record.id}`)
      .then(() => {
        if (dontAskChecked) {
          setSkipDeleteConfirm(true);
        }
        setConfirmOpen(false);
        onDelete?.();
        onClose();
      })
      .catch(() => {
        setDeleting(false);
      });
  }

  // Resolve icon + colors for the header chip
  const type = record?.record_type?.toLowerCase() ?? "";
  const IconComponent =
    type === "observation" && record ? getObservationIcon(record.fhir_resource) : RECORD_TYPE_ICONS[type];
  const colors = RECORD_TYPE_COLORS[type] ?? DEFAULT_RECORD_COLOR;

  return (
    <>
      <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
        <SheetContent
          className="w-full sm:max-w-xl overflow-auto border-l p-0"
          style={{ background: "var(--card)", borderColor: "var(--border)" }}
        >
          <SheetHeader className="px-6 pt-6 pb-5 border-b" style={{ borderColor: "var(--border)" }}>
            <SheetTitle asChild>
              <p className="kicker" style={{ margin: 0 }}>
                Record detail
              </p>
            </SheetTitle>
          </SheetHeader>

          {loading ? (
            <RetroLoadingState text="Loading record" />
          ) : !record ? (
            <div className="py-10 text-center">
              <span className="muted text-sm">Record not found</span>
            </div>
          ) : (
            <div className="px-6 py-6 space-y-5">
              {/* 1. Header: icon chip + serif title + badge + status */}
              <div className="flex items-start gap-3">
                {IconComponent && (
                  <div
                    className="flex items-center justify-center w-10 h-10 shrink-0 mt-1"
                    style={{
                      background: colors.bg,
                      color: colors.text,
                      borderRadius: "var(--radius-sm)",
                    }}
                  >
                    <IconComponent size={20} />
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <h2 className="display" style={{ fontSize: 24, margin: 0, color: "var(--text)" }}>
                    {record.display_text}
                  </h2>
                  <div className="flex items-center gap-2 mt-2.5">
                    <RetroBadge recordType={record.record_type} />
                    {record.status && (
                      <span className="text-xs" style={{ color: "var(--text-dim)" }}>
                        {record.status}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* 2. Primary content: rich type-specific renderer */}
              <div className="panel">
                <FhirResourceRenderer recordType={record.record_type} fhirResource={record.fhir_resource} />
              </div>

              {/* 2b. Recorded trend for recurring observations (neutral, descriptive). */}
              {trend.length >= 2 && (
                <div className="panel">
                  <div className="field-l" style={{ marginBottom: 4 }}>
                    Recorded results · {trend.length} on file
                  </div>
                  <Sparkline points={trend} />
                  <div className="between mono muted" style={{ fontSize: 11, marginTop: 4 }}>
                    <span>
                      {fmtShort(trend[0].effective_date)} · {trend[0].value}
                      {trend[0].unit ? ` ${trend[0].unit}` : ""}
                    </span>
                    <span>
                      {fmtShort(trend[trend.length - 1].effective_date)} · {trend[trend.length - 1].value}
                      {trend[trend.length - 1].unit ? ` ${trend[trend.length - 1].unit}` : ""}
                    </span>
                  </div>
                </div>
              )}

              {/* 3. AI extraction info (conditional) */}
              {record.ai_extracted && (
                <AIExtractionBadge aiExtracted={record.ai_extracted} confidenceScore={record.confidence_score} />
              )}

              {/* 4. Metadata fields */}
              <div>
                <Field label="Date" value={fmtDate(record.effective_date)} />
                <Field label="Source" value={record.source_format} />
                {record.code_value && (
                  <Field
                    label="Code"
                    value={`${record.code_value}${record.code_system ? ` · ${record.code_system}` : ""}`}
                    mono
                  />
                )}
                {record.category && record.category.length > 0 && (
                  <Field label="Categories" value={record.category.join(", ")} />
                )}
                <Field label="Added" value={fmtDate(record.created_at)} />
              </div>

              {/* 5. Advanced section: collapsible FHIR JSON */}
              <div className="field" style={{ borderBottom: 0, paddingBottom: 0 }}>
                <div className="field-l">FHIR resource</div>
                <AdvancedSection fhirResource={record.fhir_resource} />
              </div>

              {/* 6. Actions */}
              <div className="flex items-center gap-3 pt-1">
                <button className="btn" style={{ flex: 1 }} onClick={() => api.get(`/records/${record.id}`)}>
                  <Sparkles size={15} /> Add to summary
                </button>
                <button
                  onClick={handleDeleteClick}
                  disabled={deleting}
                  className="row-del"
                  title="Delete record"
                  aria-label="Delete record"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>

      {record && (
        <ConfirmDialog
          open={confirmOpen}
          title="Delete record?"
          description={record.display_text}
          confirmLabel="Delete"
          cancelLabel="Cancel"
          variant="destructive"
          onConfirm={performDelete}
          onCancel={() => setConfirmOpen(false)}
          showDontAskAgain
          dontAskAgainChecked={dontAskChecked}
          onDontAskAgainChange={setDontAskChecked}
        />
      )}
    </>
  );
}

function fmtDate(value: string | null): string {
  if (!value) return "Not specified";
  return new Date(value).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

function fmtShort(value: string | null | undefined): string {
  if (!value) return "";
  return new Date(value).toLocaleDateString("en-US", { year: "2-digit", month: "short" });
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="field">
      <div className="field-l">{label}</div>
      <div className={`field-v ${mono ? "mono" : ""}`} style={mono ? { fontSize: 13 } : undefined}>
        {value}
      </div>
    </div>
  );
}
