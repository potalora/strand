"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { HealthRecord } from "@/types/api";
import { usePreferencesStore } from "@/stores/usePreferencesStore";
import { RECORD_TYPE_ICONS, getObservationIcon } from "@/lib/record-icons";
import { RECORD_TYPE_COLORS, DEFAULT_RECORD_COLOR } from "@/lib/constants";
import { RetroBadge } from "@/components/retro/RetroBadge";
import { RetroLoadingState } from "@/components/retro/RetroLoadingState";
import { FhirResourceRenderer } from "@/components/retro/FhirResourceRenderer";
import { ConfirmDialog } from "@/components/retro/ConfirmDialog";
import { AIExtractionBadge, AdvancedSection } from "@/components/retro/renderers/shared";

export default function RecordDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [record, setRecord] = useState<HealthRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [shown, setShown] = useState(false);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [dontAskChecked, setDontAskChecked] = useState(false);

  const { skipDeleteConfirm, setSkipDeleteConfirm } = usePreferencesStore();

  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    api
      .get<HealthRecord>(`/records/${id}`)
      .then(setRecord)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load record"))
      .finally(() => setLoading(false));
  }, [id]);

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
        router.push("/admin?tab=all");
      })
      .catch(() => {
        setDeleting(false);
      });
  }

  if (loading) return <RetroLoadingState text="Loading record" />;

  if (error || !record) {
    return (
      <div className={`screen ${shown ? "on" : ""}`}>
        <div className="page-top">
          <Link href="/admin?tab=all" className="btn ghost sm">
            <ChevronLeft size={15} /> Back to records
          </Link>
        </div>
        <div className="py-12 text-center">
          <p className="muted text-sm">{error || "Record not found"}</p>
        </div>
      </div>
    );
  }

  // Resolve icon + colors for the header chip (record-TYPE hue — neutral, not value-judgement).
  const type = record.record_type.toLowerCase();
  const IconComponent =
    type === "observation" ? getObservationIcon(record.fhir_resource) : RECORD_TYPE_ICONS[type];
  const colors = RECORD_TYPE_COLORS[type] ?? DEFAULT_RECORD_COLOR;

  return (
    <>
      <div className={`screen ${shown ? "on" : ""}`}>
        {/* Back link */}
        <div className="page-top">
          <Link href="/admin?tab=all" className="btn ghost sm">
            <ChevronLeft size={15} /> Back to records
          </Link>
        </div>

        {/* 1. Editorial header: kicker + icon chip + serif title + badge + status */}
        <p className="kicker">Record detail</p>
        <div className="flex items-start gap-3" style={{ marginTop: 6 }}>
          {IconComponent && (
            <div
              className="flex items-center justify-center shrink-0"
              style={{
                width: 44,
                height: 44,
                marginTop: 6,
                background: colors.bg,
                color: colors.text,
                borderRadius: "var(--radius-sm)",
              }}
            >
              <IconComponent size={22} />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <h1 className="display" style={{ fontSize: 34, margin: 0, color: "var(--text)" }}>
              {record.display_text}
            </h1>
            <div className="flex items-center gap-2" style={{ marginTop: 10 }}>
              <RetroBadge recordType={record.record_type} />
              {record.status && (
                <span className="text-xs" style={{ color: "var(--text-dim)" }}>
                  {record.status}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* 2. Primary content: rich type-specific renderer in a panel */}
        <div className="panel" style={{ marginTop: 26 }}>
          <FhirResourceRenderer recordType={record.record_type} fhirResource={record.fhir_resource} />
        </div>

        {/* 3. AI extraction info (conditional) */}
        {record.ai_extracted && (
          <div style={{ marginTop: 18 }}>
            <AIExtractionBadge
              aiExtracted={record.ai_extracted}
              confidenceScore={record.confidence_score}
            />
            <p className="muted text-xs" style={{ marginTop: 8 }}>
              This record was extracted from an unstructured document using AI.
            </p>
          </div>
        )}

        {/* 4. Metadata fields */}
        <div style={{ marginTop: 18 }}>
          <Field label="Date" value={fmtDate(record.effective_date)} />
          <Field label="Source" value={record.source_format} />
          <Field label="FHIR type" value={record.fhir_resource_type} />
          {record.code_value && (
            <Field
              label="Code"
              value={`${record.code_value}${record.code_system ? ` · ${record.code_system}` : ""}`}
              mono
            />
          )}
          {record.code_display && <Field label="Code display" value={record.code_display} />}
          {record.category && record.category.length > 0 && (
            <Field label="Categories" value={record.category.join(", ")} />
          )}
          <Field label="Added" value={fmtDate(record.created_at)} />
        </div>

        {/* 5. Advanced section: collapsible FHIR JSON */}
        <div className="field" style={{ borderBottom: 0 }}>
          <div className="field-l">FHIR resource</div>
          <AdvancedSection fhirResource={record.fhir_resource} />
        </div>

        {/* 6. Delete affordance */}
        <div className="between" style={{ marginTop: 8 }}>
          <span className="muted text-xs">Remove this record from your timeline.</span>
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
    </>
  );
}

function fmtDate(value: string | null): string {
  if (!value) return "Not specified";
  return new Date(value).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
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
