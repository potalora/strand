/** Exact execution modes shared by settings, uploads, and summaries. */
export type ProcessingMode =
  | "validated_strict_local"
  | "custom_local"
  | "cloud_assisted"
  | "prompt_only";

export type LocalModelRole = "ocr" | "extraction" | "summary";
export type IngestionModelRole = Exclude<LocalModelRole, "summary">;
export type BoundedIngestionModels<T> = [] | [T] | [T, T];

export type PackState =
  | "not_installed"
  | "downloading"
  | "verifying"
  | "preview"
  | "ready"
  | "update_available"
  | "failed";

export type LocalPackStatusReason =
  | "feature_disabled"
  | "release_evidence_missing";

export type LocalPackOperationAction =
  | "install"
  | "update"
  | "verify"
  | "rollback";

export type LocalPackOperationState =
  | "queued"
  | "running"
  | "paused"
  | "failed"
  | "completed";

export interface LocalModelArtifact {
  role: LocalModelRole;
  repository: string;
  revision: string;
  quantization: string;
  runtime: string;
  license: string;
  download_bytes: number;
  expected_memory_bytes: number | null;
  installed: boolean;
  validated: boolean;
}

export interface LocalPackOperation {
  id: string;
  action: LocalPackOperationAction;
  state: LocalPackOperationState;
  current_role: LocalModelRole | null;
  bytes_done: number;
  bytes_total: number;
  message: string | null;
  retryable: boolean;
}

export interface LocalPackOperationCreated {
  operation_id: string;
  state: "queued";
}

export interface LocalPackStatus {
  platform:
    | "apple_silicon"
    | "linux_cpu"
    | "linux_cuda"
    | "linux_rocm"
    | "unsupported";
  compatible: boolean;
  enabled: boolean;
  state: PackState;
  status_reason: LocalPackStatusReason | null;
  active_revision: string | null;
  available_revision: string | null;
  models: LocalModelArtifact[];
  operation: LocalPackOperation | null;
}

/** Bounded model identity exposed on upload status/history responses. */
export interface LocalRunModel {
  role: IngestionModelRole;
  repository: string;
  revision: string;
}

export interface LocalRunInfo {
  privacy_mode: "validated_strict_local";
  models: BoundedIngestionModels<LocalRunModel>;
}

export interface LocalProcessingFailure {
  stage: string;
  code: string;
  message: string;
  model_role: LocalModelRole | null;
  repository: string | null;
  revision: string | null;
  retryable: boolean;
  checkpoint_preserved: boolean;
  cloud_fallback_attempted: false;
}

export type LocalJobKind = "ingestion" | "summary";
export type LocalJobStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

export interface LocalAIJobProgress {
  model_role?: LocalModelRole | null;
  page_index?: number | null;
  page_total?: number | null;
  worker_current?: number | null;
  worker_total?: number | null;
  attempt?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  splits_used?: number | null;
}

export interface LocalAIJobFailure {
  stage: string;
  code: string;
  model_role?: LocalModelRole | null;
  retryable: boolean;
  checkpoint_preserved: boolean;
  cloud_fallback_attempted: boolean;
}

export interface LocalAIJobResponse {
  id: string;
  upload_id: string | null;
  summary_prompt_id: string | null;
  kind: LocalJobKind;
  processing_mode: ProcessingMode;
  status: LocalJobStatus;
  stage: string;
  progress: LocalAIJobProgress | null;
  failure: LocalAIJobFailure | null;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface EvidenceReference {
  id: string;
  page_number: number | null;
  section: string | null;
  excerpt: string;
  start_offset: number | null;
  end_offset: number | null;
  field_paths: string[];
}

export interface ExtractionModelIdentity {
  role: IngestionModelRole;
  repository: string;
  revision: string;
  quantization: string;
  runtime: string;
}

export interface RecordExtractionProvenance {
  record_id: string;
  processing_mode: "validated_strict_local";
  schema_version: string;
  evidence: EvidenceReference[];
  unresolved_fields: string[];
  rejected_fields: string[];
  models: ExtractionModelIdentity[];
}

type UnknownObject = Record<string, unknown>;

function isObject(value: unknown): value is UnknownObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseIngestionModels<T extends LocalRunModel>(
  value: unknown
): BoundedIngestionModels<T> {
  if (
    !Array.isArray(value) ||
    value.length > 2 ||
    value.some(
      (model) =>
        !isObject(model) ||
        (model.role !== "ocr" && model.role !== "extraction") ||
        typeof model.repository !== "string" ||
        typeof model.revision !== "string"
    )
  ) {
    throw new TypeError("Invalid ingestion model provenance");
  }
  return value as BoundedIngestionModels<T>;
}

function parseExtractionModelHistory(
  value: unknown
): ExtractionModelIdentity[] {
  if (
    !Array.isArray(value) ||
    value.length > 512 ||
    value.some(
      (model) =>
        !isObject(model) ||
        (model.role !== "ocr" && model.role !== "extraction") ||
        typeof model.repository !== "string" ||
        typeof model.revision !== "string" ||
        typeof model.quantization !== "string" ||
        typeof model.runtime !== "string"
    )
  ) {
    throw new TypeError("Invalid ingestion model provenance");
  }
  return value as ExtractionModelIdentity[];
}

/** Validate the bounded, summary-free ingestion provenance on upload payloads. */
export function parseLocalRunInfo(value: unknown): LocalRunInfo {
  if (
    !isObject(value) ||
    value.privacy_mode !== "validated_strict_local"
  ) {
    throw new TypeError("Invalid ingestion model provenance");
  }
  return {
    privacy_mode: value.privacy_mode,
    models: parseIngestionModels<LocalRunModel>(value.models),
  };
}

/** Validate the model boundary before displaying record extraction evidence. */
export function parseRecordExtractionProvenance(
  value: unknown
): RecordExtractionProvenance {
  if (
    !isObject(value) ||
    value.processing_mode !== "validated_strict_local"
  ) {
    throw new TypeError("Invalid ingestion model provenance");
  }
  parseExtractionModelHistory(value.models);
  return value as unknown as RecordExtractionProvenance;
}

/** Validated model output: references only, with server-owned text rendering. */
export interface GroundedSummaryClaim {
  fact_id: string;
  field_paths: string[];
  evidence_ids: string[];
}

export interface GroundedSummarySection {
  heading: string;
  claims: GroundedSummaryClaim[];
}

export interface GroundedSummaryUncertainty {
  uncertainty_id: string;
  fact_ids: string[];
  evidence_ids: string[];
}

export interface GroundedSummaryDocument {
  sections: GroundedSummarySection[];
  uncertainties: GroundedSummaryUncertainty[];
}

export interface StrictLocalSummaryModelProvenance {
  processing_mode: "validated_strict_local";
  manifest_sha256: string;
  pack_revision: string;
  model: {
    role: "summary";
    repository: string;
    revision: string;
    quantization: string;
    runtime: {
      name: string;
      version: string;
    };
  };
}

export interface CustomLocalSummaryModelProvenance {
  processing_mode: "custom_local";
  provider: string;
  model: string;
}

export interface CloudSummaryModelProvenance {
  processing_mode: "cloud_assisted";
  provider: string;
  model: string;
}

export type SummaryModelProvenance =
  | StrictLocalSummaryModelProvenance
  | CustomLocalSummaryModelProvenance
  | CloudSummaryModelProvenance;
