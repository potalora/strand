export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface UserResponse {
  id: string;
  email: string;
  display_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  display_name?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface HealthRecord {
  id: string;
  patient_id: string;
  record_type: string;
  fhir_resource_type: string;
  fhir_resource: Record<string, unknown>;
  source_format: string;
  effective_date: string | null;
  status: string | null;
  category: string[] | null;
  code_system: string | null;
  code_value: string | null;
  code_display: string | null;
  display_text: string;
  ai_extracted: boolean;
  confidence_score: number | null;
  created_at: string;
}

export interface RecordListResponse {
  items: HealthRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface TimelineEvent {
  id: string;
  record_type: string;
  display_text: string;
  effective_date: string | null;
  code_display: string | null;
  category: string[] | null;
}

export interface TimelineResponse {
  events: TimelineEvent[];
  total: number;
}

export interface TimelineStats {
  total_records: number;
  records_by_type: Record<string, number>;
  date_range_start: string | null;
  date_range_end: string | null;
}

export interface DashboardOverview {
  total_records: number;
  total_patients: number;
  total_uploads: number;
  records_by_type: Record<string, number>;
  recent_records: {
    id: string;
    record_type: string;
    display_text: string;
    effective_date: string | null;
    created_at: string | null;
  }[];
  date_range_start: string | null;
  date_range_end: string | null;
}

export interface UploadResponse {
  upload_id: string;
  status: string;
  records_inserted: number;
  errors: unknown[];
  unstructured_uploads?: { upload_id: string; filename: string; status: string }[];
}

export interface PendingExtractionFile {
  id: string;
  filename: string;
  mime_type: string;
  file_category: string;
  file_size_bytes: number | null;
  created_at: string | null;
}

export interface TriggerExtractionResponse {
  triggered: number;
  failed: number;
  results: { upload_id: string; status: string }[];
}

export interface PendingExtractionResponse {
  files: PendingExtractionFile[];
  total: number;
}

export interface ExtractionProgressResponse {
  total: number;
  completed: number;
  processing: number;
  failed: number;
  pending: number;
  records_created: number;
}

export interface LabItem {
  id: string;
  display_text: string;
  effective_date: string | null;
  value: number | string | null;
  unit: string;
  reference_low: number | null;
  reference_high: number | null;
  interpretation: string;
  code_display: string | null;
  code_value: string | null;
}

export interface PromptResponse {
  id: string;
  summary_type: string;
  system_prompt: string;
  user_prompt: string;
  target_model: string;
  suggested_config: Record<string, unknown>;
  record_count: number;
  de_identification_report: Record<string, number> | null;
  copyable_payload: string;
  generated_at: string;
}

export interface ExtractedEntity {
  entity_class: string;
  text: string;
  attributes: Record<string, string>;
  start_pos: number | null;
  end_pos: number | null;
  confidence: number;
}

export interface ExtractionResult {
  upload_id: string;
  status: string;
  extracted_text_preview: string | null;
  entities: ExtractedEntity[];
  error: string | null;
}

export interface UnstructuredUploadResponse {
  upload_id: string;
  status: string;
  file_type: string;
}

export interface DuplicateWarning {
  total_records: number;
  deduped_records: number;
  duplicates_excluded: number;
  message: string | null;
}

export interface GenerateSummaryRequest {
  patient_id: string;
  summary_type: string;
  category?: string;
  date_from?: string;
  date_to?: string;
  output_format: string;
  custom_system_prompt?: string;
  custom_user_prompt?: string;
}

export interface GenerateSummaryResponse {
  id: string;
  natural_language: string | null;
  json_data: Record<string, unknown> | null;
  record_count: number;
  duplicate_warning: DuplicateWarning | null;
  de_identification_report: Record<string, number> | null;
  model_used: string;
  generated_at: string;
}

export interface PatientInfo {
  id: string;
  fhir_id: string | null;
  gender: string | null;
  name: string | null;
  birth_date: string | null;
}

export interface SourceBreakdown {
  source: string;
  count: number;
}

export interface SourcesResponse {
  items: SourceBreakdown[];
  total: number;
}

export interface AuditLogEntry {
  id: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  ip_address: string | null;
  details: Record<string, unknown> | null;
  created_at: string | null;
}

export interface AuditLogResponse {
  items: AuditLogEntry[];
  total: number;
  page: number;
  limit: number;
}

export interface SeriesPoint {
  id: string;
  effective_date: string | null;
  value: number;
  unit: string;
}

export interface SeriesResponse {
  code_value: string;
  items: SeriesPoint[];
  total: number;
}

export interface DedupCandidate {
  id: string;
  similarity_score: number;
  match_reasons: Record<string, boolean>;
  status: string;
  record_a: {
    id: string;
    display_text: string;
    record_type: string;
    source_format: string;
    effective_date: string | null;
  } | null;
  record_b: {
    id: string;
    display_text: string;
    record_type: string;
    source_format: string;
    effective_date: string | null;
  } | null;
}
