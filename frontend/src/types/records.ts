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
  created_at: string;
}

export interface RecordListResponse {
  items: HealthRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface UploadResponse {
  id: string;
  filename: string;
  ingestion_status: string;
  record_count: number;
  created_at: string;
}

export interface IngestionStatus {
  id: string;
  filename: string;
  ingestion_status: string;
  ingestion_progress: {
    current_file?: string;
    file_index?: number;
    total_files?: number;
    records_ingested?: number;
    records_failed?: number;
    records_inserted?: number;
    records_updated?: number;
    records_unchanged?: number;
    records_skipped?: number;
    duplicate_of?: string;
    record_count?: number;
    total_entries?: number;
  };
  ingestion_errors: Array<{
    file?: string;
    row?: number;
    error?: string;
  }>;
  record_count: number;
  total_file_count: number;
  processing_started_at: string | null;
  processing_completed_at: string | null;
}
