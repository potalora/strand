export interface TimelineEvent {
  id: string;
  record_type: string;
  display_text: string;
  effective_date: string | null;
  code_display: string | null;
  category: string[] | null;
  // Human-readable provider/performer, server-derived from the record's
  // fhir_resource. Null when the record carries none.
  provider?: string | null;
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
