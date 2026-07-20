// Trace list types (from /api/traces)

export interface TraceGroup {
  id: string;
  name: string;
  modified: string;
  size: number;
  event_count: number | null;
  batch_id: string | null;
}

export interface PaginatedTraceResponse {
  traces: TraceGroup[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

// OTLP span as returned by /api/trace (stored in SQLite, served back as OTLP-format dicts)

export interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: OtlpAttribute[];
  status: { code?: number; message?: string };
  events?: OtlpSpanEvent[];
  _resource?: { attributes?: OtlpAttribute[] };
}

export interface OtlpAttribute {
  key: string;
  value: OtlpValue;
}

export interface OtlpValue {
  stringValue?: string;
  intValue?: string;
  doubleValue?: number;
  boolValue?: boolean;
  arrayValue?: { values: OtlpValue[] };
  kvlistValue?: { values: { key: string; value: OtlpValue }[] };
}

export interface OtlpSpanEvent {
  name: string;
  timeUnixNano: string;
  attributes?: OtlpAttribute[];
}

export interface TraceResponse {
  format: string;
  path: string;
  events: OtlpSpan[];
  total_count: number;
  has_more: boolean;
}

// Normalized event — what plugins consume (converted from OTLP spans)

export interface TraceEvent {
  type: string;
  timestamp: string;
  ids: {
    span_id: string;
    trace_id: string;
    parent_span_id: string | null;
  };
  attributes: Record<string, unknown>;
  body?: string | null;
  span_id?: string;
  trace_id?: string;
  _span_data?: {
    start_time_ns: number;
    end_time_ns: number;
    duration_ns: number;
  };
  _is_span_event?: boolean;
  _parent_span_id?: string;
  // DFS-preorder position (set by annotateTreeOrder); siblings ordered by start time.
  _tree_rank?: number;
}

export type ViewState = 'collapsed' | 'concise' | 'expanded';

// Eval types

export interface ExperimentSummaryItem {
  id: string;
  timestamp: string;
  prompt_version: string;
  models: string[];
  test_count: number;
  passed_count: number;
  status: string;
  suite_name: string | null;
}

export interface PaginatedExperimentsResponse {
  experiments: ExperimentSummaryItem[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}
