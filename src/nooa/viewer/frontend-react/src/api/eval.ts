import type { PaginatedExperimentsResponse } from './types';

export async function fetchExperiments(params: {
  page?: number;
  limit?: number;
  search?: string;
}): Promise<PaginatedExperimentsResponse> {
  const searchParams = new URLSearchParams();
  if (params.page) searchParams.set('page', String(params.page));
  if (params.limit) searchParams.set('limit', String(params.limit));
  if (params.search) searchParams.set('search', params.search);

  const res = await fetch(`/api/eval/experiments?${searchParams}`);
  if (!res.ok) throw new Error(`Failed to fetch experiments: ${res.statusText}`);
  return res.json();
}

export interface ColumnInfo {
  key: string;
  values: string[];
}

export interface ExperimentDetail {
  experiment_id: string;
  results: TestResult[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
  metadata_keys: string[];
  columns: ColumnInfo[];
  models: string[];
  variants: string[];
  timestamp: string;
  status: string;
}

export interface TestResult {
  session_id: string;
  test_id: string;
  test_name: string;
  display_name: string;
  passed: boolean | null;
  duration_ms?: number | null;
  span_count?: number | null;
  model?: string;
  agent_class?: string;
  method?: string;
  tier?: string;
  score?: number;
  weighted_score?: number;
  error?: string;
  scores?: Record<string, { score: number; passed: boolean; reasoning?: string }>;
  [key: string]: unknown;
}

export interface ExperimentSummary {
  overall: {
    total: number;
    passed: number;
    avg_score: number;
    success_rate: number;
    run_count?: number;
    total_input_tokens?: number;
    total_output_tokens?: number;
  };
  by_model: Record<string, { total: number; passed: number }>;
  by_test_type: Record<string, { total: number; passed: number }>;
  by_tier: Record<string, { total: number; passed: number }>;
  matrix: {
    models: string[];
    test_types: string[];
    cells: Record<string, Record<string, { passed: number; total: number; rate: number }>>;
  };
}

export async function fetchExperimentDetail(
  experimentId: string,
  params?: {
    page?: number;
    limit?: number;
    sort_by?: string;
    sort_dir?: string;
    search?: string;
    meta?: Record<string, string>;
  },
): Promise<ExperimentDetail> {
  const searchParams = new URLSearchParams();
  if (params?.page) searchParams.set('page', String(params.page));
  if (params?.limit) searchParams.set('limit', String(params.limit));
  if (params?.sort_by) searchParams.set('sort_by', params.sort_by);
  if (params?.sort_dir) searchParams.set('sort_dir', params.sort_dir);
  if (params?.search) searchParams.set('search', params.search);
  if (params?.meta) {
    for (const [key, val] of Object.entries(params.meta)) {
      if (val) searchParams.set(key, val);
    }
  }
  const qs = searchParams.toString();
  const url = `/api/eval/experiment/${encodeURIComponent(experimentId)}${qs ? `?${qs}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch experiment: ${res.statusText}`);
  return res.json();
}

export async function fetchExperimentSummary(
  experimentId: string,
  params?: {
    search?: string;
    meta?: Record<string, string>;
  },
): Promise<ExperimentSummary> {
  const searchParams = new URLSearchParams();
  if (params?.search) searchParams.set('search', params.search);
  if (params?.meta) {
    for (const [key, val] of Object.entries(params.meta)) {
      if (val) searchParams.set(key, val);
    }
  }
  const qs = searchParams.toString();
  const url = `/api/eval/experiment/${encodeURIComponent(experimentId)}/summary${qs ? `?${qs}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch experiment summary: ${res.statusText}`);
  return res.json();
}

export interface HistoricalExperiment {
  experiment_id: string;
  created_at: string;
  tier_metrics: Record<string, { success_rate: number; tests_passed: number; tests_total: number }>;
}

export interface ExperimentMetrics {
  history: HistoricalExperiment[];
}

export async function fetchExperimentMetrics(limit?: number): Promise<ExperimentMetrics> {
  const params = limit && limit > 0 ? `?limit=${limit}` : '';
  const res = await fetch(`/api/eval/experiments/metrics${params}`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  return res.json();
}
