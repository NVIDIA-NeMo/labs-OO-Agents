// Typed fetchers for the Memory tab endpoints (/api/memory/*).

export interface MemoryDbInfo {
  path: string;
  size_bytes: number;
  mtime: number;
}

export interface MemoryRecordRow {
  id: string;
  type: string;
  status: string | null;
  owner: string;
  importance: number;
  importance_label: string;
  title: string | null;
  preview: string;
  tags: string[];
  created_at: number;
  last_accessed_at: number;
  archived: boolean;
  fetches: number;
  edge_count: number;
}

export interface MemoryRecordsResponse {
  records: MemoryRecordRow[];
  total: number;
  page: number;
  limit: number;
  has_more: boolean;
}

export interface MemoryReference {
  kind: string;
  key: string;
  preview: string | null;
  captured_at: number;
}

export interface MemoryAccessRecord {
  ts: number;
  channel: string;
  reader_owner: string;
  session_ref: string | null;
  trace_ref: string | null;
  query: string | null;
  score: number | null;
  rank: number | null;
  components: Record<string, number> | null;
}

export interface MemoryEdgeOut {
  target_id: string;
  type: string;
  weight: number;
  target_type: string | null;
  target_preview: string | null;
}

export interface MemoryUsage {
  fetches: number;
  recalled: number;
  searched: number;
  injected: number;
  reinforced: number;
  deref: number;
  last_channel: string | null;
  last_ts: number | null;
  last_session_ref: string | null;
  last_trace_ref: string | null;
  mean_rank: number | null;
  mean_score: number | null;
  injected_never_used: boolean;
  retention: number;
  prune_eta: number | null;
  strength: number;
  reinforcement_count: number;
}

export interface MemoryRecordDetail {
  id: string;
  type: string;
  title: string | null;
  content: string;
  owner: string;
  status: string | null;
  archived: boolean;
  importance: number;
  importance_label: string;
  salience: number;
  salience_label: string;
  confidence: number;
  confidence_label: string;
  mood: string | null;
  strength: number;
  reinforcement_count: number;
  created_at: number;
  last_accessed_at: number;
  access_count: number;
  tags: string[];
  entities: string[];
  place_or_task: string | null;
  references: MemoryReference[];
  access_log: MemoryAccessRecord[];
  usage: MemoryUsage;
  edges: MemoryEdgeOut[];
}

export interface MemoryMaintenanceEntry {
  ts: number;
  kind: string;
  report: Record<string, unknown>;
}

export interface MemoryStats {
  total: number;
  by_type?: Record<string, number>;
  by_owner?: Record<string, number>;
  with_references?: number;
  never_fetched_pct?: number | null;
  fetch_concentration_top10pct?: number;
  total_fetches?: number;
  dedup_reinforces?: number;
  injected_memories?: number;
  injected_used_rate?: number | null;
  todos_open?: number;
  todos_closed?: number;
  todo_median_open_age_hours?: number | null;
  cross_owner_reads?: number;
  maintenance: MemoryMaintenanceEntry[];
}

export interface MemoryExplainRow {
  rank: number;
  id: string;
  score: number;
  source: string;
  cos: number;
  rel: number;
  rec: number;
  imp: number;
  spread: number;
  type: string;
  owner: string;
  head: string;
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      // keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchMemoryDbs(): Promise<MemoryDbInfo[]> {
  return getJson('/api/memory/dbs');
}

export async function fetchMemoryRecords(params: {
  db: string;
  owner?: string;
  type?: string;
  status?: string;
  q?: string;
  include_archived?: boolean;
  page?: number;
  limit?: number;
}): Promise<MemoryRecordsResponse> {
  const searchParams = new URLSearchParams({ db: params.db });
  if (params.owner) searchParams.set('owner', params.owner);
  if (params.type) searchParams.set('type', params.type);
  if (params.status) searchParams.set('status', params.status);
  if (params.q) searchParams.set('q', params.q);
  if (params.include_archived) searchParams.set('include_archived', 'true');
  if (params.page) searchParams.set('page', String(params.page));
  if (params.limit) searchParams.set('limit', String(params.limit));
  return getJson(`/api/memory/records?${searchParams}`);
}

export async function fetchMemoryRecord(db: string, id: string): Promise<MemoryRecordDetail> {
  const searchParams = new URLSearchParams({ db, id });
  return getJson(`/api/memory/record?${searchParams}`);
}

export async function fetchMemoryStats(db: string): Promise<MemoryStats> {
  const searchParams = new URLSearchParams({ db });
  return getJson(`/api/memory/stats?${searchParams}`);
}

export async function fetchMemoryExplain(
  db: string,
  q: string,
  k?: number,
  dim?: number,
): Promise<MemoryExplainRow[]> {
  const searchParams = new URLSearchParams({ db, q });
  if (k) searchParams.set('k', String(k));
  if (dim) searchParams.set('dim', String(dim));
  return getJson(`/api/memory/explain?${searchParams}`);
}
