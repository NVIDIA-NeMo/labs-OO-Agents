// --- Annotation types ---

export interface Annotation {
  id: string;
  session_id: string;
  span_id: string | null;
  target: string | null;
  name: string;
  score: number | null;
  label: string | null;
  comment: string | null;
  tags: string[];
  created_at: string;
  author_id: string | null;
  source: string;
  metadata: Record<string, unknown> | null;
}

export interface AnnotationCreate {
  session_id: string;
  span_id?: string | null;
  target?: string | null;
  name: string;
  score?: number | null;
  label?: string | null;
  comment?: string | null;
  tags?: string[];
  source?: string;
}

export interface AnnotationUpdate {
  score?: number | null;
  label?: string | null;
  comment?: string | null;
  tags?: string[] | null;
  target?: string | null;
}

export interface TagInfo {
  tag: string;
  count: number;
}

// --- API functions ---

export async function fetchAnnotations(sessionId: string): Promise<Annotation[]> {
  const res = await fetch(`/api/traces/${encodeURIComponent(sessionId)}/annotations`);
  if (!res.ok) throw new Error(`Failed to fetch annotations: ${res.statusText}`);
  return res.json();
}

export async function createAnnotation(data: AnnotationCreate): Promise<Annotation> {
  const res = await fetch('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to create annotation: ${res.statusText}`);
  return res.json();
}

export async function updateAnnotation(id: string, data: AnnotationUpdate): Promise<Annotation> {
  const res = await fetch(`/api/annotations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to update annotation: ${res.statusText}`);
  return res.json();
}

export async function deleteAnnotation(id: string): Promise<void> {
  const res = await fetch(`/api/annotations/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete annotation: ${res.statusText}`);
}

export async function fetchTags(): Promise<TagInfo[]> {
  const res = await fetch('/api/tags');
  if (!res.ok) throw new Error(`Failed to fetch tags: ${res.statusText}`);
  const data = await res.json();
  return data.tags || [];
}
