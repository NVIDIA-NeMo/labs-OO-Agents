import type {
  PaginatedTraceResponse,
  TraceResponse,
  OtlpSpan,
  OtlpAttribute,
  OtlpValue,
  TraceEvent,
} from './types';
import { annotateTreeOrder } from '@/components/trace/tree_order';

export async function fetchTraces(params: {
  page?: number;
  limit?: number;
  search?: string;
  experiment?: string;
  batch_id?: string;
  sort_by?: string;
  sort_dir?: string;
}): Promise<PaginatedTraceResponse> {
  const searchParams = new URLSearchParams();
  if (params.page) searchParams.set('page', String(params.page));
  if (params.limit) searchParams.set('limit', String(params.limit));
  if (params.search) searchParams.set('search', params.search);
  if (params.experiment) searchParams.set('experiment', params.experiment);
  if (params.batch_id) searchParams.set('batch_id', params.batch_id);
  if (params.sort_by) searchParams.set('sort_by', params.sort_by);
  if (params.sort_dir) searchParams.set('sort_dir', params.sort_dir);

  const res = await fetch(`/api/traces?${searchParams}`);
  if (!res.ok) throw new Error(`Failed to fetch traces: ${res.statusText}`);
  return res.json();
}

export async function fetchTrace(sessionId: string): Promise<TraceResponse> {
  const res = await fetch(`/api/trace?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(`Failed to fetch trace: ${res.statusText}`);
  return res.json();
}

export async function fetchTraceResource(
  sessionId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  const res = await fetch(`/api/trace/resource?session_id=${encodeURIComponent(sessionId)}`, {
    signal,
  });
  if (!res.ok) throw new Error(`Failed to fetch trace resource: ${res.statusText}`);
  return res.json();
}

export async function deleteTrace(sessionId: string): Promise<void> {
  const res = await fetch(`/api/traces/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Failed to delete trace: ${res.statusText}`);
}

// OTLP attribute parsing

function extractOtlpValue(v: OtlpValue): unknown {
  if ('stringValue' in v && v.stringValue !== undefined) return v.stringValue;
  if ('intValue' in v && v.intValue !== undefined) return parseInt(v.intValue, 10);
  if ('doubleValue' in v && v.doubleValue !== undefined) return v.doubleValue;
  if ('boolValue' in v && v.boolValue !== undefined) return v.boolValue;
  if ('arrayValue' in v && v.arrayValue) return v.arrayValue.values.map(extractOtlpValue);
  if ('kvlistValue' in v && v.kvlistValue) {
    const obj: Record<string, unknown> = {};
    for (const kv of v.kvlistValue.values) {
      obj[kv.key] = extractOtlpValue(kv.value);
    }
    return obj;
  }
  return null;
}

export function otlpAttrsToDict(attrs: OtlpAttribute[]): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  if (!Array.isArray(attrs)) return result;
  for (const attr of attrs) {
    result[attr.key] = extractOtlpValue(attr.value);
  }
  return result;
}

function nsToNumber(ns: string | undefined, fallback: string = '0'): number {
  try {
    return Number(BigInt(ns || fallback));
  } catch {
    return parseInt(ns || fallback, 10);
  }
}

function nsDuration(startStr: string | undefined, endStr: string | undefined): number {
  try {
    const s = BigInt(startStr || '0');
    const e = BigInt(endStr || startStr || '0');
    return Number(e - s);
  } catch {
    const s = parseInt(startStr || '0', 10);
    const e = parseInt(endStr || startStr || '0', 10);
    return e - s;
  }
}

/**
 * Derive the viewer plugin type, using span.name for more specific grouping
 * when the plugin is a generic category (e.g. "method" -> "method.handle").
 */
function derivePluginType(attrs: Record<string, unknown>, spanName: string | undefined): string {
  const plugin = attrs['nemo_oo_agents.viewer.plugin'] as string | undefined;
  if (!plugin) return spanName || 'unknown';

  // For method/method_call plugins, the span name (e.g. "method.handle")
  // provides the specific method name — use it for sub-type grouping in the sidebar.
  if ((plugin === 'method' || plugin === 'method_call') && spanName && spanName.startsWith(plugin + '.')) {
    return spanName;
  }

  return plugin;
}

export function convertOtlpSpansToEvents(spans: OtlpSpan[]): TraceEvent[] {
  const events: TraceEvent[] = [];

  for (const span of spans) {
    const startNs = nsToNumber(span.startTimeUnixNano);
    const endNs = nsToNumber(span.endTimeUnixNano, span.startTimeUnixNano || '0');
    const durationNs = nsDuration(span.startTimeUnixNano, span.endTimeUnixNano);
    const startTimestamp = new Date(startNs / 1e6).toISOString();

    const resourceAttrs = otlpAttrsToDict(span._resource?.attributes || []);
    const spanAttrs = otlpAttrsToDict(span.attributes || []);
    const flatAttrs = { ...resourceAttrs, ...spanAttrs };

    const statusCode = span.status?.code;
    let statusStr = 'UNSET';
    if (statusCode === 1) statusStr = 'OK';
    else if (statusCode === 2) statusStr = 'ERROR';

    events.push({
      type: `span.${derivePluginType(flatAttrs, span.name)}`,
      timestamp: startTimestamp,
      ids: {
        span_id: span.spanId,
        trace_id: span.traceId,
        parent_span_id: span.parentSpanId || null,
      },
      attributes: {
        ...flatAttrs,
        span_name: span.name,
        start_time_ns: startNs,
        end_time_ns: endNs,
        duration_ns: durationNs,
        status_code: statusStr,
        status_description: span.status?.message || null,
      },
      _span_data: {
        start_time_ns: startNs,
        end_time_ns: endNs,
        duration_ns: durationNs,
      },
      span_id: span.spanId,
      trace_id: span.traceId,
    });

    if (span.events && Array.isArray(span.events)) {
      for (const se of span.events) {
        const eventNs = nsToNumber(se.timeUnixNano, span.startTimeUnixNano || '0');
        const eventTimestamp = new Date(eventNs / 1e6).toISOString();
        const eventAttrs = otlpAttrsToDict(se.attributes || []);

        events.push({
          type: se.name || 'span.event',
          timestamp: eventTimestamp,
          ids: {
            span_id: span.spanId,
            trace_id: span.traceId,
            parent_span_id: span.parentSpanId || null,
          },
          attributes: {
            ...eventAttrs,
            span_name: span.name,
          },
          body: (eventAttrs.message as string) || (eventAttrs.reasoning as string) || null,
          _span_data: {
            start_time_ns: eventNs,
            end_time_ns: eventNs,
            duration_ns: 0,
          },
          _is_span_event: true,
          _parent_span_id: span.spanId,
        });
      }
    }
  }

  events.sort((a, b) => {
    const timeA = a._span_data?.start_time_ns || 0;
    const timeB = b._span_data?.start_time_ns || 0;
    return timeA - timeB;
  });

  // Tag each event with its tree-order rank so the viewer can render spans in
  // call-graph order (de-interleaving concurrent subtrees).
  annotateTreeOrder(events);

  return events;
}
