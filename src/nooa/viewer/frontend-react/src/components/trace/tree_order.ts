import type { TraceEvent } from "@/api/types";

/**
 * Tree (call-graph) ordering for agent traces.
 *
 * The viewer shows spans in their parent-child tree, with each parent's children
 * ordered by start time (DFS preorder). This keeps every subtree contiguous, so
 * concurrent sibling subtrees (parallel sub-agents) don't interleave in the flat
 * timeline. There is no special-casing: siblings are simply ordered by when they
 * started. For a trace with no overlapping siblings, this order is identical to a
 * flat timestamp sort.
 *
 * annotateTreeOrder() tags each event with a `_tree_rank` (its DFS-preorder
 * position); sorting by it produces the tree order. A "Sort by time" toggle in the
 * viewer falls back to plain `start_time_ns` ordering.
 */

const ROOT_KEY = " root";

function startNs(e: TraceEvent): number {
  return e._span_data?.start_time_ns ?? 0;
}

function spanIdOf(e: TraceEvent): string {
  return e.ids?.span_id ?? "";
}

// Stable order: by start time, tie-broken on span id.
function compareEvents(a: TraceEvent, b: TraceEvent): number {
  const sa = startNs(a);
  const sb = startNs(b);
  if (sa !== sb) return sa - sb;
  const ia = spanIdOf(a);
  const ib = spanIdOf(b);
  return ia < ib ? -1 : ia > ib ? 1 : 0;
}

/** Tag every event with `_tree_rank` (DFS preorder, siblings by start time). */
export function annotateTreeOrder(events: TraceEvent[]): void {
  const spanIds = new Set<string>();
  for (const e of events) {
    if (!e._is_span_event) {
      const sid = e.ids?.span_id;
      if (sid) spanIds.add(sid);
    }
  }

  // Group real-span children by parent (orphans/external parents become roots), and
  // span-events by their owning span.
  const childrenOf = new Map<string, TraceEvent[]>();
  const eventsOf = new Map<string, TraceEvent[]>();
  const realSpans: TraceEvent[] = [];

  const push = (map: Map<string, TraceEvent[]>, key: string, e: TraceEvent) => {
    const list = map.get(key);
    if (list) list.push(e);
    else map.set(key, [e]);
  };

  for (const e of events) {
    if (e._is_span_event) {
      const owner = e._parent_span_id;
      if (owner) push(eventsOf, owner, e);
      continue;
    }
    realSpans.push(e);
    const parent = e.ids?.parent_span_id;
    const key = parent && spanIds.has(parent) ? parent : ROOT_KEY;
    push(childrenOf, key, e);
  }

  for (const list of childrenOf.values()) list.sort(compareEvents);
  for (const list of eventsOf.values())
    list.sort((a, b) => startNs(a) - startNs(b));

  // DFS preorder: a span, then its span-events, then its child spans.
  let rank = 0;
  const visited = new Set<string>();

  const visit = (e: TraceEvent) => {
    const id = spanIdOf(e);
    if (id && visited.has(id)) return; // cycle guard — before any rank assignment
    if (id) visited.add(id);

    e._tree_rank = rank++;
    const evs = eventsOf.get(id);
    if (evs) {
      for (const se of evs) se._tree_rank = rank++;
    }

    const kids = childrenOf.get(id);
    if (kids) {
      for (const c of kids) visit(c);
    }
  };

  for (const r of childrenOf.get(ROOT_KEY) ?? []) visit(r);

  // Any spans unreachable via the tree (e.g. cycles) still get a rank so they sort.
  for (const e of realSpans)
    if (e._tree_rank === undefined) e._tree_rank = rank++;
  for (const list of eventsOf.values())
    for (const se of list)
      if (se._tree_rank === undefined) se._tree_rank = rank++;
}
