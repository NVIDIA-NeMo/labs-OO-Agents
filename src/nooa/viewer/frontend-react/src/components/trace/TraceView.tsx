import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { fetchTrace, convertOtlpSpansToEvents } from "@/api/traces";
import {
  fetchAnnotations,
  createAnnotation,
  deleteAnnotation,
} from "@/api/annotations";
import type { Annotation } from "@/api/annotations";
import { EventList } from "@/components/trace/EventList";
import { FilterSidebar } from "@/components/trace/FilterSidebar";
import type { FilterState } from "@/components/trace/FilterSidebar";
import { AnnotationForm } from "@/components/annotations/AnnotationForm";
import { KeyboardShortcutsHelp } from "@/components/shared/KeyboardShortcutsHelp";
import { useKeyboardNav } from "@/hooks/useKeyboardNav";
import { PlaygroundProvider } from "@/components/playground/PlaygroundContext";
import type { TraceEvent, ViewState } from "@/api/types";

import "@/components/plugins";

const COLLAPSED_BY_DEFAULT = ["span.context_snapshot"];

const EXCLUDED_SEARCH_KEYS = new Set([
  "line",
  "lineno",
  "line_number",
  "col",
  "column",
  "col_offset",
  "index",
  "idx",
  "pos",
  "tb_lineno",
  "frame_lineno",
]);

function getDefaultState(eventType?: string): ViewState {
  if (eventType && COLLAPSED_BY_DEFAULT.some((p) => eventType.startsWith(p))) {
    return "collapsed";
  }
  return "concise";
}

function eventToSearchString(event: TraceEvent): string {
  const parts: string[] = [event.type];
  if (event.body) parts.push(event.body);
  for (const [key, val] of Object.entries(event.attributes)) {
    if (EXCLUDED_SEARCH_KEYS.has(key)) continue;
    if (val != null) parts.push(String(val));
  }
  return parts.join(" ").toLowerCase();
}

function applyFilters(events: TraceEvent[], filter: FilterState): TraceEvent[] {
  let result = events;

  if (filter.enabledTypes.size < countUniqueTypes(events)) {
    result = result.filter((e) => filter.enabledTypes.has(e.type));
  }

  if (filter.spanId) {
    result = result.filter(
      (e) =>
        e.ids?.span_id === filter.spanId || e._parent_span_id === filter.spanId,
    );
  }

  if (filter.textSearch) {
    const term = filter.textSearch.toLowerCase();
    result = result.filter((e) => eventToSearchString(e).includes(term));
  }

  // Call-tree order: DFS preorder, siblings by start time, so concurrent subtrees
  // stay contiguous at every depth. Degrades to a flat timestamp sort for traces
  // with no nesting. Eval-summary spans are floated to the top, as before.
  result = [...result].sort((a, b) => (a._tree_rank ?? 0) - (b._tree_rank ?? 0));
  if (result.some((e) => e.type.startsWith("span.eval"))) {
    const evalEvents = result.filter((e) => e.type.startsWith("span.eval"));
    const rest = result.filter((e) => !e.type.startsWith("span.eval"));
    result = [...evalEvents, ...rest];
  }

  return result;
}

function countUniqueTypes(events: TraceEvent[]): number {
  const types = new Set<string>();
  for (const e of events) types.add(e.type);
  return types.size;
}

interface TraceViewProps {
  sessionId: string;
  onBack?: () => void;
}

export function TraceView({ sessionId, onBack }: TraceViewProps) {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [eventStates, setEventStates] = useState<Map<number, ViewState>>(
    () => new Map(),
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const eventListRef = useRef<HTMLDivElement>(null);
  const filterSearchRef = useRef<HTMLInputElement>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [rawJsonOpenSet, setRawJsonOpenSet] = useState<Set<number>>(
    () => new Set(),
  );

  const [filterState, setFilterState] = useState<FilterState>({
    textSearch: "",
    enabledTypes: new Set<string>(),
    spanId: "",
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    return localStorage.getItem("sidebarCollapsed") === "true";
  });

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      localStorage.setItem("sidebarCollapsed", String(!prev));
      return !prev;
    });
  }, []);

  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [formSpanId, setFormSpanId] = useState<string | null>(null);

  const annotationsBySpan = useMemo(() => {
    const map = new Map<string, Annotation[]>();
    for (const ann of annotations) {
      const key = ann.span_id || "";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(ann);
    }
    return map;
  }, [annotations]);

  const loadAnnotations = useCallback(async () => {
    if (!sessionId) return;
    try {
      const anns = await fetchAnnotations(sessionId);
      setAnnotations(anns);
    } catch {
      // non-critical
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) {
      setError("No session_id provided");
      setLoading(false);
      return;
    }

    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      setEventStates(new Map());
      setSelectedIndex(null);
      try {
        const data = await fetchTrace(sessionId);
        if (cancelled) return;
        const converted = convertOtlpSpansToEvents(data.events);
        setEvents(converted);
        const types = new Set(converted.map((e) => e.type));
        setFilterState((prev) => ({ ...prev, enabledTypes: types }));
      } catch (err) {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "Failed to load trace");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (!loading && events.length > 0) {
      loadAnnotations();
    }
  }, [loading, events.length, loadAnnotations]);

  const filteredEvents = useMemo(
    () => applyFilters(events, filterState),
    [events, filterState],
  );

  const getViewState = useCallback(
    (index: number): ViewState => {
      return (
        eventStates.get(index) ?? getDefaultState(filteredEvents[index]?.type)
      );
    },
    [eventStates, filteredEvents],
  );

  const setViewState = useCallback((index: number, state: ViewState) => {
    setEventStates((prev) => {
      const next = new Map(prev);
      next.set(index, state);
      return next;
    });
  }, []);

  const handleEventClick = useCallback(
    (index: number) => {
      setSelectedIndex(index);
    },
    [],
  );

  const handleQuickFeedback = useCallback(
    async (spanId: string, label: "positive" | "negative") => {
      const existing = annotations.find(
        (a) =>
          a.span_id === spanId && a.name === "feedback" && a.label === label,
      );
      if (existing) {
        await deleteAnnotation(existing.id);
      } else {
        await createAnnotation({
          session_id: sessionId,
          span_id: spanId,
          name: "feedback",
          label,
          source: "human",
        });
      }
      loadAnnotations();
    },
    [sessionId, annotations, loadAnnotations],
  );

  const handleOpenAnnotationForm = useCallback((spanId: string) => {
    setFormSpanId(spanId);
  }, []);

  const handleFilterSearch = useCallback(() => {
    const input = filterSearchRef.current;
    if (input) {
      input.focus();
    } else {
      if (sidebarCollapsed) toggleSidebar();
    }
  }, [sidebarCollapsed, toggleSidebar]);

  const handleSetAllViewState = useCallback((state: ViewState) => {
    const count = filteredEvents.length;
    setEventStates((prev) => {
      const next = new Map(prev);
      for (let i = 0; i < count; i++) {
        next.set(i, state);
      }
      return next;
    });
  }, [filteredEvents]);

  useKeyboardNav({
    getItemCount: () => filteredEvents.length,
    getSelectedIndex: () => selectedIndex,
    setSelectedIndex: (i) => {
      setSelectedIndex(i);
    },
    getViewState: (i) => getViewState(i),
    onExpand: (i) => {
      const current = getViewState(i);
      if (current === "collapsed") setViewState(i, "concise");
      else if (current === "concise") setViewState(i, "expanded");
    },
    onCollapse: (i) => {
      const current = getViewState(i);
      if (current === "expanded") setViewState(i, "concise");
      else if (current === "concise") setViewState(i, "collapsed");
    },
    onBack: onBack,
    onShowHelp: () => setShowHelp((v) => !v),
    onSearch: handleFilterSearch,
    onAnnotate: (i) => {
      const spanId = filteredEvents[i]?.ids?.span_id;
      if (spanId) setFormSpanId(spanId);
    },
    onPositiveFeedback: (i) => {
      const spanId = filteredEvents[i]?.ids?.span_id;
      if (spanId) handleQuickFeedback(spanId, "positive");
    },
    onNegativeFeedback: (i) => {
      const spanId = filteredEvents[i]?.ids?.span_id;
      if (spanId) handleQuickFeedback(spanId, "negative");
    },
    onToggleRawJson: (i) => {
      if (getViewState(i) !== "expanded") {
        setViewState(i, "expanded");
        setRawJsonOpenSet((prev) => new Set(prev).add(i));
      } else {
        setRawJsonOpenSet((prev) => {
          const next = new Set(prev);
          if (next.has(i)) next.delete(i);
          else next.add(i);
          return next;
        });
      }
    },
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading trace...
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/30 border border-red-800 rounded p-4 text-red-300">
        {error}
      </div>
    );
  }

  // Drives the "N of M events" counter.
  const filterActive =
    filterState.textSearch !== "" ||
    filterState.spanId !== "" ||
    filterState.enabledTypes.size < countUniqueTypes(events);

  return (
    <PlaygroundProvider sessionId={sessionId}>
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm text-gray-500">
          {filterActive
            ? `${filteredEvents.length} of ${events.length} events`
            : `${events.length} event${events.length !== 1 ? "s" : ""}`}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <a
            href={`/api/trace/export?session_id=${encodeURIComponent(sessionId)}`}
            download
            className="text-xs text-gray-600 hover:text-gray-400"
            title="Export trace (.jsonl)"
          >
            Export
          </a>
          <button
            onClick={() => setShowHelp(true)}
            className="text-xs text-gray-600 hover:text-gray-400"
            title="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>

      <div className="flex gap-4">
        <FilterSidebar
          events={events}
          filterState={filterState}
          onChange={setFilterState}
          collapsed={sidebarCollapsed}
          onToggleCollapsed={toggleSidebar}
          searchInputRef={filterSearchRef}
          onSetAllViewState={handleSetAllViewState}
        />

        <div
          ref={eventListRef}
          className="flex-1 min-w-0 bg-gray-900 rounded-lg border border-gray-800"
        >
          <EventList
            events={filteredEvents}
            eventStates={eventStates}
            defaultState="concise"
            selectedIndex={selectedIndex}
            searchQuery={filterState.textSearch}
            rawJsonOpenSet={rawJsonOpenSet}
            annotationsBySpan={annotationsBySpan}
            sessionId={sessionId}
            onSelect={handleEventClick}
            onViewStateChange={setViewState}
            onQuickFeedback={handleQuickFeedback}
            onOpenAnnotationForm={handleOpenAnnotationForm}
          />
        </div>
      </div>

      {formSpanId !== null && (
        <AnnotationForm
          sessionId={sessionId}
          spanId={formSpanId}
          existing={annotationsBySpan.get(formSpanId) || []}
          onClose={() => setFormSpanId(null)}
          onSaved={loadAnnotations}
        />
      )}

      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </PlaygroundProvider>
  );
}
