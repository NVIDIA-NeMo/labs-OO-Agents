import { useCallback, useMemo } from 'react';
import type { TraceEvent, ViewState } from '@/api/types';
import type { Annotation } from '@/api/annotations';
import { EventItem } from './EventItem';

const COLLAPSED_BY_DEFAULT = ['span.context_snapshot'];

interface EventListProps {
  events: TraceEvent[];
  eventStates: Map<number, ViewState>;
  defaultState: ViewState;
  selectedIndex: number | null;
  searchQuery?: string;
  rawJsonOpenSet?: Set<number>;
  annotationsBySpan?: Map<string, Annotation[]>;
  sessionId?: string;
  onSelect: (index: number) => void;
  onViewStateChange: (index: number, state: ViewState) => void;
  onQuickFeedback?: (spanId: string, label: 'positive' | 'negative') => void;
  onOpenAnnotationForm?: (spanId: string) => void;
}

export function EventList({
  events,
  eventStates,
  defaultState,
  selectedIndex,
  searchQuery,
  rawJsonOpenSet,
  annotationsBySpan,
  sessionId,
  onSelect,
  onViewStateChange,
  onQuickFeedback,
  onOpenAnnotationForm,
}: EventListProps) {
  const getState = useCallback(
    (index: number): ViewState => {
      const explicit = eventStates.get(index);
      if (explicit !== undefined) return explicit;

      const eventType = events[index]?.type;
      if (eventType && COLLAPSED_BY_DEFAULT.some((p) => eventType.startsWith(p))) {
        return 'collapsed';
      }
      return defaultState;
    },
    [eventStates, defaultState, events],
  );

  // Build depth map based on parent span relationships
  const depthMap = useMemo(() => {
    const map = new Map<number, number>();
    const spanDepths = new Map<string, number>();

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const spanId = ev.ids?.span_id;
      const parentId = ev.ids?.parent_span_id || ev._parent_span_id;

      if (ev._is_span_event && ev._parent_span_id) {
        const parentDepth = spanDepths.get(ev._parent_span_id) ?? 0;
        map.set(i, parentDepth + 1);
        continue;
      }

      if (!parentId) {
        map.set(i, 0);
        if (spanId) spanDepths.set(spanId, 0);
      } else {
        const parentDepth = spanDepths.get(parentId) ?? 0;
        const depth = parentDepth + 1;
        map.set(i, depth);
        if (spanId) spanDepths.set(spanId, depth);
      }
    }
    return map;
  }, [events]);

  if (events.length === 0) {
    return <div className="text-center py-12 text-gray-500">No events to display</div>;
  }

  return (
    <div className="divide-y divide-gray-800/30">
      {events.map((event, index) => (
        <EventItem
          key={event.ids?.span_id ? `${event.ids.span_id}-${index}` : index}
          event={event}
          index={index}
          viewState={getState(index)}
          depth={depthMap.get(index) ?? 0}
          isSelected={selectedIndex === index}
          searchQuery={searchQuery}
          rawJsonOpen={rawJsonOpenSet?.has(index)}
          annotations={annotationsBySpan?.get(event.ids?.span_id || '')}
          sessionId={sessionId}
          onSelect={onSelect}
          onViewStateChange={onViewStateChange}
          onQuickFeedback={onQuickFeedback}
          onOpenAnnotationForm={onOpenAnnotationForm}
        />
      ))}
    </div>
  );
}
