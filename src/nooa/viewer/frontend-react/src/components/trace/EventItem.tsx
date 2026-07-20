import { useCallback, useMemo } from "react";
import type { TraceEvent, ViewState } from "@/api/types";
import type { Annotation } from "@/api/annotations";
import { getPlugin } from "@/components/plugins/registry";
import { DefaultPlugin } from "@/components/plugins/DefaultPlugin";
import { AnnotationIndicator } from "@/components/annotations/AnnotationIndicator";
import { CopyButton } from "@/components/shared/CopyButton";

interface EventItemProps {
  event: TraceEvent;
  index: number;
  viewState: ViewState;
  depth: number;
  isSelected: boolean;
  searchQuery?: string;
  rawJsonOpen?: boolean;
  annotations?: Annotation[];
  sessionId?: string;
  onSelect: (index: number) => void;
  onViewStateChange: (index: number, state: ViewState) => void;
  onQuickFeedback?: (spanId: string, label: "positive" | "negative") => void;
  onOpenAnnotationForm?: (spanId: string) => void;
}

const VIEW_STATES: ViewState[] = ["collapsed", "concise", "expanded"];


export function EventItem({
  event,
  index,
  viewState,
  depth,
  isSelected,
  searchQuery,
  rawJsonOpen,
  annotations,
  sessionId,
  onSelect,
  onViewStateChange,
  onQuickFeedback,
  onOpenAnnotationForm,
}: EventItemProps) {
  const Plugin = getPlugin(event.type) ?? DefaultPlugin;
  const spanId = event.ids?.span_id || "";

  const debugPrompt = useMemo(() => {
    if (!spanId || !sessionId) return "";
    const viewerUrl = window.location.origin;
    return [
      `# one-time setup (if necessary): uv run trace-explorer --install-skill`,
      `uv run trace-explorer --viewer ${viewerUrl} --session-id '${sessionId}' --span-id '${spanId}'`,
    ].join("\n");
  }, [spanId, sessionId]);

  const handleSelect = useCallback(() => {
    onSelect(index);
  }, [index, onSelect]);

  const handleQuickFeedback = useCallback(
    (label: "positive" | "negative") => {
      onQuickFeedback?.(spanId, label);
    },
    [spanId, onQuickFeedback],
  );

  const handleOpenForm = useCallback(() => {
    onOpenAnnotationForm?.(spanId);
  }, [spanId, onOpenAnnotationForm]);

  const hasAnnotations = annotations && annotations.length > 0;
  const viewControls = (
    <div className="inline-flex items-center gap-1">
      <div className="inline-flex rounded border border-gray-700 overflow-hidden">
        {VIEW_STATES.map((state) => (
          <button
            key={state}
            onClick={(e) => {
              e.stopPropagation();
              onViewStateChange(index, state);
            }}
            className={`cursor-pointer px-1.5 py-0.5 text-[9px] leading-none font-medium uppercase tracking-wide transition-colors ${
              viewState === state
                ? "bg-gray-700 text-gray-100"
                : "bg-gray-900 text-gray-400 hover:text-gray-200 hover:bg-gray-800"
            }`}
            aria-label={`Set event ${index} to ${state}`}
          >
            {state}
          </button>
        ))}
      </div>
      {debugPrompt && (
        <CopyButton
          text={debugPrompt}
          label="DEBUG"
          title="Copy a prompt to debug this trace with Claude Code, Cursor or other coding agents"
          className="!px-1.5 !py-0.5 !text-[9px] leading-none font-medium uppercase tracking-wide !rounded border border-gray-700 !bg-gray-900 !text-gray-400 hover:!text-gray-200 hover:!bg-gray-800"
        />
      )}
    </div>
  );

  return (
    <div
      data-event-index={index}
      data-view-state={viewState}
      onClick={handleSelect}
      className={`group/event border-b border-gray-700/50 transition-colors ${
        hasAnnotations ? "border-l-2 border-l-blue-600/40" : ""
      } ${isSelected ? "bg-gray-800/80 ring-1 ring-gray-600" : "hover:bg-gray-800/30"}`}
    >
      <div className="py-2 pr-3" style={{ paddingLeft: `${depth * 16 + 12}px` }}>
        <div className="flex items-start gap-2 min-w-0">
          <div className="flex-1 min-w-0">
            {viewState === "collapsed" ? (
              <div className="flex items-center gap-3 min-w-0">
                <div className="flex-1 min-w-0">
                  <Plugin
                    event={event}
                    viewState={viewState}
                    searchQuery={searchQuery}
                    rawJsonOpen={rawJsonOpen}
                  />
                </div>
                {viewControls}
              </div>
            ) : (
              <Plugin
                event={event}
                viewState={viewState}
                searchQuery={searchQuery}
                rawJsonOpen={rawJsonOpen}
                viewControls={viewControls}
              />
            )}
          </div>
          {spanId && (
            <AnnotationIndicator
              annotations={annotations || []}
              onOpenForm={handleOpenForm}
              onQuickFeedback={handleQuickFeedback}
            />
          )}
        </div>
      </div>
    </div>
  );
}
