import { useState, useCallback, useMemo, type Ref } from "react";
import type { TraceEvent } from "@/api/types";

const SPAN_GROUP_RE = /^(span\.[^.]+)\.(.+)$/;

type TypeEntry = { type: string; count: number };
type HierarchyItem =
  | { kind: "single"; entry: TypeEntry }
  | {
      kind: "group";
      prefix: string;
      children: { type: string; suffix: string; count: number }[];
    };

export interface FilterState {
  textSearch: string;
  enabledTypes: Set<string>;
  spanId: string;
}

interface FilterSidebarProps {
  events: TraceEvent[];
  filterState: FilterState;
  onChange: (state: FilterState) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  searchInputRef?: Ref<HTMLInputElement>;
  onSetAllViewState?: (state: 'collapsed' | 'concise' | 'expanded') => void;
}

function truncateId(id: string): string {
  if (id.length <= 12) return id;
  return id.slice(0, 8) + ".." + id.slice(-4);
}

export function FilterSidebar({
  events,
  filterState,
  onChange,
  collapsed,
  onToggleCollapsed,
  searchInputRef,
  onSetAllViewState,
}: FilterSidebarProps) {
  const [typeSearchInput, setTypeSearchInput] = useState("");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    new Set(),
  );

  const eventTypes = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of events) {
      counts.set(e.type, (counts.get(e.type) || 0) + 1);
    }
    return [...counts.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([type, count]) => ({ type, count }));
  }, [events]);

  const spanIds = useMemo(() => {
    const ids = new Set<string>();
    for (const e of events) {
      if (e.ids?.span_id) ids.add(e.ids.span_id);
    }
    return [...ids].sort();
  }, [events]);

  const filteredTypes = useMemo(
    () =>
      typeSearchInput
        ? eventTypes.filter((t) =>
            t.type.toLowerCase().includes(typeSearchInput.toLowerCase()),
          )
        : eventTypes,
    [eventTypes, typeSearchInput],
  );

  const hierarchy = useMemo(() => {
    const groups = new Map<
      string,
      { type: string; suffix: string; count: number }[]
    >();
    const result: HierarchyItem[] = [];
    const seenGroups = new Set<string>();

    for (const { type, count } of filteredTypes) {
      const match = type.match(SPAN_GROUP_RE);
      if (match) {
        const prefix = match[1];
        if (!groups.has(prefix)) groups.set(prefix, []);
        groups.get(prefix)!.push({ type, suffix: match[2], count });
      }
    }

    for (const { type, count } of filteredTypes) {
      const match = type.match(SPAN_GROUP_RE);
      if (match) {
        const prefix = match[1];
        if (!seenGroups.has(prefix)) {
          seenGroups.add(prefix);
          const children = groups.get(prefix)!;
          if (children.length === 1) {
            result.push({
              kind: "single",
              entry: { type: children[0].type, count: children[0].count },
            });
          } else {
            result.push({ kind: "group", prefix, children });
          }
        }
      } else {
        result.push({ kind: "single", entry: { type, count } });
      }
    }

    return result;
  }, [filteredTypes]);

  const allEnabled = eventTypes.every((t) =>
    filterState.enabledTypes.has(t.type),
  );

  const handleTextSearch = useCallback(
    (value: string) => {
      onChange({ ...filterState, textSearch: value });
    },
    [filterState, onChange],
  );

  const handleToggleType = useCallback(
    (type: string) => {
      const next = new Set(filterState.enabledTypes);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      onChange({ ...filterState, enabledTypes: next });
    },
    [filterState, onChange],
  );

  const handleToggleGroup = useCallback(
    (children: { type: string }[]) => {
      const next = new Set(filterState.enabledTypes);
      const allChecked = children.every((c) => next.has(c.type));
      for (const c of children) {
        if (allChecked) next.delete(c.type);
        else next.add(c.type);
      }
      onChange({ ...filterState, enabledTypes: next });
    },
    [filterState, onChange],
  );

  const toggleGroupCollapsed = useCallback((prefix: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix);
      else next.add(prefix);
      return next;
    });
  }, []);

  const handleAllTypes = useCallback(() => {
    onChange({
      ...filterState,
      enabledTypes: new Set(eventTypes.map((t) => t.type)),
    });
  }, [filterState, onChange, eventTypes]);

  const handleNoneTypes = useCallback(() => {
    onChange({ ...filterState, enabledTypes: new Set() });
  }, [filterState, onChange]);

  const handleSpanId = useCallback(
    (value: string) => {
      onChange({ ...filterState, spanId: value });
    },
    [filterState, onChange],
  );

  const handleReset = useCallback(() => {
    onChange({
      textSearch: "",
      enabledTypes: new Set(eventTypes.map((t) => t.type)),
      spanId: "",
    });
    setTypeSearchInput("");
  }, [onChange, eventTypes]);

  const hasActiveFilters =
    filterState.textSearch !== "" || filterState.spanId !== "" || !allEnabled;

  if (collapsed) {
    return (
      <div className="shrink-0 w-10">
        <button
          onClick={onToggleCollapsed}
          className="w-10 h-10 flex items-center justify-center text-gray-500 hover:text-gray-300 bg-gray-900 border border-gray-800 rounded-lg"
          title="Show filters"
        >
          <span className="text-xs">&#9654;</span>
        </button>
      </div>
    );
  }

  return (
    <div className="shrink-0 w-64 bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs text-gray-400 font-medium uppercase tracking-wider">
          Filters
        </span>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              onClick={handleReset}
              className="text-[10px] text-gray-500 hover:text-gray-300"
              title="Reset all filters"
            >
              Reset
            </button>
          )}
          <button
            onClick={onToggleCollapsed}
            className="text-gray-500 hover:text-gray-300 text-xs"
            title="Hide filters"
          >
            &#9664;
          </button>
        </div>
      </div>

      {/* View state bulk controls */}
      {onSetAllViewState && (
        <div className="px-3 py-2 border-b border-gray-800 flex items-center gap-1.5">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider mr-auto">View</span>
          <button
            onClick={() => onSetAllViewState('collapsed')}
            className="px-2 py-0.5 text-[10px] text-gray-400 hover:text-gray-200 bg-gray-800 hover:bg-gray-700 rounded border border-gray-700"
          >
            All Collapsed
          </button>
          <button
            onClick={() => onSetAllViewState('concise')}
            className="px-2 py-0.5 text-[10px] text-gray-400 hover:text-gray-200 bg-gray-800 hover:bg-gray-700 rounded border border-gray-700"
          >
            All Concise
          </button>
        </div>
      )}

      <div className="p-3 space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
        {/* Text search */}
        <div>
          <label className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">
            Search
          </label>
          <input
            ref={searchInputRef}
            type="search"
            value={filterState.textSearch}
            onChange={(e) => handleTextSearch(e.target.value)}
            placeholder="Filter events..."
            className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500"
          />
        </div>

        {/* Span ID */}
        {spanIds.length > 1 && (
          <div>
            <label className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">
              Span
            </label>
            <select
              value={filterState.spanId}
              onChange={(e) => handleSpanId(e.target.value)}
              className="w-full px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none"
            >
              <option value="">All spans</option>
              {spanIds.map((id) => (
                <option key={id} value={id} title={id}>
                  {truncateId(id)}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Event types */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-[10px] text-gray-500 uppercase tracking-wider">
              Event Types
            </label>
            <div className="flex gap-1.5">
              <button
                onClick={handleAllTypes}
                className="text-[10px] text-gray-500 hover:text-gray-300"
              >
                All
              </button>
              <button
                onClick={handleNoneTypes}
                className="text-[10px] text-gray-500 hover:text-gray-300"
              >
                None
              </button>
            </div>
          </div>

          {eventTypes.length > 8 && (
            <input
              type="text"
              value={typeSearchInput}
              onChange={(e) => setTypeSearchInput(e.target.value)}
              placeholder="Filter types..."
              className="w-full px-2 py-0.5 mb-1.5 text-[11px] bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500"
            />
          )}

          <div className="space-y-0.5">
            {hierarchy.map((item) => {
              if (item.kind === "single") {
                const { type, count } = item.entry;
                return (
                  <label
                    key={type}
                    className="flex items-center gap-1.5 cursor-pointer group"
                  >
                    <input
                      type="checkbox"
                      checked={filterState.enabledTypes.has(type)}
                      onChange={() => handleToggleType(type)}
                      className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0 focus:ring-offset-0 w-3 h-3"
                    />
                    <span className="text-[11px] text-gray-400 group-hover:text-gray-200 truncate flex-1">
                      {type}
                    </span>
                    <span className="text-[10px] text-gray-600 shrink-0">
                      {count}
                    </span>
                  </label>
                );
              }

              const { prefix, children } = item;
              const allChecked = children.every((c) =>
                filterState.enabledTypes.has(c.type),
              );
              const someChecked = children.some((c) =>
                filterState.enabledTypes.has(c.type),
              );
              const isCollapsed = collapsedGroups.has(prefix);

              return (
                <div key={prefix}>
                  <div className="flex items-center gap-1.5 cursor-pointer group">
                    <input
                      type="checkbox"
                      ref={(el) => {
                        if (el) el.indeterminate = someChecked && !allChecked;
                      }}
                      checked={allChecked}
                      onChange={() => handleToggleGroup(children)}
                      className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0 focus:ring-offset-0 w-3 h-3"
                    />
                    <span
                      onClick={() => toggleGroupCollapsed(prefix)}
                      className="text-[11px] text-gray-400 group-hover:text-gray-200 truncate flex-1"
                    >
                      {prefix}
                    </span>
                    <button
                      onClick={() => toggleGroupCollapsed(prefix)}
                      className="text-[9px] text-gray-500 shrink-0"
                    >
                      {isCollapsed ? "\u25B6" : "\u25BC"}
                    </button>
                  </div>
                  {!isCollapsed && (
                    <div className="ml-[18px]">
                      {children.map(({ type, suffix, count }) => (
                        <label
                          key={type}
                          className="flex items-center gap-1.5 cursor-pointer group"
                        >
                          <input
                            type="checkbox"
                            checked={filterState.enabledTypes.has(type)}
                            onChange={() => handleToggleType(type)}
                            className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0 focus:ring-offset-0 w-3 h-3"
                          />
                          <span
                            className="text-[11px] text-gray-400 group-hover:text-gray-200 truncate flex-1"
                            title={type}
                          >
                            {suffix}
                          </span>
                          <span className="text-[10px] text-gray-600 shrink-0">
                            {count}
                          </span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
