import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { fetchTraces, deleteTrace } from "@/api/traces";
import { formatRelativeTime } from "@/utils/time";
import { DataTable } from "@/components/DataTable";
import type { DataColumn, SortDir } from "@/components/DataTable";
import { ColumnConfigButton } from "@/components/shared/ColumnConfigButton";
import { KeyboardShortcutsHelp } from "@/components/shared/KeyboardShortcutsHelp";
import { useKeyboardNav } from "@/hooks/useKeyboardNav";
import { useColumnConfig } from "@/hooks/useColumnConfig";
import type { TraceGroup } from "@/api/types";

const PAGE_SIZE = 50;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTimestamp(ts: string): string {
  try {
    const epoch = parseFloat(ts);
    const date = !isNaN(epoch) && epoch > 1e9 ? new Date(epoch * 1000) : new Date(ts);
    return date.toLocaleString();
  } catch {
    return ts;
  }
}

export function TraceList() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const searchInputRef = useRef<HTMLInputElement>(null);

  const page = parseInt(searchParams.get("page") || "1", 10);
  const search = searchParams.get("q") || "";
  const batchId = searchParams.get("batch_id") || "";
  const sortBy = searchParams.get("sort") || null;
  const sortDir = (searchParams.get("dir") || "desc") as SortDir;

  const [traces, setTraces] = useState<TraceGroup[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);

  const loadTraces = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchTraces({
        page,
        limit: PAGE_SIZE,
        search: search || undefined,
        batch_id: batchId || undefined,
        sort_by: sortBy || undefined,
        sort_dir: sortBy ? sortDir : undefined,
      });
      setTraces(data.traces);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to load traces:", err);
    } finally {
      setLoading(false);
    }
  }, [page, search, batchId, sortBy, sortDir]);

  useEffect(() => {
    loadTraces();
  }, [loadTraces]);

  const handlePageChange = useCallback(
    (newPage: number) => {
      const params = new URLSearchParams(searchParams);
      params.set("page", String(newPage));
      setSearchParams(params, { replace: true });
      setSelectedIndex(null);
    },
    [searchParams, setSearchParams],
  );

  const setSearch = useCallback(
    (val: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("page", "1");
        if (val) next.set("q", val);
        else next.delete("q");
        return next;
      }, { replace: true });
      setSelectedIndex(null);
    },
    [setSearchParams],
  );

  const clearFilters = useCallback(() => {
    setSearchParams(new URLSearchParams(), { replace: true });
  }, [setSearchParams]);

  const handleTraceClick = useCallback(
    (trace: TraceGroup) => {
      navigate(`/traces/view?session_id=${encodeURIComponent(trace.id)}`);
    },
    [navigate],
  );

  const handleCheckChange = useCallback((key: string, checked: boolean) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(key);
      else next.delete(key);
      return next;
    });
  }, []);

  const handleCheckAll = useCallback(
    (checked: boolean) => {
      if (checked) {
        setCheckedIds(new Set(traces.map((t) => t.id)));
      } else {
        setCheckedIds(new Set());
      }
    },
    [traces],
  );

  const handleBatchDelete = useCallback(async () => {
    if (checkedIds.size === 0) return;
    const count = checkedIds.size;
    if (!confirm(`Delete ${count} trace${count !== 1 ? "s" : ""}?`)) return;
    setDeleting(true);
    try {
      await Promise.all([...checkedIds].map((id) => deleteTrace(id)));
      setCheckedIds(new Set());
      loadTraces();
    } catch (err) {
      console.error("Failed to delete traces:", err);
    } finally {
      setDeleting(false);
    }
  }, [checkedIds, loadTraces]);

  const hasActiveFilters = search !== "";

  const traceColumns: DataColumn<TraceGroup>[] = [
    {
      key: "name",
      label: "Name",
      className: "max-w-xs truncate",
      configurable: false,
      header: (
        <div className="flex items-center gap-1">
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="text-gray-500 hover:text-gray-200 text-xs"
              title="Clear all filters"
            >
              x
            </button>
          )}
          <input
            ref={searchInputRef}
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search traces..."
            className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500 w-40"
          />
        </div>
      ),
      render: (t) => (
        <span className="font-mono text-sm" title={t.name}>
          {t.name}
        </span>
      ),
    },
    {
      key: "event_count",
      label: "Spans",
      className: "w-20 text-right",
      render: (t) => t.event_count ?? "\u2014",
    },
    {
      key: "size",
      label: "Size",
      className: "w-24 text-right",
      render: (t) => formatSize(t.size),
    },
    {
      key: "batch_id",
      label: "Batch",
      className: "w-40 truncate",
      render: (t) => (
        <span className="text-gray-400 font-mono text-xs" title={t.batch_id || ""}>
          {t.batch_id || "\u2014"}
        </span>
      ),
    },
    {
      key: "modified",
      label: "Modified",
      className: "w-48",
      render: (t) => (
        <span className="text-gray-400" title={formatTimestamp(t.modified)}>
          {formatRelativeTime(t.modified)}
        </span>
      ),
    },
  ];

  const handleSort = useCallback(
    (key: string | null, dir: SortDir) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("page", "1");
        if (key) {
          next.set("sort", key);
          next.set("dir", dir);
        } else {
          next.delete("sort");
          next.delete("dir");
        }
        return next;
      }, { replace: true });
      setSelectedIndex(null);
    },
    [setSearchParams],
  );

  const columnConfig = useColumnConfig("traces-table", traceColumns);

  useKeyboardNav({
    getItemCount: () => traces.length,
    getSelectedIndex: () => selectedIndex,
    setSelectedIndex,
    onActivate: (i) => {
      if (traces[i]) handleTraceClick(traces[i]);
    },
    onSearch: () => searchInputRef.current?.focus(),
    onShowHelp: () => setShowHelp((v) => !v),
  });

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-gray-100">Traces</h1>
        <div className="flex items-center gap-2">
          {checkedIds.size > 0 && (
            <button
              onClick={handleBatchDelete}
              disabled={deleting}
              className="px-3 py-1.5 text-sm bg-red-900/60 text-red-200 rounded hover:bg-red-800/60 transition-colors disabled:opacity-50"
            >
              {deleting ? "Deleting..." : `Delete ${checkedIds.size}`}
            </button>
          )}
          <ColumnConfigButton config={columnConfig} />
          <button
            onClick={() => setShowHelp(true)}
            className="text-xs text-gray-600 hover:text-gray-400 px-1"
            title="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <DataTable<TraceGroup>
          data={traces}
          columns={columnConfig.visibleColumns}
          getKey={(t) => t.id}
          onRowClick={handleTraceClick}
          total={total}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={handlePageChange}
          loading={loading}
          selectedIndex={selectedIndex}
          emptyMessage="No traces found"
          sortKey={sortBy}
          sortDir={sortDir}
          onSort={handleSort}
          checkedKeys={checkedIds}
          onCheckChange={handleCheckChange}
          onCheckAll={handleCheckAll}
        />
      </div>

      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
}
