import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router";
import {
  fetchExperimentDetail,
  fetchExperimentSummary,
  fetchExperimentMetrics,
} from "@/api/eval";
import type { HistoricalExperiment } from "@/api/eval";
import { deleteTrace } from "@/api/traces";
import type {
  ExperimentDetail,
  ExperimentSummary,
  TestResult,
  ColumnInfo,
} from "@/api/eval";
import { DataTable } from "@/components/DataTable";
import type { DataColumn, SortDir } from "@/components/DataTable";
import { ColumnConfigButton } from "@/components/shared/ColumnConfigButton";
import { CopyButton } from "@/components/shared/CopyButton";
import { KeyboardShortcutsHelp } from "@/components/shared/KeyboardShortcutsHelp";
import { useColumnConfig } from "@/hooks/useColumnConfig";
import { useKeyboardNav } from "@/hooks/useKeyboardNav";
import { buildFilterParams } from "@/utils/evalFilters";
import { formatDurationMs } from "@/utils/time";

const PAGE_SIZE = 50;

function scoreColor(rate: number): string {
  if (rate >= 80) return "text-green-400";
  if (rate >= 60) return "text-yellow-400";
  return "text-red-400";
}

function matrixCellColor(rate: number): string {
  if (rate >= 80) return "text-green-400";
  if (rate >= 60) return "text-yellow-400";
  return "text-red-400";
}

function shortName(name: string): string {
  return name.includes("/") ? name.split("/").pop()! : name;
}

function columnLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const FILTER_LABELS: Record<string, string> = {
  model: "All Models",
  tier: "All Tiers",
  variant: "All Variants",
  passed: "All Status",
};

export function EvalExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const experimentId = decodeURIComponent(id || "");

  const page = parseInt(searchParams.get("page") || "1", 10);
  const sortBy = searchParams.get("sort") || null;
  const sortDir = (searchParams.get("dir") || "desc") as SortDir;
  const filterKeyword = searchParams.get("q") || "";

  const metaFilters = useMemo(() => {
    const meta: Record<string, string> = {};
    const skip = new Set(["page", "sort", "dir", "q"]);
    for (const [key, val] of searchParams.entries()) {
      if (!skip.has(key) && val) meta[key] = val;
    }
    return meta;
  }, [searchParams]);

  const searchInputRef = useRef<HTMLInputElement>(null);

  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [summary, setSummary] = useState<ExperimentSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  const loadData = useCallback(async () => {
    if (!experimentId) return;
    setLoading(true);
    setError(null);
    try {
      const [d, s] = await Promise.all([
        fetchExperimentDetail(experimentId, {
          page,
          limit: PAGE_SIZE,
          sort_by: sortBy || undefined,
          sort_dir: sortBy ? sortDir : undefined,
          search: filterKeyword || undefined,
          meta: Object.keys(metaFilters).length > 0 ? metaFilters : undefined,
        }),
        fetchExperimentSummary(experimentId, {
          search: filterKeyword || undefined,
          meta: Object.keys(metaFilters).length > 0 ? metaFilters : undefined,
        }),
      ]);
      setDetail(d);
      setSummary(s);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [experimentId, page, sortBy, sortDir, filterKeyword, metaFilters]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const columns = useMemo(() => {
    const cols = detail?.columns ?? [];
    const first = cols.filter((c) => c.key === "passed");
    const rest = cols.filter((c) => c.key !== "passed");
    return [...first, ...rest];
  }, [detail]);

  const hasActiveFilters =
    filterKeyword !== "" ||
    Object.values(metaFilters).some((v) => v !== "");

  const handlePageChange = useCallback(
    (newPage: number) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("page", String(newPage));
        return next;
      }, { replace: true });
    },
    [setSearchParams],
  );

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
    },
    [setSearchParams],
  );

  const setFilterKeyword = useCallback(
    (val: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("page", "1");
        if (val) next.set("q", val);
        else next.delete("q");
        return next;
      }, { replace: true });
    },
    [setSearchParams],
  );

  const setMetaFilter = useCallback(
    (key: string, val: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("page", "1");
        if (val) next.set(key, val);
        else next.delete(key);
        return next;
      }, { replace: true });
    },
    [setSearchParams],
  );

  const clearFilters = useCallback(() => {
    setSearchParams(new URLSearchParams(), { replace: true });
  }, [setSearchParams]);

  const currentFilters = useMemo(
    () => ({ keyword: filterKeyword, meta: metaFilters }),
    [filterKeyword, metaFilters],
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
      if (checked && detail) {
        setCheckedIds(new Set(detail.results.map((t) => t.session_id)));
      } else {
        setCheckedIds(new Set());
      }
    },
    [detail],
  );

  const handleBatchDelete = useCallback(async () => {
    if (checkedIds.size === 0) return;
    const count = checkedIds.size;
    if (!confirm(`Delete ${count} trace${count !== 1 ? "s" : ""}?`)) return;
    setDeleting(true);
    try {
      await Promise.all([...checkedIds].map((id) => deleteTrace(id)));
      setCheckedIds(new Set());
      loadData();
    } catch (err) {
      console.error("Failed to delete traces:", err);
    } finally {
      setDeleting(false);
    }
  }, [checkedIds, loadData]);

  const handleRowClick = useCallback(
    (test: TestResult) => {
      const filterStr = buildFilterParams(currentFilters);
      navigate(
        `/eval/experiment/${encodeURIComponent(experimentId)}/trace/${encodeURIComponent(test.session_id)}${filterStr}`,
      );
    },
    [navigate, experimentId, currentFilters],
  );

  const tableColumns = useMemo((): DataColumn<TestResult>[] => {
    const cols: DataColumn<TestResult>[] = [
      {
        key: "display_name",
        label: "Test",
        className: "truncate max-w-md",
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
              value={filterKeyword}
              onChange={(e) => setFilterKeyword(e.target.value)}
              placeholder="Search tests..."
              className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500 w-40"
            />
          </div>
        ),
        render: (t) => (
          <span className="text-gray-200" title={t.display_name}>
            {t.display_name}
          </span>
        ),
      },
    ];

    for (const col of columns) {
      const isNumeric = col.values.length > 0 && col.values.every((v) => !isNaN(Number(v)));
      const filterable = col.values.length <= 30;
      const allLabel = FILTER_LABELS[col.key] ?? `All ${columnLabel(col.key)}`;

      let header: React.ReactNode | undefined;
      if (filterable) {
        header = (
          <select
            value={metaFilters[col.key] || ""}
            onChange={(e) => setMetaFilter(col.key, e.target.value)}
            className="px-1 py-0.5 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none max-w-[13rem]"
          >
            <option value="">{allLabel}</option>
            {col.values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        );
      } else if (!isNumeric) {
        const raw = metaFilters[col.key] || "";
        const display = raw.startsWith("~") ? raw.slice(1) : raw;
        header = (
          <input
            type="search"
            value={display}
            onChange={(e) => {
              const v = e.target.value;
              setMetaFilter(col.key, v ? `~${v}` : "");
            }}
            placeholder={`Search ${columnLabel(col.key).toLowerCase()}...`}
            className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500 w-28"
          />
        );
      }

      cols.push({
        key: col.key,
        label: columnLabel(col.key),
        className: "text-xs text-gray-400",
        header,
        render: (t) => renderCellValue(t, col),
      });
    }

    const passedIdx = cols.findIndex((c) => c.key === "passed");
    const insertAt = passedIdx === -1 ? cols.length : passedIdx + 1;
    cols.splice(
      insertAt,
      0,
      {
        key: "duration_ms",
        label: "Duration",
        className: "w-24 text-right font-mono text-xs text-gray-400 tabular-nums",
        render: (t) =>
          typeof t.duration_ms === "number" && Number.isFinite(t.duration_ms)
            ? formatDurationMs(t.duration_ms)
            : "—",
      },
      {
        key: "span_count",
        label: "Spans",
        className: "w-20 text-right font-mono text-xs text-gray-400 tabular-nums",
        render: (t) =>
          typeof t.span_count === "number" && Number.isFinite(t.span_count)
            ? Math.round(t.span_count).toLocaleString()
            : "—",
      },
    );

    return cols;
  }, [
    columns,
    filterKeyword,
    metaFilters,
    hasActiveFilters,
    clearFilters,
    setMetaFilter,
    setFilterKeyword,
  ]);

  const columnConfig = useColumnConfig("eval-experiment-table", tableColumns);

  useKeyboardNav({
    getItemCount: () => detail?.results.length ?? 0,
    getSelectedIndex: () => selectedIndex,
    setSelectedIndex,
    onActivate: (i) => {
      if (detail?.results[i]) handleRowClick(detail.results[i]);
    },
    onBack: () => navigate("/eval"),
    onSearch: () => searchInputRef.current?.focus(),
    onShowHelp: () => setShowHelp((v) => !v),
  });

  if (loading && !detail) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Loading experiment...
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-[100rem] mx-auto px-4 py-6">
        <div className="bg-red-900/30 border border-red-800 rounded p-4 text-red-300">
          {error}
        </div>
      </div>
    );
  }

  if (!detail || !summary) return null;

  const overall = summary.overall;
  const passRate =
    overall.total > 0
      ? ((overall.passed / overall.total) * 100).toFixed(0)
      : "0";
  const matrix = summary.matrix;

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => navigate(-1)}
          className="text-gray-400 hover:text-gray-200 transition-colors text-sm"
        >
          &#9666; Back
        </button>
        <h1
          className="text-lg font-mono text-gray-200 truncate max-w-2xl"
          title={experimentId}
        >
          {experimentId}
        </h1>
        <CopyButton
          text={[
            `# one-time setup (if necessary): uv run trace-explorer --install-skill`,
            `uv run trace-explorer --viewer ${window.location.origin} --experiment '${experimentId}'`,
          ].join("\n")}
          label="DEBUG"
          title="Copy a prompt to debug this experiment with Claude Code, Cursor or other coding agents"
          className="!px-1.5 !py-0.5 !text-[9px] leading-none font-medium uppercase tracking-wide !rounded border border-gray-700 !bg-gray-900 !text-gray-400 hover:!text-gray-200 hover:!bg-gray-800"
        />
      </div>

      <SummaryStats overall={overall} passRate={passRate} />

      <TierBreakdown byTier={summary.by_tier} />

      {matrix.models.length >= 2 && matrix.test_types.length >= 2 && (
        <ResultsMatrix matrix={matrix} />
      )}

      <div className="flex items-center gap-2 mt-6 mb-3">
        <span className="text-sm text-gray-400">
          {detail.total} test{detail.total !== 1 ? "s" : ""}
          {hasActiveFilters && ` (filtered)`}
        </span>
        <div className="flex items-center gap-2 ml-auto">
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

      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-x-auto">
        <DataTable<TestResult>
          data={detail.results}
          columns={columnConfig.visibleColumns}
          getKey={(t) => t.session_id}
          onRowClick={handleRowClick}
          total={detail.total}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={handlePageChange}
          loading={loading}
          selectedIndex={selectedIndex}
          sortKey={sortBy}
          sortDir={sortDir}
          onSort={handleSort}
          emptyMessage="No tests match filters"
          checkedKeys={checkedIds}
          onCheckChange={handleCheckChange}
          onCheckAll={handleCheckAll}
        />
      </div>

      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
}

function SummaryStats({
  overall,
  passRate,
}: {
  overall: ExperimentSummary["overall"];
  passRate: string;
}) {
  return (
    <div className="flex items-center gap-6 mb-4 p-4 bg-gray-900 border border-gray-800 rounded-lg">
      <div>
        <div className={`text-2xl font-bold ${scoreColor(parseInt(passRate))}`}>
          {passRate}%
        </div>
        <div className="text-xs text-gray-500">Pass Rate</div>
      </div>
      <div>
        <div className="text-2xl font-bold text-gray-200">
          {overall.avg_score?.toFixed(2) || "0.00"}
        </div>
        <div className="text-xs text-gray-500">Avg Score</div>
      </div>
      <div>
        <div className="text-2xl font-bold text-gray-200">{overall.total}</div>
        <div className="text-xs text-gray-500">Total Tests</div>
      </div>
      <div>
        <div className="text-lg font-semibold text-green-400">
          {overall.passed}
        </div>
        <div className="text-xs text-gray-500">Passed</div>
      </div>
      <div>
        <div className="text-lg font-semibold text-red-400">
          {overall.total - overall.passed}
        </div>
        <div className="text-xs text-gray-500">Failed</div>
      </div>
    </div>
  );
}

function ResultsMatrix({ matrix }: { matrix: ExperimentSummary["matrix"] }) {
  return (
    <details className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <summary className="px-4 py-2 text-sm text-gray-400 cursor-pointer hover:text-gray-200">
        Results Matrix
      </summary>
      <div className="overflow-x-auto px-4 pb-4">
        <table className="text-xs font-mono">
          <thead>
            <tr>
              <th className="text-left px-2 py-1 text-gray-500" />
              {matrix.models.map((m) => (
                <th
                  key={m}
                  className="text-center px-3 py-1 text-gray-400"
                  title={m}
                >
                  {shortName(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.test_types.map((tt) => (
              <tr key={tt}>
                <th className="text-left px-2 py-1 text-gray-400 font-normal">
                  {shortName(tt)}
                </th>
                {matrix.models.map((m) => {
                  const cell = matrix.cells[tt]?.[m];
                  if (!cell)
                    return (
                      <td
                        key={m}
                        className="text-center px-3 py-1 text-gray-600"
                      >
                        -
                      </td>
                    );
                  return (
                    <td
                      key={m}
                      className={`text-center px-3 py-1 font-semibold ${matrixCellColor(cell.rate)}`}
                      title={`${cell.passed}/${cell.total}`}
                    >
                      {cell.rate}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

const TIER_ORDER = ["stable", "frontier", "horizon"];
const TIER_COLORS: Record<string, string> = {
  stable: "text-green-400",
  frontier: "text-yellow-400",
  horizon: "text-purple-400",
};

function TierBreakdown({
  byTier,
}: {
  byTier: Record<string, { total: number; passed: number }>;
}) {
  const tiers = TIER_ORDER.filter((t) => byTier[t] && byTier[t].total > 0);
  if (tiers.length === 0) return null;

  const [history, setHistory] = useState<HistoricalExperiment[]>([]);

  useEffect(() => {
    fetchExperimentMetrics(20)
      .then((m) => setHistory(m.history))
      .catch(() => {});
  }, []);

  const historicalAvgs: Record<string, number> = {};
  if (history.length > 0) {
    for (const tier of TIER_ORDER) {
      const rates = history
        .map((h) => h.tier_metrics?.[tier])
        .filter((m) => !!m && m.tests_total > 0)
        .map((m) => m.success_rate);
      if (rates.length > 0) {
        historicalAvgs[tier] = rates.reduce((a, b) => a + b, 0) / rates.length;
      }
    }
  }

  return (
    <details className="mb-4 bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <summary className="px-4 py-2 text-sm text-gray-400 cursor-pointer hover:text-gray-200 flex items-center gap-3">
        <span>Tier Breakdown</span>
        <span className="flex gap-3 text-xs ml-2">
          {tiers.map((tier) => {
            const d = byTier[tier];
            const rate = Math.round((d.passed / d.total) * 100);
            return (
              <span key={tier}>
                <span className={TIER_COLORS[tier] || "text-gray-400"}>{rate}%</span>
                <span className="text-gray-600 ml-1">{tier}</span>
              </span>
            );
          })}
        </span>
      </summary>
      <div className="px-4 pb-4 flex gap-6">
        {tiers.map((tier) => {
          const d = byTier[tier];
          const rate = Math.round((d.passed / d.total) * 100);
          const hist = historicalAvgs[tier];
          return (
            <div key={tier}>
              <div className={`text-2xl font-bold ${TIER_COLORS[tier] || "text-gray-400"}`}>
                {rate}%
              </div>
              <div className="text-xs text-gray-500">
                {tier} ({d.passed}/{d.total})
                {hist != null && (
                  <span className="text-gray-600 ml-1">{hist.toFixed(0)}% avg</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </details>
  );
}

function renderCellValue(
  test: TestResult,
  col: ColumnInfo,
): React.ReactNode {
  const val = test[col.key];
  const str = val != null ? String(val) : "";

  if (col.key === "passed") {
    const hasError = !!test.error;
    const status = hasError ? "error" : test.passed ? "passed" : "failed";
    const cls =
      status === "passed"
        ? "bg-green-900 text-green-200"
        : status === "error"
          ? "bg-orange-900 text-orange-200"
          : "bg-red-900 text-red-200";
    return (
      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${cls}`}>
        {status.toUpperCase()}
      </span>
    );
  }
  if (col.key === "score" || col.key === "weighted_score") {
    const score = col.key === "score"
      ? (test.weighted_score ?? test.score ?? (test.passed ? 1 : 0))
      : val;
    return score != null ? (
      <span className="font-mono text-gray-300">{Number(score).toFixed(2)}</span>
    ) : null;
  }
  if (col.key === "error") {
    return str ? (
      <span className="text-red-400 truncate max-w-xs" title={str}>{str.slice(0, 40)}{str.length > 40 ? "..." : ""}</span>
    ) : null;
  }
  if (col.key === "tier") {
    const cls =
      str === "gold"
        ? "bg-yellow-900 text-yellow-200"
        : str === "silver"
          ? "bg-gray-700 text-gray-200"
          : str === "bronze"
            ? "bg-orange-900 text-orange-200"
            : "bg-gray-800 text-gray-400";
    return str ? (
      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${cls}`}>
        {str}
      </span>
    ) : null;
  }
  if (col.key === "variant") {
    return str ? (
      <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-[10px]">
        {str}
      </span>
    ) : null;
  }
  if (str.length > 60) {
    return <span title={str}>{str.slice(0, 57)}...</span>;
  }
  return str;
}
