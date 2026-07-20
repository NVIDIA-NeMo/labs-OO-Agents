import { useState, useEffect, useMemo, useCallback } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router";
import { fetchExperimentDetail } from "@/api/eval";
import type { TestResult } from "@/api/eval";
import { TraceView } from "@/components/trace/TraceView";
import { CopyButton } from "@/components/shared/CopyButton";
import {
  readFiltersFromParams,
  buildFilterParams,
  applyEvalFilters,
} from "@/utils/evalFilters";

export function EvalTraceDetail() {
  const { id, traceId } = useParams<{ id: string; traceId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const experimentId = decodeURIComponent(id || "");
  const sessionId = decodeURIComponent(traceId || "");

  const [allTests, setAllTests] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(true);

  const filters = useMemo(() => readFiltersFromParams(searchParams), [searchParams]);
  const filterStr = useMemo(() => buildFilterParams(filters), [filters]);

  useEffect(() => {
    if (!experimentId) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const detail = await fetchExperimentDetail(experimentId);
        if (!cancelled) setAllTests(detail.results);
      } catch {
        // non-critical — prev/next won't work but trace still shows
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [experimentId]);

  const filteredTests = useMemo(
    () => applyEvalFilters(allTests, filters),
    [allTests, filters],
  );

  const currentIndex = useMemo(
    () => filteredTests.findIndex((t) => t.session_id === sessionId),
    [filteredTests, sessionId],
  );

  const currentTest = currentIndex >= 0 ? filteredTests[currentIndex] : null;
  const prevTest = currentIndex > 0 ? filteredTests[currentIndex - 1] : null;
  const nextTest =
    currentIndex >= 0 && currentIndex < filteredTests.length - 1
      ? filteredTests[currentIndex + 1]
      : null;

  const goToTrace = useCallback(
    (test: TestResult) => {
      navigate(
        `/eval/experiment/${encodeURIComponent(experimentId)}/trace/${encodeURIComponent(test.session_id)}${filterStr}`,
        { replace: true },
      );
    },
    [navigate, experimentId, filterStr],
  );

  const goBack = useCallback(() => {
    navigate(-1);
  }, [navigate]);

  const statusLabel = currentTest
    ? currentTest.error
      ? "ERROR"
      : currentTest.passed
        ? "PASS"
        : "FAIL"
    : null;

  const statusCls = statusLabel === "PASS"
    ? "bg-green-900 text-green-200"
    : statusLabel === "ERROR"
      ? "bg-orange-900 text-orange-200"
      : "bg-red-900 text-red-200";

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center gap-3 mb-4 min-w-0">
        <button
          onClick={goBack}
          className="shrink-0 text-gray-400 hover:text-gray-200 transition-colors text-sm whitespace-nowrap"
        >
          &#9666; Back
        </button>

        <h1
          className="text-lg font-mono text-gray-200 truncate min-w-0"
          title={sessionId}
        >
          {sessionId}
        </h1>

        {statusLabel && (
          <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-semibold ${statusCls}`}>
            {statusLabel}
          </span>
        )}

        {sessionId && (
          <CopyButton
            text={[
              `# one-time setup (if necessary): uv run trace-explorer --install-skill`,
              `uv run trace-explorer --viewer ${window.location.origin} --session-id '${sessionId}'`,
            ].join("\n")}
            label="DEBUG"
            title="Copy a prompt to debug this trace with Claude Code, Cursor or other coding agents"
            className="shrink-0 !px-1.5 !py-0.5 !text-[9px] leading-none font-medium uppercase tracking-wide !rounded border border-gray-700 !bg-gray-900 !text-gray-400 hover:!text-gray-200 hover:!bg-gray-800"
          />
        )}

        {!loading && filteredTests.length > 0 && (
          <div className="shrink-0 flex items-center gap-1 ml-auto whitespace-nowrap">
            <button
              onClick={() => prevTest && goToTrace(prevTest)}
              disabled={!prevTest}
              className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-default"
              title="Previous test"
            >
              ▲
            </button>
            <span className="text-xs text-gray-500 tabular-nums">
              {currentIndex >= 0 ? currentIndex + 1 : "?"} / {filteredTests.length}
            </span>
            <button
              onClick={() => nextTest && goToTrace(nextTest)}
              disabled={!nextTest}
              className="pl-2 py-1 text-xs text-gray-400 hover:text-gray-200 disabled:opacity-30 disabled:cursor-default"
              title="Next test"
            >
              ▼
            </button>
          </div>
        )}
      </div>

      <TraceView sessionId={sessionId} onBack={goBack} />
    </div>
  );
}
