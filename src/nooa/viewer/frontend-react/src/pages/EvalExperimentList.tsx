import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { fetchExperiments } from '@/api/eval';
import { useLiveUpdater } from '@/hooks/useLiveUpdater';
import { useKeyboardNav } from '@/hooks/useKeyboardNav';
import { KeyboardShortcutsHelp } from '@/components/shared/KeyboardShortcutsHelp';
import { formatRelativeTime } from '@/utils/time';
import type { ExperimentSummaryItem } from '@/api/types';

const PAGE_SIZE = 50;

export function EvalExperimentList() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const page = parseInt(searchParams.get('page') || '1', 10);
  const search = searchParams.get('q') || '';

  const searchInputRef = useRef<HTMLInputElement>(null);

  const [experiments, setExperiments] = useState<ExperimentSummaryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchInput, setSearchInput] = useState(search);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  const loadExperiments = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchExperiments({
        page,
        limit: PAGE_SIZE,
        search: search || undefined,
      });
      setExperiments(data.experiments);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    } finally {
      setLoading(false);
    }
  }, [page, search]);

  useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  const hasRunning = useMemo(() => experiments.some((e) => e.status === 'running'), [experiments]);

  useLiveUpdater(hasRunning, loadExperiments);

  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const params = new URLSearchParams();
      if (searchInput) params.set('q', searchInput);
      params.set('page', '1');
      setSearchParams(params, { replace: true });
    },
    [searchInput, setSearchParams],
  );

  const handlePageChange = useCallback(
    (newPage: number) => {
      const params = new URLSearchParams(searchParams);
      params.set('page', String(newPage));
      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  useKeyboardNav({
    getItemCount: () => experiments.length,
    getSelectedIndex: () => selectedIndex,
    setSelectedIndex,
    onActivate: (i) => {
      if (experiments[i]) {
        navigate(`/eval/experiment/${encodeURIComponent(experiments[i].id)}`);
      }
    },
    onSearch: () => searchInputRef.current?.focus(),
    onShowHelp: () => setShowHelp((v) => !v),
  });

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-gray-100">Evaluations</h1>
        <div className="flex items-center gap-2">
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              ref={searchInputRef}
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="Search experiments..."
              className="px-3 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500 w-64"
            />
            <button
              type="submit"
              className="px-3 py-1.5 text-sm bg-gray-700 text-gray-200 rounded hover:bg-gray-600 transition-colors"
            >
              Search
            </button>
          </form>
          <button
            onClick={() => setShowHelp(true)}
            className="text-xs text-gray-600 hover:text-gray-400 px-1"
            title="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {loading && experiments.length === 0 ? (
          <div className="text-center py-12 text-gray-500">Loading...</div>
        ) : experiments.length === 0 ? (
          <div className="text-center py-12 text-gray-500">No experiments found</div>
        ) : (
          experiments.map((exp, idx) => {
            const passRate =
              exp.test_count > 0 ? ((exp.passed_count / exp.test_count) * 100).toFixed(0) : '0';
            return (
              <div
                key={exp.id}
                data-list-index={idx}
                onClick={() => navigate(`/eval/experiment/${encodeURIComponent(exp.id)}`)}
                className={`bg-gray-900 border rounded-lg p-4 cursor-pointer hover:bg-gray-800/70 transition-colors ${
                  selectedIndex === idx ? 'border-gray-600 bg-gray-800/80 ring-1 ring-gray-600' : 'border-gray-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="font-mono text-sm text-gray-200 truncate">{exp.id}</div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-400">
                      <span title={new Date(exp.timestamp).toLocaleString()}>
                        {formatRelativeTime(exp.timestamp)}
                      </span>
                      {exp.models.length > 0 && <span>{exp.models.join(', ')}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-4 shrink-0 ml-4">
                    {exp.status === 'running' && (
                      <span className="flex items-center gap-1.5 text-xs text-yellow-400">
                        <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-400" />
                        </span>
                        Running
                      </span>
                    )}
                    <div className="text-right">
                      <div
                        className={`text-lg font-semibold ${
                          parseInt(passRate) === 100
                            ? 'text-green-400'
                            : parseInt(passRate) >= 80
                              ? 'text-yellow-400'
                              : 'text-red-400'
                        }`}
                      >
                        {passRate}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {exp.passed_count}/{exp.test_count} passed
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between mt-4 px-1">
          <span className="text-sm text-gray-400">
            {total} experiment{total !== 1 ? 's' : ''} &middot; Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => handlePageChange(page - 1)}
              disabled={page <= 1}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <button
              onClick={() => handlePageChange(page + 1)}
              disabled={page >= totalPages}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </div>
  );
}
