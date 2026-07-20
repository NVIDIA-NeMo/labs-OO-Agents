import { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";
import {
  fetchMemoryDbs,
  fetchMemoryExplain,
  fetchMemoryRecords,
  fetchMemoryStats,
} from "@/api/memory";
import type {
  MemoryDbInfo,
  MemoryExplainRow,
  MemoryRecordRow,
  MemoryStats,
} from "@/api/memory";
import { formatRelativeTime } from "@/utils/time";
import { DataTable } from "@/components/DataTable";
import type { DataColumn } from "@/components/DataTable";

const PAGE_SIZE = 50;

const MEMORY_TYPES = ["info", "skill", "episode", "intent", "todo", "reflection", "scratch"];
const TODO_STATUSES = ["open", "done", "dropped"];
const VIEWS = ["records", "dashboard", "explain"] as const;
type View = (typeof VIEWS)[number];

function dbLabel(path: string): string {
  const parts = path.split("/");
  return parts.slice(-2).join("/");
}

const selectClass =
  "px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 focus:outline-none focus:border-gray-500";
const inputClass =
  "px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500";

// ---------------------------------------------------------------------------
// Records view
// ---------------------------------------------------------------------------

function RecordsView({ db }: { db: string }) {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const page = parseInt(searchParams.get("page") || "1", 10);
  const owner = searchParams.get("owner") || "";
  const type = searchParams.get("type") || "";
  const status = searchParams.get("status") || "";
  const q = searchParams.get("q") || "";
  const includeArchived = searchParams.get("archived") === "true";

  const [records, setRecords] = useState<MemoryRecordRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const setParam = useCallback(
    (key: string, value: string) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("page", "1");
          if (value) next.set(key, value);
          else next.delete(key);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchMemoryRecords({
      db,
      owner: owner || undefined,
      type: type || undefined,
      status: status || undefined,
      q: q || undefined,
      include_archived: includeArchived,
      page,
      limit: PAGE_SIZE,
    })
      .then((data) => {
        if (cancelled) return;
        setRecords(data.records);
        setTotal(data.total);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err.message || err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [db, owner, type, status, q, includeArchived, page]);

  const columns: DataColumn<MemoryRecordRow>[] = [
    {
      key: "type",
      label: "Type",
      className: "w-28",
      render: (r) => (
        <span className="font-mono text-xs text-gray-300">
          {r.type}
          {r.status && <span className="text-gray-500"> · {r.status}</span>}
          {r.archived && <span className="text-red-400/70"> · archived</span>}
        </span>
      ),
    },
    {
      key: "owner",
      label: "Owner",
      className: "w-24 truncate",
      render: (r) => <span className="text-gray-400 text-xs">{r.owner || "(unowned)"}</span>,
    },
    {
      key: "importance_label",
      label: "Importance",
      className: "w-24",
      render: (r) => (
        <span className="text-xs text-gray-300" title={`${r.importance}`}>
          {r.importance_label}
        </span>
      ),
    },
    {
      key: "fetches",
      label: "Fetches",
      className: "w-16 text-right",
      render: (r) => r.fetches,
    },
    {
      key: "last_accessed_at",
      label: "Last accessed",
      className: "w-32",
      render: (r) => (
        <span className="text-gray-400 text-xs">{formatRelativeTime(r.last_accessed_at)}</span>
      ),
    },
    {
      key: "preview",
      label: "Preview",
      className: "max-w-md truncate",
      render: (r) => (
        <span title={r.preview}>
          {r.title && <span className="text-gray-200 font-medium">{r.title} — </span>}
          <span className="text-gray-400">{r.preview}</span>
        </span>
      ),
    },
    {
      key: "edge_count",
      label: "Edges",
      className: "w-14 text-right",
      render: (r) => r.edge_count,
    },
  ];

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <input
          type="search"
          value={q}
          onChange={(e) => setParam("q", e.target.value)}
          placeholder="Keyword search..."
          className={`${inputClass} w-52`}
        />
        <input
          type="text"
          value={owner}
          onChange={(e) => setParam("owner", e.target.value)}
          placeholder="Owner"
          className={`${inputClass} w-28`}
        />
        <select value={type} onChange={(e) => setParam("type", e.target.value)} className={selectClass}>
          <option value="">All types</option>
          {MEMORY_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <select
          value={status}
          onChange={(e) => setParam("status", e.target.value)}
          className={selectClass}
        >
          <option value="">Any status</option>
          {TODO_STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={includeArchived}
            onChange={(e) => setParam("archived", e.target.checked ? "true" : "")}
            className="rounded border-gray-700 bg-gray-800 w-3.5 h-3.5 accent-gray-500"
          />
          Include archived
        </label>
      </div>

      {error && <div className="mb-3 text-sm text-red-400">{error}</div>}

      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <DataTable<MemoryRecordRow>
          data={records}
          columns={columns}
          getKey={(r) => r.id}
          onRowClick={(r) =>
            navigate(`/memory/record?db=${encodeURIComponent(db)}&id=${encodeURIComponent(r.id)}`)
          }
          total={total}
          page={page}
          pageSize={PAGE_SIZE}
          onPageChange={(p) => setParam("page", String(p))}
          loading={loading}
          emptyMessage="No memories found"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dashboard view
// ---------------------------------------------------------------------------

function StatTile({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 px-4 py-3">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-lg text-gray-100 font-semibold mt-1">{value ?? "—"}</div>
    </div>
  );
}

function CountTable({ title, counts }: { title: string; counts: Record<string, number> }) {
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      <div className="px-4 py-2 text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800">
        {title}
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(([key, count]) => (
            <tr key={key} className="border-b border-gray-800/50 last:border-b-0">
              <td className="px-4 py-1.5 text-gray-300 font-mono text-xs">{key}</td>
              <td className="px-4 py-1.5 text-right text-gray-400">{count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DashboardView({ db }: { db: string }) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStats(null);
    setError(null);
    fetchMemoryStats(db)
      .then((data) => {
        if (!cancelled) setStats(data);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err.message || err));
      });
    return () => {
      cancelled = true;
    };
  }, [db]);

  if (error) return <div className="text-sm text-red-400">{error}</div>;
  if (!stats) return <div className="text-gray-500 py-12 text-center">Loading...</div>;

  const pct = (v: number | null | undefined) => (v == null ? null : `${v}%`);
  const rate = (v: number | null | undefined) =>
    v == null ? null : `${Math.round(v * 100)}%`;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatTile label="Memories" value={stats.total} />
        <StatTile label="Total fetches" value={stats.total_fetches} />
        <StatTile label="Never fetched" value={pct(stats.never_fetched_pct)} />
        <StatTile
          label="Fetch concentration (top 10%)"
          value={rate(stats.fetch_concentration_top10pct)}
        />
        <StatTile label="Dedup reinforces" value={stats.dedup_reinforces} />
        <StatTile label="With references" value={stats.with_references} />
        <StatTile label="Injected memories" value={stats.injected_memories} />
        <StatTile label="Injected-used rate" value={rate(stats.injected_used_rate)} />
        <StatTile label="Open todos" value={stats.todos_open} />
        <StatTile label="Closed todos" value={stats.todos_closed} />
        <StatTile
          label="Median open-todo age"
          value={
            stats.todo_median_open_age_hours == null
              ? null
              : `${stats.todo_median_open_age_hours}h`
          }
        />
        <StatTile label="Cross-owner reads" value={stats.cross_owner_reads} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {stats.by_type && <CountTable title="By type" counts={stats.by_type} />}
        {stats.by_owner && <CountTable title="By owner" counts={stats.by_owner} />}
      </div>

      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="px-4 py-2 text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800">
          Maintenance history
        </div>
        {stats.maintenance.length === 0 ? (
          <div className="px-4 py-6 text-sm text-gray-500 text-center">No maintenance runs yet</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase text-left">
                <th className="px-4 py-2 font-medium w-40">When</th>
                <th className="px-4 py-2 font-medium w-28">Kind</th>
                <th className="px-4 py-2 font-medium w-24">Trigger</th>
                <th className="px-4 py-2 font-medium w-28">Outcome</th>
                <th className="px-4 py-2 font-medium w-24">Duration</th>
                <th className="px-4 py-2 font-medium">Report</th>
              </tr>
            </thead>
            <tbody>
              {stats.maintenance.map((entry, i) => {
                const { trigger, interrupted, stopped_in, duration_ms, ...rest } = entry.report;
                return (
                  <tr key={i} className="border-b border-gray-800/50 last:border-b-0">
                    <td className="px-4 py-1.5 text-gray-400 text-xs">
                      {formatRelativeTime(entry.ts)}
                    </td>
                    <td className="px-4 py-1.5 text-gray-300 font-mono text-xs">{entry.kind}</td>
                    <td className="px-4 py-1.5 text-gray-400 font-mono text-xs">
                      {typeof trigger === "string" ? trigger : "—"}
                    </td>
                    <td className="px-4 py-1.5 text-xs">
                      {interrupted ? (
                        <span className="text-amber-400">
                          interrupted{typeof stopped_in === "string" && stopped_in ? ` @ ${stopped_in}` : ""}
                        </span>
                      ) : (
                        <span className="text-gray-400">completed</span>
                      )}
                    </td>
                    <td className="px-4 py-1.5 text-gray-400 font-mono text-xs">
                      {typeof duration_ms === "number" ? `${duration_ms.toFixed(0)} ms` : "—"}
                    </td>
                    <td className="px-4 py-1.5 text-gray-400 font-mono text-xs truncate max-w-md">
                      {JSON.stringify(rest)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Explain view
// ---------------------------------------------------------------------------

function ExplainView({ db }: { db: string }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [k, setK] = useState(10);
  const [rows, setRows] = useState<MemoryExplainRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setRows(await fetchMemoryExplain(db, query, k));
    } catch (err) {
      setRows(null);
      setError(String((err as Error).message || err));
    } finally {
      setLoading(false);
    }
  }, [db, query, k]);

  const num = (v: number) => v.toFixed(3);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Query to explain..."
          className={`${inputClass} w-96`}
        />
        <label className="text-xs text-gray-400 flex items-center gap-1">
          k
          <input
            type="number"
            min={1}
            max={100}
            value={k}
            onChange={(e) => setK(Math.max(1, parseInt(e.target.value, 10) || 10))}
            className={`${inputClass} w-16`}
          />
        </label>
        <button
          onClick={run}
          disabled={loading || !query.trim()}
          className="px-3 py-1 text-xs rounded bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Scoring..." : "Explain"}
        </button>
      </div>
      <p className="text-xs text-gray-600 mb-3">
        Dry-run recall: scores the store's candidates for the query without touching or logging.
        Uses the offline hashing embedder — cosine is only meaningful for hashing-backed stores.
      </p>

      {error && <div className="mb-3 text-sm text-red-400">{error}</div>}

      {rows && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase text-left">
                <th className="px-3 py-2 font-medium w-12">Rank</th>
                <th className="px-3 py-2 font-medium w-16 text-right">Score</th>
                <th className="px-3 py-2 font-medium w-16">Source</th>
                <th className="px-3 py-2 font-medium w-14 text-right">Cos</th>
                <th className="px-3 py-2 font-medium w-14 text-right">Rel</th>
                <th className="px-3 py-2 font-medium w-14 text-right">Rec</th>
                <th className="px-3 py-2 font-medium w-14 text-right">Imp</th>
                <th className="px-3 py-2 font-medium w-14 text-right">Spread</th>
                <th className="px-3 py-2 font-medium w-20">Type</th>
                <th className="px-3 py-2 font-medium w-24">Owner</th>
                <th className="px-3 py-2 font-medium">Head</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={11} className="text-center py-8 text-gray-500">
                    No candidates matched
                  </td>
                </tr>
              ) : (
                rows.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() =>
                      navigate(
                        `/memory/record?db=${encodeURIComponent(db)}&id=${encodeURIComponent(r.id)}`,
                      )
                    }
                    className="border-b border-gray-800/50 last:border-b-0 cursor-pointer hover:bg-gray-800/50 transition-colors"
                  >
                    <td className="px-3 py-1.5 text-gray-400">{r.rank}</td>
                    <td className="px-3 py-1.5 text-right text-gray-200 font-mono text-xs">
                      {num(r.score)}
                    </td>
                    <td className="px-3 py-1.5 text-gray-400 text-xs">{r.source}</td>
                    <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">{num(r.cos)}</td>
                    <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">{num(r.rel)}</td>
                    <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">{num(r.rec)}</td>
                    <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">{num(r.imp)}</td>
                    <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">{num(r.spread)}</td>
                    <td className="px-3 py-1.5 text-gray-300 font-mono text-xs">{r.type}</td>
                    <td className="px-3 py-1.5 text-gray-400 text-xs truncate">{r.owner || "(unowned)"}</td>
                    <td className="px-3 py-1.5 text-gray-400 text-xs truncate max-w-md" title={r.head}>
                      {r.head}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page shell: db selector + view switcher
// ---------------------------------------------------------------------------

export function MemoryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const db = searchParams.get("db") || "";
  const viewParam = searchParams.get("view") || "records";
  const view: View = (VIEWS as readonly string[]).includes(viewParam)
    ? (viewParam as View)
    : "records";

  const [dbs, setDbs] = useState<MemoryDbInfo[] | null>(null);

  useEffect(() => {
    fetchMemoryDbs()
      .then(setDbs)
      .catch(() => setDbs([]));
  }, []);

  // Default to the first discovered db when none is selected.
  useEffect(() => {
    if (!db && dbs && dbs.length > 0) {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("db", dbs[0].path);
          return next;
        },
        { replace: true },
      );
    }
  }, [db, dbs, setSearchParams]);

  const setView = useCallback(
    (v: View) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams();
          const currentDb = prev.get("db");
          if (currentDb) next.set("db", currentDb);
          next.set("view", v);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setDb = useCallback(
    (path: string) => {
      const next = new URLSearchParams();
      if (path) next.set("db", path);
      next.set("view", view);
      setSearchParams(next, { replace: true });
    },
    [setSearchParams, view],
  );

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-semibold text-gray-100">Memory</h1>
          <div className="flex gap-1">
            {VIEWS.map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1 text-sm rounded transition-colors capitalize ${
                  view === v
                    ? "bg-gray-800 text-gray-100"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <select
          value={db}
          onChange={(e) => setDb(e.target.value)}
          className={`${selectClass} max-w-md font-mono`}
          title={db}
        >
          {!db && <option value="">Select a memory db...</option>}
          {db && (!dbs || !dbs.some((d) => d.path === db)) && (
            <option value={db}>{dbLabel(db)}</option>
          )}
          {(dbs ?? []).map((d) => (
            <option key={d.path} value={d.path}>
              {dbLabel(d.path)}
            </option>
          ))}
        </select>
      </div>

      {!db ? (
        <div className="text-center py-12 text-gray-500">
          {dbs === null
            ? "Loading..."
            : "No memory databases discovered under ./.nooa — pass one with ?db=<path>"}
        </div>
      ) : (
        <>
          {view === "records" && <RecordsView db={db} />}
          {view === "dashboard" && <DashboardView db={db} />}
          {view === "explain" && <ExplainView db={db} />}
        </>
      )}
    </div>
  );
}
