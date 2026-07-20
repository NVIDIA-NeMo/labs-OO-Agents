import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { fetchMemoryRecord } from "@/api/memory";
import type { MemoryAccessRecord, MemoryRecordDetail as RecordDetail } from "@/api/memory";
import { CodeBox } from "@/components/shared/CodeBox";
import { formatRelativeTime } from "@/utils/time";

function formatTs(ts: number | null | undefined): string {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString();
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      <div className="px-4 py-2 text-xs text-gray-500 uppercase tracking-wide border-b border-gray-800">
        {title}
      </div>
      {children}
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm text-gray-200 mt-0.5">{value ?? "—"}</div>
    </div>
  );
}

function TraceRefCell({ entry }: { entry: MemoryAccessRecord }) {
  if (!entry.trace_ref) return <span className="text-gray-600">—</span>;
  // Deep-link to the trace when we know which session it lives in; otherwise
  // the bare span id is still useful for manual lookup.
  if (entry.session_ref) {
    return (
      <Link
        to={`/traces/view?session_id=${encodeURIComponent(entry.session_ref)}`}
        className="text-blue-400 hover:text-blue-300 font-mono text-xs"
        title={`span ${entry.trace_ref} in session ${entry.session_ref}`}
      >
        {entry.trace_ref}
      </Link>
    );
  }
  return <span className="font-mono text-xs text-gray-400">{entry.trace_ref}</span>;
}

export function MemoryRecordDetail() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const db = searchParams.get("db") || "";
  const id = searchParams.get("id") || "";

  const [record, setRecord] = useState<RecordDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!db || !id) return;
    let cancelled = false;
    setRecord(null);
    setError(null);
    fetchMemoryRecord(db, id)
      .then((data) => {
        if (!cancelled) setRecord(data);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err.message || err));
      });
    return () => {
      cancelled = true;
    };
  }, [db, id]);

  const usage = record?.usage;

  return (
    <div className="max-w-[100rem] mx-auto px-4 py-6">
      <div className="flex items-center gap-3 mb-4 min-w-0">
        <button
          onClick={() => navigate(-1)}
          className="shrink-0 text-gray-400 hover:text-gray-200 transition-colors text-sm whitespace-nowrap"
        >
          &#9666; Back
        </button>
        <h1 className="text-lg font-mono text-gray-200 truncate min-w-0" title={id}>
          {id}
        </h1>
        {record && (
          <span className="shrink-0 px-2 py-0.5 text-xs rounded bg-gray-800 text-gray-300 font-mono">
            {record.type}
            {record.status && ` · ${record.status}`}
            {record.archived && " · archived"}
          </span>
        )}
      </div>

      {error && <div className="text-sm text-red-400 mb-4">{error}</div>}
      {!record && !error && <div className="text-gray-500 py-12 text-center">Loading...</div>}

      {record && (
        <div className="space-y-4">
          <Section title={record.title ? `Content — ${record.title}` : "Content"}>
            <div className="p-3">
              <CodeBox code={record.content} language="markdown" showLineNumbers={false} />
            </div>
          </Section>

          <Section title="Metadata">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 p-4">
              <MetaItem label="Owner" value={record.owner || "(unowned)"} />
              <MetaItem
                label="Importance"
                value={`${record.importance_label} (${record.importance})`}
              />
              <MetaItem label="Salience" value={`${record.salience_label} (${record.salience})`} />
              <MetaItem
                label="Confidence"
                value={`${record.confidence_label} (${record.confidence})`}
              />
              <MetaItem label="Mood" value={record.mood} />
              <MetaItem label="Strength" value={record.strength} />
              <MetaItem label="Created" value={formatTs(record.created_at)} />
              <MetaItem label="Last accessed" value={formatTs(record.last_accessed_at)} />
              <MetaItem label="Access count" value={record.access_count} />
              <MetaItem
                label="Tags"
                value={record.tags.length ? record.tags.join(", ") : null}
              />
              <MetaItem
                label="Entities"
                value={record.entities.length ? record.entities.join(", ") : null}
              />
              <MetaItem label="Place / task" value={record.place_or_task} />
            </div>
          </Section>

          {usage && (
            <Section title="Usage">
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 p-4">
                <MetaItem label="Fetches" value={usage.fetches} />
                <MetaItem label="Recalled" value={usage.recalled} />
                <MetaItem label="Searched" value={usage.searched} />
                <MetaItem label="Injected" value={usage.injected} />
                <MetaItem label="Deref" value={usage.deref} />
                <MetaItem label="Reinforced" value={usage.reinforced} />
                <MetaItem label="Mean rank" value={usage.mean_rank} />
                <MetaItem label="Mean score" value={usage.mean_score} />
                <MetaItem
                  label="Last fetch"
                  value={
                    usage.last_channel
                      ? `${usage.last_channel} · ${formatRelativeTime(usage.last_ts ?? 0)}`
                      : null
                  }
                />
                <MetaItem label="Retention" value={usage.retention} />
                <MetaItem
                  label="Prune ETA"
                  value={usage.prune_eta == null ? "never" : formatTs(usage.prune_eta)}
                />
                <MetaItem
                  label="Injected, never used"
                  value={usage.injected > 0 ? (usage.injected_never_used ? "yes" : "no") : null}
                />
              </div>
            </Section>
          )}

          {record.references.length > 0 && (
            <Section title="References (previews are write-time snapshots)">
              <div className="divide-y divide-gray-800/50">
                {record.references.map((ref, i) => (
                  <div key={i} className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-gray-200">
                        {ref.kind}:{ref.key}
                      </span>
                      <span className="text-xs text-gray-600">
                        captured {formatRelativeTime(ref.captured_at)}
                      </span>
                    </div>
                    {ref.preview && (
                      <div className="mt-1 text-xs text-gray-400 font-mono whitespace-pre-wrap">
                        <span className="text-gray-600">snapshot: </span>
                        {ref.preview}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Section>
          )}

          <Section title="Access history">
            {record.access_log.length === 0 ? (
              <div className="px-4 py-6 text-sm text-gray-500 text-center">No accesses logged</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800 text-xs text-gray-500 uppercase text-left">
                      <th className="px-3 py-2 font-medium w-40">When</th>
                      <th className="px-3 py-2 font-medium w-24">Channel</th>
                      <th className="px-3 py-2 font-medium w-24">Reader</th>
                      <th className="px-3 py-2 font-medium w-40">Session</th>
                      <th className="px-3 py-2 font-medium w-32">Trace</th>
                      <th className="px-3 py-2 font-medium">Query</th>
                      <th className="px-3 py-2 font-medium w-16 text-right">Score</th>
                      <th className="px-3 py-2 font-medium w-14 text-right">Rank</th>
                      <th className="px-3 py-2 font-medium w-48">Components</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...record.access_log].reverse().map((entry, i) => (
                      <tr key={i} className="border-b border-gray-800/50 last:border-b-0">
                        <td className="px-3 py-1.5 text-gray-400 text-xs" title={formatTs(entry.ts)}>
                          {formatRelativeTime(entry.ts)}
                        </td>
                        <td className="px-3 py-1.5 text-gray-300 font-mono text-xs">
                          {entry.channel}
                        </td>
                        <td className="px-3 py-1.5 text-gray-400 text-xs">
                          {entry.reader_owner || "—"}
                        </td>
                        <td
                          className="px-3 py-1.5 text-gray-400 font-mono text-xs truncate max-w-40"
                          title={entry.session_ref || ""}
                        >
                          {entry.session_ref || "—"}
                        </td>
                        <td className="px-3 py-1.5 truncate max-w-32">
                          <TraceRefCell entry={entry} />
                        </td>
                        <td
                          className="px-3 py-1.5 text-gray-400 text-xs truncate max-w-xs"
                          title={entry.query || ""}
                        >
                          {entry.query || "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">
                          {entry.score ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-right text-gray-400 font-mono text-xs">
                          {entry.rank ?? "—"}
                        </td>
                        <td className="px-3 py-1.5 text-gray-500 font-mono text-[11px] truncate max-w-48">
                          {entry.components ? JSON.stringify(entry.components) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          {record.edges.length > 0 && (
            <Section title="Edges">
              <div className="divide-y divide-gray-800/50">
                {record.edges.map((edge, i) => (
                  <Link
                    key={i}
                    to={`/memory/record?db=${encodeURIComponent(db)}&id=${encodeURIComponent(edge.target_id)}`}
                    className="flex items-center gap-3 px-4 py-2.5 hover:bg-gray-800/50 transition-colors"
                  >
                    <span className="font-mono text-xs text-gray-500 w-24 shrink-0">
                      {edge.type} ({edge.weight})
                    </span>
                    <span className="font-mono text-xs text-blue-400 shrink-0">
                      {edge.target_id.slice(0, 12)}
                    </span>
                    {edge.target_type && (
                      <span className="text-xs text-gray-500 shrink-0">{edge.target_type}</span>
                    )}
                    <span className="text-xs text-gray-400 truncate">
                      {edge.target_preview || ""}
                    </span>
                  </Link>
                ))}
              </div>
            </Section>
          )}
        </div>
      )}
    </div>
  );
}
