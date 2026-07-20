import { Fragment } from "react";
import type { PluginProps } from "./registry";
import { CodeBox } from "@/components/shared/CodeBox";

function formatDuration(ns: number): { ms: string; sec: string } {
  const ms = (ns / 1e6).toFixed(0);
  const sec = (ns / 1e9).toFixed(2);
  return { ms, sec };
}

function formatHarnessValue(v: unknown): string {
  if (Array.isArray(v))
    return `[${v.map((x) => formatHarnessValue(x)).join(", ")}]`;
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(4);
  return String(v);
}

const TIMER_RE =
  /^harness\.time\.(\w+)\.(total_s|min_s|max_s|avg_s|count|samples)$/;
type TimerStats = {
  count?: unknown;
  avg_s?: unknown;
  min_s?: unknown;
  max_s?: unknown;
  total_s?: unknown;
  samples?: unknown;
};

function formatSamples(s: unknown): string {
  if (!Array.isArray(s)) return String(s);
  const head = s.slice(0, 3).map((x) => formatHarnessValue(x));
  return s.length > 3
    ? `[${head.join(", ")}, …${s.length - 3} more]`
    : `[${head.join(", ")}]`;
}

export function GenerationPlugin({
  event,
  viewState,
  rawJsonOpen,
  viewControls,
}: PluginProps) {
  const attrs = event.attributes || {};
  const strategy = (attrs["generation.strategy"] as string) || "unknown";
  const durationNs = (attrs.duration_ns as number) || 0;
  const agentMethod = attrs["agent.method"] as string | undefined;
  const agentName = (attrs["agent.name"] as string) || "Agent";
  const timestamp = event.timestamp
    ? new Date(event.timestamp).toLocaleTimeString()
    : "";

  const statusCode = (attrs.status_code as string) || "";
  const errorType = (attrs["error.type"] as string) || "";
  const errorMessage = (attrs["error.message"] as string) || "";
  const isError = statusCode === "ERROR" || !!errorType || !!errorMessage;
  const isMaxIter =
    errorMessage.includes("max_iterations") ||
    errorMessage.includes("max iterations");

  const { ms, sec } = formatDuration(durationNs);

  const headerLine = (
    <div className="flex items-center gap-3 text-xs">
      <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
        {strategy}
      </span>
      {durationNs > 0 && (
        <span className="text-gray-400">
          {ms}ms ({sec}s)
        </span>
      )}
      {agentMethod && (
        <span className="text-gray-400">
          {agentName}.{agentMethod}
        </span>
      )}
      {isMaxIter && (
        <span className="text-orange-400 font-semibold">MAX ITERATIONS</span>
      )}
      {isError && !isMaxIter && (
        <span className="text-red-400 font-semibold">ERROR</span>
      )}
      <span className="ml-auto text-gray-600">{timestamp}</span>
      {viewControls}
    </div>
  );

  if (viewState === "collapsed" || viewState === "concise") {
    return (
      <div
        className={
          isError
            ? `border-l-[3px] pl-2 ${isMaxIter ? "border-orange-500" : "border-red-500"}`
            : ""
        }
      >
        {headerLine}
      </div>
    );
  }

  // Expanded
  const meta: Record<string, unknown> = {};
  if (attrs["generation.id"]) meta["Generation ID"] = attrs["generation.id"];
  if (agentMethod) meta["Method"] = agentMethod;
  if (agentName) meta["Agent"] = agentName;

  const timerStats: Record<string, TimerStats> = {};
  const otherHarnessEntries: [string, unknown][] = [];
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith("harness.")) continue;
    const match = key.match(TIMER_RE);
    if (match) {
      const [, name, stat] = match;
      (timerStats[name] ??= {})[stat as keyof TimerStats] = value;
    } else {
      otherHarnessEntries.push([key, value]);
    }
  }
  const timerNames = Object.keys(timerStats).sort();
  otherHarnessEntries.sort(([a], [b]) => a.localeCompare(b));

  return (
    <div>
      {headerLine}

      {isError && (
        <div
          className={`mt-3 p-3 rounded-lg border ${
            isMaxIter
              ? "bg-gradient-to-br from-yellow-950 to-yellow-900 border-yellow-600"
              : "bg-gradient-to-br from-red-950 to-red-900 border-red-600"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg font-bold">{isMaxIter ? "!" : "X"}</span>
            <span
              className={`font-semibold text-sm ${isMaxIter ? "text-yellow-200" : "text-red-200"}`}
            >
              {isMaxIter
                ? "Max Iterations Reached"
                : errorType || "Generation Error"}
            </span>
          </div>
          {errorMessage && (
            <pre
              className={`text-xs font-mono whitespace-pre-wrap break-words ${isMaxIter ? "text-yellow-100" : "text-red-100"}`}
            >
              {errorMessage}
            </pre>
          )}
        </div>
      )}

      {Object.keys(meta).length > 0 && (
        <div className="mt-3 p-3 bg-gray-900 rounded border-l-4 border-purple-700">
          <div className="text-xs text-gray-500 mb-1">Metadata</div>
          <CodeBox code={JSON.stringify(meta, null, 2)} language="json" />
        </div>
      )}

      {(timerNames.length > 0 || otherHarnessEntries.length > 0) && (
        <div className="mt-3 p-3 bg-gray-900 rounded border-l-4 border-teal-700">
          <div className="text-xs text-gray-500 mb-2">Harness Telemetry</div>

          {timerNames.length > 0 && (
            <table className="text-xs font-mono w-full mb-3">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left pb-1 pr-3 font-normal">timer</th>
                  <th className="text-right pb-1 px-2 font-normal">count</th>
                  <th className="text-right pb-1 px-2 font-normal">avg</th>
                  <th className="text-right pb-1 px-2 font-normal">min</th>
                  <th className="text-right pb-1 px-2 font-normal">max</th>
                  <th className="text-right pb-1 px-2 font-normal">total</th>
                  <th className="text-left pb-1 pl-3 font-normal">samples</th>
                </tr>
              </thead>
              <tbody>
                {timerNames.map((name) => {
                  const s = timerStats[name];
                  return (
                    <tr key={name} className="text-gray-200">
                      <td className="py-0.5 pr-3 text-gray-400">{name}</td>
                      <td className="py-0.5 px-2 text-right">
                        {formatHarnessValue(s.count)}
                      </td>
                      <td className="py-0.5 px-2 text-right">
                        {formatHarnessValue(s.avg_s)}
                      </td>
                      <td className="py-0.5 px-2 text-right">
                        {formatHarnessValue(s.min_s)}
                      </td>
                      <td className="py-0.5 px-2 text-right">
                        {formatHarnessValue(s.max_s)}
                      </td>
                      <td className="py-0.5 px-2 text-right">
                        {formatHarnessValue(s.total_s)}
                      </td>
                      <td className="py-0.5 pl-3 break-all">
                        {formatSamples(s.samples)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}

          {otherHarnessEntries.length > 0 && (
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs font-mono">
              {otherHarnessEntries.map(([key, value]) => (
                <Fragment key={key}>
                  <span className="text-gray-400">
                    {key.slice("harness.".length)}
                  </span>
                  <span className="text-gray-200 break-all">
                    {formatHarnessValue(value)}
                  </span>
                </Fragment>
              ))}
            </div>
          )}
        </div>
      )}

      <details className="mt-2" open={rawJsonOpen}>
        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
          Raw JSON
        </summary>
        <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
      </details>
    </div>
  );
}
