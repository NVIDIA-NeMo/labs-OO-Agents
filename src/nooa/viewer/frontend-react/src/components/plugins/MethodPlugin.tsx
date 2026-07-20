import type { TraceEvent } from '@/api/types';
import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatPythonValue(value: unknown, maxLength = 40): string {
  if (value === null || value === undefined) return 'None';

  if (typeof value === 'string') {
    if (/^-?\d+\.?\d*$/.test(value)) return value;
    try {
      const parsed = JSON.parse(value);
      if (typeof parsed === 'object') {
        const json = JSON.stringify(parsed);
        return json.length > maxLength ? json.substring(0, maxLength - 3) + '...' : json;
      }
    } catch {
      // plain string
    }
    const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    const quoted = `"${escaped}"`;
    return quoted.length > maxLength ? quoted.substring(0, maxLength - 3) + '..."' : quoted;
  }

  if (typeof value === 'number' || typeof value === 'boolean') return String(value);

  if (typeof value === 'object' && value !== null) {
    const json = JSON.stringify(value);
    return json.length > maxLength ? json.substring(0, maxLength - 3) + '...' : json;
  }

  return String(value);
}

function attrWithFallback(attrs: Record<string, unknown>, agentKey: string, methodKey: string): unknown {
  return attrs[agentKey] ?? attrs[methodKey];
}

// OI-first: inputs are carried as input.value = {"args":[...],"kwargs":{...}}
// on new traces; old traces use flat agent.args/method.args (JSON strings).
function oiInput(attrs: Record<string, unknown>): { args?: unknown; kwargs?: unknown } | null {
  const iv = attrs['input.value'];
  if (iv === undefined) return null;
  try {
    return (typeof iv === 'string' ? JSON.parse(iv) : iv) as { args?: unknown; kwargs?: unknown };
  } catch {
    return null;
  }
}

function getCallArgs(attrs: Record<string, unknown>): unknown[] {
  const oi = oiInput(attrs);
  if (oi && Array.isArray(oi.args)) return oi.args;
  try {
    return JSON.parse((attrWithFallback(attrs, 'agent.args', 'method.args') as string) || '[]');
  } catch {
    return [];
  }
}

function getCallKwargs(attrs: Record<string, unknown>): Record<string, unknown> {
  const oi = oiInput(attrs);
  if (oi && oi.kwargs && typeof oi.kwargs === 'object') return oi.kwargs as Record<string, unknown>;
  try {
    return JSON.parse((attrWithFallback(attrs, 'agent.kwargs', 'method.kwargs') as string) || '{}');
  } catch {
    return {};
  }
}

function buildCallString(attrs: Record<string, unknown>, truncate = true): string | null {
  const method = (attrs['agent.method'] ?? attrs['method.name']) as string | undefined;
  if (!method) return null;

  const argParts: string[] = [];
  const maxLen = truncate ? 40 : Infinity;

  for (const arg of getCallArgs(attrs)) {
    argParts.push(formatPythonValue(arg, maxLen));
  }
  for (const [k, v] of Object.entries(getCallKwargs(attrs))) {
    argParts.push(`${k}=${formatPythonValue(v, maxLen)}`);
  }

  return `${method}(${argParts.join(', ')})`;
}

function ExpandedMetadata({
  attrs,
  agentName,
  strategy,
  event,
  rawJsonOpen,
}: {
  attrs: Record<string, unknown>;
  agentName: string;
  strategy: string;
  event: TraceEvent;
  rawJsonOpen?: boolean;
}) {
  const meta: Record<string, unknown> = {};
  if (agentName) meta['Agent'] = agentName;
  if (attrs['agent.call_id']) meta['Call ID'] = attrs['agent.call_id'];
  if (strategy) meta['Strategy'] = strategy;
  if (attrs['agent.file_path']) meta['File'] = attrs['agent.file_path'];

  return (
    <>
      {Object.keys(meta).length > 0 && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-blue-500 mb-2">
          <div className="text-xs text-gray-500 mb-1">Metadata</div>
          <CodeBox code={JSON.stringify(meta, null, 2)} language="json" />
        </div>
      )}
      <details className="mt-2" open={rawJsonOpen}>
        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
          Raw JSON
        </summary>
        <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
      </details>
    </>
  );
}

export function MethodPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const agentName = (attrs['agent.name'] as string) || '';
  const method =
    (attrs['agent.method'] as string) ||
    (attrs['method.name'] as string) ||
    (attrs.span_name as string) ||
    event.type.replace('span.', '');
  const durationNs = (attrs.duration_ns as number) || 0;
  const statusCode = (attrs.status_code as string) || 'UNSET';
  const strategy = (attrs['agent.strategy.name'] as string) || '';
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  // OI-first: output.value; fall back to native agent.result/method.result.
  const result =
    ((attrs['output.value'] ?? attrs['agent.result'] ?? attrs['method.result']) as string) ?? null;
  const hasError = !!(attrs['error.message'] || statusCode === 'ERROR');

  if (viewState === 'collapsed') {
    const collapsedCall = buildCallString(attrs, true);
    const resultStr = result != null ? formatPythonValue(result, 40) : '';
    const summary = collapsedCall
      ? `${collapsedCall} -> ${resultStr}`
      : `${method}() -> ${resultStr}`;

    return (
      <div className="flex items-center justify-between text-sm">
        <div className="flex-1 min-w-0 text-gray-300 font-mono truncate">
          <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold mr-1">
            {method}
          </span>
          {summary}
          {durationNs > 0 && (
            <span className="text-gray-500 ml-2">({formatDuration(durationNs)})</span>
          )}
          {hasError && <span className="text-red-400 ml-1">ERROR</span>}
        </div>
        <div className="flex items-center gap-3 flex-shrink-0 ml-4">
          <span className="text-[11px] opacity-60">{event.type}</span>
          <span className="text-gray-500 text-xs">{timestamp}</span>
        </div>
      </div>
    );
  }

  const fullCall = buildCallString(attrs, false);
  const errorMessage = attrs['error.message'] as string | undefined;

  let resultDisplay = result ?? '';
  let resultLang = 'markdown';
  if (result != null) {
    try {
      const parsed = JSON.parse(result);
      resultDisplay = JSON.stringify(parsed, null, 2);
      resultLang = 'json';
    } catch {
      resultDisplay = result;
    }
  }

  return (
    <div>
      <div className="flex items-center gap-3 text-xs text-gray-400 mb-2">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          {method}
        </span>
        {durationNs > 0 && <span>{formatDuration(durationNs)}</span>}
        {hasError && <span className="text-red-400">Error</span>}
        <span className="ml-auto opacity-60">{timestamp}</span>
        {viewControls}
      </div>

      {fullCall && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-blue-700 mb-2">
          <div className="text-xs text-gray-500 mb-1">Call</div>
          <CodeBox code={fullCall} language="python" />
        </div>
      )}

      {result != null && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-amber-600 mb-2">
          <div className="text-xs text-gray-500 mb-1">Result</div>
          <CodeBox code={resultDisplay} language={resultLang} />
        </div>
      )}

      {hasError && errorMessage && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-red-700 mb-2">
          <div className="text-xs text-gray-500 mb-1">Error</div>
          <pre className="text-sm text-red-300 whitespace-pre-wrap break-words font-mono">
            {errorMessage}
          </pre>
        </div>
      )}

      {viewState === 'expanded' && (
        <ExpandedMetadata attrs={attrs} agentName={agentName} strategy={strategy} event={event} rawJsonOpen={rawJsonOpen} />
      )}
    </div>
  );
}
