import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

interface ParsedResult {
  definedMethods: string[];
  returnedValue: string | null;
  stdout: string | null;
}

// OI-first code extraction: new traces carry the executed code as
// input.value = {"code": "..."} (application/json); old traces carry a flat `code`.
function getExecCode(attrs: Record<string, unknown>): string {
  const iv = attrs['input.value'];
  if (typeof iv === 'string') {
    try {
      const o = JSON.parse(iv) as { code?: unknown };
      if (typeof o?.code === 'string') return o.code;
    } catch {
      // not JSON
    }
  } else if (iv && typeof iv === 'object' && typeof (iv as { code?: unknown }).code === 'string') {
    return (iv as { code: string }).code;
  }
  return (attrs.code as string) || (attrs['code_execution.code'] as string) || '';
}

function parseResult(attrs: Record<string, unknown>): ParsedResult {
  // OI-first: output.value; fall back to native result attrs.
  const raw = attrs['output.value'] ?? attrs.result ?? attrs['code_execution.result'];
  let obj: Record<string, unknown> | null = null;

  if (typeof raw === 'string') {
    try {
      obj = JSON.parse(raw);
    } catch {
      // not JSON
    }
  } else if (typeof raw === 'object' && raw !== null) {
    obj = raw as Record<string, unknown>;
  }

  const definedMethods: string[] = [];
  let returnedValue: string | null = null;
  let stdout: string | null = null;

  if (obj) {
    if (obj.defined_methods && typeof obj.defined_methods === 'object') {
      definedMethods.push(...Object.keys(obj.defined_methods as object));
    }
    if (
      obj.returned_value !== undefined &&
      obj.returned_value !== null &&
      obj.returned_value !== ''
    ) {
      returnedValue =
        typeof obj.returned_value === 'string'
          ? obj.returned_value
          : JSON.stringify(obj.returned_value, null, 2);
    }
    if (obj.stdout && typeof obj.stdout === 'string' && obj.stdout.trim()) {
      stdout = obj.stdout;
    }
  }

  return { definedMethods, returnedValue, stdout };
}

export function CodeExecutionPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const code = getExecCode(attrs);
  const durationNs = (attrs.duration_ns as number) || 0;
  const statusCode = (attrs.status_code as string) || 'UNSET';
  const hasError = !!attrs.error || statusCode === 'ERROR';
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  const { definedMethods, returnedValue, stdout } = parseResult(attrs);
  const methodsList = definedMethods.length > 0 ? definedMethods.join(', ') : 'none';

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center justify-between text-sm">
        <div className="flex-1 min-w-0 text-gray-300 font-mono truncate">
          <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold mr-1">
            Code Execution
          </span>
          Defined: {methodsList}
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

  const headerLine = (
    <div className="flex items-center gap-3 text-xs text-gray-400 mb-2">
      <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
        Code Execution
      </span>
      <span>Defined: {methodsList}</span>
      {durationNs > 0 && <span>{formatDuration(durationNs)}</span>}
      {hasError && <span className="text-red-400">Error</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  const isExpanded = viewState === 'expanded';

  return (
    <div>
      {headerLine}

      {definedMethods.length > 0 && (
        <>
          {isExpanded && (
            <div className="text-[11px] font-semibold uppercase text-gray-500 mt-4 mb-2">
              Defined Methods ({definedMethods.length})
            </div>
          )}
          <div className="flex flex-wrap gap-1.5 mb-2">
            {definedMethods.map((m) => (
              <span
                key={m}
                className="font-mono text-xs font-semibold text-purple-400 px-2 py-1 bg-[#1a1a2e] rounded border border-purple-700"
              >
                {m}
              </span>
            ))}
          </div>
        </>
      )}

      {code && (
        <>
          {isExpanded && (
            <div className="text-[11px] font-semibold uppercase text-gray-500 mt-4 mb-2">
              Generated Code
            </div>
          )}
          <div className="mb-2">
            <CodeBox
              code={code}
              language="python"
              maxHeight={isExpanded ? 'none' : '300px'}
            />
          </div>
        </>
      )}

      {returnedValue && (
        <>
          {isExpanded && (
            <div className="text-[11px] font-semibold uppercase text-gray-500 mt-4 mb-2">
              Execution Result
            </div>
          )}
          <div className="p-3 bg-gray-900 rounded border-l-4 border-green-700 mb-2">
            <div className="text-xs text-gray-500 mb-1">Returned Value</div>
            <pre className="text-sm text-green-300 whitespace-pre-wrap break-words font-mono">
              {returnedValue}
            </pre>
          </div>
        </>
      )}

      {stdout && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-amber-600 mb-2">
          <div className="text-xs text-gray-500 mb-1">stdout</div>
          <pre className="text-sm text-amber-200 whitespace-pre-wrap break-words font-mono">
            {stdout}
          </pre>
        </div>
      )}

      {hasError && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-red-700 mb-2">
          <div className="text-xs text-gray-500 mb-1">Error</div>
          <pre className="text-sm text-red-300 whitespace-pre-wrap break-words font-mono">
            {(attrs.error as string) || (attrs.status_description as string) || 'Unknown error'}
          </pre>
        </div>
      )}

      {isExpanded && (
        <details className="mt-2" open={rawJsonOpen}>
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
            Raw JSON
          </summary>
          <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
        </details>
      )}
    </div>
  );
}
