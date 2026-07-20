import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

function getToolName(attrs: Record<string, unknown>, eventType: string): string {
  const spanName = (attrs.span_name as string) || '';
  if (spanName.startsWith('tool_execution.')) {
    return spanName.substring('tool_execution.'.length);
  }
  const typeSuffix = eventType.replace(/^span\.tool_execution\.?/, '');
  if (typeSuffix) return typeSuffix;
  return (attrs['tool.name'] as string) || 'unknown';
}

function getToolResult(attrs: Record<string, unknown>): unknown {
  // OI-first: output.value; fall back to native tool.result.
  const raw = attrs['output.value'] ?? attrs['tool.result'];
  if (typeof raw === 'string') {
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }
  return raw;
}

function hasError(attrs: Record<string, unknown>): boolean {
  return !!(attrs['error.message'] || attrs.error || attrs.status_code === 'ERROR');
}

function getErrorMessage(attrs: Record<string, unknown>): string {
  return (attrs['error.message'] as string) || (attrs.error as string) || 'Validation failed';
}

function formatAsRepr(value: unknown, indent = 0): string {
  const pad = '  '.repeat(indent);
  const next = '  '.repeat(indent + 1);

  if (value === null || value === undefined) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'string') return JSON.stringify(value);

  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    if (value.length <= 3 && value.every((v) => typeof v !== 'object' || v === null)) {
      return '[' + value.map((v) => formatAsRepr(v, 0)).join(', ') + ']';
    }
    return (
      '[\n' + value.map((v) => next + formatAsRepr(v, indent + 1)).join(',\n') + '\n' + pad + ']'
    );
  }

  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).filter((k) => k !== '__class__');
    const className = obj.__class__ as string | undefined;

    if (keys.length === 0) return className ? `${className}()` : '{}';

    if (className) {
      if (keys.length <= 2 && keys.every((k) => typeof obj[k] !== 'object' || obj[k] === null)) {
        return `${className}(${keys.map((k) => `${k}=${formatAsRepr(obj[k], 0)}`).join(', ')})`;
      }
      return `${className}(\n${keys.map((k) => next + `${k}=${formatAsRepr(obj[k], indent + 1)}`).join(',\n')}\n${pad})`;
    }

    if (keys.length <= 2 && keys.every((k) => typeof obj[k] !== 'object' || obj[k] === null)) {
      return (
        '{' + keys.map((k) => `${JSON.stringify(k)}: ${formatAsRepr(obj[k], 0)}`).join(', ') + '}'
      );
    }
    return (
      '{\n' +
      keys
        .map((k) => next + `${JSON.stringify(k)}: ${formatAsRepr(obj[k], indent + 1)}`)
        .join(',\n') +
      '\n' +
      pad +
      '}'
    );
  }

  return String(value);
}

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function collapsedSummary(toolName: string, result: unknown, isError: boolean): string {
  if (toolName === 'return_result') {
    if (isError) return 'Invalid Result returned';
    if (result && typeof result === 'object' && !Array.isArray(result)) {
      const obj = result as Record<string, unknown>;
      const className = obj.__class__ as string | undefined;
      const keys = Object.keys(obj).filter((k) => k !== '__class__');
      if (className) {
        return `-> ${className}(${keys.slice(0, 3).join(', ')}${keys.length > 3 ? ', ...' : ''})`;
      }
      if (keys.length <= 3) return `-> {${keys.join(', ')}}`;
      return `-> {${keys.slice(0, 3).join(', ')}, ...}`;
    }
    const repr = formatAsRepr(result);
    return `-> ${repr.length > 50 ? repr.substring(0, 47) + '...' : repr}`;
  }
  return `${toolName}()${isError ? ' ERROR' : ''}`;
}

export function ToolExecutionPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const toolName = getToolName(attrs, event.type);
  const durationNs = (attrs.duration_ns as number) || 0;
  const isError = hasError(attrs);
  const isReturnResult = toolName === 'return_result';
  const result = getToolResult(attrs);
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  const badgeCls = isError
    ? 'bg-red-900 text-red-200'
    : isReturnResult
      ? 'bg-green-900 text-green-200'
      : 'bg-purple-900 text-purple-200';

  const badgeLabel = isError
    ? isReturnResult
      ? 'INVALID'
      : 'ERROR'
    : isReturnResult
      ? 'RESULT'
      : toolName.toUpperCase();

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center gap-3 text-xs">
        <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${badgeCls}`}>
          {badgeLabel}
        </span>
        <span className="text-gray-300">{collapsedSummary(toolName, result, isError)}</span>
        {durationNs > 0 && <span className="text-gray-500">{formatDuration(durationNs)}</span>}
        <span className="ml-auto text-gray-600">{timestamp}</span>
      </div>
    );
  }

  const headerLine = (
    <div className="flex items-center gap-3 text-xs text-gray-400 mb-2">
      <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${badgeCls}`}>
        {badgeLabel}
      </span>
      <span
        className={
          isError
            ? 'text-red-400 font-semibold'
            : isReturnResult
              ? 'text-green-400 font-semibold'
              : 'text-orange-400 font-semibold'
        }
      >
        {isError
          ? isReturnResult
            ? 'Invalid Result'
            : 'Error'
          : isReturnResult
            ? 'Result'
            : toolName}
      </span>
      {durationNs > 0 && <span>{formatDuration(durationNs)}</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  if (viewState === 'concise') {
    return (
      <div>
        {headerLine}
        {isError ? (
          <div className="p-2.5 bg-gray-950 rounded border-l-4 border-red-600 mb-2">
            <div className="text-xs text-red-400 font-semibold mb-1">
              {isReturnResult ? 'Validation Error' : 'Error'}
            </div>
            <CodeBox
              code={getErrorMessage(attrs)}
              language="markdown"
              showLineNumbers={false}
              maxHeight="300px"
            />
          </div>
        ) : result !== null && result !== undefined ? (
          <div className="p-2.5 bg-gray-950 rounded border-l-4 border-green-600 mb-2">
            <div className="text-xs text-green-400 font-semibold mb-1">Result</div>
            <CodeBox code={formatAsRepr(result)} language="python" maxHeight="300px" />
          </div>
        ) : null}
      </div>
    );
  }

  // Expanded
  const metadataFields: [string, string][] = [];
  if (attrs['execution.id']) metadataFields.push(['Execution ID', String(attrs['execution.id'])]);
  if (attrs['generation.id'])
    metadataFields.push(['Generation ID', String(attrs['generation.id'])]);
  if (attrs['agent.name']) metadataFields.push(['Agent', String(attrs['agent.name'])]);
  if (attrs['result.type']) metadataFields.push(['Result Type', String(attrs['result.type'])]);

  return (
    <div>
      {headerLine}

      {isError ? (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">
            {isReturnResult ? 'Validation Error' : 'Error'}
          </div>
          <CodeBox
            code={getErrorMessage(attrs)}
            language="markdown"
            showLineNumbers={false}
            maxHeight="none"
          />
        </div>
      ) : result !== null && result !== undefined ? (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Result</div>
          <CodeBox code={formatAsRepr(result)} language="python" maxHeight="none" />
        </div>
      ) : null}

      {metadataFields.length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Metadata</div>
          <div className="bg-gray-950 rounded p-3 text-xs font-mono">
            {metadataFields.map(([label, value]) => (
              <div key={label} className="mb-1 last:mb-0">
                <span className="text-gray-500">{label}:</span>{' '}
                <span className="text-gray-200">{value}</span>
              </div>
            ))}
          </div>
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
