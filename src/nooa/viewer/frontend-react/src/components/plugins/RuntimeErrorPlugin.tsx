import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

export function RuntimeErrorPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes;
  const errorType = attrs.error_type as string | undefined;
  const message = (attrs.message as string) || event.body || '';
  const line = attrs.line as number | undefined;
  const traceback = attrs.traceback as string | undefined;
  const codeLines = attrs.code_lines as string[] | undefined;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    const summary = `Line ${line ?? '?'}: ${errorType || 'Error'}: ${message || 'Unknown error'}`;
    return (
      <div className="flex items-center justify-between text-sm">
        <div className="flex-1 min-w-0 text-red-300 font-mono truncate">
          <span className="px-1.5 py-0.5 rounded bg-red-900 text-red-200 text-xs font-semibold mr-1">
            ERROR
          </span>
          {summary}
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
      <span className="px-1.5 py-0.5 rounded bg-red-900 text-red-200 text-xs font-semibold">
        Runtime Error
      </span>
      {errorType && <span className="text-red-300 font-mono font-semibold">{errorType}</span>}
      {line != null && <span>Line {line}</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  if (viewState === 'concise') {
    return (
      <div>
        {headerLine}
        <div className="p-3 bg-gray-900 rounded border-l-4 border-red-600 mb-2">
          <pre className="text-sm text-gray-200 whitespace-pre-wrap break-words font-mono">
            {message || 'No message'}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div>
      {headerLine}

      <div className="p-3 bg-gray-900 rounded border-l-4 border-red-600 mb-2">
        <div className="text-xs text-gray-500 mb-1">Error Type</div>
        <div className="text-red-300 font-mono font-semibold">{errorType || 'Unknown'}</div>
      </div>

      <div className="p-3 bg-gray-900 rounded border-l-4 border-red-600 mb-2">
        <div className="text-xs text-gray-500 mb-1">Message</div>
        <pre className="text-sm text-gray-200 whitespace-pre-wrap break-words font-mono">
          {message || 'No message'}
        </pre>
      </div>

      {codeLines && codeLines.length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Code Context</div>
          <CodeBox code={codeLines.join('\n')} language="python" />
        </div>
      )}

      {traceback && (
        <div className="p-3 bg-gray-900 rounded border-l-4 border-red-800 mb-2">
          <div className="text-xs text-gray-500 mb-1">Traceback</div>
          <pre className="text-xs text-red-300 whitespace-pre-wrap break-words font-mono leading-relaxed">
            {traceback}
          </pre>
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
