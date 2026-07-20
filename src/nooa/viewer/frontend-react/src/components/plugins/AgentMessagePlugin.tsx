import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

export function AgentMessagePlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const message = event.body || (attrs.message as string) || '';
  const methodName = attrs.method_name as string | undefined;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    const preview = message.length > 80 ? message.substring(0, 80) + '...' : message;
    return (
      <div className="flex items-center justify-between text-sm">
        <div className="flex-1 min-w-0 text-gray-300 truncate">
          <span className="px-1.5 py-0.5 rounded bg-amber-900 text-amber-200 text-xs font-semibold mr-1">
            Agent
          </span>
          {preview}
          {methodName && <span className="text-gray-500 ml-2">({methodName})</span>}
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
      <span className="px-1.5 py-0.5 rounded bg-amber-900 text-amber-200 text-xs font-semibold">
        Agent Message
      </span>
      {methodName && <span>{methodName}</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  return (
    <div>
      {headerLine}

      <div className="p-3 bg-gray-900 rounded border-l-4 border-amber-600 mb-2">
        <CodeBox
          code={message}
          language="markdown"
          showLineNumbers={false}
          maxHeight={viewState === 'expanded' ? 'none' : '300px'}
        />
      </div>

      {viewState === 'expanded' && (
        <>
          {methodName && (
            <div className="p-3 bg-gray-900 rounded border-l-4 border-blue-500 mb-2">
              <div className="text-xs text-gray-500 mb-1">Metadata</div>
              <CodeBox
                code={JSON.stringify({ method_name: methodName }, null, 2)}
                language="json"
              />
            </div>
          )}
          <details className="mt-2" open={rawJsonOpen}>
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
              Raw JSON
            </summary>
            <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
          </details>
        </>
      )}
    </div>
  );
}
