import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

export function AgentReasoningPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const reasoning = event.body || (attrs.reasoning as string) || '';
  const methodName = attrs.method_name as string | undefined;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    const preview = reasoning.length > 100 ? reasoning.substring(0, 100) + '...' : reasoning;
    return (
      <div className="flex items-center justify-between text-sm">
        <div className="flex-1 min-w-0 text-gray-400 truncate italic">
          <span className="px-1.5 py-0.5 rounded bg-yellow-900 text-yellow-200 text-xs font-semibold mr-1 not-italic">
            Reasoning
          </span>
          {preview}
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
      <span className="px-1.5 py-0.5 rounded bg-yellow-900 text-yellow-200 text-xs font-semibold">
        Agent Reasoning
      </span>
      {methodName && <span>{methodName}</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  return (
    <div>
      {headerLine}

      <div className="p-4 bg-[#1a1410] rounded border-l-4 border-yellow-600 mb-2">
        <pre className="text-sm text-gray-300 whitespace-pre-wrap break-words leading-relaxed italic">
          {reasoning}
        </pre>
      </div>

      {viewState === 'expanded' && (
        <>
          {methodName && (
            <div className="text-xs text-gray-500 px-3 py-2">
              Method: <span className="text-gray-300">{methodName}</span>
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
