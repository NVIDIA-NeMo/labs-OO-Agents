import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

export function UserMessagePlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const message = event.body || (event.attributes?.message as string) || 'No message';
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    const preview = message.length > 80 ? message.substring(0, 80) + '...' : message;
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          User
        </span>
        <span className="text-gray-300 truncate">{preview}</span>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          User Message
        </span>
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-xs">{timestamp}</span>
          {viewControls}
        </div>
      </div>
      <div className="p-4 bg-gray-900 border-l-4 border-sky-600 rounded">
        <pre className="text-sm text-gray-200 whitespace-pre-wrap break-words font-mono leading-relaxed">
          {message}
        </pre>
      </div>
      {viewState === 'expanded' && (
        <details className="mt-2" open={rawJsonOpen}>
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
            Raw JSON
          </summary>
          <CodeBox code={JSON.stringify(event, null, 2)} language="json" className="mt-1" />
        </details>
      )}
    </div>
  );
}
