import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

export function DefaultPlugin({ event, viewState, viewControls }: PluginProps) {
  const badge = event.type;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          {badge}
        </span>
        <span className="text-gray-500">{timestamp}</span>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          {badge}
        </span>
        <span className="text-gray-500 text-xs">{timestamp}</span>
        {viewControls}
      </div>
      {viewState === 'expanded' && (
        <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
      )}
    </div>
  );
}
