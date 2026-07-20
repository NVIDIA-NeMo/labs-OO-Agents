import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';
import { ContextBlockRenderer } from '@/components/shared/ContextBlockRenderer';

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

const INTERNAL_KEYS = new Set([
  'span_name',
  'start_time_ns',
  'end_time_ns',
  'duration_ns',
  'status_code',
  'status_description',
]);

const NOISE_PREFIXES = ['git.', 'python.', 'hostname', 'nemo_oo_agents.version'];

function isNoiseKey(key: string): boolean {
  return NOISE_PREFIXES.some((p) => key.startsWith(p));
}

function findHeroContent(attrs: Record<string, unknown>): {
  attrKey: string;
  label: string;
  content: string;
  language: string;
} | null {
  // Known large-text attributes worth showing as a code box
  const candidates: [string, string, string][] = [
    ['nemo_oo_agents.system_message', 'System Message', 'markdown'],
    ['nemo_oo_agents.user_message', 'User Message', 'markdown'],
    ['code', 'Code', 'python'],
    ['result', 'Result', 'json'],
    ['message', 'Message', 'markdown'],
    // OI-first fallbacks: for OI-only traces the native attrs above
    // are absent, so render the OpenInference-standard I/O values instead.
    ['input.value', 'Input', 'markdown'],
    ['output.value', 'Output', 'json'],
  ];

  for (const [attrKey, label, lang] of candidates) {
    const val = attrs[attrKey];
    if (typeof val === 'string' && val.length > 0) {
      return { attrKey, label, content: val, language: lang };
    }
  }
  return null;
}

export function SpanPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const spanName = (attrs.span_name as string) || event.type.replace('span.', '');
  const durationNs = (attrs.duration_ns as number) || 0;
  const statusCode = (attrs.status_code as string) || 'UNSET';
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  const statusColor =
    statusCode === 'ERROR'
      ? 'text-red-400'
      : statusCode === 'OK'
        ? 'text-green-400'
        : 'text-gray-500';

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          SPAN
        </span>
        <span className="text-gray-300 font-mono">{spanName}</span>
        {durationNs > 0 && <span className="text-gray-500">{formatDuration(durationNs)}</span>}
        {statusCode !== 'UNSET' && statusCode !== 'OK' && (
          <span className={statusColor}>[{statusCode}]</span>
        )}
      </div>
    );
  }

  const hero = findHeroContent(attrs);

  const meaningfulAttrs = Object.entries(attrs).filter(
    ([k]) => !INTERNAL_KEYS.has(k) && !isNoiseKey(k),
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
            {spanName}
          </span>
          {durationNs > 0 && (
            <span className="text-gray-500 text-sm">{formatDuration(durationNs)}</span>
          )}
          {statusCode !== 'UNSET' && statusCode !== 'OK' && (
            <span className={`text-xs ${statusColor}`}>{statusCode}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-xs">{timestamp}</span>
          {viewControls}
        </div>
      </div>

      {hero && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">{hero.label}</div>
          {hero.attrKey === 'nemo_oo_agents.system_message' ? (
            <ContextBlockRenderer
              content={hero.content}
              plain={attrs['nemo_oo_agents.system_message.is_diff'] === true}
            />
          ) : (
            <CodeBox
              code={hero.content}
              language={hero.language}
              maxHeight={viewState === 'expanded' ? 'none' : '300px'}
            />
          )}
        </div>
      )}

      {viewState === 'expanded' && (
        <>
          {meaningfulAttrs.length > 0 && (
            <div className="p-3 bg-gray-900 rounded text-sm space-y-1 mb-2">
              {meaningfulAttrs.map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-gray-500 font-mono shrink-0 text-xs">{k}:</span>
                  <span className="text-gray-300 text-xs break-all">
                    {typeof v === 'object'
                      ? JSON.stringify(v)
                      : String(v).length > 200
                        ? String(v).substring(0, 200) + '...'
                        : String(v)}
                  </span>
                </div>
              ))}
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
