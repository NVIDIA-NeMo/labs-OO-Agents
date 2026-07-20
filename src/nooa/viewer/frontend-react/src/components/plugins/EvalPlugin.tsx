import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';

interface Scorer {
  score?: number;
  passed?: boolean;
  reasoning?: string;
}

function extractScorers(attrs: Record<string, unknown>): Record<string, Scorer> {
  const scorers: Record<string, Scorer> = {};
  const pattern = /^eval\.scorer\.(.+)\.(score|passed|reasoning)$/;

  for (const [key, value] of Object.entries(attrs)) {
    const m = key.match(pattern);
    if (m) {
      const [, name, field] = m;
      if (!scorers[name]) scorers[name] = {};
      (scorers[name] as Record<string, unknown>)[field] = value;
    }
  }

  if (Object.keys(scorers).length === 0 && attrs['eval.scores']) {
    try {
      const blob =
        typeof attrs['eval.scores'] === 'string'
          ? JSON.parse(attrs['eval.scores'] as string)
          : attrs['eval.scores'];
      for (const [name, data] of Object.entries(blob as Record<string, Record<string, unknown>>)) {
        if (data && typeof data === 'object') {
          scorers[name] = {
            score: data.score as number,
            passed: data.passed as boolean,
            reasoning: (data.reasoning || data.reason) as string,
          };
        }
      }
    } catch {
      /* ignore */
    }
  }

  return scorers;
}

function formatScore(score: unknown): string {
  if (score === null || score === undefined) return '\u2014';
  const num = Number(score);
  if (!Number.isFinite(num)) return '\u2014';
  return `${(num * 100).toFixed(0)}%`;
}

function passLabel(passed: unknown): string {
  if (passed === true) return 'PASS';
  if (passed === false) return 'FAIL';
  return '?';
}

function scoreColor(score: unknown): string {
  const num = Number(score);
  if (!Number.isFinite(num)) return 'text-gray-400';
  if (num >= 0.8) return 'text-green-400';
  if (num >= 0.5) return 'text-orange-400';
  return 'text-red-400';
}

function scoreBorderColor(score: unknown): string {
  const num = Number(score);
  if (!Number.isFinite(num)) return 'border-gray-600';
  if (num >= 0.8) return 'border-green-600';
  if (num >= 0.5) return 'border-orange-600';
  return 'border-red-600';
}

function passBadge(passed: unknown): string {
  if (passed === true) return 'bg-green-900 text-green-200';
  if (passed === false) return 'bg-red-900 text-red-200';
  return 'bg-gray-800 text-gray-400';
}

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function EvalPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const attrs = event.attributes || {};
  const testId = (attrs['eval.test_id'] as string) || 'unknown';
  const model = (attrs['eval.model'] as string) || '';
  const agentClass = (attrs['eval.agent_class'] as string) || '';
  const method = (attrs['eval.method'] as string) || '';
  const passed = attrs['eval.passed'];
  const score = attrs['eval.weighted_score'] ?? attrs['eval.score'];
  const durationNs = (attrs.duration_ns as number) || 0;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center gap-3 text-xs">
        <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${passBadge(passed)}`}>
          {passLabel(passed)}
        </span>
        <span className="text-gray-300 font-mono truncate">{testId}</span>
        {model && <span className="text-gray-500">[{model}]</span>}
        <span className={`font-semibold ${scoreColor(score)}`}>{formatScore(score)}</span>
        {durationNs > 0 && <span className="text-gray-500">{formatDuration(durationNs)}</span>}
        <span className="ml-auto text-gray-600">{timestamp}</span>
      </div>
    );
  }

  const scorers = extractScorers(attrs);
  const scorerNames = Object.keys(scorers);

  const headerLine = (
    <div className="flex items-center gap-3 text-xs text-gray-400 mb-2">
      <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${passBadge(passed)}`}>
        EVAL
      </span>
      <span className={`font-semibold ${scoreColor(score)}`}>{formatScore(score)}</span>
      {model && <span className="text-purple-400">{model}</span>}
      {durationNs > 0 && <span>{formatDuration(durationNs)}</span>}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  if (viewState === 'concise') {
    return (
      <div>
        {headerLine}

        <div
          className={`grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 p-2.5 rounded border-l-4 ${scoreBorderColor(score)} bg-gray-950 text-xs font-mono mb-2`}
        >
          <span className="text-gray-500">Test ID:</span>
          <span className="text-orange-400">{testId}</span>
          {agentClass && (
            <>
              <span className="text-gray-500">Agent:</span>
              <span className="text-green-400">{agentClass}</span>
            </>
          )}
          {method && (
            <>
              <span className="text-gray-500">Method:</span>
              <span className="text-green-400">{method}</span>
            </>
          )}
        </div>

        {scorerNames.length > 0 && (
          <div className="mt-2">
            <div className="text-xs text-gray-500 mb-1">Scorers ({scorerNames.length})</div>
            <div className="bg-gray-950 rounded overflow-hidden">
              {scorerNames.map((name, idx) => {
                const s = scorers[name];
                return (
                  <div
                    key={name}
                    className={`flex items-center gap-3 px-3 py-1.5 text-xs ${
                      idx < scorerNames.length - 1 ? 'border-b border-gray-800' : ''
                    }`}
                  >
                    <span
                      className={`px-1 py-0.5 rounded text-[10px] font-semibold ${passBadge(s.passed)}`}
                    >
                      {passLabel(s.passed)}
                    </span>
                    <span className="text-orange-400 font-semibold min-w-[80px]">{name}</span>
                    <span className={`font-semibold ${scoreColor(s.score)}`}>
                      {formatScore(s.score)}
                    </span>
                    {s.reasoning && (
                      <span className="text-gray-500 truncate flex-1">{s.reasoning}</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <OutputComparison attrs={attrs} passed={passed} />
      </div>
    );
  }

  // Expanded
  const testInfo: Record<string, string> = {
    'Test ID': testId,
  };
  if (agentClass) testInfo['Agent Class'] = agentClass;
  if (method) testInfo['Method'] = method;
  if (model) testInfo['Model'] = model;
  testInfo['Overall Passed'] = passed === true ? 'Yes' : passed === false ? 'No' : 'Unknown';
  testInfo['Weighted Score'] = formatScore(score);
  if (durationNs > 0) testInfo['Duration'] = formatDuration(durationNs);

  const metaData: Record<string, unknown> = {};
  const skipKeys = new Set([
    'eval.test_id',
    'eval.agent_class',
    'eval.method',
    'eval.model',
    'eval.passed',
    'eval.weighted_score',
    'eval.score',
    'eval.scores',
  ]);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith('eval.scorer.')) continue;
    if (skipKeys.has(key)) continue;
    if (key.startsWith('eval.')) metaData[key] = value;
  }

  return (
    <div>
      {headerLine}

      <div className="text-xs text-gray-500 mb-1">Test Info</div>
      <CodeBox
        code={JSON.stringify(testInfo, null, 2)}
        language="json"
        showLineNumbers={false}
        maxHeight="none"
        className="mb-2"
      />

      {scorerNames.length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Scorer Results ({scorerNames.length})</div>
          {scorerNames.map((name) => {
            const s = scorers[name];
            return (
              <div
                key={name}
                className={`rounded border-l-4 ${scoreBorderColor(s.score)} bg-gray-950 mb-2 overflow-hidden`}
              >
                <div className="flex items-center gap-3 px-3 py-2 bg-gray-900 border-b border-gray-800 text-xs">
                  <span
                    className={`px-1 py-0.5 rounded text-[10px] font-semibold ${passBadge(s.passed)}`}
                  >
                    {passLabel(s.passed)}
                  </span>
                  <span className="text-orange-400 font-semibold">{name}</span>
                  <span className={`font-semibold ${scoreColor(s.score)}`}>
                    {formatScore(s.score)}
                  </span>
                </div>
                {s.reasoning && (
                  <div className="px-3 py-2 text-xs font-mono text-gray-200 whitespace-pre-wrap break-words">
                    {s.reasoning}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <OutputComparison attrs={attrs} passed={passed} />

      {Object.keys(metaData).length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Additional Metadata</div>
          <CodeBox
            code={JSON.stringify(metaData, null, 2)}
            language="json"
            showLineNumbers={false}
            maxHeight="none"
          />
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

function OutputComparison({ attrs, passed }: { attrs: Record<string, unknown>; passed: unknown }) {
  let expected = (attrs['eval.expected_output'] as string) ?? (attrs['eval.expected'] as string);
  let actual = (attrs['eval.actual_output'] as string) ?? (attrs['eval.output'] as string);

  if (expected !== undefined && typeof expected === 'string') {
    try {
      expected = JSON.stringify(JSON.parse(expected), null, 2);
    } catch {
      /* keep as-is */
    }
  }
  if (actual !== undefined && typeof actual === 'string') {
    try {
      actual = JSON.stringify(JSON.parse(actual), null, 2);
    } catch {
      /* keep as-is */
    }
  }

  if (expected === undefined && actual === undefined) return null;

  const isPassed = passed === true;
  const borderCls = isPassed ? 'border-green-600' : 'border-red-600';

  return (
    <div className="mt-3">
      <div className="text-xs text-gray-500 mb-1">{isPassed ? 'Output' : 'Output Comparison'}</div>
      {expected !== undefined && (
        <div className="p-2.5 bg-gray-950 border-l-4 border-green-600 rounded mb-2 text-xs font-mono">
          <div className="text-green-400 font-semibold mb-1">Expected</div>
          <div className="text-gray-200 whitespace-pre-wrap break-words">{String(expected)}</div>
        </div>
      )}
      {actual !== undefined && (
        <div className={`p-2.5 bg-gray-950 border-l-4 ${borderCls} rounded text-xs font-mono`}>
          <div className={`font-semibold mb-1 ${isPassed ? 'text-green-400' : 'text-red-400'}`}>
            Actual
          </div>
          <div className="text-gray-200 whitespace-pre-wrap break-words">{String(actual)}</div>
        </div>
      )}
    </div>
  );
}
