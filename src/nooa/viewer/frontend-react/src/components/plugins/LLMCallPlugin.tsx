import type { PluginProps } from './registry';
import { CodeBox } from '@/components/shared/CodeBox';
import { ContextBlockRenderer } from '@/components/shared/ContextBlockRenderer';
import { usePlayground } from '@/components/playground/PlaygroundContext';

function formatDuration(ns: number): string {
  if (ns <= 0) return '';
  const ms = ns / 1e6;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

interface ToolCall {
  name: string;
  arguments: Record<string, unknown> | string;
  id: string | null;
}

interface Message {
  role: string;
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

function extractInputMessages(attrs: Record<string, unknown>): Message[] {
  // Method 1: OpenInference flattened format (most common)
  const messages: Message[] = [];
  let i = 0;
  while (attrs[`llm.input_messages.${i}.message.role`]) {
    const msg: Message = {
      role: attrs[`llm.input_messages.${i}.message.role`] as string,
      content:
        (attrs[`llm.input_messages.${i}.message.content`] as string) ||
        (attrs[`llm.input_messages.${i}.message.contents.0.message_content.text`] as string) ||
        null,
    };

    const toolCalls: ToolCall[] = [];
    let j = 0;
    while (attrs[`llm.input_messages.${i}.message.tool_calls.${j}.tool_call.function.name`]) {
      const name = attrs[
        `llm.input_messages.${i}.message.tool_calls.${j}.tool_call.function.name`
      ] as string;
      const argsStr =
        (attrs[
          `llm.input_messages.${i}.message.tool_calls.${j}.tool_call.function.arguments`
        ] as string) || '{}';
      const id =
        (attrs[`llm.input_messages.${i}.message.tool_calls.${j}.tool_call.id`] as string) || null;
      let args: Record<string, unknown> | string;
      try {
        args = JSON.parse(argsStr);
      } catch {
        args = argsStr;
      }
      toolCalls.push({ name, arguments: args, id });
      j++;
    }
    if (toolCalls.length > 0) msg.tool_calls = toolCalls;

    const toolCallId = attrs[`llm.input_messages.${i}.message.tool_call_id`] as string | undefined;
    if (toolCallId) msg.tool_call_id = toolCallId;

    messages.push(msg);
    i++;
  }
  if (messages.length > 0) return messages;

  // Method 2: input.value (OpenInference nested format)
  if (attrs['input.value']) {
    try {
      let parsed = attrs['input.value'];
      if (typeof parsed === 'string') parsed = JSON.parse(parsed);
      if (parsed && typeof parsed === 'object' && 'messages' in (parsed as Record<string, unknown>)) {
        const msgs = (parsed as Record<string, unknown>).messages;
        if (Array.isArray(msgs)) return msgs as Message[];
      }
      if (Array.isArray(parsed) && parsed.length > 0 && parsed[0]?.role) {
        return parsed as Message[];
      }
    } catch { /* not parseable */ }
  }

  // Method 3: llm.input_messages as JSON string or array
  if (attrs['llm.input_messages']) {
    try {
      let val = attrs['llm.input_messages'];
      if (typeof val === 'string') val = JSON.parse(val);
      if (Array.isArray(val)) return val as Message[];
    } catch { /* not parseable */ }
  }

  return [];
}

function extractOutputToolCalls(attrs: Record<string, unknown>): ToolCall[] {
  const toolCalls: ToolCall[] = [];
  let i = 0;
  while (attrs[`llm.output_messages.0.message.tool_calls.${i}.tool_call.function.name`]) {
    const name = attrs[
      `llm.output_messages.0.message.tool_calls.${i}.tool_call.function.name`
    ] as string;
    const argsStr =
      (attrs[
        `llm.output_messages.0.message.tool_calls.${i}.tool_call.function.arguments`
      ] as string) || '{}';
    const id =
      (attrs[`llm.output_messages.0.message.tool_calls.${i}.tool_call.id`] as string) || null;
    let args: Record<string, unknown> | string;
    try {
      args = JSON.parse(argsStr);
    } catch {
      args = argsStr;
    }
    toolCalls.push({ name, arguments: args, id });
    i++;
  }
  return toolCalls;
}

function extractOutput(attrs: Record<string, unknown>): string | null {
  if (attrs['llm.output.content']) {
    return attrs['llm.output.content'] as string;
  }

  if (attrs['output.value']) {
    try {
      const raw = attrs['output.value'] as string;
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      if (Array.isArray(parsed) && parsed[0]?.contents) {
        const text = parsed[0].contents
          .filter((c: { text?: string }) => c.text)
          .map((c: { text: string }) => c.text)
          .join('\n');
        if (text) return text;
      }
    } catch {
      // not JSON — use as plain string below
    }
    const val = attrs['output.value'];
    if (typeof val === 'string' && val.trim()) {
      return val;
    }
  }

  if (attrs['llm.output_messages.0.message.content']) {
    return attrs['llm.output_messages.0.message.content'] as string;
  }

  if (attrs['llm.output_messages.0.message.contents.0.message_content.text']) {
    return attrs['llm.output_messages.0.message.contents.0.message_content.text'] as string;
  }

  return null;
}

function formatToolCallContent(tc: ToolCall): { content: string; lang: string } {
  if (
    tc.name === 'execute_python' &&
    typeof tc.arguments === 'object' &&
    tc.arguments !== null &&
    'code' in tc.arguments
  ) {
    return { content: String(tc.arguments.code).trim(), lang: 'python' };
  }
  return {
    content:
      typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments, null, 2),
    lang: 'json',
  };
}

function MessageBox({
  role,
  content,
  toolCallId,
}: {
  role: string;
  content: string;
  toolCallId?: string;
}) {
  const borderColor =
    role === 'system'
      ? 'border-gray-600'
      : role === 'assistant'
        ? 'border-indigo-700'
        : role === 'tool'
          ? 'border-amber-700'
          : 'border-sky-700';

  let label = role;
  if (toolCallId) label += ` [${toolCallId.slice(-8)}]`;

  return (
    <div className={`p-3 bg-gray-900 rounded border-l-4 ${borderColor} mb-2`}>
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      {role === 'system' || role === 'user' ? (
        <ContextBlockRenderer content={content} />
      ) : (
        <CodeBox code={content} language="markdown" showLineNumbers={false} />
      )}
    </div>
  );
}

function ToolCallBox({ tc, index, total }: { tc: ToolCall; index: number; total: number }) {
  const { content, lang } = formatToolCallContent(tc);
  const idSuffix = tc.id ? ` [${tc.id.slice(-8)}]` : '';
  const label =
    total > 1
      ? `Tool Call ${index + 1}/${total}: ${tc.name}${idSuffix}`
      : `Tool Call: ${tc.name}${idSuffix}`;

  return (
    <details className="rounded border-l-4 border-amber-600 bg-gray-900 mb-2">
      <summary className="px-3 py-2 text-xs text-gray-500 cursor-pointer hover:text-gray-300 list-none">
        {label}
      </summary>
      <div className="px-3 pb-3">
        <CodeBox code={content} language={lang} />
      </div>
    </details>
  );
}

function extractReasoningContent(attrs: Record<string, unknown>): string | null {
  if (attrs['llm.reasoning_content']) {
    return attrs['llm.reasoning_content'] as string;
  }

  if (attrs['output.value']) {
    try {
      const outputValue =
        typeof attrs['output.value'] === 'string'
          ? JSON.parse(attrs['output.value'] as string)
          : attrs['output.value'];
      if (outputValue?.reasoning_content) {
        return outputValue.reasoning_content as string;
      }
    } catch {
      // not JSON
    }
  }

  return null;
}

function ReasoningSection({ content, defaultOpen }: { content: string; defaultOpen?: boolean }) {
  const lineCount = content.split('\n').length;
  return (
    <details
      className="mt-2 border border-purple-700 rounded-md overflow-hidden"
      open={defaultOpen}
    >
      <summary className="px-3 py-2 bg-purple-900/20 cursor-pointer flex items-center gap-2 text-xs font-semibold text-purple-300">
        <span>Reasoning</span>
        <span className="font-normal opacity-70">({lineCount} lines)</span>
      </summary>
      <div className="p-3 bg-[#1a1625] max-h-[300px] overflow-auto">
        <pre className="m-0 whitespace-pre-wrap break-words text-xs leading-relaxed text-gray-200">
          {content}
        </pre>
      </div>
    </details>
  );
}

export function LLMCallPlugin({ event, viewState, rawJsonOpen, viewControls }: PluginProps) {
  const openPlayground = usePlayground();
  const attrs = event.attributes || {};
  const model = (attrs['llm.model_name'] ??
    attrs['gen_ai.response.model'] ??
    attrs['llm.model'] ??
    attrs.model ??
    'unknown') as string;
  const shortModel = model.length > 35 ? model.substring(0, 32) + '...' : model;
  const durationNs = (attrs.duration_ns as number) || 0;
  const statusCode = (attrs.status_code as string) || 'UNSET';
  const promptTokens = (attrs['llm.token_count.prompt'] as number) || 0;
  const completionTokens = (attrs['llm.token_count.completion'] as number) || 0;
  const totalTokens = (attrs['llm.token_count.total'] as number) || promptTokens + completionTokens;
  const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
  const reasoningContent = extractReasoningContent(attrs);

  if (viewState === 'collapsed') {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
          LLM
        </span>
        <span className="text-gray-300 font-mono">{shortModel}</span>
        {reasoningContent && <span className="text-purple-300 text-xs">[reasoning]</span>}
        {totalTokens > 0 && <span className="text-gray-500">({totalTokens} tokens)</span>}
        {durationNs > 0 && <span className="text-gray-500">{formatDuration(durationNs)}</span>}
        {statusCode === 'ERROR' && <span className="text-red-400">ERROR</span>}
      </div>
    );
  }

  const inputMessages = extractInputMessages(attrs);
  const outputText = extractOutput(attrs);
  const outputToolCalls = extractOutputToolCalls(attrs);

  const handleOpenPlayground = openPlayground
    ? (e: React.MouseEvent) => {
        e.stopPropagation();
        const inputMsgs = extractInputMessages(attrs);
        const output = extractOutput(attrs);
        openPlayground({
          messages: inputMsgs,
          originalOutput: output,
          model: model !== "unknown" ? model : null,
        });
      }
    : undefined;

  const headerLine = (
    <div className="flex items-center gap-3 text-xs text-gray-400 mb-2">
      <span className="px-1.5 py-0.5 rounded bg-purple-900 text-purple-200 text-xs font-semibold">
        LLM Call
      </span>
      <span className="text-gray-300 font-mono">{shortModel}</span>
      {promptTokens > 0 && <span>in: {promptTokens}</span>}
      {completionTokens > 0 && <span>out: {completionTokens}</span>}
      {durationNs > 0 && <span>{formatDuration(durationNs)}</span>}
      {handleOpenPlayground && (
        <button
          onClick={handleOpenPlayground}
          className="text-gray-600 hover:text-blue-400 transition-colors"
          title="Open in Playground"
        >
          Sandbox
        </button>
      )}
      <span className="ml-auto opacity-60">{timestamp}</span>
      {viewControls}
    </div>
  );

  if (viewState === 'concise') {
    // Show last input message + output + tool calls
    // Separate <context> wrapper messages (from cached block formatter) into a collapsed section
    const isContextMsg = (m: Message): boolean => m.role === 'user' && (m.content?.trimStart().startsWith('<context>\n') ?? false);
    const contextMsgs = inputMessages.filter(isContextMsg);
    const lastMsg = inputMessages.length > 0
      ? inputMessages.findLast(m => !isContextMsg(m)) ?? null
      : null;

    return (
      <div>
        {headerLine}

        {lastMsg && lastMsg.content && (
          <MessageBox
            role={lastMsg.role}
            content={lastMsg.content}
            toolCallId={lastMsg.tool_call_id}
          />
        )}

        {lastMsg?.tool_calls?.map((tc, i) => (
          <ToolCallBox key={i} tc={tc} index={i} total={lastMsg.tool_calls!.length} />
        ))}

        {contextMsgs.length > 0 && (
          <details className="mb-2 rounded border border-gray-700 overflow-hidden">
            <summary className="px-3 py-1.5 text-xs text-gray-500 cursor-pointer hover:text-gray-300 bg-gray-800/40">
              Context ({contextMsgs.length} {contextMsgs.length === 1 ? 'block' : 'blocks'})
            </summary>
            <div className="p-2">
              {contextMsgs.map((msg, i) => (
                <MessageBox key={`ctx-${i}`} role={msg.role} content={msg.content!} />
              ))}
            </div>
          </details>
        )}

        {(outputText || outputToolCalls.length > 0) && (
          <div>
            <div className="text-xs text-gray-500 mb-1">Output</div>
            {outputText && (
              <div className="p-3 bg-gray-900 rounded border-l-4 border-indigo-700 mb-2">
                <CodeBox
                  code={outputText}
                  language="markdown"
                  showLineNumbers={false}
                />
              </div>
            )}
            {outputToolCalls.map((tc, i) => (
              <ToolCallBox key={`out-${i}`} tc={tc} index={i} total={outputToolCalls.length} />
            ))}
          </div>
        )}

        {reasoningContent && <ReasoningSection content={reasoningContent} />}
      </div>
    );
  }

  // Expanded: show all input messages + output + tool calls + reasoning + raw JSON
  return (
    <div>
      {headerLine}

      {inputMessages.length > 0 && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Input Messages ({inputMessages.length})</div>
          {inputMessages.map((msg, i) => (
            <div key={i}>
              {msg.content && (
                <MessageBox role={msg.role} content={msg.content} toolCallId={msg.tool_call_id} />
              )}
              {msg.tool_calls?.map((tc, j) => (
                <ToolCallBox
                  key={`in-${i}-${j}`}
                  tc={tc}
                  index={j}
                  total={msg.tool_calls!.length}
                />
              ))}
            </div>
          ))}
        </div>
      )}

      {(outputText || outputToolCalls.length > 0) && (
        <div className="mb-2">
          <div className="text-xs text-gray-500 mb-1">Output</div>
          {outputText && (
            <div className="p-3 bg-gray-900 rounded border-l-4 border-indigo-700 mb-2">
              <CodeBox
                code={outputText}
                language="markdown"
                showLineNumbers={false}
                maxHeight="none"
              />
            </div>
          )}
          {outputToolCalls.map((tc, i) => (
            <ToolCallBox key={`out-${i}`} tc={tc} index={i} total={outputToolCalls.length} />
          ))}
        </div>
      )}

      {reasoningContent && <ReasoningSection content={reasoningContent} defaultOpen />}

      <details className="mt-2" open={rawJsonOpen}>
        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
          Raw JSON
        </summary>
        <CodeBox code={JSON.stringify(event, null, 2)} language="json" />
      </details>
    </div>
  );
}
