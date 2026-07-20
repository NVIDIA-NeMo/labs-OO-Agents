import { registerPlugin } from './registry';
import { UserMessagePlugin } from './UserMessagePlugin';
import { SpanPlugin } from './SpanPlugin';
import { LLMCallPlugin } from './LLMCallPlugin';
import { CodeExecutionPlugin } from './CodeExecutionPlugin';
import { MethodPlugin } from './MethodPlugin';
import { GenerationPlugin } from './GenerationPlugin';
import { EvalPlugin } from './EvalPlugin';
import { ToolExecutionPlugin } from './ToolExecutionPlugin';
import { RuntimeErrorPlugin } from './RuntimeErrorPlugin';
import { AgentMessagePlugin } from './AgentMessagePlugin';
import { AgentReasoningPlugin } from './AgentReasoningPlugin';

registerPlugin('user_message', UserMessagePlugin);
registerPlugin('user.message', UserMessagePlugin);

registerPlugin('span.llm_call', LLMCallPlugin);
registerPlugin('span.aresponses', LLMCallPlugin);
registerPlugin('span.acompletion', LLMCallPlugin);
registerPlugin('span.completion', LLMCallPlugin);
registerPlugin('span.responses', LLMCallPlugin);

registerPlugin('span.code_execution', CodeExecutionPlugin);

registerPlugin('span.generation', GenerationPlugin);

registerPlugin('span.eval', EvalPlugin);

registerPlugin('span.tool_execution', ToolExecutionPlugin);

registerPlugin('span.method', MethodPlugin);
registerPlugin('span.plan', MethodPlugin);
registerPlugin('span.method_call', MethodPlugin);

registerPlugin('runtime_error', RuntimeErrorPlugin);

registerPlugin('agent.message', AgentMessagePlugin);

registerPlugin('agent.reasoning', AgentReasoningPlugin);

// Fallback for any span.* type not matched above
registerPlugin('span.*', SpanPlugin);
