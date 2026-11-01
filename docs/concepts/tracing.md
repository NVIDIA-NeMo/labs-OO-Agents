# Tracing

NOOA traces the program's call tree, not only the messages sent to an LLM. A
trace can connect a Python orchestrator to nested agentic methods, model calls,
generated-code cells, helper calls, and external tools.

## Start with the development viewer

From a repository checkout:

```bash
uv run nooa start-dev
```

The viewer listens on port 5001 by default. When an agent starts and the viewer
is reachable, tracing is enabled automatically.

Agents that do not configure tracing explicitly send traces to the reachable
viewer automatically. The tracing quickstart intentionally configures a durable
journal exporter instead, so run it in another terminal and then import its
output:

```bash
uv run python examples/quickstart/06_tracing.py
uv run nooa import-traces traces/quickstart-06-journal
```

Refresh the viewer after the import. Calling `enable_tracing(...)` replaces the
automatic exporter configuration; it does not send to both destinations unless
both exporters are supplied. Automatic tracing is a development convenience.
If the viewer is not reachable, there is no automatic file fallback.

## Write durable traces explicitly

For tests, benchmarks, and unattended runs, configure a file exporter:

```python
from nooa.tracing import enable_tracing, exporters, flush_traces

enable_tracing(exporters=[exporters.jsonl("traces")])

# Run the agent.

flush_traces()
```

JSONL traces can be archived, imported into the viewer, or inspected by other
OpenTelemetry-compatible tooling. Set a stable session ID when an evaluation
harness must associate files with a particular task.

### Journal calls from clients that bypass LiteLLM

`exporters.journal_file(...)` and the live viewer's journal record each model
call's messages through LiteLLM callbacks. A `UnifiedLLM` subclass that talks to
its model server directly (for example Gym's policy client) records its calls
with `journal_llm_call` once the response arrives:

```python
from nooa.tracing import journal_llm_call

response = LLMResponse(parts=parts, usage=usage, model_name="gym-policy")
journal_llm_call(
    model="gym-policy",
    messages=messages,             # the list acall() received, as is
    response=response,             # None records a failed call's prompt
    latency_ms=elapsed_ms,
    extra={"response_id": raw.id}, # stored on the call record
)
```

It writes to every installed journal sink (journal file, live viewer) and does
nothing without one. The session defaults to the current trace session and the
call is linked to the current span. Inside a NOOA turn it uses the context the
runtime rendered for the call, so the record matches the LiteLLM path. It never
raises.

Use `exporters.journal_file` by itself for such runs. It keeps every span
(LLM spans lose their message attributes) and adds the journal. Don't add
`exporters.jsonl` next to it. The viewer doesn't dedupe span ids: when both
copies end up in one imported file, every span shows up twice. When they stay
in separate files, `nooa import-traces` imports `<session>.jsonl` and then
skips `<session>.nooa.jsonl` because the session already exists, so the
journal is lost.

## What appears in a trace

All agent methods are traced by default, including regular Python methods and
private methods. Common span types include:

| Span | Meaning |
|---|---|
| `method.<name>` | One agent method call |
| `generation` | One strategy execution |
| `litellm.acompletion` | One provider call |
| `code_execution` | One CodeAct Python cell |
| `method_call.<name>` | Generated code calling another agent method |
| `tool_execution.<tool>` | An external tool invocation |

Parent-child nesting follows the Python call hierarchy. That makes a
deterministic outer workflow just as important to observability as its LLM
steps.

## Events and traces are different views

Events are the agent's working history and may be shown to later model turns.
Traces are the operational record of how the program executed. A trace includes
timing and nesting that the agent does not need in its prompt.

Use events for conversational memory and agent-visible feedback. Use traces for
debugging latency, retries, tool calls, generated code, and control flow.

## Keep important work inside the trace boundary

Logic inside agent methods becomes part of the nested call tree. Significant
preprocessing performed in a module-level `main()` helper does not. Put
meaningful workflow steps in agent methods or explicitly instrument the
application layer when you need them represented.

You can exclude a noisy method without changing its behavior:

```python
from nooa import no_trace


class AgentWithUtility(Agent, llm=llm):
    @no_trace
    def frequent_utility(self) -> int:
        return 1
```

## A practical debugging loop

1. Inspect the rendered prompt for missing or duplicated information.
2. Inspect the method span and its child generations.
3. Check validation errors and model retries.
4. Open generated-code cells and their stdout or stderr.
5. Follow nested method and tool spans to the first incorrect boundary.

This is usually more informative than expanding the prompt and trying again.

## Continue

- [Architecture](../architecture.md)
- [Orchestration](orchestration.md)
- Runnable example: [tracing](../../examples/quickstart/06_tracing.py)
- Detailed reference: [`nooa-capturing-traces`](../../skills/nooa-capturing-traces/SKILL.md)
