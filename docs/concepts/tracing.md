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

## Export recorded token usage

Use the existing `trace-explorer` command to export a completed-call usage series
from a local OpenInference OTLP JSONL file or portable `.nooa.jsonl` journal file:

```bash
uv run trace-explorer traces/my-run.jsonl --usage --until-seconds 7200 > usage.json
uv run trace-explorer traces/my-run.jsonl --usage --usage-format csv --until-seconds 7200 > usage.csv
uv run trace-explorer traces/my-run.nooa.jsonl --usage --until-seconds 7200 > journal-usage.json
```

`--usage` defaults to JSON. Omit `--until-seconds` to include all completed calls.
Time zero is the start of the first recorded LLM call, not the program launch.
Earlier non-LLM spans, including orchestrators or spans with skewed timestamps,
do not move that origin. With no LLM spans, the earliest positive span start
anchors the zero-call summary. Recorded duration extends from the origin to
the latest recorded span timestamp.
Completion at the cutoff is inclusive. All traces in the file participate;
session/root-generation selection and other Explorer views cannot be combined
with this mode.

The Python API returns the same data without building a session tree:

```python
from nooa.trace_explorer import load_usage_report

report = load_usage_report("traces/my-run.jsonl", until_seconds=7200)
print(report.totals.known_total_tokens)
print(report.totals.missing_usage_calls)
print(report.by_model)
data = report.to_dict()   # schema_version=1; summary + chronological timeline
csv_text = report.to_csv()
```

Each timeline point preserves nullable input/output/total/cache-read counts and
a cumulative snapshot. `known_*` aggregates sum available counts, not estimates
for missing fields. `missing_usage_calls` counts completed calls missing any
input/output/total count; absent cache telemetry has its own counter. A missing
total is derived only when both input and output are present. A provider-reported
total takes precedence, so known input plus known output need not equal the
known total when telemetry is partial or inconsistent. Cache/reasoning subsets
are never added again. Recorded zeros are preserved; this report cannot reverse
normalization already performed by an upstream collector. Only `LLM`-kind spans
count; parent rollups do not.
Identical accounting for a repeated trace/span ID counts once; conflicts raise.

Per-model summaries (`by_model`) are available in JSON and the Python API only;
CSV retains each call's model name but does not emit per-model summary rows.
CSV contains `call` rows with cumulative known totals and a final `summary` row
even when no calls completed. That row carries the time origin, recorded duration,
missing-data counts and `in_flight_calls`. Unknown per-call values are empty
fields. Text fields starting with `=`, `+`, `-`, `@`, `|`, or full-width `＝`
(including after whitespace), or starting with a tab or line break, are
apostrophe-prefixed for spreadsheet safety. JSON preserves their original text.

`in_flight_calls` counts recorded calls crossing the cutoff or lacking an end
timestamp. It does **not** estimate tokens consumed partway through those calls.
Exporters may not record running calls at all; absent/unexported calls cannot be
detected. A cutoff beyond the recorded duration does not establish observation
coverage up to that time. This is recorded usage, not a provider bill, answer
quality measurement, or hard spending limit.

Accounting mode rejects malformed/truncated records and empty captures with a
non-zero CLI exit, rather than silently skipping data. Ordinary Explorer loading
remains unchanged. For `exporters.journal_file(...)` captures, accounting reads
the ordinary OTLP span records and skips `nooaJournal` manifest, block and call
records. Journal `call.tokens` are never counted again or used to fill missing
span usage. Message reconstruction remains separate work in #205. A journal-only
capture without OTLP spans fails clearly instead of reporting zero usage.
The reader loads spans into memory; this is not a streaming analyzer for
arbitrarily large files.

## Journal calls from clients that bypass LiteLLM

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
nothing without one. The session defaults to the current trace session. Each
call gets an `LLM` span of its own (`llm_call`) under the current span, the way
the LiteLLM instrumentor nests `litellm.acompletion` under `generation`; the
viewer shows journal messages only on `LLM` spans, one call per span. A client
that already runs inside its own `LLM` span is linked to that span, and
`span_id=` links the call to any span. Inside a NOOA turn it uses the context
the runtime rendered for the call, so the record matches the LiteLLM path. It
never raises.

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
