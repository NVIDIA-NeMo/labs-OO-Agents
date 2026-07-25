---
name: nooa-capturing-traces
description: Capture execution traces from NVIDIA OO Agents. Use when instrumenting an agent run, writing traces to JSONL files, sending traces to the viewer or an OTLP/Langfuse/Phoenix backend, controlling which methods are traced, or when traces are mysteriously missing.
compatibility: nooa package; the [tracing] extra (opentelemetry + openinference) for exporters
---

# Capturing Traces

NVIDIA OO Agents traces every agent method call, LLM call, code execution, and tool invocation as OpenTelemetry spans using OpenInference semantic conventions. Traces are grouped by `session.id` and nested by call hierarchy.

## Automatic tracing (zero config)

Every `Agent.__init__()` auto-attempts tracing **once per process**: it probes the trace viewer at `http://localhost:5001` (or `$OTLP_ENDPOINT`) and, if reachable, streams spans to it. Nothing to import, nothing to call:

```bash
nooa start-dev            # terminal 1: viewer + OTLP receiver on :5001
uv run python my_agent.py    # terminal 2: traces appear automatically
```

**If the viewer is not reachable, tracing is silently disabled.** There is NO automatic fallback to files. The only exception: if you explicitly set `OTLP_ENDPOINT` and it's unreachable, a warning is printed to stderr. If you need traces without a viewer, use explicit file export (below).

## Explicit tracing

```python
from nooa.tracing import enable_tracing, exporters, flush_traces

# Write JSONL files — one file per session: {trace_dir}/{session_id}.jsonl
enable_tracing(exporters=[exporters.jsonl("./traces")])

# ... run the agent ...
flush_traces()   # force-flush pending spans (e.g. before process exit)
```

Signature: `enable_tracing(exporters=None, *, experiment=None, extra_resource_attrs=None) -> None`
(source: `src/nooa/tracing/__init__.py`). It is idempotent for no-arg calls; calling again with explicit `exporters` replaces the previous exporters. `experiment` tags every span's resource with an experiment name (defaults to `$TRACE_EXPERIMENT`), which the viewer uses to group eval runs.

### Exporter factories (`nooa.tracing.exporters`)

| Factory | Destination | Notes |
|---|---|---|
| `exporters.jsonl(trace_dir=None)` | `{trace_dir}/{session_id}.jsonl` | dir from arg → `$TRACE_DIR` → `./traces/` |
| `exporters.journal(endpoint=None)` | viewer (delta journal) | the default used by auto-tracing; most efficient for the viewer |
| `exporters.local_otlp(endpoint=None)` | OTLP JSON over HTTP | lightweight urllib POST to `$OTLP_ENDPOINT` |
| `exporters.otlp(endpoint, headers=None)` | real OTLP/HTTP collector | needs `opentelemetry-exporter-otlp-proto-http` |
| `exporters.langfuse(host=None, ...)` | Langfuse | reads `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY` |
| `exporters.console()` | stdout | quick debugging |

Multiple destinations at once:

```python
enable_tracing(exporters=[
    exporters.jsonl("./traces"),
    exporters.journal(),          # viewer too, if running
])
```

### Session IDs

Spans are grouped by `session.id`. Set it explicitly when you need a stable, known ID (e.g. eval harnesses):

```python
from nooa.tracing import set_session, get_session
set_session("my-run-001")   # call before running the agent
```

## What gets traced

**All agent methods are traced by default** — public, private (`_x`), and async dunders. Generation methods, plain-Python orchestrators, and deterministic helpers all produce spans. Opt out per-method:

```python
from nooa import no_trace

class MyAgent(Agent, llm=llm):
    @no_trace
    async def utility(self):
        """Runs (and generates) normally, but produces no span."""
        ...
```

(Older docs claiming "private methods are not traced" are outdated — trust this table.)

| Span name | Kind | Meaning |
|---|---|---|
| `method.{name}` | AGENT | an agent method call; carries args as `input.value`, docstring, signature |
| `generation` | CHAIN | one strategy execution (an LLM "thinking" episode) |
| `litellm.acompletion` | LLM | the actual LLM call (from openinference-litellm), nested under `generation`; carries `llm.input_messages`/`llm.output_messages`/`llm.model_name` |
| `code_execution` | TOOL | one CodeAct `execute_python` cell |
| `method_call.{name}` | TOOL | LLM-generated code calling a method on `self` |
| `tool_execution.{tool}` | TOOL | external tool invocation |

Parent-child nesting follows the call hierarchy: orchestrator → generation methods → generations → LLM calls / code executions.

## Trace file format

Files are OTLP JSON Lines: each line is one `{"resourceSpans": [...]}` object. This is the interchange format for the whole toolchain — the viewer imports it (`nooa import-traces ./traces`) and the trace explorer reads it directly (`trace-explorer ./traces/<session_id>.jsonl`).

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `OTLP_ENDPOINT` | where auto-tracing / `local_otlp` / `journal` send spans | `http://localhost:5001/v1/traces` |
| `OTLP_PROBE_TIMEOUT` | viewer reachability probe timeout (seconds) | `2.0` |
| `TRACE_DIR` | default dir for `exporters.jsonl()` | `./traces` |
| `TRACE_EXPERIMENT` | default experiment resource attribute | unset |

## Pitfalls

- **Do NOT use** legacy OpenInference instrumentation imports or `enable_tracing(trace_dir=...)` — both appear in older docs/comments but do not exist. Tracing lives in `nooa.tracing`; `trace_dir` is an argument of `exporters.jsonl()`, and `enable_tracing()` returns `None`.
- Auto-tracing is attempted once per process. If the viewer wasn't running when the first `Agent` was constructed, later agents won't retry — call `enable_tracing(...)` explicitly or restart with the viewer up.
- For short scripts, call `flush_traces()` before exit; batch exporters flush on a ~1s schedule and a fast exit can drop the tail of a trace.
- Logic that runs *outside* agent methods (module-level preprocessing, `main()` helpers) is invisible in traces. Keep interesting logic inside agent methods so failures leave trace evidence.

## Verifying capture works

```python
from nooa.tracing import enable_tracing, exporters, flush_traces
enable_tracing(exporters=[exporters.jsonl("./traces")])
agent = MyAgent()
await agent.run("test")
flush_traces()
# ls ./traces/*.jsonl  → one file per session; inspect with trace-explorer
```

## Related skills

- `nooa-trace-viewer` — run the web viewer and browse captured traces.
- `nooa-trace-explorer` — programmatic/CLI trace analysis and root-cause debugging.
