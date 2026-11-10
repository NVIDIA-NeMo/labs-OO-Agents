# NeMo Relay integration

Install the optional extra with `uv sync --extra nemo-relay`. NOOA supports
Relay `>=0.11.1,<0.12`; unsupported installed versions fail before middleware
is installed.

```python
from nooa.nemo_relay_middleware import nemo_relay_scope

async with nemo_relay_scope(agent, "research"):
    result = await agent.research("a topic")
```

The scope installs middleware for traced async agent methods, LLM calls and
Python execution, then removes it on exit. Sync helpers are not individually
intercepted. Overlapping scopes on one agent share the installation, avoiding
duplicate events and policy checks. Relay does not discover tool calls inside
generated Python.

By default, scopes preserve enclosing scope-local guardrails, redactors and
subscribers, including across sequential nested methods and managed callbacks.
Relay cannot safely share a mutable stack across concurrent tasks. NOOA rejects
unsafe concurrent sharing rather than silently losing those policies.

For concurrent runs or method fanout, explicitly opt into independent stacks:

```python
async with nemo_relay_scope(agent, "parallel-research", isolate=True):
    results = await asyncio.gather(agent.research("a"), agent.research("b"))
```

**Isolated stacks do not inherit enclosing scope-local guardrails, redactors or
subscribers.** This also applies to child-task method branches inside that run.
Register policies that must cover every branch globally, or register them on
each branch's own scope. Parent trace linkage survives isolation; scope-local
policy inheritance does not.

## Upgrading from Relay 0.6

- LLM sanitizers accept `(payload, context)` and must return the payload to keep
  it in telemetry. Returning `None` omits it. Sanitizers affect telemetry, not
  provider traffic or the value returned to NOOA.
- In async code, use `await nemo_relay.subscribers.flush_async()` before reading
  captured events or exporting trajectories.
- Tool callbacks return `ToolExecutionResult`. Tool-response sanitizers and
  scope-end event sanitizers continue to receive raw encoded values. LLM
  sanitizers and events retain the legacy `message` envelope. NOOA does not
  rename these event fields after sanitizers run.
- Inspection data is converted to JSON-safe values for Relay's Rust boundary.
  Unsupported values may be summarized or represented as text; the original
  NOOA response or `ExecutionResult` still reaches the caller.
- NOOA callback exceptions retain their original type and message for the caller,
  while Relay receives generic failure text. Cancellation waits for that call's
  nested callback cleanup before closing its enclosing scopes.

The adapter does not enable response caching or routing. Python cells mutate
live state. External execution intercepts or caches that replace a call without
invoking NOOA are unsupported and raise an explicit short-circuit error, not a
guardrail rejection.

## ATIF exports

Relay 0.11.1 emits **ATIF-v1.7**. Harbor's current models also accept v1.8, which
adds audio content; this integration does not relabel v1.7 exports as v1.8.
Use the NOOA export helper after application work has completed:

```python
from nooa.nemo_relay_middleware import export_nemo_relay_atif

trajectory = await export_nemo_relay_atif(exporter)
```

It flushes pending events and formats the native export's **already sanitized**
legacy payloads into readable assistant text, reasoning and tool observations.
No runtime projection callbacks or extra sanitizer priorities are installed.
Missing payloads stay missing. Structured tool displays are capped at 4096
characters plus a truncation marker; opaque pickle data is summarized rather
than printed as base64. Empty results are not turned into the text `null`.

Nonempty provider-issued tool-call IDs identify their issuing steps when the
ID is unique across the trajectory. The helper uses those IDs to repair native
call/observation ownership; duplicate IDs remain ambiguous and tool names are
never used to infer ownership. Empty IDs are treated as absent.
Marks, including resource samples, are not ATIF steps; use ATOF/subscribers for
the complete event stream.

Relay's ATIF exporter has a remaining attribution limitation: NOOA's
host-generated `prefill_*` Python cells can be attached to the preceding model
step, including a different nested/concurrent method's step. These cells were
not requested by that model. Schema validity and matching call IDs do not prove
model ownership. Concurrent completions may also be grouped differently by
Relay's exporter. Use the original event/function ancestry when investigating
prefills; the helper cannot infer a model issuer for these host-generated cells.

See the [Relay ATIF guide](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atif)
and [Harbor format specification](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md).

## Local diagnostics and resource measurements

The existing quickstart can additionally enable Relay diagnostics and CPU/memory
polling once for the application:

```bash
NOOA_RELAY_DIAGNOSTICS=1 uv run --extra nemo-relay python examples/quickstart/15_nemo_relay.py
```

It prints lifecycle/resource events and the paths to its ATIF examples and
bounded diagnostics/CPU-memory snapshots. Files use private temporary directories
and owner-only file permissions; no predictable shared `/tmp` filename is used.
GPU, disk, network and process-detail collection are disabled in this example. Without
the flag, no plugin host is initialized by the example.

Diagnostics can contain private runtime details. Keep these artifacts local.
The first plugin initialization fixes process-wide diagnostic settings, so
restart the process to change them. Applications own one plugin activation;
individual `nemo_relay_scope` blocks do not initialize or stop plugins.

## Live verification

The opt-in test uses existing internal gateway or OpenAI credentials, runs real
nested/concurrent Predict and CodeAct calls, and writes ATIF under pytest's
temporary directory:

```bash
NOOA_RUN_RELAY_LIVE=1 uv run --extra nemo-relay pytest -m integration -s tests/integration/test_nemo_relay_live.py
```

Optional `NOOA_RELAY_LIVE_MODEL` selects another model at the selected trusted
endpoint. The internal and OpenAI credentials are bound to their respective
HTTPS endpoints; custom base-URL overrides are rejected before constructing a
client. No saved model configuration is changed.
