# NeMo Relay integration

Install the optional extra with `uv sync --extra nemo-relay`. NOOA supports
Relay `>=0.11.1,<0.12`.

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

Concurrent method calls use isolated stacks through Relay's public propagation
APIs. Top-level NOOA run scopes also isolate their mutable stacks. Parent/root
trace relationships are retained. Ancestor scope-local middleware and subscribers
do **not** transfer into a new run or isolated fanout task. Register per-run
policies inside `nemo_relay_scope`. Sequential nested calls, including Relay's
managed callback continuations, retain those policies.

## Upgrading from Relay 0.6

- LLM sanitizers accept `(payload, context)` and must return the payload to keep
  it in telemetry. Returning `None` omits it. Sanitizers affect telemetry, not
  provider traffic or the value returned to NOOA.
- In async code, use `await nemo_relay.subscribers.flush_async()` before reading
  captured events or exporting trajectories.
- Tool callbacks return `ToolExecutionResult`. Existing tool response sanitizers
  still receive the raw encoded value, and LLM response sanitizers still receive
  the legacy `message` envelope. Scope-end event sanitizers also see the legacy
  shapes. A final event projection converts only their sanitized output to the
  exported shapes below; omitted payloads stay omitted. Native call UUIDs
  identify the event, so metadata redaction does not disable this conversion.
- NOOA's emitted Python results use
  `nemo.relay.tool_result@1`: `content` is readable text, `result` retains the
  encoded value, and `is_error` reports handled execution errors. The original
  NOOA `ExecutionResult` still reaches the caller.
- Final LLM event telemetry uses `assistant_message` instead of `message`,
  matching Relay's generic exporter. Reasoning is moved into the final `reasoning`
  field, avoiding an extra unsanitized copy. Opaque native provider replay state
  stays outside this projection. The effective method-level model is recorded.

The adapter does not enable response caching or routing. Python cells mutate
live state and their result wrappers are marked non-replayable. External
execution intercepts or caches that replace a call without invoking NOOA are
unsupported and raise an explicit short-circuit error, not a guardrail rejection.

## ATIF exports

Relay 0.11.1 emits **ATIF-v1.7**. Harbor's current models also accept v1.8, which
adds audio content; this integration does not relabel v1.7 exports as v1.8.
Text-only messages and tool observations are strings. A nonempty provider-issued
tool-call ID is passed to Relay to improve correlation; empty IDs are treated as
absent. Numeric/structured results retain a readable representation capped at
4096 characters plus a truncation marker, as well as the encoded value in extra
metadata. Opaque pickle blobs are summarized in the display copy. Empty results
are not turned into the text `null`.
Marks, including resource samples, are not ATIF steps; use ATOF/subscribers for
the complete event stream.

Relay's ATIF exporter has a remaining attribution limitation: NOOA's
host-generated `prefill_*` Python cells can be attached to the preceding model
step, including a different nested/concurrent method's step. These cells were
not requested by that model. Schema validity and matching call IDs do not prove
model ownership. Concurrent completions may also be grouped differently by
Relay's exporter. Use the original event/function ancestry when investigating
prefills; this adapter does not rewrite Relay's trajectory assembly algorithm.

See the [Relay ATIF guide](https://docs.nvidia.com/nemo/relay/v0.11.1/configure-plugins/observability/atif)
and [Harbor format specification](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md).

## Local diagnostics and resource measurements

The existing quickstart can additionally enable Relay diagnostics and CPU/memory
polling once for the application:

```bash
NOOA_RELAY_DIAGNOSTICS=1 uv run --extra nemo-relay python examples/quickstart/15_nemo_relay.py
```

It prints lifecycle/resource events, writes ATIF examples to
`/tmp/nemo_relay_trajectory_{research,demo}.json`, and writes a bounded diagnostic
page plus a CPU/memory snapshot to `/tmp/nooa-relay-diagnostics.json`. GPU, disk,
network and process-detail collection are disabled in this example. Without
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
