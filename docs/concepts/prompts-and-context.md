# Prompts and context

NOOA builds prompts from Python structure. The goal is not to place everything
in one large instruction string, but to give each kind of information a clear
home.

## Where information belongs

| Information | Put it here |
|---|---|
| Agent-wide role and stable constraints | Short class docstring |
| Instructions for one task | Method name and docstring |
| Data for the current call | Method parameters |
| Images, audio, video, and PDFs | Typed Media parameters, or captured CodeAct `show()`; [media guide](../media.md) |
| Trusted instance configuration | `{self.attribute}` in a docstring |
| Cross-call facts that should stay visible | Context blocks |
| Chronological interaction history | Events |
| Intermediate values used only by Python | Local variables and private/hidden fields |

This separation controls prompt size and keeps untrusted data out of the
instruction channel.

## Class and method docstrings

```python
class Translator(Agent, llm=llm):
    """Translate faithfully without adding new information."""

    def __init__(self, language: str, **kwargs):
        super().__init__(**kwargs)
        self.language = language

    async def translate(self, text: str) -> str:
        """Translate the text into {self.language}."""
        ...
```

The class docstring applies to every agentic method. Keep it short. The method
docstring describes only the current task. `{self.language}` is appropriate
because it is trusted configuration not present in the method signature.

Pass `Image`, `Audio`, `Video`, or PDF `File` objects as method arguments.
CodeAct can also use `show()` while generated code runs. Set the model's
`media_inputs` to the formats it accepts. See [media inputs](../media.md).

The `text` parameter is already rendered by the strategy. Do not repeat it as
`{text}` in the docstring.

## Input inspection limits

CodeAct and PurePython initially print bounded argument previews: by default,
2,000 characters per string, 25 items per container, and depth 4. The full
objects remain live REPL variables. `print_prompt()` shows the generated
inspection code, while an execution trace shows the captured output actually
sent to the model.

Set `Annotated[str, spec(max_string=None)]` to print a complete string (import
`Annotated` from `typing` and `spec` from `nooa`). Per-parameter `max_length`
and `max_depth` overrides work the same way. Defaults resolve through agent
class, instance, and method-level `@strategy(..., truncation=...)` settings;
per-parameter `spec()` overrides have final precedence.

Actual formatting or capture omissions emit a developer-facing
`PrefillTruncationWarning` identifying the method and the limits involved.
Raise the relevant `TruncationConfig.prefill_format` limits or set parameter
overrides when complete input matters. Formatting and stdout capture are
independent: even `spec(max_string=None)` can produce a clipped preview if
printed output exceeds `capture.max_stdout` (50,000 characters by default).
Raise that capture budget explicitly, or arrange to inspect the input in
smaller pieces. Warnings do not include parameter contents or enter the model's
captured stderr, and can be filtered with Python's standard warnings filters.

## Context blocks are deliberate prompt input

Context blocks are named sections added to the system prompt:

```python
from nooa import Context

# Fixed content placed in the stable, cache-friendly prefix.
agent.context["policy"] = Context(POLICY_TEXT, prefix=True)

# Live content re-evaluated for every LLM turn.
agent.context["progress"] = Context(expr="self.render_progress()")
```

Use context for bounded information that several calls should see, such as a
plan, current project state, or a stable policy. Context is eager: every selected
block consumes prompt space on every turn. It is not a general object store.

Fixed versus expression-backed content and prefix versus suffix placement are
independent choices. A bare `agent.context["key"] = value` is fixed content in
the volatile suffix.

### How the live context reaches the model

The volatile suffix is the *live context*. By default it is **recorded into
the history at least every 5 turns**: right before a request that is due, the
runtime records a `LiveContextSnapshot` event (a one-line note saying it is
periodic and may be stale, then the `<context>` envelope) and the request
carries it like any other event. Being history, it stays in every later request
unchanged, so requests are append-only, provider prompt caching covers the whole
history, and training can consume the transcript as is. Before this default the
live context was regenerated as a trailing `<context>` message on every
request; that behaviour is still available.

`RenderConfig(live_context=...)` builds the renderer with one of three policy
classes, each carrying only the parameters it reads (`kind` tells them apart in
settings files and JSON); read it back as `render_config.live_context`:

| policy | what the model sees | when to use it |
|---|---|---|
| `AppendedLiveContext(every_n_turns=5)` — `kind: append` (default) | static prefix · history, in which a `LiveContextSnapshot` event is recorded before a due request · boundary | normal operation; append-only transcripts |
| `DynamicLiveContext()` — `kind: dynamic` | static prefix · history · boundary · fresh `<context>` on every request | the previous behaviour |
| `NoLiveContext()` — `kind: off` | static prefix · history · boundary | measuring what the live blocks contribute |

A snapshot is due when the visible history holds no snapshot event (the first
request, or after every earlier one was archived) or when `every_n_turns` or
more responses follow the last one; a due snapshot is skipped only when no live
block changed since the last one, so blocks that change every turn (a context
usage counter, say) make most due snapshots record. Events recorded while a
call is pending land after the snapshot, as the model saw them; a retry after a
context-window error reuses the snapshot already recorded; archiving the oldest
events can only delay the next snapshot, and archiving the latest snapshot
brings it forward. A session recorded under `dynamic`, or switched to `append`
mid-session, simply gets a fresh snapshot on its next request. Smaller
`every_n_turns` values trade prompt size for freshness (`1` snapshots before
every call). Agents without a static block keep their fixed text in the system
message; expression-backed blocks still follow the policy.

```python
from nooa.context_blocks import DynamicLiveContext, RenderConfig

# Restore the previous behaviour: regenerate the live context on every request.
agent = MyAgent(render_config=RenderConfig(live_context=DynamicLiveContext()))
```

`RenderConfig` is the only place the policy lives. A policy is written as an
object, its dict form (`{"kind": "append", "every_n_turns": 3}`) or a bare kind
name (`"dynamic"`, `"off"`; YAML's bare `off` too), and a `RenderConfig` itself
may be given as a mapping wherever an agent takes one:

- any agent: `MyAgent(render_config={"live_context": "off"})`;
- Atom, in `settings.yaml` (a session's `SessionOptions.render_config`
  overrides it; `AtomHarness` forwards its `render_config` keyword):

  ```yaml
  atom:
    render_config:
      live_context:
        kind: "off"
  ```

- NeMo Gym: `init_kwargs.render_config.live_context.kind=off`, or a dict for
  `append`.
- Atom can change it from a session, for the running agent and future ones:
  `self.workspace_settings.set_live_context("append", every_n_turns=3)`, or
  `/live-context append 3` in the TUI/ACP (`/live-context` alone shows it).

## Events are history, not configuration

Each call records task, model, code-execution, error, and result events. The
selected event history supplies the conversation seen by later calls.

Use events when sequence matters: what was asked, tried, observed, and returned.
Use context when a fact should be deliberately restated as current information.

Long-running agents should summarize or filter history instead of copying the
entire past into a context block. See the
[summarization quickstart](../../examples/quickstart/09_summarization.py).

## Lifetime, persistence, and memory

Context blocks, events, and ordinary fields remain available across calls while
the same agent object is alive. By default, NOOA uses in-memory storage. A new
process or a newly constructed instance does not automatically continue that
state.

For durable checkpoints, pass a persistent `StorageManager`, save a snapshot,
and restore it into a freshly constructed compatible agent. Snapshots can
preserve event history, context blocks, and snapshotable fields. This is an
explicit application workflow, not an automatically checkpointed graph after
every node.

Long-term semantic memory is different again. The optional `nooa-memory`
package provides remember/recall behavior across sessions; it is not the same
as event history or an agent snapshot. See the
[memory quickstart](../../examples/quickstart/12_memory.py).

## Prompt layers are a trust boundary

Method parameters often contain user messages, documents, repository files, or
tool output. Treat those values as data. Method and class docstrings, trusted
context, and framework instructions are the instruction layer.

This is why raw argument interpolation is discouraged: it promotes untrusted,
potentially large data into trusted instructions and bypasses the normal
rendering limits.

Long trusted instructions can live in a versioned file, be read into a private
instance attribute, and be referenced as `{self._instructions}` by the one
method that needs them. Avoid placing method-specific instructions in a global
context block sent to every call.

## Let Python retain non-prompt state

Not every value the application knows must be shown to the model. Keep caches,
credentials, full datasets, open connections, and intermediate artifacts in
ordinary hidden fields or external storage. Expose a bounded rendering or a
deterministic method when the agent actually needs access.

## Common mistakes

- Writing a long class prompt that every method pays for.
- Repeating method arguments with `{argument}`.
- Using context as a database or dumping an unbounded project into it.
- Confusing event history with current authoritative state.
- Assuming a new process or agent instance automatically resumes prior state.
- Giving every method instructions intended for only one phase.
- Rendering secrets or raw internal clients into the visible agent state.

## Continue

- [Agents and methods](agents-and-methods.md)
- [Orchestration](orchestration.md)
- [Architecture](../architecture.md)
- Runnable examples: [dynamic prompts](../../examples/quickstart/07_dynamic_prompts.py)
  and [context blocks](../../examples/quickstart/08_context_blocks.py)
