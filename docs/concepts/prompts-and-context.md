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

## Output limits: `TruncationConfig` and package config models

Every configurable limit core applies to text a model or user reads is a field
of `nooa.config.TruncationConfig`. Set it on the agent class
(`class MyAgent(Agent, truncation=...)`), the instance
(`MyAgent(truncation=...)`) or one method (`@strategy(..., truncation=...)`,
while that method runs). Later levels win, and each names only the fields it
changes (sub-configs merge field by field). Code that applies a limit reads the
config in force with `nooa.config.effective(TruncationConfig)` when it applies
it. This is the one place that encodes those precedence rules. It uses context
variables: asyncio tasks and `asyncio.to_thread` keep the caller's limits, but
a plain `threading.Thread` does not. Run its target in
`contextvars.copy_context().run(...)`.

A package's own limits are fields of that package's own pydantic config model,
not of `TruncationConfig`; core never imports or names them. They follow the
same rule and are set with `configs=`, a list of config instances (one per
model, matched by exact type):

```python
from nooa import strategy
from nooa.config import effective
from nooa_atom.limits import AtomLimits

class MyAgent(AtomAgent, configs=[AtomLimits(max_job_tail=5_000)]): ...   # class
agent = MyAgent(configs=[AtomLimits(instructions_file_chars=20_000)])      # instance
# method: @strategy(CodeActStrategy(), configs=[AtomLimits(max_job_tail=500)])
effective(AtomLimits).max_job_tail   # read where the limit is applied
```

Outside any agent `effective(Model)` is the model's defaults (`Model()`).
Every layer is a partial override: the fields it sets win, the rest come from
the layer below. The layers are the model's defaults, then each class's own
`truncation=` / `configs=` (base classes first), then the instance, then the
method's `@strategy(...)`.
`nooa.config.merge_config(base, override)` is the field merge all levels use.

| Field | Default | Bounds |
| Field | Default | Bounds |
|---|---|---|
| `max_context_tokens` | `None` | whole-context token budget |
| `response_reserve_tokens` | 4,096 | planning reserve when no reply cap is known |
| `overflow_files` | `True` | whether cut output may go to a file |
| `placeholder_chars` | 50,000 | one docstring `{expr}` placeholder (method docstrings, Task prompts, the system prompt) |
| `capture.max_stdout` / `max_stderr` / `max_error` | 50,000 / 2,000 / 10,000 | one `execute_python` cell's streams and error |
| `capture.tail` | `None` (half) | tail share of the three cell streams above |
| `capture.max_shell_output` | 30,000 | each stream of a shell command result (`ShellTools`, `BashSession`; the nooa-atom terminal card shows the same, the nooa-cli card half) |
| `media_capture.max_attachments_per_execution` | 5 | `show()` attachments per cell |
| `event_format` | 10,000 chars / 200 items / depth 5 | event fields in the conversation |
| `prefill_format` | 2,000 / 25 / 4 | input inspection previews and non-string placeholders |
| `context_block_format` | unlimited | non-string `self.context[...]` values |
| `state_format` | 500 / 50 / 4 | the agent's `state` block. The prompt shows its expression, so it is fixed when the agent is built: the class and instance levels apply, a method level does not |

Package limits:

| Model | Field | Default | Bounds |
|---|---|---|---|
| `nooa_atom.limits.AtomLimits` | `max_job_tail` | 20,000 | the end of each stream on a background job's done update |
| `nooa_atom.limits.AtomLimits` | `instructions_file_chars` | 100,000 | one repository instruction file (`AGENTS.md`) |
| `nooa_atom.limits.AtomLimits` | `instructions_total_chars` | 200,000 | the whole `repository_instructions` block |
| `nooa_cli.coding.limits.CodingLimits` | `instructions_file_chars` / `instructions_total_chars` | 100,000 / 200,000 | the same for the nooa-cli coding agent |

`max_event_tokens` and `min_preserved_events` are accepted but not enforced.
Per-field `Annotated[T, spec(...)]` limits on an event or parameter type
override the `*_format` defaults for that field.

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
