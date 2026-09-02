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

The `text` parameter is already rendered by the strategy. Do not repeat it as
`{text}` in the docstring.

## Per-parameter rendering limits

Use `spec()` metadata when one parameter needs different rendering limits from
the rest of the agent:

```python
from typing import Annotated

from nooa import Agent, spec


class Analyzer(Agent, llm=llm):
    async def analyze(
        self,
        document: Annotated[str, spec(max_string=None)],
        samples: Annotated[list[dict], spec(max_length=100, max_depth=6)],
    ) -> str:
        """Analyze the document and representative samples."""
        ...
```

`max_string`, `max_length`, and `max_depth` override
`TruncationConfig.prefill_format` for that parameter. `None` disables the
corresponding formatting limit. The complete objects remain live in CodeAct's
REPL even when their printed previews are truncated.

Rendering settings resolve from framework defaults through agent class, agent
instance, and method-level `@strategy(..., truncation=...)` configuration. A
parameter's `spec()` metadata has final precedence for that parameter.

CodeAct's inspection prefill writes rendered values to stdout, which has an
independent capture limit (`capture.max_stdout`, 50,000 characters by default).
Consequently, `spec(max_string=None)` disables string formatting truncation but
does not disable stdout capture truncation. Raise `capture.max_stdout`
explicitly when a complete printed value can exceed that separate limit.

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
