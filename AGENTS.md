# NVIDIA OO Agents Framework

Python framework for building AI agents. An agent is a Python object — its methods are its capabilities.

## Core Concept

```python
class MyAgent(Agent, llm=llm):
    # Deterministic method — a tool the LLM can call
    def get_stock(self, item: str) -> int:
        return INVENTORY.get(item, 0)

    # Generation method — LLM executes this; docstring = prompt
    async def analyze(self, data: str) -> Result:
        """Analyze the data. Use self.get_stock() for inventory checks."""
        ...  # ← ellipsis triggers LLM generation
```

**Ellipsis (`...`) = LLM generation.** No ellipsis = regular Python.
**Docstring = prompt.** Instructions only — arguments are rendered to the LLM by default.
**Return type = contract.** Pydantic models force structured LLM output.

### Arguments Are Rendered By Default — Never `{param}` in Docstrings

The framework already shows the LLM every argument: the method signature accompanies the task, the default CodeAct prefill pprint()s each parameter value under the truncation config (and the values are live REPL variables), and Predict serializes parameters with size caps. Writing `{data}` in a docstring re-injects the raw value into the instruction text — redundant, **untruncated** (huge arguments blow up the context), and it moves untrusted data into the instruction channel. Reserve `{...}` templating for what the signature cannot show: `{self.attr}` instance state and computed expressions like `{len(items)}`.

## Quick Rules

### Method Design

- **One method = one LLM task.** Don't make a method do classification AND implementation. If your method identifies terms, greps, and summarizes — split it into three.
- **Orchestrators are pure Python.** Workflow sequence methods have real bodies (no `...`), calling generation methods for each step. If a class has no `...` methods it doesn't need to subclass `Agent` at all.
- **Helpers beat prompts.** Deterministic logic as regular methods > telling the LLM to figure it out. Methods are visible to the LLM via `doc(self)` (auto-generated API documentation). Define helpers as class methods, not as lambda/function references assigned to `self`.
- **Evidence before assertions.** Run tests/verification before claiming work is done. Enforce in the orchestrator.
- **Everything visible by default.** Module-level and agent-level names are visible to the LLM (in `doc(self)` and exec_globals). Hide explicitly with `@hidden`, `Annotated[T, hidden]`, or `with hidden:`.

### Visibility

Single rule (Python-style): **visible by default, hide explicitly.**

- **Module level:** Imports, constants, functions, and classes are visible to agent-generated code unless you hide them:
  - `@hidden` on module-level functions
  - `Annotated[T, hidden]` on module-level variables (e.g. `API_KEY: Annotated[str, hidden] = "..."`)
  - `with hidden:` for imports or unannotated names you want to keep out of exec_globals
- **Agent** (methods, fields): Public names visible by default. `_private` methods/fields hidden by default. Opt out with `@hidden` or `Annotated[T, hidden]`; opt a `_private` method back in with `@spec(hidden=False)`.
- **Types:** Types used in the agent's public API (return types, parameter types, fields) must be **defined or imported at module level** so they appear in exec_globals. No automatic injection.

| Scope | Default | Opt-out / Opt-in |
|-------|---------|-----------------|
| Module level | VISIBLE | `@hidden`, `Annotated[T, hidden]`, `with hidden:` |
| Class methods (public) | VISIBLE | `@hidden` |
| Class methods (`_private`) | HIDDEN | `@spec(hidden=False)` |
| Class fields (public) | VISIBLE | `Annotated[T, hidden]` |
| Class fields (`_private`) | HIDDEN | — |
| Types | Must be at module level (import or define) to be in exec_globals | — |
| `context` / `events` | **Hidden by default** | Opt in: `spec(self, "context", hidden=False)` in subclass `__init__` |

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from nooa import Agent, hidden, spec

CATEGORIES = ["billing", "technical", "general"]   # visible to LLM by default
MAX_RESULTS = 10

with hidden:
    import secrets  # LLM cannot see this

class SearchAgent(Agent, llm=llm):
    index_path: Path = Path("data/index.json")
    api_key: Annotated[str, hidden] = ""       # hidden from LLM

    def search(self, query: str) -> list[str]:
        """Search the index for the query."""
        ...

    @hidden
    def rebuild_index(self) -> None:
        raw = Path("data/raw.json").read_text()
        self._entries = json.loads(raw)

    @spec(hidden=False)
    def _shown_helper(self) -> str:
        """This private method is explicitly shown in doc() output."""
        return self._compute()
```

To unhide a parent's hidden field, re-declare in the subclass without `hidden`:

```python
class MyAgent(Agent, llm=llm):
    my_tool: MyTool  # unhides Parent's Annotated[MyTool, hidden]
```

> **Note:** `context` and `events` are exceptions — do NOT re-declare them. Use `spec()` instead (see below).

`ContextApi` and `EventsApi` are **always present** on every Agent as `self.context` and `self.events`, but **hidden from the LLM by default**. Subclasses opt in by calling `spec(self, "context", hidden=False)` (and/or `spec(self, "events", hidden=False)`) in their `__init__` to expose them via `doc(self)`.

```python
from nooa.agentdoc import spec

class MyAgent(Agent, llm=llm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "context", hidden=False)   # LLM can now see and use self.context
        spec(self, "events", hidden=False)    # LLM can now see and use self.events
```

The underlying state lives on `agent.context_manager` (a `ContextManager`) and `agent.event_manager` (an `EventManager`) — both present and hidden from the LLM.

Common typing constructs and framework symbols (`asyncio`, `typing`, strategies, `doc()`, `pprint()`) are always available in agent-generated code.

### Strategy Selection

- **Default strategy is `CodeActStrategy()`** — gives the LLM `execute_python()` + `return_result()` tools. No `@strategy` decorator needed.
- **`PredictStrategy` for extraction/classification.** If your method returns a Pydantic model and doesn't execute code, don't use `CodeActStrategy` just because it's the default. Use `@strategy(PredictStrategy())` instead.

| Strategy | Use When |
|----------|----------|
| `PredictStrategy` | Single-shot LLM call that solves the task in one go (no iteration). Output types are validated against the return type annotation. |
| **`CodeActStrategy`** (default) | Code execution + iteration in REPL; anything that needs to run code or use tools |

### Reserved Parameters

- **`reasoning`** — Reserved. Declaring it as a generation-method parameter raises `ValueError` at class creation; chain-of-thought is provided via the `reasoning()` builtin available in CodeAct-generated code. Use an alternative like `rationale`.

### Context Blocks

- **Static context blocks for once-computed values.** Use `self.context["key"] = value` for values known at assignment time. These are evaluated once and cached.
- **Dynamic context blocks for runtime values.** Use `self.context.set_dynamic("key", "python_expression")` for values that change during agent execution. The expression is re-evaluated each LLM turn.

```python
# Static: set once, never changes
self.context["plan"] = plan.format()

# Dynamic: re-evaluated each LLM turn
self.context.set_dynamic("project_state", "self.format_project_state()")
```

### Tracing

Tracing is automatic when the dev viewer is running (`nooa start-dev`, port 5001). To write JSONL files instead:

```python
from nooa.tracing import enable_tracing, exporters

enable_tracing(exporters=[exporters.jsonl("traces/my_agent")])
```

Trace viewer runs on port 5001 by default (`nooa start-dev --port` to change). See `examples/quickstart/06_tracing.py` for a full example.

- **All public methods are traced by default.** Private (`_private`) and dunder (`__method__`) methods are also traced unless you opt out.
- **Use `@no_trace` to exclude from traces.** Decorate any method (public, private, or dunder) with `@no_trace` to prevent it from appearing in traces while still allowing generation.

```python
from nooa import no_trace

class MyAgent(Agent, llm=llm):
    @no_trace
    async def utility(self):
        """Public but NOT traced"""
        ...
```

### Quality & Testing

- **Evidence before assertions.** Run tests/verification before claiming work is done. Enforce in the orchestrator.

### Understanding `doc(self)`

The `doc(self)` expression generates auto-documented API information about the agent. When used in prompts, it shows the LLM:
- All non-`@hidden` methods (generation methods and deterministic helpers)
- All non-`Annotated[T, hidden]` fields
- Method signatures with type hints
- Docstrings for each method

This enables **progressive disclosure** — the LLM can discover available tools and methods dynamically. Use `{doc(self)}` in your docstrings to give the LLM access to the agent's full API:

```python
async def task(self, data: str):
    """Process the data.

    Available methods:
    {doc(self)}
    """
    ...
```

For deeper authoring guidance, see `skills/nooa-agent-authoring/SKILL.md`.

## Python & Dependencies

- **Use `uv` exclusively** for package management — never pip, pip-tools, poetry, or conda.
- Add packages: `uv add <package>`
- Remove packages: `uv remove <package>`
- Sync lockfile: `uv sync`
- Run scripts: `uv run python <script>.py`
- Run tests: `uv run pytest`

## Experiments

All experiments in `experiments/` **must** have a `README.md` covering: research question, experiment design, key metrics, how to run, and results summary (updated after runs with quantitative findings).
