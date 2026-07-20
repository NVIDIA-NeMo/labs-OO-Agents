---
name: refine-agent-prompt
description: >-
  Render and refine the system prompt of a nooa agent. Use when the user wants
  to inspect, debug, or improve an agent's prompt, context blocks, or tool documentation.
---

# Refine Agent Prompt

Render the exact prompt that would be sent to the LLM — with real agent state, real
context blocks, and real arguments — then work through diagnostic questions to find waste.

## Step 1: Render the prompt

Use `nooa.print_prompt` with a `FakeLLMClient` to render without making a real LLM call:

```python
import asyncio, nooa
from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

class DemoAgent(Agent, llm=FakeLLMClient()):
    async def respond(self, message: str) -> str:
        """Answer the user's message."""
        ...

async def main():
    agent = DemoAgent()
    await nooa.print_prompt(agent.respond, "example user message here")

asyncio.run(main())
```

For a custom agent loaded from a file:

```python
import asyncio, importlib.util, sys, nooa
from nooa.unifiedllm import FakeLLMClient

spec = importlib.util.spec_from_file_location("agent_module", "path/to/my_agent.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["agent_module"] = mod
spec.loader.exec_module(mod)

async def main():
    agent = mod.MyAgent(llm=FakeLLMClient())
    await nooa.print_prompt(agent.respond, "example user message here")

asyncio.run(main())
```

Run with: `uv run python3 << 'EOF' ... EOF`

To render a specific sub-method (e.g. `answer_question`, `brainstorm`, `implement_step`):

```python
await nooa.print_prompt(agent.answer_question, "what is the capital of France?")
await nooa.print_prompt(agent.implement_step, '{"description": "add logging"}')
```

To capture the data programmatically instead of printing:

```python
data = await nooa.build_prompt_data(agent.respond, "hello")
print(data.system_prompt)   # the full system message
print(data.task_prompt)     # the task user-message
print(data.inspect_prefill) # the inspect/prefill code block
```

## Step 2: Diagnostic questions

Work through these one by one. Each has tradeoffs — discuss with the user before changing.

---

### Is the execution context clean?

The `<execution_context>` block lists every symbol available in the REPL.
Scan **Imported items** and ask: does the LLM have any reason to call this directly?

**Should usually be hidden:** framework internals (`SkillManager`, `TokenBudgetSummarizer`),
base classes (`BaseTUIAgent`, `Agent`), LLM client classes (`FakeLLMClient`, `get_llm_client`),
internal config (`CodeActConfig`, `SummarizationConfig`, `AgentConfig`),
pydantic internals (`BaseModel`, `Field`), private helpers (anything starting with `_`).

**Fine to expose:** tool/skill classes the LLM uses (`ShellTools`, `WebPublisher`),
`pd`, `np`, `px`, `go`, `Path`, `json`, `math`, `re`.

**Fix:** wrap the import in `with hidden:` at module level:

```python
from nooa import hidden   # must be outside

with hidden:
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig
```

**Tradeoff:** hiding something means the LLM can't use it by name in the REPL.
Only hide things the agent has no reason to call directly.

---

### Is there duplicate content?

Look for the same information in more than one place.

- **Same class in `<self>` (Referenced Types) AND as a context block or skill doc**:
  usually OK if `<self>` shows a concise one-liner. If both are full docs, that's waste.
- **Task prompt repeating a workflow** already in a context block:
  usually removable — the context block is dynamic, the docstring is stale.
- **Skill docstring examples already shown in the task prompt**: consider whether the
  repetition is reinforcing a critical pattern or just inflating tokens.

---

### Are internal methods visible?

Scroll `<self>` for methods the LLM should never call directly:
`get_summarization_status()`, `_respond_codeact()`, `_phase`, `_workflow_state`,
`_config`, `install_summarizer()`, etc.

**Fix:** `@hidden` from `nooa`:

```python
from nooa import hidden
from nooa import Agent

class MyAgent(Agent, llm=...):

    @hidden
    def get_summarization_status(self) -> dict: ...
```

Fields can be hidden with the `hidden` marker from `nooa.storage.markers`:

```python
from typing import Annotated
from nooa import hidden
from nooa.storage.markers import nosnapshot

class MyAgent(Agent, llm=...):
    _phase: Annotated[str, hidden]
    _workflow_state: Annotated[dict, hidden]
    bash: Annotated[ShellTools, nosnapshot]       # visible but not snapshotted
```

**Tradeoff:** none for internal lifecycle state. If the LLM is calling `_phase` directly,
that's a prompt problem to fix, not a reason to keep it visible.

---

### Is the main entry point hidden?

If the agent has a primary method the LLM is *called as* rather than *calling*,
it will appear in `<self>` — but the LLM never needs to call itself recursively.

**Fix:** `@hidden` the entry-point so `<self>` shows only attributes the LLM uses:

```python
@hidden
async def respond(self, user_message: str) -> None:
    """..."""
    ...
```

**Tradeoff:** none for single-method agents. For orchestrators with multiple callable
sub-methods, only hide the outermost entry point.

---

### Is the task prompt (docstring) clear and non-redundant?

Read the method docstring as if you're the LLM seeing it fresh:

- Does it repeat tool usage already shown in Skills or context blocks?
  Replace with a pointer: "See `doc(self.bash)` for shell commands."
- Are the output rules (format, structure) specific and unambiguous?
- Does `{user_message}` appear as a template substitution? That's correct.
- **Are raw `{argument}` substitutions duplicated in the prefill?**
  Check `=== PREFILL ===` — inputs are already printed there. Repeating
  `{user_message}` in the body is redundant if the prefill already shows it.

**Tradeoff:** shorter prompts are cheaper, but the LLM may miss instructions that only
appear in context blocks it skims. Keep critical constraints in the task prompt even if
they're also elsewhere.

---

### Are Skills surfaced well?

Agents with `Skill` subclass attributes expose them via `doc(self.<skill>)`.
The `<execution_context>` skills table instructs the LLM to call `doc()` before using each skill.

Check:
- Is the skill's 1-liner description clear enough for the LLM to know when to call `doc()`?
- Does the skill's full docstring (revealed by `doc()`) have enough examples for the LLM
  to use it correctly in one shot, without needing follow-up calls?
- Is any skill API already fully replicated in the task prompt? If so, that's waste —
  the skill docstring is the canonical source.

**For `WebPublisher` / `self.web`**: the agent should be able to produce a working
Plotly call on the first try. If it takes multiple LLM iterations, check whether the
skill examples cover the user's data shape (long-form vs wide-form DataFrames,
the right `px.*` function, axis mapping). Add more targeted examples to the skill docstring.

---

### Are return types and data structures strongly typed?

Look for string-in / string-out patterns where typed Pydantic models would be clearer:

```python
# Instead of:
async def classify_intent(self, msg: str) -> str:
    """Returns 'question', 'feature', 'bugfix', or 'refactor'"""

# Use Pydantic — fields render as a schema in the prompt:
from pydantic import BaseModel
from typing import Literal

class Intent(BaseModel):
    """Classification of the user's request."""
    task_type: Literal["question", "feature", "bugfix", "refactor"]
    summary: str  # One-sentence description of what the user wants.

async def classify_intent(self, msg: str) -> Intent:
    ...
```

Pydantic models render field types, defaults, and docstrings automatically in `<self>`
and Referenced Types — the LLM gets a clear schema without extra documentation effort.

---

### Are context blocks vs inline `{doc(...)}` used correctly?

The task prompt can embed tool docs inline or store them in a persistent context block:

- **Short-running methods** (< 1 min, few turns): inline `{doc(self.tool)}` is fine.
- **Long-running methods** (many turns, large context, at risk of summarization):
  prefer `self.context["key"] = doc(self.tool)` so the docs persist across summarization.

```python
# Inline (good for short tasks):
async def respond(self, user_message: str) -> None:
    """{doc(self.bash)}
    Message: {user_message}
    """

# Context block (good for long tasks):
def __init__(self, ...):
    ...
    self.context["bash_docs"] = doc(self.bash)
```

**Tradeoff:** context blocks add overhead every turn; inline is paid once at task start.

---

## nooa reference

### `@hidden` — hide a method from `<self>` and the execution context

```python
from nooa import hidden

@hidden
def get_summarization_status(self): ...

with hidden:
    from nooa.agents import TokenBudgetSummarizer
    from nooa.config import CodeActConfig
```

### `Annotated[T, hidden]` — hide a field from `<self>`

```python
from nooa import hidden
from nooa.storage.markers import nosnapshot
from typing import Annotated

class MyAgent(Agent, llm=...):
    _phase: Annotated[str, hidden]
    bash: Annotated[ShellTools, nosnapshot]   # shown but not persisted to snapshots
```

### `nosnapshot` — exclude a field from session snapshots

```python
from nooa.storage.markers import nosnapshot
from typing import Annotated

bash: Annotated[ShellTools, nosnapshot]   # live tools shouldn't be serialised
```

### `nooa.print_prompt` — render without an LLM call

```python
await nooa.print_prompt(agent.my_method, arg1, arg2)
```

### `nooa.build_prompt_data` — capture prompt sections programmatically

```python
data = await nooa.build_prompt_data(agent.my_method, arg1)
data.system_prompt   # full system message
data.task_prompt     # task user-message
data.inspect_prefill # prefill/inspect code
data.strategy_name   # e.g. 'CodeActStrategy'
```
