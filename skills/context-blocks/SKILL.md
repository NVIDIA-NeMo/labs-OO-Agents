---
name: context-blocks
description: "How to manage context blocks (system prompt sections) in NVIDIA OO Agents. Use when configuring what appears in an agent's LLM prompt — adding, overriding, suppressing, or positioning blocks. Covers the unified Context API, well-known framework keys, and the declarative/runtime interface."
compatibility: nooa >= 0.x (context-block-controls branch)
metadata:
  skill_type: api-reference
user-invocable: false
---

# Context Blocks API

Context blocks are named sections of the system prompt. They come from multiple sources (framework, strategy, skills, developer) and are managed through a **single unified interface**.

## Core Concept

Every context block entry is a dict mapping `{key: value}`:

```python
from nooa import Context

context = {
    "role": "You are a security expert",                       # str: literal text, volatile suffix
    "shell": Context(expr="doc(self.shell)", prefix=True),     # evaluated each turn, cacheable prefix
    "config": Context("stable config text", prefix=True),      # literal, cacheable prefix
    "status": Context(expr="f'{self.done}/{self.total}'"),     # evaluated each turn, volatile suffix
    "self": None,                                              # suppress block from prompt
}
```

## Value Types

| Value | Placement | Content |
|-------|-----------|---------|
| `"text"` | suffix (volatile) | Fixed literal |
| `Context("text", prefix=True)` | prefix (cacheable) | Fixed literal |
| `Context(expr="self.x()")` | suffix (volatile) | Re-evaluated each LLM turn |
| `Context(expr="self.x()", prefix=True)` | prefix (cacheable) | Re-evaluated each turn |
| `None` | — | Suppress block from rendering |

## Entry Points (same dict shape everywhere)

```python
# 1. Class-level
class MyAgent(Agent, llm=llm, context={
    "role": Context("Security expert", prefix=True),
    "self": None,  # suppress doc(self) block
}):
    pass

# 2. Instance-level
agent = MyAgent(context={"focus": "performance analysis"})

# 3. @strategy decorator (agent methods)
@strategy(CodeActStrategy(), context={
    "focus": "Write comprehensive tests",
    "state": None,
})
async def write_tests(self, code: str) -> str: ...

# 4. @strategy decorator (standalone functions)
@strategy(CodeActStrategy(), context={
    "role": "expert summarizer",
    "execution_context": None,
}, llm=llm)
async def summarise(text: str) -> str:
    """Summarise {text} in one sentence."""
    ...

# 5. Scoped (with-block, temporary)
with ScopedContext(context={"urgency": "high", "state": None}):
    result = await agent.analyze(data)

# 6. Runtime (inside agent code or constructors) — SAME TYPES
self.context["role"] = "You are an expert"
self.context["shell"] = Context(expr="doc(self.shell)", prefix=True)
self.context["status"] = Context(expr="f'{self.done}/{self.total}'")
self.context["self"] = None  # suppress
```

## Well-Known Block Keys

| Key | Source | Content | In standalone? |
|-----|--------|---------|---------------|
| `system_prompt` | Framework | Agent class docstring | ❌ |
| `self` | Framework | `doc(type(self))` — class API introspection | ❌ |
| `state` | Framework | Current instance field values (pformat) | ❌ |
| `strategy_prompt` | Strategy | Strategy instructions (## Strategy block) | ✅ |
| `execution_context` | Strategy/CodeAct | Available imports, types, functions | ✅ |

Suppress any of these with `context={"key": None}` or `self.context["key"] = None`.

## Prefix vs Suffix

- **Prefix** = cacheable across turns. Forms a stable shared prefix that LLM providers can cache. Use for blocks that rarely change (agent identity, tool docs, stable config).
- **Suffix** = volatile. Default. Changes between turns.

Rule of thumb:
- Set at class/instance init time → usually prefix-worthy
- Set per method/turn → usually suffix
- Use `prefix=True` explicitly when you swap something at runtime but know it's stable across multiple subsequent turns (e.g. hot-swapping shell tool docs)

## Inspecting Your Prompt

```python
# In agent code:
self.context.all_keys()      # All block keys (framework + strategy + user)
self.context.disabled()      # Currently suppressed keys
self.context.is_enabled("k") # Check if a specific key renders

# From outside:
agent.context_manager.keys()            # All registered keys
agent.context_manager.protected_keys    # Framework-protected keys
agent.context_manager.disabled_keys     # Suppressed set
```

## `Context` Class Reference

```python
class Context:
    def __init__(
        self,
        value: str | None = None,   # Literal text (mutually exclusive with expr)
        *,
        expr: str | None = None,    # Python expression re-evaluated each turn
        prefix: bool = False,       # Place in cacheable prefix if True
    ): ...
```

- `value` and `expr` are mutually exclusive (TypeError if both given)
- `expr` is validated at creation time (must be compilable Python)
- `prefix=False` is the default (volatile suffix)

## Legacy API (deprecated)

These still work but emit `DeprecationWarning`:

| Old | Replacement |
|-----|-------------|
| `self.context.set_static("k", "v")` | `self.context["k"] = Context("v", prefix=True)` |
| `self.context.set_static("k", expr="e")` | `self.context["k"] = Context(expr="e", prefix=True)` |
| `self.context.set_dynamic("k", "e")` | `self.context["k"] = Context(expr="e")` |
| `self.context.disable("k")` | `self.context["k"] = None` |
| `self.context.enable("k")` | `self.context["k"] = Context(expr="<original_expr>", prefix=True)` |
| `DynamicContext("expr")` | `Context(expr="expr")` |
