# Writing Generation Methods

**Status**: Guide
**Date**: 2025-12-17

## Overview

Generation methods are methods where the LLM "fills in" the implementation. With the metaclass-based system, methods with ellipsis (`...`) bodies are automatically wrapped for generation - no decorator needed!

```python
from nooa import Agent

class MyAgent(Agent, llm=my_llm):
    async def analyze(self, data: str) -> Analysis:
        """Analyze the input data and return structured insights.

        Consider:
        1. Key patterns in the data
        2. Anomalies or outliers
        3. Actionable recommendations
        """
        ...  # LLM generates this
```

The `...` (ellipsis) is the key - it signals that this method needs LLM generation. The `AgentMeta` metaclass uses AST inspection to detect this pattern at class creation time and automatically wraps the method.

## Anatomy of a Generation Method

The metaclass detects ellipsis bodies and wraps them automatically. To override the default strategy, use `@strategy`:

```python
from nooa import strategy
from nooa.strategies import ReflexionStrategy

@strategy(ReflexionStrategy(max_iterations=5))
async def complex_task(self): ...
```

### 2. The Signature

```python
async def method_name(self, arg1: Type1, arg2: Type2) -> ReturnType:
```

- **`async`**: Generation methods are always async (LLM calls are I/O)
- **`self`**: First parameter is always the agent instance
- **Parameters**: Typed arguments that become available in the prompt context
- **Return type**: Defines what the LLM should produce

### 3. The Docstring (The Prompt)

The docstring IS the prompt. It tells the LLM what to do:

```python
async def summarize(self, text: str, max_words: int = 100) -> str:
    """Summarize the text in max_words words or less.

    Provide a concise summary that captures the main points.
    """
    ...
```

**Don't interpolate parameters (`{text}`, `{max_words}`) into the docstring.**
The framework renders arguments to the LLM by default: the signature accompanies
the task, and the default CodeAct prefill pprint()s each parameter value under
the truncation config (the values are also live variables in the REPL); Predict
serializes parameters with size caps. `{param}` re-injects the raw value into
the instructions — redundant, untruncated, and it turns untrusted data into
prompt text. Reserve `{...}` templating for `{self.attr}` instance state and
computed expressions like `{len(items)}` (see `docs/guides/prompt-mechanics.md`).

### 4. The Body

```python
    ...  # Ellipsis indicates "LLM fills this in"
```

The `...` (ellipsis) is the convention for "this is a generation method" - the implementation comes from the LLM.

---

## Two Contexts: Agent Methods vs Strategy Methods

Generation methods work in two contexts:

### On Agent Classes

When used on an Agent class method, `self` is the agent instance and the runtime is implicit:

```python
class Analysis(BaseModel):
    key_patterns: list[str]
    anomalies: list[str]
    recommendation: str

class DataAnalyzer(Agent, llm=my_llm):
    async def analyze(self, data: str) -> Analysis:
        """Analyze the data and return structured insights."""
        ...

    async def summarize(self, text: str) -> str:
        """Create a concise summary of the text."""
        ...
```

### On Strategy Classes

Strategy methods can also use ellipsis for generation. They must receive `runtime` as the first parameter after `self`.

**Key difference:** Strategy methods require explicit `runtime` parameter, Agent methods do not.

---

## Choosing a Strategy

By default, ellipsis methods use `CodeActStrategy()`. Override with `@strategy`:

```python
from nooa import Agent, strategy
from nooa.strategies import PredictStrategy

class MyAgent(Agent, llm=my_llm):
    # Default strategy (CodeActStrategy) — code execution + iteration
    async def task1(self): ...

    # Single-shot prediction — use when method returns Pydantic model, no code needed
    @strategy(PredictStrategy())
    async def classify(self, msg: str) -> Intent: ...
```

### Strategy Selection Guide

| Strategy | Use When | LLM Calls |
|----------|----------|-----------|
| **`CodeActStrategy`** (default) | Needs code execution, tools, or iteration | Multiple |
| `PredictStrategy` | Single-shot; returns Pydantic model; no code execution needed | Single |

---

## Reserved Parameter Names

`reasoning` is **reserved**: declaring it as a generation-method parameter raises
`ValueError` at class creation. Chain-of-thought is provided through the
`reasoning()` builtin available in CodeAct-generated code, not a parameter.

```python
# ❌ ERROR - reserved name (raises ValueError at class creation)
async def bad_method(self, reasoning: str): ...

# ✅ OK - use different name
async def good_method(self, rationale: str): ...
```

---

## Private and Public Methods

All ellipsis methods are auto-wrapped, and **all methods are traced by default** —
public, private, and dunder. Private methods are still hidden from `doc(self)`,
but their calls appear in traces:

```python
class MyAgent(Agent, llm=my_llm):
    async def public_task(self):
        """Public → Generated + Traced + visible in doc(self)"""
        ...

    async def _private_helper(self):
        """Private → Generated + Traced, hidden from doc(self)"""
        ...
```

To opt-out of tracing for any method:

```python
from nooa import no_trace

class MyAgent(Agent, llm=my_llm):
    @no_trace
    async def utility(self):
        """Public but NOT traced"""
        ...
```

---

## Common Patterns

### Calling Other Generation Methods

Generation methods can call each other:

```python
class Analyzer(Agent, llm=my_llm):
    async def analyze(self, text: str) -> dict:
        """Main analysis method."""
        sentiment = await self.get_sentiment(text)
        entities = await self.extract_entities(text)
        return {"sentiment": sentiment, "entities": entities}

    async def get_sentiment(self, text: str) -> str:
        """Determine sentiment: positive, negative, or neutral."""
        ...

    async def extract_entities(self, text: str) -> list[str]:
        """Extract named entities from the text."""
        ...
```

### Using agentdoc for Context

Use `agentdoc` functions in your docstrings (`doc` is auto-injected in generated
code; import it from `nooa.agentdoc` in your own modules):

```python
from nooa.agentdoc import doc

class MyAgent(Agent, llm=my_llm):
    async def task(self, data: str):
        """Process the data.

        Full documentation:
        {doc(self)}
        """
        ...
```

---

## Best Practices

1. **Clear docstrings** - The docstring is the prompt. Be explicit about what you want.
2. **Type hints** - Use type hints for parameters and return values. They help the LLM understand expectations.
3. **Break down complex tasks** - Create helper methods rather than one giant generation method.
4. **Use appropriate strategies** - Choose the strategy that fits your use case.
5. **Test with real LLMs** - Don't just test with FakeLLMClient - verify with real models.

---

## Migration from Old API

If you're updating code from the old `@agent` and `@plan` decorators:

**Before:**
```text
@agent(llm=my_llm)
class MyAgent(Agent):
    @plan
    async def task(self): ...

    @plan(strategy=ReflexionStrategy())
    async def complex(self): ...
```

**After:**
```python
from nooa import Agent, strategy
from nooa.strategies import ReflexionStrategy

class MyAgent(Agent, llm=my_llm):
    async def task(self): ...  # No decorator needed!

    @strategy(ReflexionStrategy())
    async def complex(self): ...
```

The metaclass automatically wraps all ellipsis methods - you only need `@strategy` when overriding the default strategy.
