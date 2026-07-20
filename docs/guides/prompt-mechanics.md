# How Prompts Work in nooa

## The Ellipsis IS the Key

A method becomes a **generation method** (LLM-powered) when its body is `...` (ellipsis). The `@strategy` decorator is optional — it only overrides the default strategy.

```python
# This IS a generation method (ellipsis body, default CodeActStrategy)
async def analyze(self, data: str) -> AnalysisResult:
    """Analyze the data for anomalies."""
    ...

# This IS a generation method (ellipsis body, explicit strategy override)
@strategy(PredictStrategy())
async def analyze(self, data: str) -> AnalysisResult:
    """Analyze the data for anomalies."""
    ...

# This is NOT a generation method (has a real body)
async def orchestrate(self, msg: str):
    """Pure Python — no LLM involved."""
    result = await self.analyze(msg)
    return result
```

## The Docstring IS the Prompt

When a generation method is called, the runtime:

1. Extracts the method's `__doc__` (docstring)
2. Expands `{...}` template expressions (e.g. `{self.attr}`)
3. Wraps it in a Task event sent to the LLM as a USER message

```python
@strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))
async def analyze(self, data: str) -> AnalysisResult:
    """Analyze the data for anomalies.
    Focus on outliers and unexpected patterns.
    Return a structured summary."""
    ...
```

When called as `await agent.analyze("sales Q4")`, the LLM sees:

```
<user>
# Your task
Analyze the data for anomalies.
Focus on outliers and unexpected patterns.
Return a structured summary.

## Method signature:
async def analyze(self, data: str) -> AnalysisResult

Please perform the task now.
</user>
```

## Arguments Are Rendered By Default — Don't Write `{param}`

The LLM **already sees every argument value** without any templating:

- The method signature is included with the task (above).
- The default CodeAct prefill (`InspectInputsPrefill`) pprint()s each parameter
  under the truncation config, and the actual values are live variables in the
  REPL — the model can slice, inspect, and pass them around.
- Predict serializes parameters into the prompt with size caps.

Writing `{data}` in a docstring is technically supported (the expansion in step 2
accepts any expression, parameters included) but is almost always wrong:

- **Redundant** — the value is already rendered; you pay for it twice.
- **Unprotected** — docstring expansion injects the *raw* value: no smart
  truncation, so a large argument blows up the context window.
- **Injection surface** — untrusted argument content becomes part of the
  instruction text instead of staying clearly-delimited data.

Reserve `{...}` templating for what the signature cannot show: `{self.attr}`
instance state and computed expressions like `{len(items)}`.

## Reserved Parameter Names

- **`reasoning`** is reserved: declaring it as a generation-method parameter raises `ValueError` at class creation. Chain-of-thought is surfaced through the `reasoning()` builtin available to the LLM in CodeAct-generated code, not through a parameter.

## Full Prompt Structure

The LLM receives blocks in this order:

| Role | Block | Source |
|------|-------|--------|
| SYSTEM | `system_prompt` | `agent._system_prompt()` — agent identity |
| SYSTEM | `self` | `doc(self)` — auto-generated introspection of agent class |
| SYSTEM | `context_api` | `doc(self.context)` — context block API docs |
| SYSTEM | `events_api` | `doc(self.events)` — event query API docs |
| SYSTEM | `strategy_prompt` | `strategy.strategy_instructions()` — execution model |
| SYSTEM | `execution_context` | `strategy.execution_context()` — available imports/types |
| SYSTEM | user blocks | `self.context["key"]` — any blocks you set |
| USER | task | Method docstring (the actual instructions) |
| USER/ASSISTANT | events | Conversation history (messages, code executions, etc.) |

## `doc(self)` — Agent Introspection

The runtime auto-generates API documentation of the agent class and includes it as a SYSTEM block. This means the LLM sees:
- All public SW1 methods (deterministic tools) with their signatures and docstrings
- All public attributes
- The agent's class hierarchy

This is why SW1 helper methods are discoverable without explicitly telling the LLM about them in every docstring (though being explicit is still recommended).

**Private methods** (prefixed with `_`) are not included in `doc(self)` — use them for internal logic that shouldn't be visible to the LLM. (They ARE still traced by default; opt out with `@no_trace`.)

## Context Blocks vs Docstrings

- **Docstring**: Specific task instructions for THIS method call (arguments are rendered separately by default — see above). Appears as USER role.
- **Context blocks** (`self.context["key"] = value`): Persistent or dynamic supplementary information. Appears as SYSTEM role. Available across all method calls on the same agent instance.

## Implication for Agent Design

Since the docstring is the prompt, the quality of your agent is largely determined by:
1. **What methods exist** — each method is a distinct LLM task
2. **What each docstring says** — the instructions the LLM follows
3. **What return type each method has** — forces the LLM to produce structured output
4. **What context blocks are set** — supplementary information available during execution

There is no separate "prompt template" system. The code IS the configuration.
