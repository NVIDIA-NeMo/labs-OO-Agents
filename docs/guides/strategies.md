# Strategies in nooa

## The Two Strategies

| Strategy | Default? | Use When | LLM Output |
|----------|----------|----------|------------|
| `PredictStrategy` | — | Single-shot LLM call; returns a Pydantic model; no code execution needed | JSON matching return type |
| `CodeActStrategy` | **Yes** | Needs code execution, tool calls, or iteration in a REPL | `execute_python()` tool + `return_result()` |

**The default** strategy (when no `@strategy` decorator is used) is `CodeActStrategy()`.

> Other strategies (`TemplateStrategy`, `PurePythonStrategy`, `ReflexionStrategy`) are experimental
> and not yet ready for production use.

## PredictStrategy

Use when the method returns a Pydantic model and the LLM only needs to produce structured output — no code execution required:

```python
from nooa import strategy
from nooa.strategies import PredictStrategy

class Intent(BaseModel):
    task_type: Literal["question", "feature", "bugfix"]
    summary: str

@strategy(PredictStrategy())
async def classify_intent(self, msg: str) -> Intent:
    """Classify {msg} into a task type and summarize it."""
    ...
```

The LLM must return valid JSON matching the return type. If it doesn't, the strategy retries.

**Use for:** classification, routing, extraction from text, any single-shot structured response.

**Don't use for:** tasks that need to run code, call tools, or iterate.

## CodeActStrategy Configuration

```python
@strategy(CodeActStrategy(
    max_iterations=50,     # Max tool calls before stopping
    max_retries=3,         # Max consecutive errors before failure
    allow_text_response=True,  # Allow plain text without tool calls
))
async def respond(self, msg: str) -> None:
    """..."""
    ...
```

**`return_result()` from within `execute_python()`**: The LLM can call `return_result(value)` inside a code block to compute and return in a single tool call.

## Methods Without Strategy (Pure Python Orchestration)

Any method WITHOUT an ellipsis body executes directly — no LLM, no strategy:

```python
async def orchestrate(self, msg: str):
    """This runs as normal Python — no LLM involved."""
    intent = await self.classify(msg)  # This calls LLM
    if intent == "question":
        return await self.answer(msg)  # This calls LLM
    # The orchestration logic itself is deterministic Python
```

Use this for **workflow enforcement** — the sequence of steps is hardcoded in Python, each step dispatches to an LLM-powered method.

## Choosing the Right Strategy per Method

| Method Purpose | Strategy | Why |
|---------------|----------|-----|
| Classification/routing | `PredictStrategy` | Single-shot, no code needed, fast |
| Extraction from text | `PredictStrategy` | Returns Pydantic model, no REPL needed |
| Exploration/brainstorming | `CodeActStrategy(allow_text_response=True)` | May need to read files, run commands |
| Implementation | `CodeActStrategy(max_iterations=50)` | Needs code execution, iteration |
| Verification | `CodeActStrategy(max_iterations=10)` | Run tests, check output |
| Simple Q&A | `CodeActStrategy(allow_text_response=True)` | May or may not need tools |
