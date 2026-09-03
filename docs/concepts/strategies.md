# Strategies

A strategy decides how an agentic method uses an LLM. It does not change the
method's Python interface: callers still pass the same arguments and receive
the declared return type.

NOOA's two primary strategies cover most applications.

| Strategy | Use it when | Execution model |
|---|---|---|
| `PredictStrategy` | A non-tool attempt can classify, extract, or produce the typed answer | Structured LLM attempt followed by validation; invalid output may be retried |
| `CodeActStrategy` | The task needs tools, live Python objects, code execution, or iteration | Repeated LLM turns in a per-call Python REPL; locals persist across cells |

CodeAct is the default. Select Predict explicitly when the task does not need
the extra loop.

## Predict: one focused judgment

```python
from typing import Literal

from nooa import strategy
from nooa.strategies import PredictStrategy


class Router(Agent, llm=llm):
    @strategy(PredictStrategy())
    async def route(self, message: str) -> Literal["sales", "support", "abuse"]:
        """Choose the team that should receive the message."""
        ...
```

Predict serializes the inputs, asks the provider for structured output, and
validates it. It cannot call methods or execute Python during that attempt.
Validation failures can cause additional provider attempts, but there is no
iterative tool-using loop.

## CodeAct: inspect, act, and iterate

```python
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy


class RepositoryAgent(Agent, llm=llm):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=12)))
    async def investigate(self, question: str) -> str:
        """Investigate the repository and answer with source evidence."""
        ...
```

CodeAct gives the model a Python execution loop. Method arguments are live REPL
variables, helper definitions persist across cells, and generated code can call
visible methods and tools on `self`. The loop ends when a value validates
against the return annotation.

## Strategy selection is not model selection

The strategy controls the interaction pattern. The LLM setting controls which
model performs it. You can change either independently:

```python
class ResearchAgent(Agent, llm=fast_llm):
    @strategy(PredictStrategy(), llm=accurate_llm)
    async def extract_claims(self, text: str) -> list[str]:
        """Extract independently verifiable claims."""
        ...
```

The `llm=` override also accepts a registry alias or model string
(``@strategy(PredictStrategy(), llm="gpt-5-mini")``) — resolved lazily on the
first call per instance, so declaring it constructs no client at import time.

LLM overrides may also be supplied per instance or per call. Keep model routing
out of the public method contract unless the application truly needs callers to
choose it.

## Prefill is a useful middle ground

Code before the final ellipsis runs deterministically before the first CodeAct
turn. Its variables and output become available in the REPL:

```python
async def review_open_items(self) -> str:
    """Review the open items and recommend the next action."""
    open_items = [item for item in self.items if item.is_open]
    ...
```

Use prefill to prepare inputs for one agentic task. Use a real Python
orchestrator when several agentic stages must run in a fixed order.

## Concurrency

Predict and CodeAct lock an agent instance while a generation call is active.
Calling several agentic methods concurrently on the same instance therefore
serializes by default. Create one agent instance per independent concurrent
task. Custom strategies can opt out of locking when they do not share mutable
runtime state.

## Common mistakes

- Using default CodeAct for a simple typed classification.
- Choosing Predict for work that must call tools or inspect live objects.
- Increasing `max_iterations` instead of decomposing an oversized method.
- Passing configuration fields directly to a strategy constructor; advanced
  options belong in `CodeActConfig` or `PredictConfig`.
- Treating a strategy as part of the caller-facing API.

## Continue

- [Agents and methods](agents-and-methods.md)
- [Orchestration](orchestration.md)
- Runnable example: [strategy selection](../../examples/quickstart/04_strategies.py)
- Advanced reference: [`nooa-codeact-advanced`](../../skills/nooa-codeact-advanced/SKILL.md)
