# A 10-minute tour of NVIDIA-labs Object Oriented Agents

**What if a Python method could think?** NVIDIA-labs Object Oriented Agents
(NOOA) starts with that idea and keeps ordinary Python in charge. Classes define
roles, methods define capabilities, type annotations define contracts, and an
ellipsis marks the work delegated to an LLM.

This is a conceptual tour of what makes NOOA different. For a runnable,
step-by-step path, use the
[notebook tutorials](../notebook_tutorials/README.md).

> NOOA can execute LLM-generated Python. Run CodeAct agents inside an OS-level
> sandbox such as a container, VM, or
> [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell). Framework checks are
> defense in depth, not a containment boundary.

## 1. A Python method that thinks

Choose a model supported by LiteLLM and put `...` in an async method:

```python
import asyncio

from nooa import Agent
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client("gpt-5-mini")


class FeedbackAgent(Agent, llm=llm):
    """Analyze customer feedback faithfully and concisely."""

    async def analyze(self, text: str) -> str:
        """Summarize the sentiment and key topics in one sentence."""
        ...


async def main():
    agent = FeedbackAgent()
    result = await agent.analyze("Great product, but shipping was slow.")
    print(result)


asyncio.run(main())
```

Three familiar pieces of Python define the model call:

- `...` means the LLM implements this method at runtime. A real method body
  remains deterministic Python.
- The name, signature, and docstring describe the task. Arguments are already
  shown to the model, so they do not belong in `{placeholder}` prompts.
- The return annotation defines what the caller receives.

The caller does not invoke a graph, chain, or tool registry. It calls a method
on an object. That small boundary is the foundation of the framework.

Run this idea in
[Notebook 1: Your First Object-Oriented Agent](../notebook_tutorials/01_your_first_agent.ipynb),
or open the compact
[first-generation quickstart](../examples/quickstart/01_first_generation_method.py).

## 2. Ordinary methods become tools

An agentic method can use the object's other visible methods. Deterministic
Python is therefore the simplest tool layer:

```python
class InventoryAgent(Agent, llm=llm):
    def __init__(self, inventory: dict[str, int], **kwargs):
        super().__init__(**kwargs)
        self._inventory = inventory

    def get_stock(self, item: str) -> int:
        """Return the number of units currently in stock."""
        return self._inventory.get(item, 0)

    async def answer(self, question: str) -> str:
        """Answer the inventory question using current stock information."""
        ...
```

With the default `CodeActStrategy`, the LLM works in a Python REPL created for
that method call. It can inspect `self`, call `self.get_stock(...)`, calculate,
and iterate. REPL locals persist across generated cells within the call, while
the inventory remains one live Python object instead of being copied into every
prompt.

NOOA documents the visible surface through `doc(self)`, so the model can
discover capabilities as it needs them. This is progressive disclosure using
Python's object model: expose a small set of useful verbs and keep bulky or
sensitive implementation state private.

Explore the live-object workflow in
[Notebook 3: CodeAct REPL and Pass by Reference](../notebook_tutorials/03_codeact_tools_and_live_objects.ipynb).
The [tools and visibility guide](concepts/tools-and-visibility.md) explains the
full visibility rules and stateful-tool lifecycle.

## 3. Types become contracts

Free-form strings are useful for prose. When another part of the program needs
to consume the result, return a typed model:

```python
from typing import Literal

from pydantic import BaseModel, Field

from nooa import strategy
from nooa.strategies import PredictStrategy


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)


class Classifier(Agent, llm=llm):
    @strategy(PredictStrategy())
    async def classify(self, text: str) -> FeedbackAnalysis:
        """Classify the customer feedback."""
        ...
```

NOOA validates the provider output against the return type. Validation errors
can trigger another provider attempt, but Predict has no Python or tool loop.
Use it for focused classification and extraction. Use CodeAct when the method
must inspect live objects, call tools, execute code, or iterate.

The strategy changes how the method is implemented, not how callers use it.
Both remain ordinary awaited methods returning validated Python values.

[Notebook 2: Choosing a Strategy](../notebook_tutorials/02_choosing_a_strategy.ipynb)
develops this choice interactively. The
[strategy guide](concepts/strategies.md) covers configuration and concurrency.

## 4. Python remains the control plane

LLMs are useful for judgment; Python is better for rules that must always run.
Keep each generation method focused on one LLM task, then put sequencing,
retries, authorization, and verification in real method bodies:

```python
from pathlib import Path

from pydantic import BaseModel, Field

from nooa import hidden


class Finding(BaseModel):
    file: str
    line: int = Field(ge=1)
    evidence: str


class ReviewAgent(Agent, llm=llm):
    def __init__(self, repo: Path, **kwargs):
        super().__init__(**kwargs)
        self._repo = repo.resolve()

    async def find_issues(self, target: str) -> list[Finding]:
        """Find concrete issues and cite exact source evidence."""
        ...

    def evidence_exists(self, finding: Finding) -> bool:
        path = (self._repo / finding.file).resolve()
        if not path.is_relative_to(self._repo) or not path.is_file():
            return False
        lines = path.read_text().splitlines()
        return finding.line <= len(lines) and finding.evidence in lines[finding.line - 1]

    @hidden
    async def run(self, target: str) -> list[Finding]:
        candidates = await self.find_issues(target)
        return [finding for finding in candidates if self.evidence_exists(finding)]
```

The schema checks properties contained in the returned value. Python verifies
claims that depend on files, tests, databases, APIs, permissions, or other
external state. The deterministic `run()` method makes the evidence gate
unskippable and is hidden because generated code does not need to call the
whole workflow recursively.

This is the central reliability pattern: **use the model for judgment and make
Python enforce the workflow.** It remains easy to test because the control flow
is ordinary code.

See [Orchestration](concepts/orchestration.md) for decomposition and evidence
gates, and [Safety](concepts/safety.md) for the actual process and authorization
boundaries.

## 5. Objects compose into multi-agent systems

A subagent is another Python object with its own role, context, history, and
tools. Pass handoffs through typed method arguments and coordinate them with
normal Python:

```python
import asyncio

from pydantic import BaseModel


class Review(BaseModel):
    perspective: str
    concerns: list[str]


class ReviewRole(Agent, llm=llm):
    async def respond(self, change: str) -> Review:
        """Review the change from this role's perspective."""
        raise NotImplementedError


class SecurityReviewer(ReviewRole):
    async def respond(self, change: str) -> Review:
        """Review authentication, authorization, data exposure, and abuse paths."""
        ...


class ReliabilityReviewer(ReviewRole):
    async def respond(self, change: str) -> Review:
        """Review failure modes, recovery, observability, and operations."""
        ...


class ReviewPanel:
    def __init__(self, reviewers: list[ReviewRole]):
        self.reviewers = reviewers

    async def run(self, change: str) -> list[Review]:
        return await asyncio.gather(
            *(reviewer.respond(change) for reviewer in self.reviewers)
        )


panel = ReviewPanel([SecurityReviewer(), ReliabilityReviewer()])
reviews = await panel.run("Add passwordless account recovery")
```

`ReviewPanel` calls the same `respond()` interface on every object. Normal
Python dispatch selects the subclass implementation, even though those
implementations happen to be LLM-driven. The coordinator does not need to
inherit from `Agent`; it has no LLM-driven method.

The same pattern supports sequential handoffs, `asyncio.gather`, timeouts,
fallback models, queues, and approval steps. Separate worker instances isolate
agent and tool state, while shared files, databases, and services still require
normal coordination or separate sandboxes.

[Notebook 4: Composing Subagents](../notebook_tutorials/04_composing_subagents.ipynb)
shows sequential, parallel, and model-directed composition. The
[multi-agent guide](concepts/multi-agent-systems.md) covers isolation,
persistence, and concurrency in more depth.

## 6. The NOOA design in one view

The framework's shape can be summarized in six rules:

1. An ellipsis delegates one method to an LLM.
2. A real body keeps exact behavior in Python.
3. Public methods form the agent's discoverable capability surface.
4. Return types turn model output into validated program data.
5. Strategies choose the lightest execution loop for each method.
6. Objects and ordinary Python compose the complete system.

That is the whole progression: begin with one method that thinks, add a narrow
Python surface around it, and introduce more objects only when work needs an
independent role or state boundary.

Continue according to how you prefer to learn:

- [Notebook tutorials](../notebook_tutorials/README.md) — the primary hands-on
  learning path.
- [Concept guides](README.md#concepts-at-a-glance) — focused explanations of
  agents, strategies, context, tracing, orchestration, and safety.
- [Architecture](architecture.md) — follow one call through prompt assembly,
  execution, validation, events, and traces.
- [Examples catalog](../examples/README.md) — compact runnable programs by
  capability.
- [Agent-authoring skill](../skills/nooa-agent-authoring/SKILL.md) — detailed
  reference for coding agents and experienced contributors.
