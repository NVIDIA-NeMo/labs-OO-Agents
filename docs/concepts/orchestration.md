# Orchestration

Agentic methods are good at judgment. Python is better at enforcing sequence,
branching, retries, concurrency, and acceptance criteria. A reliable NOOA
workflow uses both.

## Put the workflow in a real method

```python
from pathlib import Path

from pydantic import BaseModel, Field

from nooa import hidden


class Finding(BaseModel):
    path: str
    line: int = Field(ge=1)
    evidence: str


class ReviewAgent(Agent, llm=llm):
    def __init__(self, root: Path, **kwargs):
        super().__init__(**kwargs)
        self._root = root.resolve()

    async def discover(self, target: str) -> list[Finding]:
        """Find candidate issues and cite the relevant source text."""
        ...

    def evidence_exists(self, finding: Finding) -> bool:
        path = (self._root / finding.path).resolve()
        if not path.is_relative_to(self._root) or not path.is_file():
            return False
        lines = path.read_text().splitlines()
        return finding.line <= len(lines) and finding.evidence in lines[finding.line - 1]

    @hidden
    async def run(self, target: str) -> list[Finding]:
        candidates = await self.discover(target)
        return [item for item in candidates if self.evidence_exists(item)]
```

`discover()` delegates fuzzy analysis to the LLM. `evidence_exists()` performs
an exact check. `run()` guarantees that verification happens before findings
leave the workflow.

The outer method is hidden because generated code does not need to call the
whole workflow recursively. It is still an agent method and remains visible in
traces.

## Types validate values; Python validates the world

Pydantic can enforce that a line number is positive or a category is one of
three values. It cannot prove that a cited file exists, a test passed, a row was
saved, or an API accepted a change.

Use deterministic gates for claims about external state:

```python
result = await self.implement(request)
test_run = await self.shell.run("pytest -q")
if not test_run.success:
    raise RuntimeError("Implementation did not pass verification")
return result
```

Do not ask the model to assert that it verified something and treat the
assertion as evidence.

## One method, one LLM task

Break a large workflow into focused judgments:

```python
@hidden
async def run(self, request: str) -> FinalResult:
    plan = await self.plan(request)
    change = await self.implement(plan)
    review = await self.review(change)
    return self.accept(change, review)
```

Each phase has its own signature, prompt, return contract, trace span, and test
surface. Python makes the order non-negotiable.

## Prefill and orchestration solve different problems

Code before an ellipsis prepares one agentic call and leaves values in its REPL.
A real orchestrator calls several methods and controls their relationship.

Use prefill to calculate `open_items` before one judgment. Use orchestration to
require plan → implementation → review → verification.

## You do not need an Agent for pure control flow

If a coordinator has no agentic methods, make it an ordinary Python class. It
can own agent objects and call them:

```python
class Pipeline:
    def __init__(self, researcher: Researcher, writer: Writer):
        self.researcher = researcher
        self.writer = writer

    async def run(self, topic: str) -> Article:
        sources = await self.researcher.find_sources(topic)
        return await self.writer.write(topic, sources)
```

This is often the clearest translation of a chain or graph: edges become normal
Python calls, conditions become `if`, and fan-out becomes `asyncio.gather`.

## Common mistakes

- Putting the entire workflow in one giant agentic method.
- Asking the LLM to remember mandatory steps instead of encoding them in Python.
- Accepting model-authored evidence without checking external state.
- Performing important preprocessing in module-level code, where it is absent
  from the agent call trace.
- Subclassing `Agent` for a class that contains only deterministic coordination.

## Continue

- [Multi-agent systems](multi-agent-systems.md)
- [Tracing](tracing.md)
- [Safety](safety.md)
- [Architecture](../architecture.md)
- Runnable example: [tracing quickstart](../../examples/quickstart/06_tracing.py)
- Guided tutorial: [composing subagents](../../notebook_tutorials/04_composing_subagents.ipynb)
