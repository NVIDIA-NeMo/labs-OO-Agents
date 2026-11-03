# Multi-agent systems

Use another agent when a piece of work needs its own model interaction history,
context, tool state, or reusable role. A subagent is an ordinary Python object,
not a special node type.

## Isolation is the main reason to split

Each properly isolated worker should own:

- per-instance event history and context blocks;
- a generation lock and a fresh CodeAct REPL session for each agentic method
  call (REPL variables persist across cells within that call, not across calls);
- stateful tools and connections constructed on that instance;
- visible fields and role instructions.

Python still permits class-level objects, so the framework cannot guarantee
this isolation if a mutable tool or connection is assigned on the class.

A child does not automatically inherit the parent's history or context. Pass
the data it needs through method arguments or shared application objects.

This isolation lasts for the lifetime of the Python objects. Default storage is
in memory, so a process restart or a newly constructed agent does not resume the
old history automatically. Durable agent state requires an explicit storage and
snapshot/restore workflow; semantic memory is a separate opt-in capability. See
[Prompts and context](prompts-and-context.md#lifetime-persistence-and-memory).

```python
class Researcher(Agent):
    async def research(self, question: str) -> list[str]:
        """Find evidence relevant to the question."""
        ...


class Writer(Agent):
    async def write(self, question: str, evidence: list[str]) -> str:
        """Write an answer supported only by the supplied evidence."""
        ...
```

The signatures make the handoff explicit. The writer does not silently depend
on the researcher's prompt history.

## Coordinate agents with Python

```python
class AnswerPipeline:
    def __init__(self, llm):
        self.researcher = Researcher(llm=llm)
        self.writer = Writer(llm=llm)

    async def run(self, question: str) -> str:
        evidence = await self.researcher.research(question)
        return await self.writer.write(question, evidence)
```

The coordinator does not need to inherit from `Agent`; it contains no agentic
method. Use normal Python for routing, timeouts, fallback models, and error
handling.

## Use one worker instance per concurrent task

The built-in Predict and CodeAct strategies serialize generation calls on one
agent instance. Independent fan-out should use independent instances:

```python
import asyncio


async def analyze_all(llm, targets: list[str]) -> list[str]:
    workers = [Reviewer(llm=llm) for _ in targets]
    return await asyncio.gather(
        *(worker.review(target) for worker, target in zip(workers, targets, strict=True))
    )
```

Separate worker instances isolate per-agent state only when mutable dependencies
are also constructed per worker. A mutable class attribute is shared by normal
Python semantics, just as it would be in any `asyncio` application.

Separate agent and tool instances do not isolate an external resource they all
point at. Concurrent writers need separate worktrees, sandboxes, database
namespaces, or deterministic coordination around the shared resource.

## Model inheritance and explicit construction

A child agent created inside an active parent agent call can inherit the
parent's LLM when its class does not declare one. Application coordinators and
agent constructors should normally pass `llm=` explicitly because they run
outside that active call context.

Explicit construction is easier to understand and makes per-role model choices
visible:

```python
self.researcher = Researcher(llm=accurate_llm)
self.writer = Writer(llm=fast_llm)
```

## When not to create another agent

Use a regular method when the operation has exact semantics. Use another
agentic method on the same object when the task should share the same role,
history, and tools. Use a child agent when isolation or a reusable role is the
point.

Too many agents create more prompts, histories, model calls, and failure
boundaries. Splitting every function into an agent is not a scalability
strategy.

## Compared with supervisors and graph nodes

You can implement supervisors, routers, debate patterns, and worker pools in
NOOA. The supervisor is usually a Python orchestrator or a focused routing
method, and workers are ordinary agent instances. This keeps state ownership
and handoffs visible in Python instead of hiding them in framework-managed
graph state.

## Common mistakes

- Assuming child agents inherit context blocks or conversation history.
- Sharing one stateful tool or one agent instance across parallel jobs.
- Passing implicit global state instead of typed arguments.
- Using an LLM supervisor for routing that deterministic Python can express.
- Creating many roles without a clear isolation or specialization benefit.

## Continue

- [Orchestration](orchestration.md)
- [Tools and visibility](tools-and-visibility.md)
- [Tracing](tracing.md)
- Tour: [Objects compose into multi-agent systems](../tour.md#5-objects-compose-into-multi-agent-systems)
