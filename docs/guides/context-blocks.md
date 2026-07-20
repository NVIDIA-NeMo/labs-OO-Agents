# Context Blocks in nooa

## The Model

Context blocks follow the `Block(key, expr, update, show)` model:

- **key**: Unique identifier for the block
- **expr**: Python expression (as string) evaluated at render time, OR a static value
- **update**: When to re-evaluate (`"always"` = dynamic, `"once"` = static)
- **show**: Whether to include in the prompt (`True`/`False`)

## Static vs Dynamic

```python
# Static: set once, never changes
self.context["plan"] = plan.format()

# Dynamic: re-evaluated each LLM turn
self.context.set_dynamic("project_state", "self.format_project_state()")
```

Dynamic blocks are useful for showing live status that changes as the agent works (e.g., which plan step is current, test results so far).

## Context Blocks in Orchestration

In a single agent, context blocks persist across method calls. Use this for passing state between phases:

```python
class MyAgent(Agent, llm=llm):
    async def orchestrate(self, msg: str):
        spec = await self.brainstorm(msg)

        # Set context that all subsequent methods see as SYSTEM blocks
        self.context["decisions"] = spec.format()

        plan = await self.write_plan(spec)
        self.context["plan"] = plan.format()

        # implement() sees both "decisions" and "plan"
        for step in plan.steps:
            await self.implement(step)
```

## Key Property

Context blocks are **per-agent-instance**. They are NOT shared across agents. Each agent has its own `ContextManager`. See `docs/guides/single-vs-multi-agent.md` for implications.
