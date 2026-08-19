# Context Management in NOOA

## Goal

Use one pattern for context produced by agents, skills, and future components. One agent view defines the complete model context. It may explicitly compose other views. Rendering adds no policy.

## Contract

```python
class Block:
    key: str
    content: str
    role: Role = Role.SYSTEM


ContextItem = Block | EventBase
AssembledContext = tuple[ContextItem, ...]


class ContextView[Owner](Protocol):
    def assemble(
        self,
        owner: Owner,
        call: Call,
    ) -> AsyncIterator[ContextItem]: ...
```

`Block.content` is materialized. Events remain typed. The runtime collects the selected agent view into an immutable tuple before rendering. `Call` is the immutable generation snapshot.

## Resolution

Highest precedence wins and replaces the lower view:

```text
Agent: call -> method -> instance -> class -> DefaultAgentView
Skill:        instance -> class -> DefaultSkillView
```

```python
class MyAgent(Agent, context_view=AgentView()): ...
agent = MyAgent(context_view=OtherAgentView())
await agent.run(context_view=CallView())


class SearchSkill(Skill, context_view=SearchView()): ...
skill = SearchSkill(context_view=OtherSearchView())
```

`@strategy(..., context_view=view)` is the method override. Skills have no method or call override. Composition is ordinary Python; there is no merge protocol.

## Defaults

```python
class DefaultAgentView(ContextView[Agent]):
    async def assemble(self, agent, call):
        prefix, trailing = await agent.context_manager.materialize(call)
        strategy = tuple(
            [block async for block in call.strategy.context_blocks(agent, call)]
        )

        for block in default_prefix_order(prefix, strategy):
            yield block

        for skill in agent.active_skills():
            view = resolve_context_view(skill, default=DefaultSkillView())
            async for item in view.assemble(skill, call):
                yield item

        for event in agent.event_manager.events_for_context(call.event_query):
            yield event

        for block in trailing:
            yield block


class DefaultSkillView(ContextView[Skill]):
    async def assemble(self, skill, call):
        if skill.context_block is not None:
            yield await materialize(skill.context_block, owner=skill, call=call)
```

The agent view owns skill selection and placement. A skill view owns only its emitted items. Registry skills participate when active; directly attached public skills are active by default. Hidden and inactive skills contribute nothing.

`DefaultSkillView` emits only explicitly declared context. It does not dump skill documentation. A custom skill view may emit any ordered `Block | EventBase` sequence.

## Ownership

```text
resolve agent view
    -> assemble agent and selected component views
    -> tuple[Block | EventBase, ...]
    -> ContextRenderer
    -> ProviderFormatter
```

- The selected agent view owns membership, materialization, filtering, global order, and budget policy.
- A nested view owns the content and local order of its contribution.
- The renderer expands each item in place and serializes it.
- The provider formatter only adapts message shape and cache annotations.
- Downstream stages preserve semantics or raise `UnsupportedContextLayout`; they never reorder, omit, resolve, evict, or repair context.

## Migration

Keep `agent.context`, context managers, event creation, and skill activation as state APIs. Stop giving them implicit prompt behavior.

`Skill.context_block = (key, expression)` becomes shorthand consumed by `DefaultSkillView`; activation no longer writes it into `ContextManager`. Specialized skills replace this shorthand with a class- or instance-level view.

Replace scoped context overrides and formatter-owned partitioning with views. A temporary adapter may feed assembled items to current formatters, but the final formatter boundary must preserve order.

## Required tests

- immutable, stable assembly order;
- agent and skill precedence;
- inactive and hidden skills contribute nothing;
- legacy `Skill.context_block` materializes through `DefaultSkillView`;
- the agent view controls exact skill placement;
- custom agent views may omit all managers and skills;
- event expansion preserves position;
- renderers preserve order or reject the layout;
- existing Predict, CodeAct, tool, skill, context, and quickstart behavior remains covered.

## Status

Implemented. The compatibility defaults preserve current state APIs while agent and skill views own assembly. Renderers preserve order or reject unsupported layouts.
