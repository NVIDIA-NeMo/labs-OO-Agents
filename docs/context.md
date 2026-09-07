# Context Management in NOOA

## Goal

Use one pattern for context produced by agents, skills, and future components. One agent view creates the complete ordered model context. It may explicitly compose other views. Rendering adds no content policy.

## Contract

```python
class Block:
    key: str
    content: str
    role: Role = Role.SYSTEM


ContextItem = Block | EventBase


class ContextView[Owner](Protocol):
    def assemble(
        self,
        owner: Owner,
        call: CurrentCall,
    ) -> AsyncIterator[ContextItem]: ...
```

`Block.content` is materialized. Events remain typed. The runtime collects the selected view into an immutable `tuple[Block | EventBase, ...]` before rendering. No additional assembled-context type is needed.

`CurrentCall` is the immutable invocation snapshot. In addition to method inputs and the resolved strategy and event query, context views may read the resolved `model`, `provider`, `context_window`, and `context_budget`. Internal formatting and token-counting data support the helpers. It contains no LLM client or credentials.

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

## Sources and helpers

The core contract does not require `ContextManager`, `EventManager`, string expressions, or a declarative block specification. A native view uses normal Python and yields materialized items directly:

```python
async def assemble(self, agent, call):
    state = await agent.compute_state()
    yield Block(key="state", content=context_text(state, call=call))
```

Small optional helpers provide reusable mechanisms:

```python
context_text(value, *, call) -> str
collect_context(view, owner, call) -> tuple[ContextItem, ...]
apply_context_budget(items, *, call, evictable) -> tuple[ContextItem, ...]
evaluate_context_expression(expression, *, owner, call) -> object
```

Expression evaluation supports declarative APIs; it is not required by `ContextView`.

The existing managers are concrete context sources. Source-specific helpers translate their state into the neutral contract:

```python
materialize_managed_context(agent, call) -> (prefix, trailing, evictable)
events_from_manager(agent, call) -> tuple[EventBase, ...]
```

These helpers preserve current manager semantics: protected and disabled blocks, dynamic caching, strategy and scoped overrides, event queries, and prefix/trailing placement. They are not core context abstractions.

## Defaults

`DefaultAgentView` owns the complete default policy. The runtime does not own its ordering or membership:

```python
class DefaultAgentView(ContextView[Agent]):
    async def assemble(self, agent, call):
        prefix, trailing, evictable = await materialize_managed_context(agent, call)
        items: list[ContextItem] = [*prefix]

        for skill in agent.active_skills():
            view = resolve_context_view(skill, default=DefaultSkillView())
            items.extend(await collect_context(view, skill, call))

        items.extend(events_from_manager(agent, call))
        items.extend(trailing)
        items = apply_context_budget(items, call=call, evictable=evictable)

        for item in items:
            yield item


class DefaultSkillView(ContextView[Skill]):
    async def assemble(self, skill, call):
        if skill.context_block is not None:
            key, expression = skill.context_block
            value = await evaluate_context_expression(
                expression,
                owner=call.agent,  # preserves existing agent-scoped semantics
                call=call,
            )
            yield Block(key=key, content=context_text(value, call=call))
```

Registry skills participate when active; directly attached public skills are active by default. Hidden and inactive skills contribute nothing. `DefaultSkillView` emits only explicitly declared context; it never dumps skill documentation.

## Model-specific content

Model-specific prompt content remains view policy, not formatter policy. A custom view may collect items, run any application-defined adapter, apply its budget, then yield the final sequence. No framework-level prompt-adapter stage is required.

## Ownership

```text
resolve view
    -> view assembles Block | EventBase items
    -> tuple[Block | EventBase, ...]
    -> ContextRenderer
    -> ProviderFormatter
```

- The selected agent view owns content, membership, materialization, filtering, global order, adaptation, and budget policy.
- A nested view owns the content and local order of its contribution.
- Source-specific helpers translate existing state APIs; they do not choose global placement.
- The renderer expands each item in place and serializes it.
- The provider formatter only adapts message shape and cache annotations.
- Downstream stages preserve semantics or raise `UnsupportedContextLayout`; they never reorder, omit, resolve, evict, repair, or add context.

## Migration

Keep `agent.context`, context managers, event creation, strategy/scoped overrides, skill activation, and `Skill.context_block` as public state APIs. Reimplement their internals as sources consumed by the default views; do not make them requirements for custom views.

Move default assembly policy out of `ActorRuntime` and into `DefaultAgentView`. `ActorRuntime` resolves `CurrentCall` and the selected view, collects its output into a tuple, renders it, and calls the LLM.

Do not introduce `BlockSpec`, a separate assembled-context object, or permanent runtime-owned compatibility assembly. Temporary bridges are acceptable only during migration.

## Required tests

- immutable, stable assembly order;
- agent and skill precedence;
- resolved model and budget data on `CurrentCall`, including method and call overrides;
- inactive and hidden skills contribute nothing;
- existing manager, scoped override, event, and dynamic-cache behavior is preserved;
- `Skill.context_block` preserves agent-scoped expression behavior;
- the agent view controls exact skill and event placement;
- custom views may omit all managers and skills;
- custom views can adapt before applying their own budget;
- `DefaultAgentView` uses public helpers, not private runtime assembly;
- event expansion preserves position;
- renderers preserve order or reject the layout;
- existing Predict, CodeAct, tool, skill, context, capability, and quickstart behavior remains covered.

## Status

Implemented on this branch as a proof of concept. `DefaultAgentView` owns assembly policy; `ActorRuntime` only resolves the call and view, collects the tuple, renders it, and invokes the model.
