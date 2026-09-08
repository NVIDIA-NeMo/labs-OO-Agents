# Context Management in NOOA

## Goal

Use one pattern for context produced by agents, skills, and future components. One agent view creates the complete ordered model context and may explicitly compose other views. The default view is a readable reference implementation. Rendering adds no content policy.

## Contract

```python
class Block:
    key: str
    content: str
    role: Role = Role.SYSTEM
    metadata: BlockMetadata | None = None


ContextItem = Block | EventBase


class ContextView[Owner](Protocol):
    def assemble(
        self,
        owner: Owner,
        call: CurrentCall,
    ) -> AsyncIterator[ContextItem]: ...
```

`Block.content` is materialized; `metadata` carries optional rendering and budget hints. Events remain typed. The runtime collects the selected view into an immutable `tuple[Block | EventBase, ...]` before rendering. No additional assembled-context type is needed.

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

`ContextView` is a projection interface, not a storage API. Agents retain `self.context` and `context_manager`; the default view uses them as one source. Custom views may use them, another state system, or neither.

A native view uses normal Python and yields materialized items directly:

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

A source helper may retrieve, evaluate, or format one source. It does not select other sources or decide global precedence, placement, or order.

## Defaults

`DefaultAgentView` lives in its own ordinary module and contains the complete default policy:

```python
class DefaultAgentView(ContextView[Agent]):
    async def assemble(self, agent, call):
        blocks = [
            await system_prompt_block(agent, call),
            await agent_interface_block(agent, call),
            await agent_state_block(agent, call),
        ]

        # Later sources replace earlier blocks with the same key.
        for source in (
            await stored_context_blocks(
                agent.context_manager, agent, call, exclude={"system_prompt", "self", "state"}
            ),
            await strategy_context_blocks(call.strategy, agent, call),
            await decorator_context_blocks(call),
            await scoped_context_blocks(call),
        ):
            blocks = replace_by_key(blocks, source)

        blocks = remove_disabled(blocks, agent.context_manager.disabled())
        blocks = order_blocks(blocks, call.strategy.get_block_order())
        prefix, trailing = partition_blocks(blocks)

        skills = []
        for skill in agent.active_skills():
            view = resolve_context_view(skill, default=DefaultSkillView())
            skills.extend(await collect_context(view, skill, call))

        items = [*prefix, *skills, *visible_events(agent, call), *trailing]
        evictable = [*reversed(user_blocks(trailing)), *reversed(framework_blocks(trailing))]
        for item in apply_context_budget(items, call=call, evictable=evictable):
            yield item
```

Each `*_blocks` helper materializes only its named source. The generic list helpers contain no source selection or ordering policy. Changing or removing a source is a local edit to `assemble()`.

```python
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

Keep `agent.context`, context managers, event creation, strategy/scoped overrides, skill activation, and `Skill.context_block` as public state APIs.

`DefaultAgentView` and its helpers use only public agent, call, manager, strategy, and event interfaces. `ActorRuntime` only creates `CurrentCall`, resolves and collects the view, renders it, and calls the LLM.

## Required tests

- immutable, stable assembly order;
- agent and skill precedence;
- resolved model and budget data on `CurrentCall`, including method and call overrides;
- inactive and hidden skills contribute nothing;
- existing manager, scoped override, event, dynamic-cache, and snapshot behavior is preserved;
- `Skill.context_block` preserves agent-scoped expression behavior;
- the agent view controls exact skill and event placement;
- custom views may ignore all managers and skills;
- custom views can adapt before applying their own budget;
- `DefaultAgentView` uses public helpers, not private runtime assembly;
- the standalone default module imports no `nooa.runtime.*`;
- an external custom view ignores a sentinel `context_manager` and completes Predict and CodeAct calls;
- event expansion preserves position;
- renderers preserve order or reject the layout;
- existing Predict, CodeAct, tool, skill, context, capability, and quickstart behavior remains covered.
