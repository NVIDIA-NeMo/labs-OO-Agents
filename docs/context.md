# Context Management in NOOA

## Goal

Use one pattern for context produced by agents, skills, and future components. One agent view creates the complete ordered model context and may explicitly compose other views. The default view is a readable reference implementation. Rendering adds no context sources or placement policy.

## Contract

```python
class Block:
    key: str
    content: str
    role: Role = Role.SYSTEM
    metadata: BlockMetadata | None = None


class CacheBoundary:
    """Request caching of the rendered prefix ending here, where supported."""


ContextItem = Block | EventBase | CacheBoundary


class ContextView[Owner](Protocol):
    def assemble(
        self,
        owner: Owner,
        call: CurrentCall,
    ) -> AsyncIterator[ContextItem]: ...
```

`Block.content` is materialized; `metadata` carries optional rendering and budget hints. Events remain typed. `CacheBoundary` is a structural marker: it has no content or token cost and is not evictable. In the default view, `metadata.static` controls prefix placement independently of evaluation timing.

The runtime assembles the selected view for every LLM request and collects it into a `tuple[ContextItem, ...]`. Rendering preserves that item order except when expanding an atomic tool-call replay group. No additional assembled-context type is needed.

`CurrentCall` is the immutable per-request view of an invocation. Its invocation identity and method inputs stay stable; mutable manager and scoped selections are captured again for each request. Views may read the resolved strategy, event query, `model`, `context_window`, and `context_budget`. Internal formatting and token-counting data support the helpers. It contains no LLM client or credentials.

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

`Agent` and `Skill` expose `__context_view__()` as the owner-resolution hook. Registration storage is internal to those classes; generic resolution does not inspect it.

## Sources and helpers

`ContextView` is a projection interface, not a storage API. Agents retain `self.context` and `context_manager`; the default view uses them as one source. Custom views may use them, another state system, or neither.

A native view uses normal Python and yields materialized items directly:

```python
async def assemble(self, agent, call):
    state = await agent.compute_state()
    yield Block(key="state", content=context_text(state, call=call))
```

Small optional helpers each do one job:

```python
context_text(value, *, call) -> str
async collect_context_items(view, owner, call) -> tuple[ContextItem, ...]
apply_context_budget(items, *, call, evictable) -> tuple[ContextItem, ...]
async evaluate_context_expression(expression, *, owner, call) -> object
select_context_events(events, *, call) -> tuple[EventBase, ...]
```

`context_text` formats one value; `collect_context_items` validates and retains yielded order; expression evaluation binds `self` to `owner`; event selection reads active events, applies the resolved query with stable `call.invocation_id`, and excludes non-model events; budgeting follows the caller's eviction order and counts blocks only, so it is not a full rendered-request limit. A source helper may retrieve, evaluate, or format one source. It does not select other sources or decide global precedence, placement, or order.

Iterative views must explicitly include the task, model outputs, and execution feedback they need. Strategies produce these as typed events; rendering injects none. `self.events.query()` searches stored history, including archived events, so it is an inspection API rather than prompt selection. `call.invocation_id` stays stable while a strategy may change `call.id`. Tool schemas and execution policy remain strategy-owned.

## Defaults

`DefaultAgentView` lives in its own ordinary module and contains the complete default policy. In abbreviated form:

```python
class DefaultAgentView(ContextView[Agent]):
    async def assemble(self, agent, call):
        disabled = agent.context_manager.disabled()
        declared = frozenset(key for key, _ in agent.context_manager.declarations())
        blocks = await self._materialize_agent_blocks(
            agent, call, disabled=disabled, declaration_keys=declared
        )
        blocks, skill_items = await self._compose_skill_views(
            agent, call, blocks, disabled=disabled, declaration_keys=declared
        )
        blocks = await self._apply_override_layers(agent, call, blocks, disabled=disabled)
        blocks = order_blocks(blocks, call.strategy.get_block_order())
        prefix, trailing = partition_blocks(blocks)

        items = [*prefix, *skill_items, *visible_events(agent, call)]
        boundaries = sum(isinstance(item, CacheBoundary) for item in items)
        if boundaries > 1:
            raise ValueError("multiple cache boundaries")
        if items and boundaries == 0:
            items.append(CacheBoundary())
        items.extend(trailing)
        evictable = [*reversed(user_blocks(trailing)), *reversed(framework_blocks(trailing))]
        for item in apply_context_budget(items, call=call, evictable=evictable):
            yield item
```

The three private phase methods contain source-specific detail. `assemble()` keeps global order, event placement, cache placement, and budgeting visible. `apply_block_overrides` preserves replacement, deletion, inherited placement, and explicit prefix hints.

The three built-in helpers derive content directly from the agent. Protected-key registration selects which defaults apply; disable state and stored overrides are checked before evaluation. The manager stores those controls, user declarations, and the last materialized values for compatible named reads, but does not create defaults.

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
            yield Block(
                key=key,
                content=context_text(value, call=call),
                role=Role.USER,
                metadata=BlockMetadata(expr=expression, source_dynamic=True),
            )
```

A skill view produces only that skill's ordered contribution; the agent view decides whether and where to include it. Native views normally read `skill` state directly. The fallback preserves the legacy `context_block` expression and agent binding as prompt projection, without registering it in `self.context`; explicit manager declarations win. Custom contributions are inserted intact without manager policy.

Registry skills participate when active; directly attached public skills are active by default. Hidden and inactive skills contribute nothing. `DefaultSkillView` emits only explicitly declared context; it never dumps skill documentation. The default agent view places custom skill contributions before visible events; a custom agent view may place volatile skill state after the cache boundary.

## Model-specific content

Model-specific prompt content remains view policy, not formatter policy. A custom view may collect items, run any application-defined adapter, apply its budget, then yield the final sequence. No framework-level prompt-adapter stage is required.

## Ownership

```text
resolve view
    -> view assembles ContextItem items
    -> tuple[ContextItem, ...]
    -> render_context
    -> ProviderFormatter
    -> UnifiedLLM
```

- The selected agent view owns content, membership, materialization, filtering, global order, adaptation, and budget policy.
- A nested view owns the content and local order of its contribution.
- Source-specific helpers translate existing state APIs; they do not choose global placement.
- The renderer expands items in place and emits no boundary text. A canonical assistant tool-call turn and its linked results form one atomic replay group; a boundary cannot split it.
- Provider formatting preserves a cache marker when its output can carry one and otherwise removes it. UnifiedLLM maps explicit boundaries to provider annotations or removes them when unsupported. Without a boundary, NOOA emits no explicit `cache_control` annotation.
- The selected agent view owns cache placement. The default preserves one boundary contributed by a skill, rejects multiple, or adds one after visible history and before trailing context; custom agent views receive none implicitly.
- Provider-managed automatic caching and transport-routing hints such as `prompt_cache_key` are independent of explicit cache breakpoints.
- Bounded serialization is formatting; recovery for missing data or other invented content belongs to event production or view policy.
- Downstream stages preserve view order, except for declared atomic replay groups, or raise `UnsupportedContextLayout`; they do not resolve, evict, repair, or invent content.
- A view omission is prompt policy, not an access-control boundary; tools and generated code may expose data available through other APIs.

## Migration

Keep `agent.context`, context managers, event creation, strategy/scoped overrides, skill activation, and `Skill.context_block` as public state APIs.

These are the default application's context APIs, not requirements of `ContextView`. A custom view may use an independent state API. Native iterative strategies still use NOOA events unless replaced together with the strategy.

Legacy implicit policy is removed: `cache_control_injection_points` raises `ValueError`; `CachedBlockFormatter` no longer chooses placement; and `render_context(context_limit=...)` reports the limit but leaves eviction to the view.

`DefaultAgentView` and its helpers use only public agent, call, manager, strategy, and event interfaces. `ActorRuntime` supplies call facts and budget support, resolves and collects the view, renders it, and handles transport and observability; it does not assemble context content.
