# Context Management in NOOA

In NOOA, context views decide what a model sees on each turn. A strategy's default view selects and organizes instructions, agent state, conversation history, and information from skills. A method or call may replace that view to change what is included and how it is presented, using NOOA's built-in context API or your own data structures.

## Contract

```python
class Block:
    """A named unit of materialized context.

    content is an evaluated value, not a deferred expression.
    """
    key: str
    content: object
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

`Block.content` is materialized; `metadata` carries optional rendering and budget hints. Events remain typed. `CacheBoundary` is a structural marker: it has no content or token cost and is not evictable. In `CodeActView` and `PredictView`, `metadata.static` controls prefix placement independently of evaluation timing.

The runtime assembles the selected view for every model request and collects it into a `tuple[ContextItem, ...]`. Rendering preserves that item order except when expanding an atomic tool-call replay group. No additional assembled-context type is needed.

`CurrentCall` is a per-request snapshot of an invocation. Its invocation identity and method inputs stay stable; mutable manager and scoped state are captured again for each request. Views may read the resolved strategy, `model`, `context_window`, and `context_budget`. Internal formatting and token-counting data support the helpers. It contains no LLM client or credentials.

## Resolution

Highest precedence wins and replaces the lower view:

```text
call -> method -> strategy default
```

```python
class MyAgent(Agent):
    @strategy(CodeActStrategy(), context_view=ResearchView())
    async def run(self, question: str) -> str: ...

agent = MyAgent()
await agent.run("What did we find?", context_view=CallView())
```

`@strategy(..., context_view=view)` is the method override. Views may call helpers or other views through ordinary Python; there is no implicit registration or merge protocol.

Resolve the strategy before its view. Agents and skills expose state and documentation, not view configuration. The selected view owns complete context assembly, including strategy instructions and skill contributions.

## Sources and helpers

`ContextView` is a projection interface, not a storage API. Agents retain `self.context` and `context_manager`; default views may use them as one source. Custom views may use them, another state system, or neither.

A native view uses normal Python and yields materialized items directly:

```python
async def assemble(self, agent, call):
    state = await agent.compute_state()
    yield Block(key="state", content=state)
```

Small optional helpers each do one job:

```python
context_text(value, *, call) -> str
async collect_context_items(view, owner, call) -> tuple[ContextItem, ...]
apply_context_budget(items, *, call, evictable) -> tuple[ContextItem, ...]
async evaluate_context_expression(expression, *, owner, call) -> object
is_model_visible(event) -> bool
```

`context_text` formats one value; `collect_context_items` validates and retains yielded order; expression evaluation binds `self` to `owner`; `is_model_visible` identifies nonempty model-facing events without selecting history. Budgeting follows the caller's eviction order and counts blocks only, so it is not a full rendered-request limit. A source helper may retrieve, evaluate, or format one source. It does not select other sources or decide global precedence, placement, or order.

Iterative views must explicitly include the task, model outputs, and execution feedback they need. Strategies produce these as typed events; rendering injects none. `self.events.query()` searches stored history, including archived events, so it is an inspection API rather than prompt selection. `call.invocation_id` stays stable while a strategy may change `call.id`. Tool schemas and execution policy remain strategy-owned.

Views select active history directly. For example, a current-invocation view uses:

```python
for event in agent.event_manager.values():
    if call.invocation_id in event.related_call_ids and is_model_visible(event):
        yield event
```

## Defaults

`CodeActView`, `PredictView`, and `DecisionContextView` live in ordinary Python modules and contain their strategies' default context policies, including instructions and block ordering. They may share source helpers. In abbreviated form:

```python
class CodeActView(ContextView[Agent]):
    async def assemble(self, agent, call):
        disabled = agent.context_manager.disabled()
        declared = frozenset(key for key, _ in agent.context_manager.declarations())
        blocks = await self._materialize_agent_blocks(
            agent, call, disabled=disabled, declaration_keys=declared
        )
        skill_blocks = await collect_skill_blocks(
            agent, call, disabled=disabled, declaration_keys=declared
        )
        blocks = replace_blocks_by_key(blocks, skill_blocks)
        blocks = await self._apply_override_layers(agent, call, blocks, disabled=disabled)
        blocks = order_blocks(blocks, self.block_order)
        prefix, trailing = partition_blocks(blocks)

        history = visible_events(agent)
        items = [*prefix, *history]
        if items:
            items.append(CacheBoundary())
        items.extend(trailing)
        evictable = [*reversed(user_blocks(trailing)), *reversed(framework_blocks(trailing))]
        for item in apply_context_budget(items, call=call, evictable=evictable):
            yield item
```

The private phase methods contain source-specific detail. `assemble()` keeps global order, event placement, cache placement, and budgeting visible. `apply_block_overrides` preserves replacement, deletion, inherited placement, and explicit prefix hints. `visible_events` selects active model-facing history and projects recorded tool calls without results as outcome-unknown, without changing stored events or reconstructing absent records.

The three built-in helpers derive content directly from the agent. Protected-key registration selects which defaults apply; disable state and stored overrides are checked before evaluation. The manager stores those controls, user declarations, and the last materialized values for compatible named reads, but does not create defaults.

The decision default evaluates explicit decorator/scoped context and includes no history by default. Custom views may select other agent state or active history. Here, `evaluate_value` resolves one literal or expression:

```python
class DecisionContextView(ContextView[Agent]):
    async def assemble(self, agent, call):
        selected = {**(call.decorator_context or {}), **(call.scoped_context or {})}
        disabled = agent.context_manager.disabled()
        for key, value in selected.items():
            if key not in disabled and value is not None:
                yield Block(key=key, content=await evaluate_value(value, agent, call))
```

`collect_skill_blocks` iterates active skills and evaluates their declared `context_block`, with `self` bound to the agent, without registering it in `self.context`. It skips disabled, explicitly declared, and protected keys before evaluation. The view chooses whether to include these blocks and where to place them. Other views may use skill state or documentation directly.

Registry skills participate when active; directly attached public skills are active by default. Hidden and inactive skills contribute nothing in the default views.

## Formatting independently of views

Use `RenderConfig` to select presentation without implementing another view:

```python
from nooa.context_blocks import MarkdownBlockFormatter, RenderConfig

agent = MyAgent(
    render_config=RenderConfig(block_formatter=MarkdownBlockFormatter()),
)
```

`XMLBlockFormatter` is the XML alternative. The legacy
`nooa.context_blocks.renderers.CachedBlockFormatter` import remains a compatibility
subclass of `XMLBlockFormatter`, with identical serialization and guidance.
Neither XML nor Markdown enables caching: both preserve the boundary chosen by
the view. `RenderConfig` chooses block/event serialization and client message
shape; it does not choose context sources, history membership, order, or eviction.

`Agent`'s base instructions are format-neutral. The built-in manager-backed
context views explicitly include the selected formatter's `format_description()`
in the built-in Agent prompt. User class docstrings, `_resolve_system_prompt()`
implementations, and manager/view/decorator/scoped prompt overrides retain ownership of their
instructions. Disabling `system_prompt` also disables default format guidance.
Guidance is evaluated only after override resolution, when the built-in prompt
survives; replacing or removing that prompt does not call `format_description()`.
Custom views receive no implicit guidance; they may explicitly use
`agent.render_config.block_formatter.format_description()` when desired.
Rendering itself never injects instructions.

`context_text(value, call=call)` is separate Python-value serialization. It
applies the call's structural bounds to **non-string** values. Existing strings
pass through unchanged, even when longer than `max_string`; budget or shorten
such content explicitly in the view.

For plain event-field serialization, select
`RenderConfig(block_formatter=PlainBlockFormatter())`, importing
`PlainBlockFormatter` from `nooa.plain_formatter`. This retains XML wrappers for
materialized blocks and optional event role/tag metadata. It is not a tag-free
format; canonical assistant responses retain their original text and replay data.

Block formatters receive one ordered resolved segment at a time, with non-tool
event text already serialized. They may coalesce adjacent materialized blocks of
the same role, without partitioning all system blocks ahead of history. The
renderer handles boundary markers between segments; it does not truncate inputs.

### Atomic tool replay exception

A canonical `LLMResponse` tool-call batch and its linked `ToolCallEvent` results
form one replay group. Stock formatters emit the response at its original
position, followed by all results in tool-call order; linked execution sidecars
emit nothing at their stored positions. Thus a result stored later than an
unrelated block can move into its response's group. A cache boundary cannot
split a response from its linked results.

At the formatter level, incomplete canonical groups are safely omitted as a
group, including orphan linked executions. The renderer rejects a supplied
`ToolCallEvent` with `result=None` before formatting. This does not authorize
arbitrary reordering or fabrication of missing results. Default-view recovery of an **existing** execution record
with an unavailable result is a separate, explicit view policy; absent/archived
records are not reconstructed. Wrapper-style customizations can override
`wrap_system_block()` and `wrap_message_block()` on `BlockFormatter` while
inheriting `format()` to preserve these rules. The first hook formats materialized
blocks of any role; the second formats ordinary events. Canonical assistant
replay bypasses the event wrapper.

`RenderedMessage` is the neutral intermediate representation, including a
private replay carrier. `OpenAIProviderFormatter.format()` returns client input
containing ordinary dictionaries, original `LLMResponse` objects, and
`CacheBoundary` markers. It is not final provider wire JSON. UnifiedLLM owns
provider-specific encoding, native reasoning replay, exact tool argument data,
and cache annotation mapping. Do not flatten a retained response into a
mapping merely to satisfy an annotation or change XML/Markdown wrapping.

## Model-specific content

Model-specific prompt content remains view policy, not formatter policy. A custom view may collect items, run any application-defined adapter, apply its budget, then yield the final sequence. No framework-level prompt-adapter stage is required.

## Ownership

```text
resolve view
    -> view assembles ContextItem items
    -> tuple[ContextItem, ...]
    -> model request pipeline
```

- The selected view owns content, membership, materialization, filtering, global order, adaptation, and budget policy. Strategies retain method inputs, tool/output schemas, decision questions, execution, and result handling.
- Views may compress their output. State summarizers independently compact stored events. When triggered, `TokenBudgetSummarizer` pauses the agent and applies completed summaries before the next turn. Parent-request cache reuse is an optimization, not a requirement for custom views.
- Events expose `call_id` (origin) and `related_call_ids` (represented calls; summaries retain their sources' union). Views filter with `id in event.related_call_ids`. `events.query(call_id=id)` matches origin; `events.query(related_call_ids={a, b})` matches either represented call. Both filters combine with AND; relevance is not call isolation.
- Helpers and nested views provide explicit contributions; the selected view decides their inclusion and placement.
- Source-specific helpers translate existing state APIs; they do not choose global placement.
- The renderer expands items in place and emits no boundary text. A canonical assistant tool-call turn and its linked results form one atomic replay group; a boundary cannot split it.
- Provider formatting preserves a cache marker when its output can carry one and otherwise removes it. UnifiedLLM maps explicit boundaries to provider annotations or removes them when unsupported. Without a boundary, NOOA emits no explicit `cache_control` annotation.
- The selected view owns cache placement. The CodeAct example adds a boundary after visible history and before trailing context; custom views receive none implicitly.
- Provider-managed automatic caching and transport-routing hints such as `prompt_cache_key` are independent of explicit cache breakpoints.
- Bounded serialization is formatting; recovery for missing data or other invented content belongs to event production or view policy.
- Downstream stages preserve view order, except for declared atomic replay groups, or raise `UnsupportedContextLayout`; they do not resolve, evict, repair, or invent content.
- A view omission is prompt policy, not an access-control boundary; tools and generated code may expose data available through other APIs.

## Migration

Keep `agent.context`, context managers, event creation, strategy/scoped overrides, skill activation, and `Skill.context_block` as public state APIs.

These are the default application's context APIs, not requirements of `ContextView`. A custom view may use an independent state API. Native iterative strategies still use NOOA events unless replaced together with the strategy.

Legacy implicit policy is removed: `cache_control_injection_points` raises `ValueError`; `CachedBlockFormatter` no longer chooses placement; and `render_context(context_limit=...)` reports the limit but leaves eviction to the view.

Direct UnifiedLLM callers place boundaries explicitly; without one, NOOA adds no explicit cache marker:

```python
from nooa import CacheBoundary

response = await client.acall([
    {"role": "system", "content": "Stable instructions"},
    CacheBoundary(),
    {"role": "user", "content": "Current task"},
])
```

`EventQuery`, agent `event_query=`, `ScopedContext(events=...)`, and manager query overrides are removed. Event membership is view code; event inspection and context-block overrides remain available.

Default views and their helpers use only public agent, call, manager, strategy, and event interfaces. `ActorRuntime` supplies call facts and budget support, resolves and collects the view, renders it, and handles transport and observability; it does not assemble context content.
