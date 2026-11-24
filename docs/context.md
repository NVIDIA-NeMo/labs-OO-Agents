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

`CurrentCall` is a per-request snapshot of an invocation. Its invocation identity and method inputs stay stable; mutable manager and scoped state are captured again for each request. Views may read the resolved strategy, `model`, `context_window`, and `context_budget`; the runtime resolves these once per request from the client and limits the request is sent with, so the window and budget always agree. Internal formatting and token-counting data support the helpers. It contains no LLM client or credentials.

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
replayable(events) -> tuple[EventBase, ...]
```

`context_text` formats one value; `collect_context_items` validates and retains yielded order; expression evaluation binds `self` to `owner`; `is_model_visible` identifies nonempty model-facing events without selecting history; `replayable` makes a selection sendable by giving tool calls without a recorded result an outcome-unknown result and omitting tool-call turns or results the selection separated, on copies, without changing stored events. Budgeting follows the caller's eviction order and counts blocks only, so it is not a full rendered-request limit. A source helper may retrieve, evaluate, or format one source. It does not select other sources or decide global precedence, placement, or order.

Iterative views must explicitly include the task, model outputs, and execution feedback they need. Strategies produce these as typed events; rendering injects none. `self.events.query()` searches stored history, including archived events, so it is an inspection API rather than prompt selection. Tool schemas and execution policy remain strategy-owned.

Views select active history directly. The default views select it in `history(owner, call)` and pass the result through `replayable()`, so a subclass can change only the history and keep the instructions, blocks, cache boundary, and budget:

```python
class CurrentCallView(CodeActView):
    def history(self, agent, call):
        return (event for event in visible_events(agent) if call.id in event.related_call_ids)
```

A view written from scratch is the whole request; one that yields only events sends no system prompt or context blocks. It selects events directly:

```python
for event in replayable(
    event
    for event in agent.event_manager.values()
    if call.id in event.related_call_ids and is_model_visible(event)
):
    yield event
```

## Defaults

`CodeActView`, `PredictView`, and `DecisionContextView` live in ordinary Python modules and contain their strategies' default context policies, including instructions and block ordering. They may share source helpers. In abbreviated form:

```python
class CodeActView(ContextView[Agent]):
    def __init__(self, *, state=True): ...

    async def assemble(self, agent, call):
        # system prompt, self, events_api, state (unless state=False), strategy instructions
        own = await self.view_blocks(agent, call)
        # self.context blocks, skill blocks, then decorator and scoped context
        added = await self.context_blocks(agent, call)
        blocks = order_blocks((*own, *added), self.block_order)
        prefix, trailing = partition_blocks(blocks)

        history = replayable(self.history(agent, call))  # visible_events(agent)
        items = [*prefix, *history]
        if items:
            items.append(CacheBoundary())
        items.extend(trailing)
        evictable = [*reversed(user_blocks(trailing)), *reversed(framework_blocks(trailing))]
        for item in apply_context_budget(items, call=call, evictable=evictable):
            yield item
```

`view_blocks()` and `context_blocks()` contain source-specific detail. `assemble()` keeps global order, event placement, cache placement, and budgeting visible. `apply_block_overrides` applies decorator and scoped context to the added blocks: replacement, deletion, inherited placement, and explicit prefix hints. `history()` returns `visible_events` (active model-facing history) and is the override point for event selection; `replayable` then pairs its tool calls and results without reconstructing absent records.

The view owns the blocks it builds; their helpers derive content directly from the agent or strategy, and only for an `Agent` owner (standalone functions get the strategy blocks). The view's constructor documents what can be changed, such as `CodeActView(state=False)`; anything else needs a custom view. `PredictView` leaves out the events API guidance, since a Predict call has no REPL to run `doc(self.events)`. `self.context` only adds blocks: a stored block named `system_prompt` is an extra block, and `None` suppresses user, skill, and decorator/scoped blocks by name but never a view's own. So removing a built-in block is now a view choice: `context={"state": None}` on the class becomes `context_view=CodeActView(state=False)` on the method. The manager stores user declarations and that suppression set, but not evaluated values: reading an expression block returns its `Context(expr=...)` declaration, so call the expression's code for its current value.

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

`collect_skill_blocks` iterates active skills and materializes their declared `context_block = (key, Context(...))`, evaluating expressions with `self` bound to the skill, without registering it in `self.context`. `Context` validates the expression when the skill class is defined; `prefix=True` marks the block cacheable as for agent blocks. It skips disabled and explicitly declared keys before evaluation. The view chooses whether to include these blocks and where to place them. Other views may use skill state or documentation directly.

Registry skills participate when active; directly attached public skills are active by default. Hidden and inactive skills contribute nothing in the default views.

## Formatting independently of views

Use `RenderConfig` to select presentation without implementing another view:

```python
from nooa.context_blocks import MarkdownBlockFormatter, RenderConfig

agent = MyAgent(
    render_config=RenderConfig(block_formatter=MarkdownBlockFormatter()),
)
```

`XMLBlockFormatter` is the XML alternative. Neither XML nor Markdown enables
caching: both preserve the boundary chosen by the view. `RenderConfig` chooses
serialization and client message shape; it does not choose context sources,
history membership, order, or eviction. `CachedBlockFormatter` remains as a
compatibility wrapper with XML-equivalent serialization; cache placement stays
view-owned, so use explicit view cache boundaries.

`Agent`'s base instructions are format-neutral. Built-in context views include
the selected formatter's `format_description()` in the built-in Agent prompt.
Custom class docstrings own their instructions. A view that replaces or removes
its own `system_prompt` does not evaluate default format guidance. User, skill,
decorator, and scoped context blocks are separate additions; a same-name block
or suppression does not replace or disable the view-owned prompt or guidance.
Custom views may explicitly select formatter guidance. Rendering itself never
injects instructions.

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

At the formatter level, incomplete canonical groups and orphan linked executions
raise `UnsupportedContextLayout`. The renderer also rejects a supplied
`ToolCallEvent` with `result=None`. Select history through `replayable()` before
rendering: it repairs missing outcomes and omits separated groups on copies,
without mutating stored events. Cache boundaries must still keep each batch
intact. Wrapper-style customizations can override
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
- `call.id` identifies one generation-method call, with or without `@no_trace`. Events recorded during it, including by deterministic helpers it calls, carry it as `call_id`; events recorded outside any generation call of the agent have none. Tracing keeps its own method call tree.
- Events expose `call_id` (origin) and `related_call_ids` (represented calls; summaries retain their sources' union). Views filter with `id in event.related_call_ids`. `events.query(call_id=id)` matches origin; `events.query(related_call_ids={a, b})` matches either represented call. Both filters combine with AND; relevance is not call isolation.
- Helpers and nested views provide explicit contributions; the selected view decides their inclusion and placement.
- Source-specific helpers translate existing state APIs; they do not choose global placement.
- The renderer expands items in place and emits no boundary text. A canonical assistant tool-call turn and its linked results form one atomic replay group; a boundary cannot split it.
- Provider formatting preserves a cache marker when its output can carry one and otherwise removes it. UnifiedLLM maps explicit boundaries to provider annotations or removes them when unsupported. Without a boundary, NOOA emits no explicit `cache_control` annotation.
- The selected view owns cache placement and places at most one `CacheBoundary`; rendering a second raises `UnsupportedContextLayout`. The CodeAct example adds a boundary after visible history and before trailing context; custom views receive none implicitly.
- Provider-managed automatic caching and transport-routing hints such as `prompt_cache_key` are independent of explicit cache breakpoints.
- Bounded serialization is formatting; recovery for missing data or other invented content belongs to event production or view policy.
- Downstream stages preserve view order, except for declared atomic replay groups, or raise `UnsupportedContextLayout`; they do not resolve, evict, repair, or invent content.
- Assembly is deterministic local code. An exception that escapes the view's `assemble()` or rendering surfaces as `ContextAssemblyError` (`UnsupportedContextLayout` is a subclass); strategies propagate it without retrying or recording model-visible feedback. How a view handles a failing source is its own policy: `CodeActView` and `PredictView` render the error (`ExcType: message`) as that block's content and log a warning, while `DecisionContextView` raises, so a decision is never made on context that failed to resolve.
- A view omission is prompt policy, not an access-control boundary; tools and generated code may expose data available through other APIs.

## Migration

Keep `agent.context`, context managers, event creation, strategy/scoped overrides, skill activation, and `Skill.context_block` as public state APIs.

These are the default application's context APIs, not requirements of `ContextView`. A custom view may use an independent state API. Native iterative strategies still use NOOA events unless replaced together with the strategy.

Legacy implicit policy is removed: `cache_control_injection_points` raises `ValueError`; `CachedBlockFormatter` remains only as an XML-serialization compatibility wrapper and does not choose cache placement; and `render_context()` takes no context limit: it counts blocks the view marked evicted and never evicts.

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
