---
name: nooa-context-and-state
description: Manage what a NOOA agent sees and remembers — context blocks, event history and queries, history summarization, and persistent memory/storage. Use when pinning information into the system prompt, querying past events, bounding context growth in long conversations, or persisting agent state.
compatibility: nooa package
---

# Context, Events, and State

Every agent has two managers, always present, hidden from the LLM by default:

- `agent.context_manager` (`ContextManager`) — named **context blocks** rendered into the system prompt each turn.
- `agent.event_manager` (`EventManager`) — the **event history** (tasks, messages, code executions, LLM calls).

Their agent-facing APIs are `self.context` (`ContextApi`) and `self.events` (`EventsApi`). To let the *LLM* see and manage them, opt in per subclass:

```python
from nooa.agentdoc import spec

class MyAgent(Agent, llm=llm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "context", hidden=False)   # LLM can now use self.context
        spec(self, "events", hidden=False)    # LLM can now query self.events
```

Do NOT re-declare `context`/`events` as class annotations to unhide them — use `spec()`.

## Context blocks

Blocks appear as labelled SYSTEM sections, visible across all method calls on
the instance (per-instance only — subagents don't inherit them). They are not
the only information retained between turns of an active generation call: the
current task description stays present, and event history carries messages,
generated code, tool results, and printed output. Event history can later be
filtered, summarized, or evicted as it grows; a context block is rendered again
each turn and is not removed by history summarization. Context is eager prompt
input, not general-purpose object storage. Use it only for bounded information
that must remain continuously visible.

```python
from nooa import Context

# Fixed content in the stable, provider-cache-friendly prefix
self.context["plan"] = Context(plan.format(), prefix=True)

# Live expression re-evaluated every LLM turn in the volatile suffix
self.context["progress"] = Context(expr="self.format_project_state()")

# Fixed literal in the volatile suffix (the bare-value shorthand)
self.context["latest_decision"] = "Use the canary deployment."

# Remove
del self.context["plan"]           # or self.context.pop("plan")

# Class-level default blocks
class MyAgent(
    Agent,
    llm=llm,
    context={"focus": Context(expr="self.topic")},
): ...
```

Content and placement are independent: `value` is fixed while `expr` is
re-evaluated; `prefix=True` requests stable-prefix placement while the default
uses the volatile suffix. Use docstrings for per-call task instructions and
context blocks for bounded cross-call facts such as decisions, plans, and live
status. Method arguments remain the right place for per-call input.

Prefix placement directly affects provider prompt/KV caching. Put only truly
stable content in the prefix: changing a prefix block invalidates cache reuse
for that block and everything after it. Keep live or frequently changing blocks
in the volatile suffix so the stable prefix remains reusable.

The default context view marks where that volatile suffix begins. UnifiedLLM
maps this boundary to provider-specific cache metadata when supported.

Per-method event selection belongs to the context view. To change only the
history, subclass the strategy's default view and override `history()`; the
system prompt, context blocks, cache boundary, and budget stay as they are:

```python
from nooa import strategy
from nooa.default_context_view import CodeActView, visible_events

class CurrentCallView(CodeActView):
    def history(self, agent, call):
        return (event for event in visible_events(agent) if call.id in event.related_call_ids)

@strategy(context_view=CurrentCallView())
async def solve(self, problem: str) -> str:
    """Solves with only this call's events, no prior history."""
    ...
```

Use `PredictView` or `PurePythonView` as the base for those strategies. A view
written from scratch is the whole request: if it yields only events, the model
sees no system prompt, instructions, or context blocks.

The default views pass `history()` through `replayable()`. In a view written
from scratch, pass any event selection through `replayable()` after filtering. It gives tool calls without a
recorded result an outcome-unknown result and omits tool-call turns or results the selection
separated, on copies only; stored history is unchanged. Rendering rejects such gaps with
`UnsupportedContextLayout` instead of repairing them.

## Events

Event history is what fills the LLM's conversation window. Key model-visible event types (names have no "Event" suffix): `Task`, `Message`, `Reasoning`, `Error`, `Feedback`, `LLMResponse`, `PythonOutput`, `Summary`, `Notification`. `LLMResponse` is both the canonical assistant turn and the home of its token/cost metadata; renderers expose only its conversational fields. Runtime-only events (never shown to the LLM) include `BeforeAgentCall`/`AfterAgentCall` and `LLMCallStart`/`LLMCallEnd`.

`LLMResponse.reasoning` is provider-exposed text and remains useful across model switches, where UnifiedLLM replays it as ordinary assistant text. Opaque state lives on ordered `LLMResponse.parts`; UnifiedLLM alone interprets it and replays it through a matching provider/API/model gate.

Pass prior `LLMResponse` objects directly in the next call's message list.
Responses expose read-only public mapping access (`response["content"]`,
`response.get("role")`). To edit a turn, replace that history element with
`{**dict(response), "content": "edited text"}`; the replacement has no opaque
state. `dict(response)` is a public projection, while `model_dump()` is for
durable archives. JSON-only integrations must explicitly project public messages.

An intentional model/API mismatch logs a warning and falls back to the portable text. A malformed current-version part or provider signature raises `ReasoningReplayError`; do not catch it and silently continue, because it signals archive corruption or an unsupported provider contract change.

```python
# Query (AND semantics; chronological; limit keeps most recent)
recent = agent.events.query(limit=20)
errors = agent.events.query(type="Error")
hits   = agent.events.query(query="timeout")            # text search; regex=True for regex

# Views select from active history; query() also searches archived events.
active = agent.event_manager.values()

# Subscribe
agent.event_manager.on("Message", lambda e: print(e.content))

# Archive a range into a one-line summary (LLM can do this too when events is exposed)
agent.events.collapse("3", "17", summary_text="Explored the repo layout")
```

## History summarization

Unbounded histories eventually overflow the model context. Install a summarizer:

```python
from nooa.agents import TokenBudgetSummarizer, MethodSummarizer, context_budget
from nooa.config import TokenBudgetConfig, MethodSummarizerConfig

# Compress oldest events when the token budget is crossed (open-ended conversations)
TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10))

# Or compress each completed method call's events (batch-style agents)
MethodSummarizer.install(agent, config=MethodSummarizerConfig(min_events=3))

# Size the budget from the model's context window
TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=context_budget(llm, percent=0.8)))
```

Summarizers are themselves agents; they inherit the host agent's LLM by default. `agent.context_stats` reports context-window usage.

Token-budget compaction summarizes stored events independently of the context view.
Views may separately compress selected prompt content without replacing stored history.

## Persistent state

```python
from nooa.storage import SQLiteStorageManager

agent = MyAgent(storage=SQLiteStorageManager("agent_state.db"))   # snapshots + resume
```

Events, context blocks, LLM-defined methods, and user attributes are serialized; exclude a field with `Annotated[T, nosnapshot]` (`from nooa.storage import nosnapshot`). Note this is **agent state** persistence (`src/nooa/storage/`) — unrelated to trace storage (`traces.db`, owned by the viewer).

For long-term semantic memory (remember/recall across sessions) there is an opt-in memory subsystem:

```python
from nooa_memory import MemoryConfig, MemoryManager, MemoryToolsMixin

class MyAgent(MemoryToolsMixin, Agent, llm=llm): ...
MemoryManager.install(agent, config=MemoryConfig(enabled=True))
```

See `examples/advanced/memory.py` and `examples/quickstart/12_memory.py`.

## Pitfalls

- Context blocks and events are **per-instance**. Subagents start empty — pass data explicitly (constructor args, shared dataclasses).
- Expression blocks are evaluated every turn — keep them cheap and bounded (a huge `self.render_everything()` bloats every prompt).
- Reading an expression block returns its declaration, not a value:
  `self.context["progress"]` is `Context(expr="self.format_project_state()")`.
  Call `self.format_project_state()` for the current value.
- `self.context["k"] = None` suppresses a user, skill, or decorator/scoped
  block from prompt rendering. Use `del self.context["k"]` / `.pop("k")` to
  remove a user block entirely.
- Blocks the view builds (`system_prompt`, `self`, `state`, strategy
  instructions) are not in `self.context`. Storing one of those names adds a
  separate block; turn the view's blocks off with its options, e.g.
  `@strategy(context_view=CodeActView(state=False))`.
- A bare `self.context["k"] = value` is fixed content in the volatile suffix.
  Use `Context(value, prefix=True)` for stable, cache-friendly content.
- `set_static()`, `set_dynamic()`, and `DynamicContext` are legacy APIs. They
  still work, but the unified `Context(value=...|expr=..., prefix=...)` API is
  the current interface.

## Related skills

- `nooa-agent-authoring` — the core authoring model this builds on.
- `nooa-capturing-traces` — events vs spans: traces are the observability view of the same run.
