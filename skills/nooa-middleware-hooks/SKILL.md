---
name: nooa-middleware-hooks
description: Intercept and observe NOOA execution — middleware via event_manager.intercept() (guardrails, input/output transforms, blocking), event observers via event_manager.on() (react to Task/Error/LLMComplete/turn events), and the InstrumentationHooks protocol for observability backends. Use when adding guardrails, redacting or rewriting prompts, blocking or faking an LLM call or code execution, rate-limiting agent methods, subscribing to lifecycle events, or wiring custom telemetry.
compatibility: nooa package
---

# Middleware and Hooks

Three interception surfaces, one decision rule:

| Surface | Registration | Can it change behavior? | Errors | Scope |
|---|---|---|---|---|
| **Middleware** | `agent.event_manager.intercept(kind, fn)` | YES — transform inputs/outputs, short-circuit, block | propagate (a guardrail raising aborts the call) | per-agent |
| **Observers** | `agent.event_manager.on(event_type, fn)` | no — fire-and-forget after the event is recorded | isolated | per-agent |
| **InstrumentationHooks** | `set_hooks(obj)` (`nooa.runtime.hooks`) | no — observational timing/tracing pairs | swallowed + logged, overhead metered | one global slot per async context |

Rule of thumb: enforcing or transforming → `intercept()`; reacting → `on()`; building an observability backend → you probably want a tracing exporter (`nooa-capturing-traces`), not raw hooks — the tracing system already occupies the hooks slot.

## Middleware (`intercept`)

Each middleware is `async def mw(ctx, nxt) -> ctx` — a typed context object and a `nxt` callable running the rest of the chain. Three kinds (`nooa.runtime.middleware`):

| Kind | Wraps | Context (`ctx`) mutables in | result out |
|---|---|---|---|
| `"agent_call"` | one whole agent-method call (all turns) | `args`, `kwargs` (+ `agent`, `method_name`) | `ctx.result` |
| `"llm_call"` | one LLM round-trip inside `runtime.generate()` | `messages` (the rendered prompt), `params` (tools, output_model, max_tokens, ...) | `ctx.response` (`LLMResponse`) |
| `"execute_python"` | one CodeAct cell in `runtime.execute_code()` | `code`, `params` (timeout, restrictions, ...) | `ctx.result` (`ExecutionResult`) |

```python
from nooa.runtime.middleware import LLMCallContext, LLMCallNext

async def redact_secrets(ctx: LLMCallContext, nxt: LLMCallNext) -> LLMCallContext:
    for m in ctx.messages:                          # transform inputs
        if isinstance(m.get("content"), str):
            m["content"] = m["content"].replace(SECRET, "[redacted]")
    ctx = await nxt(ctx)                            # run the rest of the chain + the real call
    return ctx                                      # (could also inspect/patch ctx.response here)

unsubscribe = agent.event_manager.intercept("llm_call", redact_secrets)
...
unsubscribe()                                       # intercept() returns a remover
```

Verified semantics:

- **Order**: registration order = execution order; first registered is outermost. Nesting across kinds: `agent_call` → per-turn `llm_call` → per-cell `execute_python`.
- **Short-circuiting** (don't call `nxt`) is allowed for guardrails/caching, but you MUST set the output slot (`ctx.result` / `ctx.response`) — the runtime raises `RuntimeError` if middleware returns without it. To fake an LLM turn, construct an `LLMResponse` (`content`, `tool_calls=[]`, `finish_reason="stop"`, `assistant_message={...}`, `raw_response=None`).
- **Blocking**: raise from the middleware — the exception propagates to the caller exactly like a failure of the wrapped operation (for `llm_call`, CodeAct counts it against its session error budget).
- **Exceptions are NOT swallowed** — middleware is control flow, unlike hooks.
- Per-agent: registered on that agent's `EventManager`; subagents have their own.
- `execute_python` has a re-entry guard: code the middleware itself triggers (e.g. it calls agent methods) skips the middleware, while nested generation methods called by executed code re-enter it for their own cells.
- On a context-window error the runtime archives events, rebuilds messages, and retries — so `llm_call` middleware can run more than once per logical turn; keep it idempotent.
- The tracing `on_messages_built` hook fires inside the innermost core, so traces show the **post-middleware** messages.

Worked production example: `src/nooa/nemo_relay_middleware.py` installs all three kinds to route calls through NeMo Relay (guardrails/ATIF); `nemo_relay_scope(agent, name)` wraps install/uninstall. Runnable: `examples/quickstart/13_nemo_relay.py`. Test patterns (mutate/short-circuit/ordering): `tests/test_event_middleware.py`.

## Observers (`on`)

Fire-and-forget callbacks after an event is recorded — they cannot alter it.

```python
unsub = agent.event_manager.on("Error", lambda e: log.warning("agent error: %s", e.content))
agent.event_manager.on("*", audit)                 # wildcard: every event
```

- Useful runtime-only events (never rendered to the model): `BeforeTurn` / `AfterTurn` (per generation turn; `AfterTurn.is_final` marks method completion) and `LLMComplete` (tokens, cost, model_name, tool_calls, reasoning metadata per round-trip — emitted precisely so you don't need `intercept("llm_call")` just to read LLM metrics).
- Model-visible events (`Task`, `Message`, `Error`, `PythonOutput`, ...) are observable the same way — see `nooa-context-and-state` for the full list.
- The summarizers are the house pattern: subscribe to `AfterTurn` to *schedule* work, apply it at the next `BeforeTurn` (`agents/summarization.py:159-160`).

## InstrumentationHooks (`set_hooks`)

A `Protocol` of paired callbacks (`nooa.runtime.hooks`): `before_/after_agent_call`, `before_/after_generation`, `before_/after_code_execution`, `before_/after_method_invocation`, `before_/after_tool_execution`, plus the point-in-time `on_messages_built`. Each `before_*` may return a context object that is handed to its `after_*` (span/timing state without globals).

```python
from nooa.runtime.hooks import set_hooks

class TimingHooks:
    def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id, **kw):
        return {"t0": time.perf_counter()}
    def after_agent_call(self, agent, method_name, result, exception, context, **kw):
        print(f"{method_name}: {time.perf_counter() - context['t0']:.2f}s")
    # unimplemented methods are fine — calls are defensive

set_hooks(TimingHooks())   # set_hooks(None) removes
```

- Hook exceptions never affect the agent (logged at WARNING); time spent in hooks is metered as tracing overhead.
- **One slot, contextvar-scoped — and tracing uses it.** `enable_tracing()` installs `OpenInferenceHooks` via this same `set_hooks` (`tracing/__init__.py:136-148`); calling `set_hooks(MyHooks())` afterwards silently replaces tracing (and vice versa). If you need both, delegate to the previous hooks object from yours, or prefer a tracing exporter / `on()` observers.
- `after_agent_call` can receive `context=None` when `agent_call` middleware short-circuits before the before-hook ran — handle it.

## Pitfalls

- Don't use hooks for app logic (they're swallowed-exception observational); don't use middleware for metrics you can get from `LLMComplete` (you'd pay complexity for nothing).
- `AgentCallContext.result` uses a not-set sentinel — a short-circuiting `agent_call` middleware that "returns None" on purpose must still assign `ctx.result = None`.
- Middleware lives on the instance's event manager: install in `__init__` (after `super().__init__()`) or on the constructed agent, not on the class.
- Keep `llm_call` middleware fast — it's on the critical path of every turn, and runs again on context-window retries.

## Related skills

- `nooa-context-and-state` — the event model that `on()` observes; `EventQuery` filtering.
- `nooa-capturing-traces` — the tracing system that occupies the hooks slot; exporters are usually the right telemetry surface.
- `nooa-codeact-advanced` — what `execute_python` middleware wraps (validator pipeline, restrictions, cells).
