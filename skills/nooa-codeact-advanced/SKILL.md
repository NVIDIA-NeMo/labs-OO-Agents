---
name: nooa-codeact-advanced
description: Advanced tuning of NVIDIA OO Agents strategies — CodeAct prefill (understanding, disabling, custom, pre-ellipsis code), loop guards (max_iterations, retries, text-only stop), truncation tuning (TruncationConfig/CaptureConfig/FormatConfig), code restrictions (RestrictionsConfig), execution environment internals, and PredictStrategy tuning (retries, param guards, output_serialization). Use when configuring CodeActConfig or PredictConfig beyond defaults, writing a custom prefill, restricting generated code, or debugging truncation/eviction behavior.
compatibility: nooa package
---

# Advanced CodeAct (and Predict) Tuning

The authoring basics are in `nooa-agent-authoring`. This skill covers the deep configuration surface, verified against `strategies/codeact.py`, `strategies/prefill.py`, `strategies/predict.py`, and `config/`.

## Config plumbing rules (read first)

- Strategy constructors take **`config=` only**: `CodeActStrategy(config=CodeActConfig(...))`, `PredictStrategy(PredictConfig(...))`. Flat kwargs (`CodeActStrategy(max_iterations=10)`, `CodeActStrategy(prefill=...)`) do not exist, despite some docstring examples.
- All config objects are frozen Pydantic models with `merge_with(other)`: only fields explicitly set on `other` override. **Configs must be freshly constructed** — anything round-tripped through `model_dump()`/`model_validate()` has an empty `model_fields_set` and `merge_with` raises.
- Truncation layers: framework default → `Agent` class kwarg `truncation=` → instance kwarg → `@strategy(..., truncation=...)` per method. `TruncationConfig.merge_with` deep-merges sub-configs field-by-field; the strategy configs merge flat.

## Prefill

Before the first LLM turn, CodeAct runs up to two **synthetic `execute_python` cells** (tool-call id prefixed `prefill_`, `execution_count=0`). They cost no loop iterations, their failures are non-fatal (logged as an `Error` event), and any variables they define persist as live REPL locals for the model's first real turn.

1. **The prefill plugin** (default `InspectInputsPrefill()`): generates code that prints the call signature, `pprint()`s each parameter under `truncation.prefill_format` limits (defaults `max_string=2000, max_length=25, max_depth=4`), auto-`show()`s `Media` parameters for multimodal perception, and prints `doc()` of a complex return type. Per-parameter limits come from `Annotated[T, spec(max_string=..., max_length=...)]` on the signature.
2. **Pre-ellipsis code**: statements between the docstring and the `...` are extracted from the AST and run verbatim as the second prefill cell — with or without the plugin.

```python
class Notifier(Agent, llm=llm):
    async def notify_stale(self) -> str:
        """Notify owners of stale tasks. Decide who needs pinging."""
        overdue = [t for t in self.all_tasks() if t.overdue]      # pre-ellipsis: runs first,
        print(f"{len(overdue)} overdue")                          # LLM sees output + variables
        ...
```

- **Disable** the plugin: `CodeActConfig(prefill=None)` (pre-ellipsis code still runs). The `CodeActStrategy.__init__` docstring claiming "prefill is always enabled" is outdated — the config is honored.
- **Custom prefill**: any object implementing `get_code(self, call: CurrentCall, config=None) -> str | None` (return `None` to skip; `config` is the agent's `TruncationConfig`). Pass it via `CodeActConfig(prefill=MyPrefill())`.
- Inside prefill cells only, the builtin `_call` (the `CurrentCall`) is available — that's how `InspectInputsPrefill` prints `doc(_call.return_type)`.
- Pre-ellipsis code runs through the full validator/restriction pipeline; `await` is allowed (cells are wrapped in an async function), and async `self.method()` calls must be awaited or the whole cell is rejected.

## Loop guards and knobs (`CodeActConfig`)

| Knob | Default | Verified semantics |
|---|---|---|
| `max_iterations` | `None` | **Unlimited.** The loop then stops only on completion, the error budget, or a hard abort. |
| `max_retries` | `3` | **Cumulative session error budget, not consecutive** (the counter is never reset). LLM API errors, bad tool JSON, empty code, and `return_result` validation failures all count. |
| `max_consecutive_text_only` | `3` | Consecutive no-tool-call text replies before hard abort; `0` disables. Any real tool call resets the counter. |
| `text_only_stop_behavior` | `"return_result"` | Text-only reply → try to validate the text as the final result; on failure, a visible correction `Error` is added. `"synthetic_reasoning"` instead converts the text to a no-op `reasoning(...)` cell whose tool result says the task is NOT finished. |
| `cell_timeout` | `None` | Per-cell `asyncio.wait_for` limit in seconds; `None` = unlimited. Cannot interrupt a truly blocking sync syscall — that's what the blocking-call AST validation is for. |
| `max_tokens` / `temperature` / `top_p` | `None` | Passed to every generation call when set (model defaults otherwise). On empty responses with `finish_reason="length"` CodeAct aborts and tells you to raise `max_tokens` (16384+ for reasoning models). |
| `translate_tool_calls` | `False` | When a weak model calls an agent method directly as a tool (instead of via `execute_python`), rewrite it into equivalent code and run it — teaching the right pattern. Off = error listing the two valid tools. |
| `restrictions` | `RestrictionsConfig()` | See Restrictions below. |
| `prefill` | `InspectInputsPrefill()` | See Prefill above. |
| `max_tool_calls` | `None` | **Dead — declared but never read.** Setting it does nothing. |

`tool_choice` is hardcoded `"auto"`. There is no `allow_text_response` option (older docs mention one) — text handling is entirely the two text-only knobs.

## `return_result` mechanics

The LLM finishes via the `return_result` tool, or `return_result(value)` inside a cell (raises a control-flow signal — not an error — stopping the cell immediately), or an explicit `return X` statement whose value validates against the return type. All three route through the same validation, and inline completions emit a synthetic `return_result` ToolCallEvent so trajectories always show the final answer.

Coercion before validation, in order: bare values wrapped to the schema; string results resolved as REPL variable names; JSON / `ast.literal_eval` parsing; `ClassName(...)` constructor strings evaluated in an empty-builtins sandbox. Validation failures are fed back to the model (with a tip to call `return_result(variable)` from inside code) and count against `max_retries`. Non-JSON-schemable return types (DataFrames, arbitrary classes) degrade to an `Any` schema plus `doc(<type>)` in the tool description and an `isinstance` check at the end.

## Execution environment

- **exec_globals** = the agent module's namespace across the full MRO (minus `hidden` names) + method parameters as variables + auto-injected symbols: `self`, `asyncio`, `typing` (+`Annotated/Any/Literal/Optional/Union`), `doc`, `pprint`, `show`, media classes, `strategy` + all strategy classes, `reasoning`, `return_result`. `help` is shadowed to `doc` so nothing blocks on stdin.
- **REPL state persists** across cells (globals-declared wrapper); helper functions defined in cells persist as REPL locals — they are never attached to the agent, and attaching callables to `self` is validator-rejected. Return-type names are re-injected if the model clobbers them. `Out[n]` gives Jupyter-style access to prior cell values.
- **Media**: `show(obj)` attaches images/files to the cell's output, capped by `truncation.media_capture.max_attachments_per_execution` (default 5, **per cell**).
- **Custom error formatting**: `CodeActStrategy(config=..., error_formatter=obj)` where `obj.format(error, code=None, *, line_offset=0) -> str`. Default is the IPython-style formatter (cell/line references, framework frames filtered, signature hint on call-shape TypeErrors).

## Truncation tuning (`TruncationConfig`)

Two independent mechanisms — set them separately:

1. **Capture time** (`capture: CaptureConfig`) — per-cell stream caps with head/tail truncation: `max_stdout=50_000`, `max_stderr=2_000`, `max_error=10_000` chars; `tail=None` (half the budget); `file_backed=True` (default `False`) streams the *full* output to a temp file and puts its path in the truncation notice so the LLM can grep it.
2. **Render time** (`FormatConfig` triplet = `pformat` bounds `max_string/max_length/max_depth`): `event_format` (event fields, every turn; defaults 10_000/200/5), `prefill_format` (parameter inspection; 2_000/25/4), `context_block_format` (non-string context values; unlimited — overflow handled by block eviction).

Context budget: `max_context_tokens` — or, when unset and the model exposes a context window, half the *usable* window (`(context_window - reserved_output) // 2`, where `reserved_output` is the call's `max_tokens`, else `TruncationConfig.response_reserve_tokens`, default 4096; setting `response_reserve_tokens=0` disables the auto-derived budget). Over-budget triggers **whole-block eviction** of SYSTEM context blocks — newest non-`static` user blocks first, marked `EVICTED: over context budget` in place. Static framework blocks (`system_prompt`, `self`) are never evicted. No token counter is required: eviction sizes blocks as `chars × ratio`, where the ratio is calibrated from each provider response's reported prompt tokens (cold-start ~4 chars/token before the first response). Inspect usage via `agent.context_stats` — token figures are provider-reported (`prompt_tokens`); the context-vs-events breakdown is attributed by character share and `ctx N%` is measured against the usable window.

**Dead/stale knobs**: `max_event_tokens` and `min_preserved_events` are read by nothing — events are never evicted by the renderer. Older notes may mention removed flat fields such as `max_block_chars` or `max_stdout_chars`; trust `config/truncation_config.py`.

## Restricting generated code (`RestrictionsConfig`)

```python
from nooa.runtime.restrictions import RestrictionsConfig
CodeActConfig(restrictions=RestrictionsConfig(
    blocked_modules=frozenset({"subprocess", "socket", "requests"}),  # hard: stripped + import-denied
    restricted_imports=frozenset({"os", "shutil"}),                    # soft: import statements denied
    blocked_calls={"time": frozenset({"sleep"})},                      # per-module call denylist
))
```

- Defaults: 12 event-loop-hazard modules hard-blocked (subprocess, socket, smtplib, ...); blocking calls denied (`time.sleep`, `os.system`, `Thread.join`, `asyncio.run`, ...); `restricted_imports` **empty** — any other installed module is importable.
- Always blocked regardless of config: `exec`/`eval`/`compile`/`__import__`/`input`/`breakpoint`/`globals`/`locals`/`vars`, `from x import *`, dangerous dunder access (`__class__`, `__globals__`, ...), process-exit calls, self-recursion into the currently-executing method, and infinite `while True` without break.
- Process-global default: `set_restricted_imports(frozenset({...}))` / `get_restricted_imports()` (from `nooa.runtime.restrictions`) — applies to every subsequently-constructed `RestrictionsConfig` that doesn't set the field explicitly. Generated code cannot call these (they're in the forbidden-builtins list).
- Violations are **not fatal**: the validator error comes back as the cell's error output with a fix hint, and the model retries (within the `max_retries` budget).
- Agent-in-agent recursion depth is a separate class-level guard: `class MyAgent(Agent, execution=ExecutionConfig(max_nesting_depth=5))` (default 10; class kwarg only).

## Overriding the strategy prompt

Not a config field — use the decorator's scoped-context:

```python
from nooa.context_blocks import ScopedContext

@strategy(CodeActStrategy(), ScopedContext(context={"strategy_prompt": "My leaner instructions..."}))
async def solve(self, q: str) -> str: ...
```

Block order is `[system_prompt, strategy_prompt, execution_context, self]`; both strategy blocks are dynamic expressions you can replace per method.

## Advanced PredictStrategy (`PredictConfig`)

Single LLM turn, no tools, no code. The prompt is the docstring plus each parameter serialized as a Python assignment (bounded by `truncation.prefill_format` + per-param `spec()` overrides); the output schema is enforced via the provider's structured-output `response_format`, not prompt text.

| Knob | Default | Semantics |
|---|---|---|
| `max_retries` | `10` | Validation retries. Each failure adds an `Error` event with the formatted validation error plus the raw output truncated to `max_error_chars` (1000). Exhaustion raises `GenerationError` (not ValidationError). |
| `max_param_chars` | `200_000` | **Hard pre-flight guard, not a truncator**: any parameter whose `repr` exceeds it raises `ValueError` — Predict is single-shot, so a silently-truncated input would mean silently-wrong output. Chunk the input or raise/disable (`None`) the limit. |
| `max_tokens` / `temperature` / `top_p` | `None` | Forwarded to the LLM call when set. |
| `output_serialization` | `"event"` | How the result is recorded in history: `"event"` keeps the LLMOutput (replayed as a plain assistant message); `"tool_call"` replaces it with a synthetic `return_result` ToolCallEvent — prefer it when a downstream tool-using model reads this history. |

- Return-type handling: `Optional[X]` unwraps; `dict[K,V]` uses a root-object schema; bare `list`/scalars are wrapped in a hidden `{"value": ...}` schema (Responses-API rejects array-rooted schemas) and unwrapped after validation; models with hidden fields get a public-subset schema and are rehydrated. Non-JSON-serializable **return types** (DataFrame, ndarray) are rejected up front with a pointer to CodeAct (parameters are only size-checked via `max_param_chars`).
- Reasoning models: the JSON must land in `content`; `reasoning` is only used as a fallback when content is empty. Prose-in-content + JSON-in-reasoning fails and retries.
- Predict sees **all prior conversation events** by default. Isolate it with `@strategy(PredictStrategy(), ScopedContext(events=EventQuery.current_call()))`.
- `PredictStrategy(max_retries=3)` (from an old docstring) constructs a broken strategy — always wrap in `PredictConfig`.

## Related skills

- `nooa-agent-authoring` — the basics this builds on (strategy selection, contracts, visibility).
- `nooa-context-and-state` — the context blocks that truncation/eviction act on.
- `nooa-capturing-traces` / `nooa-trace-explorer` — see every prefill cell, tool call, and validation retry in the trace.
