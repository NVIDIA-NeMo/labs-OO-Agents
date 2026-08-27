# Design: Replace LiteLLM with AnyLLM behind the UnifiedLLM boundary

**Status:** Proposed for design review — no implementation has started<br>
**Repository baseline:** `origin/main` at `97f52dec84ed88ca3b202f91bee0bc0074626246` (fetched 2026-08-27)<br>
**Implementation worktree:** `/localhome/local-pfurgale/dev/labs-OO-Agents-anyllm`, branch `feat/anyllm-unified-boundary`
**AnyLLM reference:** Mozilla `any-llm-sdk` 1.26.0 (published 2026-08-17); upstream source also inspected at `e822b281dc6d41578c02ba33cf6c50e9081651cf`

## 1. Decision summary

Replace LiteLLM with Mozilla AnyLLM by preserving NOOA's `nooa.unifiedllm` API as the framework-facing boundary and putting all AnyLLM knowledge in one private adapter module:

```text
NOOA runtime / strategies / viewer / relay / tracing / memory
                         |
                         v
 nooa.unifiedllm public protocol + NOOA-owned request/response/errors
                         |
                         v
          nooa.unifiedllm._anyllm (private adapter)
                         |
                         v
                  Mozilla any-llm-sdk
```

The migration is **not** an import substitution. LiteLLM currently owns transport, provider routing, response types, exceptions, retries, HTTP clients, model metadata, token estimation, tracing callbacks, message journaling, cost calculation, and a few process-wide switches. Those responsibilities must either move behind the boundary or be removed.

The proposed end state is:

1. `nooa` outside `nooa.unifiedllm._anyllm` imports neither `any_llm` nor provider SDK response/exception types.
2. Public NOOA objects contain only NOOA/Python/Pydantic types. There is no public `raw_response` escape hatch.
3. Public model configuration is provider-neutral and validated. Backend-specific options are quarantined under an explicit `provider_options` mapping.
4. NOOA performs **no local tokenization, token estimation, token calibration, or input-context decisions based on tokens**. Context policy is character-bounded plus provider-error-driven recovery. Provider-native output limits such as `max_output_tokens` remain request controls; they tell the provider how much it may generate and are not token counting by NOOA.
5. Provider-reported usage and provider-declared `context_window` metadata may be copied into passive observability fields, but are never used for sizing, eviction, summarization, retry, scoring, or other control flow. This is observation, not NOOA token counting.
6. Tracing and message journaling use NOOA's own call lifecycle, not LiteLLM callbacks or AnyLLM internals.
7. NVIDIA inference is configured as an OpenAI-compatible endpoint inside the private adapter and is covered by credential-gated live tests.

## 2. Goals and non-goals

### Goals

- Replace the direct runtime dependency on `litellm` and `openinference-instrumentation-litellm` with `any-llm-sdk`.
- Preserve the common user flow (`get_llm_client(...)`, `CompletionClient`, `ResponsesClient`, `.call()`, `.acall()`) where it is provider-neutral.
- Normalize completion, tool-call, reasoning, structured-output, streaming, usage, and error behavior into NOOA-owned types.
- Remove every LiteLLM bypass in production code, including viewer inference, eval setup, memory embeddings, tracing, and model registry assumptions.
- Remove deprecated token-counting APIs and all local token-based estimation/calibration.
- Test normal, structured, tool-call, streaming, retry, error, and cancellation behavior against hermetic fixtures, then smoke-test the NVIDIA inference endpoint.

### Non-goals

- Exposing AnyLLM's complete provider API through NOOA.
- Preserving arbitrary LiteLLM kwargs or LiteLLM routing strings indefinitely.
- Recreating LiteLLM's model-price database or local cost calculator.
- Building a second general-purpose provider router beside AnyLLM.
- Guaranteeing every AnyLLM provider in the first implementation. Existing documented NOOA providers and NVIDIA receive explicit coverage; others are best-effort until tested.
- Changing agent/strategy semantics unrelated to the LLM transport migration.

## 3. Constraints established from current code

The current core depends unconditionally on LiteLLM and its instrumentor (`pyproject.toml:24-34`, with the instrumentor repeated in the tracing extra at `pyproject.toml:99-105`). The primary adapter is a 2,600-line module that imports and configures LiteLLM at import time (`src/nooa/unifiedllm/unifiedllm.py:17-36`). There are 31 production Python files containing LiteLLM references, plus tests, docs, examples, and package metadata.

AnyLLM 1.26.0 supports Python >=3.11 and NOOA supports >=3.12,<3.14. The base AnyLLM install includes OpenAI and Anthropic SDKs. It exposes sync/async Chat Completions and Responses APIs, normalized OpenAI-style Pydantic responses, tool calls, structured output, streaming, and `AnyLLM.create_openai_compatible(...)`. It does **not** supply equivalents for LiteLLM's tokenizer/model-cost database or callback-based OpenInference instrumentation.

## 4. Current leaks and their disposition

### 4.1 Public response leaks

| Current leak | Evidence | Disposition |
|---|---|---|
| `LLMResponse.raw_response: Any` exposes the backend object | `src/nooa/unifiedllm/unifiedllm.py:673-683` | Remove from public response. Private adapter may hold the provider object only while normalizing it. |
| NeMo Relay serializes the raw LiteLLM object | `src/nooa/nemo_relay_middleware.py:185-194` | Serialize `LLMResponse.to_wire()` / normalized fields. |
| CodeAct reads `raw_response.output` for empty-response diagnostics | `src/nooa/strategies/codeact.py:1094-1104` | Add provider-neutral bounded `diagnostics: Mapping[str, JSONValue]`; never expose provider object. |
| `usage` is annotated as `dict[str, int]`, but actor handles dicts, objects, aliases, details, and float costs | `src/nooa/runtime/actor.py:1140-1212` | Normalize once to `LLMUsage`; consumers do no shape probing. |

### 4.2 Configuration/routing leaks

| Current leak | Evidence | Disposition |
|---|---|---|
| Arbitrary `**config` and `**kwargs` flow directly to LiteLLM | `unifiedllm.py:1153-1159, 1696-1729, 2216-2249` | Typed common config/call options; explicit `provider_options` escape hatch. |
| Registry is intentionally raw for LiteLLM passthrough | `src/nooa/unifiedllm/registry.py:180-186` | Validate entries into `ModelConfig`; keep raw YAML only at loader edge. |
| LiteLLM routing strings are public (`openai/...`, `nvidia_nim/...`) | `registry.py:297-335`; `README.md:142-150` | Canonical separate `provider` and opaque `model_name`. Parse legacy strings only in a deprecation shim at the boundary. |
| LiteLLM controls (`drop_params`, `allowed_openai_params`, `additional_drop_params`) are first-class config | `registry.py:358-390` | Remove. Unsupported common parameters fail clearly; endpoint-specific ones live under `provider_options`. |
| Process globals are mutated at import (`modify_params`, `disable_aiohttp_transport`) | `unifiedllm.py:26-36` | Delete. Adapter construction is explicit and per-client. |
| LiteLLM internals are used to build HTTP handlers/clients | `unifiedllm.py:104-307` | Construct and retain a NOOA-owned `httpx.AsyncClient`, then pass it as the ordinary OpenAI client kwarg `http_client` through AnyLLM's public provider constructor. Do not invent a `client_args` wrapper or import AnyLLM internals. |

### 4.3 Error leaks

- Retry logic classifies exceptions via status attributes, class-name strings, and provider message text (`src/nooa/unifiedllm/retry.py:23-85,155-202`).
- Runtime imports `litellm.exceptions.ContextWindowExceededError` and parses message strings (`src/nooa/runtime/actor.py:262-349`).

**Disposition:** the private adapter catches AnyLLM/provider exceptions and raises a NOOA hierarchy:

```python
class LLMError(Exception):
    provider: str | None
    status_code: int | None
    code: str | None

class LLMAuthenticationError(LLMError): ...
class LLMRateLimitError(LLMError):
    retry_after: str | None
class LLMInvalidRequestError(LLMError): ...
class LLMModelNotFoundError(LLMError): ...
class LLMContextLengthError(LLMInvalidRequestError):
    # Provider-declared metadata for display/diagnostics only; never a sizing input.
    context_window: int | None
class LLMTransportError(LLMError): ...
class LLMProviderError(LLMError): ...
```

The original exception remains only as `__cause__`, not as a typed public field. Retry consumes this hierarchy and structured HTTP metadata. Message parsing remains a bounded compatibility fallback inside `_anyllm`, never in runtime.

AnyLLM unified exceptions are currently controlled by the process environment and checked at exception-handling time. NOOA will **not** mutate `ANY_LLM_UNIFIED_EXCEPTIONS`: the adapter locally normalizes both AnyLLM exceptions and raw provider SDK exceptions. Mappings cover authentication, rate limit/retry-after, invalid request, model-not-found, context length, content filter, insufficient funds, upstream/gateway timeout, transport failure, unsupported parameter, and provider failure. Pydantic `ValidationError` remains a structured-output validation error. Tests cover both AnyLLM-wrapped and raw OpenAI SDK paths so behavior is independent of ambient environment.

### 4.4 Direct backend bypasses

| Area | Current behavior | Target behavior |
|---|---|---|
| Viewer playground | Calls `litellm.acompletion` and consumes LiteLLM stream classes (`src/nooa/viewer/trace_routes.py:644-720`) | Calls `get_llm_client().acall(stream=True)` and consumes a NOOA `LLMStream`. |
| Eval pipeline | Imports LiteLLM and mutates `drop_params` globally (`util/eval_pipeline/.../model_factory.py:129-146`, `config.py:410-416`, `subprocess_worker.py:485-503`) | Uses registry/client configuration only. |
| Memory embeddings | Public `LiteLLMEmbedder` calls `litellm.embedding` (`packages/nooa-memory/src/nooa_memory/embeddings.py:60-105`) | Rename to `LLMEmbedder`; use a small NOOA embedding port backed by AnyLLM, with normalized vectors/errors. |
| Runtime | Imports LiteLLM context errors and mentions provider implementation | Imports only normalized UnifiedLLM errors/types. |
| Tracing/journal | Mutates LiteLLM callback lists, subclasses `CustomLogger`, monkey-patches instrumentor internals, computes LiteLLM costs | Uses provider-neutral NOOA call hooks and normalized request/response snapshots. |
| Docs/examples/tests | Use LiteLLM names, model prefixes, mocks, and classes | Use UnifiedLLM names and NOOA fixtures; only private adapter tests mention AnyLLM. |

### 4.5 Media/message-shape leaks

Media code calls OpenAI-style blocks “LiteLLM format” (`src/nooa/runtime/media_capture.py:8-92`, `context_blocks/models.py:271-303`). OpenAI-compatible message dictionaries are a useful **NOOA canonical wire shape**, not an AnyLLM type. Rename documentation accordingly and centralize any provider conversion in `_anyllm`. No AnyLLM class enters event/context models.

### 4.6 Tracing and journaling leaks

Current tracing depends on:

- LiteLLM callback registration and global callback-list mutation (`_journal_exporter.py`, `_journal_file_exporter.py`);
- `CustomLogger` and `litellm_call_id` (`_litellm_journal.py`);
- OpenInference LiteLLM internals and monkey patches (`_litellm_patch.py`);
- LiteLLM response-cost headers and local cost calculation (`_litellm_patch.py:65-153`);
- `LiteLLMInstrumentor` (`src/nooa/tracing/__init__.py:440-456`).

Replace these with a provider-neutral call lifecycle owned by UnifiedLLM:

```python
@dataclass(frozen=True)
class LLMCallRecord:
    call_id: str
    model: str
    messages: tuple[Message, ...]
    options: LLMCallOptions
    response: LLMResponse | None
    error: LLMError | None

# Internal, context-local observer; not an AnyLLM callback.
with observe_llm_call(record_builder):
    ...
```

This lifecycle sits **above the transport adapter**, in the public UnifiedLLM/runtime call path, so it also observes `FakeLLMClient`, duck-typed clients, and future backends. One logical call receives one `call_id` and one terminal event. A context-recovery rebuild creates a new logical call because it sends a different prompt; retries of an identical payload remain attempts under the same logical call (`attempt=1..N`). Each attempt records monotonic start/end time and one of success/error/cancelled; streaming additionally records first-chunk time and a terminal stream event. Only the terminal successful attempt contributes a response/usage record, preventing double counting.

Hooks are `before_llm_call(snapshot)`, `before_llm_attempt(snapshot)`, `on_llm_chunk(snapshot)`, and `after_llm_call(snapshot)`. Snapshots deep-copy JSON-safe messages/options, apply the existing secret scrubber before dispatch to exporters, cap diagnostic strings/containers, and never retain mutable caller objects. Cancellation always emits exactly one cancelled terminal event and re-raises. The existing content-addressed journal builder consumes these snapshots, and the NOOA OpenInference hook emits the LLM span beneath the active generation span. Cost is retained only when supplied by the endpoint/gateway as an authoritative value; local model-price calculation is removed.

## 5. Token counting removal

“Token counting is deprecated” is interpreted as a hard architectural rule: **NOOA will not locally tokenize, estimate, calibrate, or make input-context decisions from token counts.** Provider-native output limits remain request controls. Provider-reported usage and provider-declared context metadata may remain passive telemetry, because copying an API observation is not counting; neither may influence sizing, eviction, summarization, retry, scoring, or other control flow.

### Remove

- `UnifiedLLM.count_tokens` (`unifiedllm.py:1299-1313`).
- `TokenCalibration`, `_token_calibration`, and `_update_token_calibration` (`:1045-1150`) and all calibration calls.
- `FakeLLMClient.count_tokens` (`src/nooa/unifiedllm/fake.py:41-48`).
- `nooa.char_approximate_token_counter`, `src/nooa/token_counter.py`, and root export (`src/nooa/__init__.py:82,173`).
- Summarizer probing of `llm.count_tokens` and `_input_token_counter` (`src/nooa/agents/summarization.py:464,491-503`).
- Runtime `tokens_per_char` calibration and the local `count_tokens` closure (`src/nooa/runtime/actor.py:631-635,1177-1197,3031-3057`).
- `max_context_tokens`, `max_event_tokens`, and `response_reserve_tokens` as active controls; token-derived context statistics and token-based truncation callbacks.
- The task-token accumulator as a runtime service (`src/nooa/runtime/token_usage.py`) and any benchmark **scoring/control** based on token totals. Bench may aggregate passive `reported_usage` for reporting only.
- Token totals in context utilization UI/trace explorer/ACP as control or capacity fields. Passive provider usage can be shown under `reported_usage`, clearly labeled as provider-reported and possibly absent.

### Replace with

- Existing deterministic character caps for captured and formatted values.
- New `max_context_chars`, `max_event_chars`, and `response_reserve_chars` configuration, with no model-specific claim of equivalence.
- Character-only context stats (`context_chars`, `event_chars`, dropped counts).
- Error-driven context recovery: catch normalized `LLMContextLengthError`, archive a deterministic character-sized batch of the oldest eligible events, rebuild, and retry with a bounded attempt count. The retry may reduce the provider-native `max_output_tokens` only when the error provides a structured maximum/required delta; that is provider-error remediation, not local counting. If no structured values exist, retain the output limit and reduce input by characters.
- Summarization input bounded by `max_input_chars`, configured directly. There is no model-window-derived default.

For one release, old token config keys can raise a targeted migration error rather than being silently ignored. The removed callable APIs should fail at import/access with a clear changelog entry; they are deprecated by product decision and should not retain behavior behind aliases.

## 6. Public API: before and after

### 6.1 Client creation

Before:

```python
llm = get_llm_client(
    "openai/qwen/qwen3-coder-480b-a35b-instruct",
    api_base="https://integrate.api.nvidia.com/v1",
    api_key=key,
    drop_params=True,
    allowed_openai_params=["extra_body"],
)
```

After:

```python
llm = get_llm_client(
    "qwen/qwen3-coder-480b-a35b-instruct",
    provider="openai-compatible",
    endpoint="https://integrate.api.nvidia.com/v1",
    api_key=key,
    request=LLMRequestDefaults(
        temperature=0.0,
        max_output_tokens=4096,
    ),
    provider_options={"extra_body": {...}},  # explicit quarantine, only if needed
)
```

Registry YAML before:

```yaml
models:
  coder:
    model_name: openai/qwen/qwen3-coder-480b-a35b-instruct
    api_base: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
    drop_params: true
```

Registry YAML after:

```yaml
models:
  coder:
    model_name: qwen/qwen3-coder-480b-a35b-instruct
    provider: openai-compatible
    endpoint: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
    context_window: 131072       # optional declarative metadata, never inferred
    request:
      temperature: 0.0
      max_output_tokens: 4096
    provider_options: {}         # explicit, audited escape hatch
```

`api_base` remains accepted as a one-release deprecated alias for `endpoint`; legacy LiteLLM-prefixed model strings are normalized with a warning where unambiguous. Unknown provider-specific top-level keys become validation errors rather than silently leaking through.

### 6.2 Call API

Before:

```python
response = await llm.acall(
    messages,
    tools=tools,
    output_model=Answer,
    max_tokens=4096,
    prompt_cache_key="agent-codeact",
    **arbitrary_litellm_kwargs,
)
```

After:

```python
response = await llm.acall(
    messages,
    tools=tools,
    output_model=Answer,
    options=LLMCallOptions(
        max_output_tokens=4096,
        cache_key="agent-codeact",
    ),
    provider_options={},
)
```

To reduce migration blast radius, named common kwargs may remain accepted initially and be converted immediately to `LLMCallOptions`; arbitrary kwargs do not. Sync `call` and async `acall` remain. Streaming is added to the boundary as `stream()/astream()` returning normalized `LLMChunk` values; it is not represented by backend iterators leaking from `acall`.

### 6.3 Response API

Before:

```python
@dataclass
class LLMResponse:
    raw_response: Any
    content: str | BaseModel
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    assistant_message: dict[str, Any]
    reasoning: str | None = None
    usage: dict[str, int] | None = None
```

After:

```python
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]

@dataclass(frozen=True)
class LLMUsage:
    # Passive provider observations only; never used for runtime control.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None

@dataclass(frozen=True)
class LLMResponse:
    content: str | BaseModel
    tool_calls: tuple[ToolCall, ...]
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter", "error", "unknown"]
    assistant_message: dict[str, JSONValue]
    reasoning: str | None = None
    reported_usage: LLMUsage | None = None
    diagnostics: dict[str, JSONValue] | None = None

    def to_wire(self) -> dict[str, JSONValue]: ...
```

`diagnostics` is bounded and provider-neutral (status, incomplete reason, refusal text); it cannot hold arbitrary objects. `to_wire()` supports Relay and trace journaling without exposing backend models.

### 6.4 Client interface

Before, annotations require `UnifiedLLM`, while runtime deliberately accepts duck types with `acall` (`src/nooa/method_llm.py:35-66`). After, formalize that behavior:

```python
@runtime_checkable
class LLMClient(Protocol):
    model: str
    # Provider-declared metadata for display/diagnostics only; never a sizing input.
    context_window: int | None
    async def acall(...) -> LLMResponse: ...
    def call(...) -> LLMResponse: ...
    async def aclose(self) -> None: ...
    def close(self) -> None: ...
```

The full protocol also includes `stream`, `astream`, `__enter__`, `__exit__`, `__aenter__`, and `__aexit__`; the abbreviated listing emphasizes the common calls. `UnifiedLLM` remains the standard implementation/base for source compatibility. `CompletionClient`, `ReasoningCompletionClient`, `ResponsesClient`, and `FakeLLMClient` remain public and satisfy `LLMClient`. No AnyLLM class appears in signatures, fields, exceptions, or docs outside the backend implementation section.

`context_window` is retained only as provider-declared metadata for display and diagnostics. Remove its control reads from `interactive.py`, summarization, ACP capacity events, and runtime budgeting/recovery. `get_model_info()` is removed; `.config` becomes a read-only provider-neutral `ModelConfig` snapshot. Context-manager methods are compatibility-guaranteed.

### 6.5 Compatibility and precedence matrix

| Surface | Migration behavior |
|---|---|
| `CompletionClient`, `ResponsesClient`, `ReasoningCompletionClient`, `FakeLLMClient` | Retained public names. |
| `call`, `acall`, context managers, `close`, `aclose` | Retained and covered. |
| `retry_config`, `http_config`, `cache_control_injection_points` constructors | Retained as provider-neutral options. Cache-control is represented by typed `CacheControlPoint(role, position)` rules and translated only by the adapter for declared-capable providers. |
| `max_tokens`, `prompt_cache_key` call kwargs | One-release deprecated aliases for `options.max_output_tokens` and `options.cache_key`; supplying both old and new forms is an error. |
| `api_base` | One-release deprecated alias for `endpoint`; both is an error. |
| Legacy provider-prefixed model strings | One-release warning and unambiguous normalization; ambiguous strings fail with migration guidance. |
| Arbitrary constructor/call kwargs and LiteLLM controls | Hard break; move intentional endpoint data under `provider_options`. |
| `LLMResponse.usage` | One-release read-only alias to `reported_usage`; values remain passive. |
| `LLMResponse.raw_response` | Hard removal; replaced by `to_wire()` and bounded diagnostics. |
| `tool_calls` list → tuple and frozen response | Deferred. Keep the current mutable/list shape during this migration to avoid an unrelated source break. |
| `.config` | One-release read-only mapping view of the typed config, then removal. |
| `get_model_info()` | Hard removal; explicit registry metadata replaces backend lookup. |
| `context_window` | Retained provider-declared metadata for display/diagnostics only; never read by control code. |
| Mutable raw `MODELS` values | Internal break. Migrate CLI, eval, resolved config, and viewer readers in the same phase before values become typed snapshots. |

Concrete common option types:

```python
@dataclass(frozen=True)
class LLMRequestDefaults:
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: str | tuple[str, ...] | None = None
    reasoning_effort: str | None = None
    cache_key: str | None = None
    timeout: float | None = None

@dataclass(frozen=True)
class LLMCallOptions(LLMRequestDefaults):
    stream: bool = False
    tool_choice: str | dict[str, JSONValue] | None = None
    parallel_tool_calls: bool | None = None
```

Effective values resolve in this order: typed call `options` > one-release named compatibility kwargs > constructor `request` defaults. A conflict between `options` and a compatibility alias raises `TypeError`; it never silently chooses. `provider_options` is merged last only after rejecting keys that collide with common options, credentials, endpoint, model, tools, or response format. `call/acall` always return a complete `LLMResponse`; `stream/astream` alone expose iterators of normalized `LLMChunk`.

### 6.6 Replay schema

`assistant_message` becomes a NOOA-owned replay structure with allow-listed fields: role, text content, normalized tool calls, refusal text, reasoning text, and `replay_metadata`. The latter accepts only bounded JSON under provider-name namespaces and only keys explicitly required for round trips (for example Anthropic thinking signatures or Gemini thought signatures). The adapter strips unknown extension fields. Multi-turn tests prove that tool and reasoning replay survives without carrying an AnyLLM/OpenAI object or arbitrary provider response.

## 7. Private AnyLLM adapter

Create `src/nooa/unifiedllm/_anyllm.py`, the only core module allowed to import `any_llm`.

Responsibilities:

1. Parse the provider-neutral model config into an AnyLLM provider instance.
2. For arbitrary OpenAI-compatible endpoints such as NVIDIA, call:

   ```python
   AnyLLM.create_openai_compatible(
       name="nvidia",
       api_base=endpoint,
       api_key=api_key,
       http_client=http_client,  # NOOA-owned httpx.AsyncClient
   )
   ```

3. Call `.completion/.acompletion` with an unprefixed model ID.
4. Use `.responses/.aresponses` for `ResponsesClient`. AnyLLM 1.26 marks generic custom OpenAI-compatible providers as not supporting Responses, while its built-in OpenAI provider does. Therefore the private adapter may construct `AnyLLM.create("openai", api_base=endpoint, ...)` for a configured Responses client, but that implementation detail must not change the public provider identity. Add a live NVIDIA Responses smoke test only for a model/endpoint known to support it.
5. Normalize AnyLLM/OpenAI-style responses into `LLMResponse`, including reasoning, all tool calls, finish reason, assistant replay message, passive usage, and bounded diagnostics.
6. Collect stream chunks into the same response for existing `call/acall(stream=True)` compatibility and expose normalized chunks for new `stream/astream` consumers. Accumulate tool-call fragments by stable index/id and preserve reasoning/extra content required for replay without exposing AnyLLM classes.
7. Translate exceptions to NOOA errors and preserve the original as `__cause__`.
8. Own provider lifecycle and close the AnyLLM/OpenAI async client safely. Preserve cancellation: on caller cancellation, cancel/close an unfinished stream and re-raise `CancelledError` without retry.
9. Participate in—but do not own—the provider-neutral lifecycle emitted above the adapter. It may attach attempt-level transport status through the internal hook protocol.

### Parameter mapping

| UnifiedLLM | AnyLLM | Notes |
|---|---|---|
| `model` | `model=` | Strip legacy LiteLLM provider prefix only in compatibility parser. |
| `endpoint` | `api_base=` | Adapter-only mapping. |
| `max_output_tokens` | completion parameter selected by endpoint capability; `max_output_tokens=` for Responses | AnyLLM's OpenAI base rewrites `max_tokens` to wire-level `max_completion_tokens`. NVIDIA profiles declare `output_limit_parameter = "max_tokens" | "max_completion_tokens"`; hermetic request-capture tests pin the emitted body per profile/model before live tests. |
| `cache_key` | `prompt_cache_key=` | Only send when endpoint/provider capability permits. |
| `tools` | OpenAI-style tool dicts | Produced from NOOA `Tool`; no AnyLLM callable wrapper. |
| `output_model` | `response_format=` | Prefer native parsed result; retain NOOA client-side parsing/validation fallback. |
| `provider_options` | final `**kwargs` | Explicitly isolated and never reflected as public attributes. |
| `HttpConfig` | NOOA-owned `httpx.AsyncClient`, passed as ordinary provider-constructor `http_client` kwarg | AnyLLM has no public close API in 1.26.0. NOOA owns this client: `aclose()` awaits it; `close()` runs the same coroutine directly when no loop is active and through a short-lived helper thread when a loop is active. Both are idempotent and contract-tested; no access to undocumented `provider.client`. |

### Responses capability and conversion

`ModelConfig` gains an explicit `capabilities.responses: bool = False`. `ResponsesClient` fails before network I/O unless it is true. The adapter maps the existing `ResponsesProviderFormatter` output as follows: system content to `instructions`; user/assistant text to Responses input messages; assistant tool calls to `function_call` items; tool results to `function_call_output`; and NOOA call IDs unchanged. It supports `previous_response_id` only as a typed option and normalizes incomplete reason, refusal, output items, and tool calls into `LLMResponse`. Formatter selection continues to use the public `ResponsesClient` class during compatibility, then may move to a declared `api_style` capability. A hermetic `/responses` test is mandatory even if the live NVIDIA profile does not enable Responses.

### Structured output

AnyLLM may return `ParsedChatCompletion.message.parsed`; the adapter extracts it when it is an instance of the requested model. Otherwise it uses normalized text plus existing `extract_and_parse_json()` and Pydantic validation. Existing retry-on-empty/invalid-output policy remains owned by UnifiedLLM, not AnyLLM.

## 8. Model registry migration

Evolve `ModelConfig` from `extra="allow"` to `extra="forbid"` with:

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_name: str
    provider: str
    endpoint: str | None = None
    api_key_env: str | None = None
    client_type: Literal["completion", "responses"] = "completion"
    # Provider-declared metadata for display/diagnostics only; never a sizing input.
    context_window: int | None = None
    request: LLMRequestDefaults = LLMRequestDefaults()
    provider_options: dict[str, JSONValue] = {}
```

The YAML loader still layers raw files, but validation happens before entries enter `MODELS`. `MODELS` should become a mapping of typed snapshots; external readers use `get_model_config()` or a JSON-safe `get_registry_config()`, not mutable internals.

Migration warnings identify exact renamed/removed keys. Secrets remain environment-variable references; logs never emit values.

## 9. Tracing/observability migration

1. Add a provider-neutral LLM span in NOOA's existing `OpenInferenceHooks` beneath the generation span. Populate standardized model, input/output message, tool, finish, error, and passive reported-usage attributes.
2. Move content-addressed message journal capture from LiteLLM callbacks to the same lifecycle. The runtime already publishes final rendered messages through `set_journal_payload_from_messages`; consume that sideband at `before_llm_call` and pair it by NOOA call ID.
3. Replace `_litellm_journal.py` with `_llm_journal.py`; delete `_litellm_patch.py` after porting only backend-neutral output/tool-call attribute behavior.
4. Remove `LiteLLMInstrumentor`, callback-list scans, `litellm_call_id`, and LiteLLM callback reset fixtures.
5. Record cost only when the endpoint reports it. Do not estimate from a local model/cost table.
6. Rename trace documentation and assertions from a nested `litellm.acompletion` span to a NOOA-owned `llm.call` span.

This is part of the provider replacement, not a later cleanup: otherwise changing transport silently removes production traces and journals.

## 10. Implementation sequence

### Phase A — establish the neutral contract

- Add `LLMClient`, `LLMCallOptions`, `LLMUsage`, normalized errors, normalized diagnostics, and `to_wire()`.
- Adapt `FakeLLMClient` and consumer tests first.
- Move runtime, CodeAct, Relay, and event emission off `raw_response` and loose usage.

### Phase B — remove token counting and change context policy

- Delete token-count APIs, calibration, and runtime token estimators.
- Introduce explicit character budgets and error-driven context recovery.
- Update summarization, context rendering/stats, bench, ACP, trace explorer, docs, and tests.
- Keep provider-reported usage passive and optional.

### Phase C — introduce AnyLLM transport

- Add private `_anyllm.py`; initially pin `any-llm-sdk==1.26.0` so a moving adapter API cannot drift during migration.
- Implement completion, Responses, streaming collection, tools, structured output, reasoning, cancellation, and error normalization. Logical lifecycle stays above the adapter; transport attempts provide internal details only.
- Capture NVIDIA-like request bodies hermetically before live tests, especially AnyLLM's `max_tokens` → `max_completion_tokens` rewrite.
- Make public clients delegate by composition; no subclassing from AnyLLM and no exported AnyLLM types.

### Phase D — eliminate bypasses and LiteLLM observability

- Migrate viewer, eval pipeline, memory embeddings, quickstarts, and integration helpers.
- Port tracing/journaling to NOOA lifecycle.
- Delete LiteLLM patches, callbacks, imports, environment settings, dependencies, notices, and filters.

### Phase E — compatibility cleanup and documentation

- Migrate bundled/project model YAML to provider + model fields.
- Update README, architecture, tour, tracing docs, skills, examples, changelog, and third-party notices.
- Add a CI guard: production paths may import `any_llm` only from `src/nooa/unifiedllm/_anyllm.py`; no production path may import/reference `litellm`. The public embedding protocol/client also lives in `nooa.unifiedllm` and delegates to `_anyllm`, so `nooa-memory` imports only the public NOOA boundary.

Each phase lands with passing tests; the final branch has zero `litellm` matches except an intentional changelog migration note.

## 11. Test plan

### Hermetic boundary tests

- Text completion, empty content, multiple choices policy, finish-reason mapping.
- Tool calls: multiple calls, missing IDs/names, malformed argument JSON preservation, streamed fragments.
- Structured output: native parsed object, JSON fallback, RootModel, invalid JSON, Pydantic failure, length/content-filter finish.
- Reasoning: normalized reasoning string and replay-required provider metadata converted into NOOA-owned JSON.
- Streaming: sync and async, content-only, usage-only chunks, tool fragments, cancellation, mid-stream exception, no leaked task.
- Errors: 400, auth, 404/model, context length, 429/retry-after, timeout/connection, 500/502/504; retry only normalized retryable errors.
- HTTP: per-client limits/timeouts, exact constructor kwargs, ownership of the injected `httpx.AsyncClient`, sync-close while an event loop is/is not active, async-close, idempotence, and concurrent-client isolation.
- Registry: typed validation, legacy-key warnings, env key resolution, unknown-key rejection, secret redaction.
- Boundary guard: AST/import scan proving AnyLLM does not appear outside private adapter/tests and LiteLLM is absent from production/dependencies.
- Token-removal guard: no `count_tokens`, tokenizer library, token calibration, or token-budget API remains; provider `reported_usage` is never read by context/summarization control code.

### Consumer tests

- Agent generation through `FakeLLMClient` and AnyLLM adapter fixture.
- CodeAct and Predict text/tool/structured paths.
- Relay serialization from `LLMResponse.to_wire()`.
- Viewer completion/streaming through UnifiedLLM only.
- Tracing span parentage, journal payload, tool-call IDs, errors, and shutdown flushing without callbacks.
- Memory embedding batches through normalized embedding port.
- Eval subprocess startup without global provider mutation.

### Named migration inventory

- Registry readers/mutators: CLI model/config commands, eval config/model factory/subprocess worker, `config.resolved`, viewer model loader, NAT/plugin readers, and tests that mutate `MODELS`.
- Token-policy consumers: `interactive.py` summarizer thresholds, `agents/summarization.py`, actor context/error recovery, context renderer/stats, ACP event bridge, `runtime/token_usage.py`, benchmark protocol/runner/analyzer, trace explorer, ATIF/event schemas, and debug handler.
- Message/provider behavior: cache-control injection, multimodal image/audio/video/file blocks, reasoning and provider replay metadata, provider formatter selection, and runtime restrictions.
- Observability: session setup, OpenInference hooks, journal HTTP/file exporters, viewer routes, shutdown/flush and integration reset fixtures.

### NVIDIA-compatible hermetic matrix

An in-process HTTP server runs in ordinary CI and captures exact requests produced through `AnyLLM.create_openai_compatible`. It covers `/chat/completions` and `/responses`, auth/header redaction, both output-limit wire parameters by declared profile, `extra_body`, cache key, fragmented SSE, usage-only chunks, tool/reasoning replay, structured output, 400 context errors, 401, 429, 5xx, timeout, cancellation, and connection closure. These tests establish adapter behavior without secrets; live tests establish endpoint/model compatibility.

### NVIDIA live acceptance

Credential-gated and secret-safe, using `NVIDIA_INFERENCE_API_KEY` (falling back to the existing documented synonym only in config resolution):

1. Use an explicitly configured endpoint/model/capability tuple (no runtime model discovery): internal `https://inference-api.nvidia.com/v1` with a reviewed model ID when `NVIDIA_INFERENCE_API_KEY` is present; public NIM `https://integrate.api.nvidia.com/v1` is a separate profile. A missing credential/profile is a reported **skip**, never a pass.
2. Non-streaming text: assert non-empty normalized content and no backend type in the returned object graph.
3. Async streaming: assert at least one normalized text chunk and clean close.
4. Tool calling: require a deterministic tool, validate normalized ID/name/JSON arguments, then replay a tool result.
5. Structured output: request a small Pydantic object and validate exact type/fields.
6. Context error (if safely reproducible): verify normalized exception and bounded recovery; otherwise cover by wire fixture.
7. Trace/journal: verify one NOOA `llm.call` child span and complete normalized message journal.
8. Run with secret scrubbing enabled and scan generated logs/artifacts for the key.

Live capability failures are reported per model rather than hidden by dropping parameters. The selected endpoint/model and response IDs are recorded; prompts/credentials are not.

## 12. Acceptance criteria

- `git merge-base HEAD origin/main` reflects the reviewed baseline (or the branch is deliberately rebased and the design rechecked before implementation).
- Core dependency is `any-llm-sdk==1.26.0` initially; `litellm` and `openinference-instrumentation-litellm` are gone from dependency metadata and lockfile.
- No production import/reference to LiteLLM remains.
- No AnyLLM/provider SDK type appears in public NOOA signatures, dataclass/Pydantic fields, emitted events, middleware contexts, or serialized output.
- All LLM traffic—agent, viewer, eval, and embeddings—travels through an explicit NOOA-owned boundary.
- No token-counting/estimation/calibration/control-flow behavior remains.
- Existing provider-neutral `CompletionClient`, `ResponsesClient`, `ReasoningCompletionClient`, `FakeLLMClient`, `call/acall`, tools, retries, structured outputs, and context-manager usage remain covered.
- Unit/integration suite, lint/type checks, package build, import smoke tests, and NVIDIA acceptance smoke pass.
- Worktree is clean and implementation is committed only after review findings are resolved.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| AnyLLM API moves quickly | Exact 1.26.0 pin for migration; adapter contract tests; loosen only after proven compatibility. |
| Generic OpenAI-compatible provider does not advertise Responses support | Private adapter selects the built-in OpenAI implementation for Responses while retaining NOOA endpoint identity; live-test only known-supported endpoint/model. |
| Parameter parity differs from LiteLLM | Typed common options, explicit provider options, fail loudly; fixture and NVIDIA request-body tests. |
| Tracing disappears when LiteLLM instrumentor is removed | Land NOOA-native LLM lifecycle and journal before switching the transport. |
| Tool/reasoning replay loses provider extension data | Normalize required JSON into assistant message/diagnostics and add multi-turn tests. Never retain provider objects. |
| Removing token heuristics changes truncation behavior | Character caps are deterministic; context overflow recovery is bounded and tested; no pretence that chars equal tokens. |
| Legacy model names/config break users | One-release parser/warnings and migration guide; no silent ambiguous rewrite. |
| AnyLLM unified exceptions depend on environment setup | Do not mutate the process environment; locally normalize both AnyLLM and raw SDK exceptions in the adapter. |
| Custom HTTP pooling semantics regress | Per-client construction/close tests and repeated NVIDIA calls; no global monkey patches. |
| Scope expands into benchmark/trace UI data migrations | Separate passive reported usage from forbidden counting. Preserve fields only where externally consumed, clearly renamed; remove them from control paths. |

## 14. Design review questions

1. **Compatibility window:** approve one release of deprecated `api_base`, legacy model-prefix, and common call-kwarg shims, or require a hard break now?
2. **Passive usage:** approve retaining provider-reported usage for traces/ATIF/bench reporting while forbidding all counting and control-flow use? The proposed answer is **yes**.
3. **Responses API:** retain public `ResponsesClient` using the private AnyLLM OpenAI-provider workaround for custom endpoints, or temporarily support Responses only on named AnyLLM providers? The proposed answer is **retain it** and validate NVIDIA per model.
4. **Provider coverage:** install base `any-llm-sdk` first (OpenAI/Anthropic included), adding provider extras only for documented/tested NOOA providers, rather than `[all]`? The proposed answer is **base plus explicit extras**.
5. **Raw response removal:** accept a direct removal of `raw_response` in favor of bounded diagnostics and `to_wire()`, rather than a deprecation period that would continue the leak? The proposed answer is **remove directly** because no supported public behavior should depend on backend objects.

## 15. Sources checked

- NOOA baseline source at the exact SHA above, especially `src/nooa/unifiedllm`, runtime actor, tracing, Relay middleware, viewer routes, memory embeddings, eval pipeline, tests, docs, and dependency metadata.
- Mozilla AnyLLM source and docs: <https://github.com/mozilla-ai/any-llm>, <https://docs.mozilla.ai/any-llm/>, and its custom OpenAI-compatible endpoint guide.
- PyPI metadata for `any-llm-sdk` 1.26.0: <https://pypi.org/project/any-llm-sdk/>.
- NVIDIA NIM API reference entry point: <https://docs.api.nvidia.com/nim/reference/llm-apis>. The environment returned HTTP 403 for documentation content; no model-specific capability is asserted without a live test.
