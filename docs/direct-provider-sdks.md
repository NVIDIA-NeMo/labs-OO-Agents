# Opt-in direct provider SDKs

The default transport remains **LiteLLM**. Use the strict boolean constructor
setting `direct=True` to send through the official OpenAI or Anthropic SDK.
There is no environment transport selector. Saved Connect `transport` or
`api_style` metadata does not enable direct mode or select its protocol.

## Current-main compatibility

The opt-in SDK path retains the latest main fixes: session-affinity headers
(#411), HTTP 408 retries (#413), and stable cell-context imports (#415). No
model, transport, or strategy defaults are changed by opting in to this path.

- Chat, Responses and Messages, both sync and async, mirror a nonempty string
  `prompt_cache_key` into `x-session-affinity` before SDK dispatch. Without a
  usable key no affinity header is generated. An explicit `extra_headers`
  affinity value wins case-insensitively; unrelated headers are preserved and
  caller mappings are not mutated. Native Messages removes the unsupported
  JSON `prompt_cache_key` but keeps the affinity header. Headers remain HTTP
  headers, not JSON fields, and retries retain the same affinity value.
- HTTP 408 uses NOOA's ordinary retry budget/backoff, not the extra rate-limit
  budget. SDK-internal retries remain disabled. Explicit
  `retryable_status_codes` is authoritative: include 408 to retry it or omit
  it to fail immediately. Mocked HTTP tests cover success, exclusion and
  exhaustion for every sync/async SDK protocol without opening network sockets.
- Stable cell-context rendering continues to use bound objects even when a
  child re-imports or removes their module, preserving the stable prompt prefix.
  This is a shared strategy fix, not a direct-transport-specific renderer.

## Constructors

```python
from nooa.unifiedllm import CompletionClient, ResponsesClient

# OpenAI Chat Completions
chat = CompletionClient("openai/gpt-4o-mini", direct=True, max_tokens=1024)

# OpenAI Responses
responses = ResponsesClient("openai/gpt-5", direct=True, max_tokens=4096)

# Native Anthropic Messages (requires an explicit positive reply limit)
messages = CompletionClient(
    "anthropic/claude-sonnet-4-5", direct=True, max_tokens=4096,
)

# Explicit OpenAI-compatible endpoint; the wire model can contain slashes.
gateway = CompletionClient(
    "openai/my-org/my-model", direct=True,
    base_url="https://gateway.example/v1", api_key="your-key", max_tokens=1024,
)
```

Use `with client:` for synchronous calls or `async with client:` for asynchronous
calls. `close()` releases the owned synchronous pool; `await aclose()` releases
both pools. An asynchronous example:

```python
async with CompletionClient(
    "anthropic/claude-sonnet-4-5", direct=True, max_tokens=4096,
) as client:
    reply = await client.acall([{"role": "user", "content": "Hello"}])
    print(reply.content)
```

Examples make real requests if executed; the test suite uses synthetic SDK HTTP
responses and does not require provider credentials.

## Registry

```yaml
models:
  direct-chat:
    model_name: openai/my-org/my-model
    direct: true
    api_base: https://gateway.example/v1
    api_key_env: GATEWAY_API_KEY
    max_tokens: 1024
  direct-messages:
    model_name: anthropic/claude-sonnet-4-5
    direct: true
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4096
  direct-responses:
    model_name: openai/gpt-5
    client_type: responses
    direct: true
    max_tokens: 4096
```

`get_llm_client("direct-chat")` uses the registry setting.
`get_llm_client("direct-chat", direct=False)` explicitly restores LiteLLM;
`get_llm_client("another-alias", direct=True)` explicitly enables the SDK path.
Absent `direct`, existing aliases retain LiteLLM. Strings such as `"true"`,
integers, and null are not booleans and are rejected. `direct` is constructor-only:
it must not appear in `call`/`acall` kwargs, reasoning-level settings or
`extra_body`, and is never serialized to the provider.

## Routing and request behavior

- `anthropic/<wire-model>` on `CompletionClient` selects native Messages.
  Other Completion routes use OpenAI Chat. A name containing `anthropic` behind
  `openai/` remains Chat. `ResponsesClient` selects OpenAI Responses; it does not
  support a native `anthropic/` route.
- Recognized prefixes are stripped once: `openai`, `anthropic`, `deepseek`,
  `nvidia_nim`, `openrouter`, `hosted_vllm`, `together_ai`, `xai`, and `gemini`.
  All except `openai` and `anthropic` require an explicit compatible endpoint
  (or the corresponding SDK base-URL environment variable). Other slash-containing
  wire model IDs are sent intact, with an endpoint required. Prefer
  `openai/<wire-model>` for arbitrary gateways.
- Native Azure, Bedrock, Vertex AI and SageMaker routing is unsupported. Use
  LiteLLM for those native protocols, or an explicit OpenAI-compatible gateway.
  No errors fall back to LiteLLM.
- Explicit `base_url` wins over `api_base`, including per-call overrides.
  Without explicit credentials/URLs the SDK's `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` and `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` defaults apply.
  LiteLLM global endpoint settings and `OPENAI_API_BASE` do not control SDK calls.
- Each client owns its HTTP pools and `HttpConfig`; per-request SDK wrappers
  borrow them so URL/key overrides cannot reuse stale credentials.
  **Direct requests do not follow redirects**, avoiding cross-origin leakage
  of Anthropic's `x-api-key` and conversation bodies. Supply the final endpoint.
- SDK retries are disabled (`max_retries=0`); the existing NOOA `RetryConfig`
  owns retries, including rate limits, timeouts, transient errors and optional
  empty-content retries (including native thinking-only Messages replies).
  Native direct Messages adds HTTP 529 overload to the default retryable status
  set, including when other retry settings are customized. An explicitly supplied
  `retryable_status_codes` set is authoritative: omit 529 to opt out. This never
  changes LiteLLM's default policy or a caller's frozen `RetryConfig`. Cancellation propagates normally. Streaming is not
  supported in direct mode (`stream=False` only).
- Provider extensions are serialized through SDK `extra_body` and appear at
  the top level of the wire JSON. NOOA rejects attempts to overwrite managed
  conversation, model, schema, tool, stream, client or routing controls. Use
  top-level standard fields; conflicting translated aliases are rejected.
  LiteLLM drop/allow-list flags do not filter direct fields, and nonempty
  `additional_drop_params` is rejected. Use `retry_config`, not SDK retry kwargs.
- Native Messages translates function tools declared through `Tool` or constructor
  `tools` schemas, preserving boolean `function.strict` (true **and** false) and
  top-level `cache_control` with optional `ttl="5m"` / `"1h"`. Unknown tool/function
  fields and malformed nested option shapes are rejected rather than discarded.
  The `call`/`acall` `tools` argument remains the existing list of callable `Tool`
  objects, not a native provider-tool input API. Server-managed native tool types
  are not supported by this translator.
- Responses structured `output_model` merges its strict schema into `text.format`,
  retaining other `text` options such as `verbosity`. A conflicting format is an
  error, not silently overwritten. Messages likewise rejects conflicts between
  `output_config.format` and the managed structured response format.
- Native OpenAI Chat translates the shared `max_tokens` budget to
  `max_completion_tokens`; compatible gateways keep `max_tokens`. Responses uses
  `max_output_tokens`. These decisions follow the actual endpoint, not a model
  name heuristic. Native Messages requires a positive integer `max_tokens`.

## History, cache and response contracts

Direct dispatch consumes the **current prepared request**, after canonical
projection, typed `CacheBoundary` consumption and cache policy. It does not
restore the old PR's projection or cache policy. `LLMResponse`, tools, structured
outputs, finish reasons, token counting/calibration and normalized usage retain
their existing public interfaces.

- Auto Anthropic cache mapping follows native Messages, not an Anthropic-looking
  gateway model ID. Gateways that accept Anthropic Chat cache blocks may declare
  `cache_breakpoint="anthropic"` explicitly. Marked nested tool results remain
  nested, and signed thinking blocks are never rewritten as cacheable text.
- Late dynamic system/developer updates become user text blocks **in place** in
  the volatile Messages suffix. They are not rejected or hoisted into leading
  `system` instructions. Adjacent roles are coalesced without moving their blocks.
- Responses retains the latest **80 eligible** stable checkpoints and current
  stable content-block wrapping. `prompt_cache_options` is serialized correctly;
  this preserves eligible prefixes but does not promise a provider cache hit.
- Replay scope follows direct wire model, protocol and provider prefix, independent
  of URL/key rotation. Signed Anthropic thinking, encrypted OpenAI Responses
  reasoning and compatible Chat reasoning/signatures use the current immutable
  part adapters. **`gemini/` is an explicit declaration of the Gemini-compatible
  Chat tool-signature protocol**, not automatic gateway detection or native
  Gemini/Vertex SDK support. Only this prefix normalizes tool signatures from/to
  `extra_content.google.thought_signature` without changing public IDs. Use
  `gemini/<wire-model>` with an explicit trusted compatible endpoint for custom
  direct providers implementing that extension. `openai/<wire-model>` does not
  capture/replay this Google extension, even if its model name contains `gemini`.
  Direct non-Gemini routes likewise do not retain signatures spelled as
  `provider_specific_fields.thought_signature` or inline `__thought__` call-ID
  suffixes (the suffix is removed from public IDs). Declare `gemini/` for those
  tool-signature variants too; the default LiteLLM adapters remain unchanged.
  In direct mode raw dictionaries with Google signatures are not a canonical replay API.
  URL/key changes intentionally do not invalidate scope: callers rotating a
  `gemini/` endpoint must ensure the destination implements the same protocol
  and is trusted with retained opaque state. Provider/model changes demote it.
  Summary-only Responses reasoning behind gateways stays native there; switching
  to a native OpenAI endpoint demotes it to readable text, as on the default path.
- **Native Anthropic reply fidelity is narrower than native block fidelity.**
  Replies normalize through the existing grouped Chat adapter: thinking blocks,
  concatenated text, then tool calls. Signed/redacted thinking dictionaries and
  tool IDs/names/JSON input values survive durable replay, but original cross-kind
  block order, individual text block boundaries, text citations/other metadata,
  and tool-use block extensions do not. Text concatenation adds no separators.
  Prepared-request equality across rendering/storage is not a claim that these
  original response blocks survive. Caller-supplied supported content blocks and
  their cache markers are preserved on input; NOOA chooses cache markers after
  projection. This does not guarantee response-block cache checkpoint fidelity
  or a cache hit. Full native capture/projection is intentionally not part of
  this selective transport addition. Unsupported response block kinds fail
  explicitly, rather than silently disappearing.
- Unknown Anthropic stop reasons requiring server-managed continuation (for
  example `pause_turn`) raise `UnsupportedStopReasonError` from
  `nooa.unifiedllm.errors`; they are not silently treated as successful stops.

## Observability limitations

This is a selective transport addition, **not a tracing rewrite**. Default
LiteLLM tracing/journaling remains unchanged. Direct SDK requests bypass LiteLLM
collectors and its instrumentation callbacks, so provider LLM spans and
LiteLLM request/response journals are not emitted for this opt-in path. Existing
outer agent/method tracing, call tracking, usage/calibration and applicable
httpx debug logging remain available. Direct calls do not estimate provider cost;
`usage.cost_usd` retains the existing `0.0` unavailable-cost default (not a
claim of a free call). No removed universal tracing hook is imported.

SDK serializers were verified against the locked OpenAI and Anthropic versions
with mocked HTTP; live provider acceptance, provider-specific extensions and
cache hit rates still require endpoint-specific validation. Native providers
outside the routes above, streaming, and server-managed continuations remain
out of scope.
