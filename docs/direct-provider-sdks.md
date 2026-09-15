# Direct provider SDKs

NOOA can send requests through the official OpenAI and Anthropic SDKs without
using LiteLLM. LiteLLM remains the default while the direct path is tested in
real sessions. A direct request that fails does not fall back to LiteLLM.

## Configure a model

Add `transport: direct` to a model entry. Keep its existing model name and
credentials. `api_style` selects the request format, not the company that
trained the model:

```yaml
models:
  my-model:
    model_name: openai/my-model
    client_type: completion
    api_style: chat
    transport: direct
    api_base: https://models.example/v1
    api_key_env: MODEL_API_KEY
    context_window: 32768
    max_tokens: 2048
```

The formats are `chat` (OpenAI-compatible Chat Completions), `responses`
(OpenAI Responses, with `client_type: responses`), and `anthropic` (Anthropic
Messages, with `client_type: completion`). If `api_style` is omitted, the
`anthropic/` routing prefix selects Messages; otherwise the client class selects
Chat or Responses. This compatibility default does not infer model capabilities.

The routing prefixes `openai/`, `anthropic/`, `azure/`, `deepseek/`,
`nvidia_nim/`, `openrouter/`, `hosted_vllm/`, `together_ai/`, `xai/`, and
`gemini/` are removed before sending the model name. All except `openai/` and
`anthropic/` require an explicit server URL. Other names, including names with
slashes, are sent verbatim. They do not select a provider: without `api_base`,
the selected SDK uses its default endpoint. Set the URL explicitly for custom
servers; native Bedrock and Vertex routing prefixes are not supported.

For a temporary soak run, `NOOA_LLM_TRANSPORT=direct uv run ...` selects the
direct path for clients in that process. Set it to `litellm` to compare the
legacy path, or unset it to use each registry entry. This is a client setting,
not a per-call setting. The environment override also takes precedence over an
explicit constructor `transport` argument. There is no automatic retry through
the other transport.

## What stays shared

1. UnifiedLLM applies the entry's reasoning-level settings and prepares the
   messages, tools, output schema and cache boundary.
2. The selected transport sends the request. Both use the client's HTTP pool
   and NOOA retry policy; the direct SDKs' own retries are disabled.
3. The same capture code returns an `LLMResponse`. History, storage, agents,
   relay and summarization do not need to know which transport was used.

Replay compatibility still uses the API style, provider and exact model name
sent to the server. Switching transports alone does not invalidate stored
reasoning. `replay_vendor` can declare the vendor of native state explicitly.
Readable `reasoning_content` is retained on all resolved Chat routes, including
OpenRouter, NIM and vLLM. Unknown opaque formats are dropped with a warning.
Editing a stored turn or switching to an incompatible model still drops its
private fields and retains readable text.

Reasoning settings come from the registry, not a new model-name mapping.
Cache configuration follows the shared clients' `cache_breakpoint` policy.
An endpoint accepting a request does not establish that it used the
reasoning setting or served a cache hit; inspect reported usage during soaking.

## Tracing and token estimates

Both transports use one UnifiedLLM tracing boundary. It records the sanitized
outbound HTTP body and the public response, plus reported usage. Credentials
and native reasoning state are redacted. Journal errors are logged without
failing or repeating the provider call. Tracing no longer instruments unrelated
raw `litellm.completion` calls made outside UnifiedLLM.

Legacy model/URL/key overrides construct temporary bound HTTP wrappers and close
them after dispatch, keeping the request hook attached without reusing the wrong
credentials. Handler-backed vLLM, DeepSeek and Gemini routes use the handler
wrapper, and Gemini `contents` are included in journal input. Providers that do
not accept a supplied HTTP client can still bypass the hook; complete wire capture
is not promised for arbitrary third-party LiteLLM adapters.

The LiteLLM tracing instrumentor, its monkeypatch, token calibration and
`UnifiedLLM.count_tokens` are removed. No tokenizer replaces them. Direct
clients use the registry's context window; the existing summarizer character
estimate and actor usage-based sizing remain. Unknown costs are not calculated
from a model table on the direct path.
LiteLLM may estimate Anthropic's reasoning-token breakdown; the direct path
does not. When the server reports only total output tokens, that total is
preserved and the separate reasoning count remains unknown (represented as zero
by the existing usage type).

### Migration notes (including the default LiteLLM path)

This is an opt-in transport, but not a promise that every default-path observable
is unchanged. The following shared changes apply to both transports:

- `UnifiedLLM.count_tokens` and `TokenCalibration` are removed. Use reported
  `response.usage` for actual usage. Applications that need preflight estimates
  must supply their own counter; the summarizer uses its existing character
  heuristic when no application counter is available. This is not a tokenizer
  accuracy guarantee.
- Readable `reasoning_content` on resolved compatible Chat routes is now replayed
  in that field, not concatenated into the assistant's visible answer. This changes
  second-turn bodies for routes outside the old provider allow-list. Opaque state
  still requires a recognized dialect and a compatible scope.
- Raw `reasoning_content` is not converted into spoken text by the direct native
  Anthropic adapter. Native replay uses signed thinking blocks; portable text
  produced by the shared cross-model projection is unchanged.
- LiteLLM globals and its cache-control preservation patch initialize when a
  legacy client is constructed. Unrelated raw `litellm.*` calls made earlier see
  LiteLLM's own defaults and are not traced by NOOA.
- Spans are named `llm.call`, one per logical call including retries. `input.value`
  is scrubbed wire JSON; `output.value` is JSON for the public response. Model and
  invocation attributes describe the actual request, not pre-translation kwargs.
  Journal input follows wire messages; Anthropic input usage includes cache reads
  and writes. Consumers relying on old span names, message indices, or cost-breakdown
  attributes need updating. Unknown cost remains zero in the current usage type.

`api_style` remains supported so existing registry/Connect entries do not break
and native format selection need not depend on a model-name heuristic.
`replay_vendor` is validated consistently before either transport is constructed.
The prefix rules above remain unchanged to preserve saved replay scopes; an
OpenRouter model whose literal name starts with a recognized prefix must retain
the outer routing prefix (for example `openai/deepseek/deepseek-chat`).

Direct HTTP clients preserve redirect handling and environment-based client
certificates (`SSL_CERTIFICATE`) and verification (`SSL_VERIFY`, `SSL_CERT_FILE`,
`SSL_CERT_DIR`). They do not read mutable LiteLLM Python globals. Provider SDK
User-Agent headers may differ; HTTP-body equality does not assert header equality.

## Testing and limits

The mocked HTTP matrix runs both transports through sync and async calls,
SQLite reopen, renderer and relay round trips, edits and model switches. It
asserts on outbound fields, including signed thinking, encrypted reasoning,
inline signatures and readable reasoning. Separate tests cover no-fallback
errors, SDK retries, registry selection, lazy imports, structured output and
trace/journal parity.

This first version does not implement streaming, native Google APIs, Azure
deployment authentication, Bedrock or Vertex SDKs. Use an OpenAI-compatible
endpoint or keep LiteLLM for those routes. Anthropic direct calls require an
positive `max_tokens`; only leading system messages are accepted. The Anthropic
base URL may end in `/v1` (removed before SDK dispatch), but must not include
`/messages`. This suffix handling is case-sensitive.

Direct calls reject `stream=True`, `num_retries` other than `0`, and the
LiteLLM-specific `client` and `custom_llm_provider` overrides. They discard
`drop_params`, `allowed_openai_params`, an empty `additional_drop_params`, and
`context_window`; non-empty `additional_drop_params` is rejected. Anthropic
also omits `prompt_cache_key`, which its API does not accept. Other provider
fields go through the SDK's `extra_body` without translation. Image `detail`
has no Anthropic equivalent and is omitted during image conversion.

OpenAI is pinned to 2.44.0 because structured-output conversion uses its private
schema helper. An SDK upgrade must rerun the structured-output and wire tests;
the pin avoids silently depending on a changing private interface.

No live soak result is claimed by these offline tests. Live release checks
must use configured `release-gate-*` aliases; endpoints, credentials and
route-specific evidence belong in the private release configuration.

SDK references: [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [Anthropic Messages](https://platform.claude.com/docs/en/api/python/messages/create).
