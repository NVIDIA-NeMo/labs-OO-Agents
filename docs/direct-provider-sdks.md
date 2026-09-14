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
Messages, with `client_type: completion`). Existing `openai/` and `anthropic/`
routing prefixes are removed before sending the model name. Other recognized
legacy prefixes require an explicit server URL; the direct path does not
discover endpoints or convert authentication schemes.

For a temporary soak run, `NOOA_LLM_TRANSPORT=direct uv run ...` selects the
direct path for clients in that process. Set it to `litellm` to compare the
legacy path, or unset it to use each registry entry. This is a client setting,
not a per-call setting. There is no automatic retry through the other transport.

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
Likewise, cache markers remain opt-in through the existing `cache_breakpoint`
setting. An endpoint accepting a request does not establish that it used the
reasoning setting or served a cache hit; inspect reported usage during soaking.

## Tracing and token estimates

Both transports use one UnifiedLLM tracing boundary. It records the sanitized
outbound HTTP body and the public response, plus reported usage. Credentials
and native reasoning state are redacted. Journal errors are logged without
failing or repeating the provider call. Tracing no longer instruments unrelated
raw `litellm.completion` calls made outside UnifiedLLM.

The LiteLLM tracing instrumentor, its monkeypatch, token calibration and
`UnifiedLLM.count_tokens` are removed. No tokenizer replaces them. Direct
clients use the registry's context window; the existing summarizer character
estimate and actor usage-based sizing remain. Unknown costs are not calculated
from a model table on the direct path.

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
explicit `max_tokens`; only leading system messages are accepted. Unknown
provider fields are sent through the SDK's `extra_body`, not translated or
silently dropped.

No live soak result is claimed by these offline tests. Live release checks
must use configured `release-gate-*` aliases; endpoints, credentials and
route-specific evidence belong in the private release configuration.

SDK references: [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [Anthropic Messages](https://platform.claude.com/docs/en/api/python/messages/create).
