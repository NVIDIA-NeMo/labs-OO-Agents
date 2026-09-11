# Stable-prefix caching

NOOA places live context after stable instructions and conversation history.
One boundary identifies where that changing suffix begins. UnifiedLLM consumes
the boundary after projecting assistant turns into provider messages, so native
reasoning and multi-item Responses turns remain on the correct side.

This replaces positional rules such as "cache every system message and the last
tool result." Those rules can select changing context, and maintaining both
policies would make their interactions harder to test.

## Configuration

- `cache_breakpoint="auto"` is the CompletionClient default. Recognized Anthropic routes get a
  native `cache_control` breakpoint. Other routes use provider-default caching.
- `cache_breakpoint="anthropic"` explicitly selects the Anthropic Chat mapping,
  including gateway aliases that cannot be recognized automatically.
- `cache_breakpoint="openai"` opts a Responses client into explicit OpenAI
  breakpoints. Use it only on routes supporting those wire fields; it is not
  inferred from a model name.
- `cache_breakpoint=None` disables NOOA-generated cache markers, not the
  provider's implicit cache. This is the ResponsesClient default; that client
  accepts only `None` or `"openai"`, not the Anthropic Chat mapping.

Registry YAML accepts the same setting. Explicit mappings are tied to the client
model: use a new client when switching models. The automatic mapping is resolved
against the effective per-call model.

`CacheBoundary` belongs to the UnifiedLLM interface, alongside `LLMResponse`.
The cached renderer inserts it immediately before live context; the formatter
and middleware pass the same object through unchanged. Direct callers use it
in their message list too:

```python
from nooa.unifiedllm import CacheBoundary

messages = [*history, CacheBoundary(), {"role": "user", "content": live_state}]
response = await client.acall(messages)
```

Only UnifiedLLM interprets and removes the boundary before sending the request.
NeMo Relay projects it to public metadata JSON at its serialization boundary,
then restores the original object if that entry is unchanged. A raw dictionary
with `nooa_cache_boundary` is rejected with instructions to use `CacheBoundary()`;
there is only one accepted boundary type. Without a boundary, the policy marks
only leading system/developer instructions, not arbitrary history.

## Provider mapping

Anthropic marks the latest eligible content block before the boundary, never a
thinking or redacted-thinking block. OpenAI Responses marks the latest eligible
input-text block or function result and enables explicit mode. If necessary,
stable Responses instructions become an input-text block to carry that marker.
If no eligible stable block exists, OpenAI explicit mode remains enabled with
no breakpoint and logs a warning: the request does not cache anything. This can
happen with wholly dynamic input or stable history containing only unmarkable
output blocks. It deliberately avoids implicit writes beyond the chosen boundary.
Gemini receives no invented inline marker: this change uses its implicit cache,
not a separately managed explicit cached-content resource.

These mappings follow the [OpenAI prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic's content-block breakpoints](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
and [Gemini's implicit versus explicit caching](https://ai.google.dev/gemini-api/docs/caching).
OpenAI documents explicit mode and content-block breakpoints for GPT-5.6 and later;
gateway support must still be checked independently.

A boundary makes the stable prefix eligible for reuse; it does not guarantee a
hit. Provider thresholds, expiry, routing, model configuration and earlier edits
still matter. Changing effort or tools may invalidate an otherwise stable prefix.

## Migration

`cache_control_injection_points` has been removed. Constructor and per-call use
raise a message naming `cache_breakpoint` and `CacheBoundary()` as replacements.
Use `cache_breakpoint=None` instead of an empty injection list. To cache completed
history, place a boundary after that history rather than selecting a message by
role or position.

`cache_breakpoint` belongs on the client constructor. Passing it through
`call`/`acall` or `extra_body` raises a configuration error before dispatch;
the framework setting must never become a provider request field.

## Code walkthrough: what changed and why

1. `unifiedllm/cache_policy.py` owns the single policy. It consumes the boundary
   and changes only the final marker target's containers; unrelated messages
   and large strings are shared.
   `CacheBoundary` is a small, immutable UnifiedLLM input type. The cached renderer
   places it before live context using the same pass-through path as assistant
   responses. The formatter does not translate it or know what it means;
   ordinary messages carry no cache flag and are not edited.
2. Both clients apply the policy after provider projection. This keeps boundary
   placement correct when one stored assistant turn expands into several wire
   items, and keeps providers' fields out of renderers and middleware.
3. The former positional injection methods and their scenario tests are removed.
   Contract tests cover defaults, migration errors, ownership, dynamic context,
   native replay expansion and actual mocked HTTP payloads instead.
4. The opt-in live test closes SQLite, opens a new storage manager and client,
   changes trailing live state, and compares the stable HTTP prefix while
   checking native replay and reported cache tokens.

## Evidence and limits

The object-in-list stack was tested through NVIDIA Inference Hub on 2026-09-11,
with production code frozen at `e527f9ce`. All three providers have passing live
checks. SQLite events, native state and the stable HTTP prefix were equal after
reopen with a fresh client; the trailing live context changed.

| Model | Passing test revision | Resumed input tokens | Cached input tokens |
|---|---|---:|---:|
| GPT-5.6 Sol | `fef83178` | 6,120 | 6,096 |
| Claude Sonnet 5 | `fef83178` | 10,858 | 10,828 |
| Gemini 3.1 Pro Preview | `808ddbfe` | 24,667 | 20,350 |

The warm requests reported zero cache-read tokens. OpenAI's
`prompt_cache_breakpoint` and `prompt_cache_options` fields were verified on the
serialized HTTP request and accepted by this live route, not inferred from the
installed SDK schema. Support on other routes is not established. Gemini used
implicit caching; exact replay does not control how much a provider caches.

Two test assumptions were corrected during validation: compare durable public
projections rather than transient SDK response objects after SQLite reopen,
and accept either a final answer or a valid tool continuation after cache reuse.
The latter does not guarantee identical sampled output. Gemini initially passed
the state/wire checks but selected another tool call; its isolated rerun passed
all assertions after the test correction. No production changes were needed.
Including these attempts, the round used 14 requests, 140,307 input tokens and
5,856 output tokens, with retries disabled.

Offline tests:

```sh
uv run --extra nemo-relay pytest tests/unifiedllm/test_cache_policy.py tests/unifiedllm/test_explicit_cache_boundary.py
```

The live suite is opt-in and spends tokens:

```sh
NOOA_RUN_CACHE_RESUME_LIVE=1 uv run --extra nemo-relay pytest tests/integration/test_cache_resume_live.py -m integration -k reasoning_and_prompt_cache -s
```

Set `NVIDIA_INFERENCE_API_KEY` separately. Three same-provider cases use nine
requests with retries disabled, roughly 90k input tokens. Additional cases cover
cross-provider stripping and portable reasoning from Nemotron, Qwen and DeepSeek.
The suite prints usage and checks, not opaque contents.
