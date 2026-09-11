# Stable-prefix caching

NOOA places live context after stable instructions and conversation history.
One boundary identifies where that changing suffix begins. UnifiedLLM consumes
the boundary after projecting assistant turns into provider messages, so native
reasoning and multi-item Responses turns remain on the correct side.

This replaces positional rules such as "cache every system message and the last
tool result." Those rules can select changing context, and maintaining both
policies would make their interactions harder to test.

## Configuration

- `cache_breakpoint="auto"` is the default. Recognized Anthropic routes get a
  native `cache_control` breakpoint. Other routes use provider-default caching.
- `cache_breakpoint="anthropic"` explicitly selects the Anthropic Chat mapping,
  including gateway aliases that cannot be recognized automatically.
- `cache_breakpoint="openai"` opts a Responses client into explicit OpenAI
  breakpoints. Use it only on routes supporting those wire fields; it is not
  inferred from a model name.
- `cache_breakpoint=None` disables NOOA-generated cache markers, not the
  provider's implicit cache.

Registry YAML accepts the same setting. Explicit mappings are tied to the client
model: use a new client when switching models. The automatic mapping is resolved
against the effective per-call model.

The cached renderer inserts `{"nooa_cache_boundary": true}` immediately before
live context. Direct UnifiedLLM callers may insert the same dictionary, or add
the key to their first dynamic message. Without a marker, the policy marks only
leading system/developer instructions. It never assumes arbitrary history is
stable. The metadata key does not reach the provider.

## Provider mapping

Anthropic marks the latest eligible content block before the boundary, never a
thinking or redacted-thinking block. OpenAI Responses marks the latest eligible
input-text block or function result and enables explicit mode. If necessary,
stable Responses instructions become an input-text block to carry that marker.
Gemini receives no invented inline marker: this change uses its implicit cache,
not a separately managed explicit cached-content resource.

A boundary makes the stable prefix eligible for reuse; it does not guarantee a
hit. Provider thresholds, expiry, routing, model configuration and earlier edits
still matter. Changing effort or tools may invalidate an otherwise stable prefix.

## Migration

`cache_control_injection_points` has been removed. Constructor and per-call use
raise a message naming `cache_breakpoint` and `nooa_cache_boundary` as replacements.
Use `cache_breakpoint=None` instead of an empty injection list. To cache completed
history, place a boundary after that history rather than selecting a message by
role or position.

## Code walkthrough: what changed and why

1. `unifiedllm/cache_policy.py` owns the single policy. It consumes the boundary
   and changes only the final marker target's containers; unrelated messages
   and large strings are shared.
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

The validated prototype used NVIDIA Inference Hub on 2026-09-11. After SQLite
resume and a changed trailing context block, reported cache hits were
6,136/6,160 input tokens for GPT-5.6 Sol, 10,775/10,805 for Claude Sonnet 5, and
24,491/24,677 for Gemini 3.1 Pro Preview. Gemini's warm request already had cache
hits, so this is not a controlled cold-cache comparison.

OpenAI's `prompt_cache_breakpoint` and `prompt_cache_options` fields were verified
on that live route, not inferred from the installed SDK schema. Support on
other routes is not established. These are prototype results; a final extracted
branch run must be recorded separately.

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
