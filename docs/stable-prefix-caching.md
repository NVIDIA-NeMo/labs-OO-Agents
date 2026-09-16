# Stable-prefix caching

NOOA places live context after stable instructions and conversation history.
One boundary identifies where that changing suffix begins. UnifiedLLM consumes
the boundary after projecting assistant turns into provider messages, so native
reasoning and multi-item Responses turns remain on the correct side.

This replaces positional rules such as "cache every system message and the last
tool result." Those rules can select changing context, and maintaining both
policies would make their interactions harder to test.

## Configuration

- `cache_breakpoint="auto"` is the default for both clients; registry entries need
  no cache setting. Completion clients mark recognized Anthropic routes with a
  native `cache_control` breakpoint; other Chat routes use provider-default caching.
  Responses clients enable explicit caching when a `CacheBoundary()` and eligible
  stable input are present. Without them, they send no extra cache fields.
- `cache_breakpoint="anthropic"` explicitly selects the Anthropic Chat mapping,
  including gateway aliases that cannot be recognized automatically.
- `cache_breakpoint="openai"` forces the Responses explicit-cache policy, including
  its leading-instructions fallback without a boundary. It is an override, not
  required for normal use.
- `cache_breakpoint=None` disables NOOA-generated cache markers, not the
  provider's implicit cache. Use this opt-out for a Responses endpoint that does
  not support explicit-cache fields. Responses clients do not accept the
  Anthropic Chat mapping.

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
there is only one accepted boundary type. Without a boundary, Completion and
forced explicit Responses policies mark only leading system/developer instructions,
not arbitrary history. Automatic Responses adds no cache fields.

## Provider mapping

Anthropic marks the latest eligible content block before the boundary, never a
thinking or redacted-thinking block. Images and documents are eligible, including
the `image_url` and `file` forms that LiteLLM converts for Anthropic. OpenAI
Responses marks the latest eligible input text, image, file or function result
and enables explicit mode. Stable media must be inside the breakpoint rather
than left after a preceding text block. If necessary,
stable Responses instructions become an input-text block to carry that marker.
If no eligible stable block exists, forced OpenAI explicit mode remains enabled with
no breakpoint and logs a warning: the request does not cache anything. This can
happen with wholly dynamic input or stable history containing only unmarkable
output blocks. It deliberately avoids implicit writes beyond the chosen boundary.
Automatic Responses instead leaves provider-default caching unchanged in this case.
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

### Offline request contracts

`tests/unifiedllm/test_history_wire_contract.py` runs on ordinary PR CI without
credentials. Synthetic provider replies pass through the real clients and SDKs;
an HTTP mock captures the outgoing requests and unexpected httpx requests fail.
It covers Responses encrypted reasoning, Anthropic signed thinking, Gemini tool
signatures on OpenAI-compatible routes, and Chat `reasoning_content`.

The matrix compares complete requests before and after rendering, SQLite
close/reopen, relay JSON reconciliation, and their combination, in both sync
and async calls. Only the trailing dynamic-context text is excluded. For
Anthropic that text can share a user message with tool results, which remain
inside the comparison. Tool schemas, instructions, argument strings, reasoning
fields and cache markers are checked, not just successful responses. Deliberate
edits and model/provider switches must instead discard incompatible native state
and retain readable reasoning. Negative controls corrupt captured reasoning,
arguments, instructions and tools and require the assertions to fail.

This follows [Pydantic AI's history-round-trip testing approach](https://github.com/pydantic/pydantic-ai/blob/8762545fa01fabfee6904e6166dd18aac759a005/tests/test_cache_prefix_stability.py), with assertions
on the requests generated by the current code rather than saved HTTP recordings.
It adds no runtime mechanism or test dependency. Run it with:

```bash
uv run pytest tests/unifiedllm/test_history_wire_contract.py
```

These tests prove request preservation, not provider cache hits. The bounded
live release checks remain necessary to detect provider and gateway changes.

### Provider validation

Multimodal placement has regression tests through both serialized HTTP paths.
The opt-in resume test now ends the OpenAI and Anthropic stable prefixes with a
fixed image, and checks that the image itself carries the breakpoint. This adds
no calls to the existing three-call scenario. Offline preservation tests do not
establish cache support on a particular deployment.

```sh
uv run --extra nemo-relay pytest tests/unifiedllm/test_cache_policy.py tests/unifiedllm/test_explicit_cache_boundary.py
```

Live tests are opt-in and spend tokens. Configure their registry aliases for
your deployment. Keep endpoint details, credentials guidance and measured
results alongside that private configuration. Exact replay does not control
how much a provider caches; compare reported usage as well as outgoing requests.
