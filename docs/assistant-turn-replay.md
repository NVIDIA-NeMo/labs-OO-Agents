# Assistant turns, persistence, and reasoning replay

NOOA stores each model response as one `LLMResponse` event containing ordered
text, tool-call, and reasoning parts. UnifiedLLM interprets provider extensions;
the renderer and middleware pass the response object alongside ordinary message
dictionaries. Public read-only mapping access keeps consumers independent of
provider formats. JSON-only integrations project explicitly at their boundary.

## Why ordered parts

Flattening a response into independent text and tool-call lists loses the order
and grouping that the provider returned. Reconstructing it later requires extra
bookkeeping. Signed thinking, encrypted reasoning, and thought signatures also
must not be attached to edited public content or sent to an incompatible model.

The ordered response is the stored record. Wire messages are generated at
dispatch, using native extensions only for a response object whose destination
scope matches. Replacing the response with a dictionary discards native state.
This preserves supported provider message order,
boundaries, and fields rather than reconstructing them from separate carriers.
Stable replay is necessary for prompt-cache reuse; it does not guarantee a cache
hit or select a provider cache policy.

Compared with the superseded replay implementation, this removes order ledgers,
batch identities, content fingerprints, private dict-subclass carriers, and relay
lookup machinery. It retains provider validation and final wire adapters. The
relay has one public-content equality check to recover unchanged objects after
its JSON round trip; that cost is local to the integration, not every dispatch.

## Caller-facing contract

- `LLMResponse.parts` is an immutable tuple. Each part is text, a tool call, or
  reasoning; each may have an opaque `native` extension read only by UnifiedLLM.
- `.content`, `.tool_calls`, and `.reasoning` are derived public views. They are
  not a second stored copy of the response. `.usage` stores normalized input,
  output, cache-read, cache-write, reasoning-token, and estimated-cost fields.
- `call` and `acall` accept ordinary message dictionaries and prior
  `LLMResponse` objects in the same list. There is no ID key or separate lookup.
- Responses expose public read-only mapping access: `reply["content"]`,
  `reply.get("role")`, and `dict(reply)`. Native state is never part of that view.
  Nested projected containers are detached; changing them alone does not edit
  the response.
- Middleware can replace any list element with an ordinary dictionary. That
  replacement is sent without native state. Assignment into the
  response raises a helpful error explaining this edit contract.
- In Chat input, an explicit `reasoning_content` field on a raw dictionary is
  preserved, not automatically folded into `content`. Portable reasoning
  projection applies to retained `LLMResponse` objects. API-shape translation
  and rejection of raw opaque fields still apply at the transport boundary.
- The renderer retains the original object when text and calls are unchanged;
  truncation or omitted calls produce a portable dictionary instead.
- The relay receives only public JSON. Unchanged entries at the same index regain
  their original response objects on return. Insertions/deletions conservatively
  demote shifted entries; they never associate native state by fuzzy matching.
- Repeated mapping reads share one cached immutable public view. Returned
  dictionaries get their own mutable containers; edits cannot alter that cache.
  Copies with changed parts start with an empty cache.
- The standalone Anthropic formatter exports portable Anthropic-shaped JSON
  (`tool_use`/`tool_result`), without framework cache markers or native state.
  UnifiedLLM clients handle native replay using the standard message history.

For direct callers:

```python
reply = await client.acall(messages)
messages.append(reply)
# Append matching tool results if reply contains calls, then continue:
next_reply = await client.acall(messages)

# To edit a historical response, replace that element:
messages[index] = {**dict(messages[index]), "content": "edited text"}
```

Use `replace_parts()` or `replace_text()` to construct edited responses. These
strip native extensions and replay scope. The `model_copy(update=...)` guard
enforces the same rule for public-part edits; metadata-only copies can share
parts. Pydantic's frozen fields alone would not protect that copy path.

## Capture and compatibility

The scope consists of API style, LiteLLM-resolved provider, and exact model ID.
Transport URLs and credentials are not part of this identity. The gate therefore
prevents cross-model/provider/API-style replay, but is not an issuer- or
account-isolation policy. A successful gateway test does not establish that
every gateway or account accepts another issuer's opaque data.

OpenAI encrypted reasoning, Anthropic signed/redacted thinking, and Gemini
signatures stay with their owning parts. Plain reasoning is retained as readable
text and can be sent to another model without the source's native extensions.
When plain text arrived in Chat's `reasoning_content`, a small origin marker
restores that field for compatible replay. This matters for DeepSeek thinking
tool turns: moving reasoning into answer content is not equivalent. The marker
contains no copy of the reasoning text. Changed models or edited turns still
receive the text as ordinary assistant content, without the source field.
Unknown capture routes warn and keep portable text while dropping opaque state.
The `__thought__` tool-call ID separator is reserved for inline signatures on
every route, including OpenAI-compatible gateways with no separate signature
field. Capture removes that suffix from public IDs before applying the replay
gate. Raw input dictionaries containing it are rejected rather than exposed as
public history. LiteLLM can still strip restored inline signatures on routes it
does not recognize as Gemini; the HTTP tests use a Gemini-named gateway route,
while capture/privacy tests also cover generic and unknown routes.
Reasoning-only replies remain in history, including length-stopped replies;
an empty visible answer does not mean the turn is empty.
Nonempty provider fields in input dictionaries raise with instructions to pass
an `LLMResponse` instead; both clients enforce that rule. Null or empty optional
fields from SDK message dumps carry no state and are accepted. Malformed recognized
state raises; it is not silently treated as a successful
capture. Empty public tool IDs are accepted where no retained native state needs
binding; nonempty duplicate IDs and ambiguous native bindings raise.

Valid Responses outputs outside our replay format, such as built-in search
actions and refusal blocks, do not discard an otherwise usable answer. Capture
keeps readable text (including the refusal), warns about the unsupported shape,
and removes native state from the whole turn. The original SDK response remains
available on the live result; it is not saved in the session. Malformed recognized
fields still raise, as does unsupported output with no readable outcome.

Provider strings remain immutable in storage. Container detachment happens at
capture and final projection, without repeatedly copying large immutable string
leaves. Live SDK responses and parsed Python results are excluded from archives.

### Open-model tool replay check

On 2026-09-14, two capped calls each through NVIDIA Inference Hub passed for
DeepSeek V4 Pro, Kimi K3, GLM 5.3 and Qwen 3.5 397B. Each produced nonempty
`reasoning_content` and a tool call. After closing/reopening SQLite, a new client
sent exactly that text in the next request's `reasoning_content`; all four
continuations completed. Both calls included trailing live-context messages.
Total: 2,759 input tokens (including cached input), 797 output tokens, no retries.
These tests establish field preservation and successful continuation, not whether
each route rejects an omitted field or guarantees a cache hit. Run with
`NOOA_RUN_OPEN_MODEL_REPLAY=1` and `NVIDIA_INFERENCE_API_KEY`:

```bash
uv run pytest -m integration -s tests/integration/test_open_model_tool_reasoning_live.py
```

Set `NOOA_TEST_OMITTED_REASONING=1` to add one continuation per model with
`reasoning_content` removed from the final HTTP body, after LiteLLM serialization.
The test checks that this is the only changed field and reports the HTTP outcome.
A successful omitted-field request does not prove that reasoning was retained or
used: gateways differ in how they enforce the field. Rate limits, authentication
failures and transport failures are inconclusive, not evidence of a required field.

## Archives and collapse

New archives preserve ordered parts and native extensions through SQLite and
generic snapshots. Old flat `LLMResponse`/legacy `LLMOutput` records load as
portable reasoning, text, and calls. Their old opaque sidecar is intentionally
discarded: ordering cannot be recovered reliably. That is a one-time loss of
opaque replay on migration, not a promise to replay old private blobs.

Collapse still archives the requested tag range. A `ToolCallEvent` keeps its
nested `ToolResult`, but the originating `LLMResponse` is a separate event.
Splitting that linked batch is not rejected or automatically expanded today.
The formatter omits incomplete batches: missing results warn; results whose
source turn is absent are omitted. Independent user-role `PythonOutput` events
are unaffected by this pairing rule. Omitting a split batch from a request does
not archive both halves together. Changing collapse's range selection remains
a separate decision.

## Observability

Readable reasoning remains available to tracing. Opaque native fields are
excluded from the public journal/search projections and scrubbed from trajectory
export. SQLite is a replay archive; tracing is not. Usage fields can be zero when
a route does not expose a measurement: zero reasoning tokens does not prove the
absence of reasoning, and a zero cost estimate does not prove free inference.

## Code walkthrough: what changed and why

1. `llm_types.py`: defines ordered immutable parts and public views so a response
   is stored once, with its original relationships intact.
2. `unifiedllm/response_parts.py` and `chat_parts.py`: capture native extensions
   and project them at the provider boundary. Provider-specific fields stay here
   rather than spreading into strategy and UI code.
3. `unifiedllm/replay_state.py`: resolves scope and validates provider variations.
   It rejects opaque fields supplied in ordinary wire dictionaries.
4. `unifiedllm/unifiedllm.py`: projects response objects for the effective model.
   All clients, including the reasoning wrapper and fake, accept the same history
   shape. Runtime lookup building and per-dispatch public equality are deleted.
5. Renderer/formatter/runtime: the formatter recognizes the public LLMResponse
   event type and preserves its response object, with existing public text/tool
   views for display and budgeting. EventBase owns generic event identity and
   search/emptiness behavior, not assistant-specific replay hooks. Only the
   UnifiedLLM adapters interpret native provider state.
   Responses lifts only leading system messages into `instructions`: moving a
   later system message there would reorder the conversation. This needs no
   cache marker. Cache-boundary metadata and policy belong to the follow-up PR.
6. Storage and event hooks: serialization, searchable fields, and empty-event
   handling are owned by the event type. Generic consumers do not branch on the
   response representation. Archives preserve replay state; public export does not.
7. Contract tests: cover exact wire order, edits/truncation, same-scope and
   cross-scope replay, SQLite resume, collapse, provider variation, and public
   tracing. Identity assertions detect accidental conversion to dictionaries;
   replacement tests verify that edits discard native state.

Prototype evidence (before the object-in-list interface): live tests included all three closed providers
with SQLite resume and changing trailing context; all six directed opaque-state
exclusion checks; and real Nemotron, Qwen, and DeepSeek reasoning transferred to
OpenAI, Anthropic, and Gemini. Those runs used 30 requests, 95,055 input tokens,
and 7,672 output tokens. They covered clients, rendering, storage, and HTTP, not
the TUI or execution of generated Python. Cache-policy evidence belongs to the
follow-up; these results are not a fresh run of the extracted commit.
