# Assistant turns, persistence, and reasoning replay

NOOA stores each model response as one `LLMResponse` event containing ordered
text, tool-call, and reasoning parts. UnifiedLLM interprets provider extensions;
the renderer, middleware, and relay continue to exchange ordinary public JSON
messages. This preserves the information needed for replay without exposing
provider formats throughout the framework.

## Why ordered parts

Flattening a response into independent text and tool-call lists loses the order
and grouping that the provider returned. Reconstructing it later requires extra
bookkeeping. Signed thinking, encrypted reasoning, and thought signatures also
must not be attached to edited public content or sent to an incompatible model.

The ordered response remains the authority. Wire messages are generated at
dispatch, using native extensions only when the public message is unchanged and
the destination scope matches. This preserves supported provider message order,
boundaries, and fields rather than reconstructing them from separate carriers.
Stable replay is necessary for prompt-cache reuse; it does not guarantee a cache
hit or select a provider cache policy.

Compared with the superseded replay implementation, this removes order ledgers,
batch identities, content fingerprints, private dict-subclass carriers, and relay
index reconciliation. It retains one equality comparison at dispatch, provider
validation, and the final wire adapters. Those costs are explicit, not eliminated.

## Caller-facing contract

- `LLMResponse.parts` is an immutable tuple. Each part is text, a tool call, or
  reasoning; each may have an opaque `native` extension read only by UnifiedLLM.
- `.content`, `.tool_calls`, and `.reasoning` are derived public views. They are
  not a second stored copy of the response. `.usage` stores normalized input,
  output, cache-read, cache-write, reasoning-token, and estimated-cost fields.
- `call` and `acall` accept public message dictionaries plus an optional,
  keyword-only `turns: Mapping[str, LLMResponse]`. This lookup is request-local,
  never a provider parameter. Runtime callers receive it automatically.
- A public assistant dictionary can carry `nooa_turn`, the original event ID.
  UnifiedLLM removes it before the network request. The runtime borrows the
  original events already rendered; it does not reload the archive to build the
  lookup. Context-window recovery builds a new lookup from the rebuilt history.
- Middleware may edit ordinary dictionaries. If a referenced assistant message
  differs from the event's public view, UnifiedLLM warns and sends only portable
  content. Missing references also warn and replay portably. Invalid reference
  types or mismatched ID-to-event associations raise an error.

For direct callers:

```python
turns = {}
reply = await client.acall(messages)
messages.append({**reply.public_message(), "nooa_turn": reply.id})
turns[reply.id] = reply
# Append matching tool results if reply contains tool calls, then continue:
next_reply = await client.acall(messages, turns=turns)
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
Unknown capture routes warn and keep portable text while dropping opaque state.
Malformed recognized state raises; it is not silently treated as a successful
capture. Empty public tool IDs are accepted where no retained native state needs
binding; nonempty duplicate IDs and ambiguous native bindings raise.

Provider strings remain immutable in storage. Container detachment happens at
capture and final projection, without repeatedly copying large immutable string
leaves. Live SDK responses and parsed Python results are excluded from archives.

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
are unaffected by this pairing rule. Protocol-safe omission is not atomic
collapse. Changing that policy is a separate decision.

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
   It grants no authority to opaque fields supplied in ordinary wire dictionaries.
4. `unifiedllm/unifiedllm.py`: resolves public IDs, compares with the authoritative
   event, and dispatches to the effective model. This is where edits lose native
   replay authority. Its abstract and concrete methods expose the same lookup
   contract, including the reasoning wrapper and fake client.
5. Renderer/formatter/runtime: carry only public values and an event reference;
   one shared assistant-message builder keeps both sides of the comparison equal.
   The live-context boundary prevents Responses from moving trailing system
   context into leading instructions. Provider cache mapping is not added here.
6. Storage and event hooks: serialization, searchable fields, and empty-event
   handling are owned by the event type. Generic consumers do not branch on the
   response representation. Archives preserve replay state; public export does not.
7. Contract tests: cover exact wire order, edits/truncation, same-scope and
   cross-scope replay, SQLite resume, collapse, provider variation, and public
   tracing. Unexpected edited-turn downgrade warnings fail UnifiedLLM tests.

Live evidence from the preceding prototype included all three closed providers
with SQLite resume and changing trailing context; all six directed opaque-state
exclusion checks; and real Nemotron, Qwen, and DeepSeek reasoning transferred to
OpenAI, Anthropic, and Gemini. Those runs used 30 requests, 95,055 input tokens,
and 7,672 output tokens. They covered clients, rendering, storage, and HTTP, not
the TUI or execution of generated Python. Cache-policy evidence belongs to the
follow-up; these results are not a fresh run of the extracted commit.
