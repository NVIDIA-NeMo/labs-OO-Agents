# Reasoning Capture and Replay Design

**Status:** Proposed for design review
**Parent:** `docs/design/llm-reasoning-observability-program.md`
**Cross-harness evidence:** `/localhome/local-pfurgale/dev/tmp/reasoning-survey/reports/`

## 1. Problem

Provider reasoning appears in several incompatible forms:

- OpenAI Responses reasoning items with encrypted content;
- Anthropic signed thinking and redacted-thinking blocks;
- provider compaction/continuation checkpoints;
- plain-text `reasoning_content` from GLM, Kimi, DeepSeek, Qwen, Nemotron,
  and OpenAI-compatible endpoints;
- reasoning token counts with no replayable body.

NOOA must preserve the provider's output without assuming that every artifact is
safe or valid to send to every later model. Capture, storage, replay, export,
and display are therefore separate policy points.

## 2. Goals

1. Capture every provider-supplied replayable reasoning artifact.
2. Preserve exact provider output order and provider-authored boundaries.
3. Replay opaque state only to a verified compatible provider context.
4. By default, carry plain reasoning text across model switches as ordinary,
   labeled context—not as another provider's native reasoning field.
5. Let users turn replay off without losing captured history.
6. Preserve reasoning on tool-owning turns even if later requests disable new
   reasoning generation.
7. Recover from stale/invalid replay state and persist the repair.
8. Make compaction and redaction reasoning-aware.

## 3. Non-goals

- Requesting hidden chain of thought not exposed by a provider.
- Treating displayed reasoning summaries as equivalent to provider replay state.
- Replaying opaque state based only on model-name substring equality.
- Guaranteeing that plain-text reasoning improves a different model.
- Keeping every historical reasoning artifact active after compaction forever.

## 4. Data model

```python
class ReasoningKind(StrEnum):
    OPAQUE = "opaque"
    TEXT = "text"
    CHECKPOINT = "checkpoint"

class ProviderIdentity(BaseModel):
    provider: str
    api_style: str
    model: str
    endpoint_id: str | None
    account_scope: str | None
    transport: str
    opaque_replay_key: str | None

class ReasoningRecord(BaseModel):
    version: int = 1
    kind: ReasoningKind
    payload: JsonValue
    provenance: ProviderIdentity
    provider_item_type: str | None = None
    sequence: int
    provider_token_count: int | None = None
    replayable: bool = True
    redaction_class: Literal["opaque", "plain_reasoning"]
```

Reasoning records live on the provider-authored assistant event, alongside its
text, calls, and output-item sequence. They are not attached to framework-
synthetic corrections.

### Why not `reasoning_items` + `reasoning_content` forever?

Those prototype fields split by current wire shape and encourage formatter
logic to inspect ad-hoc dictionaries. `ReasoningRecord[]` gives one versioned,
provider-neutral persistence format and lets adapters own wire conversion.
Backward readers translate #261/#268/#301 fields into records.

## 5. Capture policy

**Capture is unconditional when a provider returns an artifact.** Replay policy
does not affect capture.

Capture rules:

- preserve output sequence and item type;
- convert provider objects to bounded JSON immediately;
- stamp full `ProviderIdentity` at capture time;
- retain opaque bodies without logging/repr exposure;
- retain plain text locally under a reasoning-specific privacy class;
- backfill terminal-only artifacts (for example Azure encrypted content emitted
  at `response.completed`) before finalizing the assistant event;
- if only token counts are available, record usage but create no fake reasoning
  body.

For streams, the adapter accumulates fragments privately and emits one ordered
record set at terminal success. Partial/failed streams may retain bounded
observability diagnostics, but never claim replayable complete state.

## 6. Replay policy

```python
class ReasoningReplayMode(StrEnum):
    OFF = "off"
    AUTO = "auto"
    NATIVE_ONLY = "native_only"
    TEXT_CONTEXT = "text_context"
```

### Default: `AUTO`

1. Opaque/checkpoint state:
   - replay natively only when the destination's `opaque_replay_key` matches;
   - otherwise drop with a structured reason.
2. Plain text:
   - replay in the provider-native reasoning field when destination capability
     declares it compatible;
   - otherwise demote only when a declared policy/capability allows labeled
     cross-model context (the default product policy enables this for non-
     sensitive text); never claim demoted text is native destination reasoning.
3. If replay capability is unknown:
   - opaque state fails closed;
   - text demotes only under the approved AUTO policy; otherwise drops.

### Other modes

- `OFF`: capture persists; no reasoning artifact is emitted.
- `NATIVE_ONLY`: compatible native replay only; incompatible text and opaque
  state drop.
- `TEXT_CONTEXT`: explicitly force all captured plain reasoning into labeled
  ordinary context—even when a provider-native field exists. This is a
  diagnostic/portability mode, not the default. Opaque state remains
  fail-closed.

A global emergency `reasoning_replay=off` overrides per-model/session policy.

### 6.1 How cross-model text demotion works, concretely

The mechanism is a **render-time content transformation of one message**, not
a new message type and not a history rewrite:

1. Each stored assistant turn already carries `ReasoningRecord(kind=TEXT, payload=<string>)`
   next to its text, calls, and outputs. Nothing is copied out of the turn.
2. When rendering history for a request, the ReplayPlanner classifies each
   reasoning record against the destination's `CapabilityProfile`:
   - `native` → the record is emitted in the destination's native reasoning field
     (e.g. `reasoning_content` on the assistant message), still on the same turn;
   - `labeled_context` → the record is emitted **on the same assistant turn** as an
     ordinary text block prefixed with a non-negotiable label, e.g.
     `[Prior assistant reasoning — source: glm-5.3, not generated by the current model]`;
   - `drop` → omitted, with a structured drop reason recorded on the call.
3. Ordering is unchanged: on a tool turn the demoted block stays attached to the
   assistant message that made the call, so the provider still sees
   `assistant(+labeled reasoning) → function_call → function_call_output`.
   No item is moved before the reasoning-bearing turn or after the tool result.
4. The transform is **idempotent and stateless**: it is computed from the stored
   records on every render. If you switch models again, the same record is
   re-classified against the new destination — no persisted "demoted" copies,
   no migration, nothing to un-do.
5. Wire shapes stay per-provider: the OpenAI formatter emits the labeled block as
   a plain text part of the assistant message; the Responses formatter emits it as
   an assistant input message part; Anthropic emits it as a text block. The label
   text itself is provider-neutral and generated by the planner, so formatters
   never invent their own attribution.

Why this is clean: the stored event never mutates; there is exactly one decision
point (the planner); the provider sees a self-describing history where any prior
reasoning is explicitly attributed to its source model; and the label makes it
impossible for the destination model (or a reader) to mistake demoted text for
its own reasoning. Cost note: demoted text consumes ordinary context tokens, so
the planner bounds it (per-record char cap + total demoted-token budget, oldest
first) and the drop reason records when the budget excluded a record.

## 7. Compatibility

Opaque compatibility is adapter-declared and may include:

- logical provider;
- API style;
- endpoint or gateway identity;
- account/credential scope if state is bound to it;
- model compatibility group or revision;
- provider-specific prompt/signature constraints.

The compatibility key is a non-secret digest. It is not derived in runtime from
model strings. If the adapter cannot prove compatibility, replay is refused.

Plain-text native compatibility is a separate capability. A provider may accept
`reasoning_content` but use a different field or reject it on tool calls; the
capability profile declares the wire dialect.

## 8. Tool-call and append-only behavior

### Prior reasoning is independent of the current reasoning toggle

Pi and Prime Agent provide the clearest reference implementation: one
message-transform choke point makes replay decisions from provenance stamped on
each prior assistant message. The *current request's* thinking/effort toggle
only controls generation parameters. It never rewrites history or removes prior
signed/encrypted thinking.

NOOA adopts that invariant:

- `reasoning.effort=none` means “generate no new reasoning this turn,” not
  “discard reasoning required to replay prior turns”;
- ReplayPlanner is blind to the current generation effort except where a
  provider capability explicitly says prior replay is forbidden;
- prior tool-owning reasoning remains paired with its tool call and result;
- orphan tool calls receive an explicit synthetic error result with the original
  call ID, preserving provider pairing constraints.

The provider-authored assistant turn is immutable:

```text
ReasoningRecord(s)
Assistant text (optional)
Function call(s)
Function output(s)
Framework correction (optional, separate synthetic record)
```

- CodeAct/Experimental CodeAct never delete the provider turn.
- Synthetic comment/return/custom corrections are explicit framework events.
- A later request that disables new reasoning generation still replays the
  retained reasoning belonging to prior tool calls when compatibility requires
  it.
- Orphan tool calls are repaired with explicit synthetic results; reasoning
  remains on the original assistant turn.

## 9. Rejection recovery ladder

### Durable repair shape

OpenClaw's useful model is copy-on-write branch repair: branch before the first
rejected entry, re-append the repaired suffix with new IDs, and leave the old
rows on an abandoned branch. Resume selects the newest branch. This gives
append-only storage, durable repair, and free undo, but OpenClaw lacks a typed
repair marker.

NOOA should preserve the same benefits in its event model:

- never mutate or delete the provider-authored event;
- append a typed `ReplayRepair` record that references affected event/reasoning
  IDs and digests;
- active-view rendering applies repair records deterministically;
- the original branch remains queryable and the repair is reversible;
- session resume reconstructs the repaired active view without relying on an
  in-process boolean or warning log.

When a provider rejects replay state with a classified compatibility error:

1. retry once with rejected reasoning/checkpoint records stripped;
2. if relevant, retry with stale compaction state stripped and rebuilt full
   retained history;
3. persist a `ReplayRepair` event recording:
   - affected event/reasoning IDs and digests (not body);
   - destination identity and error class;
   - action taken and bounded attempt number;
   - replacement/retirement state;
4. future renders honor the repair and do not repeat the same invalid replay;
5. bounded exhaustion surfaces the provider error.

Repair is append-only where possible. If an active-view override is needed, it
must be durable, inspectable, and reversible; the original event remains stored.

## 10. Compaction

Compaction is a replay boundary, not only a text-summary operation.

- provider checkpoints replace older native replay state only when the adapter
  declares the checkpoint valid;
- keep the newest required signed/native turn where providers require it;
- retire stale opaque records before summarizing their visible effects;
- plain reasoning text may be summarized into ordinary context; the summary is
  not provider-native reasoning;
- record which reasoning IDs/checkpoints were retired;
- never estimate retained encrypted-state cost from ciphertext byte length;
  use provider-reported usage.

## 11. Privacy, redaction, and lifecycle

- **Product decision (overrides the earlier draft): reasoning is a first-class
  traced artifact, not a secret.** OTLP spans, the message journal, trace
  downloads, and the normal Event Explorer previews include reasoning content
  (plain text and opaque blobs alike), because tracing must retain everything
  that was sent to the model. Export is ON by default.
- The local session/event store persists reasoning records as ordinary event
  JSON (confirmed: Pydantic-validated, SQLite-backed, survives shutdown/resume
  — the same store that survives events today), so captured reasoning survives
  restart and session resume with no extra mechanism.
- Redaction is therefore **opt-in, not default**: `export_reasoning=false`
  (or a coarser `redact_reasoning=true`) suppresses reasoning bodies from
  external exports for compliance-sensitive deployments. Provenance and policy
  metadata always remain visible even when a payload is redacted.
- Opaque blobs are still kept out of *accidental* leakage channels — exception
  tracebacks, debug logs, `repr()` output, and stack traces — because those are
  not tracing surfaces and serve no diagnostic purpose. But journal/OTLP/Event
  Explorer render them.
- Reasoning-aware *optional* redaction must not mistake ciphertext for harmless
  noise when enabled; the redaction toggle is a documented compliance feature.
- Add an erase/retention operation for reasoning bodies that preserves event
  structure and audit metadata.
- Bug reports default to including reasoning (same rationale: full trace of what
  the model saw), with the same opt-out toggle honored.

## 12. Provider capability profile

Minimum fields:

```python
class ReasoningCapabilities(BaseModel):
    capture_kinds: frozenset[ReasoningKind]
    native_replay_kinds: frozenset[ReasoningKind]
    effort_map: dict[str, str | None]   # null means unsupported
    supports_reasoning_when_disabled: bool | None
    requires_tool_turn_reasoning_replay: bool | None
    replay_field: str | None
    signature_prefix_sensitive: bool | None
    terminal_backfill: bool = False
```

Profiles may come from generated catalogs, provider metadata, or explicit
registry overrides. Unknowns are not silently promoted to supported.

## 13. Tests

Required cross-provider fixtures:

- a follow-up request with reasoning disabled still replays prior compatible
  tool-owning reasoning/signatures;
- orphaned tool calls receive synthetic matching results before replay;
- OpenAI Responses encrypted item, tool loop, terminal-only backfill;
- OpenAI bridge chat-shaped reasoning items;
- Anthropic signed/redacted thinking, prefix mismatch, compaction boundary;
- GLM/Kimi/DeepSeek/Qwen/Nemotron reasoning-content tool and text turns;
- same context native replay; different context opaque drop; text demotion;
- capture with replay off; session save/resume; archive/compaction;
- save/resume reconstructs the next request entirely from local history with
  `store=false` and no `previous_response_id`/server continuation dependency;
- rejection recovery persists and prevents repeat failure;
- reasoning bodies present in journal/OTLP/trace-download/Event Explorer by
  default, and absent when `export_reasoning=false` is set (both paths tested).

## 14. Compatibility migration

- Translate old `reasoning_items` and `reasoning_content` optional event fields
  into `ReasoningRecord[]` on read.
- Unknown old opaque provenance fails closed.
- Migrate `retain_reasoning=true` to replay `AUTO`; `false`/unset does not disable
  capture in the new design.
- Keep old wire serialization fields for **at least one full release**. Remove
  only when: legacy-reader usage is below an agreed threshold for one release,
  compatibility CI has no legacy-only fixtures, the rollback window has closed,
  and the release owner explicitly approves removal.
