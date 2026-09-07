# LLM Reasoning, Replay, and Observability Program

**Status:** Proposed for design review
**Date:** 2026-09-07
**Repository baseline:** `origin/dev/tui` at `a335096`
**Prototype stack:** #261 → #268 → #301
**Detailed designs:**
- `docs/design/llm-reasoning-capture-replay.md`
- `docs/design/llm-transaction-telemetry-tui.md`
- `docs/design/anyllm-sequencing-decision.md`
- `docs/design/llm-program-pr-roadmap.md`
- `docs/design/llm-program-execution-backlog.md`

## 1. Decision summary

NOOA will treat provider reasoning and provider-reported usage as first-class,
transport-neutral data owned by the framework.

The target architecture separates four concerns that are currently entangled:

1. **Capture** — retain every reasoning artifact the provider actually returns.
2. **Replay policy** — decide independently whether and how each retained artifact
   is sent on a later request.
3. **Transaction telemetry** — normalize one provider round-trip into a typed,
   optional usage/cache/latency observation with explicit provenance.
4. **Presentation** — aggregate immutable transaction events for the TUI, traces,
   evaluations, and external viewers without feeding telemetry back into prompt
   control flow.

```text
provider SDK / LiteLLM / future AnyLLM
                 |
                 v
          private transport adapter
                 |
       normalized NOOA call result
       + ReasoningRecord[]
       + UsageObservation
                 |
       provider-neutral lifecycle
       (logical call + attempts)
          /                   \
         v                     v
 event/session store      trace/journal sinks
         |                     |
         v                     v
 ReplayPlanner            session aggregates
         |                     |
         v                     v
 provider formatter       /usage + toolbar + viewer
```

The program includes AnyLLM as a **gated transport workstream**, not as the
prerequisite for the contracts above. We first land the provider-neutral
identity, capability, replay, usage, and call-lifecycle contracts on the current
mainline. The AnyLLM adapter is then rebased/reimplemented against those
contracts and must pass the same conformance suite before a transport cutover.

## 2. Product goals

1. **Always retain reasoning supplied by a provider.** “Always” means capture
   every reasoning artifact present in a successful response; it does not mean
   requesting hidden chain of thought that the provider does not expose.
2. **Make replay independently controllable.** Capturing an artifact must not
   imply that it is sent again.
3. **Default replay behavior:**
   - provider-bound opaque state (encrypted content, signed/redacted thinking,
     provider checkpoints) replays only in a verified compatible provider
     context;
   - plain-text reasoning replays natively in compatible contexts and is
     demoted to ordinary, labeled context across model/provider changes;
   - operators can disable replay entirely or select a stricter native-only
     policy.
4. **Observe every logical LLM transaction:** input, output, reasoning, cache
   read, cache write, cache outcome, cost when authoritative, latency, model,
   provider, API style, status, and attempts.
5. **Surface trustworthy usage in the TUI:** last call and session totals, with
   unknown values represented as unknown rather than zero.
6. **Preserve stateless operation.** Local history remains the source of truth;
   server continuation IDs may optimize requests but never become the only
   durable state.
7. **Stay transport-neutral.** Runtime, strategies, event schemas, TUI, and
   exporters do not depend on LiteLLM, AnyLLM, or provider SDK classes.

## 3. Non-goals

- Exposing private chain of thought that a provider intentionally withholds.
- Treating provider-reported usage as exact enough for prompt sizing,
  summarization, eviction, or retry control flow.
- Computing authoritative cost from a local pricing table when the endpoint did
  not report cost.
- Guaranteeing replay compatibility merely because two model strings share a
  substring or provider brand.
- Redesigning the trace viewer in the same PR as foundational event contracts.
- Merging the existing AnyLLM prototype unchanged.

## 4. Core invariants

### 4.1 Reasoning

- Capture and replay are separate decisions.
- Provider-authored output is append-only. Framework corrections are separate,
  explicitly synthetic records.
- Every retained reasoning artifact carries provenance and a versioned replay
  kind.
- Opaque artifacts fail closed: unknown or incompatible provenance means drop,
  never speculative replay.
- Plain text may cross model boundaries only as ordinary labeled context; it is
  never emitted under another provider’s native reasoning field.
- Tool-call ordering is preserved exactly: provider reasoning → assistant/tool
  call → matching tool result.
- A provider rejection cannot permanently brick a session: retry without the
  incompatible state, persist the repair, and continue.
- Compaction owns an explicit boundary. It either preserves a provider-issued
  checkpoint or deliberately retires older replay state.

### 4.2 Telemetry

- One **logical call** has one terminal status and zero or more transport
  attempts. Retries of an identical payload are attempts; a rebuilt prompt is a
  new logical call.
- Only the successful terminal response contributes usage totals. Attempts are
  observable but never double-counted.
- Provider-reported values are optional. Missing is `None`, not `0`.
- Cache read/write and reasoning tokens are subsets/categories, not additive
  extras. They must not be added again to input/output totals.
- “Cache miss” is not inferred from `cached_tokens == 0` unless the adapter knows
  the endpoint performed and reported a cache lookup.
- Raw usage may be retained for diagnostics, but all consumers use the canonical
  schema.
- Telemetry never controls context-window behavior.

### 4.3 Identity and capabilities

- `ProviderIdentity` separates logical provider, API style, endpoint identity,
  model ID, transport adapter, and credential/account scope.
- `CapabilityProfile` is explicit model/endpoint metadata, not inferred from a
  provider SDK class or routing prefix.
- Compatibility for opaque replay is an adapter-supplied compatibility key, not
  a coarse `model_family()` string comparison.

## 5. Target public contracts

The exact module layout is a PR-level decision; the semantic contracts are not.

```python
class ProviderIdentity(BaseModel):
    provider: str                 # openai, anthropic, moonshot, zai, ...
    api_style: str                # responses, chat-completions, messages, ...
    model: str
    endpoint_id: str | None       # stable non-secret endpoint fingerprint
    account_scope: str | None     # non-secret hash/identifier if replay-bound
    transport: str                # litellm, anyllm, direct, fake
    opaque_replay_key: str | None # adapter-declared compatibility boundary

class ReasoningKind(StrEnum):
    OPAQUE = "opaque"             # encrypted/signed/redacted provider state
    TEXT = "text"                 # provider-exposed plain reasoning
    CHECKPOINT = "checkpoint"     # compaction/continuation checkpoint

class ReasoningRecord(BaseModel):
    version: int = 1
    kind: ReasoningKind
    payload: JsonValue            # NOOA-owned JSON only; no SDK object
    provenance: ProviderIdentity
    provider_item_type: str | None
    sequence: int                 # exact order within provider output
    provider_token_count: int | None = None
    replayable: bool = True
    redaction_class: Literal["opaque", "plain_reasoning"]

class ReasoningReplayMode(StrEnum):
    OFF = "off"
    AUTO = "auto"                # default policy below
    NATIVE_ONLY = "native_only"
    TEXT_CONTEXT = "text_context" # force plain text to labeled context

class UsageObservation(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None
    cache_eligible_input_tokens: int | None
    cost_usd: Decimal | None
    cache_status: CacheStatus
    field_sources: dict[str, UsageSource]
    raw: dict[str, JsonValue] | None

class LLMAttemptRecord(BaseModel):
    attempt: int
    started_at: float
    ended_at: float
    first_token_at: float | None
    status: Literal["success", "error", "cancelled"]
    usage: UsageObservation | None  # billed/provider-observed attempt usage

class LLMCallRecord(BaseModel):
    call_id: str
    generation_id: str | None
    provider: ProviderIdentity
    request_fingerprint: str
    started_at: float
    ended_at: float
    first_token_at: float | None
    status: Literal["success", "error", "cancelled"]
    finish_reason: str | None
    attempts: tuple[LLMAttemptRecord, ...]
    usage: UsageObservation | None  # successful terminal response usage
    reasoning: list[ReasoningRecord]
```

## 6. Default replay policy

| Retained artifact | Compatible destination | Incompatible destination | Replay off |
|---|---|---|---|
| OpenAI encrypted reasoning item | Native item if `opaque_replay_key` matches | Drop | Drop |
| Anthropic signed/redacted thinking | Native block if signature context is valid | Drop | Drop |
| Provider compaction checkpoint | Native item if compatibility key matches | Drop | Drop |
| Plain reasoning text | Native field in `AUTO`; labeled context in `TEXT_CONTEXT` | `AUTO`: labeled context only when approved; `TEXT_CONTEXT`: always labeled | Drop |

Compatibility is stronger than “same family.” It may include provider, API
style, endpoint/account scope, model compatibility group, and provider-specific
prefix/signature constraints.

## 7. Cross-harness evidence and current-state assessment

A five-project survey of Prime Agent, OpenCode, Hermes, Pi, and OpenClaw found
that all five:

- capture provider reasoning unconditionally and apply policy at replay time;
- preserve encrypted/signed provider state in local durable history rather than
  relying on `previous_response_id` as the primary state mechanism;
- fence opaque state to compatible provider provenance;
- preserve reasoning on tool-owning assistant turns;
- treat provider rejection as a recoverable replay-compatibility failure.

The surveyed systems deliberately carry plain reasoning text across model
switches as labeled context while refusing to replay provider-bound state. That
supports the proposed `AUTO` default and challenges #301's capture gate and
same-family-only text rule. Evidence lives outside this worktree at
`/localhome/local-pfurgale/dev/tmp/reasoning-survey/reports/`.

The open stack is useful proof and partial foundation:

- **#261** captures/replays OpenAI Responses reasoning and corrects cache-read
  extraction.
- **#268** makes CodeAct text correction append-only and adds provenance gating.
- **#301** captures/replays chat-family reasoning behind a per-alias capture
  switch and fixes gateway-prefix family parsing.

The target design changes #301’s policy: reasoning capture becomes unconditional
when supplied; `retain_reasoning` is replaced by a replay policy. The stack also
needs stronger provider identity than `model_family`, fail-closed opaque replay,
reasoning-aware compaction/redaction, and rejection repair.

Current telemetry is fragmented:

- `LLMComplete` has prompt/output/cache-read/reasoning/cost flat integers, with
  missing values forced to zero and no cache-write field.
- normalization is duplicated between clients, runtime, trace patch, and journal;
- the journal loses Responses and Anthropic cache shapes;
- ATIF and headless aggregate a subset;
- the TUI exposes context utilization but no last-call/session usage or cache
  metrics.

## 8. Program phases

1. **Contract foundation** — identity, capabilities, reasoning records, replay
   policy, canonical usage, logical call/attempt lifecycle.
2. **Always-capture + policy replay** — revise/supersede #301; add repair and
   compatibility tests.
3. **Canonical telemetry pipeline** — adapters normalize once; events, journal,
   ATIF/headless consume the same schema.
4. **TUI usage experience** — session accumulator, `/usage`, toolbar item, event
   explorer fixes.
5. **AnyLLM conformance adapter** — rebase/reimplement the private adapter on the
   contracts; dual-transport fixture and live smoke tests.
6. **Cutover and cleanup** — migrate bypasses/tracing, remove LiteLLM-specific
   callbacks and globals only after parity gates pass.

Detailed sequencing and delegation packets are in
`docs/design/llm-program-pr-roadmap.md`.

## 9. Decision points for review

1. Should plain reasoning text cross models by default (`AUTO`) as labeled
   context? Proposed: **yes**, matching all five surveyed harnesses.
2. Should raw plain reasoning be exportable by default? Proposed: stored locally,
   **redacted from external exports unless explicitly enabled**.
3. How long should stale reasoning survive compaction? Proposed: keep provider
   checkpoints and the newest necessary signed/native turn; demoted text may be
   summarized, older opaque state retired at an explicit boundary.
4. Should the initial TUI toolbar show call or session cache rate? Proposed:
   compact session summary; `/usage` supplies last-call detail.
5. Should #301 be amended or replaced? Proposed: treat it as a prototype and
   **replace its capture gate with always-capture + replay policy before merge**.
