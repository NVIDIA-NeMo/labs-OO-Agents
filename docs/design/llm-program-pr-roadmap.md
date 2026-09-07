# LLM Reasoning and Observability PR Roadmap

**Status:** Proposed execution plan
**Parent architecture:** `docs/design/llm-reasoning-observability-program.md`
**Telemetry design:** `docs/design/llm-transaction-telemetry-tui.md`
**AnyLLM decision:** `docs/design/anyllm-sequencing-decision.md`

## 1. Sequencing principles

- Merge behavior before presentation: contracts → capture/replay → telemetry → TUI.
- Keep every PR independently reviewable and releasable where possible.
- Do not mix transport migration with semantic policy changes.
- Add compatibility readers before changing writers; remove compatibility only in
  a later PR.
- Each PR includes contract tests at its boundary, not only unit tests of helper
  functions.
- Open stacked PRs only when the dependency is semantic; otherwise branch from
  current `dev/tui` after the predecessor merges.
- Existing #261/#268/#301 are prototypes/partial foundations. Rework them rather
  than layering contradictory policies indefinitely.

## 2. Recommended treatment of the current stack

### #261 — OpenAI Responses reasoning state

Keep and land after review. It proves native item retention and fixes cache-read
extraction, but its loose usage dictionary is superseded later by the canonical
usage contract.

### #268 — append-only CodeAct + coarse provenance

Keep append-only recovery. Before final merge, either narrow its provenance
claim or add a follow-up that replaces coarse model-family matching with typed
`ProviderIdentity` and a fail-closed opaque compatibility key.

### #301 — opt-in plain-text reasoning capture

Treat as a prototype. The target policy is **always capture, replay by policy**.
Preferred choices:

- amend #301 before merge to remove the capture gate and introduce the first
  replay policy; or
- if reviewability suffers, close/supersede #301 with PRs 2–4 below.

Do not merge `retain_reasoning` as a long-lived capture control and then invert
it immediately unless release timing requires a compatibility bridge.

## 3. PR series

### PR 0 — Design-only program record

**Base:** `dev/tui`
**Contents:** these design documents only.
**Purpose:** agree semantics, names, sequencing, and AnyLLM decision before more
behavior code lands.

Acceptance:
- docs link to one another;
- all key invariants and open decisions are explicit;
- canonical schema names/types agree across the overview and detailed designs;
- no runtime changes.

### PR 1 — Provider identity and capability contracts

**Base:** `dev/tui` after design acceptance.
**Scope:** introduce transport-neutral types without changing behavior.

Deliverables:
- `ProviderIdentity` with logical provider, API style, model, endpoint/account
  fingerprints, transport, opaque compatibility key;
- `CapabilityProfile` for reasoning forms, replay, tool calls, Responses,
  streaming, cache metrics, effort-level map, output-limit semantics;
- typed model configuration using declared capabilities;
- legacy LiteLLM model-string compatibility parser at the adapter edge;
- deterministic identity/capability fixtures for NVIDIA gateway aliases.

Tests:
- golden JSON/schema fixtures freeze `ProviderIdentity`, `ReasoningRecord`,
  `UsageObservation`, `LLMAttemptRecord`, and `LLMCallRecord` names/types before
  implementation PRs depend on them;
- gateway prefixes never determine logical provider;
- opaque compatibility differs across incompatible endpoint/account scopes;
- unknown capability fails closed for Responses/opaque replay;
- generated/upstream capability data can be overridden explicitly.

Does not:
- change reasoning capture/replay;
- migrate transport;
- change TUI.

### PR 2 — Versioned reasoning records and always-capture

**Base:** PR 1.
**Scope:** replace ad-hoc reasoning fields with a normalized event contract.

Deliverables:
- `ReasoningRecord` and `ReasoningKind` (opaque, text, checkpoint);
- every provider-supplied reasoning artifact captured when present;
- ordering/sequence retained;
- provider metadata converted to NOOA-owned JSON;
- event JSON backward reader for #261/#268/#301 fields;
- reasoning-aware repr/redaction classification;
- default-deny external export of reasoning bodies across logs, bug reports,
  clipboard, journal, OTLP, trace download, and normal event-explorer views;
- documented at-rest posture and erase/retention operation (these block
  always-capture acceptance, not deferred hardening);
- terminal-stream backfill for providers that emit encrypted content only at
  completion.

Migration:
- optional event fields require no SQLite schema migration;
- event semantic version distinguishes old `reasoning_items` rows;
- old rows are interpreted conservatively, opaque provenance unknown/fail-closed.

Tests:
- Responses full output, terminal backfill, Anthropic signed/redacted blocks,
  chat reasoning text, tool and text turns, save/resume round-trip;
- capture occurs even when replay is disabled;
- no SDK class crosses the event boundary;
- non-leak tests cover repr/log, bug report, clipboard, journal, OTLP, trace
  download, and default event explorer;
- stateless save/resume reconstructs the next request from local history with
  `store=false` and no continuation ID.

### PR 3 — Replay policy planner and provider adapters

**Base:** PR 2.
**Scope:** make replay policy explicit and independently configurable.

Deliverables:
- `ReasoningReplayMode`: off, auto, native-only, text-context;
- a `ReplayPlanner` consumes history + destination capabilities and emits:
  native replay items, labeled text context, or drops with a reason code;
- default `AUTO`: compatible opaque state native; compatible plain text native;
  incompatible plain text as labeled ordinary context; incompatible opaque drop;
- transport formatters accept the plan rather than inspect model strings;
- capture-gated `retain_reasoning` deprecated/migrated to replay policy.

Tests:
- same-context opaque replay;
- provider/account/endpoint/model incompatibility drops opaque state;
- cross-model text demotion, never native-field impersonation;
- replay off; native-only; explicit text-context;
- tool-call ordering and disabled-thinking follow-up behavior;
- old sessions fail closed for opaque unknown provenance.

### PR 4 — Replay repair, compaction, and privacy

**Base:** PR 3.
**Scope:** robustness around invalid/stale replay state.

Deliverables:
- provider rejection classification for invalid encrypted/signature/checkpoint
  state;
- recovery ladder: retry stripped reasoning → retry stripped checkpoint/full
  history as policy allows → persist repair;
- persisted repair is append-only where possible; if a transcript rewrite is
  necessary, record a repair event with original IDs/digest and reason;
- compaction boundaries retire stale opaque state explicitly and retain only
  provider-compatible checkpoint/newest required signed turn;
- reasoning-aware external-export redaction default;
- erase/retention API for locally persisted plain reasoning.

Tests:
- recovery survives session resume and does not retry forever;
- prefix/signature mismatch, invalid encrypted content, compaction boundary;
- no opaque blob in logs/repr/bug reports/OTLP by default;
- completion and provider-switch behavior unchanged.

### PR 5 — Canonical usage and logical-call lifecycle

**Base:** PR 1; may proceed in parallel with PRs 2–4 after identity contracts
freeze.
**Scope:** typed telemetry contract and normalization only.

Deliverables:
- `UsageObservation`, `CacheStatus`, `LLMCallRecord`, `LLMAttemptRecord`;
- one logical call lifecycle above transport;
- one provider-shape normalizer for dict/object forms;
- input/output/total/reasoning/cache-read/cache-write/cost/latency fields;
- raw usage retained only bounded and scrubbed;
- existing `LLMComplete` gains nested canonical fields plus compatibility
  accessors;
- attempt telemetry but exactly-once terminal usage accumulation.

Tests:
- OpenAI Chat, Responses, Anthropic/LiteLLM, NVIDIA gateway fixture shapes;
- explicit zero vs missing; malformed/negative/NaN rejected or ignored;
- retries, cancellation, streaming, prompt rebuild, double-count prevention;
- no telemetry read by context/summarization control code.

### PR 6 — Persistence and exporter convergence

**Base:** PR 5.
**Scope:** every sink consumes canonical call/usage records.

Deliverables:
- journal token JSON v2 with dual-read of legacy shape;
- provider-neutral trace attributes and logical-call IDs;
- ATIF/headless/task totals include cache write and coverage semantics;
- Trace Explorer reads canonical and legacy fields;
- gateway authoritative cost/header values feed canonical observation;
- no duplicate normalization in actor/journal/trace patches.

Tests:
- legacy/new event and journal reconstruction;
- retries do not double-count;
- OTLP/journal/ATIF/headless totals agree on the same fixture;
- absent stays absent.

### PR 7 — TUI `/usage`, session accumulator, and toolbar

**Base:** PR 6.
**Scope:** presentation only.

Deliverables:
- session `UsageAccumulator` subscribed to terminal call events;
- `/usage`, `/usage last`, `/usage session`, `/usage reset`, `/usage json`;
- `usage` toolbar provider;
- corrected Event Explorer `LLMComplete` field mapping;
- optional config for toolbar compactness;
- metrics remain separate from `ctx N%` context utilization.

Tests:
- full/partial/no usage coverage;
- cache hit/partial/miss/unknown/not-applicable labels;
- last-call vs session totals;
- reset affects display accumulator only, not persisted event history;
- narrow terminal rendering and plugin failure isolation.

### PR 8 — Trace viewer usage cards

**Base:** PR 6 (can run parallel with PR 7).
**Scope:** trace explorer and React viewer.

Deliverables:
- LLM call card: input/output/reasoning/cache-read/cache-write/cost/attempts/
  latency and source labels;
- session aggregate and cache coverage;
- new/legacy trace compatibility;
- frontend assets rebuilt under repository policy.

### PR 9 — AnyLLM adapter on frozen contracts

**Base:** after PRs 1, 3, and 5; may be developed in parallel after contracts
freeze.
**Scope:** rebase/reimplement the private adapter, not a semantic redesign.

Deliverables:
- pinned AnyLLM version;
- Chat, Responses, sync, async, streaming, tools, structured output;
- exact `ProviderIdentity`, `CapabilityProfile`, `ReasoningRecord`, and
  `UsageObservation` mapping;
- normalized errors and attempt telemetry;
- NVIDIA OpenAI-compatible endpoint support without conflating transport and
  logical provider;
- no AnyLLM type outside private adapter/tests.

Gates:
- same conformance fixture suite passes under LiteLLM and AnyLLM;
- credentialed smoke tests for GPT Responses, GLM chat reasoning, and one other
  family;
- no behavior regression in reasoning replay or telemetry.

### PR 10 — AnyLLM staged cutover and LiteLLM removal

**Depends:** PR 4 (repair/privacy), PR 6 (sinks), PR 9 (adapter conformance).
**Scope:** operations and cleanup.

Deliverables:
- provider/API-style adapter selection and rollback switch;
- viewer/eval/memory/tracing bypasses migrated;
- provider-neutral tracing lifecycle active;
- remove LiteLLM patches, callbacks, globals, dependency, docs, and CI filters;
- CI guard prevents new backend imports outside private adapters.

Do not land until live parity and rollback checks pass.

## 4. Parallelism and dependency graph

```text
PR 0 design
  |
  v
PR 1 identity/capabilities
  |\
  | +--> PR 5 usage/lifecycle --> PR 6 sinks --> PR 7 TUI
  |                                  \-------> PR 8 viewer
  v
PR 2 reasoning records/capture
  v
PR 3 replay policy
  v
PR 4 repair/compaction/privacy

PR 1 + PR 3 + PR 5 --> PR 9 AnyLLM adapter
PR 4 + PR 6 + PR 9 --> PR 10 cutover
```

Safe parallel work after PR 1:

- reasoning records/replay policy;
- usage/lifecycle schema;
- TUI mockups against fixture models (without runtime wiring);
- AnyLLM request/response fixture exploration after contracts freeze.

Avoid concurrent mutation of the same checkout. Give mutating subagents isolated
worktrees and one PR-sized objective.

## 5. Delegation packets

### Packet A — Provider identity and capabilities

**Objective:** implement PR 1.
**Files:** model config/registry, new contract module, adapter boundary, fixtures.
**Inputs:** parent design + AnyLLM decision.
**Must prove:** gateway prefixes cannot corrupt logical identity; unknown
capabilities fail closed; no behavior change.

### Packet B — Reasoning schema and backward reader

**Objective:** implement PR 2.
**Files:** events/context models, provider normalization, SQLite round-trip tests.
**Must prove:** unconditional capture, exact ordering, old events load, no SDK
objects persist, export redaction class attached.

### Packet C — Replay planner

**Objective:** implement PR 3.
**Files:** new planner/policy types, formatters, config migration.
**Must prove:** opaque fail-closed; text demotion across models; all tool-call
pairing remains valid; replay disabled works independently of capture.

### Packet D — Repair and compaction

**Objective:** implement PR 4.
**Files:** normalized error mapping, retry/recovery, context collapse, repair
records.
**Must prove:** one bounded ladder, persistent repair, signature/checkpoint
invalidations, no infinite retry.

### Packet E — Usage and lifecycle

**Objective:** implement PR 5.
**Files:** UnifiedLLM public types/lifecycle, adapters, runtime event emission,
fixtures.
**Must prove:** exact-once terminal usage, attempt visibility, optionality, no
control-flow reads.

### Packet F — Sinks

**Objective:** implement PR 6.
**Files:** journal, tracing, ATIF, headless, token accumulator, compatibility
readers.
**Must prove:** totals agree across all sinks; old data reads; no duplicate
normalization.

### Packet G — TUI

**Objective:** implement PR 7.
**Files:** session accumulator, toolbar, commands, event explorer.
**Must prove:** deterministic labels, coverage semantics, last/session/reset,
context display stays separate.

### Packet H — Viewer

**Objective:** implement PR 8.
**Files:** Trace Explorer, viewer API/frontend/tests.
**Must prove:** new + legacy cards, usage/cache/cost source labels, no fabricated
zeros.

### Packet I — AnyLLM conformance adapter

**Objective:** implement PR 9 using the frozen contracts.
**Files:** private adapter, public client delegation, adapter fixtures/live tests.
**Must prove:** transport parity without leaking AnyLLM types or changing policy.

## 6. Program verification matrix

Every PR must run its focused tests and a minimum integration spine:

- event JSON save/resume;
- stateless local reconstruction after process restart with `store=false` and
  no `previous_response_id`/Conversation dependency;
- Chat and Responses tool-loop replay;
- model switch: opaque drop, text demotion;
- compaction/resume boundary;
- logical retry/cancel/stream lifecycle;
- usage fixture normalization and exact-once aggregate;
- trace/journal reconstruction;
- TUI deterministic rendering;
- legacy event/trace compatibility.

Credential-gated matrix before cutover:

- OpenAI GPT Responses encrypted replay;
- NVIDIA gateway GPT Responses;
- GLM chat reasoning capture/replay;
- Kimi chat reasoning capture/replay;
- Anthropic signed/redacted thinking;
- a non-reasoning model for absence semantics.

## 7. Rollout and compatibility

- Add new optional event fields before readers require them.
- Dual-read legacy events/journal rows for at least one full release.
- Keep old flat usage accessors for at least one full release with deprecation
  warnings.
- Remove compatibility only after all four gates: legacy-reader usage below the
  agreed threshold for one release; no legacy-only CI fixtures; documented
  rollback window closed; release-owner approval recorded.
- Convert `retain_reasoning` to replay-policy migration:
  - `true` → `auto`;
  - `false`/unset → target product default selected at design approval.
- Add a global emergency replay-off switch and per-model override.
- AnyLLM cutover is per provider/API style and rollback-safe.
- Close or supersede obsolete stacked prototypes only after replacement tests
  land.

## 8. Decisions required before implementation

1. Confirm `AUTO` cross-model text demotion is the default.
2. Confirm always-capture includes local plaintext persistence and establish
   export/redaction defaults.
3. Confirm #301 is amended/replaced rather than shipping a permanent capture
   gate.
4. Confirm `ProviderIdentity.opaque_replay_key` includes endpoint/account scope,
   not only provider/model.
5. Confirm TUI compact label preference (session totals vs last call).
6. Confirm the AnyLLM prototype is a reference branch, not a merge target.
