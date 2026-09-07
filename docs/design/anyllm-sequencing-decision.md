# Decision Record: AnyLLM in the Reasoning and Observability Program

**Status:** Proposed
**Decision:** Include AnyLLM as a gated transport workstream after provider-neutral contracts, not as the program prerequisite
**Parent:** `docs/design/llm-reasoning-observability-program.md`

## Context

NOOA has two active lines of work:

1. The current LiteLLM-based reasoning stack (#261 → #268 → #301), which adds
   Responses reasoning replay, append-only correction, provenance fencing, and
   chat-family reasoning retention.
2. `feat/anyllm-unified-boundary`, a broad prototype that replaces LiteLLM with
   Mozilla AnyLLM behind `nooa.unifiedllm`.

The AnyLLM branch contains useful architectural work but diverged from the
current line at `97f52de`; the current reasoning branch is roughly 325 commits
newer and the AnyLLM branch has three unique commits. Forty-one changed paths
intersect, including registry, actor, tracing, UnifiedLLM, and tests. A direct
merge would turn conflict resolution into an accidental specification.

## Decision

**DECIDED (D-06): include AnyLLM, but leave it strictly for last.** No AnyLLM
adapter work begins until the reasoning and telemetry tracks are actually
working end to end on the LiteLLM line. The prior recommendation (parallel
adapter work after contract freeze) is superseded: contracts may still be
written transport-neutral so AnyLLM can slot in cleanly later, but implementation
of the adapter is deferred until the other program tracks are demonstrably
working. The existing `feat/anyllm-unified-boundary` branch remains reference
material and is not merged.

**Do not migrate first and do not merge the existing AnyLLM branch unchanged.**

Land provider-neutral contracts and behavior on the current mainline first:

- `ProviderIdentity` and `CapabilityProfile`;
- `ReasoningRecord`, `ReplayPolicy`, and replay repair;
- `UsageObservation` and logical call/attempt lifecycle;
- transport-independent events and conformance fixtures.

Then rebase or reimplement the private AnyLLM adapter against those contracts.
The adapter may proceed in parallel only after the contracts are frozen. It
must not define public semantics.

## Why include AnyLLM at all?

The migration is valuable for this program because it provides:

- explicit provider identity instead of LiteLLM routing-prefix inference;
- a private adapter boundary that can quarantine SDK/provider types;
- normalized errors and streaming;
- a path away from process-global callbacks and LiteLLM-specific tracing;
- a clean place to translate provider-native reasoning and usage into NOOA
  contracts.

This directly addresses the discovered `model_family()` bug: gateway model
strings such as `openai/nvidia/zai-org/glm-5.3` expose transport/routing syntax,
not semantic provider identity. AnyLLM's explicit provider field is better raw
material for `ProviderIdentity`.

## Why not make AnyLLM the prerequisite?

- The prototype predates #261/#268/#301 and does not preserve their reasoning
  semantics.
- Its `ResponsesClient` preserves full output only on tool-call paths and sets
  normalized reasoning to `None` in important paths.
- Its capability model lacks reasoning forms, replay compatibility, cache
  reporting, and retention policy.
- It changes 234 files and removes/reduces many context/token tests. That is a
  migration prototype, not a validated foundation.
- AnyLLM's custom OpenAI-compatible provider does not advertise Responses
  support; the adapter constructs the OpenAI provider internally for compatible
  endpoints. Logical provider identity must therefore remain separate from
  transport implementation.

## Dependency graph

```text
A. Provider-neutral contracts
   ├─ ProviderIdentity
   ├─ CapabilityProfile
   ├─ ReasoningRecord + ReplayPolicy
   ├─ UsageObservation
   └─ logical-call / attempt lifecycle
          |
          +--> B. LiteLLM adapter conformance
          |      + current reasoning behavior
          |      + usage/cache fixtures
          |
          +--> C. AnyLLM adapter reimplementation
                 + same conformance suite
                 + explicit provider identity
                 + Chat + Responses + streaming

B + C --> D. dual-adapter credential-gated smoke tests
          |
          v
E. cutover configuration / staged default
          |
          v
F. remove LiteLLM callbacks, patches, globals, and dependency
```

## Sequencing rules

1. Public events, replay policy, usage schema, and TUI models may not import
   LiteLLM or AnyLLM.
2. Both adapters must satisfy a **semantic-equivalence contract** for the same
   provider recordings: equal canonical identity (except transport), reasoning
   order/replay decisions, usage values and field provenance, tool calls,
   finish status, and errors. Transport-specific diagnostics may differ.
3. The adapter supplies identity and capabilities; runtime never derives them
   from model-string substrings.
4. Tracing is driven by the provider-neutral logical-call lifecycle above both
   adapters.
5. Provider usage remains passive observability under either adapter.
6. Cutover happens per provider/API-style capability, not all-at-once.
7. LiteLLM remains the fallback until the AnyLLM path passes replay, telemetry,
   cancellation, retry, streaming, NVIDIA endpoint, and journal conformance.

## Alternatives rejected

### Migrate to AnyLLM first

Rejected: it would force reasoning/replay and telemetry policy to be redesigned
inside transport code and would regress current behavior before a conformance
contract exists.

### Finish the whole program on LiteLLM, then consider AnyLLM separately

Safe but wasteful: provider identity, call lifecycle, and usage normalization
would likely be designed twice. The contract-first approach preserves momentum
without coupling semantics to LiteLLM.

### Merge the two branches now

Rejected: large divergence, 41 overlapping paths, conflicting response
semantics, and broad test deletion make conflict resolution unreliable.

### Stay on LiteLLM indefinitely

A valid fallback if AnyLLM provider parity fails, but it retains weak provider
identity, callback/global-state coupling, and multiple usage-normalization
paths. The contract program is still worthwhile even if the transport never
changes.

## Entry criteria for AnyLLM implementation

- Provider identity/capability schema merged and versioned.
- Reasoning capture/replay policy merged with fixture conformance tests.
- Replay repair, compaction boundaries, and default privacy/non-leak gates merged.
- Canonical usage and call lifecycle merged.
- No open semantic questions about cache status, cross-model text demotion, or
  opaque compatibility keys.
- A pinned AnyLLM version and support matrix are agreed.

## Exit criteria for transport cutover

- Chat and Responses conformance fixtures pass under both adapters.
- OpenAI encrypted reasoning, Anthropic signed/redacted thinking, and chat
  reasoning-content replay tests pass or are explicitly capability-disabled.
- Usage/cache fixtures produce identical `UsageObservation` values.
- No retry double counting; streaming/cancellation lifecycle closes exactly
  once.
- NVIDIA gateway live smoke tests pass for representative GPT, GLM, and one
  other chat family.
- Tracing/journal/viewer output remains backward compatible.
- A per-provider rollback switch is documented and tested.

## Relationship to the earlier AnyLLM migration design

This decision refines `docs/design/anyllm-unifiedllm-migration.md` on the
prototype branch `feat/anyllm-unified-boundary` (the file is intentionally not
copied into this branch). In particular, the older document proposes removing the task
token accumulator and token totals from TUI/trace surfaces. The expanded
program instead retains **passive provider-reported usage** and deliberately
adds TUI reporting. It preserves the older invariant that usage must never drive
context sizing, summarization, eviction, retries, or scoring.

The older branch remains evidence and reusable implementation material, not the
current specification. Where the documents conflict, this decision and the
parent reasoning/observability program govern.

## Consequences

- The program gains a stable semantic boundary independent of either backend.
- AnyLLM work can be delegated after the contracts land without blocking early
  reasoning/telemetry PRs.
- Some work in the current AnyLLM branch will be reused, but the branch is a
  reference implementation rather than a merge target.
- LiteLLM-specific code survives longer, but its responsibilities shrink behind
  a measurable conformance boundary before removal.
