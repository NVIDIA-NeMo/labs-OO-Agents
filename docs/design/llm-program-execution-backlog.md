# LLM Reasoning and Observability Execution Backlog

**Status:** Proposed
**Roadmap:** `docs/design/llm-program-pr-roadmap.md`

This is the durable, delegation-ready backlog. IDs are stable across issue/PR
creation. Each task is sized for one owner or one small worker team. A worker
must read the parent design and the named task packet before editing.

## Program decisions (human owners)

| ID | Decision | Proposed default | Blocks |
|---|---|---|---|
| D-01 | Cross-model plain reasoning | Demote to labeled ordinary context in `AUTO` | R-03 |
| D-02 | Plain-reasoning privacy | **DECIDED:** export ON by default (tracing retains everything sent to the model); `export_reasoning=false` opt-in suppression; event-store persistence confirmed | R-02, R-04 |
| D-03 | Prototype #301 disposition | **DECIDED:** supersede | R-02 |
| D-04 | Opaque compatibility scope | Provider + API + endpoint/account + model compatibility | C-01, R-03 |
| D-05 | TUI compact metric | **DECIDED:** `↑in ↓out ↻cached% (n/m)`; cache segment hidden when endpoint lacks cache capability; `c` ASCII fallback | U-02 |
| D-06 | AnyLLM branch treatment | **DECIDED:** include but strictly last — no adapter work until reasoning + telemetry tracks are working; branch stays reference-only | A-01 |

## Contract foundation

### C-01 — ProviderIdentity

**Depends:** D-04
**Objective:** define a versioned provider identity independent of transport
routing strings.

- [ ] Provider, API style, model, endpoint fingerprint, account scope, transport.
- [ ] Adapter-declared opaque replay compatibility key.
- [ ] Legacy LiteLLM model-string parser isolated at adapter edge.
- [ ] NVIDIA gateway fixtures: GPT, GLM, Kimi, DeepSeek, Qwen, Nemotron.
- [ ] Unknown identity fails closed for opaque replay.

**Acceptance:** no runtime/provider formatter infers semantic identity from model
substrings; all fixtures round-trip JSON.

### C-02 — CapabilityProfile

**Depends:** C-01
**Objective:** model/endpoint capabilities declared rather than guessed.

- [ ] Chat/Responses/tool/streaming/structured-output support.
- [ ] Reasoning capture and native replay kinds.
- [ ] Neutral effort map with null=unsupported and clamping policy.
- [ ] Cache-read/write reporting and cache-key/retention controls.
- [ ] Output-limit wire parameter semantics.
- [ ] Generated catalog inputs plus explicit override precedence.

**Acceptance:** `/reasoning`, formatter selection, and adapter options consume the
profile; unsupported values fail before network I/O.

## Reasoning capture and replay

### R-01 — ReasoningRecord schema and backward reader

**Depends:** C-01
**Objective:** replace ad-hoc reasoning fields with versioned normalized records.

- [ ] Opaque/text/checkpoint kinds, sequence, provider item type, provenance.
- [ ] Convert provider objects to bounded JSON at adapter edge.
- [ ] Translate legacy #261/#268/#301 event fields on read.
- [ ] SQLite/session save-resume and archive tests.
- [ ] Redaction class attached independently of payload kind.

**Acceptance:** no provider SDK objects persist; old sessions load; exact output
order survives save/resume.

### R-02 — Always-capture

**Depends:** R-01, D-02, D-03, X-01
**Objective:** capture every supplied reasoning artifact independent of replay.

- [ ] OpenAI Responses complete output including terminal backfill.
- [ ] Anthropic signed and redacted thinking blocks.
- [ ] Chat-family reasoning content for GLM/Kimi/DeepSeek/Qwen/Nemotron.
- [ ] Tool and terminal-text turns.
- [ ] Capture with replay disabled.
- [ ] Reasoning exported by default to journal, OTLP, trace download, bug
  reports, and normal Event Explorer previews (D-02 override: tracing retains
  everything sent to the model).
- [ ] Opt-in `export_reasoning=false` suppression honored across all sinks,
  with tests for both default and suppressed paths.
- [ ] Opaque blobs kept out of traceback/repr/debug-log noise channels.
- [ ] Reasoning erase/retention operation.
- [ ] Stateless save/resume reconstructs the next request from local history
  with `store=false` and no continuation ID.

**Acceptance:** capture fixtures pass for every family; export-both-paths and
stateless-resume tests pass; #301 capture switch no longer controls storage.

### R-03 — ReplayPlanner

**Depends:** R-01, R-02, C-02, D-01
**Objective:** plan native replay, labeled text context, or drop.

- [ ] Modes: off, auto, native-only, text-context.
- [ ] Compatible opaque replay; incompatible/unknown opaque fail-closed.
- [ ] Compatible native text replay.
- [ ] Cross-model text demotion to labeled ordinary context.
- [ ] Global emergency off + per-session/model override.
- [ ] Disabled-thinking follow-up retains prior tool-turn state.

**Acceptance:** matrix tests cover provider/API/endpoint/model switches and tool
ordering; formatters do not inspect model strings.

### R-04 — Replay repair, compaction, and privacy

**Depends:** R-03
**Objective:** robustly retire invalid replay state and persist repair.

- [ ] Classify invalid encrypted/signature/checkpoint errors.
- [ ] Bounded retry ladder with reasoning/checkpoint stripping.
- [ ] Durable append-only `ReplayRepair` record.
- [ ] Compaction retirement/checkpoint rules.
- [ ] Anthropic prefix/signature invalidation fixtures.
- [ ] Reasoning body erase/retention API.
- [ ] Export/log/bug-report/clipboard redaction tests.

**Acceptance:** a repaired session resumes without repeating the failure; no
opaque body leaks to default external surfaces.

## Usage, cache, and call lifecycle

### M-01 — UsageObservation normalizer

**Depends:** C-01
**Objective:** one typed usage schema and one adapter-edge normalizer.

- [ ] Input/output/total/reasoning/cache-read/cache-write/cost fields.
- [ ] Cache status and source; bounded raw payload.
- [ ] Dict/object fixture shapes for Chat, Responses, Anthropic, NVIDIA gateway.
- [ ] Preserve zero vs absent; reject invalid numerics.
- [ ] Provider/gateway authoritative cost mapping.

**Acceptance:** fixtures yield exact values and optionality; no consumer probes
provider aliases.

### M-02 — Logical-call and attempt lifecycle

**Depends:** C-01, M-01
**Objective:** exact-once terminal telemetry across retries/streams/cancellation.

- [ ] Call ID, request fingerprint, attempt records, timing/first-token timing.
- [ ] Retry vs prompt-rebuild semantics.
- [ ] Success/error/cancel terminal states.
- [ ] One successful usage contribution per logical call.
- [ ] Fake client participates in the same lifecycle.

**Acceptance:** duplicate/double-count tests; cancelled and abandoned streams
close exactly once.

### M-03 — Event and compatibility bridge

**Depends:** M-01, M-02
**Objective:** evolve `LLMComplete` without breaking consumers.

- [ ] Nested provider/call/usage fields.
- [ ] One-release flat compatibility accessors.
- [ ] Old event JSON reader tests.
- [ ] Task/headless/ATIF aggregates updated.
- [ ] Remove runtime's second usage-shape normalization.

**Acceptance:** legacy and new fixtures produce identical canonical aggregates.

### M-04 — Journal and trace convergence

**Depends:** M-03
**Objective:** all sinks consume canonical records.

- [ ] Journal token JSON v2 + dual reader.
- [ ] Trace attributes for cache write/status/source/attempt/latency/cost.
- [ ] Trace Explorer new + legacy readers.
- [ ] No duplicated provider normalization in journal/trace patches.
- [ ] Cross-sink parity tests.

**Acceptance:** event/journal/OTLP/ATIF/headless totals agree on one fixture.

## TUI and viewer

### U-01 — Session usage accumulator

**Depends:** M-03
**Objective:** replayable/read-only aggregate over terminal call events.

- [ ] Last-call record and session totals.
- [ ] Coverage for usage/cache/cost.
- [ ] Independent totals; no subset double counting.
- [ ] Reset display accumulator without deleting persisted events.
- [ ] Resume/rebuild from existing events.

### U-02 — `/usage` and toolbar provider

**Depends:** U-01, D-05
**Objective:** useful terminal metrics without confusing context utilization.

- [ ] `/usage`, `last`, `session`, `reset`, `json`.
- [ ] `usage` toolbar provider with bounded compact label.
- [ ] Hit/partial/miss/not-applicable/unknown rendering.
- [ ] Narrow-terminal, missing-data, partial-coverage tests.
- [ ] Documentation and completion entries.

### U-03 — Event explorer and trace viewer

**Depends:** M-04
**Objective:** per-call and session metrics in diagnostics/UI.

- [ ] Fix `LLMComplete` field mapping.
- [ ] React card input/output/reasoning/cache read/write/cost/attempt/latency.
- [ ] Source/coverage labels.
- [ ] Legacy trace compatibility and rebuilt frontend assets.

## AnyLLM transport workstream

### A-01 — Rebase/reimplement AnyLLM adapter

**DECIDED (D-06): do not start until R-02, R-04, M-04, U-02, and U-03 are
complete and working. Depends:** C-01, C-02, R-03, M-02
**Objective:** port the private adapter to current mainline contracts.

- [ ] Pin AnyLLM version.
- [ ] Chat + Responses + tools + structured + streaming.
- [ ] Normalize identity, capability, reasoning, usage, errors, and attempts.
- [ ] No AnyLLM type outside private adapter/tests.
- [ ] Same fixture suite under LiteLLM and AnyLLM.

### A-02 — Live conformance matrix

**Depends:** A-01
**Objective:** prove endpoint behavior before cutover.

- [ ] GPT Responses encrypted replay.
- [ ] NVIDIA GPT Responses.
- [ ] GLM reasoning-content replay.
- [ ] Kimi reasoning-content replay.
- [ ] Anthropic signed/redacted thinking.
- [ ] Streaming, cancellation, retries, cache telemetry.

### A-03 — Staged cutover

**Depends:** A-02, M-04, R-04
**Objective:** migrate providers/API styles with rollback.

- [ ] Per-provider adapter selection and rollback.
- [ ] Viewer/eval/memory/tracing bypasses migrated.
- [ ] Provider-neutral tracing lifecycle enabled.
- [ ] LiteLLM patches/callbacks/globals/dependency removed only after parity.
- [ ] CI import-boundary guard.

## Cross-cutting review tasks

### X-01 — Security/privacy gate

**Depends:** D-02
**Blocks:** R-02

- [ ] Specify at-rest posture and threat model before always-capture lands.
- [ ] Specify export defaults, redaction, erase, bug reports, clipboard, traces.
- [ ] Endpoint/account compatibility and key rotation.
- [ ] Raw usage size/secrets.

**Acceptance:** test plan names every external surface and proves default
non-leak behavior; product/security owner approves the at-rest posture.

### X-02 — Migration/release review

- [ ] Event and journal versioning.
- [ ] Old sessions/traces and stateless resume without server continuation IDs.
- [ ] Config deprecation (`retain_reasoning` → replay mode).
- [ ] Compatibility-removal gates: reader usage threshold, minimum one full
  release, rollback window, CI fixture conversion, and owner approval.
- [ ] Stacked PR closure/supersession.

### X-03 — Documentation

- [ ] User guide: capture vs replay vs export.
- [ ] Provider support matrix.
- [ ] `/usage` and toolbar docs.
- [ ] Operator rollback and live-test runbook.
