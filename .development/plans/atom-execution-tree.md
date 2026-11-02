# Opt-in Atom execution trees (replacement for #347)

Target dev/atom, starting at 15330d92. Do not carry the old shared-runner stack.

## Architecture

- Add --execution-tree to `nooa atom`, default false, passed into AtomACPAgent and SessionRegistry in worker and single-process paths. Router workers re-execute the original argv; ensure programmatic fixtures also forward the policy. HTTP follows the same argv-based launch contract.
- Add a session-owned, ACP-independent execution observer, installed via Session.add_loop_context_hook and composed with existing instrumentation hooks. Expose method lifecycle observations as typed data-only session updates; existing bridges subscribe, so detach/reattach does not own instrumentation.
- Each root turn has a fresh run UUID; children linked to a captured creator keep its causal run across their continuations. A separate session/turn token scopes cancellation, so causal grouping never defines cancellation scope. Suspended generators and already-started background tasks keep their owning run. For children created/opened from an active observed method, capture an explicit parent session/span relationship and causal run; do not copy the parent's entire Context. Children loaded without a calling method are roots. Document run/turn semantics and no durable historical tree replay.
- Correlate Python/file/terminal events in the originating context before crossing threads, emit immutable snapshots, and serialize observer state/subscription changes and session delivery with the owner loop (a lock may protect synchronous capture). No client/ACP types in the observer.
- Include version/run/span/parent/type/name/optional agent/timestamps in _meta['nooa.dev/execution']; method arguments, return values and agent context remain absent. Existing ACP status and content carry completion/errors.
- Bridge translates logical method/tool keys into its actual wire IDs consistently, including child mirroring; no duplicated Python/file/terminal cards. Existing concise rolling IDs cannot represent a detailed tree; detailed tree is emitted only in verbose mode. Each attachment tracks emitted starts. On mid-run attach or verbosity re-enable, unknown parent references become roots and unmatched completions are suppressed (no historical backfill). Turning verbosity down must finalize visible method cards without cancelling session work; silent and concise retain their existing behavior.
- Scoped subscriptions observe transient nested Agent event managers only for active method lifetimes; release after final completion (including errors, cancellation and generator close). Keep root subscription existing. All callbacks reject the wrong session context; never retain completed transient agents for a session lifetime. Session shutdown closes all remaining observations/subscriptions.
- Completion cleanup is idempotent and run-scoped. Cancelling a parent turn must not fail a suspended method or independently running child from a different run/session. Existing method after hooks normally finalize their own nodes; exceptional cleanup acts only on that turn.

## Files and tests

1. session/execution.py: observer and immutable execution payloads; items.py: typed update/correlation fields; session.py and registry.py: installation/lifetime.
2. acp/event_bridge.py: method rendering + correlate/decorate existing cards, source/wire ID mapping, verbosity and cancellation. acp/server.py and cli.py: policy wiring through roots/workers and children.
3. Tests fail before implementing new behavior: default-off, nested methods + handled failure + real file/terminal, two prompts, repeated provider IDs, parallel child sessions, cancellation/reuse, suspended generators across turns, emitter thread, subscription collection, tracing composition, verbosity, attach/detach, and real stdio router/worker metadata round trip. Wait explicitly for terminal notifications in protocol tests.
4. Port the scripted Aion demo to Atom/Done, update experiment/package docs and launch commands. Reuse the already validated Aion renderer contract; test fixture captured stream against renderer and Pool. Full AionCore/browser smoke if runnable; explicitly record any unavailable integration rather than implying it passed.
5. Run targeted Atom/tracing tests and lint; broaden to Atom package + core turn/hook suites. Record commands/results. Independent implementation review and fixes, then signed-off commit and draft replacement PR targeting dev/atom through repository's OE publication policy. Do not close/comment on original #347 without explicit instruction.

## Non-goals

No change to default Pool display, no frontend fork edits, no protocol standardization, no persistent execution-store schema, no revival of #330 or unrelated #337 code.


## Validation and review outcome

Plan review approved after separating causal run ids from cancellation tokens
and defining attachment/verbosity behavior. Implementation review approved the
session-owned design after fixes for uncommitted cancellation, active ancestry
eviction, and unprintable exceptions. A final deterministic worker-thread
start/failure regression exposed delivery overtaking; the owner-loop FIFO now
drains older emitter-context snapshots before lifecycle transitions. The final
session/bridge/cancellation gate passed 207 tests, and the real protocol/hook/
child-tracing gate passed 27 tests. The scripted router/worker demo completed
four turns with the feature off/on; the pinned Aion component compatibility
suite passed against the actual producer stream. Pool was tested with the real
demo off/on. Full AionCore/browser testing was not rerun.

Broad Atom/runtime-hook/tracing checks were partitioned after correcting new
fixtures: 2,462 passed, 6 skipped, 1 xfailed, and three baseline failures. All
three reproduce from a pristine archive of dev/atom 15330d92: repository
instruction symlinks in a symlinked workspace, burst-progress wake timing, and
skill search matching the word 'review' in an MCP command's filesystem path.
The optional running viewer's exporter was isolated for offline network tests
using OTLP_ENDPOINT=http://127.0.0.1:1/v1/traces; runtime/tracing tests remain live.
No baseline fixes are included. Ruff lint/format, SPDX headers, focused pyright,
and diff whitespace checks pass. New behavior is disabled by default.


The staged-file Pyright hook also checks older declarations in touched modules.
Constructor-preserving display decorators, explicit Pydantic defaults, dynamic
hook casts and test-double narrowing now make the entire staged Python set
pass with zero type errors; these adjustments preserve runtime model values.
The session/model/port/child/doc regression gate passed 165 tests after them.
