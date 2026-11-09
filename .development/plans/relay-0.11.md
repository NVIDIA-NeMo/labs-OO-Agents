# Relay 0.11 integration update

## Scope

Upgrade the optional integration from Relay 0.6 to >=0.11.1,<0.12. Preserve
NOOA execution, response identity/private replay state, tool policies and tracing
architecture. Do not add caching, routing, capability interception or a daemon.

## Implementation

1. Update pyproject and uv.lock with a package-specific exclude-newer exception;
   avoid unrelated dependency upgrades.
2. Return Relay ToolExecutionResult from tool callbacks. Update LLM sanitizers
   and asynchronous flush calls in tests/quickstart. Read effective model identity
   from ctx.client, falling back to the agent only for contexts without a client.
3. Replace private scope ContextVar access with public use_scope_stack and
   create_scope_stack_from_propagation/capture_propagation_context. Keep the
   same-task nesting behavior; isolate sibling tasks and restore context on
   errors/cancellation. Document that ancestor scope-local registrations are not
   inherited by isolated tasks.
4. Audit ATIF emitted by real Relay 0.11.1. Determine its latest supported ATIF
   schema and validate exports against the authoritative schema. Fix adapter
   projection/annotations if necessary so messages, model IDs, token usage,
   tool-call IDs and observations are represented correctly. Do not expose
   opaque provider replay state. Preserve the native NOOA response/result.
5. Update quickstart with optional diagnostics/resource-metrics configuration,
   application-owned plugin activation and clean async shutdown. Keep normal
   execution simple and the optional features disabled by default.

## Verification

Plan approved by the independent reviewer. Explicit tool-call IDs and useful
ATIF observation content are acceptance requirements, not just nonempty exports.

- First add and run regressions for effective client identity and ATIF semantic
  correctness against the unchanged adapter; existing compatibility tests already
  reproduce 15 failures on 0.11.1 (43/43 pass on 0.6).
- Run targeted Relay tests, concurrent/cancellation cleanup, native replay privacy
  checks, and relevant middleware/tracing regressions; run lint/format checks.
- Run live end-to-end NOOA generation through configured model credentials,
  covering CodeAct tool execution, Predict structured output, nested/concurrent
  methods and effective-client override. Inspect/validate generated ATIF, not just
  mock callback outputs. Include a live run of the optional diagnostics/metrics
  example. Keep live artifacts and logs under /tmp, out of the PR.
- An independent subagent reviews the whole diff and validation evidence. Fix
  findings and repeat review until both implementer and reviewer approve.
- Commit only intended files, preserving pre-existing LFS worktree artifacts;
  follow this repository's oe publishing mechanism for a draft PR.

## Implementation and validation findings

- Relay emits ATIF-v1.7; current Harbor accepts v1.7 and v1.8 (audio extension).
  No version relabeling or custom exporter was introduced.
- Fixed a pre-existing ATIF projection error: `message` was not recognized by
  Relay's generic extractor. `assistant_message` now yields actual answer text,
  tool calls and public reasoning without provider-native state.
- Tool results include readable content, structured encoded data, error status,
  explicit provider-issued IDs and a non-replayable flag for Python cells.
- Original-adapter regression reproduction: four new model/ATIF tests fail.
  Updated adapter: 382 middleware/tracing tests pass; Ruff lint/format pass.
- Live quickstart with diagnostics and CPU/memory polling completed using
  existing internal gateway credentials. Public NIM credential failed with 403;
  no saved configuration or credentials were changed.
- Live concurrent workflows completed with real typed Predict and multi-turn
  CodeAct calls, method model overrides, six model calls and six observations.
  Research, return-type and concurrent ATIF artifacts passed the current
  authoritative Harbor validator.
- Independent review identified an upstream limitation: synthetic prefill cells
  can attach to a preceding unrelated model step. Documented it and separated
  prefills from provider-issued calls in live assertions. Correcting Relay's
  assembly algorithm is outside this minimal update.
- Review also found ineffective in-place header mutation in the existing
  example. It now copies the request and describes telemetry-only behavior.
- Final independent implementation review: APPROVE, no required changes. The
  reviewer independently ran 47 targeted tests and validated all three final
  live artifacts. Final broader regressions: 382 passed. Strengthened live test:
  passed in 22.58 seconds, with two synthetic prefills reported separately.
