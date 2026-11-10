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

## Follow-up review triage

The 382-test run above was a selected suite, not the full CI suite. The follow-up
review exposed missing fake-Relay tests and additional real runtime regressions.

| Finding | Verification and resolution |
| --- | --- |
| Red CI / old test doubles | Reproduced all 13 failures; updated tool keywords, effective-client fixture and fake-module cleanup. |
| Scope cleanup masks results/errors | Reproduced; guarded manual push/pop/reset/binding cleanup preserves the NOOA outcome. |
| Collection-valued dataclass serialization | Reproduced for set, frozenset, range and deque; display projection now uses sanitized native JSON after execution. |
| Legacy redactors bypassed | Reproduced for response and scope-end sanitizers; retain legacy payloads through both, then project only their sanitized output. Omitted data and metadata stay omitted. |
| Duplicate reasoning / missing migration entry | Reasoning is moved after redaction, not copied. CHANGELOG and migration documentation updated. |
| Live-test credential routing | Reproduced with sentinel credentials; existing keys are bound to their fixed trusted endpoint, rejecting custom overrides before client construction. |
| Unbounded/opaque/empty displays | Cap human text at 4096 characters plus marker; summarize pickle; preserve omitted/empty results instead of displaying null. |
| Empty tool-call ID | Normalize to None; real native execution-context regression added. |
| Cache hit called a guardrail rejection | Real short-circuit tests now get an explicit unsupported replacement/cache result error. No cache replay feature added. |
| Nested scope-local policy loss | Reproduced; native callback tasks are logical method continuations, while genuine fanout still isolates. |
| Exception text in scope metadata | Reproduced; function/run scopes record status codes without copying exception strings. |
| Concurrent roots / overlapping installs | Reproduced stack corruption and duplicate events; isolate root stacks and reference-count one installation per EventManager. |
| Missing behavioral coverage | Added 20 real-Relay regressions plus six off-network credential-routing cases; flush event queues before loop teardown. |
| Dependency cutoff/runtime check | The narrow release exception was explicitly approved in the upgrade plan. Clarified the frozen cutoff is not a rolling cooldown; supported versions remain constrained by the package extra. No redundant runtime version parser added. |
| ATIF attribution claims | Narrowed documentation; provider IDs improve correlation but do not eliminate Relay's documented upstream ordering/ownership limitations. |

Follow-up verification: 212 targeted tests pass, including the formerly missed
unit files. Real concurrent model workflows and the diagnostics quickstart pass;
all three fresh exports pass the current official Harbor validator. The complete
non-live CI-equivalent run passed 13,720 tests (five skipped, three xfailed);
the final overlapping-installation regression also passes in the targeted run.
The independent reviewer approved the fixes after reproducing the original
issues and independently passing 73 targeted tests. Final live verification
passed in 19.73 seconds. GitHub CI must complete before reporting this follow-up
as finished.

## Re-review: policy boundaries and native lifecycle

The previous triage's runtime projection and automatic isolation solutions are
superseded by the fixes below. Reproductions used installed native Relay 0.11.1,
not substitute Relay implementations.

| Current finding | Confirmed resolution |
| --- | --- |
| Cancellation leaves scopes open | Wait for this call's own Python callback cleanup before enclosing scopes pop, including repeated cancellation; no global flush or unrelated-call wait. |
| Datetime, oversized integers, lone surrogates abort completed calls | Convert inspection copies to Rust-compatible JSON; original caller values remain intact. Add real-Relay regressions. |
| LLM/tool exception metadata exposes text | Native callbacks receive fixed generic failure text; original exceptions/cancellation still reach NOOA callers. |
| Missing runtime version check | Install and all three directly callable middleware functions reject unsupported versions before execution. |
| ATIF prefill misattribution | Keep upstream limitation explicit. Export helper repairs only uniquely identified provider calls/observations and their execution metadata; host prefills remain upstream evidence. |
| Root isolation silently drops enclosing local policies | Default roots preserve the ambient stack and policies. Unsafe concurrent sharing fails before work; explicit isolate=True opts into a documented independent policy boundary. |
| Concurrent projector deregistration leaks | Remove runtime projectors entirely, including their registration/deregistration lifecycle. |
| Unflushed exit crashes | Baseline reproduced 7 native crashes in 60 subprocesses; final source passes 100/100 unflushed exits. Add process-teardown regression coverage. |
| Maximum-priority tie bypasses legacy redactor | Keep legacy payload shapes throughout all runtime sanitizers; format only final sanitized exports. A late maximum-priority redactor is covered against real Relay. |
| Quickstart mislabels pickle | Inspect raw encoded event values; opaque pickle is correctly identified. |
| Anonymous cleanup warnings | Identify cleanup phase and scope name/UUID or stack identifier without copying exception text. |
| Predictable diagnostics path | Use private unpredictable temporary directories and owner-only files for every example artifact. |
| Stale comments and untested behavior | Update migration/changelog/comments; add native policy inheritance, fail-closed concurrency, JSON bridge, exception privacy, repeated cancellation and shutdown cases. |
| Unreadable/duplicated ATIF results | Bound exported human text, remove raw duplicate tool values and summarize opaque pickle. Preserve sanitizer omissions. |
| ATIF schema edge cases | Preserve malformed provider arguments as evidence while emitting an argument object; retain native ATIF-v1.7 rather than relabeling as v1.8. |
| Previously fixed CI/credentials/cache/install issues | Retain fixes and include their tests in targeted/full validation. |

Independent lifecycle and boundary audits reproduced the reported failures;
a third agent reviewed the entire final change and approved it. No reviewer
found a remaining blocking issue. Native prefill ownership remains an upstream
limitation, documented rather than inferred from tool names or completion order.
