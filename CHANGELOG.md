# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- `nooa connect`'s reasoning-level puzzle checks now end with a one-line
  "Reasoning tokens · max: N · high: N · low: N (wrong)" summary, so a
  cross-level comparison doesn't require scrolling back through the run.
- `nooa connect`'s catalogue lookup now offers a fuzzy "did you mean" picklist
  when no exact/suffix match is found — common for gateway-routed model IDs
  (`aws/anthropic/bedrock-claude-opus-5`) whose routing prefix the catalogue
  never records. Never auto-selects a guess; only offered interactively.
- `nooa connect`'s reasoning-level puzzle check now shows the actual
  reasoning token count alongside the correct/incorrect result, so a wrong
  answer can be told apart from reasoning effort having no real effect.
- `nooa connect`'s reply-budget dropdown now truncates `high`/`extended`
  presets against the model's actual declared max output, and adds an
  explicit "Model maximum" choice showing that number, so picking exactly
  the ceiling never requires `--custom`.
- `nooa connect`'s `--budget-tokens` for API checks now defaults to
  unlimited instead of a fixed 131,072-token cap; it's only capped when the
  flag is passed explicitly. Previously the default could cause "some
  checks will be skipped" even with no explicit budget set.
- Add `nooa connect --working-dir`/`-w`: save to a project's own registry
  (`<working-dir>/.nooa/llm_config.yaml`) instead of the user-global one —
  the same file `nooa tui -w <working-dir>` reads. Mutually exclusive with
  `--output`.
- Fix `nooa connect` claiming "Using saved key variable X for this endpoint"
  and then immediately prompting for that same key. The registry only
  remembers which variable *name* an endpoint used last time, not whether a
  value is currently set; the message now says so honestly when the value is
  missing, instead of contradicting the prompt that follows it.
- Add `nooa connect`: a model-setup wizard, staged JSON interface and reusable
  `nooa.unifiedllm.connect` library. Prompts remain in `nooa-cli`, without new
  core dependencies. Configured checks send the saved reply limit, including
  reasoning-level overrides; insufficient budget skips checks instead of lowering
  caps. Save-time validation rejects caps that leave no input room. New entries
  default to `transport: direct`, forward-compatible with the direct SDK runtime;
  older runtimes ignore it and explicit existing transport choices are preserved.
  Registry writes follow symlinks and retain mode/newlines. Diagnostic handoffs
  scrub active keys throughout reports, including model names and mapping keys.
- Add the dedicated `nooa-model-configuration` coding-agent skill and route
  Connect help and diagnostic handoffs to it. Agent authoring links to model
  setup instead of embedding registry and reasoning-configuration instructions.
- Connect prefers endpoint-reported limits to catalogue values, labels input-only
  context bounds, and lets staged/offline plans reuse discovery JSON. Cache checks
  consider both continuations without treating provider misses as setup failures.
  Add puzzle-result feedback and a user-selected 120-second routing retry within
  the original approved budget.
- `ShellTools.run_stream` and the coding activity wrapper now accept
  `command, *, stdin=None, timeout=30.0`, matching `run`. Streaming uses the
  same stdin handling; pass an existing positional timeout as `timeout=...`.
- Context status, percentages, automatic summarization and overflow recovery now
  reserve the selected UnifiedLLM client's effective reply cap, including
  reasoning levels and per-call overrides. Unknown caps use a labelled planning
  reserve. Automatic summaries trigger at 80% of the usable input window;
  explicit summary thresholds remain fixed across model switches. Responses
  requests translate reply-cap aliases to `max_output_tokens`, and cap overrides
  replace inherited aliases rather than sending conflicting limits.
  Context management rejects a configured reply cap at or above the known
  context window instead of repeatedly summarizing against a one-token budget.
- Restore legacy Todo notes and statuses through the stored-session deserializer,
  and retain completed worker results when a delegated Todo disappears.
  Cleanup handles child-task re-entry and continues after a callback is cancelled,
  without cancelling unrelated callers.
- Keep CodeAct call correlation IDs separate from task display tags, report live
  input types after reassignment, and correct V2 tool/delegation hints. Benchmark
  working-directory context is untraced; failed trajectory exports no longer
  reuse a previous task's metrics. No-ID trace attribution requires matching code
  before selecting a later LLM turn.
- `self.events.collapse()` accepts integer endpoints that identify existing events,
  including mixed string/integer ranges over prior summaries, without warnings.
  Invalid numeric endpoints leave history unchanged.
- Rename the benchmark `TaskResult.command_to_verify` field to `how_to_verify`
  ("How to Verify"): concrete verification steps and expected results, not
  necessarily a shell command. Result JSON and runner answers use the new field.
- Reject ambiguous `ShellTools.replace(match, old, new)` calls before file access,
  with guidance for full-region versus path-based substring replacement.
- Add `CodeActV2`, the single-`python_cell` strategy with in-cell `return_result`.
  The benchmark agents use it; `CodeActStrategy` remains the default.
  Its cacheable Python-cell context includes the execution namespace's typed
  stub without a second execution-context block. Names and runtime helpers use
  one Python-style block; internal delegation errors are not advertised there.
  Benchmark agents omit the automatic `python_cell_state` inventory block.
- Trace explorer viewer requests now send configured viewer authentication and
  honor proxy environment settings, including `NO_PROXY` for direct access.
  Authenticated HTTP requests warn that bearer tokens are unencrypted; existing
  HTTP viewer/exporter setups remain supported. Use HTTPS or a trusted local
  connection/tunnel. The warning includes neither the token nor the URL.
- `CurrentCall` is a mutable invocation record; strategies bind its task tag and
  live execution namespace with ordinary public-field assignment during setup.
- Breaking: remove `CodeActLiteStrategy` and its experimental exports. Use
  `CodeActStrategy` for the existing two-tool contract or `CodeActV2` for the
  single-tool contract. The evaluation CLI option is now `codeact_v2`.
- Preserve inline completion values in replay and archived events in benchmark
  trajectories; make Todo updates/restores atomic and delegation merge failures
  recoverable. Behavior reports use schema version 2; regenerate older reports
  from their trajectories before comparing results.
- Benchmark agents release resources through `aclose()` as well as `close()`.
  Supplied delegation context uses ordinary method-argument formatting, without
  a custom renderer or redaction policy. Todo metadata
  and comment read-back methods are now included in model-facing documentation.
  Cancellation during shutdown is propagated only after background cleanup drains.
- `CodeActStrategy` remains the default strategy, but its model-facing behavior
  changes: revised delegation guidance, validated inline completion values in
  PythonOutput (None on validation failure), no replay of synthetic inline-return
  tool pairs, and explicit error/retry feedback for non-object tool arguments.
- Todo snapshot upgrades preserve legacy tasks, but downgrading to the previous
  implementation silently loses descriptions, active-task selection and comment
  IDs. Back up sessions before downgrading. `TodoVars` is now an alias for
  `PersistentVars`; helper-name keys must use explicit `get`/`set` access, and
  private/helper attribute writes are rejected. `InteractiveAgent.v` retains its
  separate `AgentVars` implementation.
- Delegation merge conflicts raise `DelegationMergeError` carrying the completed
  result and worker state. Benchmark agents no longer pre-seed a planning Todo;
  they expose tools through `python_cell_tools`, retain concise `context_usage`
  status and compaction guidance, and recreate the shell for each evaluation's
  working directory.

- Responses clients now honor the cached renderer's stable-prefix boundary by default,
  without a cache setting in the model registry. Requests without a usable boundary
  retain provider-default caching; `cache_breakpoint=None` opts out of NOOA markers.

- Security: the sandbox parent no longer unpickles worker bytes. Brokered `self.*`
  arguments, `self.x = value` assignments, cell return values and `return_result`
  payloads now cross as msgpack; rich values are rebuilt only from a fixed set of
  value types, numpy arrays, and the agent's declared pydantic models / dataclasses
  / enums (validated on the way in). Anything else is a `CellSerializationError`
  instead of code running in the parent. Adds the `msgpack` dependency.
- Breaking: custom CodeAct error formatters must implement
  `format(error, code=None, *, line_offset=0, max_error=None, tail_chars=None)`.
  Reduced legacy signatures are no longer supported.
- Breaking: sandboxed user-code failures are exposed as `SandboxExecutionError`;
  inspect `original_type`, `original_error`, and `diagnostic` for worker-side details.
- Add composable, context-scoped instrumentation hooks and trace-session scopes so hosts can observe NOOA execution without replacing native tracing.
- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
- Fixed: generator agent methods (`def`/`async def` containing `yield`) are now
  traced correctly. Their span previously covered only the *creation* of the
  generator, so LLM calls made by the body were recorded as children of whichever
  method drained it. Body calls now nest under the generator, and calls the
  consumer makes between yields do not.
- Breaking: a generator method with the `...` generation marker (including
  `yield ...`) now raises `TypeError` at class-creation time. Generation
  strategies commit one final result and do not define a stream protocol.
  Deterministic generators remain supported without `@strategy`.
