# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Fix a second round of code-review findings on the Connect work above:
  - Encrypted-reasoning detection could mislabel a response with a real
    visible summary as "withheld" whenever encrypted_content was also
    present alongside it (a normal OpenAI-style shape: a visible reasoning
    summary plus an opaque replay blob at the same time). Both the
    Responses-native and openai/azure Chat "reasoning_items" branches now
    require empty visible text before flagging withheld, matching the
    Anthropic `thinking` branch's existing rule.
  - `--stage catalogue` never used `fuzzy_match_models()` — the gateway-ID
    "did you mean" fallback only reached the interactive wizard, not this
    scripted/agent-facing JSON entry point. Exact/suffix matches that find
    nothing now also surface fuzzy suggestions under a `fuzzy_models` key;
    still never auto-selected, same as the wizard.
  - `--working-dir`'s existence check reported "does not exist" for a path
    that exists but is a file, whenever `~`-expansion was needed (a literal
    absolute path was already caught correctly by click's own check).
    Existence and directory-ness are now checked and reported separately.
  - `shadowing_source()` couldn't recognize a `--working-dir` save target as
    an effective registry layer, so it warned "still resolved from X" on
    every `--working-dir` save with a same-named alias defined elsewhere --
    even though that is `--working-dir`'s whole intended, correct use
    (`entries()`'s `extra_path` is always the highest-priority layer,
    matching what a later `nooa tui -w` read of the same directory would
    do). A plain `--output` path has no such guarantee and keeps the
    original, stricter behavior.
  - `--edit-model` combined with `--working-dir` could silently copy a
    different project's entry for the same alias name into the
    `--working-dir` target, because `entries()` only redirects to the
    target's own file when that file already exists -- otherwise the
    edited alias comes from the general (cwd/global) chain instead. Now
    prints a clear warning naming both files when this happens, instead of
    looking like an in-place edit of the target's own prior settings.
  - Ran `ruff format` on `test_connect_reply_budget.py`; the diff was
    previously unformatted.

- Fix a first round of code-review findings on the Connect work above:
  - Ran `ruff format` on `_connect_wizard.py`; the diff was previously unformatted.
  - `nooa connect`'s encrypted-reasoning detection now also recognizes the
    openai/azure Chat Completions shape (`part.native["reasoning_items"]`), a
    third wire dialect distinct from both the Anthropic `thinking_blocks`
    wrapper and the unwrapped Responses-style native — previously always read
    as "not encrypted" on that route.
  - `nooa connect --working-dir` combined with any `--stage` other than
    `save` used to crash with a confusing, generic usage error; it now
    raises a clear one explaining that only the full interactive run and
    `--stage save` write a registry file.
  - `nooa connect --working-dir ~` (a literal, unexpanded tilde) is no longer
    rejected as nonexistent; existence is now checked after expansion,
    not before.
  - `format_budget()` no longer mislabels a genuine, very large, explicit
    `--budget-tokens` value as "unlimited"; the threshold is now anchored to
    the actual sentinel with a fixed buffer for in-run spend, not a
    magnitude cutoff independent of it.
  - A level check whose response carried no `usage` object no longer
    silently vanishes from the end-of-run "Reasoning tokens · ..." summary
    when reasoning was genuinely observed; it now shows "reasoning observed"
    instead of being omitted. Levels where reasoning was never observed
    still stay out of the summary entirely, unchanged.
  - `_reasoning_tokens_summary()` (the summary line) now checks
    `reasoning_observed` the same way its sibling `_reasoning_tokens_label()`
    (the per-check row) already did, so the two can no longer disagree about
    whether a reasoning cost was actually measured for a level.
  - Removed the dead, misleading manual `status` computation in
    `CheckProgress.update()` that `check_status()` always overwrote anyway;
    kept only the `detail` text those branches actually control.
  - `docs/model-connect.md` no longer documents the old 131,072-token default
    `--budget-tokens` (now unlimited by default) or omits the "Model maximum"
    reply-budget choice.
  - De-duplicated the identical `normalized()` id-matching closure between
    `match_models()` and `fuzzy_match_models()` into one shared function.
  - Removed a redundant `except (binascii.Error, ValueError)` — `binascii.Error`
    is already a `ValueError` subclass — and its now-unused import.

- `nooa connect`'s session check's replay/repeat turns now change one rule of
  the puzzle instead of asking the model to "verify the recorded result."
  Verifying a known answer from memory is itself trivial enough that a model
  with adaptive/content-dependent reasoning effort could skip reasoning on
  it, even right after reasoning on the original puzzle in the same session
  (observed live for `gpt-6-astra` via its Azure route: turn 1 reasoned,
  turns 2-3 did not). Forcing a genuine re-solve fixed it — live-verified
  `reasoning_observed_by_turn: [True, True, True]` for both `gpt-6-astra`
  and Claude Opus 5, cache reuse unaffected.

- `nooa connect`'s session check now reuses the reasoning-level puzzle for its
  conversation task instead of trivial arithmetic. The old task was easy
  enough that a model with adaptive/content-dependent reasoning effort could
  legitimately skip reasoning on it even at a real reasoning level — a false
  "reasoning not retained" result indistinguishable from an actual replay
  bug (observed live for `gpt-6-astra`; live-verified fixed for Claude Opus
  5 and Qwen, both now showing reasoning on every turn). As with level
  checks, a wrong answer is not graded, only whether reasoning was observed.

- Fix `nooa connect` always reporting `reasoning_encrypted: false` on
  Responses-API routes (e.g. `gpt-6-astra`) even when the provider genuinely
  returned encrypted reasoning. Detection only recognized the Chat-style
  native shape (`part.native["thinking_blocks"]`); Responses routes store the
  raw output item directly on `.native` (`{"type": "reasoning",
  "encrypted_content": "..."}`), which was never checked. Both shapes are now
  detected.

- `nooa connect`'s session check now references trains by short hex id
  instead of zero-padded decimal record/item numbers. Live testing against
  Claude Opus 5 via Bedrock found the original wording's `finish_reason:
  "error"` on the replay turn was a real, provider-side content-filter
  false-positive (confirmed by replaying the identical request outside
  Connect: same request, ~50% pass rate) rather than a request-construction
  bug. Removing the digit-run-shaped ids roughly halved the observed failure
  rate (about 1 in 6-7 live runs) but did not eliminate it; a retry on
  `content_filter` would be needed to fully close this out.

- `nooa connect`'s reasoning-level rows now show a character count for real,
  visible reasoning text when litellm never attempted a token-count estimate
  for it (observed live for Qwen and DeepSeek routes; litellm's text-length
  estimate only exists for Anthropic/Bedrock). Never the text itself, only
  its length, matching the existing encrypted-blob byte-size treatment.

- A wrong answer on `nooa connect`'s reasoning-level puzzle no longer flags
  that check for attention. The puzzle exists to elicit reasoning, not to
  prove the model can solve it; only a missing reasoning signal (the thing
  the check actually verifies) still does.

- Fix reasoning-text-withheld detection missing the dialect actually observed
  live for Claude Sonnet 5/Opus 5 via Azure or Bedrock: a normal *signed*
  `thinking` block whose visible text is empty, not Anthropic's distinct
  `redacted_thinking` block type the original detection only checked for.
  Both are now detected and shown as "reasoning text withheld by the
  provider"; only the genuine `redacted_thinking` case (which carries an
  opaque data blob) reports a byte size, since a signature's length doesn't
  scale with how much was thought.

- `nooa connect`'s encrypted-reasoning-bundle message now includes the
  decoded byte size of Anthropic's `redacted_thinking` blob when available
  ("~N bytes of encrypted state"), a rough size signal since there is no way
  to convert an opaque encrypted payload into an actual token count.

- `nooa connect` no longer shows a misleading "0 reasoning tokens" for
  providers whose reasoning-token estimate is a text-length count of a
  deliberately-empty reasoning field (Claude Sonnet 5/Opus 5 via Azure or
  Bedrock). Detects Anthropic's `redacted_thinking` block specifically —
  real reasoning occurred; the provider withholds the text — and shows
  "encrypted reasoning bundle returned (N output tokens, not split out)".
  Any other case where reasoning was observed but not separately counted
  falls back to showing `output_tokens` (which does include the reasoning
  cost, just not broken out) instead of a bare, misleading 0. A real,
  positive reasoning-token count from the endpoint still displays as before.
- `nooa connect`'s "Ran out of reply tokens before finishing" message now
  adds "if you plan to use this reasoning level, increase the reply budget"
  for level checks specifically.

- `nooa connect` now says "Ran out of reply tokens before finishing" for a
  check whose reply was truncated by the reply cap (`finish_reason: length`),
  distinct from the generic "Reply incomplete" message still used for a
  provider error or content filter.
- Fix `nooa connect` reporting no reasoning observed for providers that
  return a reasoning part with a signature but deliberately empty text
  (Claude Sonnet 5/Opus 5 via Azure or Bedrock). The level-check probe was
  reading `response.reasoning`, which joins only non-empty parts, instead of
  checking for the part's presence the way session checks already do.
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
