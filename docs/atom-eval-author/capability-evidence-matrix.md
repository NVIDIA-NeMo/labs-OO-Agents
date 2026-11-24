<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ATOM capability evidence matrix

Read the [scope, tool inventory and source index](SOURCES.md) first. This matrix targets
NOOA `107a65d77e4c2c644bb90553ac404bb4ed8a064c`, refreshed on 2026-11-16; it does not
describe a measured benchmark run. C01–C30 retain their identities; C31 adds the
separate session-reflection mechanism. These 31 capability IDs are stable references for
later authoring. The [Ethos draft](ETHOS.md) captures the intended behavior that can be
derived from these sources; unresolved product intent remains explicit.

## Evidence and failure labels

- **P:** explicit prompt/role expectation. Instructions alone do not enforce it.
- **D:** documented contract or limitation.
- **I:** implementation supports or enforces the specified mechanism.
- **T:** relevant test source/assertions inspected; execution is recorded separately in
  [validation](SOURCES.md#validation-of-this-documentation-pass). T alone does not mean
  the test was run or that model behavior was demonstrated. Tests may use fake/scripted
  models or a related framework/derived agent; see
  [test-fixture scope](SOURCES.md#test-fixture-scope).
- **Proposed evaluation check:** an authoring suggestion, not an existing task or
  evidence of model success. These checks address what source inspection cannot
  establish.

"Tested failure path" means regression coverage exists in source; it does not mean an
open defect or a measured failure rate. "Documented limit" is an intentional or
acknowledged boundary. Prompt-only behavioral risks are identified as unmeasured. The
list is scoped to evidence inspected here, not a complete issue-tracker audit.

For generated evaluations, score the agent's observable choices and task outcomes.
Check runtime mechanics such as retry-budget enforcement, OAuth callback freshness,
session locks and process cleanup with deterministic tests before evaluating the agent.
Those harness prerequisites help distinguish infrastructure failures from agent
failures; they are not model capabilities to score. C26 is explicitly a harness
prerequisite.

## Capabilities

| ID | Capability |
|---|---|
| [C01](#c01) | Discover and use the current API |
| [C02](#c02) | Read instructions and preserve unrelated work |
| [C03](#c03) | Find definitions and references |
| [C04](#c04) | Read and edit files accurately |
| [C05](#c05) | Execute commands with persistent shell state |
| [C06](#c06) | Handle command deadlines and cancellation |
| [C07](#c07) | Manage background work and resource requests |
| [C08](#c08) | Use timers and bounded progress |
| [C09](#c09) | Delegate bounded tasks and verify results |
| [C10](#c10) | Continue kept-child conversations |
| [C11](#c11) | Recover an awaited child at a cell deadline |
| [C12](#c12) | Maintain plans and durable item notes |
| [C13](#c13) | Reuse Python state across live turns |
| [C14](#c14) | Preserve argument/object identity and Python semantics |
| [C15](#c15) | Distinguish live state from restart persistence |
| [C16](#c16) | Use installed libraries in the correct environment |
| [C17](#c17) | Recover from actionable API/runtime errors |
| [C18](#c18) | Handle large and truncated output |
| [C19](#c19) | Inspect and manage context/history |
| [C20](#c20) | Discover and activate the right skill |
| [C21](#c21) | Use MCP with correct configuration and auth state |
| [C22](#c22) | Write reusable skills and helper calls |
| [C23](#c23) | Save workspace preferences with correct scope |
| [C24](#c24) | Complete, ask or wait according to role |
| [C25](#c25) | Verify outcomes before asserting success |
| [C26](#c26) | Respect session input and cancellation lifecycle |
| [C27](#c27) | Run headlessly with attributable results |
| [C28](#c28) | Use media only where supported |
| [C29](#c29) | Perform optional bounded post-Done maintenance |
| [C30](#c30) | Interpret diagnostics and trace metrics correctly |
| [C31](#c31) | Handle configured session housekeeping |

<a id="c01"></a>

### C01 — Discover and use the current API

**Support and tool mapping**

P/I/T; T01–T02, T14. Inspect `doc` and use `python_cell` rather than guessing flat
tools.

**Known failure path or boundary**

Tested rejection: old delegate/spawn calls. The default agent has no self.shell and does
not load nemo.shell; hidden implementation names are not advertised tools.

**Source/test evidence**

- [S01](SOURCES.md#s01)
- [S03](SOURCES.md#s03)
- [test_legacy_delegation_is_not_offered](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L448)
- [test_prompt_leaks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_prompt_leaks.py#L1)

**Proposed evaluation check**

Give an unfamiliar tool object; require discovery and a successful call. Check that
feedback corrects an initial wrong API guess.

<a id="c02"></a>

### C02 — Read instructions and preserve unrelated work

**Support and tool mapping**

P/I; T03–T04. Read repository instructions/relevant code before making focused edits;
follow current steering.

**Known failure path or boundary**

Prompt boundary: repository instructions cannot expand authorization. Root-scoped search
is not filesystem isolation; external paths are technically accepted.

**Source/test evidence**

- [S14](SOURCES.md#s14)
- [test_refs_outside_the_root_does_not_raise](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_repo_tools_refs.py#L51)

**Proposed evaluation check**

Seed unrelated modifications and scoped instructions; verify the requested change and
unchanged unrelated files. Model compliance remains unmeasured.

<a id="c03"></a>

### C03 — Find definitions and references

**Support and tool mapping**

I/T; T04. Search the intended workspace/session filesystem and consume
diagnostics/truncation flags.

**Known failure path or boundary**

Tested failure path: unreadable or disappeared files; wrong host filesystem; parser/text
fallback and result caps can omit evidence.

**Source/test evidence**

- [S04](SOURCES.md#s04)
- [test_session_file_failure_is_reported_without_unpaired_symbols](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_repo_tools_session_paths.py#L148)

**Proposed evaluation check**

Use a multi-file repository with shadowed symbols and session-only files; verify
selected locations and edits, including a capped first search.

<a id="c04"></a>

### C04 — Read and edit files accurately

**Support and tool mapping**

P/I/T; T03. Use read ranges, unique text replacement, Match anchors or full writes as
appropriate.

**Known failure path or boundary**

Guarded failure: missing/ambiguous replacement, stale anchor or oversized file (default
replacement cap 8 MiB). Anchor checks do not lock out concurrent writers. `FileWrite`
reports a write, not successful task validation.

**Source/test evidence**

- [S01](SOURCES.md#s01)
- [S04](SOURCES.md#s04)
- [test_file_write_documents_real_string_fields](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_doc_contracts.py#L40)
- [test_stale_region_never_writes](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/tools/test_shell_tools_stale_match.py#L30)

**Proposed evaluation check**

Include repeated snippets and an intervening edit; verify exact final content and safe
recovery from replacement rejection.

<a id="c05"></a>

### C05 — Execute commands with persistent shell state

**Support and tool mapping**

I/T; T05. Reuse one Shell for intended cwd/environment continuity; separate Shells
isolate state.

**Known failure path or boundary**

Tested failure path: shell exit/restart can lose state; missing retained cwd triggers
explicit fallback. No PTY/live input.

**Source/test evidence**

- [S05](SOURCES.md#s05)
- [test_run_reuses_the_objects_bash](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L422)
- [test_a_restarted_shell_whose_kept_cwd_is_gone_starts_where_it_started](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L656)

**Proposed evaluation check**

Set cwd/env, run later commands, and distinguish a normal nonzero exit from a
destroyed/restarted shell.

<a id="c06"></a>

### C06 — Handle command deadlines and cancellation

**Support and tool mapping**

I/T; T05–T07. Per-run deadline starts when the command starts; interpret timeout and
cancellation accurately. At a cell deadline an unfinished inline/queued run can hand
over to self.jobs with its advertised name.

**Known failure path or boundary**

Tested failure paths: blocked stdin, hung command, queued work, cell call behind a
background job. Timeout gives returncode 124 and timed_out; earlier background processes
may survive until close, depending on ownership. Cell timeout is distinct from command
timeout: successful handover keeps work running; user cancellation stops it. No
available manager means no handover. A handed-over busy Shell rejects later inline calls
with the owning job name.

**Source/test evidence**

- [S05](SOURCES.md#s05)
- [test_run_takes_a_timeout_per_call](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L775)
- [test_earlier_background_job_survives_timeout_and_cancel_until_close](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L1047)
- [shell deadline/cancel assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_deadline_detach.py#L252)

**Proposed evaluation check**

Timeout a command, verify the next command works, identify surviving owned work and
close it deliberately. Do not equate one command timeout with all-process teardown.
Separately force a cell deadline (including queued runs), collect/cancel the advertised
job without duplicating side effects, and verify user cancellation stops the command.

<a id="c07"></a>

### C07 — Manage background work and resource requests

**Support and tool mapping**

I/T; T06–T07. Launch, await/iterate or receive JobEvent; use request jobs for resource
inputs.

**Known failure path or boundary**

Tested misuse: awaiting/iterating a resource itself, duplicate/reserved names, wrong
launch value, too many in-flight requests. Cancelling a waiter is not necessarily
cancelling work.

**Source/test evidence**

- [S06](SOURCES.md#s06)
- [test_background_request_arrives_once_and_waiting_checks_targets](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L194)

**Proposed evaluation check**

Run concurrent distinguishable jobs and consume every intended result once through the
selected delivery route; check attribution and explicit cancellation.

<a id="c08"></a>

### C08 — Use timers and bounded progress

**Support and tool mapping**

I/T; T06–T07. Observe opt-in previews separately from final results; use after/every for
live work.

**Known failure path or boundary**

Documented bounds: quiet default; every coalesces unread ticks; previews can be omitted;
terminal results supersede unread previews. Timers do not survive restart.

**Source/test evidence**

- [S06](SOURCES.md#s06)
- [test_default_launch_is_quiet_iterated_awaited_and_detached](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_progress.py#L359)
- [test_final_supersedes_unread_progress_and_is_never_delayed](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_progress.py#L410)

**Proposed evaluation check**

Generate noisy output and timer ticks while doing other work; verify bounded previews
and final-result correctness without treating preview omission as task failure.

<a id="c09"></a>

### C09 — Delegate bounded tasks and verify results

**Support and tool mapping**

P/I/T; T08. Supply sufficient briefs, use TaskResult, independently check reported work;
one editing child at a time is prompted.

**Known failure path or boundary**

Tested failure path: child failure reaches parent. Prompt-only risks: incomplete brief,
accepting unsupported reports, concurrent conflicting edits.

**Source/test evidence**

- [S07](SOURCES.md#s07)
- [S02](SOURCES.md#s02)
- [test_one_shot_subagent_returns_its_task_result_and_notes_are_job_items](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L78)
- [test_a_failed_subagent_raises_in_the_cell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L222)

**Proposed evaluation check**

Delegate a change with acceptance checks; inject an incorrect child report and verify
the parent detects it. Runtime routing tests do not prove this judgment.

<a id="c10"></a>

### C10 — Continue kept-child conversations

**Support and tool mapping**

I/T; T07–T08. Distinguish TaskResult from Question; answer through send and treat
message as a note.

**Known failure path or boundary**

Tested misuse: reading Question.report or Subagent.report; request cancellation stops a
turn, resource cancellation closes the child.

**Source/test evidence**

- [S07](SOURCES.md#s07)
- [S15](SOURCES.md#s15)
- [test_kept_subagent_question_answer_and_follow_up](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L156)

**Proposed evaluation check**

Have a child ask for missing context, answer it, then send follow-up work. Verify
results belong to the correct request and cancellation preserves/closes the intended
scope.

<a id="c11"></a>

### C11 — Recover an awaited child at a cell deadline

**Support and tool mapping**

I/T; T06–T08. An inline awaited child may detach into a managed job; later wait/collect
or cancel it.

**Known failure path or boundary**

Explicit distinction: a cell deadline can leave the child running; ordinary cancellation
closes it. Reissuing work can duplicate side effects.

**Source/test evidence**

- [S07](SOURCES.md#s07)
- [test_a_child_awaited_at_the_deadline_runs_on_as_a_job_and_wakes_waiting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_deadline_detach.py#L93)

**Proposed evaluation check**

Force the parent cell deadline before child completion; require using the advertised job
to retrieve the original result without spawning duplicate work.

<a id="c12"></a>

### C12 — Maintain plans and durable item notes

**Support and tool mapping**

P/I/T; T09–T10. Activate current work, keep findings/checks on the item and resolve
exact IDs/titles.

**Known failure path or boundary**

Tested failure paths: ambiguous or stale titles, completing without a target, snapshot
migration. Status display is bounded and is not the complete store.

**Source/test evidence**

- [S08](SOURCES.md#s08)
- [test_ambiguous_titles_fail_without_side_effects](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_todo_list.py#L119)
- [test_snapshot_round_trip_keeps_ids_notes_and_the_active_item](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_todo_list.py#L402)

**Proposed evaluation check**

Use duplicate titles, rename an item, then resume; verify the correct item and stored
evidence remain associated without accidental retargeting.

<a id="c13"></a>

### C13 — Reuse Python state across live turns

**Support and tool mapping**

I/T; T02. Values, functions/classes and global rebindings survive Done/Waiting/NeedInput
within handle.

**Known failure path or boundary**

Documented limits: in-process only; other generation methods use fresh namespaces;
roots/children are distinct; writes are not transactional and del does not close
resources.

**Source/test evidence**

- [S12](SOURCES.md#s12)
- [test_kept_session_retains_names_across_done_and_need_input](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L35)
- [test_nested_generated_method_has_independent_owner_and_del](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_nested.py#L19)

**Proposed evaluation check**

Create a helper and mutable state, yield/end a turn, update the state and reuse the
helper. Check isolated child/nested-call state and partial effects after an error.

<a id="c14"></a>

### C14 — Preserve argument/object identity and Python semantics

**Support and tool mapping**

I/T at the framework/derived-agent level; T02, T18. Pass objects directly, preserve
aliases, closures/defaults and one-time definitions.

**Known failure path or boundary**

Evidence boundary: identity-specific test uses a derived ParameterCollisionAgent method;
it is not a live-model ATOM capability score or a cross-process reference guarantee.

**Source/test evidence**

- [S12](SOURCES.md#s12)
- [test_current_call_parameters_keep_identity_and_are_not_provider_advertised](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_review_repairs.py#L358)
- [scope/default semantics](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_execution_namespace.py#L97)

**Proposed evaluation check**

Use a large mutable object with aliases and helper calls; check identity/mutation
in-process and avoid serializing/reconstructing it. Treat this as a targeted evaluation
gap, not a demonstrated model ability.

<a id="c15"></a>

### C15 — Distinguish live state from restart persistence

**Support and tool mapping**

I/T; T09–T10. Store serializable durable facts in self.v/todo state; acknowledge restart
loss of live globals/jobs.

**Known failure path or boundary**

Documented limits: globals are not snapshots; running jobs become one lost failure
notice and are not relaunched; earlier children are not adopted as jobs. A cell-defined
class instance restores only while the class is resolvable; unavailable classes cause
affected self.v entries to be dropped with warnings, preserving other entries.

**Source/test evidence**

- [S13](SOURCES.md#s13)
- [test_namespace_nosnapshot_restart_notice_once_and_self_v_survives](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L462)
- [test_after_restore_reports_lost_jobs_once](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L692)
- [cell-class restore boundary](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_class_restore.py#L1)

**Proposed evaluation check**

Restart after storing facts and launching work; verify retained facts, one lost-job
notice and deliberate recovery rather than pretending the job completed. Compare
fresh-agent restore with a live class against simulated process loss; require
recognition of dropped values without claiming all stored facts vanished.

<a id="c16"></a>

### C16 — Use installed libraries in the correct environment

**Support and tool mapping**

P/I; T02, T05, T18. Use available runtime libraries or run task code in its own
environment through Shell.

**Known failure path or boundary**

Documented failure: ModuleNotFoundError in agent Python need not mean the task
environment lacks the package. Benchmark extra is opt-in; no blanket import guarantee.

**Source/test evidence**

- [S15](SOURCES.md#s15)
- [S20](SOURCES.md#s20)

**Proposed evaluation check**

Place a library only in the task environment, trigger the hint and verify successful
execution there. Include installed-library object reuse separately.

<a id="c17"></a>

### C17 — Recover from actionable API/runtime errors

**Support and tool mapping**

I/T; T01–T08, T14–T16. Use name inventory, documented result fields and namespaced MCP
calls after feedback.

**Known failure path or boundary**

Tested feedback paths: NameError, missing module, bare/wrong-server tool call, incorrect
result attributes, forbidden subprocess import. Emitting a hint is not model recovery.
Invalid return_result values permit ten retries after the first invalid value, failing
on the eleventh invalid value; this is a separate budget.

**Source/test evidence**

- [S15](SOURCES.md#s15)
- [S14](SOURCES.md#s14)
- [test_result_attribute_errors](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_result_attribute_errors.py#L1)
- [test_an_atom_cell_importing_subprocess_is_told_to_use_a_shell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_restriction_hints.py#L34)
- [retry limit](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_return_result_fixups.py#L166)

**Proposed evaluation check**

Inject a controlled first error; observe correction and task completion without repeated
identical misuse, or a truthful blocker when recovery is unavailable. Score hint
delivery separately from the agent's recovery and independently verified outcome.

<a id="c18"></a>

### C18 — Handle large and truncated output

**Support and tool mapping**

I/T; T02–T05. Distinguish bounded object formatting, irrecoverable cell print cuts, and
retained shell capture files.

**Known failure path or boundary**

Default limits (capture is configurable): cell stdout >50,000 chars / stderr >2,000 cut
without file; shell summaries >1 MiB use files, themselves bounded (default 64
MiB/stream) and removed with owner/pruning. Newly added snapshot/overflow libraries have
not replaced these ATOM capture paths at this revision.

**Source/test evidence**

- [S15](SOURCES.md#s15)
- [test_a_long_cell_print_is_cut_with_the_file_hint_and_no_file](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_print_cut.py#L15)
- [test_direct_run_keeps_long_output_until_aclose](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L114)

**Proposed evaluation check**

Hide needed evidence in omitted output; require rerunning safely to a file or reading
retained capture before its owner closes. Do not infer omitted values.

<a id="c19"></a>

### C19 — Inspect and manage context/history

**Support and tool mapping**

P/I/T; T12–T13. Query actual events, summarize completed work and retain relevant
facts/original references.

**Known failure path or boundary**

Documented failure modes: wrong event type/tag; protected block edits; confusing a
summary with complete original evidence. Automatic summarization is not perfect recall.

**Source/test evidence**

- [S11](SOURCES.md#s11)
- [test_context_usage_examples_run](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_advertised_examples.py#L129)

**Proposed evaluation check**

Build a long history with a late correction; compact it and require an answer grounded
in retained/original events, including recovery of an archived error.

<a id="c20"></a>

### C20 — Discover and activate the right skill

**Support and tool mapping**

I/T on the inherited skill manager and public built-in installation; T14–T15. Search
without importing, activate selected skills and inspect docs before use.

**Known failure path or boundary**

Tested failure paths: unavailable name, name conflicts, hidden-method exposure; text
instructions may be deferred to next turn. Built-ins install through the registry; saved
local repo/todo skills cannot replace those built-ins. Additional inventory is
deployment-dependent.

**Source/test evidence**

- [S09](SOURCES.md#s09)
- [S03](SOURCES.md#s03)
- [test_search_does_not_import_a_code_skill](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skill_manager.py#L167)
- [test_activating_a_text_skill_sends_its_doc_to_the_system_messages_channel](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skill_manager.py#L194)

**Proposed evaluation check**

Supply distractor skills and one useful unfamiliar skill; verify targeted discovery,
activation, instruction retrieval and successful invocation.

<a id="c21"></a>

### C21 — Use MCP with correct configuration and auth state

**Support and tool mapping**

I/T on the inherited manager/connection components; T16. Discover connected server
methods, honor approval/sign-in prerequisites and report missing access.

**Known failure path or boundary**

Explicit limits: deactivation leaves connections callable; configuration is not
authorization; wrong call form gets a hint. OAuth failure/recovery is not ordinary task
success. Repeated pasted OAuth flows must wait for fresh code/state; the previous
callback cannot complete the next flow.

**Source/test evidence**

- [S16](SOURCES.md#s16)
- [test_an_unapproved_mcp_server_asks_for_approval](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skill_manager.py#L241)
- [test_fingerprint_binds_the_workspace_scope](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_approval_fixes.py#L11)
- [repeated sign-in](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_signin_lifetime.py#L125)

**Proposed evaluation check**

Use a local fixture server with missing approval, stale configuration and a rejected
argument. Evaluate whether the agent reports missing access, respects approval
boundaries and corrects recoverable misuse before completing the task. Use no live
external service; callback freshness is covered by the linked deterministic tests.

<a id="c22"></a>

### C22 — Write reusable skills and helper calls

**Support and tool mapping**

I/T; T17–T18. Scaffold/edit/reload/test code or text skills; use typed generated
sub-calls where needed. Skill tests use the injected workspace shell without requiring
self.shell.

**Known failure path or boundary**

Tested failure paths: reserved/path-like names, old .nooa/libs layout and code/text
directory collision. Skill persistence is filesystem persistence, not retained Python
closures. Standalone @strategy helpers require source-verifiable ellipsis bodies; pass,
docstring-only or implemented bodies are rejected.

**Source/test evidence**

- [S10](SOURCES.md#s10)
- [test_a_code_skill_is_saved_in_nooa_skills_and_active_in_a_new_session](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_libs_persist.py#L70)
- [test_a_text_skill_is_a_skill_md_found_and_active_in_a_new_session](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_libs_persist.py#L95)
- [shared-runtime helper-source assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_strategy_helper_source.py#L1)

**Proposed evaluation check**

Create a small skill with tests, use it, reopen a session and use it again. Separately
test helper composition and validation of nested generation output. Reject a
missing-ellipsis helper, then define a valid helper and call it in a later cell; verify
model-call evidence rather than accepting a silent None.

<a id="c23"></a>

### C23 — Save workspace preferences with correct scope

**Support and tool mapping**

I/T; T19. Persist requested defaults while distinguishing current session, workspace and
higher-priority overrides.

**Known failure path or boundary**

Tested failure paths: literal credentials, leaking personal defaults into workspace
settings, save failure. Saved default model does not validate credentials or switch the
current model.

**Source/test evidence**

- [S17](SOURCES.md#s17)
- [test_remember_mcp_rejects_a_literal_secret_header](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_workspace_settings_mcp.py#L36)

**Proposed evaluation check**

Save a placeholder-based MCP definition and skill preference; inspect saved diff and a
new session. Verify live/session-local behavior is not misreported as changed.

<a id="c24"></a>

### C24 — Complete, ask or wait according to role

**Support and tool mapping**

P/I/T; T01, T11. Interactive answers go to user; batch returns TaskResult without
asking; Waiting names work that can deliver.

**Known failure path or boundary**

Tested rejection: waiting on the person/idle or nonexistent work, batch Done without
result, batch Done with running work. Interactive Done may leave background work; idle
resources may remain in batch.

**Source/test evidence**

- [S02](SOURCES.md#s02)
- [test_unattended_done_waits_for_running_work](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L412)
- [test_interactive_done_may_leave_background_work](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L429)

**Proposed evaluation check**

Run equivalent scenarios in each role; verify allowed completion, wait delivery and
blocked result semantics without treating Done as verifier success.

<a id="c25"></a>

### C25 — Verify outcomes before asserting success

**Support and tool mapping**

P; I/T only for result shape and transport. T03, T05, T08. Run relevant checks, verify
delegated work and report observed evidence/limits.

**Known failure path or boundary**

Unmeasured behavioral risk: fabricated test evidence or treating a returned TaskResult
as proof. Type validation does not verify truth.

**Source/test evidence**

- [S01](SOURCES.md#s01)
- [S02](SOURCES.md#s02)
- [S18](SOURCES.md#s18)

**Proposed evaluation check**

Seed a convincing but false success report and a failing acceptance check; require
truthful failure/revision and grader-observed final correctness.

<a id="c26"></a>

### C26 — Respect session input and cancellation lifecycle

**Support and tool mapping**

I/T host integration; T11 plus Session/ACP controls. Preserve input ownership and
outcomes through dispatch/cancel/close. The interactive role explicitly warns that a
running cell holds back new messages.

**Known failure path or boundary**

Documented limit: replay is not exactly-once side-effect execution; consumed Waiting
obligations/coroutines do not resume. Dispatch recording failures pause and require
explicit recovery. Shared runtime guards idle readiness mismatches and preserves caller
cancellation; owner probes must not take the session lock. ACP watchdog cleanup must
respect process-group ownership.

**Source/test evidence**

- [S21](SOURCES.md#s21)
- [test_preparation_cancel_keeps_admission_and_resolves_prompt](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_dispatch_invariants.py#L33)

**Harness prerequisite (deterministic)**

Use the lifecycle regressions in [S21](SOURCES.md#s21) to validate dispatch, cancellation,
storage-failure handling and worker cleanup before running behavioral evaluations.
Classify failures here as harness failures rather than model task failures.

<a id="c27"></a>

### C27 — Run headlessly with attributable results

**Support and tool mapping**

I/T host integration (generic BatchAgent fixtures plus ATOM adapter source); T01, T08.
Follow Waiting until Done or a classified stop, retaining task results, usage and
traces.

**Known failure path or boundary**

Documented limits: turn/time limits and failures are not solved tasks; native benchmark
policy and separate HCB adapter certification differ; alias changes may be forbidden.
Session reflection defaults off headlessly but can be enabled separately (C31); the
native policy does not itself forbid those ordinary turns.

**Source/test evidence**

- [S18](SOURCES.md#s18)
- [S25](SOURCES.md#s25)
- [test_the_turn_limit_stops_a_run_that_is_still_waiting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_headless_run.py#L75)
- [test_a_failed_turn_is_reported_not_raised](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_headless_run.py#L94)

**Proposed evaluation check**

Smoke-test done, timeout, failure and child work; verify stop reason, evidence
attribution and policy receipt. This row does not choose a benchmark baseline.

<a id="c28"></a>

### C28 — Use media only where supported

**Support and tool mapping**

D/I; shared runtime with conditional ATOM integration. T21. Attach supported media and
distinguish text fallback from actual model inspection.

**Known failure path or boundary**

Documented limits: text-only default, provider/format limits, per-cell capture cap, PDFs
only for File. No ATOM end-to-end multimodal evaluation inspected.

**Source/test evidence**

- [S20](SOURCES.md#s20)

**Proposed evaluation check**

Enable one supported media format and test a known-content fixture; disable it and
require accurate acknowledgment of the text notice. Establish host transport separately.

<a id="c29"></a>

### C29 — Perform optional bounded post-Done maintenance

**Support and tool mapping**

D/I/T; opt-in experimental_done_reflection, off by default; separate from session
reflection (C31). Preserve delivered result and summarize only captured completed
evidence.

**Known failure path or boundary**

Documented limits: gated by both turn count/time; restarts reset gate; stale plans
rejected; no TTL, memory bound or automatic general cleanup promise.

**Source/test evidence**

- [S12](SOURCES.md#s12)
- [S23](SOURCES.md#s23)
- [test_reflection_failure_rejection_budget_keeps_original_done](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L313)

**Proposed evaluation check**

If this experimental track is selected later, inject stale evidence and timeout; verify
the delivered answer is unchanged and unrelated/current state is retained.

<a id="c30"></a>

### C30 — Interpret diagnostics and trace metrics correctly

**Support and tool mapping**

I/T; host telemetry plus T13 (child-tracing test uses BaseAtom). Distinguish execution
error/rejection/hint counts from completed task outcomes. Shell handover and returned
inline waits over 60 seconds have separate counters.

**Known failure path or boundary**

Evidence limit: hint count is not recovery rate; counters do not establish ergonomic
quality; delegated histories require explicit inclusion when aggregating.

**Source/test evidence**

- [S19](SOURCES.md#s19)
- [test_atom_metrics](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_atom_metrics.py#L1)
- [test_child_session_tracing](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_child_session_tracing.py#L1)

**Proposed evaluation check**

Correlate controlled failures and child calls with saved events; independently score
outcome and recovery. Report missing trace evidence separately from absence of behavior.
Distinguish cell-deadline handover from command timeout and long-await accounting;
counter presence alone must not pass a recovery task.

<a id="c31"></a>

### C31 — Handle configured session housekeeping

**Support and tool mapping**

P/D/I/T; factory/host mechanism using T09–T10, T12–T13, T20. A root session posts
ordinary reflection messages; prompts request title, context, retained-variable and
todo/fact cleanup while preserving rules, promises and continuation references.

**Known failure path or boundary**

Defaults: on for ACP, off for other hosts; session override wins over workspace
settings. Both 50 cells and 60 minutes are required by default; pending waiting prompts
defer it; idle time alone does not wake a session. Configuration survives reload;
counters restart. Children are excluded by the session hook. Prompt delivery does not
enforce useful or safe cleanup. Independent of C29 and still subject to title/other host
restrictions.

**Source/test evidence**

- [S25](SOURCES.md#s25)
- [timing assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_reflection.py#L79)
- [waiting-prompt ownership](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_reflection.py#L352)
- [saved settings on reload](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_reflection.py#L378)

**Proposed evaluation check**

With housekeeping enabled and its delivery verified by the linked deterministic tests,
request cleanup amid standing rules and unfinished work. Independently verify retained
facts/promises and valid cleanup. Do not score a delivered message or an extra Done as
successful maintenance.

## Gaps that matter for Eval Author

The largest gap is between a scripted test exercising an API and a real model choosing
that API appropriately. In particular, C09/C25 (verification of delegated work), C14
(object/reference use), C16 (library/environment choice), C17 (recovery after feedback)
and C18–C19 (using partial/archived evidence) need behavioral tasks. Their runtime
plumbing is not equivalent to measured task coverage.

Default single-session behavior, orchestration, host lifecycle, conditional
extensions/media, session reflection (C31), and experimental post-Done reflection (C29)
should be separately tagged. Unsupported deployment features should not become automatic
agent failures. Changing those scopes requires a later decision; this pass does not
choose them.

Existing framework capability cases are starting evidence ([S22](SOURCES.md#s22)), not
proof that ATOM or the existing evaluation suite exercises these rows. Historical
unified-work results ([S23](SOURCES.md#s23)) belong to their original commits.

Every generated case should eventually identify its capability ID, required environment,
observable success check, relevant failure control, and evidence needed to show the
capability was exercised. These are preparation criteria, not a generated suite or a
configured mutation study.
