<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ATOM capability evidence for evaluation authoring

This package is a source-grounded first pass for preparing Eval Author inputs for
**NOOA ATOM** (`nooa_atom.agent:AtomAgent`). It inventories tools, expected behavior,
failure cases and evidence, then derives an intent draft for review.

- [Capability evidence matrix](capability-evidence-matrix.md): expectations,
  implementation/test evidence, failure boundaries and proposed evaluation checks.
- [ETHOS.md](ETHOS.md): the source-derived behavioral draft. It is structurally
  validated, but its product intent is **not yet confirmed**.

See the [preparation README](README.md) for artifact status and the fixtures benchmark link.

## Scope and provenance

The initial pass extracted capabilities and an Ethos draft. The fixtures
benchmark plan proposes task-set comparison and mutation analysis; this package does
not select the existing benchmark roster, configure models, run ATOM evaluations,
or establish a performance result. Source expectations remain distinct from
human-confirmed product intent and measured behavior.

Evidence was refreshed on 2026-11-11 after rebasing this checkout onto fetched
`origin/main`, commit `fd62a602aea69192f1d1a312ba57ab612a014d15`. ATOM is now present in
this checkout. This is a **source provenance record**, not a benchmark pin.
Current code/test links point to that immutable commit with updated line anchors.

The first pass inspected merge candidate `31c7a4494c49df8649f5e4ea045077cfecedc6b1`
on 2026-11-10. The refresh reviewed changes from that candidate to main, including
shared runtime changes used by ATOM. The historical unified-work experiment was
removed from main; [S23](#s23) deliberately retains its old immutable link and
does not treat its scores or interfaces as current evidence.

The inventory follows the public subclass, constructor registration, runtime
builtins, explicit visibility rules, host wiring and selected test assertions.
It does not treat every public-looking method in the private base as an offered
tool. Tests were inspected as source, **not executed** in this documentation pass.
Most ATOM integration tests use scripted/fake models: they establish executable
contracts, not whether a real model discovers and uses those contracts reliably.

When sources disagree, prefer current executable code and current assertions for
mechanics, record prompt expectations separately, and treat historical experiment
text as historical. Two examples: the current agent uses `self.jobs`/`Subagent`,
and live `handle` globals survive turns; older base instructions describe flat
delegation and call-local state. The package README also contains an older
"later changes" host-migration statement alongside implemented benchmark hosts.

### Test-fixture scope

The tests do not all instantiate the public ATOM. In
[test fixture definitions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/atom_test_agents.py#L358),
`ATOM_SPEC` names `BaseAtom`, a private-base test subclass; `PUBLIC_ATOM_SPEC`
names the public agent. The linked `test_agent`, `test_waiting_targets`,
`test_skill_manager` and `test_child_session_tracing` include base-agent fixtures.
They support inherited component contracts, not the complete current public
interface. `test_headless_run` uses a generic BatchAgent; it establishes host-loop
behavior. Public-agent evidence is also linked from `test_unified_agent`,
`test_atom_roles`, `test_session_namespace` and the public restriction tests.

## Changes incorporated from main

- **Shell deadlines (C06):** an inline or queued `Shell.run` can continue as a
  named job at the cell deadline. Retrieve or cancel that job; distinguish this
  from the command's own timeout and from user cancellation ([S05](#s05)).
- **State restoration (C15):** cell-defined class instances can restore while
  their class is resolvable in a live namespace. Unrestorable `self.v` entries
  are dropped individually with warnings; other entries survive ([S13](#s13)).
- **Recovery and helpers (C17/C22):** invalid results allow ten retries after the
  first invalid value. Standalone generated helpers require source-verifiable
  ellipsis bodies; ordinary helpers do not take `@strategy` ([S10](#s10), [S15](#s15)).
- **Authentication (C21):** each repeated pasted OAuth sign-in waits for its own
  callback instead of reusing the earlier code/state ([S16](#s16)).
- **Host lifecycle (C26):** running cells delay new messages; shared turn-loop
  and cancellation paths, session-owner probes and ACP worker shutdown have
  additional regression assertions ([S02](#s02), [S21](#s21)). POSIX import cleanup
  does not add native Windows execution support.
- **Telemetry (C30):** shell deadline handovers and completed inline waits longer
  than 60 seconds have separate counters; neither proves task success ([S19](#s19)).

## Tool inventory

These are the advertised core tool families and supporting runtime surfaces found
in the inspected source. **Only `python_cell` is a provider-facing tool**; the
others are Python APIs called inside it. `return_result` is a cell builtin.
External MCP tools, installed skills and task-environment commands are open-ended:
their exact names require a configured deployment inventory, which was not read.

| ID | Surface and supported operations | Availability and important boundaries | Sources |
|---|---|---|---|
| T01 | `python_cell(code)`; `return_result(...)`; `Done`, `Waiting`, `NeedInput`, `TaskResult` | Default ATOM execution/typed completion. Bare text is not completion. Host role controls allowed outcomes. | [S01](#s01), [S02](#s02), [S12](#s12) |
| T02 | `print`, `pprint`, `doc`, `help` (doc alias), `methods`, `variables`, `python_cell_state`; `asyncio`, `typing`; cell-defined Python helpers and imports | Inspect values/APIs, batch computations and reuse names. Printing is bounded; installed modules and execution restrictions constrain imports. `doc` returns text and must be printed. | [S12](#s12), [S14](#s14), [S15](#s15) |
| T03 | `self.files.read(path, lines=None)`, `replace(path, old, new)`, `replace_match(match, new)`, `write_file(path, content)` | Workspace file operations; reads return `Match`, writes return `FileWrite`. Line ranges are 1-indexed/inclusive; text replacement expects a unique occurrence. Writes have filesystem side effects. | [S01](#s01), [S04](#s04) |
| T04 | `self.repo.symbols(path='.', query='', max_results=50, cwd=None)`, `refs(name, path='.', max_results=50, cwd=None)` | Definitions/references with printable results, diagnostics and edit anchors. Supports session-backed filesystems. Results can be truncated; parser/text fallback is not a perfect semantic index. Root scoping is guidance, not containment. | [S04](#s04) |
| T05 | `Shell(...)`; `await sh.run(command, stdin=None, timeout=...)`; `await sh.aclose()`; async context management | A Shell lazily owns persistent Bash state. Different Shells are independent. Commands can modify files/processes/network according to their environment. No PTY/live-input protocol. At a cell deadline, an unfinished inline run can become a job; user cancellation still stops it. | [S05](#s05) |
| T06 | `self.jobs.launch(work, name=None, progress=None)`, `after(seconds, fn=None, name=None)`, `every(seconds, fn=None, name=None)`, `get(name)`, `list()` | Launch awaitables, async iterables or resources; timers are live-session work. `every` coalesces unread ticks and reports a tick exception as an item. Jobs are not durable scheduled services. | [S06](#s06) |
| T07 | `Job`: await/async iteration, `send(item)`, `message(item)`, `cancel()`, `set_progress(...)`; `name`, `status`, `result`, `error`, `summary`, `progress`, `output`; `JobEvent`, `Throttle` | Send creates a request job on a resource; message is supported only by resources implementing it. Await request jobs rather than the resource itself. Progress previews are opt-in/bounded and distinct from final output. | [S06](#s06) |
| T08 | `Subagent(description='', model=None).run(task, brief=None)`; launched `Subagent` resource via T06–T07 | One-shot call returns `TaskResult`; kept worker accepts follow-ups and may return `Question`. Children have separate histories/namespaces but share the checkout. Model alias override is host-policy-dependent. | [S07](#s07), [S18](#s18) |
| T09 | `self.todo.add`, `get`, `delete`, `activate`, `complete`; `Todo.title`, `description`, `comments`, `status`, `id`, `v` | Five advertised list methods; edit item content directly. IDs win over exact unique titles; duplicate titles require IDs. Host compatibility helpers are not the advertised planning interface. | [S08](#s08) |
| T10 | `self.v.<name>` and mapping-backed `self.vars`; per-Todo `v` | Serializable durable facts distinct from live Python globals. Restore also needs resolvable classes; unrestorable entries are dropped with warnings. Arbitrary resources/jobs are not recoverable snapshots. | [S08](#s08), [S13](#s13) |
| T11 | `self.message(text, echo=False)`; `self.user_messages.get(timeout=...)`, `qsize`, `status`, `flush`, `name` | Communication/readable input channel inherited from InteractiveAgent. Child messages go to the parent and are bounded; unattended results must carry the answer. Reading/flushing inputs affects delivery, not just display. | [S02](#s02), [S03](#s03), [S21](#s21) |
| T12 | `self.context[key]`, assignment/deletion, `keys`, `get`, `pop`, `set`, `all_keys`, `disabled`, `is_enabled`; `Context(...)` | ATOM explicitly exposes ContextApi. Protected blocks reject mutation. `set_static`, `set_dynamic`, `disable`, `enable` remain deprecated compatibility methods. | [S03](#s03), [S11](#s11) |
| T13 | `self.events.query`, `get`, indexed access, `keys`, `collapse(start_tag, end_tag, summary_text=None)` | Inspect execution/history and archive ranges synchronously. Archived originals remain addressable. Cell errors are `PythonOutput` events; not all errors are standalone `Error` events. | [S11](#s11) |
| T14 | `self.skills.search(query, limit=10)`, `activate(names)`, `deactivate(names)`, `doc(name)`, `status()` | Unified code/text/MCP discovery. Search loads nothing. Text instructions may arrive next turn; `doc` reads them now and can import an unloaded code skill. Registry/connection management helpers are hidden host APIs. | [S09](#s09) |
| T15 | Activated `self.<skill>` methods; text skill `source_dir`, `read_file(path)`, `run_script(...)` | Deployment-dependent plugins. TextSkill supplies file/script helpers; generated text packages may instead be wrapped code skills. Discover the actual object before assuming a helper exists. | [S09](#s09), [S10](#s10) |
| T16 | Connected `self.<server>.<tool>(...)` MCP methods | Configured service schemas determine actual tools, credentials and side effects. Exact configuration approval and possibly OAuth are prerequisites. Deactivation does not necessarily disconnect or revoke access. | [S09](#s09), [S16](#s16) |
| T17 | `self.skill_writing.path`, `list`, `create`, `reload`, `repo_tree`, `run_tests` | Scaffolds code/text skills under workspace `.nooa/skills`; reloads/tests agent-authored code. New sessions can load saved skills. This is persistent workspace modification. | [S10](#s10) |
| T18 | `self.methodwriting` guidance; `@strategy(PredictStrategy())` and CodeAct generated helper functions | Installed by base construction. MethodWriting has no public operational methods: it teaches defining reusable helpers and LLM sub-calls, with automatic argument rendering. Standalone generated functions require a source-verifiable ellipsis body. | [S03](#s03), [S10](#s10) |
| T19 | `self.workspace_settings.remember_skill`, `forget_skill`, `set_default_model`, `remember_mcp`, `forget_mcp`, `status` | Added by configured session factory. Writes `.nooa/settings.yaml`; future-session defaults need not change current/resumed sessions. Saving MCP configuration grants no approval/sign-in. | [S03](#s03), [S17](#s17) |
| T20 | `self.rename_session(title)` | Host-requested metadata change; preserves a title chosen by the person. Native guarded benchmarks disallow title generation. | [S03](#s03), [S18](#s18) |
| T21 | `show(obj)`; imported `nooa.media.Image`, `Audio`, `Video`, PDF `File` | Shared runtime capability, conditional on host transport and model `media_inputs`. Unsupported formats become text notices. No blanket ATOM multimodal-performance claim. | [S20](#s20) |

### Host surfaces and exclusions

`Session` submission, steering, withdrawal, cancellation, subscription, model/mode
selection, persistence and recovery; `open_tree`/`run_task`; and ACP slash commands
such as `/skills`, `/mcp`, `/verbosity`, `/recover` are **host/control surfaces**.
`/mcp-add` routes server details to the agent; it is not automatic approval.
They belong in lifecycle or integration evaluations when in scope ([S18](#s18),
[S21](#s21)). They are not additional provider tools.

The public agent hides `self.shell`, `delegates`, `producers`, `system_messages`,
`cwd`, the session port and runtime managers. Legacy `delegate`/`spawn` raise
errors directing callers to `Subagent`. Old `JobsApi.run/spawn/tail` is replaced
by WorkManager. `Shell.send` and `Subagent.send/message` are adapter plumbing;
the advertised model interface uses `Job.send/message`. `self.skills.read` and
Todo compatibility methods are hidden. Explicit configuration can expose additional
skills, including the legacy shell skill; that does not make them the default
surface ([S01](#s01), [S03](#s03), [S09](#s09)).

Memory/knowledge-graph integration, the retired web publisher and producers are
not automatically loaded by base construction. General NOOA features or optional
packages are not evidence that default ATOM has those capabilities ([S03](#s03)).
Optional post-Done reflection is separately configured, off by default, and is
not an automatic expiry or memory-growth guarantee ([S12](#s12), [S23](#s23)).

## How to interpret and maintain the evidence

The [matrix](capability-evidence-matrix.md) distinguishes prompt expectations,
implementation, regression-test source, explicit limitations and proposed checks.
There are no newly measured task-level results in this package. A test named for
a regression is evidence that a failure path is addressed, not proof of an open bug.
An error hint being emitted is not proof that a model recovers after reading it.

For future updates: read the public subclass and its actual factory configuration;
reconcile visibility and dynamic tools; inspect the linked assertions; update
source revision/links together; and retain capability IDs when their meaning is
unchanged. Record any new execution evidence with its command, revision, setup
and result. Do not silently promote a source assertion into measured coverage.

## Source index

Each source group below states what was used from it. Code links are immutable
commit links with line anchors; tests are linked to their file or specific test.
If the hosting site is unavailable, the same contents are locally recoverable:

```sh
git show fd62a602aea69192f1d1a312ba57ab612a014d15:packages/nooa-atom/src/nooa_atom/agent/agent.py
```

<a id="s01"></a>

### S01 — Public ATOM interface and visibility

Current prompt, file facade, role postconditions, namespace setup, and explicit hiding of legacy APIs. The public subclass overrides its private base.

[AtomAgent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L613); [Files](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L72); [visibility overrides](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L1107); [test_public_atom_agent_hides_the_base_surface](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_unified_agent.py#L785); [test_prompt_leaks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_prompt_leaks.py#L1).

<a id="s02"></a>

### S02 — Roles and result contracts

Interactive, unattended, orchestrator and child policies; completion and waiting requirements. Role instructions include behavioral expectations beyond type validation.

[packages/nooa-atom/src/nooa_atom/agent/roles.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/roles.py#L1); [waiting/completion checks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L397); [TaskResult and turn types](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/interactive.py#L1); [test_atom_roles](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_atom_roles.py#L1); [test_a_batch_turn_must_carry_a_task_result](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_agent.py#L66); [test_waiting_targets](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_waiting_targets.py#L1).

<a id="s03"></a>

### S03 — Construction and default capabilities

Base tool registration, deferred memory integration, context/events opt-in, summarizer installation, and host configuration. Use the subclass to resolve superseded base instructions.

[base construction](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/base.py#L96); [packages/nooa-atom/src/nooa_atom/agent/factory.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/factory.py#L1); [workspace skills](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/workspace/options.py#L78); [test_agent_factory](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_agent_factory.py#L1).

<a id="s04"></a>

### S04 — Repository discovery and file editing

Definitions/references, result truncation and diagnostics, Match anchors, replacement guards, and filesystem selection. Repository search roots are not an enforced filesystem sandbox.

[RepoTools](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/tools/repo_tools.py#L512); [file read/replace/write](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/tools/shell_tools.py#L1363); [test_repo_tools_refs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_repo_tools_refs.py#L1); [test_repo_tools_session_paths](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_repo_tools_session_paths.py#L1); [test_file_write_documents_real_string_fields](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_doc_contracts.py#L40).

[Shared file-layer stale-anchor tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/tools/test_shell_tools_stale_match.py#L30).

<a id="s05"></a>

### S05 — Shell execution

Persistent Bash per object, stdin, per-command deadlines, output capture, queueing, cancellation, restarts, and cleanup.

[Shell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_shell.py#L412); [CommandResult](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_shell.py#L132); [test_work_shell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_work_shell.py#L1); [test_shell_quickstart_executes_as_written](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_doc_contracts.py#L50).

Cell-deadline handover, queued-command ownership and user cancellation: [shell handover assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_cell_deadline_detach.py#L252). These use the public ATOM with a scripted model and a shortened cell deadline. A handover requires an available session work manager; otherwise cancellation stops the command.

<a id="s06"></a>

### S06 — Jobs, timers, and progress

Generic work manager, request/result ownership, cancellation, bounded progress, timer coalescing and retained job history.

[WorkManager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_manager.py#L1033); [Job](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_manager.py#L364); [packages/nooa-atom/src/nooa_atom/agent/work_progress.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_progress.py#L1); [test_work_manager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_work_manager.py#L1); [test_work_progress](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_work_progress.py#L1); [test_shell_progress_sessions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_shell_progress_sessions.py#L1).

<a id="s07"></a>

### S07 — Subagents and deadline detachment

One-shot versus kept workers, questions, notes, resource/request cancellation, and detachment of an awaited child at a cell deadline.

[Subagent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/work_subagent.py#L167); [packages/nooa-atom/src/nooa_atom/session/registry.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/session/registry.py#L1); [test_unified_agent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_unified_agent.py#L1); [test_cell_deadline_detach](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_cell_deadline_detach.py#L1).

<a id="s08"></a>

### S08 — Plans and per-item state

Advertised TodoList API, exact ID/title resolution, mutable item fields, bounded status display, persistence and legacy compatibility.

[packages/nooa-atom/src/nooa_atom/tools/todo_list.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/tools/todo_list.py#L1); [test_todo_list](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_todo_list.py#L1); [test_per_todo_state_is_advertised_and_round_trips_without_a_new_field](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_doc_contracts.py#L25).

<a id="s09"></a>

### S09 — Skill discovery and dynamic tools

Unified code/text/MCP discovery, activation, docs, name conflicts and explicit hidden host helpers. The installed skill set is deployment-dependent.

[SkillManager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/skills/manager.py#L99); [TextSkill](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/skill.py#L352); [test_skill_manager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_skill_manager.py#L1); [test_mcp_discovery](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_mcp_discovery.py#L1).

<a id="s10"></a>

### S10 — Authoring reusable skills and generated helpers

Code/text skill scaffolding, reload/testing and persistence; MethodWriting teaches cell-defined helpers and nested generation rather than exposing a method-writing RPC.

[SkillWriting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/tools/library_writing_lib.py#L69); [src/nooa/tools/method_writing_lib.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/tools/method_writing_lib.py#L1); [test_libs_persist](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_libs_persist.py#L1); [test_namespace_nested](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_namespace_nested.py#L1).

Standalone helpers now require inspectable ellipsis source: [strategy validation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/decorators.py#L26); [cross-cell helper tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/runtime/test_strategy_helper_source.py#L23); [decorator rejection tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/test_decorator_strategy.py#L1). The cross-cell tests exercise shared actor/sandbox backends, not a real-model ATOM task.

<a id="s11"></a>

### S11 — Context and event inspection

Exposed prompt-block API, event queries and archival collapse, context-usage guidance and automatic summarization installation.

[src/nooa/runtime/context.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/context.py#L1); [src/nooa/runtime/events.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/events.py#L1); [src/nooa/agents/summarization.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/agents/summarization.py#L1); [context/events exposure](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/base.py#L284); [test_context_usage_examples_run](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_advertised_examples.py#L129).

<a id="s12"></a>

### S12 — Python namespace and reference semantics

Owned in-process globals, argument identity, scopes, one-time definitions, dynamic skill providers, and isolation between calls/sessions.

[src/nooa/runtime/execution_namespace.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/execution_namespace.py#L1); [src/nooa/strategies/codeact_v2.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/strategies/codeact_v2.py#L1); [test_session_namespace](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_session_namespace.py#L1); [test_namespace_review_repairs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_namespace_review_repairs.py#L1); [test_namespace_nested](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_namespace_nested.py#L1); [tests/runtime/test_execution_namespace.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/runtime/test_execution_namespace.py#L1).

[Framework namespace providers](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/namespace_providers.py#L1); [ATOM framework-name availability test](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_namespace_providers.py#L30).

<a id="s13"></a>

### S13 — Durable state and restore

Serializable self.v/todo state and session storage. Python globals and live work do not resume from snapshots.

[src/nooa/storage/snapshot_vars.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/storage/snapshot_vars.py#L1); [packages/nooa-atom/src/nooa_atom/session/store.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/session/store.py#L1); [after_restore](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L962); [test_namespace_nosnapshot_restart_notice_once_and_self_v_survives](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_session_namespace.py#L462); [test_after_restore_reports_lost_jobs_once](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_unified_agent.py#L692); [test_atom_resume](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_atom_resume.py#L1).

Cell-defined class lookup and entry-wise restore: [live class resolver](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/cells.py#L1); [lenient restore](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/storage/serialization.py#L689); [public ATOM restore tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_cell_class_restore.py#L1). Restoring a fresh agent in the same process can reuse a live class; simulated absence of that class drops only affected entries. This is not persistence of executable class definitions across process restart.

<a id="s14"></a>

### S14 — Instructions and execution boundaries

Repository instruction discovery/refresh, explicit prompt rules and runtime restriction hints. Visibility and prompt guidance must not be mistaken for security isolation.

[packages/nooa-atom/src/nooa_atom/agent/instructions.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/instructions.py#L1); [agent instructions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L613); [docs/concepts/safety.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/docs/concepts/safety.md#L1); [test_restriction_hints](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_restriction_hints.py#L1); [test_refs_outside_the_root_does_not_raise](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_repo_tools_refs.py#L51).

<a id="s15"></a>

### S15 — Error feedback and output truncation

Hints for undefined names, missing libraries, incorrect MCP call forms and wrong result attributes; irrecoverable cell-output truncation versus file-backed shell capture.

[error hints](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L165); [error formatting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/agent.py#L335); [test_result_attribute_errors](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_result_attribute_errors.py#L1); [test_return_result_fixups](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_return_result_fixups.py#L1); [test_cell_print_cut](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_cell_print_cut.py#L1); [test_direct_run_keeps_long_output_until_aclose](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_work_shell.py#L114).

Invalid-result budget: [retry accounting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/strategies/codeact.py#L2711); [bounded retry assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_return_result_fixups.py#L166). ATOM allows ten retries after the initial invalid value; an eleventh invalid value fails the turn. This budget is separate from ordinary session errors.

<a id="s16"></a>

### S16 — MCP approval, connection, and authentication

Exact-definition/workspace approval, OAuth sign-in and recovery, and connected versus active states. No configured service inventory is inferred from the framework.

[packages/nooa-atom/src/nooa_atom/workspace/mcp_approval.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/workspace/mcp_approval.py#L1); [packages/nooa-atom/src/nooa_atom/skills/mcp_servers.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/skills/mcp_servers.py#L1); [packages/nooa-atom/src/nooa_atom/skills/mcp_auth.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/skills/mcp_auth.py#L1); [test_mcp_approval_fixes](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_mcp_approval_fixes.py#L1); [test_mcp_auth_recovery](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_mcp_auth_recovery.py#L1); [test_activating_an_mcp_server_connects_it_and_its_tools_are_methods](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_skill_manager.py#L225).

Repeated sign-in must use a fresh callback: [successive OAuth flows](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_mcp_signin_lifetime.py#L125). This is a local authentication-component test, not a live provider measurement.

<a id="s17"></a>

### S17 — Saved workspace preferences

Skills, model defaults and MCP startup configuration; credentials must remain placeholders. Saving preferences is distinct from approving or authenticating a connection.

[WorkspaceSettings](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/workspace/workspace_settings.py#L69); [registration](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/workspace/options.py#L99); [test_workspace_settings_mcp](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_workspace_settings_mcp.py#L1); [test_skills_control_settings_scope](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_skills_control_settings_scope.py#L1).

<a id="s18"></a>

### S18 — Headless and benchmark hosts

run_task, Harbor AtomBenchAgent and native Gym AtomHarness; stop reasons, outputs, benchmark guard and model-client attribution.

[packages/nooa-atom/src/nooa_atom/hosts/headless.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/hosts/headless.py#L1); [packages/nooa-atom/src/nooa_atom/bench_agent.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/bench_agent.py#L1); [packages/nooa-atom/src/nooa_atom/gym.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/gym.py#L1); [packages/nooa-atom/src/nooa_atom/benchmark_policy.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/benchmark_policy.py#L1); [test_headless_run](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_headless_run.py#L1); [test_gym_policy_worker](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_gym_policy_worker.py#L1); [test_bench_agent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_bench_agent.py#L1).

<a id="s19"></a>

### S19 — Metrics and traces

ATOM-specific instrumentation counts hint/rejection/detachment events. Tests verify event attribution; these are not task-success or recovery-rate measurements.

[packages/nooa-atom/src/nooa_atom/agent/metrics.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/agent/metrics.py#L1); [packages/nooa-atom/src/nooa_atom/session/tracing.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/session/tracing.py#L1); [test_atom_metrics](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_atom_metrics.py#L1); [test_child_session_tracing](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_child_session_tracing.py#L1); [packages/nooa-bench/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-bench/README.md#L1).

New counters are `harness.atom.shell_detached_at_deadline.count` and `harness.atom.shell_long_await.count`: [long-await assertion](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_atom_metrics.py#L131) and the public-agent shell handover tests in [S05](#s05). Long-await accounting occurs when an inline run returns after more than 60 seconds; detached work and already-managed jobs are not counted by that return path.

<a id="s20"></a>

### S20 — Media and library availability

Shared CodeAct media capture and model-format gates; benchmark dependencies are opt-in and do not preload names or guarantee target-runtime imports.

[docs/media.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/docs/media.md#L1); [src/nooa/runtime/media_capture.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/media_capture.py#L1); [packages/nooa-atom/pyproject.toml](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/pyproject.toml#L1); [benchmark profile](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/README.md#L14); [test_managed_job_output_when_no_cell_is_active](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_session_namespace.py#L601).

<a id="s21"></a>

### S21 — Session/ACP lifecycle

Host submission/steering/cancellation, durable dispatch, recovery, settings and presentation. These are host interfaces, not extra model tools.

[packages/nooa-atom/src/nooa_atom/session/session.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/session/session.py#L1); [durable dispatch](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/README.md#L477); [packages/nooa-atom/src/nooa_atom/acp/server.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/acp/server.py#L1); [test_dispatch_invariants](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_dispatch_invariants.py#L1); [test_recover](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_recover.py#L1); [acp_adapter/test_questions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/acp_adapter/test_questions.py#L1); [acp_adapter/test_verbosity](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/acp_adapter/test_verbosity.py#L1).

[ChannelReader](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/channels.py#L645).

Additional lifecycle evidence: [shared turn loop](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/src/nooa/runtime/turn_loop.py#L1); [idle readiness regression](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/runtime/test_turn_loop.py#L166); [cancellation propagation tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/runtime/test_channels_queue_ergonomics.py#L517); [non-locking owner probe](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_store.py#L457); [parent watchdog](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/src/nooa_atom/acp/worker.py#L78); [process-group boundary](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/acp_router/test_worker.py#L156); [POSIX import/feature boundaries](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/tests/test_posix_imports.py#L1). Owner probes read PID/host state without acquiring the session lock. On parent loss, a group-leading worker kills its group; a worker that does not lead the group kills only itself. Shared runtime tests are component evidence, not model behavior.

<a id="s22"></a>

### S22 — Existing framework capability cases

Framework-level examples and evaluation configuration; their presence does not establish current ATOM coverage or identify Paul’s chosen baseline.

[tests/capability/config.yaml](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/capability/config.yaml#L1); [tests/capability/agents/error_recovery.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/capability/agents/error_recovery.py#L1); [tests/capability/agents/large_data.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/capability/agents/large_data.py#L1); [tests/capability/agents/repl_exploration.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/tests/capability/agents/repl_exploration.py#L1).

<a id="s23"></a>

### S23 — Architecture history and benchmark caveats

The unified-work experiment records earlier interfaces and frozen results. It was
removed from main; the historical link below intentionally remains at
`31c7a4494c49df8649f5e4ea045077cfecedc6b1`. Its caveats prevent transferring historical
scores to this revision. The ATOM package README link follows the current source
revision and supplies current architecture/reflection documentation.

[experiments/unified-work/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/31c7a4494c49df8649f5e4ea045077cfecedc6b1/experiments/unified-work/README.md#L1); [packages/nooa-atom/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/packages/nooa-atom/README.md#L1).

<a id="s24"></a>

### S24 — Ethos schema and authoring procedure

The separate Eval Author checkout supplied `team-skills/ethos/ethos-orchestrate`,
`ethos-explore`, and `ethos-write` instructions, the schema-v1 template and its
standard-library validator. That checkout was at
`f2e3b78cd19bdee599a76e4d7bdf922595d6c192` (`feat/metric-benchmark-record`). Immutable source links:

- [Orchestrate skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/team-skills/ethos/ethos-orchestrate/SKILL.md)
- [Explore skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/team-skills/ethos/ethos-explore/SKILL.md)
- [Write skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/team-skills/ethos/ethos-write/SKILL.md)
- [Template](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/team-skills/ethos/ethos-write/references/templates/ETHOS.md)
- [Validator](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/team-skills/ethos/ethos-write/scripts/validate_ethos.py)

These links record the inspected Eval Author revision and establish document
structure, not ATOM behavior. The user explicitly requested a source-derived
first draft before the intent discussion, so the document preserves unconfirmed
intent instead of claiming the skills' interview/sign-off workflow is complete.

## Validation of this documentation pass

Evidence validation on 2026-11-11 (rechecked when packaged on 2026-11-15):

- The bundled Ethos schema-v1 validator passed, without warnings.
- All source link paths and line anchors were checked against the inspected Git
  revision (and the historical revision for S23); all local file/section links
  in this package resolved. Existing source anchors were remapped to the same
  source lines after the rebase.
- Tool, capability and source IDs were checked for uniqueness and completeness.
- `git diff --check` passed for tracked changes; the new documents were also
  checked for trailing whitespace and Markdown table column consistency.

These checks establish document structure and reference targets. Remote HTTP
availability was not tested. No ATOM runtime tests or model evaluations were run,
and no product-intent confirmation is implied.
