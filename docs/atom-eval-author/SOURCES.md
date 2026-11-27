<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ATOM capability evidence for evaluation authoring

This package is a source-grounded first pass for preparing Eval Author inputs for **NOOA
ATOM** (`nooa_atom.agent:AtomAgent`). It inventories tools, expected behavior, failure
cases and evidence, then derives an intent draft for review.

- [Capability evidence matrix](capability-evidence-matrix.md): expectations,
  implementation/test evidence, failure boundaries and proposed evaluation checks.
- [ETHOS.md](ETHOS.md): the source-derived behavioral draft. It is structurally
  validated, but its product intent is **not yet confirmed**.

See the [preparation README](README.md) for artifact status and benchmark ownership.

## Scope and provenance

The evidence categories and draft status are defined in the
[ETHOS introduction](ETHOS.md) and
[matrix legend](capability-evidence-matrix.md#evidence-and-failure-labels). This
inventory records the source basis and validation results for those artifacts.

The reviewed source baseline is NOOA `107a65d77e4c2c644bb90553ac404bb4ed8a064c`
(2026-11-16). Code/test links use that immutable commit. This is source provenance,
not an agreed benchmark runtime pin. [S23](#s23) retains the explicitly historical
unified-work experiment at `31c7a4494c49df8649f5e4ea045077cfecedc6b1`; its scores and
interfaces are not current evidence.

The inventory follows the public subclass, constructor registration, runtime builtins,
explicit visibility rules, host wiring and selected test assertions. It does not treat
every public-looking method in the private base as an offered tool. **T means inspected
test assertions**, with the executed subset recorded under
[validation](#validation-of-this-documentation-pass). Most ATOM integration tests use
scripted/fake models: they establish executable contracts, not whether a real model
discovers and uses those contracts reliably.

When sources disagree, prefer current executable code and current assertions for
mechanics, record prompt expectations separately, and treat historical experiment text
as historical. Two examples: the current agent uses `self.jobs`/`Subagent`, and live
`handle` globals survive turns; older base instructions describe flat delegation and
call-local state. The package README also contains an older "later changes"
host-migration statement alongside implemented benchmark hosts.

### Test-fixture scope

The tests do not all instantiate the public ATOM. In
[test fixture definitions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/atom_test_agents.py#L358),
`ATOM_SPEC` names `BaseAtom`, a private-base test subclass; `PUBLIC_ATOM_SPEC` names the
public agent. The linked `test_agent`, `test_waiting_targets`, `test_skill_manager` and
`test_child_session_tracing` include base-agent fixtures. They support inherited
component contracts, not the complete current public interface. `test_headless_run` uses
a generic BatchAgent; it establishes host-loop behavior. Public-agent evidence is also
linked from `test_unified_agent`, `test_atom_roles`, `test_session_namespace` and the
public restriction tests.

## Tool inventory

These are the advertised core tool families and supporting runtime surfaces found in the
inspected source. **Only `python_cell` is a provider-facing tool**; the others are
Python APIs called inside it. `return_result` is a cell builtin. External MCP tools,
installed skills and task-environment commands are open-ended: their exact names require
a configured deployment inventory, which was not read.

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
| T20 | `self.rename_session(title)` | Host-requested metadata change; preserves a title chosen by the person. Native guarded benchmarks disallow title generation. Mapped to C31 housekeeping; standalone naming evaluation remains undecided. | [S03](#s03), [S18](#s18) |
| T21 | `show(obj)`; imported `nooa.media.Image`, `Audio`, `Video`, PDF `File` | Shared runtime capability, conditional on host transport and model `media_inputs`. Unsupported formats become text notices. No blanket ATOM multimodal-performance claim. | [S20](#s20) |

### Host surfaces and exclusions

`Session` submission, steering, withdrawal, cancellation, subscription, model/mode
selection, persistence and recovery; `open_tree`/`run_task`; and ACP slash commands
such as `/skills`, `/mcp`, `/verbosity`, `/recover` are **host/control surfaces**.
`/mcp-add` routes server details to the agent; it is not automatic approval.
`/reflection [on|off|config [BLOCK on|off]]` controls factory-registered session
housekeeping and its saved settings ([S25](#s25)); it is also a host surface. They
belong in lifecycle or integration evaluations when in scope ([S18](#s18), [S21](#s21)). They are not additional provider tools.

The default public agent has **no `self.shell` attribute** and does not load
`nemo.shell`. Its private `_workspace_shell` backs `self.files`, `self.repo` and
`self.skill_writing.run_tests`; model commands use `Shell()`. It hides `delegates`,
`producers`, `system_messages`, `cwd`, the session port and runtime managers. Legacy
`delegate`/`spawn` raise errors directing callers to `Subagent`. Old
`JobsApi.run/spawn/tail` is replaced by WorkManager. `Shell.send` and
`Subagent.send/message` are adapter plumbing; the advertised model interface uses
`Job.send/message`. `self.skills.read` and Todo compatibility methods are hidden.
Explicit configuration can expose additional skills, including the legacy shell skill;
that does not make them the default surface ([S01](#s01), [S03](#s03), [S09](#s09)).

Memory/knowledge-graph integration, the retired web publisher and producers are not
automatically loaded by base construction. General NOOA features or optional packages
are not evidence that default ATOM has those capabilities ([S03](#s03)). Experimental
post-Done reflection is separately configured and off by default ([S12](#s12),
[S23](#s23)). Session reflection is a different mechanism, enabled by default for ACP
root sessions ([S25](#s25)). Neither establishes automatic expiry, bounded memory growth
or reliable model cleanup.

## How to interpret and maintain the evidence

The [matrix](capability-evidence-matrix.md) distinguishes prompt expectations,
implementation, regression-test source, explicit limitations and proposed checks. There
are no newly measured task-level results in this package. A test named for a regression
is evidence that a failure path is addressed, not proof of an open bug. An error hint
being emitted is not proof that a model recovers after reading it.

For future updates: read the public subclass and its actual factory configuration;
reconcile visibility and dynamic tools; inspect the linked assertions; update source
revision/links together; and retain capability IDs when their meaning is unchanged.
Record any new execution evidence with its command, revision, setup and result. Do not
silently promote a source assertion into measured coverage.

## Source index

Each source group below states what was used from it. Code links are immutable commit
links with line anchors; tests are linked to their file or specific test. If the hosting
site is unavailable, the same contents are locally recoverable:

```sh
git show 107a65d77e4c2c644bb90553ac404bb4ed8a064c:packages/nooa-atom/src/nooa_atom/agent/agent.py
```

<a id="s01"></a>

### S01 — Public ATOM interface and visibility

Current prompt, file facade, role postconditions, namespace setup, and explicit hiding
of legacy APIs. The public subclass overrides its private base.

[AtomAgent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L613);
[Files](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L72);
[visibility overrides](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L1104);
[test_public_atom_agent_hides_the_base_surface](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L785);
[test_prompt_leaks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_prompt_leaks.py#L1).

<a id="s02"></a>

### S02 — Roles and result contracts

Interactive, unattended, orchestrator and child policies; completion and waiting
requirements. Role instructions include behavioral expectations beyond type validation.

[packages/nooa-atom/src/nooa_atom/agent/roles.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/roles.py#L1);
[waiting/completion checks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L397);
[TaskResult and turn types](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/interactive.py#L1);
[test_atom_roles](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_atom_roles.py#L1);
[test_a_batch_turn_must_carry_a_task_result](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_agent.py#L66);
[test_waiting_targets](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_waiting_targets.py#L1).

<a id="s03"></a>

### S03 — Construction and default capabilities

Built-ins are installed through registry activation, with a private workspace shell for
repository/file operations and skill tests. `nemo.shell` is excluded from default
loading. The public subclass replaces the base todo manager with TodoList through
registration. Context/events opt-in, summarizers and host configuration remain distinct
from optional deployment tools.

[Built-in installation, collision and restore assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_builtin_skill_install.py#L1);
[absence of the shell attribute](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_agent_core.py#L48).

[base construction](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/base.py#L96);
[packages/nooa-atom/src/nooa_atom/agent/factory.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/factory.py#L1);
[workspace skills](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/workspace/options.py#L80);
[test_agent_factory](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_agent_factory.py#L1).

<a id="s04"></a>

### S04 — Repository discovery and file editing

Definitions/references, result truncation and diagnostics, Match anchors, replacement
guards, and filesystem selection. Repository search roots are not an enforced filesystem
sandbox.

[RepoTools](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/tools/repo_tools.py#L512);
[file read/replace/write](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/tools/shell_tools.py#L1363);
[test_repo_tools_refs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_repo_tools_refs.py#L1);
[test_repo_tools_session_paths](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_repo_tools_session_paths.py#L1);
[test_file_write_documents_real_string_fields](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_doc_contracts.py#L40).

[Shared file-layer stale-anchor tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/tools/test_shell_tools_stale_match.py#L30).

<a id="s05"></a>

### S05 — Shell execution

Persistent Bash per object, stdin, per-command deadlines, output capture, queueing,
cancellation, restarts, and cleanup.

[Shell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_shell.py#L412);
[CommandResult](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_shell.py#L132);
[test_work_shell](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L1);
[test_shell_quickstart_executes_as_written](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_doc_contracts.py#L50).

Cell-deadline handover, queued-command ownership and user cancellation:
[shell handover assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_deadline_detach.py#L252).
These use the public ATOM with a scripted model and a shortened cell deadline. A
handover requires an available session work manager; otherwise cancellation stops the
command.

<a id="s06"></a>

### S06 — Jobs, timers, and progress

Generic work manager, request/result ownership, cancellation, bounded progress, timer
coalescing and retained job history.

[WorkManager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_manager.py#L1033);
[Job](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_manager.py#L364);
[packages/nooa-atom/src/nooa_atom/agent/work_progress.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_progress.py#L1);
[test_work_manager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_manager.py#L1);
[test_work_progress](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_progress.py#L1);
[test_shell_progress_sessions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_shell_progress_sessions.py#L1).

<a id="s07"></a>

### S07 — Subagents and deadline detachment

One-shot versus kept workers, questions, notes, resource/request cancellation, and
detachment of an awaited child at a cell deadline.

[Subagent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_subagent.py#L167);
[packages/nooa-atom/src/nooa_atom/session/registry.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/session/registry.py#L1);
[test_unified_agent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L1);
[test_cell_deadline_detach](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_deadline_detach.py#L1).

<a id="s08"></a>

### S08 — Plans and per-item state

Advertised TodoList API, exact ID/title resolution, mutable item fields, bounded status
display, persistence and legacy compatibility.

[packages/nooa-atom/src/nooa_atom/tools/todo_list.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/tools/todo_list.py#L1);
[test_todo_list](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_todo_list.py#L1);
[test_per_todo_state_is_advertised_and_round_trips_without_a_new_field](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_doc_contracts.py#L25).

<a id="s09"></a>

### S09 — Skill discovery and dynamic tools

Unified code/text/MCP discovery, activation, docs, name conflicts and explicit hidden
host helpers. The installed skill set is deployment-dependent.

[SkillManager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/skills/manager.py#L99);
[TextSkill](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/skill.py#L352);
[test_skill_manager](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skill_manager.py#L1);
[test_mcp_discovery](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_discovery.py#L1).

<a id="s10"></a>

### S10 — Authoring reusable skills and generated helpers

Code/text skill scaffolding, reload/testing and persistence; MethodWriting teaches
cell-defined helpers and nested generation rather than exposing a method-writing RPC.
ATOM injects its private workspace shell into SkillWriting, so `run_tests` does not need
a loaded `nemo.shell` or a `self.shell` attribute. Other users of SkillWriting may still
use its legacy shell dependency.

[SkillWriting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/tools/library_writing_lib.py#L69);
[src/nooa/tools/method_writing_lib.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/tools/method_writing_lib.py#L1);
[test_libs_persist](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_libs_persist.py#L1);
[test_namespace_nested](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_nested.py#L1).

Standalone helpers now require inspectable ellipsis source:
[strategy validation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/decorators.py#L26);
[cross-cell helper tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_strategy_helper_source.py#L23);
[decorator rejection tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/test_decorator_strategy.py#L1).
The cross-cell tests exercise shared actor/sandbox backends, not a real-model ATOM task.

<a id="s11"></a>

### S11 — Context and event inspection

Exposed prompt-block API, event queries and archival collapse, context-usage guidance
and automatic summarization installation.

[src/nooa/runtime/context.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/context.py#L1);
[src/nooa/runtime/events.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/events.py#L1);
[src/nooa/agents/summarization.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/agents/summarization.py#L1);
[context/events exposure](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/base.py#L293);
[test_context_usage_examples_run](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_advertised_examples.py#L129).

<a id="s12"></a>

### S12 — Python namespace and reference semantics

Owned in-process globals, argument identity, scopes, one-time definitions, dynamic skill
providers, and isolation between calls/sessions.

[src/nooa/runtime/execution_namespace.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/execution_namespace.py#L1);
[src/nooa/strategies/codeact_v2.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/strategies/codeact_v2.py#L1);
[test_session_namespace](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L1);
[test_namespace_review_repairs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_review_repairs.py#L1);
[test_namespace_nested](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_nested.py#L1);
[tests/runtime/test_execution_namespace.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_execution_namespace.py#L1).

[Framework namespace providers](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/namespace_providers.py#L1);
[ATOM framework-name availability test](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_namespace_providers.py#L30).

<a id="s13"></a>

### S13 — Durable state and restore

Serializable self.v/todo state and session storage. Python globals and live work do not
resume from snapshots.

[src/nooa/storage/snapshot_vars.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/storage/snapshot_vars.py#L1);
[packages/nooa-atom/src/nooa_atom/session/store.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/session/store.py#L1);
[after_restore](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L959);
[test_namespace_nosnapshot_restart_notice_once_and_self_v_survives](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L462);
[test_after_restore_reports_lost_jobs_once](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_unified_agent.py#L692);
[test_atom_resume](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_atom_resume.py#L1).

Cell-defined class lookup and entry-wise restore:
[live class resolver](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/cells.py#L1);
[lenient restore](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/storage/serialization.py#L689);
[public ATOM restore tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_class_restore.py#L1).
Restoring a fresh agent in the same process can reuse a live class; simulated absence of
that class drops only affected entries. This is not persistence of executable class
definitions across process restart.

<a id="s14"></a>

### S14 — Instructions and execution boundaries

Repository instruction discovery/refresh, explicit prompt rules and runtime restriction
hints. Visibility and prompt guidance must not be mistaken for security isolation.

[packages/nooa-atom/src/nooa_atom/agent/instructions.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/instructions.py#L1);
[agent instructions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L613);
[docs/concepts/safety.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/docs/concepts/safety.md#L1);
[test_restriction_hints](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_restriction_hints.py#L1);
[test_refs_outside_the_root_does_not_raise](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_repo_tools_refs.py#L51).

<a id="s15"></a>

### S15 — Error feedback and output truncation

Hints for undefined names, missing libraries, incorrect MCP call forms and wrong result
attributes; default irrecoverable cell-output truncation versus file-backed shell
capture.

At this baseline, cell capture still uses
[CaptureScope](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/capture_scope.py#L46)
and
[CaptureConfig](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/config/truncation_config.py#L42),
while Shell uses its own
[capture implementation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/work_shell.py#L151).
The newly available
[snapshot](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/docs/snapshot-model.md#L1)
and
[overflow](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/docs/overflow.md#L1)
libraries do not replace those paths yet. Their guarantees must not be attributed to
default ATOM output. The public agent's default cell capture is memory-only; explicit
capture configuration can change that behavior.

[error hints](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L165);
[error formatting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/agent.py#L335);
[test_result_attribute_errors](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_result_attribute_errors.py#L1);
[test_return_result_fixups](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_return_result_fixups.py#L1);
[test_cell_print_cut](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_cell_print_cut.py#L1);
[test_direct_run_keeps_long_output_until_aclose](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_work_shell.py#L114).

Invalid-result budget:
[retry accounting](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/strategies/codeact.py#L2711);
[bounded retry assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_return_result_fixups.py#L166).
ATOM allows ten retries after the initial invalid value; an eleventh invalid value fails
the turn. This budget is separate from ordinary session errors.

<a id="s16"></a>

### S16 — MCP approval, connection, and authentication

Exact-definition/workspace approval, OAuth sign-in and recovery, and connected versus
active states. No configured service inventory is inferred from the framework.

[packages/nooa-atom/src/nooa_atom/workspace/mcp_approval.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/workspace/mcp_approval.py#L1);
[packages/nooa-atom/src/nooa_atom/skills/mcp_servers.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/skills/mcp_servers.py#L1);
[packages/nooa-atom/src/nooa_atom/skills/mcp_auth.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/skills/mcp_auth.py#L1);
[test_mcp_approval_fixes](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_approval_fixes.py#L1);
[test_mcp_auth_recovery](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_auth_recovery.py#L1);
[test_activating_an_mcp_server_connects_it_and_its_tools_are_methods](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skill_manager.py#L224).

Repeated sign-in must use a fresh callback:
[successive OAuth flows](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_mcp_signin_lifetime.py#L125).
This is a local authentication-component test, not a live provider measurement.

<a id="s17"></a>

### S17 — Saved workspace preferences

Skills, model defaults and MCP startup configuration; credentials must remain
placeholders. Saving preferences is distinct from approving or authenticating a
connection.

[WorkspaceSettings](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/workspace/workspace_settings.py#L69);
[registration](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/workspace/options.py#L101);
[test_workspace_settings_mcp](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_workspace_settings_mcp.py#L1);
[test_skills_control_settings_scope](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_skills_control_settings_scope.py#L1).

<a id="s18"></a>

### S18 — Headless and benchmark hosts

run_task, Harbor AtomBenchAgent and native Gym AtomHarness; stop reasons, outputs,
benchmark guard and model-client attribution.

[packages/nooa-atom/src/nooa_atom/hosts/headless.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/hosts/headless.py#L1);
[packages/nooa-atom/src/nooa_atom/bench_agent.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/bench_agent.py#L1);
[packages/nooa-atom/src/nooa_atom/gym.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/gym.py#L1);
[packages/nooa-atom/src/nooa_atom/benchmark_policy.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/benchmark_policy.py#L1);
[test_headless_run](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_headless_run.py#L1);
[test_gym_policy_worker](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_gym_policy_worker.py#L1);
[test_bench_agent](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_bench_agent.py#L1).

<a id="s19"></a>

### S19 — Metrics and traces

ATOM-specific instrumentation counts hint/rejection/detachment events. Tests verify
event attribution; these are not task-success or recovery-rate measurements.

[packages/nooa-atom/src/nooa_atom/agent/metrics.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/metrics.py#L1);
[packages/nooa-atom/src/nooa_atom/session/tracing.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/session/tracing.py#L1);
[test_atom_metrics](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_atom_metrics.py#L1);
[test_child_session_tracing](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_child_session_tracing.py#L1);
[packages/nooa-bench/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-bench/README.md#L1).

New counters are `harness.atom.shell_detached_at_deadline.count` and
`harness.atom.shell_long_await.count`:
[long-await assertion](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_atom_metrics.py#L131)
and the public-agent shell handover tests in [S05](#s05). Long-await accounting occurs
when an inline run returns after more than 60 seconds; detached work and already-managed
jobs are not counted by that return path.

<a id="s20"></a>

### S20 — Media and library availability

Shared CodeAct media capture and model-format gates; benchmark dependencies are opt-in
and do not preload names or guarantee target-runtime imports.

[docs/media.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/docs/media.md#L1);
[src/nooa/runtime/media_capture.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/media_capture.py#L1);
[packages/nooa-atom/pyproject.toml](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/pyproject.toml#L1);
[benchmark profile](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/README.md#L14);
[test_managed_job_output_when_no_cell_is_active](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_session_namespace.py#L601).

<a id="s21"></a>

### S21 — Session/ACP lifecycle

Host submission/steering/cancellation, durable dispatch, recovery, settings and
presentation. These are host interfaces, not extra model tools.

[packages/nooa-atom/src/nooa_atom/session/session.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/session/session.py#L1);
[durable dispatch](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/README.md#L502);
[packages/nooa-atom/src/nooa_atom/acp/server.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/acp/server.py#L1);
[test_dispatch_invariants](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_dispatch_invariants.py#L1);
[test_recover](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_recover.py#L1);
[acp_adapter/test_questions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/acp_adapter/test_questions.py#L1);
[acp_adapter/test_verbosity](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/acp_adapter/test_verbosity.py#L1).

[ChannelReader](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/channels.py#L645).

Additional lifecycle evidence:
[shared turn loop](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/src/nooa/runtime/turn_loop.py#L1);
[idle readiness regression](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_turn_loop.py#L166);
[cancellation propagation tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/runtime/test_channels_queue_ergonomics.py#L517);
[non-locking owner probe](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_store.py#L457);
[parent watchdog](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/acp/worker.py#L78);
[process-group boundary](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/acp_router/test_worker.py#L156);
[POSIX import/feature boundaries](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_posix_imports.py#L1).
Owner probes read PID/host state without acquiring the session lock. On parent loss, a
group-leading worker kills its group; a worker that does not lead the group kills only
itself. Shared runtime tests are component evidence, not model behavior.

<a id="s22"></a>

### S22 — Existing framework capability cases

Framework-level examples and evaluation configuration; their presence does not establish
current ATOM coverage or identify the existing evaluation suite.

[tests/capability/config.yaml](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/capability/config.yaml#L1);
[tests/capability/agents/error_recovery.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/capability/agents/error_recovery.py#L1);
[tests/capability/agents/large_data.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/capability/agents/large_data.py#L1);
[tests/capability/agents/repl_exploration.py](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/tests/capability/agents/repl_exploration.py#L1).

<a id="s23"></a>

### S23 — Architecture history and benchmark caveats

The unified-work experiment records earlier interfaces and frozen results. It was
removed from main; the historical link below intentionally remains at
`31c7a4494c49df8649f5e4ea045077cfecedc6b1`. Its caveats prevent transferring historical
scores to this revision. The ATOM package README link follows the current source
revision and supplies current architecture/reflection documentation.

[experiments/unified-work/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/31c7a4494c49df8649f5e4ea045077cfecedc6b1/experiments/unified-work/README.md#L1);
[packages/nooa-atom/README.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/README.md#L1).

<a id="s24"></a>

### S24 — Ethos schema and authoring procedure

The ETHOS `author` field records document authorship; SPDX records copyright ownership.
Its source-derived draft status does not assert product-owner confirmation. Schema v1
requires an Open Questions heading; the section retains a placeholder while questions
are omitted from this document.

The Eval Author schema and validator are pinned to
`7ba81e6f954c52b3035762a64b11595397cbef2e`. Its ethos authoring instructions use schema
v1 and the standard-library validator. Immutable source links:

- [Orchestrate skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/team-skills/ethos/ethos-orchestrate/SKILL.md)
- [Explore skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/team-skills/ethos/ethos-explore/SKILL.md)
- [Write skill](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/team-skills/ethos/ethos-write/SKILL.md)
- [Template](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/team-skills/ethos/ethos-write/references/templates/ETHOS.md)
- [Validator](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/team-skills/ethos/ethos-write/scripts/validate_ethos.py)

The current
[Ethos confirmation adapter](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/skills/eval-author/references/ethos-confirmation.md)
accepts an explicitly selected agent-local `ETHOS.md`; no root copy or symlink is
required. It requires structural validation and explicit confirmation of the exact saved
content before downstream authoring. Its
[file validator integration](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/7ba81e6f954c52b3035762a64b11595397cbef2e/skills/eval-author/scripts/rubric_ethos.py)
uses the same writer validator and records content/validator digests.

These links establish the document/confirmation contract. ATOM's intended behavior
is derived from its package README ([S23](#s23)), public prompt ([S01](#s01)) and role
instructions ([S02](#s02)). Those sources support the mission, principles and partial
trade-offs in ETHOS; they do not establish a product roadmap, quantitative success
targets or product-owner confirmation.

<a id="s25"></a>

### S25 — Session reflection and host defaults

The factory registers `nemo.reflection` with `rename`, `compact`, `variables`, `todo`
and `vars` blocks, leaving the skill inactive in the model's advertised API. Root
sessions check it after committed turns and post an ordinary `system_messages` item when
due. Both 50 code cells and 60 minutes must pass by default; context fill is reported,
not a trigger. Pending waiting prompts defer reflection, its answering turn resets the
counters, and idle sessions do not wake just because time passes. Child sessions do not
post these messages.

`SessionOptions.reflection` overrides `atom.reflection.enabled`; absent either, ACP
enables it and other hosts disable it. `/reflection` persists on/off and block choices;
on/off clears a live session override. Saved configuration survives reload, while
counters start anew on construction. These messages prompt cleanup; they do not enforce
correct deletion, useful compaction or a memory bound.

This is independent of `experimental_done_reflection` (C29). Gym exposes the separate
`reflection` option; the native policy's ban on the experimental planner is not a ban on
these ordinary housekeeping turns. Its title restrictions still apply, so the reflection
`rename` block is not permission to bypass them.

[Reflection implementation and blocks](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/skills/reflection.py#L1);
[factory registration](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/agent/factory.py#L112);
[root-session hook](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/session/session.py#L848);
[component and scripted-session tests](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_reflection.py#L1);
[Gym option assertions](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/tests/test_gym_plain.py#L180);
[native policy validation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/packages/nooa-atom/src/nooa_atom/benchmark_policy.py#L38).

The test named `test_the_factory_activates_the_skill_with_the_core_blocks` actually
asserts that it is **not activated**; the assertions take precedence over the name.
Likewise, the headless/child-named test exercises a headless root; the child exclusion
is supported by the session hook's `parent_id` check, not by a child scenario in that
test.

## Validation of this documentation pass

- Evidence baseline: NOOA `107a65d77e4c2c644bb90553ac404bb4ed8a064c`.
- Runtime checked on 2026-11-24: main `d8b09515fcc35e7a9ce1a711d585890116fc812b`
  with this documentation-only change. Linux, CPython 3.13.15, `uv sync --frozen`:
  **370 tests passed in 148.60 seconds**, with 24 Gym policy RuntimeWarnings about
  middleware not wrapping synchronous, unwrapped or otherwise uninstrumented methods.
  This does not establish universal middleware enforcement or a filesystem sandbox.
- The schema-v1 validator and `rubric_ethos.check_ethos` at Eval Author
  `7ba81e6f954c52b3035762a64b11595397cbef2e` passed without warnings or diagnostics.
  Validator SHA-256: `11880291767d815b2a22891ee291f255e4cbda347318fc4ac1927fc801065d4b`.
- All 193 immutable source URLs were checked against Git objects for file existence
  and line bounds; 52 named-test anchors were checked against definitions. Local links,
  T01–T21, S01–S25, C01–C31, Markdown table widths and whitespace were checked.
- Ruff lint/format and SPDX license checks passed on the rebased branch. Subsequent
  prose-only changes were checked for schema validity, links and whitespace.

With `EVAL_AUTHOR_CHECKOUT` at the pinned Eval Author revision, validate from this
repository root:

```sh
uv run --frozen python "$EVAL_AUTHOR_CHECKOUT/team-skills/ethos/ethos-write/scripts/validate_ethos.py" docs/atom-eval-author/ETHOS.md
```

To reproduce the runtime selection, use the runtime revision above and run
`uv sync --frozen`, then:

```sh
uv run --frozen pytest \
  packages/nooa-atom/tests/test_builtin_skill_install.py \
  packages/nooa-atom/tests/test_unified_agent.py \
  packages/nooa-atom/tests/test_doc_contracts.py \
  packages/nooa-atom/tests/test_prompt_leaks.py \
  packages/nooa-atom/tests/test_skill_manager.py \
  packages/nooa-atom/tests/test_libs_persist.py \
  packages/nooa-atom/tests/test_reflection.py \
  packages/nooa-atom/tests/test_gym_plain.py \
  packages/nooa-atom/tests/test_gym_policy_worker.py \
  packages/nooa-atom/tests/test_session_namespace.py \
  packages/nooa-atom/tests/test_cell_print_cut.py \
  packages/nooa-atom/tests/test_work_shell.py \
  tests/test_skill_registry.py \
  tests/tools/test_library_writing_lib.py
```

This selection uses scripted/fake models and local fixtures. Other linked tests remain
source evidence only; no live-model evaluation or benchmark execution was performed.
Remote HTTP availability and the contents of cross-repository fixtures branch links
were not checked; those links are navigation, not execution pins.
