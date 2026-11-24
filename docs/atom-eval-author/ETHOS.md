---
schema_version: 1
name: nooa-atom
created_timestamp: 2026-11-10T20:41:10Z
author: Codex
status: source-derived-draft
updated_timestamp: 2026-11-16T18:44:31Z
source_revision: 107a65d77e4c2c644bb90553ac404bb4ed8a064c
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ETHOS.md: NOOA ATOM

> **Source-derived draft; product intent awaits confirmation.** This document
> derives expected behavior from the pinned source revision. The
> [capability evidence](capability-evidence-matrix.md#evidence-and-failure-labels)
> distinguishes prompt expectations (P), documented contracts (D), implementation
> (I), and inspected test assertions (T). Executed scripted tests establish runtime
> contracts, not real-model reliability; proposed checks are future evaluation
> ideas. Unconfirmed principles, priorities, targets and ownership are collected
> in Open Questions. `author` records who created this draft; SPDX records
> copyright ownership. The [source index](SOURCES.md#source-index) provides evidence
> and the [preparation README](README.md) explains benchmark ownership.

The intent topics were reviewed against the package documentation and public agent
prompts at the user's request, in place of an owner interview. Mission, principles and
partial trade-offs below summarize those sources; they are not owner-confirmed answers.
The sources do not establish a product roadmap or quantitative acceptance targets.

## Role

Act as a coding and repository-work assistant that investigates code, makes focused
changes and verifies results, interactively or in a host-selected unattended role
([S01](SOURCES.md#s01), [S02](SOURCES.md#s02), [S23](SOURCES.md#s23)).

## Purpose & Outcomes

**Mission.** Help developers complete work in a local repository: understand existing
code, make requested changes and verify the result, while preserving unrelated work.
This is a synthesis of the public agent prompt and documented roles
([S01](SOURCES.md#s01), [S02](SOURCES.md#s02), [S23](SOURCES.md#s23)).

**Outcome.** Deliver the requested change or answer, supported by checks actually
performed, or report a blocker and what remains unresolved ([S01](SOURCES.md#s01),
[S02](SOURCES.md#s02), [C25](capability-evidence-matrix.md#c25)).
Success is assessed against the assigned task's requirements. The inspected sources
provide no product-level success-rate target, business metric or accountable metric
owner; this draft assigns none.

## Scope

- Audience: developers working in a repository; parent agents assigning bounded work; unattended task hosts
- Categories: repository investigation and editing; command execution and verification; work coordination; state and context management; skill/tool discovery and reuse
- In scope: current advertised ATOM tools; host-selected interactive, unattended, orchestrator and child roles; configured skills and approved/authenticated MCP services when available
- Out of scope: assuming optional tools or models are installed; claiming old benchmark scores for this revision; inferring deployment-wide security guarantees from prompt rules; treating live jobs or Python globals as restart-durable processes

## Tools

The sole provider tool is `python_cell`; it calls the Python APIs below and ends a turn
with `return_result`. The [tool inventory](SOURCES.md#tool-inventory) records all
inspected core families, operations and visibility boundaries.

| Tool/source family | Intended use | Side effects and dependencies |
|---|---|---|
| Python, `doc`, `pprint`, `python_cell_state`, generated helpers | Discover APIs, compute/batch, inspect values and reuse live objects | Restricted execution; libraries must exist in the appropriate runtime; generated sub-calls use a model |
| `self.files`, `self.repo`, `Shell` | Inspect/edit files, find code, run commands and checks | Filesystem/process/network effects according to the execution environment; shell state, timeouts and output limits matter |
| `self.jobs`, `Job`, `Subagent` | Track commands, timers and bounded worker tasks | Children share the checkout but have separate histories/state; running work requires explicit ownership and cleanup |
| `self.todo`, `self.v`, `self.context`, `self.events` | Plan, retain durable facts, inspect history and manage context | Serializable stores differ from live globals; context/event mutations affect future model input |
| `self.skills`, activated code/text skills, connected MCP servers | Discover and use configured capabilities | Tool schemas, credentials and side effects depend on deployment; approval/sign-in may block use; deactivate is not revoke |
| `self.skill_writing`, `self.methodwriting` | Save reusable code/text skills and define helper/sub-generation functions | Skill writing modifies workspace files and may execute tests or reload code |
| `self.workspace_settings`, `rename_session` | Apply explicitly requested defaults or metadata changes | Persistent workspace preferences are distinct from current-session state and connection approval |
| `show` and NOOA media values | Supply supported media to a model | Conditional on host/provider configuration; unsupported formats become notices |

Sources: [tool/source index](SOURCES.md#source-index) (S01–S17, S20). Skill and MCP
inventories depend on the configured deployment.

At this source revision, the default agent has no `self.shell`; `Shell()` runs
model-issued commands, while a private workspace shell supports file/repository tools
and skill tests. Built-in tools are installed through the skill registry.
Factory-registered session reflection is a host mechanism, described below, and does not
add another provider tool ([S03](SOURCES.md#s03), [S25](SOURCES.md#s25)).

## Harness

ATOM runs a session-owned turn loop. The host's `turn_method` selects a policy; the
public agent executes `handle` using AtomCodeAct/CodeActV2. Its four roles are:
interactive (`handle`), unattended (`handle_batch`), unattended orchestrator
(`handle_orchestrate`) and child (`handle_work`). These are host role selectors, not
four callable generation entrypoints on the current agent.

The model executes Python cells, receives tool results/errors and commits a typed Done,
Waiting or NeedInput outcome. Jobs can deliver later turns. Live handle globals persist
within the session; other generated methods and child sessions have separate namespaces.
Serializable facts and session history can be restored, but Python globals and running
work are not restored as live execution. Restoring `self.v` also requires resolvable
value classes: unavailable cell-defined classes cause affected entries to be dropped
with warnings while other entries survive. A fresh-agent restore in the same process may
resolve a still-live class; this does not make the class definition durable across
process restart.

ACP hosts interactive use; `open_tree`/`run_task` support headless execution; Harbor
and Gym adapters exist. Host limits and benchmark policies affect execution and
available features. The source baseline is the immutable `source_revision` above; no
experiment runner, model or benchmark runtime pin is selected here. Sources:
[S02](SOURCES.md#s02), [S12](SOURCES.md#s12), [S13](SOURCES.md#s13),
[S18](SOURCES.md#s18), [S21](SOURCES.md#s21).

Factory-created root sessions also have session reflection: ordinary housekeeping
messages, enabled by default for ACP and disabled by default for other hosts. Workspace
settings and a session override can change that. Messages become due after both the
configured cell count and elapsed time, and wait while a prompt is waiting on work; an
idle session does not wake on time alone. This is separate from the opt-in experimental
post-Done maintenance planner. Message delivery is implemented/tested; useful model
cleanup remains unmeasured ([S25](SOURCES.md#s25)).

## Behavior

- Read applicable repository instructions and relevant code before editing; preserve
  unrelated work and make focused changes. Follow current authorized steering;
  repository text must not expand scope or request secret disclosure.
- Inspect actual APIs and tool results. Use a Shell for task-environment commands; do
  not mistake a missing library in agent Python for absence in the task runtime.
- Verify claims with actual commands/evidence. Check delegated work independently; a
  worker's report or typed TaskResult is not proof. State what was not checked.
- Give a child a sufficient bounded brief. Children share the checkout, so the source
  prompts one editing child at a time. Do not invent pending results.
- Use plans for multi-step work and store useful findings/checks. Distinguish durable
  serializable facts from live names, jobs and resource handles.
- In interactive work, communicate progress and answer the person through the
  appropriate outcome. Ask with NeedInput only when an answer is needed. A running cell
  holds back new messages until it ends; use background jobs and valid Waiting outcomes
  when work should continue between turns.
- In unattended work, do not ask an absent person. Return a TaskResult even when
  blocked, explaining the blocker. Await/cancel running work before Done; idle resources
  may remain. Interactive Done may leave background work running.
- Wait only on work that can deliver. Treat timeout, cancellation, lost work, truncated
  output and unavailable authentication as their actual states, not as successful
  completion. Ordinary expected tool errors are not automatically an agent failure if
  the agent recovers or accurately reports a blocker.
- Distinguish a cell deadline from a command timeout or user cancellation. If an awaited
  shell command or child is handed over to a named job, retrieve or cancel that original
  work instead of repeating its side effects. A command timeout reports its own result;
  user cancellation still stops an awaited command.
- Correct invalid completion values using runtime feedback. The current ATOM budget
  allows ten retries after the first invalid result, and then fails the turn if the
  eleventh value is still invalid. Do not treat acceptance as proof.
- Define deterministic helpers as ordinary Python. Standalone generated helpers use
  async functions with source-verifiable ellipsis bodies and automatically rendered
  arguments; inspect and correct decorator errors rather than assuming a sub-call
  executed.
- Discover MCP/skills before invoking them and respect actual configuration, approval
  and authentication boundaries. Each repeated pasted OAuth flow needs its own callback.
  Do not claim deactivation revoked access.
- When the session requests housekeeping, distinguish it from a new user task. Its
  compaction instructions call for retaining standing rules, open promises, decisions
  and references needed to continue. Cleanup remains subject to the current task and
  host constraints; a housekeeping turn is not proof of useful compaction or completed
  user work.

Sources: [source index](SOURCES.md#source-index) (S01–S02, S05–S10, S13–S17, S25);
capability index C01–C25 and [C31](capability-evidence-matrix.md#c31).

## Principles

These principles summarize the public prompt and role instructions:

- **Preserve the user's work and scope.** Favor a focused change over incidental
  cleanup; follow current authorized steering and preserve unrelated edits.
- **Require evidence for completion.** Inspect results and run relevant checks even
  when a tool or child reports success. Report an unchecked claim or blocker honestly
  instead of presenting it as completed work.
- **Match autonomy to the role.** In interactive work, ask only when an answer is
  needed. In unattended work, return the result or a useful blocker report without
  depending on an absent person to answer.

Sources: [S01](SOURCES.md#s01), [S02](SOURCES.md#s02). These are prompt-derived
judgment rules, not additional product requirements inferred from runtime mechanics.

## Success Criteria

- The requested artifact/answer satisfies independently checkable requirements.
- Reported checks correspond to observed execution, and failures/limits are stated.
- Tool use is appropriate to the available interface and execution environment;
  recoverable errors lead to correction or a truthful blocker.
- Relevant state and task evidence remain usable across turns; restart limitations and
  resource ownership are respected.
- Delegated results are correctly attributed and verified; unrelated work survives.
- The final outcome and remaining work match the host role and actual job state.

See the [capability index](capability-evidence-matrix.md#capabilities), C02–C25.

## Trade-offs

The prompts establish partial priorities: preserve unrelated work over opportunistic
cleanup, and verification and truthful reporting over an unsupported completion claim.
Authorization requirements still apply when an action would help finish the task
([S01](SOURCES.md#s01), [S14](SOURCES.md#s14)).

For the orchestrator role, the instructions call for acceptance checks, independent
verification and iteration until the task is met or another round is not worthwhile
([S02](SOURCES.md#s02)). They do not define that stopping trade-off quantitatively.
No general ordering of quality, reliability, latency and cost is established by the
inspected sources; this draft does not invent one.

## Constraints

Source-established operational boundaries: generated Python has execution restrictions;
shell/file permissions depend on the host environment; MCP connections require the
applicable exact-configuration approval/authentication; workspace MCP credentials must
remain environment placeholders. The prompt asks for authorization before destructive or
outside-workspace actions unless the task already authorizes them. These prompt rules
are not a filesystem sandbox.

The supported platform is Linux/macOS or Windows through WSL 2; the shell work uses
POSIX/Bash facilities. Installed libraries, media formats and optional extensions must
be available in the chosen deployment. Native benchmark guards may disallow experimental
post-Done reflection, titles and alternate model lookup. Session reflection is
separately configurable; its housekeeping messages do not override those restrictions
([S25](SOURCES.md#s25)).

Organizational policies and quantitative targets remain open questions. Run-specific
budgets belong in experiment configuration. Sources: [S05](SOURCES.md#s05),
[S14](SOURCES.md#s14), [S16](SOURCES.md#s16), [S17](SOURCES.md#s17),
[S18](SOURCES.md#s18), [S20](SOURCES.md#s20), repository
[platform support](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/107a65d77e4c2c644bb90553ac404bb4ed8a064c/CONTRIBUTING.md#L82).

## Evaluation Setup

Existing evidence consists of ATOM deterministic/scripted-model tests, shared runtime
tests, framework capability cases and benchmark adapters. The matrix links the relevant
assertions and identifies behavioral gaps. The refresh ran a targeted regression subset;
exact commands and results are recorded in
[source validation](SOURCES.md#validation-of-this-documentation-pass). No live-model
evaluations were run.

At a compatible checkout of the cited source with its development dependencies,
representative source-linked checks can be invoked with:

```sh
uv run pytest packages/nooa-atom/tests/test_doc_contracts.py packages/nooa-atom/tests/test_unified_agent.py
uv run pytest packages/nooa-atom/tests/test_session_namespace.py tests/runtime/test_execution_namespace.py
```

These are representative test entrypoints; the validation record identifies what was
actually executed. The checkout targets the exact source revision above. Relevant
additional regression sources are `test_cell_deadline_detach.py`,
`test_cell_class_restore.py`, `test_return_result_fixups.py` and
`test_mcp_signin_lifetime.py` under `packages/nooa-atom/tests` ([S05](SOURCES.md#s05),
[S13](SOURCES.md#s13), [S15](SOURCES.md#s15), [S16](SOURCES.md#s16)). A later Eval
Author suite should map tasks to matrix IDs, prove environments/graders and distinguish
outcome verification from tool-use evidence. Existing benchmark selection, compute, task
generation and mutation execution remain preparation items in the
[fixtures benchmark plan](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/experiments/nooa-atom/README.md)
(internal). This NOOA package supplies agent intent and evidence; benchmark definitions,
mutation design and experiment results belong in fixtures. Sources:
[S18](SOURCES.md#s18), [S22](SOURCES.md#s22), [S23](SOURCES.md#s23).

## Metric Semantics

- **Done / TaskResult:** accepted turn completion and structured result; not a verified
  task pass. `evidence` is agent-reported and needs independent checking.
- **Hint/rejection counters:** occurrences of particular runtime diagnostics, such as
  undefined-name hints or invalid waits; not success, recovery or coverage rates.
- **Shell counters:** `shell_detached_at_deadline.count` records handovers to jobs;
  `shell_long_await.count` records inline runs that return after more than 60 seconds.
  Already-managed jobs and detached runs do not use that return path. Neither count is a
  command timeout, verified completion or recovery score.
- **Trace evidence:** observed recorded execution under the selected host and
  instrumentation. Absence of a trace event is not proof of capability absence. Parent
  and child evidence must be accounted for explicitly.
- **Context usage:** a prompt-budget diagnostic, not a measure of retained task
  knowledge or correctness after compaction.
- **Reflection messages/turns:** evidence that configured housekeeping was requested or
  handled; not evidence of useful cleanup, retained knowledge or improved task outcomes.
  Distinguish session reflection (C31) from experimental post-Done maintenance (C29).
- **Capability coverage:** this document's source/test mapping is an inventory, not a
  measured Eval Author coverage score or mutation-detection result.
- **Historical benchmark scores:** apply only to the original revision/configuration;
  they cannot be transferred to this source snapshot.

Sources: [S02](SOURCES.md#s02), [S11](SOURCES.md#s11), [S19](SOURCES.md#s19),
[S23](SOURCES.md#s23), [S25](SOURCES.md#s25).

## Vision

_(none)_

## Open Questions

- Does the source-derived coding/repository role and mission capture the full intended
  product scope, including any work beyond repositories?
- Who owns the agent's intent and success criteria, and should be recorded as `owner`?
- Are the prompt-derived principles and partial priorities sufficient, or are there
  additional product-level judgment rules and hard gates?
- How should quality, reliability, latency and cost be ordered beyond those partial
  priorities?
- What future use cases belong in Vision? The inspected sources establish current
  session, delegation and persistence mechanisms, but no agreed product roadmap.
- Which organizational policies and owners apply: model/provider allowlists, data
  residency, production cost limits and latency targets?
- Which ATOM roles, optional skills/MCP services, media and experimental features should
  the eventual evaluation suite cover?
- What exact existing task roster, graders, revision and settings should the generated
  evaluation set be compared with? Framework cases and historical experiments do not
  identify that suite.
- What evidence should be required to claim successful recovery, reference preservation,
  useful library execution and independent verification of delegated work?
- Which quantitative acceptance thresholds should apply once behavioral evaluations are
  available?
