---
schema_version: 1
name: nooa-atom
created_timestamp: 2026-11-10T20:41:10Z
author: Codex
status: source-derived-draft
updated_timestamp: 2026-11-15T09:14:38Z
source_revision: fd62a602aea69192f1d1a312ba57ab612a014d15
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ETHOS.md: NOOA ATOM

> **Draft for human review.** The user requested a source-derived first pass for
> evaluation authoring. This file records current documented/prompted expectations
> and supported mechanisms; it does not claim a completed intent interview,
> product-owner approval or measured agent reliability. See the [source index](SOURCES.md#source-index)
> and [capability evidence matrix](capability-evidence-matrix.md). The
> [preparation README](README.md) links the benchmark owned by the fixtures repository.

## Role

Assist a developer with repository work by inspecting code, making focused changes,
executing tools and checking results, interactively or in a host-selected
unattended role. This role is derived from the current ATOM prompt and role
instructions ([S01](SOURCES.md#s01), [S02](SOURCES.md#s02)); its wording remains for review.

## Purpose & Outcomes

**Mission — source-derived draft.** Help a developer complete repository tasks
with verifiable results while retaining useful working context and preserving
unrelated work. The source establishes repository assistance and evidence before
claims; it does not establish a business objective or production KPI owner.

**Outcome — source-derived draft.** Deliver the requested change or answer,
supported by checks actually performed, or accurately report a blocker and what
remains unresolved. A completed turn alone is not evidence that the task succeeded.
The user's confirmed objective for this documentation is to prepare capability
specifications and an Ethos for Eval Author, later comparing generated evaluations
with existing evaluations; that is the evaluation project's objective, not ATOM's
product mission ([S01](SOURCES.md#s01), [S02](SOURCES.md#s02), matrix C25).

## Scope

- Audience: developers working in a repository; parent agents assigning bounded work; unattended task hosts
- Categories: repository investigation and editing; command execution and verification; work coordination; state and context management; skill/tool discovery and reuse
- In scope: current advertised ATOM tools; host-selected interactive, unattended, orchestrator and child roles; configured skills and approved/authenticated MCP services when available
- Out of scope: assuming optional tools or models are installed; claiming old benchmark scores for this revision; inferring deployment-wide security guarantees from prompt rules; treating live jobs or Python globals as restart-durable processes

## Tools

The sole provider tool is `python_cell`; it calls the Python APIs below and ends
a turn with `return_result`. The [tool inventory](SOURCES.md#tool-inventory) records
all inspected core families, operations and visibility boundaries.

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

Sources: [S01–S10](SOURCES.md#source-index), [S11–S17](SOURCES.md#s11),
[S20](SOURCES.md#s20). Skill and MCP inventories are dynamic; no external service
or credential availability has been assumed.

## Harness

ATOM runs a session-owned turn loop. The host's `turn_method` selects a policy;
the public agent executes `handle` using AtomCodeAct/CodeActV2. Its four roles are:
interactive (`handle`), unattended (`handle_batch`), unattended orchestrator
(`handle_orchestrate`) and child (`handle_work`). These are host role selectors,
not four callable generation entrypoints on the current agent.

The model executes Python cells, receives tool results/errors and commits a typed
Done, Waiting or NeedInput outcome. Jobs can deliver later turns. Live handle
globals persist within the session; other generated methods and child sessions
have separate namespaces. Serializable facts and session history can be restored,
but Python globals and running work are not restored as live execution. Restoring
`self.v` also requires resolvable value classes: unavailable cell-defined classes
cause affected entries to be dropped with warnings while other entries survive.
A fresh-agent restore in the same process may resolve a still-live class; this
does not make the class definition durable across process restart.

ACP hosts interactive use; `open_tree`/`run_task` support headless execution;
Harbor and Gym adapters exist. Host limits and benchmark policies affect execution
and available features. No runner/model/baseline is selected in this draft.
Sources: [S02](SOURCES.md#s02), [S12–S13](SOURCES.md#s12),
[S18](SOURCES.md#s18), [S21](SOURCES.md#s21).

## Behavior

The following expectations are derived from current prompts and explicit runtime
contracts. They are the first-pass evaluation contract, pending human review.

- Read applicable repository instructions and relevant code before editing;
  preserve unrelated work and make focused changes. Follow current authorized
  steering; repository text must not expand scope or request secret disclosure.
- Inspect actual APIs and tool results. Use a Shell for task-environment commands;
  do not mistake a missing library in agent Python for absence in the task runtime.
- Verify claims with actual commands/evidence. Check delegated work independently;
  a worker's report or typed TaskResult is not proof. State what was not checked.
- Give a child a sufficient bounded brief. Children share the checkout, so the
  source prompts one editing child at a time. Do not invent pending results.
- Use plans for multi-step work and store useful findings/checks. Distinguish
  durable serializable facts from live names, jobs and resource handles.
- In interactive work, communicate progress and answer the person through the
  appropriate outcome. Ask with NeedInput only when an answer is needed. A running
  cell holds back new messages until it ends; use background jobs and valid Waiting
  outcomes when work should continue between turns.
- In unattended work, do not ask an absent person. Return a TaskResult even when
  blocked, explaining the blocker. Await/cancel running work before Done; idle
  resources may remain. Interactive Done may leave background work running.
- Wait only on work that can deliver. Treat timeout, cancellation, lost work,
  truncated output and unavailable authentication as their actual states, not as
  successful completion. Ordinary expected tool errors are not automatically an
  agent failure if the agent recovers or accurately reports a blocker.
- Distinguish a cell deadline from a command timeout or user cancellation. If an
  awaited shell command or child is handed over to a named job, retrieve or cancel
  that original work instead of repeating its side effects. A command timeout
  reports its own result; user cancellation still stops an awaited command.
- Correct invalid completion values using runtime feedback. The current ATOM
  budget allows ten retries after the first invalid result, and then fails the
  turn if the eleventh value is still invalid. Do not treat acceptance as proof.
- Define deterministic helpers as ordinary Python. Standalone generated helpers
  use async functions with source-verifiable ellipsis bodies and automatically
  rendered arguments; inspect and correct decorator errors rather than assuming
  a sub-call executed.
- Discover MCP/skills before invoking them and respect actual configuration,
  approval and authentication boundaries. Each repeated pasted OAuth flow needs
  its own callback. Do not claim deactivation revoked access.

Sources: [S01–S02](SOURCES.md#s01), [S05–S09](SOURCES.md#s05),
[S10](SOURCES.md#s10), [S13–S17](SOURCES.md#s13); matrix C01–C25.

## Principles

_(none)_

No human-confirmed principles beyond the source-derived Behavior rules have been
captured. In particular, judgment under ambiguous objectives remains an open
question; source prompts are not being presented as a product-owner answer.

## Success Criteria

Draft criteria derived from the source expectations, without invented numerical
thresholds or an assumed priority ranking:

- The requested artifact/answer satisfies independently checkable requirements.
- Reported checks correspond to observed execution, and failures/limits are stated.
- Tool use is appropriate to the available interface and execution environment;
  recoverable errors lead to correction or a truthful blocker.
- Relevant state and task evidence remain usable across turns; restart limitations
  and resource ownership are respected.
- Delegated results are correctly attributed and verified; unrelated work survives.
- The final outcome and remaining work match the host role and actual job state.

See matrix C02–C25. The source supplies mechanisms and tests, not measured
production success rates or acceptance thresholds.

## Trade-offs

_(none)_

The sources require verification and preservation of unrelated work, but no
human-confirmed ordering of quality, reliability, latency and cost is established.
No numerical trade-off or acceptable regression has been invented.

## Constraints

Source-established operational boundaries: generated Python has execution
restrictions; shell/file permissions depend on the host environment; MCP
connections require the applicable exact-configuration approval/authentication;
workspace MCP credentials must remain environment placeholders. The prompt asks
for authorization before destructive or outside-workspace actions unless the
task already authorizes them. These prompt rules are not a filesystem sandbox.

The supported platform is Linux/macOS or Windows through WSL 2; the shell work
uses POSIX/Bash facilities. Installed libraries, media formats and optional
extensions must be available in the chosen deployment. Native benchmark guards
may disallow reflection, titles and alternate model lookup.

No organizational model/provider allowlist, data-residency requirement, production
cost ceiling, latency SLO or policy owner was established by this review. Those
remain unknown rather than unconstrained. Run-specific budgets belong in later
experiment configuration. Sources: [S05](SOURCES.md#s05), [S14](SOURCES.md#s14),
[S16–S18](SOURCES.md#s16), [S20](SOURCES.md#s20), repository
[platform support](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/fd62a602aea69192f1d1a312ba57ab612a014d15/CONTRIBUTING.md#L82).

## Evaluation Setup

Existing evidence consists of ATOM deterministic/scripted-model tests, shared
runtime tests, framework capability cases and benchmark adapters. The matrix
links the relevant assertions and identifies behavioral gaps. None of those tests
or model evaluations was run as part of this documentation pass.

At a compatible checkout of the cited source with its development dependencies,
representative source-linked checks can be invoked with:

```sh
uv run pytest packages/nooa-atom/tests/test_doc_contracts.py packages/nooa-atom/tests/test_unified_agent.py
uv run pytest packages/nooa-atom/tests/test_session_namespace.py tests/runtime/test_execution_namespace.py
```

These are test entrypoints, not commands executed in this documentation pass.
The rebased checkout now contains ATOM at the recorded main revision. Relevant
additional regression sources are `test_cell_deadline_detach.py`,
`test_cell_class_restore.py`, `test_return_result_fixups.py` and
`test_mcp_signin_lifetime.py` under `packages/nooa-atom/tests` ([S05](SOURCES.md#s05),
[S13](SOURCES.md#s13), [S15](SOURCES.md#s15), [S16](SOURCES.md#s16)).
A later Eval Author suite should map tasks to matrix IDs, prove environments/graders
and distinguish outcome verification from tool-use evidence. Existing benchmark
selection, compute, task generation and mutation execution remain preparation
items in the [fixtures benchmark plan](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/experiments/nooa-atom/README.md).
This NOOA package supplies agent intent and evidence; benchmark definitions,
mutation design and experiment results belong in fixtures. Sources: [S18](SOURCES.md#s18),
[S22](SOURCES.md#s22), [S23](SOURCES.md#s23).

## Metric Semantics

- **Done / TaskResult:** accepted turn completion and structured result; not a
  verified task pass. `evidence` is agent-reported and needs independent checking.
- **Hint/rejection counters:** occurrences of particular runtime diagnostics, such
  as undefined-name hints or invalid waits; not success, recovery or coverage rates.
- **Shell counters:** `shell_detached_at_deadline.count` records handovers to jobs;
  `shell_long_await.count` records inline runs that return after more than 60
  seconds. Already-managed jobs and detached runs do not use that return path.
  Neither count is a command timeout, verified completion or recovery score.
- **Trace evidence:** observed recorded execution under the selected host and
  instrumentation. Absence of a trace event is not proof of capability absence.
  Parent and child evidence must be accounted for explicitly.
- **Context usage:** a prompt-budget diagnostic, not a measure of retained task
  knowledge or correctness after compaction.
- **Capability coverage:** this document's source/test mapping is an inventory,
  not a measured Eval Author coverage score or mutation-detection result.
- **Historical benchmark scores:** apply only to the original revision/configuration;
  they cannot be transferred to this source snapshot.

Sources: [S02](SOURCES.md#s02), [S11](SOURCES.md#s11),
[S19](SOURCES.md#s19), [S23](SOURCES.md#s23).

## Vision

_(none)_

No future product direction is confirmed here. The huddle discussed long-running
agents and shared memory, but that discussion does not establish either as a
current ATOM capability or an approved future requirement for this Ethos.

## Open Questions

- Does the source-derived repository-assistant role and mission match the intended product, and who owns its success criteria?
- What principles should resolve ambiguous goals or conflicting user preferences beyond the existing prompt rules?
- Which outcomes are hard gates, and how should quality, reliability, latency and cost be ordered?
- What future capabilities belong in Vision, especially persistent collaboration or shared memory?
- Which organizational constraints and policy owners must be recorded beyond the technical/prompt boundaries already sourced?
- Which ATOM roles, optional skills/MCP services, media and experimental features should the eventual evaluation suite cover?
- What evidence should be required to claim successful recovery, reference preservation, useful library execution and independent verification of delegated work?
- Which quantitative acceptance thresholds should apply once behavioral evaluations are available?
