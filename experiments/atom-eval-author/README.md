<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ATOM evaluations with Eval Author

**Status: draft plan and initial inputs, for joint iteration. No evaluation tasks
have been generated and no mutation experiment has run.**

We are preparing NOOA's public ATOM agent for Eval Author. The first contribution
is a source-grounded `ETHOS.md`: it gives Eval Author an explicit account of the
agent's intended behavior from which to author and audit evaluations. A valid
Ethos file is a necessary input, not proof that the runtime, graders or authoring
workflow are ready. Its product intent still needs confirmation.

The experiment will compare Eval Author-generated evaluations with the benchmark
currently used to evaluate ATOM, using controlled agent mutations as an approximate
reference for regression-detection quality. It will also package ATOM as a case
example for Eval Author's metric benchmark. The case lives here because that
benchmark consumes externally prepared cases; a separate Eval Author change is
not required to start this draft.

## Artifacts and completed work

| Artifact | What is ready | What remains |
|---|---|---|
| [ETHOS.md](ETHOS.md) | Schema-valid source-derived intent draft | Confirm mission, priorities, constraints and selected evaluation scope |
| [SOURCES.md](SOURCES.md) | 21 tool families, 24 source groups and immutable code/test links | Inventory deployment-specific skills/MCP services if included |
| [Capability evidence matrix](capability-evidence-matrix.md) | 30 stable capability IDs, failure boundaries and proposed checks | Turn selected rows into independently graded behavioral tasks |
| This plan | Proposed comparison, metrics, mutation controls and run sequence | Agree on the existing benchmark, sizes, budget and acceptance criteria |
| [ATOM case example](case/README.md) | Experiment/agent/task/subset templates and intended-behavior JSON | Materialize real tasks and mutants, validate launch and trace export, then pin inputs |

The evidence snapshot is NOOA `fd62a602aea69192f1d1a312ba57ab612a014d15`
(main inspected on 2026-11-11). Eval Author's metric format was inspected at
`f2e3b78cd19bdee599a76e4d7bdf922595d6c192`. These are provenance records;
select and verify actual execution revisions before freezing the experiment.
The source inventory distinguishes inspected test assertions from executed tests
and real-model evidence. No runtime success is inferred from schema validation.

## Research questions

1. With the same ATOM baseline, model, runtime limits and mutant cohort, do
   generated tests detect regressions that the existing benchmark misses?
2. Does expanding the existing suite add useful detection beyond simply spending
   more task executions? Compare both full suites and equal-size subsets.
3. Across fixed subsets, does Eval Author's measured coverage track mutation
   sensitivity, using the metric benchmark's existing analysis?

Mutations give a ground-truth-like signal for faults we deliberately introduce
and independently verify. They do not define all good behavior. Results remain
conditional on the selected faults, task difficulty, verifiers and model noise.
A suite that scores highly on these mutants may still miss production failures.

## Experiment design

Start with unattended `handle_batch`, the default public ATOM benchmark path.
Interactive steering, multi-agent orchestration, optional MCP/media and reflection
are separately tagged follow-up tracks; include them only with a host that can
exercise and grade them. Do not make absent optional services automatic failures.

### Evaluation sets

| Case ID | Task set | Authoring access |
|---|---|---|
| `atom-existing` | Exact existing benchmark roster and verifiers, supplied/confirmed by its owner | Preserve its original definition; record any adapter changes |
| `atom-generated-zero` | New suite generated from confirmed Ethos, source evidence and environment contracts | No existing tasks, baseline traces, mutant patches or mutant results; label this the specification-only arm |
| `atom-generated-expanded` | Existing suite plus new tests authored from its measured gaps | Existing tasks and baseline traces allowed; no mutant patches or results |

The huddle mentions harness capability bench, Terminal-Bench and SWE-bench.
Those names are candidate context, **not a selected roster**. Obtain the exact
benchmark version, task IDs, grading rules, model and historical run settings
before claiming a comparison to the benchmark already in use.

Fix the task inventories and subset membership before inspecting mutant outcomes.
Use the same mutant cohort in every case. Include a full-suite subset and multiple
preselected subsets at agreed equal task counts; report costs because equal task
counts need not mean equal cost. Record sampling seed, selection rule and exact
memberships. Keep copied existing tasks identical in the expanded arm and report
the new tasks' contribution separately. Remove accidental duplicates before
freezing, using a recorded rule rather than results-driven selection.

### Mutants and controls

Prepare one small, attributable fault per mutant on the same baseline revision.
These are **candidates, not implemented mutants or claims of open bugs**:

| Candidate | Capability IDs | Intended fault | Independent activation/control evidence |
|---|---|---|---|
| `shell-deadline-loss` | C06 | Cancel a command at the cell deadline instead of handing it to a job | A short-deadline fixture loses the original result; normal short commands still work |
| `stale-edit-accepted` | C04 | Bypass stale Match protection | An intervening-edit fixture changes the wrong region; a fresh-anchor edit still works |
| `live-namespace-reset` | C13 | Discard handle globals between turns | A deterministic two-turn fixture loses a stored name; a one-turn task still works |
| `child-error-hidden` | C09–C11 | Suppress a child's failure notification | A failing-child fixture loses the expected error; a successful child still returns its result |
| `missing-library-hint` | C16–C17 | Remove the task-environment recovery hint | The diagnostic fixture changes; ordinary installed imports still work |

Use mutation controls that are separate from the evaluated suites. Record the
patch, expected effect, activation check, unaffected control, and admission
reason. Exclude mutants that do not start, mutate the verifier/environment,
fail to activate, or are equivalent for the scoped behavior; disclose every
exclusion. A valid mutant's zero or negative task effect stays in the results.
Do not discard a valid mutant because the generated suite performs poorly on it.

Balance candidates across the agreed capability groups, including behavior the
existing suite targets. Keep an independent held-out mutant cohort if we tune
prompts or tasks using development mutants. Freeze both cohorts before scoring;
report development and held-out results separately. The authoring process may
see public failure specifications and regression-test sources, but must not see
the held-out patches, activation fixtures or measured outcomes. Record that source
access so this is not misrepresented as a blind generalization study.

### Controlled execution

Pin model/provider settings, task images, dependency locks, ATOM/adapter revisions,
turn/time/token limits, workspace initialization and credentials policy across
arms and agent variants. Start each task with a fresh session/workspace. Verify
that each mutant actually installs its own pinned revision. The task's verifier,
not `Done`, `TaskResult` or ATOM's reported `success`, decides reward.

The metric runner currently runs one trial per agent/task pair, baseline first,
with agent jobs sequential and task concurrency configurable. With M mutants and
T tasks, one case costs `(M + 1) × T` executions. Cases with overlapping tasks
currently repeat them; budget the sum across cases. If variability needs repeated
runs, use distinct run/output identities and report them separately: the current
scorer does not pool repetitions. A combined estimate needs an explicitly agreed
analysis extension, not silently selected attempts.

## Key metrics and interpretation

The [metric benchmark stage contract](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/labs/metric-benchmark/docs/stages.md)
defines the primary mutation score. For higher-is-better reward:

```text
S(agent, subset) = mean task reward on that subset
M(subset) = mean over mutants [S(baseline, subset) - S(mutant, subset)]
```

Compare `M` on the full suites and the equal-size subsets. Preserve signed effects:
a negative value is an inspection signal, not zero. Report baseline success and
individual mutant effects so baseline failure floors and a single easy-to-kill
mutant do not disappear in the mean. Agree on task reward scales before comparing
collections; do not average incomparable rewards without a documented transform.

Supplementary reporting should show per-mutant detection and incremental detections
from added tasks. For binary rewards, a proposed detection rule is at least one
baseline-pass/mutant-fail pair on the subset. This rule and its noise/repetition
policy need confirmation and **are not a built-in metric-benchmark output**.
Report runtime failures, missing evidence, execution cost and wall time separately.
Do not reinterpret missing/failed executions as legitimate zero rewards.

For the metric-validity question, the existing pipeline computes coverage on each
subset's baseline traces and pairs it with that subset's mutant score. It reports
Spearman correlation with average ranks for ties, separately per case. Keep
mutant evidence inaccessible to coverage authoring; the request omits it but the
local process is not sandboxed. Coverage is an observed metric under study, not
the definition of test quality. Use the same confirmed behavior description,
measurement methods and authoring settings in all cases.

Keep the default missing-pair policy `error`. If exclusions are justified later,
report their reasons and the included count. Overlapping subsets are dependent;
correlation is descriptive, not a significance test. Constant scores or too few
pairs yield an undefined result. Do not pool case scores or claim confidence
intervals that the current tooling does not compute.

## How to prepare and run

1. Review this plan, the Ethos and capability matrix together. Resolve the choices
   below and confirm Ethos through Eval Author's existing workflow before authoring.
2. Select the existing benchmark roster and runtime integration. Materialize tasks
   with their verifiers; if porting Gym tasks to Harbor, establish grading parity
   or select a compatible execution/results adapter explicitly.
3. Generate the specification-only suite and the expansion in separate authoring
   workspaces with recorded inputs/settings. Prove each environment and verifier
   with a known-correct solution and negative control; retain rejected tasks.
4. Prepare and admit the mutant cohort using independent activation/control tests.
   Freeze agent versions, task sets and subsets before scoring.
5. Complete the [case templates and launch prerequisites](case/README.md). Exercise
   one baseline task and one mutant, then check rewards and canonical ATIF evidence.
   Artifact identity checks do not substitute for this runtime check.
6. Sign the completed bundle and run Eval Author's four stages. Commands and
   required adapter settings are in the case guide. Capture native evidence,
   exclusions, costs, mutant effects and per-case correlation outputs.
7. Update this README with quantitative results, limitations and artifact locations.
   Review failed cases before changing tasks; subsequent changes form a new version.

This PR needs no model credentials or Docker to review. Compute choice and
interactive versus unattended authoring remain open. The metric runner supports
local Harbor 0.20.0; its docs list Helix as future work. Interactive authoring can
produce a bundle for later unattended execution, provided the actual audit-author
command and semantic-judgment outputs are implemented and checked.

## Decisions for the next iteration

- Which exact existing benchmark, roster and owner-confirmed settings are the comparator?
- Does the source-derived Ethos describe the intended product, and which capabilities are hard gates?
- What generated-suite sizes, subset sizes/counts, mutation families and held-out split should we freeze?
- Which model, execution platform, budgets and repeat policy make the comparison useful and affordable?
- Which threshold would justify calling a generated or expanded suite better, given baseline difficulty and cost?
- Can the existing benchmark run unchanged through Harbor, and do ATOM/child traces satisfy the chosen coverage methods?

## Results summary

**No generated tasks, admitted mutants, ATOM runs, coverage measurements or
correlation results yet.** This contribution supplies the plan, source evidence,
Ethos draft and case-format examples. Documentation/format validation is recorded
in the PR; it is not experimental evidence.

After runs, record: pinned inputs and roster sizes; baseline pass rates; valid and
excluded mutant counts; full-suite and equal-size scores; individual effects and
negative diagnostics; incremental detections; coverage/correlation with included
subset counts; missing observations; cost/time; and limitations. Keep raw traces,
credentials and large run directories outside Git, with stable artifact references.
