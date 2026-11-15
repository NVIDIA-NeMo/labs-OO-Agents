<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ATOM case example for Eval Author's metric benchmark

This is a **preparation example**, not a runnable or signed benchmark bundle.
The `.example` suffixes are intentional: tasks, installed mutant revisions,
model selection and the audit-author command are not available yet. The real
`description.json` is a draft intended-behavior input; it contains no coverage
scores or mutant outcomes. Complete it alongside the confirmed [Ethos](../ETHOS.md).
The example stays in NOOA with the [experiment plan](../README.md); Eval Author
consumes prepared external bundles, so no registry entry or new runner is needed.

## Contents and final bundle layout

- [experiment.yml.example](experiment.yml.example): three task-set cases sharing
  one ATOM agent manifest, local Harbor execution and the Eval Author metric adapter.
- [agents/atom/agent.yml.example](agents/atom/agent.yml.example): public ATOM baseline
  and one proposed shell-deadline mutant, using NOOA's actual Harbor adapter.
- [agents/atom/description.json](agents/atom/description.json): intended behavior
  for `case_description`. The metric format accepts arbitrary JSON; our author
  command must explicitly interpret these fields. There is no universal Ethos JSON schema.
- [tasks/tasks.yml.example](tasks/tasks.yml.example) and
  [tasks/subsets.json.example](tasks/subsets.json.example): format examples to
  instantiate separately for each selected task collection. Their task IDs are
  placeholders, not generated or existing benchmark tasks.

Materialize a bundle with this structure outside the repository's source tree:

```text
atom-bundle/
  experiment.yml
  agents/atom/
    agent.yml
    description.json
    baseline/                 # pinned prepared source/dependencies
    mutants/shell-deadline-loss/
  tasks/
    existing/tasks.yml, subsets.json, <task directories>/
    generated-zero/tasks.yml, subsets.json, <task directories>/
    generated-expanded/tasks.yml, subsets.json, <task directories>/
```

All manifest references are relative to `atom-bundle/`, even references inside
nested manifests. No absolute paths, traversal or symlinks are allowed for input
references. Copying `.example` files without preparing the referenced inputs will
fail full loading. Review the
[format contract](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/labs/metric-benchmark/docs/benchmark-format.md)
before changing field names. Agent revisions are strings; task/experiment revisions
are positive integers. A final bundle requires at least one actual mutant and task.

## Launch and evidence requirements

The starting host adapter is
[`harbor_adapter:NooaBenchAgent`](../../../examples/benchmarks/harbor_adapter.py),
with `agent_type: atom` and `enable_atif: true`. Make the pinned NOOA
`examples/benchmarks` directory importable in the process that runs Harbor.
This adapter installs the agent from `git_url`/`git_ref`; listing a source directory
under `inputs` does **not** install it or apply a patch. Each admitted mutant
therefore needs a reachable immutable Git commit, or a separately implemented
local-bundle installer. Verify the installed revision against the prepared source
and preserve dependency locks. The current example adapter resolves dependencies
during installation; it does not establish a reproducible locked environment.

The [`AtomBenchAgent`](../../../packages/nooa-atom/src/nooa_atom/bench_agent.py)
selects the public agent and runs unattended. Configure model/provider settings
identically across variants. Model aliases, credentials, time/turn limits,
container image and ATOM/Harbor compatibility must be checked at preparation time.
Do not copy the sample model or environment in NOOA's Harbor demo as an experimental
decision. The metric runner's backend is local Harbor 0.20.0; no new Helix support
is supplied by this example.

The metric adapter expects canonical ATIF at `agent/trajectory.json` in each native
baseline trial. ATOM's runner can export ATIF when enabled, while its ordinary
NOOA event dump is a different format. Check actual trace content and child-call
attribution, not just the filename. Keep task-verifier rewards and trial status;
ATOM's reported completion is not the grader. If the chosen benchmark uses Gym,
porting and trace conversion need independent validation before this Harbor bundle
can represent it faithfully.

## Coverage authoring boundary

Use the real
[Eval Author metric adapter](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/labs/metric-benchmark/metric_benchmark/eval_author.py).
Fill in absolute paths to its Python interpreter and `eval_author_coverage.py`.
Its settings require `eval_author_root`, `source_sha256` and `author_command`.
The supplied author command receives `--request request.json --output-dir authored`
and must produce `authored/audit.md` and `authored/judgments.json` plus any semantic
judgment sidecars. It interprets the draft description and permitted baseline
traces; the runner does not automatically run an authoring skill or read our
Markdown sources. When materializing the description, include the confirmed
Ethos text and selected capability/failure specifications in the JSON request
input, rather than assuming links will be dereferenced.

Do not replace the author command with a constant score, omit semantic judgments
while retaining their denominator, or reuse mutant outcomes as coverage evidence.
The adapter measures real audits/traces and reports unavailable values for missing
judgments or empty/unmeasured denominators. Audit creation is currently per subset;
there is no built-in shared-rubric pass. For this draft, retain that native method
and report its variability rather than promising a fixed-rubric experiment.

## Validation and eventual execution

From the pinned Eval Author checkout, with its dependencies installed, validate
an **actually completed** bundle without model calls:

```bash
PYTHONPATH=labs/metric-benchmark uv run python - <<'PYTHON'
from pathlib import Path
from metric_benchmark.config import load_benchmark

bundle = load_benchmark(Path('/path/to/atom-bundle/experiment.yml'), allow_unsigned=True)
print(bundle.definition.benchmark_id)
PYTHON
```

Obtain the coverage implementation pin, fill it into the settings, then sign the
prepared inputs and verify them:

```bash
uv run python labs/metric-benchmark/eval_author_coverage.py --print-source-digest /path/to/eval-author
uv run labs/metric-benchmark/tools/sha-sign /path/to/atom-bundle/experiment.yml
uv run labs/metric-benchmark/tools/sha-sign /path/to/atom-bundle/experiment.yml --check
```

Only after the baseline/mutant/trace checks and resource choices in the plan are
complete, run from that checkout (with NOOA's adapter importable):

```bash
PYTHONPATH=/path/to/pinned-nooa/examples/benchmarks uv run python labs/metric-benchmark/run_experiment.py \
  --benchmark /path/to/atom-bundle/experiment.yml \
  --run-id atom-comparison-001 --output /path/to/private-results/atom-comparison-001
```

The stages execute agents, compute signed mutation loss, compute coverage and
correlate within each case. The existing tool does not produce the cross-case
comparison or incremental detection report proposed in the plan; add those
analyses explicitly after agreeing their definitions. Follow the
[tool guide](https://github.com/NVIDIA-NeMo/labs-eval-author/blob/f2e3b78cd19bdee599a76e4d7bdf922595d6c192/labs/metric-benchmark/docs/tools.md)
for validated reuse of saved results. An input/configuration change requires a new
identity/output; do not re-sign completed inputs to relabel old observations.

No command above has executed the ATOM experiment. Format checks on these examples
validate shapes only, not missing references, model access, installed revisions,
ATIF validity or task quality.
