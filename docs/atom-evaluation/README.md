<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare ATOM for Eval Author

This package supplies source-grounded inputs for using Eval Author with NOOA's
public ATOM agent (`nooa_atom.agent:AtomAgent`). The Ethos describes intended
behavior for evaluation authoring; the inventory and matrix let reviewers trace
those expectations to code, documentation and test assertions.

- [ETHOS.md](ETHOS.md): schema-valid source-derived draft, awaiting product-intent confirmation.
- [Capability evidence matrix](capability-evidence-matrix.md): 30 stable capabilities, failure boundaries and proposed behavioral checks.
- [Source inventory](SOURCES.md): 21 tool families and 24 evidence groups with immutable source links.

Evidence was inspected at NOOA `fd62a602aea69192f1d1a312ba57ab612a014d15`.
Schema/link checks are documentation validation, not ATOM task-performance evidence.
No runtime changes, generated evaluation tasks or mutations are included here.

## Benchmark ownership

The benchmark lives in **nemo-eval-author-fixtures**, following its existing
`benchmark-dev/` layout, in [draft fixtures PR #180](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/pull/180).
See the [ATOM benchmark preparation](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/agents/nooa-atom/README.md),
[comparison plan](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/experiments/nooa-atom/README.md), and
[mutant design](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/ablation-tools/nooa-atom/README.md).
Eval Author supplies the authoring/audit workflows and metric runner; it does not
own this benchmark either.

The proposed comparison uses generated and existing evaluation sets against the
same ATOM baseline and independently qualified mutants. The fixtures PR owns
that design, its case/configuration examples, and future results. No mutants have
been created. Cross-repository branch links support joint draft review and must
be pinned to exact artifact commits/digests before experiment execution.

## Next iteration

Confirm the Ethos's mission, priorities, constraints and evaluation scope together.
Keep capability IDs stable while refining the evidence. Use the fixtures plan to
resolve the benchmark roster, mutation families, runtime integration and budgets.
A confirmed Ethos is necessary for Eval Author authoring; it does not replace
runtime, environment, verifier or evidence checks.
