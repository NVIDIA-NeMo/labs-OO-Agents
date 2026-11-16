<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare ATOM for Eval Author

[Eval Author](https://github.com/NVIDIA-NeMo/labs-eval-author) provides workflows for
creating and assessing evaluations of AI agents. An ETHOS describes an agent's intended
purpose, capabilities, behavior and boundaries so Eval Author can generate evaluations
against those expectations. We are preparing this ETHOS and its supporting capability
evidence for NOOA's public ATOM agent (`nooa_atom.agent:AtomAgent`) to run experiments
that determine whether Eval Author can generate a more effective evaluation set than
the one we currently use.

- [ETHOS.md](ETHOS.md): schema-valid source-derived draft, awaiting product-intent
  confirmation.
- [Capability evidence matrix](capability-evidence-matrix.md): 31 stable capabilities,
  failure boundaries and proposed behavioral checks.
- [Source inventory](SOURCES.md): 21 tool families and 25 evidence groups with immutable
  source links.

Evidence targets NOOA `107a65d77e4c2c644bb90553ac404bb4ed8a064c`, fetched main on
2026-11-16. The documentation branch was already based on that commit; the refresh
reconciles source changes since the prior evidence revision,
`fd62a602aea69192f1d1a312ba57ab612a014d15`, rather than treating the rebase as review.
It covers removal of the legacy shell attribute, built-in skill installation, and
session reflection separately from experimental post-Done maintenance. This is the
documentation's source baseline, not an agreed benchmark runtime pin. See
[validation](SOURCES.md#validation-of-this-documentation-pass) for the current Eval
Author contract, checks run and their limits. Schema/link checks and scripted regression
tests are not ATOM task-performance evidence. No runtime changes, generated evaluation
tasks or mutations are included here.

## Benchmark ownership

The benchmark lives in **nemo-eval-author-fixtures**, following its existing
`benchmark-dev/` layout, in
[draft fixtures PR #180](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/pull/180)
(internal). See the
[ATOM benchmark preparation](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/agents/nooa-atom/README.md)
(internal),
[comparison plan](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/benchmark-dev/experiments/nooa-atom/README.md)
(internal), and
[mutant design](https://github.com/NVIDIA-dev/nemo-eval-author-fixtures/blob/docs/atom-benchmark-preparation/ablation-tools/nooa-atom/README.md)
(internal). [Eval Author](https://github.com/NVIDIA-NeMo/labs-eval-author) supplies the
authoring/audit workflows and metric runner. Fixtures links require internal repository
access; this package contains the agent intent and evidence needed for its own review.

The proposed comparison uses generated and existing evaluation sets against the same
ATOM baseline and independently qualified mutants. The fixtures PR owns that design, its
case/configuration examples, and future results. Mutation proposals there remain
unapproved draft material; this refresh neither uses them to derive intent nor qualifies
them. Cross-repository branch links support joint draft review and must be pinned to
exact artifact commits/digests before experiment execution.

## Next iteration

Review the source-derived mission, principles and partial priorities in the ETHOS.
These answers were gathered from ATOM's documentation and prompts in place of an owner
interview; unresolved product intent remains in Open Questions. Keep capability IDs
stable while refining the evidence. Use the fixtures plan to resolve the benchmark
roster, mutation families, runtime integration and budgets. A confirmed Ethos is
necessary for Eval Author authoring; it does not replace runtime, environment, verifier
or evidence checks.
