# Atom execution trees in ACP clients

## Research question and design

Can Atom expose real method/tool relationships without changing its default ACP
presentation? Does the optional private extension remain compatible with the
paired Aion renderer from NOOA #347?

`demo.py` runs Atom's actual ACP router and workers with a scripted fake model.
A Python cell calls `DemoWorkflow.run`, whose verifier catches one deliberately
failed `DemoVerifier.check` before succeeding. The workflow writes and executes
`atom_tree_demo.py` using real Atom file and terminal tools. No model requests or
credentials are needed. Use a scratch workspace: the demo writes this file,
refuses to overwrite a different file, and checks its actual stdout/exit status.

`smoke.py` performs two prompts on each of two independent connections, first
with observation disabled, then enabled. It validates method/tool parent links,
all four node types, completion and the single handled failure, and the artifact.
Key metrics are tool-card/node counts, serialized session-update bytes, recovered
failure visibility, and reuse of one session across successive prompts.

## Run

From the NOOA checkout:

```bash
uv sync --frozen --extra atom
uv run --no-sync python experiments/atom_execution_tree/smoke.py
uv run --no-sync pytest packages/nooa-atom/tests/test_execution_tree.py
```

The smoke writes captures and `summary.json` under `results/` (ignored by Git).
To connect a client to the deterministic demo, register:

- Command: absolute path to this checkout's `.venv/bin/python`.
- Arguments: absolute path to `experiments/atom_execution_tree/demo.py`, followed
  by `--execution-tree`.
- Workspace: an empty scratch directory.

The default starts a router and one worker per root; `--single-process` also
works. The demo supports the same CLI options as `nooa atom`. Omit
`--execution-tree` to compare the default tool cards. For a live model, use
`.venv/bin/nooa` with arguments `atom --execution-tree`, and Atom's ordinary
workspace model configuration. This does not add a model picker or permissions.

The tree is visible only with **verbose** ACP activity (`/verbosity verbose`).
The flag is off by default. Pool needs no configuration change and should leave
it off. Pool can display enabled method observations as flat entries, but does
not use this private metadata to nest them.

## Paired Aion renderer

Use the renderer in [ryana/aion PR #1](https://github.com/ryana/aion/pull/1), pinned
at `c38d7e123b23c1f5f4823306efbdfc967a6f7b1b`. Its setup guide describes the
stock AionCore v0.2.2 backend and the forked WebUI. Replace the older NOOA
launcher with the Atom demo command above. Stock Aion does not implement this
extension. The wire namespace and fields are unchanged from #347.

## Results (2026-11-02)

Real ACP router/worker smoke completed all four prompts, including persistent
session reuse and a generated program whose output was `ATOM_TREE_OK`:

| Observation | Cards per turn | Execution nodes per turn | Deliberately failed nodes |
|---|---:|---:|---:|
| Off (default) | 5 | 0 | 0 |
| On | 14 | 13 | 1 |

Serialized session updates measured 17,381/3,958 bytes for the two default turns
and 29,208/16,120 bytes for the enabled turns. First-turn setup and ordinary tool
content contribute to those totals; these are scripted-demo wire measurements,
not runtime performance estimates.

The pinned Aion graph/React DOM suites plus the actual-producer compatibility test passed **36
tests**. The actual Atom capture formed one tree, retained the handled verifier
failure under a successful parent, displayed its error, and collapsed/expanded
**13 → 1 → 13** method/tool rows. The tests use Aion's actual stream reducer,
graph, tree and details components, with the existing IPC/translation/diff-card
mocks; camelCase wire fields are adapted to AionCore's stored field names. The
full AionCore/browser application was not rerun. No frontend source is changed.

Lifecycle/transport regressions cover two prompts, provider ID reuse, transient
agent collection, thread delivery, cross-turn generators, prepare-time and
committed cancellation, independent child cancellation, bounded ancestry,
tracing-hook composition, attachment/verbosity transitions and both direct
stdio and router/worker metadata round trips. See the package tests for the
current executable assertions.


## Reproduce renderer compatibility

Copy `aion_compat.dom.test.tsx` into the pinned Aion fork's
`tests/unit/executionTree/` directory (with dependencies installed according to
that fork's instructions). Set `ATOM_TREE_CAPTURE` to the absolute path of this
experiment's `results/tree-0.json` and run its Vitest execution-tree suite:

```bash
ATOM_TREE_CAPTURE=/absolute/NOOA/experiments/atom_execution_tree/results/tree-0.json \
  bun run test -- tests/unit/executionTree
```

The supplied test uses the fork's real stream reducer and components; its mocks
match the focused DOM suite. The original fork suites plus this supplied
actual-producer compatibility test comprise 36 tests.
