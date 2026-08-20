# Bench-agent context audit and design

## Goal

Build two benchmark agents whose prompts expose only actionable state:

1. `BenchAgent`: a compact single-agent baseline using CodeAct.
2. `RLMBenchAgent`: the same controller plus bounded, context-isolated coding workers.

The variants intentionally share task parsing, tools, result schema, and automatic
history summarization so benchmark differences measure delegation rather than prompt
or harness drift.

## Context-block audit

| Existing block/surface | Decision | Reason |
|---|---|---|
| `task` dynamic block | Remove | `_solve_task(description)` already places the task in the method prompt. Duplicating it spends tokens and can create stale state. |
| `instructions` | Keep when supplied | Benchmark-specific constraints are authoritative and cannot safely be inferred. |
| `initial_observation` | Keep when supplied | It may contain environment state available only at task start. |
| `python_tools` | Keep, narrowed to `ShellTools` and `RepoTools` | Generated code needs signatures for the two primary capabilities. |
| `todo` static API documentation | Remove | The class method prompt and exposed typed `todo` attribute are enough; full API documentation is disproportionate. |
| `todo_status` | Keep, status only | A compact live plan prevents abandoned work across long runs. It renders IDs, dependencies, small vars, and notes, but not comment journals. Do not pre-seed a ritual todo or duplicate the full API documentation. |
| `context_usage` | Remove from benchmark agents | Automatic summarization owns the budget, so per-turn metrics are not actionable benchmark context. The interactive `CodingAgent` retains metrics for host observability. |
| public `context` / `events` APIs | Hide | They invite prompt surgery and manual compaction instead of task work. Automatic summarization owns history maintenance. |
| repeated tool/workflow prose | Collapse | State read-before-edit, minimum scope, verification, and return contract once. Tool signatures already document mechanics. |

## Prompt principles adopted

From Prime Agent: preserve a persistent controller notebook; delegate only
self-contained, context-heavy work; make delegation requests explicit; inherit the
controller model; return concise findings rather than child transcripts; retain
recent work when compacting.

From AVO: read before editing, make the smallest sufficient change, parallelize only
independent work, finish the requested task, and verify with real commands. We do not
copy AVO's repeated tool descriptions, generic warnings, or duplicated planning text.

## Automatic summarization

Both agents install NOOA's token-budget summarizer. The default trigger is 80% of the
resolved model context window, preserving ten recent events and targeting a 4,000
character summary. This is model-relative, preserves tool-call structure through the
runtime summarizer, and avoids asking the model to manually select event ranges.

## RLM/delegation invariants

- The controller remains responsible for planning, integration, final verification, and final output.
- A worker receives a self-contained objective plus only the context supplied by the
  controller; it does not inherit the controller's event history. Supplied context may
  be typed objects or collections (for example Todo items, paths, matches, or text),
  not merely a flattened string.
- Workers share the resolved LLM and filesystem root, and have their own shell/repo,
  event history, and summarizer. They deliberately have no TodoManager or persistent
  `self.v`: bounded workers report back instead of maintaining a second plan or durable state.
- Delegation is for bounded exploration, diagnosis, review, or an independently
  verifiable implementation. The controller must inspect and integrate the result.
- Independent calls may be run concurrently; dependent calls must be sequential.
- Worker output is a concise structured report, not a raw transcript.
- The same worker primitive will be exposed by `CodingAgent`/`TUIAgent` after the
  benchmark implementation is tested.

## Todo and persistent-variable audit

`TodoManager` is snapshotable. Its status block renders every todo's ID, effective
status, dependencies, variables, and notes; append-only comments remain available
through the API but are not injected every turn. This makes the status block useful
for the long-lived controller but wasteful for a bounded worker. A controller can pass
a `Todo`, a list of todos, or another structured object directly as
`supplied_context`; delegation does not force it through a text-only API.

`self.v` is an `InteractiveAgent` feature backed by `SnapshotVars`. It remains on
`CodingAgent`/`TUIAgent`, where it supports state across turns and session restore. It
is already represented (with depth, collection-length, and string-length bounds) by
the framework's protected dynamic `state` block, so adding another vars context block
would duplicate it. `BenchAgent` and `CodingWorker` are non-interactive `Agent`s and
intentionally do not gain persistent vars. Workers also do not inherit controller vars
implicitly: durable or sensitive controller state crosses the isolation boundary only
when explicitly selected as `supplied_context`.
