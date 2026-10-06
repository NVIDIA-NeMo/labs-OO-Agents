# nooa-bench

Coding benchmark agents and Harbor runner for
[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents), supporting SWE-bench
and Terminal-Bench tasks.

```bash
uv add nooa-bench
nemo-harbor --help
```

See the [main repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents) for
documentation.

## Running the coding agent headless

`nooa-bench run` runs one task with nooa-coder's coding agent, without
Harbor. The agent works unattended in the workspace directory, and its
session is stored in `.nooa/sessions` there (or in `NOOA_SESSIONS_DIR`), so
it can be opened afterwards like any other session:

```bash
NEMO_OO_LLM_CONFIG=llm_config.yaml nooa-bench run --workspace /tmp/task --model glm-5.3 \
    "Create a file hello.py that prints hello, then run it."
```

Standard output carries only the result, one JSON object with `status`
(`done` or `stopped`), `stopped` (why the run stopped without finishing),
`message`, `explanation`, `result` (the `TaskResult`), `session_id`,
`turns` and `usage`. Logs go to standard error. The exit code is 0 when the
agent finished and 1 otherwise. A turn that ends waiting on a job is
followed by the turn the job's delivery starts; `--max-turns` (default 10)
stops a run whose last allowed turn is still waiting, and `--timeout`
(seconds, no default) stops it after that time.

To run a Harbor benchmark with the same agent, set `agent_type: coder` (and a
`git_ref` that has it) in the agent's `kwargs` in
`examples/benchmarks/harbor_minimal.yaml`, then run from the repository root:

```bash
PYTHONPATH=examples/benchmarks harbor run --config examples/benchmarks/harbor_minimal.yaml
```

Harbor then runs `nemo-harbor --agent-type coder` in each task container.
The agent works in the task's working directory (`--working-dir`, else
`/testbed` or `/app`). The example adapter sets
`NOOA_SESSIONS_DIR=/logs/agent/sessions`, so the session is kept with the
task's logs; `NOOA_SESSIONS_DIR` pointed at that directory lists it. Token
counts come from the session's usage, children included, and
`trajectory.json` holds the root agent's events.

## Agent types

Three agent types are available through `nemo-harbor --agent-type`: `coder`
(above) and two variants on the older agent loop:

- `bench` — `BenchAgent` in `nooa_bench.bench_agent`: compact CodeAct baseline
  with automatic summarization and optional delegation.
- `rlm` — `RLMBenchAgent` in `nooa_bench.rlm_bench_agent`: the same capabilities
  with instructions emphasizing delegation for bounded work.

Both use `CodeActV2` with the single `python_cell` tool and return a structured `TaskResult`.
Its `how_to_verify` field describes concrete checks and expected results; commands
are optional. The `evidence` field records results the agent actually observed.
Both delegate through an awaited call returning a `TaskResult`; neither exposes
the interactive coding agent's background `spawn()` / job-handle API.
The strategy allows ten retries, uses a 1,800-second cell timeout, and has no
fixed iteration cap; configure the enclosing benchmark's time/token budget.
Awaited delegation runs inside that same parent cell deadline. A timeout cancels
the worker and merges no partial Todo state; the parent receives a cell timeout
error and may try again. Cell timeouts do not consume the strategy's retry counter,
so the enclosing harness budget is the overall limit on repeated delegations.
Workers use the same agent type, model client and working directory, with their
own execution context and shell. Delegation defaults to a maximum depth of four.
Passing a Todo gives the worker an independent task copy; successful worker
updates are merged after cleanup. Conflicts or worker-only dependencies raise
`DelegationMergeError`, retaining the completed `result` and full `worker_state`
for explicit reconciliation without rerunning the worker. Failed execution or
cleanup does not merge partial state. Task-local state stays on Todos, and automatic
summarization handles context maintenance. Method-writing tools are available in
both variants.

Context usage is the last provider-reported input count divided by the model
window minus UnifiedLLM's effective reply cap (including reasoning-level and
per-call settings). For example, 32,000 input tokens with a 128,000-token window
and 64,000-token reply cap is 50%. Automatic summarization triggers at 80% of
that usable input window. Explicit summarization thresholds stay fixed; an
unknown reply cap uses a labelled planning reserve, not a claimed model limit.

The runner writes `result.json`, `trajectory.json` and aggregate `behavior.json`
under `/logs/agent`, and the verification instructions to `/app/answer.txt`. Behavior
metrics count both Python tool names and exclude framework prefill. Set
`NOOA_INTERFACE_CHANGE_ID` to label a comparison; the default is `baseline`.
Set `NOOA_TASK_ID` to identify the task when logs share the `/logs/agent` path.
Exports include archived events after summarization, with the actual event IDs.
The original task inputs also remain in bounded, non-summarizable prompt context.
Metrics cover the controller's history; delegated workers keep separate histories
and their cells are not included. Recovery and retry metrics are omitted until
framework events carry explicit attempt linkage. Schema version 2 removes
unsupported error-code guesses from stdout. Code metrics count syntactic call
sites, not actual runtime loop iterations; fan-out recognizes direct and starred
`asyncio.gather` arguments, including comprehensions and same-cell list aliases.
Simple same-cell aliases of Todo, shell and repo objects are recognized; this is
not general cross-cell dataflow analysis. Reports with another schema, content
policy or unknown metrics are rejected; regenerate them from `trajectory.json`.
Supplied delegation context is an ordinary worker-method argument, displayed by
NOOA's standard parameter formatting. There is no delegation-specific renderer
or redaction policy; pass only the data the worker needs.
Failure to generate the behavior report does not fail an otherwise completed
task. Agents close their shells; the runner closes the shared model client.

These are the current agent prompts and strategy. Reproducing a historical tech
report run requires its original code revision and configuration.

Apache-2.0 licensed.
