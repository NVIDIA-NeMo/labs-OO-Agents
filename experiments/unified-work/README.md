# Unified work: one `self.jobs` vocabulary for shells and subagents

Status: a spike on `spike/atom-unified-work`. Nothing here has been evaluated
on a benchmark; no evaluation results are claimed.

Update: on this branch the spike agent is now the public
`nooa_atom.agent:AtomAgent`. `agent/unified.py` became `agent/agent.py`, the
earlier `AtomAgent` became the private `agent/base.py:AtomAgentBase`, and
there is one turn method, `handle`, with the role selected by `turn_method`
(see `packages/nooa-atom/README.md`). The architecture notes below describe
the spike as it was at b612e2d46, which is what frozen benchmark pins measure.

## Research question

Does an agent work better, and with a smaller prompt, when every piece of
background work (a command, a persistent shell, a one-shot or kept child agent,
any coroutine) goes through **one** small manager with one way to start it, wait
for it, read it and cancel it? The Atom agent offers two today: `self.jobs.run/
spawn/tail/after/every` for commands and timers, and
`self.delegate/spawn/child/children` + `ChildRef` + the `delegates` channel for
child sessions.

## Architecture boundary

```
WorkManager (agent/work_manager.py)          generic: names, delivery, cancel, scratch
  ├─ knows three shapes only: awaitable · async iterable · resource (async CM + send())
  ├─ imports nothing from nooa / sessions / shells (test_manager_imports_no_work_kind)
  └─ helpers work code may call: emit(), emitter(), describe(), job_dir(), current_name()

Shell (agent/work_shell.py)                  work: one-shot run() or a persistent bash resource
Subagent (agent/work_subagent.py)            work: one-shot run() or a kept child resource
  └─ uses SessionPort.delegate(route=...) and SessionRegistry.route_child(): a routed
     child's notes and outcomes go to the adapter, never to `delegates`

UnifiedAtomAgent (agent/unified.py)          integration: an AtomAgent subclass
  ├─ self.jobs = WorkManager(outbox=the `jobs` channel)
  ├─ self.files = read/replace/replace_match/write_file only
  ├─ legacy delegate/spawn/children/child/shell/delegates/producers hidden (delegate/spawn raise)
  ├─ UnifiedCodeAct: cells see this module's names only; Shell() defaults to the workspace
  ├─ `status` context block replaces `queues`
  └─ turn conditions: Waiting names only work that will deliver; unattended Done
     leaves no running command/task/request (idle resources may stay)
```

`AtomAgent` is unchanged; the registry and port changes are additive and
inert without a route (`test_agent_delegation`, `test_child_*` keep passing).

Select it with `agent_spec="nooa_atom.agent.unified:UnifiedAtomAgent"`;
children inherit the spec and run `handle_work`.

## Workflows (as the model writes them)

```python
r = await Shell().run("pytest -q tests/test_x.py")              # short: await in the cell
self.jobs.launch(Shell().run("pytest -q"), name="tests")         # long: JobEvent later
return_result(Waiting(explanation="tests running", on=["tests"]))

sh = self.jobs.launch(Shell(), name="sh")                       # persistent bash
r = await sh.send("cd src && make")                              # cwd/env carry over

done = await Subagent("fix parser").run(todo)                    # one-shot child -> TaskResult
fix = self.jobs.launch(Subagent("fix").run(todo), name="fix")    # ... in the background

rev = self.jobs.launch(Subagent("review"), name="review")      # kept child
q = await rev.send(todo)                                         # TaskResult or Question
r = await rev.send("Use src/parser.py")                          # the answer / a follow-up
req = rev.send(todo); await req.cancel()                         # stops the child's turn
await rev.cancel()                                               # closes the child
```

The examples above are shown to the model once, at the top of the work types
in the `python_cell_tools` block; the class doc points there.

Delivery: each outcome goes to one receiver, a pending `await job` /
`async for x in job`, else a `JobEvent(name, id, kind, value)` in the next
turn's `notification["jobs"]`. A child's `self.message()` notes are items of
the job (the request's while one runs). Requests are named `<name>#N` and are
valid `Waiting` targets. Jobs belong to the session across turns and are
cancelled when it closes; after a restore, jobs that were running are reported
once as `failed` `JobEvent`s whose value starts with `lost:`. A `Job` in
`self.v` is not snapshotted (keep names; `self.jobs.get(name)`).

## How to test

```bash
cd packages/nooa-atom
uv run pytest -q tests/test_work_manager.py tests/test_work_shell.py   # stage 1
uv run pytest -q tests/test_unified_agent.py                            # adapter + agent
uv run pytest -q tests/test_atom_prompt_budget.py -s                    # sizes, both agents
```

`test_unified_agent.py` uses a real `SessionRegistry` with strict fake models
scripted per session name: one-shot and kept subagents, question → answer →
follow-up, background requests and `Waiting`, failure, request cancellation,
a child closed under a request, resource cancel and session close (bash and
child gone), workspace cwd, unattended `Done` with live work, the prompt's
vocabulary in all four modes, the status block, and restore.

## Metrics plan

1. Prompt: fixed system blocks and first-call totals per mode (the budget
   test), and the per-turn task text (repeated every turn).
2. Behaviour, on the same tasks and the requested benchmark model for both
   agents (development workers may use another model; evaluated agents must
   not): task success; turns and model calls per task; cells that fail
   because of API misuse (wrong method, wrong await, waiting on idle work);
   `Waiting`/`Done` postcondition rejections; wall time spent blocked in a
   cell vs. waiting in the background.
3. Orchestration: child count, re-sends after failed checks, orphaned
   (never awaited or cancelled) work at `Done`.

## Current results (measured, not evaluated)

All Atom sessions now execute `handle()`. The saved host `turn_method` selects
only the policy and static role instructions. The default public agent is
`nooa_atom.agent:AtomAgent`; no special `--agent` flag is needed.

Latest first-call measurements (`test_atom_prompt_budget.py`, workspace paths normalized):

| Host selection | System prompt | First-call total |
|---|---:|---:|
| `handle` | 18,669 | 19,398 |
| `handle_batch` | 18,897 | 19,626 |
| `handle_work` | 19,006 | 19,735 |
| `handle_orchestrate` | 19,200 | 19,929 |

Only the applicable role instructions appear, once in the static prefix.
Empty externally supplied instructions add no block. Workspace information is
also static: one line, with a separate repository boundary only when it differs.
Skills helpers and the work API are documented in `python_cell_tools`; current
skill inventory is available on demand. The Todo type is documented once.

The old frozen benchmark spike `b612e2d46` used a separately selectable
`UnifiedAtomAgent` and different turn methods. Its canary/partial results must
not be presented as results for this rewrite. The paired baseline remains
`7f9a09dc1`; new runs must pin and report the verified rewrite commit.

## Current shell and messaging behavior

A `Shell` starts lazily and keeps one bash per object. `run()` and a launched
resource's `send()` share its cwd, variables, functions and serialized commands.
Use `aclose()` or `async with` for explicit cleanup outside Atom; Atom closes
its live shells at session shutdown. A temporary shell's background processes
end when the shell is dropped; retain the object to retain a server. Interrupts
that require restarting bash report lost state explicitly.

A retained subagent distinguishes a request from a note:

```python
request = review.send(task)           # await its own TaskResult or Question
await review.message("also check X")  # admission only; may reach the running turn
```

Notes that miss the current turn can start another turn; its outcome arrives
as a resource item. Later requests are queued and matched to their own turn,
not given the result of an earlier note. Answer a pending question with `send`.
No old `delegates` delivery path is advertised by this agent.

## Open questions

- `self.files`/`self.repo` paths stay relative to the workspace; a `Shell`
  session's `cd` does not move them.
- The legacy `nemo.shell` skill stays loaded (repo and files use it) but is
  deactivated; `/skills` can still turn it on.
- The repo-tool docs are shared; this agent's tools block rewrites their
  three legacy-shell phrases (`_DOC_FIXES`), so a doc change upstream needs
  the list updated (the vocabulary test fails if `self.shell` reappears).
- Routes are in memory. After a restore, a `Subagent` resource that was
  running is reported lost like any job, but its child session (which the
  store keeps) is not re-routed: if that child ran again, its outcomes would
  go to the legacy `delegates` channel, which this agent does not read.
- A request cancelled after its outcome was handed over (before the request
  read it) ends `cancelled` and the outcome is dropped, without an event;
  the child stays open (pinned in `test_unified_agent.py`).
- `CommandResult.stdout`/`stderr` are shown cut to 2,000 characters in
  notifications and `pprint` (display only, `spec(max_string=2000)`); the
  object keeps its 10 kB summaries and `*_path` the full output. A 9,000 +
  5,000 character result pprints in 4,274 characters instead of 14,202.
- Lingering pumps of background processes in a persistent shell are released
  only at session exit (stage-1 review, not blocking).