# Unified work: one `self.jobs` vocabulary for shells and subagents

Status: a spike on `spike/atom-unified-work`. Nothing here has been evaluated
on a benchmark; no evaluation results are claimed.

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

First call, workspace path counted as a placeholder (`test_atom_prompt_budget.py`):

| agent | mode | system prompt | first-call total | turn docstring |
|---|---|---|---|---|
| AtomAgent | handle | 19,737 | 21,807 | 713 |
| AtomAgent | handle_batch | 19,737 | 22,066 | 948 |
| AtomAgent | handle_work | 19,737 | 22,581 | 1,467 |
| AtomAgent | handle_orchestrate | 19,737 | 24,061 | 2,919 |
| UnifiedAtomAgent | handle | 18,269 | 19,865 | 161 |
| UnifiedAtomAgent | handle_batch | 18,269 | 19,885 | 157 |
| UnifiedAtomAgent | handle_work | 18,269 | 19,882 | 158 |
| UnifiedAtomAgent | handle_orchestrate | 18,269 | 19,922 | 170 |

The turn docstring is re-sent with every turn; the stable mode guidance moved
into the class doc (system prompt, 3,030 characters vs. AtomAgent's 3,353), once.

## Open questions

- `self.files`/`self.repo` paths stay relative to the workspace; a `Shell`
  session's `cd` does not move them.
- The legacy `nemo.shell` skill stays loaded (repo and files use it) but is
  deactivated; `/skills` can still turn it on.
- The repo-tool docs are shared; this agent's tools block rewrites their
  three legacy-shell phrases (`_DOC_FIXES`), so a doc change upstream needs
  the list updated (the vocabulary test fails if `self.shell` reappears).
- Lingering pumps of background processes in a persistent shell are released
  only at session exit (stage-1 review, not blocking).
