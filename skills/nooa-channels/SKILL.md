---
name: nooa-channels
description: Reactive input for NOOA agents — Channel/QueueManager for queued and event-mode input, race() turn dispatch, spawn() background jobs with JobHandle, and Atom's model-facing self.jobs API built on it (background commands, file tails, timers, ticks). Use when an agent must react to external input mid-run — user messages, CI output, timers, job completions — or when building an interactive/long-running agent loop.
compatibility: nooa package
---

# Channels: Reactive Agent Input

`Channel` + `QueueManager` (`nooa.runtime.channels`) are the producer side of agent input: outside events flow in through named channels; the agent's dispatch loop races them and reacts. Peer of `EventManager`/`ContextManager`, but NOT auto-created — you construct it.

```python
from nooa.runtime.channels import QueueManager

class Watcher(Agent, llm=llm):
    def __init__(self, **kw):
        super().__init__(**kw)
        qm = QueueManager(agent=self, event_manager=self.event_manager)
        self.qm = qm
        self.user_messages_in = qm.queue("user_messages")   # producer side (put)
        self.user_messages = self.user_messages_in.reader   # LLM-facing side (get/snapshot only)
        self.notifications = qm.event("notifications")      # fire-and-forget → prompt

    async def run(self) -> None:                            # pure-Python dispatcher
        self.qm.spawn(watch_ci(), channel="user_messages", buffer=100)  # an async generator
        while True:
            items = await self.qm.race()                    # block until any channel produces
            if items:                                       # [(channel_name, item)] — queue winner
                await self.handle(items[0])
            # [] means an event-mode put woke us — it's already rendered
            # into the next turn's prompt; just proceed to a turn.

    async def handle(self, item: tuple[str, object]) -> str:
        """React to the new input."""
        ...
```

## Two channel modes

| Mode | Factory | `put()` does | Consumption |
|---|---|---|---|
| **queue** | `qm.queue(name)` | append to a deque, wake one waiter | `race()` in the dispatcher, or agent code drains mid-turn with `await self.<chan>.get()` |
| **event** | `qm.event(name)` | add a `QueueOutput` event to `event_manager` — value renders inline in the next turn's prompt | none — no buffer, no `get()` |

- Use **queue** for inputs the agent must consume one at a time (user messages, job results); **event** for notifications the agent should merely notice ("build finished", status pings).
- `QueueOutput` renders `source`/`value_type`/`value_preview` to the model; the full `value` is hidden (`repr=False`) but reachable from code via `event_manager.get(tag).value`.
- Expose the read side to the LLM as `channel.reader` — a `get(timeout=5.0)`/`status()` wrapper without `put` (readers time out with `QueueReadTimeoutError` instead of blocking a cell forever).
- `channel.snapshot()` peeks without consuming; `qm.status()` renders a composite status block for all channels (pin it as a dynamic context block so the model sees pending input).

## `race()` — the dispatch primitive

- Returns a **length-1 list** `[(name, item)]` for a queue-mode winner, or **`[]` when an event-mode put woke it** (events are already in the prompt; the empty list tells you no queue item was consumed). The list shape is the contract.
- If multiple puts land on adjacent ticks, the first channel in **registration order** wins; the other drained items are restored to the head of their channels — nothing is dropped. `on_get` hooks fire only for the winner.
- Raises `ValueError` when no channels are registered (dispatchers treat this as "exit cleanly"); propagates `CancelledError` after cleaning up its drain tasks.
- Thread-safe wakeups: producers may `put()` from another thread (e.g. a UI thread); the manager wakes `race()` via the owning event loop.

## `spawn()` — background jobs feeding channels

```python
handle = qm.spawn(watch_ci(), channel="ci", buffer=100)  # async generator: one put per yield
qm.spawn(fetch_report(url), channel="reports")             # coroutine: one put of its result
```

- A **coroutine** job puts its single result on completion; an **async generator** puts each yielded value.
- Lifecycle events: `StreamEnd(channel_name=...)` when a job finishes is a runtime event (subscribe with `event_manager.on("StreamEnd", ...)`; never recorded or shown to the model); `JobError(channel_name, error_type, error_message)` on failure goes to the event log and is also put on the data channel so `race()` wakes the agent to react.
- `JobHandle`: `.state` (`running/done/cancelled/failed`), `await handle.cancel()` (awaits generator `finally` cleanup), `.values` (buffered outputs — `buffer=True` unbounded, `buffer=N` ring of last N, default off).

## `self.jobs` — a model-facing view built on `spawn()` (Atom)

The Atom agent (`nooa_atom.agent.agent.AtomAgent`) registers a `jobs` queue channel and exposes `self.jobs` (`nooa_atom.agent.jobs.JobsApi`), built on `spawn()`; core `InteractiveAgent` has no `self.jobs`. Every item it delivers is a `JobUpdate` with `.label` and `.path`, so a turn can end with `Waiting(on=["<label>"])` and read `notification["jobs"]`. Jobs and child sessions share one namespace of live names per session: a label (or child name) is a Python identifier (`tests`, `ci_check`; not `ci-check`) and a name a running job or open child already has is rejected. Paths are `/root/<name>` (`/root/review_pr_462/verifier` deeper down); the per-turn `queues` block counts what arrived and was not read yet under "Unread:", one line per channel with the senders' paths and never any content (`delegates: 2 (from /root/ask, /root/fix)`, `jobs: 3 (from /root/build)`; an item names its sender with `status_source()`), then a "Read now:" line, your own running jobs and children under "Running:" and retained children with nothing to do under "Idle:", and `Waiting(on=[...])` takes a name, a child's path or id. Exactly one update with `done=True` ends each job; its result is `items[0]` (a `ShellResult` for `run`, a coroutine's return value, `after`'s payload).


```python
await self.jobs.run("uv run pytest -q", label="tests")       # own bash process; one done update (items[0] is the ShellResult)
await self.jobs.run("make build", label="build", items=200, seconds=60)  # + JobUpdate progress batches
self.jobs.spawn(fetch_report(url), label="report")            # any coroutine or async iterator
self.jobs.tail("app.log", label="log", seconds=30)            # lines appended from now on
self.jobs.after(300, label="wakeup")                          # one done update after 5 minutes
self.jobs.every(60, label="tick")                             # ticks 1, 2, 3, ... until cancelled
self.jobs.throttle("log", items=100)                          # re-batch a running stream
await self.jobs.cancel("build")                               # kills the process group
await self.jobs.run("make serve", label="srv", replace=True)  # restart a running job with that label
```

- A `run`'s `ShellResult` keeps only the last 20,000 chars of stdout and, separately, of stderr (after a `[... N chars omitted]` line); there is no log file. For more, have the command redirect to a file itself (`cmd > out.log 2>&1`) and read it.
- Streams end with a done update carrying their last pending values (or `[]`); on an error `items` is `[]` and `error` is set; a cancelled job delivers nothing more. Jobs do not survive a restart: a resumed session gets one `JobsLost` event listing those that were running.
- **Who uses what.** `self.jobs` is for the model's own background work. Skills that own a named channel or long-lived infrastructure (e.g. agent_mesh's `mesh` inbox pump, CI monitors) keep using `agent.queue_manager.queue(name, replace=True)` + `spawn(...)`, and call `agent.queue_manager.set_cancel_hint(name, "Stop a monitor: await self.<skill>.stop('<label>')")` if the model may stop them. Without a hint, `status()` shows the job but no cancel command.

## Pitfalls

- `QueueManager` is hidden from the LLM by default; expose deliberately (`spec(self, "queue_manager", hidden=False)`) or, better, expose only the `reader` attributes and keep `put`/`spawn` on the Python side.
- Registration order is priority order for `race()` — register the highest-priority channel (usually user messages) first.
- Don't `await channel.get()` (producer object) from LLM code — give the LLM the `.reader`, whose `get(timeout=...)` can't deadlock a cell.
- The dispatcher belongs in a pure-Python orchestrator method (`run()` above) — see "Orchestrators are pure Python" in `nooa-agent-authoring`.
- `spawn()` requires the channel to already exist (`ValueError` otherwise).

## Related skills

- `nooa-agent-authoring` — the orchestrator pattern the dispatch loop lives in.
- `nooa-context-and-state` — `QueueOutput`/`JobError` are recorded events; query them like any others (`StreamEnd` is runtime-only).
- `nooa-middleware-hooks` — `on()` observers if you only need to react to recorded events, not consume input.
