---
name: nooa-channels
description: Reactive input for NVIDIA OO agents — Channel/QueueManager for queued and event-mode input, race() turn dispatch, spawn() background jobs with JobHandle, and the bundled producers (monitor a shell command, cron ticks, one-shot timers, file tails). Use when an agent must react to external input mid-run — user messages, CI output, timers, job completions — or when building an interactive/long-running agent loop.
compatibility: nooa package
---

# Channels: Reactive Agent Input

`Channel` + `QueueManager` (`nooa.runtime.channels`) are the producer side of agent input: outside events flow in through named channels; the agent's dispatch loop races them and reacts. Peer of `EventManager`/`ContextManager`, but NOT auto-created — you construct it.

```python
from nooa.runtime.channels import QueueManager
from nooa.runtime.producers import monitor, cron

class Watcher(Agent, llm=llm):
    def __init__(self, **kw):
        super().__init__(**kw)
        qm = QueueManager(agent=self, event_manager=self.event_manager)
        self.qm = qm
        self.user_messages_in = qm.queue("user_messages")   # producer side (put)
        self.user_messages = self.user_messages_in.reader   # LLM-facing side (get/snapshot only)
        self.notifications = qm.event("notifications")      # fire-and-forget → prompt

    async def run(self) -> None:                            # pure-Python dispatcher
        self.qm.spawn(monitor("make test"), channel="user_messages", buffer=100)
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
from nooa.runtime.producers import monitor, after, cron, tail, run_job

handle = qm.spawn(monitor("pytest -q"), channel="ci", buffer=100)  # stream stdout lines
qm.spawn(after(300), channel="wakeup")                              # one-shot timer → None
qm.spawn(cron(60), channel="ticks")                                 # yields 1, 2, 3, ... per minute
qm.spawn(tail("app.log"), channel="logs")                           # new lines as they appear
qm.spawn(run_job(some_coro(), job_id="batch-7"), channel="jobs")    # {"job_id": ..., "result": ...}
```

- A **coroutine** job puts its single result on completion; an **async generator** puts each yielded value.
- Lifecycle events go to the agent's event log: `StreamEnd(channel_name=...)` when a job finishes, `JobError(channel_name, error_type, error_message)` on failure — `JobError` is also put on the data channel so `race()` wakes the agent to react.
- `JobHandle`: `.state` (`running/done/cancelled/failed`), `await handle.cancel()` (awaits generator `finally` cleanup), `.values` (buffered outputs — `buffer=True` unbounded, `buffer=N` ring of last N, default off).
- `monitor()` runs the command in its own process group and kills the whole group on cancel — no orphaned children; stderr is merged into stdout.

## Pitfalls

- `QueueManager` is hidden from the LLM by default; expose deliberately (`spec(self, "queue_manager", hidden=False)`) or, better, expose only the `reader` attributes and keep `put`/`spawn` on the Python side.
- Registration order is priority order for `race()` — register the highest-priority channel (usually user messages) first.
- Don't `await channel.get()` (producer object) from LLM code — give the LLM the `.reader`, whose `get(timeout=...)` can't deadlock a cell.
- The dispatcher belongs in a pure-Python orchestrator method (`run()` above) — see "Orchestrators are pure Python" in `nooa-agent-authoring`.
- `spawn()` requires the channel to already exist (`ValueError` otherwise).

## Related skills

- `nooa-agent-authoring` — the orchestrator pattern the dispatch loop lives in.
- `nooa-context-and-state` — `QueueOutput`/`StreamEnd`/`JobError` are events; query them like any others.
- `nooa-middleware-hooks` — `on()` observers if you only need to react to recorded events, not consume input.
