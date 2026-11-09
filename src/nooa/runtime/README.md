# Channel-driven turns

Every `InteractiveAgent` has `agent.turns`, a `TurnLoop` (`nooa.runtime.turn_loop`).
The loop is host-neutral: it does not depend on a Session, storage, receipt IDs
or a protocol. It runs only after a host starts it:

```python
agent.turns.start(prepare=prepare, commit=commit)
```

## Public API

- `start(*, turn_method="handle", context=None, prepare=None, commit=None, after_settle=None)`:
  start the loop. Does nothing if it is already started.
- `pause()`: stop new dispatch. A committed turn keeps running.
- `resume()`: reopen the same loop and retry queued input. Clears `dispatch_error`.
  Does nothing while the loop is stopping.
- `stop_starting()`: compatibility alias for `pause()`. Use `stop()` to end the loop.
- `cancel(*, by="user")`: cancel the current turn and wait until it settles.
  Returns `False` if no turn is running.
- `stop(*, by="host")`: pause, cancel and settle the turn, end the loop and unsubscribe.
- `started`: the loop task is alive.
- `running`: a turn is in progress, including preparation and settlement.
- `paused`: dispatch is gated.
- `dispatch_error`: the exception that blocked dispatch, or `None`.
- `in_turn`: the caller is the turn task or the post-settlement task.

Events, all runtime events (never recorded, never shown to the model):

- `TurnBegan(notification)`: published right before the turn method is called.
- `TurnSettled`: published once per turn, however it ended (see Settlement).
- `TurnLoopEnded(error, message)`: the loop ended because `wait_ready()` raised.
  It is not published when `stop()` ends the loop.

`TurnCancelled(by, interrupted)` is an ordinary event in the agent's events.
It tells the model that its turn was stopped.

## One dispatch cycle

1. **Wait.** `QueueManager.wait_ready()` waits and does not dequeue.
2. **Prepare.** The async `prepare()` hook runs inside the cancellable turn task.
   All input is still queued.
3. **Recheck.** After `prepare()` returns, the loop checks for a pending cancel,
   pause or stop, and checks readiness again.
4. **Commit.** `claim_batch(commit)` selects the input and calls `commit(batch)`.
5. **Dequeue.** Only after `commit` succeeds are all selected items dequeued.
6. **Observe.** `ChannelItemConsumed` observers run, then `TurnBegan` is published.
7. **Invoke.** The loop looks up and calls the turn method (`handle` by default).

No `await` separates selection, recording, ownership transfer and invocation setup.

## Readiness

- `ready()` counts buffered queue items and unclaimed event-channel wakes.
- Registry changes wake `wait_ready()`.
- With no channels, `wait_ready()` raises and the loop ends (`TurnLoopEnded`).
  This holds even if an event wake is pending.
- A wake from a removed event channel still dispatches once if other channels
  remain. The event is already in context.
- A stale wake cannot produce an empty batch after the last channel is removed.
- An event-only batch is `{}`. No readiness is `None`, and no turn settles.

## Commit hook

- `claim_batch` selects input in channel registration order, FIFO per channel.
- `commit(batch)` must be synchronous. An awaitable result raises `TypeError`.
- `commit` must record mandatory work and let failures propagate.
- `commit` must not call observers or mutate channels.
- `commit` receives a copy of the containers. It does not own the queue containers.
- If `commit` raises, the input stays queued.

## Turn method

- Lookup and call errors settle the turn as `error`. The dispatch loop keeps running.
- A non-awaitable return value settles as `error`.
- A return value that is not a turn result settles as `error` with `error=None`.
- A result settles by its class variable `_turn_kind`: `done`, `need_input` or `waiting`.

## Pause, cancel and stop

- `pause()` gates new dispatch. It does not stop a committed turn.
- `resume()` is the only way to retry input after a pause or dispatch failure.
- A failure or cancel before commit pauses the loop. Input stays queued.
- Untouched input is never retried automatically.
- `cancel()` during preparation also pauses. A `prepare()` that swallows
  `CancelledError` still cannot commit.
- `cancel()` after the turn method ran records `TurnCancelled`, then settles as `cancelled`.
- `cancel()` from inside the turn does not wait for its own settlement.
- `stop()` and `InteractiveAgent.aclose()` from the executing turn raise `RuntimeError`.
  This prevents a task from joining itself. Close from the external owner after settlement.
- A stopped or ended loop can restart. `start()` does not add duplicate subscriptions.

## Settlement

`TurnSettled.kind` is `done`, `need_input`, `waiting`, `cancelled` or `error`.

- `result`: the turn result, for the result kinds.
- `cancelled_by`: who cancelled (`by`; `"host"` if not set).
- `interrupted`: tag of the interrupted cell's `PythonOutput`, or `None`.
- `error`, `message`: the exception and its description, for `error`.
- `ran`: the turn method was called. `False` means setup or preparation ended the turn.
- `committed`: the batch was committed. `False` means the input never left its
  channels and no durable turn started.

Rules:

- A preparation or commit failure publishes `TurnSettled(kind="error", ran=False,
  committed=False)`, sets `dispatch_error` and pauses.
- An error after commit settles as `error` with `committed=True` and does not pause.
- A `CancelledError` raised inside the turn without `cancel()` settles as `error`
  ("turn was cancelled from inside").
- Settlement is published in a nested `finally`.
- `after_settle(settled)` runs after `TurnSettled` is published, in its own task.
  It cannot replace the settlement. Its exceptions are logged.

## Observers

- `EventManager` calls each synchronous observer separately.
- An observer that raises `Exception` or `CancelledError` is logged. Later observers still run.
- Event dispatch is best effort. Mandatory recording must use explicit hooks
  (`commit`, `record_batch`), not observers.
- `ChannelItemConsumed.batch` is `True` for items claimed by a turn, and
  `False` for `get()`, drain and `race()` consumers.
- Consumption observers run after the full ownership transfer. They may put new
  input; it goes to the next batch.

## Concurrency

- All channels registered with one `QueueManager` share one reentrant lock for
  selection and queue mutation.
- Cross-thread puts serialize against commit. A put blocked by a commit goes to the next batch.
- During commit, these raise `RuntimeError` before they emit observers or change input:
  reentrant channel mutations (including queue and event puts), registry changes,
  and nested claims.
- Registry changes, dispatch control and Session admission belong to the event loop.
  They are not cross-thread APIs.
- Do not run `race()` and `wait_ready()` dispatchers at the same time on one manager.
- Existing `race()` callers keep their winner and restore behaviour.

## Context and close

- The loop runs in a fresh `contextvars.Context` by default.
- An explicit `context` can carry host hooks.
- Each turn task copies the loop context, so changes in one turn do not reach the next.
- `InteractiveAgent.aclose()` stops turns and cancels producer jobs before other cleanup.
  This also holds across stop/start and repeated close epochs.
- Async component cleanup must cooperate with cancellation. A coroutine that
  never yields to cancellation can block close.

## Mandatory batch recording

`EventManager.record_batch(events)` is a separate ledger path. It calls no observers.

- The backend must implement `append_batch`. Otherwise `record_batch` raises `TypeError`.
  There is no fallback to partial single-event writes.
- The in-memory and SQLite backends implement it.
- Only fresh, durable, non-summary events are accepted.
- Rejected before any change: tagged or reused objects, the same object twice in a
  batch, duplicate event IDs, IDs already in storage, and events whose tag cannot
  be set by ordinary field assignment (frozen, validated or custom setters).
- `append_batch` validates the batch once. Backends can be called directly.
- SQLite allocates tags and inserts events and active tags in one locked transaction.
- SQLite advances its counters only after success. It does not use the single-row
  reconnect/retry path.
- SQLite serializes tagged copies and sets the caller's tags only after commit,
  so the same inputs can be retried after a rollback.
- Ordinary `add()` keeps its existing order: observers run before the event is stored.
