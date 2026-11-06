# Channel-driven turns

Every `InteractiveAgent` has `agent.turns`, a host-neutral `TurnLoop`. It has no
Session, storage, receipt IDs or protocol dependency. It starts only when asked:

```python
agent.turns.start(prepare=prepare, commit=commit)
```

- `QueueManager.wait_ready()` waits without dequeuing. `ready()` includes buffered
  queue items and unclaimed event-channel wakes. Registry changes wake the wait;
  no channels ends the loop, even with a pending event wake. A wake from a removed
  event channel still dispatches once if other channels remain (the event is already
  in context); it cannot produce an empty batch after the final channel is removed.
  Event-only batches are `{}`, absent readiness is `None`.
- `prepare()` is async and runs inside the cancellable turn task with input still
  queued. After it returns, the loop rechecks pause/stop and current readiness.
- `claim_batch(commit)` selects buffered input in channel registration order and
  per-channel FIFO order. The synchronous `commit(batch)` must record mandatory
  work, propagate failures, and not call observers or mutate channels. It receives
  a separate container copy, not ownership of the selected queue containers.
- Only after commit succeeds are all selected items dequeued. Consumption observers
  then run, followed by `TurnBegan` and method lookup/invocation. No await separates
  selection, recording, ownership transfer and invocation setup. Lookup/call errors
  and non-awaitable returns settle as errors without killing the dispatch loop.

`pause()` gates new dispatch but lets a committed turn finish. `resume()` opens the
same loop and explicitly retries queued input. `stop_starting()` is a compatibility
alias for pause. `cancel(by=...)` cancels/settles a turn; cancellation during
preparation also pauses and leaves input queued. `stop()` pauses, cancels/settles,
ends and unsubscribes. A stopped or naturally ended loop can restart cleanly.
`running` includes preparation/settlement; `paused` and `dispatch_error` expose
blocked dispatch. Preparation/commit failure publishes `TurnSettled(error,
ran=False, committed=False)` and pauses, never automatically retries untouched input.

`TurnSettled.committed` distinguishes a started batch from untouched input; `ran`
distinguishes method invocation from setup/preparation. Settlement signaling runs
in a nested finally. Synchronous event observers raising Exception or CancelledError
are logged individually and cannot skip later subscribers. Mandatory recording must
use explicit hooks, not rely on this best-effort event dispatch. `ChannelItemConsumed`
has `batch=True` for initial claimed input, `False` for get/drain/race consumers.

All channels registered with a manager share a reentrant lock for selection and
queue mutation. Cross-thread puts serialize against commit; they belong to the next
batch when blocked by an ongoing commit. All reentrant channel mutations (including
queue/event puts), registry changes and nested claims during commit are rejected
before they can emit observers or alter input. Consumption observers see the complete ownership transfer and may put
new input without extending the current batch. Registry changes, dispatch control
and Session admission are event-loop-owned, not cross-thread APIs. Do not run `race`
and `wait_ready` dispatchers concurrently on one manager. Existing `race()` callers
retain their original winner/restore semantics.

Loop context is fresh by default; an explicit context can carry host hooks. Each
turn task copies it, so turn-local changes do not leak into the next turn.
`InteractiveAgent.aclose()` joins turns and producer jobs before other cleanup,
including across stop/start and repeated close epochs. Self-stop/self-close from
an executing turn is rejected with RuntimeError to prevent circular task joins;
close from the external owner after settlement. Async component cleanup still must
cooperate with cancellation; a permanently uncooperative coroutine can block close.

`EventManager.record_batch()` is a separate, observer-free mandatory ledger path.
SQLite `append_batch()` allocates tags and inserts events/active tags in one locked
transaction, advancing counters only after success; it does not use single-row
reconnect/retry. In-memory supports it; other backends without atomic append support
fail explicitly, with no partial-write fallback. Only fresh durable non-summary
events are accepted: tagged/reused objects, aliased batch entries, duplicate event
IDs and IDs already in storage are rejected before mutation. SQLite serializes
copies and publishes caller tags only after commit, so fresh inputs can be retried
unchanged after rollback. Ordinary `add()` retains its existing pre-store observer order.
