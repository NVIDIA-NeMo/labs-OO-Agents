# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``Channel`` + ``QueueManager``.

No LLM / runtime required — just ``asyncio`` behaviour.

Queue state is surfaced to the LLM through a dynamic context block
(``queues``, composed via ``QueueManager.status()``), not through
events. Event-mode channels are tested separately where they need an
``event_manager`` — most tests here are queue-mode.
"""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import Channel, QueueManager, QueueReadTimeoutError

# ---------------------------------------------------------------------------
# Basic producer/consumer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_get_fifo():
    q: Channel[str] = Channel("user_messages", "queue")
    q.put("a")
    q.put("b")
    q.put("c")
    assert await q.get() == "a"
    assert await q.get() == "b"
    assert await q.get() == "c"
    assert q.qsize() == 0


@pytest.mark.asyncio
async def test_get_blocks_then_wakes():
    q: Channel[str] = Channel("q", "queue")
    # No item yet — get() must block.
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)  # let the task run & register the waiter
    assert not getter.done()
    assert q.has_waiters()
    # Producer pushes — getter should complete with that item.
    q.put("hello")
    assert await asyncio.wait_for(getter, timeout=0.5) == "hello"
    assert not q.has_waiters()


@pytest.mark.asyncio
async def test_put_delivers_directly_when_waiter_exists():
    """When a waiter is pending, put() hands the item straight to the waiter.

    The item should NOT land on the backing deque — qsize stays 0.
    """
    q: Channel[int] = Channel("q", "queue")
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    q.put(42)
    assert await asyncio.wait_for(getter, timeout=0.5) == 42
    assert q.qsize() == 0


@pytest.mark.asyncio
async def test_cancelled_get_leaves_no_phantom_waiter():
    q: Channel[str] = Channel("q", "queue")
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getter
    # After cancellation the waiter list should be cleaned up.
    assert not q.has_waiters()
    # And a subsequent put() should land in the deque, not attempt to
    # deliver to a dead waiter.
    q.put("x")
    assert q.qsize() == 1


@pytest.mark.asyncio
async def test_cancelled_get_late_result_wakes_next_waiter_without_fifo_inversion():
    q: Channel[str] = Channel("q", "queue")
    first_getter = asyncio.create_task(q.get())
    second_getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)

    q.put("a")
    first_getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_getter

    assert q.snapshot() == []
    assert not q.has_waiters()
    assert await asyncio.wait_for(second_getter, timeout=0.5) == "a"

    q.put("b")
    assert await q.get() == "b"


@pytest.mark.asyncio
async def test_cancelled_get_restored_item_stays_before_later_put():
    q: Channel[str] = Channel("q", "queue")
    first_getter = asyncio.create_task(q.get())
    second_getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)

    q.put("a")
    first_getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_getter

    q.put("b")
    assert await asyncio.wait_for(second_getter, timeout=0.5) == "a"
    assert await q.get() == "b"


@pytest.mark.asyncio
async def test_cancelled_get_late_result_stays_before_put_before_cancel_unwinds():
    q: Channel[str] = Channel("q", "queue")
    first_getter = asyncio.create_task(q.get())
    second_getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)

    q.put("a")
    first_getter.cancel()
    q.put("b")
    with pytest.raises(asyncio.CancelledError):
        await first_getter

    assert await asyncio.wait_for(second_getter, timeout=0.5) == "a"
    assert await q.get() == "b"


# ---------------------------------------------------------------------------
# Non-consuming accessors
# ---------------------------------------------------------------------------


def test_pop_last_and_snapshot():
    q: Channel[str] = Channel("q", "queue")
    assert q.pop_last() is None
    q.put("a")
    q.put("b")
    q.put("c")
    assert q.snapshot() == ["a", "b", "c"]
    # pop_last removes the tail.
    assert q.pop_last() == "c"
    assert q.snapshot() == ["a", "b"]
    # Snapshot is a copy — mutating it doesn't affect the queue.
    snap = q.snapshot()
    snap.append("z")
    assert q.snapshot() == ["a", "b"]


def test_peek_removed():
    """peek() invited multi-turn branching on queue state; removed."""
    q: Channel[str] = Channel("q", "queue")
    assert not hasattr(q, "peek")


def test_clear_empties_the_queue():
    q: Channel[str] = Channel("q", "queue")
    q.put("a")
    q.put("b")
    q.clear()
    assert q.qsize() == 0
    assert q.snapshot() == []


def test_reader_qsize_tracks_buffered_items():
    q: Channel[str] = Channel("user_messages", "queue")
    reader = q.reader
    assert reader.qsize() == 0
    q.put("a")
    q.put("b")
    assert reader.qsize() == 2


@pytest.mark.asyncio
async def test_reader_get_times_out_with_actionable_error():
    q: Channel[str] = Channel("user_messages", "queue")
    reader = q.reader

    with pytest.raises(QueueReadTimeoutError) as exc_info:
        await reader.get(timeout=0.01)

    message = str(exc_info.value)
    assert "Timed out after 0.01s waiting for queue 'user_messages'" in message
    assert "WAIT/NEED_INPUT" in message
    assert not q.has_waiters()


def test_reader_get_default_timeout_is_five_seconds():
    q: Channel[str] = Channel("user_messages", "queue")
    assert q.reader.get.__defaults__ == (5.0,)


@pytest.mark.asyncio
async def test_reader_get_timeout_none_preserves_indefinite_wait():
    q: Channel[str] = Channel("user_messages", "queue")
    reader = q.reader

    getter = asyncio.create_task(reader.get(timeout=None))
    await asyncio.sleep(0)
    assert not getter.done()
    q.put("hello")
    assert await asyncio.wait_for(getter, timeout=0.5) == "hello"


# ---------------------------------------------------------------------------
# on_get hook — fires once per returned item, on every path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_get_fires_on_slow_path_backlog_drain():
    """When the item was already on the deque, ``get()`` fires ``on_get``
    before returning. The TUI relies on this for the mid-turn dequeue
    case where the agent's ``await self.user_messages.get()`` drains
    a message that was typed while the agent was busy."""
    seen: list[str] = []
    q: Channel[str] = Channel("user_messages", "queue", on_get=seen.append)
    q.put("hi")
    got = await q.get()
    assert got == "hi"
    assert seen == ["hi"]


@pytest.mark.asyncio
async def test_on_get_fires_on_fast_path_waiter_handoff():
    """When a waiter was already blocked on ``get()`` and ``put()`` hands
    the item straight through, the hook must still fire exactly once."""
    seen: list[str] = []
    q: Channel[str] = Channel("user_messages", "queue", on_get=seen.append)
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)  # register the waiter
    q.put("hello")
    assert await asyncio.wait_for(getter, timeout=0.5) == "hello"
    assert seen == ["hello"]


@pytest.mark.asyncio
async def test_on_get_fires_once_per_item_across_multiple_gets():
    """Two puts, two gets → hook fires twice, in order."""
    seen: list[str] = []
    q: Channel[str] = Channel("user_messages", "queue", on_get=seen.append)
    q.put("a")
    q.put("b")
    assert await q.get() == "a"
    assert await q.get() == "b"
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_on_get_does_not_fire_on_cancelled_get():
    """If a waiter is cancelled, no item transitioned queued → accepted
    — the hook must NOT fire. A producer that already set the waiter's
    result pushes the item back onto the deque; the hook will fire when
    some other consumer actually takes it."""
    seen: list[str] = []
    q: Channel[str] = Channel("q", "queue", on_get=seen.append)

    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0)
    getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getter

    assert seen == []
    # Now a real consumer — the hook should fire for them.
    q.put("later")
    assert await q.get() == "later"
    assert seen == ["later"]


@pytest.mark.asyncio
async def test_on_get_hook_exception_is_swallowed_and_item_still_returned():
    """A buggy hook must not eat the item. Dropping it on the floor is
    strictly worse than a missing UI echo."""

    def boom(_item: str) -> None:
        raise RuntimeError("bad hook")

    q: Channel[str] = Channel("q", "queue", on_get=boom)
    q.put("payload")
    # Must not raise.
    got = await q.get()
    assert got == "payload"


@pytest.mark.asyncio
async def test_set_on_get_late_binds():
    """Session installs its ``_on_user_message`` hook AFTER the agent
    has already created its ``_user_messages_in`` queue, so late
    binding must work."""
    seen: list[str] = []
    q: Channel[str] = Channel("user_messages", "queue")  # no hook yet
    q.set_on_get(seen.append)
    q.put("x")
    await q.get()
    assert seen == ["x"]


# ---------------------------------------------------------------------------
# status() — pending-count + preview rendering on the queue itself
# ---------------------------------------------------------------------------


def test_status_empty_queue_returns_empty_string():
    q: Channel[str] = Channel("user_messages", "queue")
    assert q.status() == ""


def test_status_includes_name_count_and_numbered_previews():
    q: Channel[str] = Channel("user_messages", "queue")
    q.put("first")
    q.put("second")
    status = q.status()
    lines = status.splitlines()
    assert lines[0] == "user_messages: 2 pending"
    assert lines[1] == "  1. first"
    assert lines[2] == "  2. second"


def test_status_flattens_newlines_and_clips_overlong_items():
    q: Channel[str] = Channel("user_messages", "queue")
    q.put("line1\nline2\nline3")
    q.put("x" * 200)
    status = q.status(max_items=3, max_chars=30)
    # No raw newline inside preview content (only between lines).
    preview_lines = status.splitlines()[1:]
    for line in preview_lines:
        # Strip "  N. " prefix; remaining content has no embedded \n.
        assert "\n" not in line
        assert len(line) <= len("  N. ") + 30
    assert "↵" in status


def test_status_overflow_summary_when_more_than_max_items():
    q: Channel[int] = Channel("jobs", "queue")
    for i in range(7):
        q.put(i)
    status = q.status(max_items=3)
    assert "jobs: 7 pending" in status
    assert "… 4 more" in status


def test_status_previews_non_string_items_via_pformat():
    q: Channel[dict] = Channel("jobs", "queue")
    q.put({"id": 42, "kind": "build"})
    status = q.status()
    assert "jobs: 1 pending" in status
    assert "'id': 42" in status


def test_output_queue_status_delegates_to_input_queue():
    """OutputQueue exposes status() so the LLM can peek mid-turn without
    reaching into the hidden producer-side queue."""
    q: Channel[str] = Channel("user_messages", "queue")
    q.put("waiting")
    assert q.reader.status() == q.status()


# ---------------------------------------------------------------------------
# QueueManager.race()
# ---------------------------------------------------------------------------


def _qm_with(*names: str) -> tuple[QueueManager, list[Channel]]:
    """Build a QueueManager + N queue-mode channels in registration order."""
    qm = QueueManager()
    channels = [qm.queue(n) for n in names]
    return qm, channels


@pytest.mark.asyncio
async def test_race_returns_fast_path():
    """If a channel already has an item, race returns it immediately."""
    qm, (q1, q2) = _qm_with("q1", "q2")
    q2.put("fromq2")
    items = await qm.race()
    assert items == [("q2", "fromq2")]
    assert q2.qsize() == 0
    # q1 was never touched.
    assert q1.qsize() == 0
    assert not q1.has_waiters()


@pytest.mark.asyncio
async def test_race_races_blocking_waiters():
    qm, (q1, q2) = _qm_with("q1", "q2")
    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    q1.put("hello")
    items = await asyncio.wait_for(waiter, timeout=0.5)
    assert items == [("q1", "hello")]
    # Losing waiter should be cancelled — no items stranded on q2.
    assert q2.qsize() == 0
    assert not q2.has_waiters()


@pytest.mark.asyncio
async def test_race_fast_path_does_not_touch_other_channels():
    """Pre-loaded item on q2 → fast path returns it; q1 stays untouched."""
    qm, (q1, q2) = _qm_with("q1", "q2")
    q2.put("preloaded")
    items = await qm.race()
    assert items == [("q2", "preloaded")]
    assert q1.qsize() == 0
    assert not q1.has_waiters()
    assert q2.qsize() == 0


@pytest.mark.asyncio
async def test_race_multi_done_restores_losers_to_head():
    """When multiple racing tasks complete in the same tick, the first
    is the winner and the rest must be re-pushed to the head of their
    source channels — not silently dropped.

    Drives the multi-done branch by issuing puts back-to-back (no
    intervening yield) so all racing tasks become done together.
    """
    qm, (q1, q2, q3) = _qm_with("q1", "q2", "q3")
    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    q1.put("a")
    q2.put("b")
    q3.put("c")
    items = await asyncio.wait_for(waiter, timeout=0.5)
    name, item = items[0]
    consumed = {(name, item)}
    for q, expected in [(q1, ("q1", "a")), (q2, ("q2", "b")), (q3, ("q3", "c"))]:
        if expected in consumed:
            assert q.qsize() == 0
        else:
            assert q.snapshot() == [expected[1]], (
                f"Loser {expected!r} must be restored to {q.name}'s head"
            )


@pytest.mark.asyncio
async def test_race_multi_done_winner_is_first_in_registration_order():
    """When multiple racing tasks complete in the same tick, the winner
    must be the channel registered earliest — same FIFO-by-position
    contract the fast path documents. ``done`` is a set, so picking by
    iteration would be non-deterministic.
    """
    for queue_order in (["q1", "q2", "q3"], ["q3", "q2", "q1"], ["q2", "q3", "q1"]):
        qm, channels = _qm_with(*queue_order)
        by_name = dict(zip(queue_order, channels, strict=True))

        waiter = asyncio.create_task(qm.race())
        await asyncio.sleep(0)
        for n in queue_order:
            by_name[n].put(f"item-from-{n}")
        items = await asyncio.wait_for(waiter, timeout=0.5)

        assert items[0][0] == queue_order[0], (
            f"For order {queue_order}, expected winner={queue_order[0]} got {items[0][0]}"
        )
        # Losers' items must still be on their channels.
        for n in queue_order[1:]:
            assert by_name[n].snapshot() == [f"item-from-{n}"]


@pytest.mark.asyncio
async def test_race_only_fires_on_get_for_winner():
    """The winner's on_get fires exactly once; losers' hooks must not fire
    even if their racing tasks completed in the same tick."""
    seen_q1: list[str] = []
    seen_q2: list[str] = []
    qm = QueueManager()
    q1 = qm.queue("q1", on_get=seen_q1.append)
    q2 = qm.queue("q2", on_get=seen_q2.append)
    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    q1.put("a")
    q2.put("b")
    items = await asyncio.wait_for(waiter, timeout=0.5)
    name, _ = items[0]
    if name == "q1":
        assert seen_q1 == ["a"]
        assert seen_q2 == []
    else:
        assert seen_q1 == []
        assert seen_q2 == ["b"]


@pytest.mark.asyncio
async def test_race_outer_cancel_restores_late_set_results():
    """If race is cancelled while a producer set a waiter's result
    mid-cancellation, the item must end up restored to its source
    channel (not stranded in the cancelled future).
    """
    qm, (q,) = _qm_with("q")
    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    waiter.cancel()
    q.put("late-result")
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert q.snapshot() == ["late-result"]


@pytest.mark.asyncio
async def test_on_get_hook_swallows_base_exception():
    """A hook that raises BaseException (not just Exception) must still
    let get() return the item. Catching BaseException is a deliberate
    trade-off: the hook is fire-and-forget UI bookkeeping and must
    never affect item delivery.
    """
    seen: list[str] = []

    def boom(item: str) -> None:
        seen.append(item)
        raise BaseException("simulated hook failure")  # noqa: TRY002

    q: Channel[str] = Channel("q", "queue", on_get=boom)
    q.put("hi")
    assert await q.get() == "hi"
    assert seen == ["hi"]


@pytest.mark.asyncio
async def test_race_no_channels_raises():
    """race() requires at least one channel (queue or event);
    empty managers raise ValueError."""
    qm = QueueManager()
    with pytest.raises(ValueError):
        await qm.race()


# ---------------------------------------------------------------------------
# Robustness — locks the bug-hunt findings on MR !139
# ---------------------------------------------------------------------------


def test_channel_init_rejects_unknown_mode():
    """Mode is a string parameter; ``Literal["queue", "event"]`` doesn't
    enforce at runtime. A typo'd mode (e.g. ``"evnet"``) used to fall
    through to queue-mode logic in ``put()`` and silently buffer items
    that would never be ``get()``'d. Constructor must reject unknown
    values explicitly.
    """
    with pytest.raises(ValueError, match="mode"):
        Channel("foo", "evnet")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_race_propagates_drain_one_failure_and_restores_other_channels():
    """If ``_drain_one`` raises a non-CancelledError (hypothetical
    today; future-proof guard), ``race()`` must:

    1. Propagate the exception (not swallow it),
    2. Restore items consumed by losing channels back to their head,
    3. Not double-restore — the failed channel's already-popped item
       is acceptably lost (Task.result() raises on retrieval, the
       cleanup branch swallows it).

    Locks the contract that future ``_drain_one`` failure modes don't
    silently strand items across the channel set.
    """

    class _ExplodingChannel(Channel):
        async def _drain_one(self):
            # Simulate an internal failure that completes the task with
            # an exception rather than a value.
            raise RuntimeError("simulated drain failure")

    # Build a manager with one exploding channel + one normal channel.
    # The exploding one must be FIRST in registration order so it wins
    # the race (race picks by registration order on multi-done).
    qm = QueueManager()
    explode = _ExplodingChannel("explode", "queue")
    qm._channels["explode"] = explode
    normal = qm.queue("normal")

    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    # Trigger the drain on both. Exploding channel's _drain_one raises;
    # normal channel's get-task completes with "saved".
    explode.put("would-have-won")
    normal.put("saved")

    with pytest.raises(RuntimeError, match="simulated drain failure"):
        await asyncio.wait_for(waiter, timeout=0.5)

    # Loser's item must still be reachable on its source channel. The
    # exploding channel may or may not retain its put — that's
    # acceptable (the failed task's value is unreachable via Task.result).
    assert normal.snapshot() == ["saved"]


@pytest.mark.asyncio
async def test_race_documented_exceptions_only():
    """``race()`` should raise only the exceptions documented in its
    contract: ``ValueError`` (no channels at all) and propagation
    of any exception raised inside racing tasks. No catch-alls that
    would hide bugs from the dispatcher.

    Smoke test that the empty case raises ValueError and the happy
    path doesn't raise anything.
    """
    qm = QueueManager()
    with pytest.raises(ValueError):
        await qm.race()

    qm.queue("q1").put("hi")
    items = await qm.race()
    assert items == [("q1", "hi")]
