# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for generic live-session turn ownership and resource cleanup."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

import pytest
from nooa_coder.sessions import (
    SessionBusyError,
    SessionRuntime,
    SessionRuntimeClosedError,
    SessionRuntimePool,
)

# Bounds a hang, not the expected duration.
_HANG_TIMEOUT = 30


@pytest.fixture(autouse=True, params=[False, True], ids=["scheduled", "eager"])
async def task_scheduling(request):
    """Exercise both deferred task startup and execution at create_task time."""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory if request.param else None)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


class _RuntimeValue:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


async def test_same_session_rejects_a_second_foreground_turn():
    runtime = SessionRuntime("one", object())

    async def _claim_again() -> None:
        async with runtime.turn():
            pass

    async with runtime.turn():
        assert runtime.busy is True
        # Bounded: if turns start queueing instead of failing fast — the exact
        # regression — this wedges on the inner lock and hangs the suite.
        with pytest.raises(SessionBusyError):
            await asyncio.wait_for(_claim_again(), timeout=_HANG_TIMEOUT)


@pytest.mark.parametrize("contenders", [2, 64])
async def test_simultaneous_turn_claims_do_not_queue(contenders):
    """Only one simultaneous caller wins; the rest fail instead of queueing."""
    runtime = SessionRuntime("one", object())
    start = asyncio.Event()
    release = asyncio.Event()
    attempted = asyncio.Event()
    outcomes: list[str] = []

    def record(outcome: str) -> None:
        outcomes.append(outcome)
        if len(outcomes) == contenders:
            attempted.set()

    async def claim() -> None:
        await start.wait()
        try:
            async with runtime.turn():
                record("entered")
                await release.wait()
        except SessionBusyError:
            record("busy")

    tasks = [asyncio.create_task(claim()) for _ in range(contenders)]
    start.set()
    await asyncio.wait_for(attempted.wait(), timeout=_HANG_TIMEOUT)
    release.set()
    await asyncio.gather(*tasks)

    assert outcomes.count("entered") == 1
    assert outcomes.count("busy") == contenders - 1
    assert not runtime.busy


async def test_waiting_turn_runs_after_current_turn():
    runtime = SessionRuntime("one", object())
    release = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with runtime.turn():
            order.append("first-enter")
            await release.wait()
            order.append("first-exit")

    async def second() -> None:
        async with runtime.turn(wait=True):
            order.append("second-enter")

    first_task = asyncio.create_task(first())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert order == ["first-enter"]
    release.set()
    await asyncio.gather(first_task, second_task)
    assert order == ["first-enter", "first-exit", "second-enter"]


async def test_cancelled_waiting_turn_releases_its_claim():
    runtime = SessionRuntime("one", object())
    queued = asyncio.Event()

    async def wait_for_turn():
        queued.set()
        async with runtime.turn(wait=True):
            pytest.fail("Cancelled waiter must not enter the turn")

    async with runtime.turn():
        waiter = asyncio.create_task(wait_for_turn())
        await queued.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert runtime.busy is True

    assert runtime.busy is False
    async with runtime.turn():
        assert runtime.busy is True
    await runtime.close()


async def test_different_sessions_run_foreground_turns_concurrently():
    pool: SessionRuntimePool[str] = SessionRuntimePool()
    first = await pool.add("first", "A")
    second = await pool.add("second", "B")
    both_entered = asyncio.Event()
    entered: set[str] = set()

    async def run(runtime: SessionRuntime[str]) -> None:
        async with runtime.turn() as value:
            entered.add(value)
            if len(entered) == 2:
                both_entered.set()
            await both_entered.wait()

    # Deadlock detector: if the two sessions serialized, neither would reach
    # both_entered and the gather would hang. Generous so a loaded runner
    # cannot flake it; a real serialization bug still fails, just later.
    await asyncio.wait_for(asyncio.gather(run(first), run(second)), timeout=30)
    assert entered == {"A", "B"}
    await pool.close()


async def test_close_waits_for_active_turn_and_is_idempotent():
    value = _RuntimeValue()
    runtime = SessionRuntime("one", value)
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def active_turn() -> None:
        async with runtime.turn():
            turn_started.set()
            await release_turn.wait()

    turn_task = asyncio.create_task(active_turn())
    await turn_started.wait()
    close_task = asyncio.create_task(runtime.close())
    # A single yield is satisfied by scheduling latency — _close_once has not
    # even started — so it passes with the turn lock removed entirely.
    for _ in range(20):
        await asyncio.sleep(0)
    assert close_task.done() is False
    assert value.close_calls == 0
    release_turn.set()
    await asyncio.gather(turn_task, close_task)
    await runtime.close()

    assert value.close_calls == 1
    assert runtime.is_closed is True
    with pytest.raises(SessionRuntimeClosedError):
        async with runtime.turn():
            pass


async def test_close_cleanup_survives_caller_cancellation():
    value = _RuntimeValue()
    runtime = SessionRuntime("one", value)
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def active_turn() -> None:
        async with runtime.turn():
            turn_started.set()
            await release_turn.wait()

    turn_task = asyncio.create_task(active_turn())
    await turn_started.wait()
    cancelled_close = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    cancelled_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_close

    release_turn.set()
    await turn_task
    await runtime.close()
    assert runtime.is_closed is True
    assert value.close_calls == 1


async def test_pool_remove_and_close_release_each_runtime_once():
    pool: SessionRuntimePool[_RuntimeValue] = SessionRuntimePool()
    first_value = _RuntimeValue()
    second_value = _RuntimeValue()
    await pool.add("first", first_value)
    await pool.add("second", second_value)

    assert await pool.remove("first") is first_value
    assert await pool.ids() == ("second",)
    with pytest.raises(KeyError):
        await pool.remove("first")
    await pool.close()
    await pool.close()

    assert first_value.close_calls == 1
    assert second_value.close_calls == 1
    with pytest.raises(SessionRuntimeClosedError):
        await pool.add("third", _RuntimeValue())


async def test_remove_unregisters_even_when_teardown_fails():
    """A failing close must not strand the session id in the pool.

    Leaving the entry registered would prevent reuse of its identifier and
    return a closed runtime to later callers.
    """

    class _Failing:
        async def close(self) -> None:
            raise RuntimeError("teardown blew up")

    pool: SessionRuntimePool[_Failing] = SessionRuntimePool()
    await pool.add("one", _Failing())

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await pool.remove("one")

    assert await pool.ids() == ()
    with pytest.raises(KeyError):
        await pool.get("one")


async def test_cancelled_remove_keeps_id_reserved_until_teardown_finishes():
    """Cancellation must not expose the id while its old runtime is still live."""
    started = asyncio.Event()
    release = asyncio.Event()

    class _Slow:
        async def close(self) -> None:
            started.set()
            await release.wait()

    pool: SessionRuntimePool[_Slow] = SessionRuntimePool()
    await pool.add("one", _Slow())

    remover = asyncio.create_task(pool.remove("one"))
    await asyncio.wait_for(started.wait(), timeout=_HANG_TIMEOUT)
    remover.cancel()
    with pytest.raises(asyncio.CancelledError):
        await remover

    assert await pool.ids() == ("one",)
    with pytest.raises(ValueError, match="already registered"):
        await pool.add("one", _Slow())

    # Join the still-running removal before releasing cleanup, avoiding polling.
    joined = asyncio.Event()

    async def join_removal():
        joined.set()
        return await pool.remove("one")

    survivor = asyncio.create_task(join_removal())
    await joined.wait()
    release.set()
    await survivor
    assert await pool.ids() == ()


@pytest.mark.parametrize("finish", ["exception", "cancel"])
async def test_interrupted_active_turn_releases_waiting_turn(finish):
    runtime = SessionRuntime("one", object())
    active = asyncio.Event()
    fail = asyncio.Event()
    queued = asyncio.Event()
    entered = asyncio.Event()

    async def first():
        async with runtime.turn():
            active.set()
            await fail.wait()
            raise ValueError("turn failed")

    async def second():
        queued.set()
        async with runtime.turn(wait=True):
            entered.set()

    first_task = asyncio.create_task(first())
    await active.wait()
    second_task = asyncio.create_task(second())
    await queued.wait()
    assert not entered.is_set()
    if finish == "cancel":
        first_task.cancel()
    else:
        fail.set()
    with pytest.raises(asyncio.CancelledError if finish == "cancel" else ValueError):
        await first_task
    await second_task
    assert entered.is_set()
    assert not runtime.busy
    async with runtime.turn():
        pass
    await runtime.close()


async def test_queued_turn_reserves_unlocked_handoff_against_fail_fast_caller():
    runtime = SessionRuntime("one", object())
    queued = asyncio.Event()
    entered = asyncio.Event()

    async def waiter():
        queued.set()
        async with runtime.turn(wait=True):
            entered.set()

    async with runtime.turn():
        task = asyncio.create_task(waiter())
        await queued.wait()
    # Exiting releases the lock, but this task has not yielded to its successor.
    assert not entered.is_set()
    assert runtime.busy
    with pytest.raises(SessionBusyError):
        async with runtime.turn():
            pytest.fail("A new caller must not bypass the reserved turn")
    await task
    assert entered.is_set()
    assert not runtime.busy


@pytest.mark.parametrize("cancel_at_handoff", [False, True])
async def test_cancelled_front_waiter_wakes_next_in_fifo_order(cancel_at_handoff):
    runtime = SessionRuntime("one", object())
    queued = [asyncio.Event() for _ in range(3)]
    order: list[int] = []

    async def waiter(index):
        queued[index].set()
        async with runtime.turn(wait=True):
            order.append(index)
            await asyncio.sleep(0)

    async with runtime.turn():
        tasks = []
        for index in range(3):
            tasks.append(asyncio.create_task(waiter(index)))
            await queued[index].wait()
        if not cancel_at_handoff:
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
    if cancel_at_handoff:
        # Cancel after release wakes this waiter, before it can acquire.
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
    await asyncio.gather(*tasks[1:])
    assert order == [1, 2]
    assert not runtime.busy
    async with runtime.turn():
        pass


async def test_close_rejects_queued_and_new_turns_before_cleanup():
    value = _RuntimeValue()
    runtime = SessionRuntime("one", value)
    queued = asyncio.Event()
    closing = asyncio.Event()

    async def waiter():
        queued.set()
        async with runtime.turn(wait=True):
            pytest.fail("A queued turn must recheck state after acquiring the lock")

    async def close():
        closing.set()
        await runtime.close()

    async with runtime.turn():
        waiting = asyncio.create_task(waiter())
        await queued.wait()
        closer = asyncio.create_task(close())
        await closing.wait()
        for wait in (False, True):
            with pytest.raises(SessionRuntimeClosedError):
                async with runtime.turn(wait=wait):
                    pytest.fail("Closing must reject new turn claims")
        assert not runtime.is_closed
        assert value.close_calls == 0
    with pytest.raises(SessionRuntimeClosedError):
        await waiting
    await closer
    assert value.close_calls == 1
    assert runtime.is_closed
    assert not runtime.busy


class _BlockingCleanup:
    def __init__(self, *, error: BaseException | None = None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.error = error

    async def close(self):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("failure", [None, ValueError("close failed"), asyncio.CancelledError()])
async def test_concurrent_close_callers_share_cleanup_and_its_result(failure):
    value = _BlockingCleanup(error=failure)
    runtime = SessionRuntime("one", value)
    callers = [asyncio.create_task(runtime.close()) for _ in range(4)]
    await value.started.wait()
    callers[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await callers[0]
    assert value.calls == 1
    assert not runtime.is_closed
    with pytest.raises(SessionRuntimeClosedError):
        async with runtime.turn():
            pass
    value.release.set()
    results = await asyncio.gather(*callers[1:], return_exceptions=True)
    if failure is None:
        assert results == [None] * 3
        await runtime.close()
    else:
        assert all(isinstance(result, type(failure)) for result in results)
        with pytest.raises(type(failure)):
            await runtime.close()
    assert value.calls == 1
    assert runtime.is_closed
    assert not runtime.busy


@pytest.mark.parametrize("failure", [None, ValueError("remove failed"), asyncio.CancelledError()])
async def test_concurrent_removers_reserve_id_until_one_shared_cleanup_finishes(failure):
    value = _BlockingCleanup(error=failure)
    pool: SessionRuntimePool[object] = SessionRuntimePool()
    original = await pool.add("one", value)
    removers = [asyncio.create_task(pool.remove("one")) for _ in range(4)]
    await value.started.wait()
    removers[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await removers[0]
    assert await pool.get("one") is original
    with pytest.raises(ValueError, match="already registered"):
        await pool.add("one", object())
    assert value.calls == 1
    value.release.set()
    results = await asyncio.gather(*removers[1:], return_exceptions=True)
    if failure is None:
        assert all(result is value for result in results)
    elif isinstance(failure, asyncio.CancelledError):
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
    else:
        assert all(result is failure for result in results)
    assert original.is_closed
    assert await pool.ids() == ()
    replacement = await pool.add("one", object())
    assert await pool.get("one") is replacement
    async with replacement.turn():
        pass
    await pool.close()
    assert value.calls == 1


async def test_pool_close_overlaps_remove_and_survives_cancelled_caller():
    pool: SessionRuntimePool[_BlockingCleanup] = SessionRuntimePool()
    first, second = _BlockingCleanup(), _BlockingCleanup()
    runtimes = [await pool.add("one", first), await pool.add("two", second)]
    remover = asyncio.create_task(pool.remove("one"))
    await first.started.wait()
    closer = asyncio.create_task(pool.close())
    await second.started.wait()
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer
    with pytest.raises(SessionRuntimeClosedError):
        await pool.add("three", _BlockingCleanup())
    assert set(await pool.ids()) == {"one", "two"}
    first.release.set()
    assert await remover is first
    assert await pool.ids() == ("two",)
    second.release.set()
    await pool.close()
    assert all(runtime.is_closed for runtime in runtimes)
    assert first.calls == second.calls == 1
    assert await pool.ids() == ()


async def test_pool_close_waits_for_every_cleanup_and_reports_all_failures():
    pool: SessionRuntimePool[_BlockingCleanup] = SessionRuntimePool()
    failures = [ValueError("one"), RuntimeError("two")]
    values = [_BlockingCleanup(error=error) for error in (*failures, None)]
    runtimes = [await pool.add(str(index), value) for index, value in enumerate(values)]
    closer = asyncio.create_task(pool.close())
    await asyncio.gather(*(value.started.wait() for value in values))
    # Failure of early cleanups must not skip or cancel the remaining one.
    values[0].release.set()
    values[1].release.set()
    await asyncio.gather(*(runtime.close() for runtime in runtimes[:2]), return_exceptions=True)
    assert not closer.done()
    assert not runtimes[2].is_closed
    values[2].release.set()
    with pytest.raises(ExceptionGroup) as exc_info:
        await closer
    assert list(exc_info.value.exceptions) == failures
    with pytest.raises(ExceptionGroup) as repeated:
        await pool.close()
    assert repeated.value is exc_info.value
    assert all(runtime.is_closed for runtime in runtimes)
    assert all(value.calls == 1 for value in values)
    assert await pool.ids() == ()


async def test_many_queued_turns_preserve_exclusivity_through_cancellation_and_errors():
    pool: SessionRuntimePool[int] = SessionRuntimePool()
    runtimes = [await pool.add(str(index), index) for index in range(3)]
    active = [0, 0, 0]
    entered: list[tuple[int, int]] = []
    peak_parallel = 0

    async def turn(runtime, index, queued):
        nonlocal peak_parallel
        queued.set()
        async with runtime.turn(wait=True) as session:
            active[session] += 1
            try:
                assert active[session] == 1, "Two turns own the same session"
                peak_parallel = max(peak_parallel, sum(active))
                entered.append((session, index))
                for _ in range(3):
                    await asyncio.sleep(0)
                if index % 7 == 0:
                    raise ValueError("turn failed")
            finally:
                active[session] -= 1

    tasks: list[tuple[int, int, asyncio.Task]] = []
    async with AsyncExitStack() as owners:
        for runtime in runtimes:
            await owners.enter_async_context(runtime.turn())
        for index in range(24):
            for runtime in runtimes:
                queued = asyncio.Event()
                task = asyncio.create_task(turn(runtime, index, queued))
                tasks.append((runtime.value, index, task))
                await queued.wait()
        cancelled = [task for _, index, task in tasks if index % 5 == 0]
        for task in cancelled:
            task.cancel()
        results = await asyncio.gather(*cancelled, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)

    results = await asyncio.gather(*(task for _, _, task in tasks), return_exceptions=True)
    for (_, index, _), result in zip(tasks, results, strict=True):
        if index % 5 == 0:
            assert isinstance(result, asyncio.CancelledError)
        elif index % 7 == 0:
            assert isinstance(result, ValueError)
        else:
            assert result is None
    for session in range(3):
        assert [index for owner, index in entered if owner == session] == [
            index for index in range(24) if index % 5 != 0
        ]
    assert active == [0, 0, 0]
    assert peak_parallel == 3
    assert all(not runtime.busy for runtime in runtimes)
    await pool.close()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_cleanup_callback_receives_value_and_overrides_value_close(asynchronous):
    value = _RuntimeValue()
    cleaned = []

    def cleanup(received):
        cleaned.append(received)

    async def async_cleanup(received):
        await asyncio.sleep(0)
        cleanup(received)

    runtime = SessionRuntime("one", value, close=async_cleanup if asynchronous else cleanup)
    await asyncio.gather(runtime.close(), runtime.close())
    assert cleaned == [value]
    assert value.close_calls == 0
    assert runtime.is_closed


async def test_unpublished_runtime_is_reserved_but_not_served():
    """add(available=False) reserves the id for setup work (e.g. a transcript
    replay) without letting get()/ids() serve it; publish() opens it, and the
    owning setup path can still remove() it on failure via include_unavailable.
    """
    pool: SessionRuntimePool[str] = SessionRuntimePool()
    await pool.add("one", "A", available=False)

    with pytest.raises(KeyError):
        await pool.get("one")
    assert await pool.ids() == ()
    with pytest.raises(ValueError):
        await pool.add("one", "duplicate")  # still reserved
    with pytest.raises(KeyError):
        await pool.remove("one")  # invisible to ordinary callers

    await pool.publish("one")
    assert (await pool.get("one")).value == "A"
    assert await pool.ids() == ("one",)

    await pool.add("two", "B", available=False)
    assert await pool.remove("two", include_unavailable=True) == "B"
    with pytest.raises(KeyError):
        await pool.publish("two")
    assert await pool.ids() == ("one",)
