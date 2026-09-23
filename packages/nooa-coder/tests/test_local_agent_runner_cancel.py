# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed owner-thread cancel dispatch must not permanently wedge cancellation.

``_request_cancel_on_owner`` is reached from ``swap_agent``/``cancel_for_transition``
via ``_cancel_turn_in_lifecycle``. Unlike ``request_cancel``, it did not roll back
``_cancel_requested`` when dispatching the cancel failed (a done ``ConcurrentFuture``
returning ``False`` from ``.cancel()``, or a closing loop raising ``RuntimeError``
from ``call_soon_threadsafe``). That left ``_cancel_requested`` stuck ``True``
forever, so the early "already requested" guard turned every later cancel request
into a silent no-op that never actually cancelled anything.
"""

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

import pytest
from nooa_coder.interactive.local_agent import LocalAgentRunner, TurnAbandoned, TurnCancelled


class _FakeQueueManager:
    def set_notify_callback(self, callback: Any) -> None:
        pass

    async def shutdown(self, **kwargs: Any) -> None:
        pass

    def running_work_handles(self) -> list[Any]:
        return []

    def channels(self) -> dict[str, Any]:
        return {}

    def handles(self) -> list[Any]:
        return []


class _FakeUserMessages:
    def qsize(self) -> int:
        return 0

    def snapshot(self) -> list[Any]:
        return []

    def set_on_get(self, callback: Any) -> None:
        pass


class _FakeAgent:
    def __init__(self) -> None:
        self.queue_manager = _FakeQueueManager()
        self._user_messages_in = _FakeUserMessages()
        self.cwd = None


def _make_runner() -> LocalAgentRunner:
    return LocalAgentRunner(_FakeAgent(), emit_text=lambda text: None, agent_id="test-agent")


@pytest.mark.asyncio
async def test_failed_cancel_dispatch_rolls_back_cancel_requested():
    runner = _make_runner()
    runner._lifecycle_state = "active"
    runner._in_handle = True
    runner._task = asyncio.get_running_loop().create_future()
    runner._source_task = None
    already_done_future: ConcurrentFuture = ConcurrentFuture()
    already_done_future.set_result(None)
    runner._source_future = already_done_future

    result = runner._request_cancel_on_owner(force=False, notify=True)

    assert result is False
    assert runner._cancel_requested is False
    assert runner._notify_cancelled is False


@pytest.mark.asyncio
async def test_failed_cancel_dispatch_does_not_wedge_later_cancel_requests():
    """Without the rollback, a second call would find ``_cancel_requested`` still
    True and hit the early "already requested" guard, returning True without ever
    dispatching a real cancel -- a permanent, silent no-op."""
    runner = _make_runner()
    runner._lifecycle_state = "active"
    runner._in_handle = True
    runner._task = asyncio.get_running_loop().create_future()
    runner._source_task = None
    already_done_future: ConcurrentFuture = ConcurrentFuture()
    already_done_future.set_result(None)
    runner._source_future = already_done_future

    first = runner._request_cancel_on_owner(force=False, notify=True)
    assert first is False

    # A real cancel becomes dispatchable again: swap in a pending future whose
    # .cancel() succeeds, and confirm the second call actually dispatches it
    # instead of returning a stale True from the "already requested" guard.
    live_future: ConcurrentFuture = ConcurrentFuture()
    runner._source_future = live_future

    second = runner._request_cancel_on_owner(force=False, notify=True)

    assert second is True
    assert runner._cancel_requested is True
    assert live_future.cancelled()


@pytest.mark.asyncio
async def test_cancel_work_settles_a_pending_foreground_with_turn_cancelled():
    """A host awaiting submit_and_wait() used to get a bare None for a cancel,
    indistinguishable from a turn the runner simply abandoned; it now gets a
    typed TurnCancelled so it can confirm the cancel without a timed guess.
    """
    runner = _make_runner()
    runner._lifecycle_state = "active"
    completion: ConcurrentFuture = ConcurrentFuture()
    runner._foreground = completion

    await runner.cancel_work()

    assert isinstance(completion.exception(), TurnCancelled)


@pytest.mark.asyncio
async def test_dispatch_exit_without_a_result_settles_the_foreground_with_turn_abandoned():
    """When the dispatch task finishes normally without any turn producing a
    result (queue manager torn down, exit exception fired), the pending
    foreground is settled with TurnAbandoned instead of None."""
    runner = _make_runner()
    runner._lifecycle_state = "active"
    completion: ConcurrentFuture = ConcurrentFuture()
    runner._foreground = completion

    async def dispatch_that_exits_early() -> None:
        return None

    task = asyncio.get_running_loop().create_task(dispatch_that_exits_early())
    runner._task = task
    await task
    runner._on_done(task)

    error = completion.exception()
    assert isinstance(error, TurnAbandoned)
    assert "before the turn produced a result" in error.reason
