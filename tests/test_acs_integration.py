# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import asyncio
import threading
from nooa.acs.schema import ACSDecision, ACSHook
from nooa.acs.handler import ACSHandler, AuditOnlyHandler
from nooa.acs.integration import (
    ACSApprovalRequiredError,
    ACSBlockedError,
    ACSDeferExhaustedError,
    acs_thread_safe,
    enforce_acs_decision,
)

class MockHandler(ACSHandler):
    def __init__(self, decisions):
        self.decisions = decisions
        self.call_count = 0
        
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        decision = self.decisions[self.call_count]
        self.call_count += 1
        return decision

@pytest.mark.asyncio
async def test_enforce_allow():
    handler = MockHandler([ACSDecision(status="allow")])
    
    async def mock_execute():
        return "success"
        
    result = await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, mock_execute
    )
    assert result == "success"

@pytest.mark.asyncio
async def test_enforce_deny():
    handler = MockHandler([ACSDecision(status="deny", reason="Unauthorized")])
    
    async def mock_execute():
        return "success"
        
    with pytest.raises(ACSBlockedError, match="Unauthorized"):
        await enforce_acs_decision(
            handler, "toolCallRequest", {}, {}, mock_execute
        )

@pytest.mark.asyncio
async def test_enforce_defer_then_allow():
    # First it defers for 10ms, then it allows
    handler = MockHandler([
        ACSDecision(status="defer", retry_after_ms=10),
        ACSDecision(status="allow")
    ])
    
    async def mock_execute():
        return "success after retry"
        
    result = await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, mock_execute
    )
    assert result == "success after retry"
    assert handler.call_count == 2


@pytest.mark.asyncio
async def test_ask_requires_approval_without_running_action():
    handler = MockHandler(
        [ACSDecision(status="ask", reason="Needs review", decision_id="decision-7")]
    )
    executed = False

    def execute():
        nonlocal executed
        executed = True

    with pytest.raises(ACSApprovalRequiredError) as error:
        await enforce_acs_decision(handler, "toolCallRequest", {}, {}, execute)

    assert not isinstance(error.value, ACSBlockedError)
    assert error.value.reason == "Needs review"
    assert error.value.decision_id == "decision-7"
    assert not executed


@pytest.mark.asyncio
async def test_defer_retry_limit_has_distinct_outcome():
    handler = MockHandler([ACSDecision(status="defer", retry_after_ms=0)] * 3)

    with pytest.raises(ACSDeferExhaustedError, match="retry limit"):
        await enforce_acs_decision(
            handler,
            "toolCallRequest",
            {},
            {},
            lambda: None,
            max_defer_retries=2,
        )

    assert handler.call_count == 3


@pytest.mark.asyncio
async def test_defer_deadline_has_distinct_outcome():
    handler = MockHandler([ACSDecision(status="defer", retry_after_ms=1000)])

    with pytest.raises(ACSDeferExhaustedError, match="deadline"):
        await enforce_acs_decision(
            handler,
            "toolCallRequest",
            {},
            {},
            lambda: None,
            max_defer_retries=10,
            defer_timeout=0.01,
        )

    assert handler.call_count == 1


@pytest.mark.asyncio
async def test_slow_deferred_evaluation_expires_without_executing_action():
    class SlowAfterDeferHandler(ACSHandler):
        def __init__(self):
            self.call_count = 0

        async def evaluate(self, hook):
            self.call_count += 1
            if self.call_count == 1:
                return ACSDecision(status="defer", retry_after_ms=0)
            await asyncio.sleep(0.05)
            return ACSDecision(status="allow")

    handler = SlowAfterDeferHandler()
    executed = False

    def execute():
        nonlocal executed
        executed = True

    with pytest.raises(ACSDeferExhaustedError, match="deadline"):
        await enforce_acs_decision(
            handler,
            "toolCallRequest",
            {},
            {},
            execute,
            max_defer_retries=2,
            defer_timeout=0.01,
        )

    assert handler.call_count == 2
    assert not executed


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_timeout", [None, 1.0])
async def test_handler_timeout_propagates_before_defer_deadline(defer_timeout):
    class TimedOutHandler(ACSHandler):
        async def evaluate(self, hook):
            raise asyncio.TimeoutError("handler-specific timeout")

    with pytest.raises(asyncio.TimeoutError, match="handler-specific timeout"):
        await enforce_acs_decision(
            TimedOutHandler(),
            "toolCallRequest",
            {},
            {},
            lambda: None,
            defer_timeout=defer_timeout,
        )


@pytest.mark.asyncio
async def test_sync_callback_returning_coroutine_is_called_once_and_awaited():
    handler = MockHandler([ACSDecision(status="allow")])
    calls = []

    def execute():
        calls.append("called")

        async def finish():
            calls.append("awaited")
            return "done"

        return finish()

    assert await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, execute
    ) == "done"
    assert calls == ["called", "awaited"]


@pytest.mark.asyncio
async def test_only_marked_callback_runs_in_worker_thread():
    handler = MockHandler([ACSDecision(status="allow")])
    event_loop_thread = threading.get_ident()
    callback_threads = []

    @acs_thread_safe
    def execute():
        callback_threads.append(threading.get_ident())
        return "done"

    assert await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, execute
    ) == "done"
    assert callback_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_unmarked_callback_stays_on_event_loop_thread():
    handler = MockHandler([ACSDecision(status="allow")])
    event_loop_thread = threading.get_ident()
    callback_threads = []

    def execute():
        callback_threads.append(threading.get_ident())
        return "done"

    assert await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, execute
    ) == "done"
    assert callback_threads == [event_loop_thread]


@pytest.mark.asyncio
async def test_marked_callback_returning_coroutine_is_awaited():
    handler = MockHandler([ACSDecision(status="allow")])
    calls = []

    @acs_thread_safe
    def execute():
        calls.append("called")

        async def finish():
            calls.append("awaited")
            return "done"

        return finish()

    assert await enforce_acs_decision(
        handler, "toolCallRequest", {}, {}, execute
    ) == "done"
    assert calls == ["called", "awaited"]


@pytest.mark.asyncio
async def test_audit_only_sync_sink_is_offloaded_and_does_not_block():
    sink_started = threading.Event()
    release_sink = threading.Event()
    sink_threads = []
    event_loop_thread = threading.get_ident()

    def sink(hook):
        sink_threads.append(threading.get_ident())
        sink_started.set()
        release_sink.wait(timeout=1)

    handler = AuditOnlyHandler(sink)
    try:
        decision = await asyncio.wait_for(
            handler.evaluate(ACSHook(hook_type="toolCallRequest", payload={})),
            timeout=0.1,
        )
        assert decision.status == "allow"
        assert await asyncio.to_thread(sink_started.wait, 0.5)
        assert sink_threads[0] != event_loop_thread
    finally:
        release_sink.set()


@pytest.mark.asyncio
async def test_audit_sink_failure_does_not_change_allow_decision():
    def failing_sink(hook):
        raise RuntimeError("audit unavailable")

    handler = AuditOnlyHandler(failing_sink)
    decision = await handler.evaluate(
        ACSHook(hook_type="toolCallRequest", payload={})
    )
    await handler.drain()

    assert decision.status == "allow"


@pytest.mark.asyncio
async def test_audit_only_async_sink_is_scheduled_without_awaiting():
    sink_started = asyncio.Event()
    release_sink = asyncio.Event()

    async def sink(hook):
        sink_started.set()
        await release_sink.wait()

    handler = AuditOnlyHandler(sink)
    decision = await handler.evaluate(
        ACSHook(hook_type="toolCallRequest", payload={})
    )

    assert decision.status == "allow"
    await asyncio.wait_for(sink_started.wait(), timeout=0.1)
    release_sink.set()


@pytest.mark.asyncio
async def test_audit_delivery_capacity_is_bounded_and_shutdown_drains():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def stalled_sink(hook):
        calls.append(hook)
        started.set()
        await release.wait()

    handler = AuditOnlyHandler(stalled_sink)
    handler.MAX_PENDING_DELIVERIES = 1
    hooks = [
        ACSHook(hook_type="toolCallRequest", payload={"number": number})
        for number in range(3)
    ]

    assert (await handler.evaluate(hooks[0])).status == "allow"
    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert (await handler.evaluate(hooks[1])).status == "allow"
    assert len(handler._delivery_tasks) == 1
    assert (await handler.evaluate(hooks[2])).status == "allow"
    assert len(handler._delivery_tasks) == 1
    assert len(calls) == 1

    shutdown = asyncio.create_task(handler.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    await shutdown

    assert handler._delivery_tasks == set()
    assert (await handler.evaluate(hooks[2])).status == "allow"
    assert len(calls) == 1
