# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The Python/native boundary must preserve outcomes and callback lifetimes."""

import asyncio

import pytest

from nooa._relay_callback import execute_callback


@pytest.mark.parametrize("swallow_native_error", [False, True])
async def test_callback_exception_is_private_but_original_reaches_caller(swallow_native_error):
    original = ValueError("private diagnostic")
    native_errors = []

    async def callback():
        raise original

    async def execute(wrapped):
        try:
            await wrapped()
        except Exception as exc:
            native_errors.append(str(exc))
            if swallow_native_error:
                return "replacement"
            raise

    with pytest.raises(ValueError) as caught:
        await execute_callback(execute, callback)
    assert caught.value is original
    assert native_errors == ["NOOA callback failed"]


async def test_native_short_circuit_and_native_failure_are_preserved():
    async def callback():
        pytest.fail("short-circuited callback must not run")

    async def cached(wrapped):
        return "cached"

    assert await execute_callback(cached, callback) == "cached"
    original = RuntimeError("native guardrail rejection")

    async def rejected(wrapped):
        raise original

    with pytest.raises(RuntimeError) as caught:
        await execute_callback(rejected, callback)
    assert caught.value is original


async def test_callback_cancellation_text_is_private_but_original_reaches_caller():
    original = asyncio.CancelledError("PRIVATE cancellation detail")
    native_errors = []

    async def callback():
        raise original

    async def execute(wrapped):
        try:
            await wrapped()
        except asyncio.CancelledError as exc:
            native_errors.append(str(exc))
            raise

    with pytest.raises(asyncio.CancelledError) as caught:
        await execute_callback(execute, callback)
    assert caught.value is original
    assert native_errors == [""]


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
async def test_real_relay_never_publishes_private_callback_exception_text(error_type):
    relay = pytest.importorskip("nemo_relay")
    from uuid import uuid4

    events = []
    registration = f"nooa-private-callback-{uuid4().hex}"
    relay.subscribers.register(registration, events.append)
    original = error_type("PRIVATE_CALLBACK_DIAGNOSTIC")

    async def callback(args):
        raise original

    try:
        with (
            relay.use_scope_stack(relay.create_scope_stack()),
            relay.scope.scope("private-callback-test", relay.ScopeType.Agent),
        ):
            with pytest.raises(error_type) as caught:
                await execute_callback(
                    lambda wrapped: relay.tools.execute("private-test-tool", {}, wrapped), callback
                )
            assert caught.value is original
        await relay.subscribers.flush_async()
        assert any(event.name == "private-test-tool" for event in events)
        assert all("PRIVATE_CALLBACK_DIAGNOSTIC" not in str(event.metadata) for event in events)
        assert all("PRIVATE_CALLBACK_DIAGNOSTIC" not in str(event.data) for event in events)
    finally:
        await relay.subscribers.flush_async()
        relay.subscribers.deregister(registration)


async def test_repeated_cancellation_waits_for_own_callback_cleanup():
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    callback_task = None

    async def callback():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async def execute(wrapped):
        nonlocal callback_task
        callback_task = asyncio.create_task(wrapped())
        # Simulate Relay: cancelling its awaitable schedules callback cancellation
        # but reports cancellation before that callback's finally has finished.
        try:
            await asyncio.shield(callback_task)
        except asyncio.CancelledError:
            callback_task.cancel()
            raise

    call = asyncio.create_task(execute_callback(execute, callback))
    await entered.wait()
    call.cancel("original caller cancellation")
    await cleaning.wait()
    call.cancel()
    await asyncio.sleep(0)
    assert not call.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await call
    assert str(caught.value) == "original caller cancellation"
    assert callback_task is not None and callback_task.done()


async def test_cancelled_native_call_cannot_start_a_late_callback():
    ready = asyncio.Event()
    wrapped_callback = None

    async def callback():
        pytest.fail("a cancelled call must not start user work")

    async def execute(wrapped):
        nonlocal wrapped_callback
        wrapped_callback = wrapped
        ready.set()
        await asyncio.Event().wait()

    call = asyncio.create_task(execute_callback(execute, callback))
    await ready.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert wrapped_callback is not None
    with pytest.raises(asyncio.CancelledError):
        await wrapped_callback()


async def test_real_relay_nested_callback_scopes_end_after_repeated_cancellation():
    relay = pytest.importorskip("nemo_relay")
    from uuid import uuid4

    from nooa import Agent
    from nooa.nemo_relay_middleware import (
        nemo_relay_agent_call_middleware,
        nemo_relay_scope,
        nemo_relay_tool_middleware,
    )
    from nooa.runtime.middleware import AgentCallContext, ExecutePythonContext
    from nooa.unifiedllm import FakeLLMClient

    class TestAgent(Agent, llm=FakeLLMClient()):
        pass

    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    events = []
    registration = f"nooa-callback-cancellation-{uuid4().hex}"
    root_name = f"cancel-root-{uuid4().hex}"
    relay.subscribers.register(registration, events.append)

    async def inner(ctx):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    async def execute(ctx):
        return await nemo_relay_agent_call_middleware(AgentCallContext(method_name="nested"), inner)

    async def run():
        async with nemo_relay_scope(TestAgent(), root_name):
            await nemo_relay_agent_call_middleware(
                AgentCallContext(method_name="outer"),
                lambda ctx: nemo_relay_tool_middleware(ExecutePythonContext(code="pass"), execute),
            )

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), timeout=10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        await relay.subscribers.flush_async()
        starts = {
            str(event.uuid): event.name for event in events if event.scope_category == "start"
        }
        ends = {str(event.uuid) for event in events if event.scope_category == "end"}
        assert root_name in starts.values()
        assert {"NoneType.outer", "execute_python", "NoneType.nested"} <= set(starts.values())
        assert set(starts) <= ends
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await relay.subscribers.flush_async()
        relay.subscribers.deregister(registration)
