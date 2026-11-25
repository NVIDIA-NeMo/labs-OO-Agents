# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 — optional dependency check must precede imports
"""Native session propagation on Relay versions exposing public stack APIs.

Check native event roots, parentage, scope-local delivery and stack restoration
on the supported Relay integration, without live model calls.
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

relay = pytest.importorskip("nemo_relay")
from nooa import Agent
from nooa.nemo_relay_middleware import install_nemo_relay, nemo_relay_scope
from nooa.runtime.middleware import MIDDLEWARE_AGENT_CALL, AgentCallContext


@pytest.fixture
def relay_events():
    events = []
    relay.subscribers.register(
        "nooa-propagation-test", lambda event: events.append(event.to_dict())
    )
    try:
        with relay.use_scope_stack(relay.create_scope_stack()):
            yield events
    finally:
        relay.subscribers.flush()
        relay.subscribers.deregister("nooa-propagation-test")


@contextmanager
def _session(root):
    with relay.scope.scope("causal-parent", relay.ScopeType.Agent):
        propagation = relay.capture_propagation_context_with_root(root)
        with relay.use_scope_stack(relay.create_scope_stack_from_propagation(propagation)):
            with relay.scope.scope("request", relay.ScopeType.Agent) as handle:
                assert str(handle.uuid) != root
                yield handle


@contextmanager
def _installed(agent):
    uninstall = install_nemo_relay(agent.event_manager)
    try:
        yield
    finally:
        uninstall()


async def _call(agent, name, body):
    return await agent.event_manager.run_middleware(
        MIDDLEWARE_AGENT_CALL, AgentCallContext(agent=agent, method_name=name), body
    )


async def _function_events(events):
    await relay.subscribers.flush_async()
    return [event for event in events if event["category"] == "function"]


@pytest.mark.parametrize("use_helper", [False, True], ids=["ambient-install", "scope-helper"])
async def test_concurrent_calls_keep_explicit_session_root(relay_events, use_helper):
    agent = Agent(llm=MagicMock())
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_finished = asyncio.Event()
    completions = []

    async def first(ctx):
        first_started.set()
        await second_started.wait()
        completions.append("first")
        first_finished.set()
        return ctx

    async def second(ctx):
        await first_started.wait()
        second_started.set()
        await first_finished.wait()
        completions.append("second")
        return ctx

    for _ in range(2):
        root = str(uuid4())
        first_started.clear()
        second_started.clear()
        first_finished.clear()
        relay_events.clear()
        with _session(root) as request:
            if use_helper:
                async with nemo_relay_scope(agent, "nooa") as parent:
                    await asyncio.gather(
                        _call(agent, "first", first), _call(agent, "second", second)
                    )
                    assert relay.scope.get_handle().uuid == parent.uuid
            else:
                parent = request
                with _installed(agent):
                    await asyncio.gather(
                        _call(agent, "first", first), _call(agent, "second", second)
                    )
            assert relay.scope.get_handle().uuid == request.uuid
        events = await _function_events(relay_events)
        assert sorted((event["name"], event["scope_category"]) for event in events) == [
            ("Agent.first", "end"),
            ("Agent.first", "start"),
            ("Agent.second", "end"),
            ("Agent.second", "start"),
        ]
        assert {event["propagation_root_uuid"] for event in events} == {root}
        assert {event["parent_uuid"] for event in events} == {str(parent.uuid)}
    assert completions == ["first", "second", "first", "second"]


@pytest.mark.parametrize("nested_helper", [False, True], ids=["methods", "nested-scope"])
async def test_same_task_nesting_keeps_scope_local_subscribers(relay_events, nested_helper):
    agent = Agent(llm=MagicMock())
    local_events = []
    root = str(uuid4())
    handles = {}

    async def inner(ctx):
        handles["inner"] = relay.scope.get_handle()
        return ctx

    async def outer(ctx):
        handles["outer"] = relay.scope.get_handle()
        await _call(agent, "inner", inner)
        assert relay.scope.get_handle().uuid == handles["outer"].uuid
        return ctx

    with _session(root):
        async with nemo_relay_scope(agent, "nooa") as parent:
            relay.scope_local.register_subscriber(
                parent, "nooa-local-test", lambda event: local_events.append(event.to_dict())
            )
            if nested_helper:
                async with nemo_relay_scope(agent, "nested"):
                    await _call(agent, "outer", outer)
                assert relay.scope.get_handle().uuid == parent.uuid
            else:
                await _call(agent, "outer", outer)
            await relay.subscribers.flush_async()
    events = await _function_events(relay_events)
    assert sorted((event["name"], event["scope_category"]) for event in events) == [
        ("Agent.inner", "end"),
        ("Agent.inner", "start"),
        ("Agent.outer", "end"),
        ("Agent.outer", "start"),
    ]
    assert {event["propagation_root_uuid"] for event in events} == {root}
    assert str(handles["inner"].parent_uuid) == str(handles["outer"].uuid)
    assert {event["name"] for event in local_events if event["category"] == "function"} == {
        "Agent.outer",
        "Agent.inner",
    }


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_failed_call_restores_ambient_stack(relay_events, failure):
    agent = Agent(llm=MagicMock())
    root = str(uuid4())

    async def fail(ctx):
        raise failure("interrupted")

    async def succeed(ctx):
        ctx.result = 42
        return ctx

    with _session(root) as request, _installed(agent):
        with pytest.raises(failure, match="interrupted"):
            await _call(agent, "failed", fail)
        assert relay.scope.get_handle().uuid == request.uuid
        assert relay.capture_propagation_context().root_uuid == root
        assert (await _call(agent, "after", succeed)).result == 42
        assert relay.scope.get_handle().uuid == request.uuid
    events = await _function_events(relay_events)
    assert sorted((event["name"], event["scope_category"]) for event in events) == [
        ("Agent.after", "end"),
        ("Agent.after", "start"),
        ("Agent.failed", "end"),
        ("Agent.failed", "start"),
    ]
    assert {event["propagation_root_uuid"] for event in events} == {root}
    assert {event["parent_uuid"] for event in events} == {str(request.uuid)}


@pytest.mark.parametrize("failure_point", ["push", "capture"])
async def test_setup_failure_restores_ambient_stack(relay_events, monkeypatch, failure_point):
    agent = Agent(llm=MagicMock())
    root = str(uuid4())
    push = relay.scope.push
    capture = relay.capture_propagation_context

    def fail_push(name, *args, **kwargs):
        if name == "Agent.failed":
            raise RuntimeError("setup failed")
        return push(name, *args, **kwargs)

    def fail_capture():
        if relay.scope.get_handle().name == "request":
            raise RuntimeError("setup failed")
        return capture()

    async def body(ctx):
        ctx.result = 42
        return ctx

    with _session(root) as request, _installed(agent):
        with monkeypatch.context() as patch:
            if failure_point == "push":
                patch.setattr(relay.scope, "push", fail_push)
            else:
                patch.setattr(relay, "capture_propagation_context", fail_capture)
            with pytest.raises(RuntimeError, match="setup failed"):
                await _call(agent, "failed", body)
        assert relay.scope.get_handle().uuid == request.uuid
        assert capture().root_uuid == root
        assert (await _call(agent, "after", body)).result == 42
    events = await _function_events(relay_events)
    assert sorted(
        event["scope_category"] for event in events if event["name"] == "Agent.after"
    ) == ["end", "start"]
    assert {event["propagation_root_uuid"] for event in events} == {root}
    assert {event["parent_uuid"] for event in events} == {str(request.uuid)}


async def test_distinct_concurrent_sessions_do_not_share_roots(relay_events):
    agent = Agent(llm=MagicMock())
    roots = [str(uuid4()), str(uuid4())]
    ready = [asyncio.Event(), asyncio.Event()]
    parents = {}

    async def session(index):
        # Each request owns its stack, as in an incoming request boundary.
        with relay.use_scope_stack(relay.create_scope_stack()):
            with _session(roots[index]) as request:
                parents[index] = str(request.uuid)

                async def body(ctx):
                    ready[index].set()
                    await ready[1 - index].wait()
                    assert relay.capture_propagation_context().root_uuid == roots[index]
                    return ctx

                await _call(agent, f"session_{index}", body)
                assert relay.scope.get_handle().uuid == request.uuid
                assert relay.capture_propagation_context().root_uuid == roots[index]

    with _installed(agent):
        await asyncio.gather(session(0), session(1))
    events = await _function_events(relay_events)
    assert len(events) == 4
    for index, root in enumerate(roots):
        own = [event for event in events if event["name"] == f"Agent.session_{index}"]
        assert {event["scope_category"] for event in own} == {"start", "end"}
        assert {event["propagation_root_uuid"] for event in own} == {root}
        assert {event["parent_uuid"] for event in own} == {parents[index]}


async def test_external_cancellation_restores_child_and_parent_context(relay_events):
    agent = Agent(llm=MagicMock())
    root = str(uuid4())
    started = asyncio.Event()

    async def blocked(ctx):
        started.set()
        await asyncio.Future()

    async def succeed(ctx):
        return ctx

    with _session(root) as request, _installed(agent):

        async def child():
            try:
                await _call(agent, "cancelled", blocked)
            except asyncio.CancelledError:
                assert relay.scope.get_handle().uuid == request.uuid
                assert relay.capture_propagation_context().root_uuid == root
                await _call(agent, "recovered", succeed)
                raise

        task = asyncio.create_task(child())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert relay.scope.get_handle().uuid == request.uuid
        assert relay.capture_propagation_context().root_uuid == root
        await _call(agent, "parent_after", succeed)
    events = await _function_events(relay_events)
    assert len(events) == 6
    assert {event["propagation_root_uuid"] for event in events} == {root}
    assert {event["parent_uuid"] for event in events} == {str(request.uuid)}
    for name in ("cancelled", "recovered", "parent_after"):
        assert {
            event["scope_category"] for event in events if event["name"] == f"Agent.{name}"
        } == {"start", "end"}


@pytest.mark.parametrize("child_task", [False, True], ids=["same-task", "child-task"])
async def test_scope_local_delivery_at_task_boundary(relay_events, child_task):
    agent = Agent(llm=MagicMock())
    local_events = []
    root = str(uuid4())
    stacks = {}

    async def body(ctx):
        stacks["method"] = relay.get_scope_stack()
        return ctx

    with _session(root):
        async with nemo_relay_scope(agent, "boundary") as parent:
            stacks["parent"] = relay.get_scope_stack()
            relay.scope_local.register_subscriber(
                parent, "boundary-local", lambda event: local_events.append(event.to_dict())
            )
            if child_task:
                await asyncio.create_task(_call(agent, "boundary", body))
            else:
                await _call(agent, "boundary", body)
            assert relay.get_scope_stack() is stacks["parent"]
            await relay.subscribers.flush_async()
    events = await _function_events(relay_events)
    assert len(events) == 2  # Global exporters see both start and end in either case.
    assert {event["propagation_root_uuid"] for event in events} == {root}
    assert {event["parent_uuid"] for event in events} == {str(parent.uuid)}
    assert (stacks["method"] is stacks["parent"]) is not child_task
    local_functions = [event for event in local_events if event["category"] == "function"]
    assert len(local_functions) == (0 if child_task else 2)


async def test_nested_method_keeps_scope_local_tool_guardrail(relay_events):
    from nooa.nemo_relay_middleware import nemo_relay_tool_middleware
    from nooa.runtime.middleware import ExecutePythonContext

    agent = Agent(llm=MagicMock())
    checked = []

    def deny(name, args):
        checked.append(name)
        return "email blocked by request policy"

    async def execute(ctx):
        pytest.fail("The request-local policy must prevent execution")

    async def inner(ctx):
        await nemo_relay_tool_middleware(ExecutePythonContext(code="send_email()"), execute)
        return ctx

    async def outer(ctx):
        await _call(agent, "send_email", inner)
        return ctx

    async with nemo_relay_scope(agent, "support-request") as request:
        relay.scope_local.register_tool_conditional_execution(request, "no-email", 1, deny)
        with pytest.raises(Exception, match="email blocked"):
            await _call(agent, "resolve_ticket", outer)
        assert relay.scope.get_handle().uuid == request.uuid
    assert checked == ["execute_python"]


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_same_task_failure_keeps_parent_locals(relay_events, failure):
    agent = Agent(llm=MagicMock())
    local_events = []

    async def fail(ctx):
        raise failure("interrupted")

    async def succeed(ctx):
        return ctx

    async with nemo_relay_scope(agent, "request") as request:
        stack = relay.get_scope_stack()
        root = relay.capture_propagation_context().root_uuid
        relay.scope_local.register_subscriber(
            request, "recovery-local", lambda event: local_events.append(event.to_dict())
        )
        with pytest.raises(failure, match="interrupted"):
            await _call(agent, "failed", fail)
        assert relay.scope.get_handle().uuid == request.uuid
        assert relay.get_scope_stack() is stack
        assert relay.capture_propagation_context().root_uuid == root
        await _call(agent, "recovered", succeed)
        await relay.subscribers.flush_async()
    functions = [event for event in local_events if event["category"] == "function"]
    assert sorted((event["name"], event["scope_category"]) for event in functions) == [
        ("Agent.failed", "end"),
        ("Agent.failed", "start"),
        ("Agent.recovered", "end"),
        ("Agent.recovered", "start"),
    ]
    assert {event["parent_uuid"] for event in functions} == {str(request.uuid)}
    assert {event["propagation_root_uuid"] for event in functions} == {root}
