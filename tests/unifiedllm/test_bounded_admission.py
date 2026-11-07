# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded FIFO admission, cross-loop contention, and application integration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import litellm
import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.context_blocks import ToolCallEvent
from nooa.events import AfterTurn, PythonOutput, ResultStatus
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionControl,
    AdmissionControlConfig,
    CompletionClient,
    ResponsesClient,
    RetryConfig,
)
from nooa.unifiedllm.admission import (
    AdmissionPolicy,
    AdmissionQueueFullError,
    AdmissionTimeoutError,
)


def policy(cap=1, pending=2, **kwargs):
    """Create a unique group so test policies never leak into one another."""
    return AdmissionPolicy(
        max_in_flight=cap,
        max_pending=pending,
        concurrency_group=str(uuid4()),
        queue_timeout=kwargs.pop("queue_timeout", None),
        api_base=None,
        **kwargs,
    )


def chat(code=None, tool="execute_python"):
    """Return an offline provider response, optionally with a submission cell."""
    message = {"role": "assistant", "content": "ok"}
    finish = "stop"
    if code:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "cell",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps({"code": code})},
            }
        ]
        finish = "tool_calls"
    return litellm.ModelResponse(
        model="test-model", choices=[{"index": 0, "finish_reason": finish, "message": message}]
    )


def response():
    """Return the minimum Responses payload consumed by the production parser."""
    r = MagicMock()
    r.output = [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
    ]
    r.output_text, r.usage = "ok", None
    return r


async def tick_until(predicate):
    """Wait cooperatively for a causal state transition."""
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.parametrize("pending", [None, 0, 1, 16])
@pytest.mark.parametrize("cap", [1, 4])
@pytest.mark.asyncio
async def test_actual_fifo_retains_at_most_pending(cap, pending):
    """The process-wide bound excludes active calls and preserves FIFO order."""
    p = policy(cap, pending)
    first = [await p.acquire(lambda _: None) for _ in range(cap)]
    observed = []
    tasks = [asyncio.create_task(p.acquire(observed.append)) for _ in range(100)]
    await asyncio.sleep(0)
    keep = 100 if pending is None else pending
    assert sum(t.done() for t in tasks) == 100 - keep
    for permit in first:
        permit.release()
    results = []
    for task in tasks:
        try:
            permit = await task
        except AdmissionQueueFullError:
            results.append("full")
        else:
            permit.release()
            results.append("ok")
    assert results == ["ok"] * keep + ["full"] * (100 - keep)
    assert sum(o["outcome"] == "queue_full" for o in observed) == 100 - keep
    permit = await p.acquire(lambda _: None)
    permit.release()


@pytest.mark.parametrize("transport", ["chat", "responses"])
@pytest.mark.parametrize("retries", [0, 3])
@pytest.mark.asyncio
async def test_reject_never_reaches_provider_or_retries(transport, retries):
    """An overflow rejection never dispatches, including with retries enabled."""
    cls, target, result = (
        (CompletionClient, "litellm.acompletion", chat())
        if transport == "chat"
        else (ResponsesClient, "litellm.aresponses", response())
    )
    base = cls(
        "test-model", retry_config=RetryConfig(max_retries=retries, rate_limit_extra_retries=0)
    )
    wrapper = AdmissionControl(
        base, AdmissionControlConfig(max_in_flight=1, max_pending=0, concurrency_group=str(uuid4()))
    )
    permit = await wrapper.controller.acquire(lambda _: None)
    try:
        with patch(target, AsyncMock(return_value=result)) as mock:
            with pytest.raises(AdmissionQueueFullError):
                await wrapper.acall([{"role": "user", "content": "synthetic"}])
            assert mock.await_count == 0
            permit.release()
            assert (await wrapper.acall([{"role": "user", "content": "synthetic"}])).content == "ok"
            assert mock.await_count == 1
    finally:
        permit.release()
        await wrapper.aclose()


@pytest.mark.asyncio
async def test_same_group_two_clients_share_pending_bound():
    """Wrapper instances share queue positions rather than getting private limits."""
    name = str(uuid4())
    opts = {"max_in_flight": 1, "max_pending": 1, "concurrency_group": name}
    a = AdmissionControl(CompletionClient("test-model"), AdmissionControlConfig(**opts))
    b = AdmissionControl(CompletionClient("test-model"), AdmissionControlConfig(**opts))
    first = await a.controller.acquire(lambda _: None)
    waiter = asyncio.create_task(b.controller.acquire(lambda _: None))
    await asyncio.sleep(0)
    with pytest.raises(AdmissionQueueFullError):
        await a.controller.acquire(lambda _: None)
    first.release()
    (await waiter).release()
    await a.aclose()
    await b.aclose()


@pytest.mark.parametrize("cause", ["cancel", "timeout"])
@pytest.mark.asyncio
async def test_pending_slot_returns_on_abandonment(cause):
    """Cancellation and expiry make a queue position available immediately."""
    p = policy(pending=1, queue_timeout=0.02 if cause == "timeout" else None)
    first = await p.acquire(lambda _: None)
    task = asyncio.create_task(p.acquire(lambda _: None))
    await asyncio.sleep(0)
    if cause == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(AdmissionTimeoutError):
            await task
    replacement = asyncio.create_task(p.acquire(lambda _: None))
    await asyncio.sleep(0)
    assert not replacement.done()
    first.release()
    (await replacement).release()


@pytest.mark.parametrize("gate", ["pacing", "cooldown"])
@pytest.mark.asyncio
async def test_time_delayed_call_counts_as_pending(gate):
    """Pacing and cooldown waits obey the same pending bound as concurrency waits."""
    p = policy(
        pending=1,
        requests_per_second=0.01 if gate == "pacing" else None,
        max_cooldown=100 if gate == "cooldown" else None,
    )
    permit = await p.acquire(lambda _: None)
    if gate == "cooldown":
        await permit.cooldown(100)
    permit.release()
    waiter = asyncio.create_task(p.acquire(lambda _: None))
    await asyncio.sleep(0)
    try:
        with pytest.raises(AdmissionQueueFullError):
            await p.acquire(lambda _: None)
    finally:
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter


@pytest.mark.asyncio
async def test_grant_cancel_race_repeated_500_times():
    """Losing a grant/cancel race leaks neither active nor pending capacity."""
    p = policy(pending=1)
    for _ in range(500):
        first = await p.acquire(lambda _: None)
        waiter = asyncio.create_task(p.acquire(lambda _: None))
        await asyncio.sleep(0)
        first.release()
        waiter.cancel()
        try:
            (await waiter).release()
        except asyncio.CancelledError:
            pass
        check = await asyncio.wait_for(p.acquire(lambda _: None), 0.2)
        check.release()


@pytest.mark.asyncio
async def test_10000_attempt_saturation_recovery():
    """Repeated saturation rejects exactly the excess and recovers every wave."""
    p = policy(cap=4, pending=16)
    counts = {"full": 0, "ok": 0}
    for _ in range(50):
        first = [await p.acquire(lambda _: None) for _ in range(4)]
        tasks = [asyncio.create_task(p.acquire(lambda _: None)) for _ in range(200)]
        await asyncio.sleep(0)
        for permit in first:
            permit.release()
        for task in tasks:
            try:
                permit = await task
            except AdmissionQueueFullError:
                counts["full"] += 1
            else:
                counts["ok"] += 1
                permit.release()
    assert counts == {"ok": 800, "full": 9200}


def test_two_owner_loops_share_the_same_bound():
    """Concurrent owner threads cannot each allocate a private queue allowance."""
    p = policy(cap=2, pending=8)
    gate = threading.Barrier(2)
    counts = []

    def worker():
        """Run one independent event loop behind a shared enqueue barrier."""

        async def run():
            """Hold a slot until both loops have enqueued all contenders."""
            first = await p.acquire(lambda _: None)
            gate.wait(timeout=5)
            tasks = [asyncio.create_task(p.acquire(lambda _: None)) for _ in range(20)]
            await asyncio.sleep(0)
            gate.wait(timeout=5)
            first.release()
            results = []
            for task in tasks:
                try:
                    permit = await task
                except AdmissionQueueFullError:
                    results.append("full")
                else:
                    permit.release()
                    results.append("ok")
            return results

        return asyncio.run(run(), debug=True)

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        counts = [r for f in futures for r in f.result(timeout=10)]
    assert counts.count("ok") == 8
    assert counts.count("full") == 32


@pytest.mark.parametrize("bad", [True, 1.5, "2", -1])
def test_validate_pending(bad):
    """Queue bounds are integers, not booleans or implicitly coerced numbers."""
    with pytest.raises((TypeError, ValueError)):
        AdmissionControlConfig(max_in_flight=1, concurrency_group=str(uuid4()), max_pending=bad)
    with pytest.raises((TypeError, ValueError)):
        policy(pending=bad)


def test_group_policy_conflicts_fail_eagerly():
    """All wrappers joining a group must agree on its pending limit."""
    name = str(uuid4())
    AdmissionControl(
        CompletionClient("test-model"),
        AdmissionControlConfig(max_in_flight=1, max_pending=1, concurrency_group=name),
    )
    with pytest.raises(ValueError, match="max_pending"):
        AdmissionControl(
            CompletionClient("test-model"),
            AdmissionControlConfig(max_in_flight=1, max_pending=2, concurrency_group=name),
        )


class ServiceAgent(Agent):
    """Exercise actual CodeAct submission after an overload clears."""

    @strategy(CodeActStrategy(config=CodeActConfig(prefill=None)))
    async def _answer(self) -> int:
        """Return the verified result of this synthetic service request."""
        ...


@pytest.mark.asyncio
async def test_real_agents_apply_overload_policy():
    """A rejected agent and another shared client recover when capacity returns."""
    base = CompletionClient(
        "test-model", retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0)
    )
    wrapped = AdmissionControl(
        base, AdmissionControlConfig(max_in_flight=1, max_pending=0, concurrency_group=str(uuid4()))
    )
    agents = [ServiceAgent(llm=wrapped) for _ in range(2)]
    held = await wrapped.controller.acquire(lambda _: None)
    try:
        with patch("litellm.acompletion", AsyncMock(return_value=chat("return_result(7)"))) as mock:
            with pytest.raises(AdmissionQueueFullError):
                await agents[0]._answer()
            assert mock.await_count == 0
            held.release()
            assert await agents[0]._answer() == 7
            assert await agents[1]._answer() == 7
            assert mock.await_count == 2
    finally:
        held.release()
        for agent in agents:
            await agent.aclose()
        await wrapped.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2, PredictStrategy])
@pytest.mark.parametrize("backend", ["local", "broker"])
@pytest.mark.asyncio
async def test_agent_queue_full_is_terminal(strategy_type, backend):
    """Strategies preserve the typed policy error and acquire only once."""
    broker = (
        AdmissionBroker(max_in_flight=1, max_pending=0).start() if backend == "broker" else None
    )
    controller = broker.controller() if broker else None
    settings = (
        AdmissionControlConfig(controller=controller)
        if controller
        else AdmissionControlConfig(max_in_flight=1, max_pending=0, concurrency_group=str(uuid4()))
    )
    llm = AdmissionControl(CompletionClient("test-model"), settings)
    config = (
        PredictConfig(max_retries=3)
        if strategy_type is PredictStrategy
        else CodeActConfig(prefill=None, max_retries=3)
    )

    class RejectedAgent(Agent):
        """One generation task under an intentionally saturated admission group."""

        @strategy(strategy_type(config=config))
        async def _answer(self) -> int:
            """Return an integer if admission is granted."""
            ...

    agent = RejectedAgent(llm=llm)
    held = await llm.controller.acquire(lambda _: None)
    try:
        with patch("litellm.acompletion", AsyncMock()) as provider:
            with patch.object(
                llm.controller, "acquire", AsyncMock(wraps=llm.controller.acquire)
            ) as acquire:
                with pytest.raises(AdmissionQueueFullError):
                    await agent._answer()
                assert acquire.await_count == 1
                assert provider.await_count == 0
                assert agent.events.query(type="Error") == []
    finally:
        held.release()
        await agent.aclose()
        await llm.aclose()
        if controller:
            controller.close()
        if broker:
            broker.close()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize("nested_strategy_type", [CodeActStrategy, PredictStrategy])
@pytest.mark.parametrize("backend", ["local", "broker"])
@pytest.mark.parametrize("entry", ["cell", "pre_ellipsis", "prefill_plugin"])
@pytest.mark.parametrize("failure", ["queue_full", "ordinary"])
@pytest.mark.parametrize(
    "execution",
    ["inprocess", pytest.param("sandbox", marks=pytest.mark.sandbox)],
)
@pytest.mark.asyncio
async def test_nested_queue_full_is_terminal_but_cell_errors_recover(
    strategy_type, nested_strategy_type, backend, entry, failure, execution
):
    """Nested overload escapes cells and prefills without changing normal error recovery."""
    if execution == "sandbox" and sys.platform != "linux":
        pytest.skip("Guarded worker validation requires Linux")
    broker = (
        AdmissionBroker(max_in_flight=1, max_pending=0).start() if backend == "broker" else None
    )
    controller = broker.controller() if broker else None
    settings = (
        AdmissionControlConfig(controller=controller)
        if controller
        else AdmissionControlConfig(max_in_flight=1, max_pending=0, concurrency_group=str(uuid4()))
    )
    llm = AdmissionControl(
        CompletionClient(
            "test-model", retry_config=RetryConfig(max_retries=3, rate_limit_extra_retries=0)
        ),
        settings,
    )
    nested_config = (
        PredictConfig(max_retries=3)
        if nested_strategy_type is PredictStrategy
        else CodeActConfig(prefill=None, max_retries=3)
    )

    class NestedPrefill:
        """Run a nested generation from the configured prefill plugin."""

        def get_code(self, call, config=None):
            """Use the same deterministic nested-call setup as a model-written cell."""
            return "self.note_pid(os.getpid())\nprint('before nested call')\nawait self.nested()"

    def outer_config(prefill=None):
        """Select the execution backend, requiring real network guards for workers."""
        return CodeActConfig(
            prefill=prefill,
            max_retries=3,
            execution_backend=execution,
            sandbox=SandboxConfig(require=True, filesystem=False, network=False),
        )

    class NestedAgent(Agent):
        """Exercise nested admission from real Python execution and parent RPC."""

        def __init__(self, **kwargs):
            """Track the process that actually runs the nested-call cell."""
            super().__init__(**kwargs)
            self._pids = []

        def note_pid(self, pid: int) -> None:
            """Record worker-side execution without relying on configuration alone."""
            self._pids.append(pid)

        @strategy(nested_strategy_type(config=nested_config))
        async def _inner(self) -> int:
            """Generate an integer if admission is granted."""
            ...

        async def nested(self) -> int:
            """Saturate admission only while the nested generation attempts to enter."""
            if failure == "ordinary":
                raise ValueError("ordinary cell failure")
            held = await llm.controller.acquire(lambda _: None)
            try:
                return await self._inner()
            except AdmissionQueueFullError as error:
                self._rejection = error
                raise
            finally:
                held.release()

        @strategy(strategy_type(config=outer_config()))
        async def _cell(self) -> int:
            """Execute a model-written cell, then submit an integer."""
            ...

        @strategy(strategy_type(config=outer_config()))
        async def _pre_ellipsis(self) -> int:
            """Run nested generation in pre-ellipsis setup."""
            self.note_pid(os.getpid())
            print("before nested call")
            await self.nested()
            ...

        @strategy(strategy_type(config=outer_config(NestedPrefill())))
        async def _prefill_plugin(self) -> int:
            """Run nested generation through the configured prefill plugin."""
            ...

    agent = NestedAgent(llm=llm)
    recovery = ServiceAgent(llm=llm)
    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    responses = (
        [
            chat(
                "self.note_pid(os.getpid())\nprint('before nested call')\nawait self.nested()",
                tool=tool,
            )
        ]
        if entry == "cell"
        else []
    ) + [chat("return_result(7)", tool=tool)]
    try:
        with patch("litellm.acompletion", AsyncMock(side_effect=responses)) as provider:
            with patch.object(agent.event_manager, "add", wraps=agent.event_manager.add) as add:
                if failure == "queue_full":
                    with pytest.raises(AdmissionQueueFullError) as caught:
                        await getattr(agent, f"_{entry}")()
                    if execution == "inprocess":
                        assert caught.value is agent._rejection
                    else:
                        assert str(caught.value) == str(agent._rejection)
                    assert provider.await_count == (1 if entry == "cell" else 0)
                    assert agent.events.query(type="Error") == []
                    turns = [
                        item.args[0]
                        for item in add.call_args_list
                        if isinstance(item.args[0], AfterTurn)
                    ]
                    assert turns
                    assert all(turn.is_final and not turn.success for turn in turns)
                    assert all(turn.exception_type == "AdmissionQueueFullError" for turn in turns)
                else:
                    assert await getattr(agent, f"_{entry}")() == 7
                    assert provider.await_count == (2 if entry == "cell" else 1)

        outputs = agent.events.query(type="PythonOutput")
        assert agent._pids
        assert all((pid != os.getpid()) == (execution == "sandbox") for pid in agent._pids)
        failed = [output for output in outputs if output.execution_status is ResultStatus.ERROR]
        assert len(failed) == 1
        assert isinstance(failed[0], PythonOutput)
        assert "before nested call" in failed[0].stdout
        expected_error = "AdmissionQueueFullError" if failure == "queue_full" else "ValueError"
        assert expected_error in failed[0].error
        call = next(
            event
            for event in agent.events.query(type="ToolCallEvent")
            if event.tool_call_id == failed[0].tool_call_id
        )
        assert isinstance(call, ToolCallEvent)
        assert call.result is not None
        assert call.result.result_status is ResultStatus.ERROR

        # Use the same owner and client after unwinding; no restart or policy reset.
        with patch("litellm.acompletion", AsyncMock(return_value=chat("return_result(7)"))):
            assert await recovery._answer() == 7
    finally:
        await agent.aclose()
        await recovery.aclose()
        await llm.aclose()
        if controller:
            controller.close()
        if broker:
            broker.close()


def test_external_controller_owns_its_queue_policy():
    """Built-in settings cannot silently override an application controller."""
    with pytest.raises(ValueError, match="controller cannot be combined"):
        AdmissionControlConfig(controller=policy(), max_pending=0)


@pytest.mark.asyncio
async def test_local_queue_full_observer_cannot_replace_policy_error():
    """A failing observer neither hides overload nor allocates a pending waiter."""
    p = policy(pending=0)
    first = await p.acquire(lambda _: None)

    def broken(_detail):
        """Simulate a failing telemetry sink."""
        raise RuntimeError("observer failure")

    try:
        with pytest.raises(AdmissionQueueFullError):
            await p.acquire(broken)
    finally:
        first.release()
    (await p.acquire(lambda _: None)).release()
