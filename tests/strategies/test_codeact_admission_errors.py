# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve admission failures at the model-generation boundary, not as API retries."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import litellm
import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.events import AfterTurn, PythonOutput
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    AdmissionControl,
    AdmissionControlConfig,
    AdmissionError,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    CompletionClient,
    FakeLLMClient,
    LLMResponse,
    ResponsesClient,
    RetryConfig,
    ToolCall,
)


class ApplicationRejection(AdmissionError):
    """An external controller's policy error must follow the same base-class contract."""


ERRORS = [
    AdmissionCallCapError,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    ApplicationRejection,
]


class RejectingController:
    """A real protocol implementation rejecting every acquisition with one policy error."""

    def __init__(self, error):
        """Retain the exception so tests can verify its identity at the Agent boundary."""
        self.error = error
        self.attempts = 0

    async def acquire(self, observer):
        """Count acquisitions without creating a permit or invoking the provider."""
        self.attempts += 1
        raise self.error


def _config(strategy_type, *, retries=3, execution="inprocess"):
    """Select a real strategy configuration without live-provider setup."""
    if strategy_type is PredictStrategy:
        return PredictConfig(max_retries=retries)
    return CodeActConfig(
        prefill=None,
        max_retries=retries,
        execution_backend=execution,
        sandbox=SandboxConfig(require=True, filesystem=False, network=False),
    )


def _reply(tool="execute_python", code="return_result(7)"):
    """Build a valid transport response rather than bypassing the production parser."""
    return litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": uuid4().hex,
                            "type": "function",
                            "function": {"name": tool, "arguments": json.dumps({"code": code})},
                        }
                    ],
                },
            }
        ],
    )


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2, PredictStrategy])
@pytest.mark.parametrize("error_type", ERRORS)
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("provider_retries", [0, 3])
@pytest.mark.parametrize("strategy_retries", [1, 3])
async def test_admission_failure_is_not_a_model_api_error(
    strategy_type, error_type, client_type, provider_retries, strategy_retries
):
    """Reject once, preserve the exact exception, and record no fictitious API failures."""
    failure = error_type("application policy rejected this attempt")
    controller = RejectingController(failure)
    llm = AdmissionControl(
        client_type(
            "openai/gpt-4o-mini",
            retry_config=RetryConfig(max_retries=provider_retries, rate_limit_extra_retries=0),
        ),
        AdmissionControlConfig(controller=controller),
    )

    class PolicyAgent(Agent):
        """Exercise the normal public generation method under application admission."""

        @strategy(strategy_type(config=_config(strategy_type, retries=strategy_retries)))
        async def answer(self) -> int:
            """Generate an integer only if the application admits the request."""
            ...

    agent = PolicyAgent(llm=llm)
    try:
        with (
            patch("litellm.acompletion", AsyncMock()) as chat,
            patch("litellm.aresponses", AsyncMock()) as responses,
            patch.object(agent.event_manager, "add", wraps=agent.event_manager.add) as add,
            patch.object(HarnessMetrics, "llm_api_error", autospec=True) as api_error,
        ):
            with pytest.raises(error_type) as caught:
                await agent.answer()
            assert caught.value is failure
            assert controller.attempts == 1
            chat.assert_not_awaited()
            responses.assert_not_awaited()
            api_error.assert_not_called()
            assert agent.events.query(type="Error") == []
            if strategy_type is not PredictStrategy:
                turns = [c.args[0] for c in add.call_args_list if isinstance(c.args[0], AfterTurn)]
                assert len(turns) == 1
                assert turns[0].is_final and not turns[0].success
                assert turns[0].exception_type == error_type.__name__
    finally:
        await agent.aclose()
        await llm.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize("error_type", ERRORS)
async def test_standalone_admission_failure_preserves_policy_error(strategy_type, error_type):
    """Standalone strategy stubs need no Agent-only attributes to propagate policy errors."""
    failure = error_type("standalone admission rejection")
    controller = RejectingController(failure)
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini", retry_config=RetryConfig(max_retries=3)),
        AdmissionControlConfig(controller=controller),
    )

    @strategy(strategy_type(config=_config(strategy_type)), llm=llm)
    async def answer() -> int:
        """Generate an integer only when application policy allows it."""
        ...

    try:
        with (
            patch("litellm.acompletion", AsyncMock()) as provider,
            patch.object(HarnessMetrics, "llm_api_error", autospec=True) as api_error,
        ):
            with pytest.raises(error_type) as caught:
                await answer()
            assert caught.value is failure
            assert controller.attempts == 1
            provider.assert_not_awaited()
            api_error.assert_not_called()
    finally:
        await llm.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize("kind", ["timeout", "cap", "unavailable"])
async def test_real_controller_failures_escape_without_strategy_retries(strategy_type, kind):
    """Use actual queue deadlines, a committed broker cap and a closed controller."""
    broker = AdmissionBroker(max_in_flight=1, max_calls=1).start() if kind != "timeout" else None
    controller = broker.controller() if broker else None
    held = None
    if kind == "cap":
        (await controller.acquire(lambda _: None)).release()
        expected = AdmissionCallCapError
    elif kind == "unavailable":
        controller.close()
        expected = AdmissionUnavailableError
    else:
        expected = AdmissionTimeoutError
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini", retry_config=RetryConfig(max_retries=3)),
        AdmissionControlConfig(controller=controller)
        if controller
        else AdmissionControlConfig(
            max_in_flight=1, concurrency_group=str(uuid4()), queue_timeout=0.02
        ),
    )
    if kind == "timeout":
        held = await llm.controller.acquire(lambda _: None)

    class PolicyAgent(Agent):
        """Run the actual admission owner through the normal strategy entrypoint."""

        @strategy(strategy_type(config=_config(strategy_type)))
        async def answer(self) -> int:
            """Generate an integer if capacity and run policy allow it."""
            ...

    agent = PolicyAgent(llm=llm)
    try:
        with (
            patch("litellm.acompletion", AsyncMock()) as provider,
            patch.object(
                llm.controller, "acquire", AsyncMock(wraps=llm.controller.acquire)
            ) as acquire,
        ):
            with pytest.raises(expected):
                await agent.answer()
            assert acquire.await_count == 1
            provider.assert_not_awaited()
            assert agent.events.query(type="Error") == []
        if held:
            held.release()
            held = None
            with patch(
                "litellm.acompletion",
                AsyncMock(
                    return_value=_reply(
                        "python_cell" if strategy_type is CodeActV2 else "execute_python"
                    )
                ),
            ):
                assert await agent.answer() == 7
    finally:
        if held:
            held.release()
        await agent.aclose()
        await llm.aclose()
        if controller:
            controller.close()
        if broker:
            broker.close()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize(
    "execution", ["inprocess", pytest.param("sandbox", marks=pytest.mark.sandbox)]
)
async def test_exhausted_cap_after_executed_cell_preserves_output_and_final_turn(
    strategy_type, execution
):
    """A later model rejection stays terminal after a real in-process or guarded cell."""
    if execution == "sandbox" and sys.platform != "linux":
        pytest.skip("Guarded worker validation requires Linux")
    broker = AdmissionBroker(max_in_flight=1, max_calls=1).start()
    controller = broker.controller()
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini", retry_config=RetryConfig(max_retries=3)),
        AdmissionControlConfig(controller=controller),
    )

    class PolicyAgent(Agent):
        """Use the first allowed attempt for a cell, then exhaust the shared cap."""

        def note_pid(self, pid: int) -> None:
            """Record where the first cell actually executed."""
            self._cell_pid = pid

        @strategy(strategy_type(config=_config(strategy_type, execution=execution)))
        async def answer(self) -> int:
            """Run a cell before asking for the final model result."""
            ...

    agent = PolicyAgent(llm=llm)
    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    try:
        with (
            patch(
                "litellm.acompletion",
                AsyncMock(
                    return_value=_reply(
                        tool, "self.note_pid(os.getpid())\nprint('completed first cell')"
                    )
                ),
            ) as provider,
            patch.object(controller, "acquire", AsyncMock(wraps=controller.acquire)) as acquire,
            patch.object(agent.event_manager, "add", wraps=agent.event_manager.add) as add,
        ):
            with pytest.raises(AdmissionCallCapError):
                await agent.answer()
            assert provider.await_count == 1
            assert acquire.await_count == 2  # One admitted attempt, one terminal rejection.
            assert (agent._cell_pid != os.getpid()) == (execution == "sandbox")
            outputs = agent.events.query(type="PythonOutput")
            assert any(
                isinstance(e, PythonOutput) and "completed first cell" in e.stdout for e in outputs
            )
            assert agent.events.query(type="Error") == []
            turns = [c.args[0] for c in add.call_args_list if isinstance(c.args[0], AfterTurn)]
            assert len(turns) == 2
            assert not turns[0].is_final
            assert turns[-1].is_final and not turns[-1].success
            assert turns[-1].exception_type == "AdmissionCallCapError"
    finally:
        await agent.aclose()
        await llm.aclose()
        controller.close()
        broker.close()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
@pytest.mark.parametrize("error_type", ERRORS[:3])
@pytest.mark.parametrize("entry", ["cell", "prefill"])
@pytest.mark.parametrize(
    "execution", ["inprocess", pytest.param("sandbox", marks=pytest.mark.sandbox)]
)
async def test_parent_can_handle_child_policy_failure_in_cells_and_prefill(
    strategy_type, error_type, entry, execution
):
    """Application-side catches work for nested generation without changing RPC error policy."""
    if execution == "sandbox" and sys.platform != "linux":
        pytest.skip("Guarded worker validation requires Linux")
    failure = error_type("child policy rejected this attempt")
    controller = RejectingController(failure)
    inner_llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini", retry_config=RetryConfig(max_retries=3)),
        AdmissionControlConfig(controller=controller),
    )

    class Child(Agent):
        """Expose a real generation method to the parent's deterministic orchestrator."""

        @strategy(strategy_type(config=_config(strategy_type)))
        async def answer(self) -> int:
            """Generate an integer only when admitted."""
            ...

    child = Child(llm=inner_llm)

    class Parent(Agent):
        """Handle a child policy rejection in application code, including sandbox RPCs."""

        def note_pid(self, pid: int) -> None:
            """Prove which process executes the parent's cell."""
            self._cell_pid = pid

        async def handle_child_failure(self) -> int:
            """Choose an explicit application fallback instead of hiding the policy error."""
            try:
                return await child.answer()
            except AdmissionError as error:
                self._handled = error
                return 7

        @strategy(strategy_type(config=_config(strategy_type, execution=execution)))
        async def cell(self) -> int:
            """Execute the scripted cell which invokes deterministic error handling."""
            ...

        @strategy(strategy_type(config=_config(strategy_type, execution=execution)))
        async def prefill(self) -> int:
            """Handle the nested rejection before the parent's first model request."""
            self.note_pid(os.getpid())
            value = await self.handle_child_failure()  # noqa: F841 -- Used by the scripted cell.
            ...

    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    code = (
        "self.note_pid(os.getpid())\nreturn_result(await self.handle_child_failure())"
        if entry == "cell"
        else "return_result(value)"
    )
    outer_llm = FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="finish", name=tool, arguments=json.dumps({"code": code}))],
            )
        ],
        strict_exhaustion=True,
    )
    parent = Parent(llm=outer_llm)
    try:
        with patch("litellm.acompletion", AsyncMock()) as provider:
            assert await getattr(parent, entry)() == 7
            provider.assert_not_awaited()
        assert parent._handled is failure
        assert (parent._cell_pid != os.getpid()) == (execution == "sandbox")
        assert controller.attempts == 1
        assert len(outer_llm.calls) == 1
        assert child.events.query(type="Error") == []
    finally:
        await parent.aclose()
        await child.aclose()
        await inner_llm.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
async def test_cancelled_admission_wait_stays_cancelled_and_owner_recovers(strategy_type):
    """Cancellation does not become a policy/API error or strand queued admission."""
    broker = AdmissionBroker(max_in_flight=1).start()
    controller = broker.controller()
    held = await controller.acquire(lambda _: None)
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini"), AdmissionControlConfig(controller=controller)
    )

    class PolicyAgent(Agent):
        """Wait under a real shared broker and retain normal task cancellation."""

        @strategy(strategy_type(config=_config(strategy_type)))
        async def answer(self) -> int:
            """Generate an integer after capacity opens."""
            ...

    agent = PolicyAgent(llm=llm)
    task = None
    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    try:
        with patch("litellm.acompletion", AsyncMock(return_value=_reply(tool))) as provider:
            task = asyncio.create_task(agent.answer())
            async with asyncio.timeout(5):
                while broker.snapshot().queued != 1:
                    await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            provider.assert_not_awaited()
            held.release()
            assert await agent.answer() == 7
        assert agent.events.query(type="Error") == []
        async with asyncio.timeout(5):
            while broker.snapshot().active or broker.snapshot().queued:
                await asyncio.sleep(0)
    finally:
        held.release()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()
        await llm.aclose()
        controller.close()
        broker.close()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, CodeActV2])
async def test_ordinary_provider_error_still_retries_and_recovers(strategy_type):
    """Do not globally disable CodeAct's existing recovery for ordinary API failures."""
    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    response = LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id="done", name=tool, arguments=json.dumps({"code": "return_result(7)"}))
        ],
        finish_reason="tool_calls",
    )
    llm = FakeLLMClient()

    class RecoveringAgent(Agent):
        """Keep the existing recoverable API-failure path intact."""

        @strategy(strategy_type(config=_config(strategy_type)))
        async def answer(self) -> int:
            """Return an integer after the temporary provider failure."""
            ...

    agent = RecoveringAgent(llm=llm)
    try:
        with patch.object(
            llm,
            "acall",
            AsyncMock(
                side_effect=[
                    litellm.InternalServerError(
                        message="transient provider failure",
                        model="gpt-4o-mini",
                        llm_provider="openai",
                    ),
                    response,
                ]
            ),
        ) as provider:
            assert await agent.answer() == 7
            assert provider.await_count == 2
        assert len(agent.events.query(type="Error")) == 1
    finally:
        await agent.aclose()
