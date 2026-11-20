# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal strategy failures escape unchanged, without policy or provider retries."""

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
from nooa.config import CodeActConfig
from nooa.context_blocks.events import ResultStatus
from nooa.events import AfterTurn, PythonOutput
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import (
    TERMINAL_LLM_ERRORS,
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
    ReasoningReplayError,
    RetryConfig,
    ToolCall,
)
from nooa.unifiedllm.errors import TERMINAL_LLM_ERRORS as SHARED_TERMINAL_ERRORS

STRATEGIES = [CodeActStrategy, CodeActV2, PurePythonStrategy]
ADMISSION_ERRORS = [AdmissionCallCapError, AdmissionTimeoutError, AdmissionUnavailableError]


def test_terminal_rule_is_reexported_and_includes_application_subclasses():
    """One public tuple retains the two error families and custom admission policies."""

    class ApplicationRejection(AdmissionError):
        """An external policy may specialize the public admission error contract."""

    assert TERMINAL_LLM_ERRORS is SHARED_TERMINAL_ERRORS
    assert TERMINAL_LLM_ERRORS == (ReasoningReplayError, AdmissionError)
    assert isinstance(ApplicationRejection("application policy"), TERMINAL_LLM_ERRORS)


class RejectingController:
    """A concrete protocol implementation rejecting acquisition with one policy error."""

    def __init__(self, error):
        """Retain the exception so tests can verify identity at the Agent boundary."""
        self.error = error
        self.attempts = 0

    async def acquire(self, observer):
        """Count acquisitions without creating a permit or invoking the provider."""
        self.attempts += 1
        raise self.error


def _strategy(strategy_type, *, execution="inprocess", prefill=None):
    """Use one retry budget and vary only strategy behavior relevant to the contract."""
    if strategy_type is PurePythonStrategy:
        return PurePythonStrategy(max_retries=3, prefill=prefill)
    return strategy_type(
        config=CodeActConfig(
            prefill=prefill,
            max_retries=3,
            execution_backend=execution,
            sandbox=SandboxConfig(require=True, filesystem=False, network=False),
        )
    )


def _response(strategy_type, code=None):
    """Build a valid successful response for either strategy's production parser."""
    if strategy_type is PurePythonStrategy:
        return LLMResponse(
            raw_response=None, content=code or "return 7", finish_reason="stop", tool_calls=[]
        )
    tool = "python_cell" if strategy_type is CodeActV2 else "execute_python"
    return LLMResponse(
        raw_response=None,
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="finish", name=tool, arguments=json.dumps({"code": code or "return_result(7)"})
            )
        ],
    )


def _reply(strategy_type, code=None):
    """Return a transport response rather than bypassing the completion parser."""
    response = _response(strategy_type, code)
    return litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[
            {
                "index": 0,
                "finish_reason": response.finish_reason,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tool.id,
                            "type": "function",
                            "function": {"name": tool.name, "arguments": tool.arguments},
                        }
                        for tool in response.tool_calls
                    ]
                    or None,
                },
            }
        ],
    )


@pytest.mark.parametrize("strategy_type", STRATEGIES)
@pytest.mark.parametrize("error_type", ADMISSION_ERRORS)
async def test_admission_failure_is_not_a_model_api_error(strategy_type, error_type):
    """One acquisition preserves the exception and records no fictitious API failure."""
    failure = error_type("application policy rejected this attempt")
    controller = RejectingController(failure)
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini", retry_config=RetryConfig(max_retries=3)),
        AdmissionControlConfig(controller=controller),
    )

    class PolicyAgent(Agent):
        """Exercise the public generation method under application admission."""

        @strategy(_strategy(strategy_type))
        async def answer(self) -> int:
            """Generate an integer only if the application admits the request."""
            ...

    agent = PolicyAgent(llm=llm)
    try:
        with (
            patch("litellm.acompletion", AsyncMock()) as provider,
            patch.object(agent.event_manager, "add", wraps=agent.event_manager.add) as add,
            patch.object(HarnessMetrics, "llm_api_error", autospec=True) as api_error,
        ):
            with pytest.raises(error_type) as caught:
                await agent.answer()
            assert caught.value is failure
            assert controller.attempts == 1
            provider.assert_not_awaited()
            api_error.assert_not_called()
            assert agent.events.query(type="Error") == []
            turns = [c.args[0] for c in add.call_args_list if isinstance(c.args[0], AfterTurn)]
            assert len(turns) == 1
            assert turns[0].is_final and not turns[0].success
            assert turns[0].exception_type == error_type.__name__
    finally:
        await agent.aclose()
        await llm.aclose()


@pytest.mark.parametrize("strategy_type", STRATEGIES)
async def test_standalone_admission_failure_preserves_policy_error(strategy_type):
    """Standalone strategy stubs need no Agent-only attributes to propagate policy errors."""
    failure = AdmissionUnavailableError("standalone admission rejection")
    controller = RejectingController(failure)
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini"), AdmissionControlConfig(controller=controller)
    )

    @strategy(_strategy(strategy_type), llm=llm)
    async def answer() -> int:
        """Generate an integer only when application policy allows it."""
        ...

    try:
        with patch("litellm.acompletion", AsyncMock()) as provider:
            with pytest.raises(AdmissionUnavailableError) as caught:
                await answer()
            assert caught.value is failure
            assert controller.attempts == 1
            provider.assert_not_awaited()
    finally:
        await llm.aclose()


@pytest.mark.parametrize("strategy_type", STRATEGIES)
@pytest.mark.parametrize("entry", ["plugin", "execution"])
@pytest.mark.parametrize("error_type", [AdmissionCallCapError, ReasoningReplayError, ValueError])
async def test_prefill_uses_the_same_terminal_rule(strategy_type, entry, error_type):
    """Both raised and executed prefill errors stop only for terminal failure types."""
    failure = error_type("prefill failure")

    class FailingPrefill:
        """Fail in the plugin itself or execute real prefill code that fails."""

        def get_code(self, call, *, config):
            """Cover both the outer exception handler and captured execution results."""
            if entry == "plugin":
                raise failure
            return (
                "import sys\n"
                "print('prefill diagnostic')\n"
                "print('prefill stderr', file=sys.stderr)\n"
                "await self.fail_prefill()"
            )

    class PolicyAgent(Agent):
        """Let a policy error escape without an application-side catch."""

        async def fail_prefill(self) -> None:
            """Reject before the main model response is requested."""
            raise failure

        @strategy(_strategy(strategy_type, prefill=FailingPrefill()))
        async def answer(self) -> int:
            """Generate an integer only after prefill succeeds or recovers."""
            ...

    llm = FakeLLMClient([_response(strategy_type)], strict_exhaustion=True)
    agent = PolicyAgent(llm=llm)
    try:
        if error_type is ValueError:
            assert await agent.answer() == 7
            assert len(llm.calls) == 1
        else:
            with pytest.raises(error_type) as caught:
                await agent.answer()
            assert caught.value is failure
            assert not llm.calls
            assert agent.events.query(type="Error") == []
        if entry == "execution":
            outputs = [
                event
                for event in agent.events.query(type="PythonOutput")
                if event.metadata.get("prefill")
            ]
            assert len(outputs) == 1
            output = outputs[0]
            assert "prefill diagnostic\n" in output.stdout
            assert "prefill stderr\n" in output.stderr
            assert output.execution_status is ResultStatus.ERROR
            assert f"{error_type.__name__}: prefill failure" in output.error
            assert output.metadata["prefill_type"] == "inspect_inputs"
            assert output.metadata["execution_error"] is True
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type", STRATEGIES)
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
        """Run the admission owner through the normal strategy entrypoint."""

        @strategy(_strategy(strategy_type))
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
            with patch("litellm.acompletion", AsyncMock(return_value=_reply(strategy_type))):
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
    """A later rejection stays terminal after a real in-process or guarded cell."""
    if execution == "sandbox" and sys.platform != "linux":
        pytest.skip("Guarded worker validation requires Linux")
    broker = AdmissionBroker(max_in_flight=1, max_calls=1).start()
    controller = broker.controller()
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini"), AdmissionControlConfig(controller=controller)
    )

    class PolicyAgent(Agent):
        """Use the first attempt for a cell, then exhaust the shared cap."""

        def note_pid(self, pid: int) -> None:
            """Record where the first cell executed."""
            self._cell_pid = pid

        @strategy(_strategy(strategy_type, execution=execution))
        async def answer(self) -> int:
            """Run a cell before asking for the final model result."""
            ...

    agent = PolicyAgent(llm=llm)
    try:
        with (
            patch(
                "litellm.acompletion",
                AsyncMock(
                    return_value=_reply(
                        strategy_type, "self.note_pid(os.getpid())\nprint('completed first cell')"
                    )
                ),
            ) as provider,
            patch.object(controller, "acquire", AsyncMock(wraps=controller.acquire)) as acquire,
            patch.object(agent.event_manager, "add", wraps=agent.event_manager.add) as add,
        ):
            with pytest.raises(AdmissionCallCapError):
                await agent.answer()
            assert provider.await_count == 1
            assert acquire.await_count == 2
            assert (agent._cell_pid != os.getpid()) == (execution == "sandbox")
            assert any(
                isinstance(e, PythonOutput) and "completed first cell" in e.stdout
                for e in agent.events.query(type="PythonOutput")
            )
            assert agent.events.query(type="Error") == []
            turns = [c.args[0] for c in add.call_args_list if isinstance(c.args[0], AfterTurn)]
            assert len(turns) == 2 and not turns[0].is_final
            assert turns[-1].is_final and not turns[-1].success
            assert turns[-1].exception_type == "AdmissionCallCapError"
    finally:
        await agent.aclose()
        await llm.aclose()
        controller.close()
        broker.close()


@pytest.mark.parametrize("strategy_type", STRATEGIES)
async def test_cancelled_admission_wait_stays_cancelled_and_owner_recovers(strategy_type):
    """Cancellation neither becomes a policy/API error nor strands queued admission."""
    broker = AdmissionBroker(max_in_flight=1).start()
    controller = broker.controller()
    held = await controller.acquire(lambda _: None)
    llm = AdmissionControl(
        CompletionClient("openai/gpt-4o-mini"), AdmissionControlConfig(controller=controller)
    )

    class PolicyAgent(Agent):
        """Wait under a shared broker and retain normal task cancellation."""

        @strategy(_strategy(strategy_type))
        async def answer(self) -> int:
            """Generate an integer after capacity opens."""
            ...

    agent = PolicyAgent(llm=llm)
    task = None
    try:
        with patch(
            "litellm.acompletion", AsyncMock(return_value=_reply(strategy_type))
        ) as provider:
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


@pytest.mark.parametrize("strategy_type", STRATEGIES)
async def test_ordinary_provider_error_still_retries_and_recovers(strategy_type):
    """Keep existing recovery for transient provider failures, unlike terminal policy errors."""
    llm = FakeLLMClient()

    class RecoveringAgent(Agent):
        """Retain the recoverable provider-failure path."""

        @strategy(_strategy(strategy_type))
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
                    _response(strategy_type),
                ]
            ),
        ) as provider:
            assert await agent.answer() == 7
            assert provider.await_count == 2
        assert len(agent.events.query(type="Error")) == 1
    finally:
        await agent.aclose()
