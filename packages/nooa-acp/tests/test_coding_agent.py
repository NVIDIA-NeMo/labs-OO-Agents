# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NOOA coding agent and dispatcher."""

import asyncio
from typing import Any

from nooa_acp.coding_agent import CodingInteractiveAgent
from nooa_acp.dispatcher import InteractiveSessionDispatcher

from nooa.context_blocks.events import ToolCallEvent
from nooa.events import PythonOutput
from nooa.interactive import AgentMessage, RespondReason, RespondResult
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _completed_llm(message: str = "Finished **successfully**.") -> FakeLLMClient:
    return FakeLLMClient.with_tool_call(
        "execute_python",
        {
            "code": (
                f"self.message({message!r})\n"
                "return_result(RespondReason.DONE, explanation='completed and verified')"
            )
        },
    )


async def test_coding_agent_runs_through_nooa_codeact(tmp_path):
    agent = CodingInteractiveAgent(llm=_completed_llm(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("inspect the repository")

    assert result is not None
    assert result.kind is RespondReason.DONE
    assert agent.cwd == tmp_path.resolve()
    assert agent.shell.session is agent.repo.session
    events = agent.event_manager.values()
    assert any(isinstance(event, AgentMessage) for event in events)
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, PythonOutput) for event in events)
    await dispatcher.close()


class _BlockingLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def acall(self, *args, **kwargs) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _WaitingAgent(CodingInteractiveAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.handle_calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        self.handle_calls += 1
        if self.handle_calls == 1:
            self.queue_manager.get_channel("system_messages").put("job finished")
            return RespondResult(kind=RespondReason.WAIT, explanation="waiting for job")
        assert notification == {"system_messages": ["job finished"]}
        return RespondResult(kind=RespondReason.DONE, explanation="job finished")


class _BackgroundAgent(CodingInteractiveAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.job_started = asyncio.Event()
        self.job: Any = None

    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        async def background_job() -> None:
            self.job_started.set()
            await asyncio.Event().wait()

        self.job = self.queue_manager.spawn(background_job(), channel="system_messages")
        return RespondResult(kind=RespondReason.WAIT, explanation="waiting for job")


class _RestartableAgent(CodingInteractiveAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.started = asyncio.Event()
        self.handle_calls = 0

    async def handle(self, notification: dict[str, list[Any]]) -> RespondResult:
        self.handle_calls += 1
        if self.handle_calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return RespondResult(kind=RespondReason.DONE, explanation="second prompt completed")


async def test_dispatcher_cancels_active_nooa_turn(tmp_path):
    llm = _BlockingLLM()
    agent = CodingInteractiveAgent(llm=llm, cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    prompt_task = asyncio.create_task(dispatcher.submit("wait forever"))
    await asyncio.wait_for(llm.started.wait(), timeout=2)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(prompt_task, timeout=2) is None
    assert dispatcher.active is False
    await dispatcher.close()


async def test_dispatcher_accepts_another_prompt_after_cancellation(tmp_path):
    agent = _RestartableAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    first = asyncio.create_task(dispatcher.submit("cancel this"))
    await asyncio.wait_for(agent.started.wait(), timeout=1)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(first, timeout=1) is None
    result = await asyncio.wait_for(dispatcher.submit("try again"), timeout=1)

    assert result is not None
    assert result.kind is RespondReason.DONE
    assert agent.handle_calls == 2
    await dispatcher.close()


async def test_dispatcher_resumes_after_wait_notification(tmp_path):
    agent = _WaitingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)

    result = await dispatcher.submit("wait for the job")

    assert result is not None
    assert result.kind is RespondReason.DONE
    assert agent.handle_calls == 2
    await dispatcher.close()


async def test_dispatcher_cancels_background_jobs(tmp_path):
    agent = _BackgroundAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    prompt_task = asyncio.create_task(dispatcher.submit("start a background job"))
    await asyncio.wait_for(agent.job_started.wait(), timeout=1)

    assert await dispatcher.cancel() is True
    assert await asyncio.wait_for(prompt_task, timeout=1) is None
    assert agent.job is not None
    assert agent.job.state == "cancelled"
    await dispatcher.close()
