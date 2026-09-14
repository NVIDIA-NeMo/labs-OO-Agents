# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One persistent engine serves native admission and ACP foreground requests."""

import asyncio

import pytest
from nooa_cli.coding import CodingAgent
from nooa_cli.interactive.dispatcher import InteractiveSessionDispatcher
from nooa_cli.interactive.local_agent import LocalAgentRunner

from nooa.interactive import RespondReason, RespondResult
from nooa.unifiedllm import FakeLLMClient


class ScriptedAgent(CodingAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notifications = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.background_received = asyncio.Event()

    async def handle(self, notification):
        self.notifications.append(notification)
        if notification.get("user_messages") == ["block"]:
            self.entered.set()
            await self.release.wait()
        if notification.get("user_messages") == ["wait"]:
            self.queue_manager.get_channel("system_messages").put("completed")
            return RespondResult(kind=RespondReason.WAIT, explanation="background work")
        if "system_messages" in notification:
            self.background_received.set()
        return RespondResult(kind=RespondReason.DONE, explanation="complete")


@pytest.mark.parametrize("frontend", ["native", "acp"])
async def test_wait_and_background_after_done_share_one_engine(tmp_path, frontend):
    agent = ScriptedAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent) if frontend == "acp" else None
    runtime = (
        dispatcher.runtime
        if dispatcher
        else LocalAgentRunner(agent, emit_text=lambda _: None, agent_id="native")
    )
    try:
        if dispatcher:
            result = await dispatcher.submit("wait")
            assert result.kind is RespondReason.DONE
        else:
            assert runtime.submit("wait")
            await asyncio.wait_for(agent.background_received.wait(), 2)
        assert agent.notifications[:2] == [
            {"user_messages": ["wait"]},
            {"system_messages": ["completed"]},
        ]
        agent.background_received.clear()
        # No new prompt is required to wake an ACP session after DONE.
        agent.queue_manager.get_channel("system_messages").put("late report")
        await asyncio.wait_for(agent.background_received.wait(), 2)
        assert agent.notifications[-1] == {"system_messages": ["late report"]}
        with pytest.raises(RuntimeError, match="active runner"):
            LocalAgentRunner(agent, emit_text=lambda _: None, agent_id="duplicate")
    finally:
        if dispatcher:
            await dispatcher.close()
        else:
            await runtime.shutdown()
            await agent.close()


async def test_new_foreground_waits_for_its_consumed_input(tmp_path):
    agent = ScriptedAgent(llm=FakeLLMClient(), cwd=tmp_path)
    runtime = LocalAgentRunner(agent, emit_text=lambda _: None, agent_id="native")
    try:
        assert runtime.submit("block")
        await agent.entered.wait()
        followup = asyncio.create_task(runtime.submit_and_wait("followup"))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already running"):
            await runtime.submit_and_wait("duplicate")
        agent.release.set()
        result = await asyncio.wait_for(followup, 2)
        assert result.kind is RespondReason.DONE
        assert agent.notifications[-1] == {"user_messages": ["followup"]}
        assert runtime.pending_user_messages() == ()
    finally:
        await runtime.shutdown()
        await agent.close()


async def test_cancelled_waiter_stops_work_and_releases_admission(tmp_path):
    agent = ScriptedAgent(llm=FakeLLMClient(), cwd=tmp_path)
    runtime = LocalAgentRunner(agent, emit_text=lambda _: None, agent_id="native")
    try:
        task = asyncio.create_task(runtime.submit_and_wait("block"))
        await agent.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await asyncio.wait_for(runtime.submit_and_wait("again"), 2)
        assert result.kind is RespondReason.DONE
        assert agent.notifications[-1] == {"user_messages": ["again"]}
    finally:
        await runtime.shutdown()
        await agent.close()


async def test_cancel_stops_background_work_after_foreground_done(tmp_path):
    agent = ScriptedAgent(llm=FakeLLMClient(), cwd=tmp_path)
    dispatcher = InteractiveSessionDispatcher(agent)
    started = asyncio.Event()

    async def background():
        started.set()
        await asyncio.Event().wait()

    try:
        await dispatcher.submit("first")
        job = agent.queue_manager.spawn(background(), channel="system_messages")
        await started.wait()
        assert await dispatcher.cancel() is True
        assert job.state == "cancelled"
        assert (await dispatcher.submit("next")).kind is RespondReason.DONE
    finally:
        await dispatcher.close()


@pytest.mark.parametrize("error", [ValueError("failed handle"), asyncio.CancelledError()])
async def test_handle_failure_reaches_foreground_waiter(tmp_path, error):
    class FailingAgent(ScriptedAgent):
        async def handle(self, notification):
            raise error

    dispatcher = InteractiveSessionDispatcher(FailingAgent(llm=FakeLLMClient(), cwd=tmp_path))
    try:
        with pytest.raises(type(error)):
            await asyncio.wait_for(dispatcher.submit("fail"), 2)
    finally:
        await dispatcher.close()
