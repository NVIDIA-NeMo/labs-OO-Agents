# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for translating NOOA events into ACP updates."""

import asyncio
from typing import Any, cast

import pytest
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart, UsageUpdate
from nooa_acp.coding_agent import CodingInteractiveAgent
from nooa_acp.event_bridge import ACPEventBridge

from nooa.context_blocks.events import ResultStatus, ToolCallEvent
from nooa.events import LLMComplete, PythonOutput
from nooa.interactive import AgentMessage
from nooa.unifiedllm import FakeLLMClient


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


async def test_bridge_preserves_message_tool_and_usage_order(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(AgentMessage(content="Final answer"))
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="prefill-1",
            name="execute_python",
            arguments={"code": "print('internal setup')"},
            metadata={"prefill": True},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="prefill-1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="internal setup\n",
        )
    )
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "print('hello')"},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="hello\n",
        )
    )
    agent.event_manager.add(LLMComplete(prompt_tokens=40, completion_tokens=10, cost_usd=0.25))
    await bridge.flush()

    updates = [update for _, update in client.updates]
    assert {session_id for session_id, _ in client.updates} == {"session-1"}
    assert [type(update) for update in updates] == [
        AgentMessageChunk,
        ToolCallStart,
        ToolCallProgress,
        UsageUpdate,
    ]
    assert cast(AgentMessageChunk, updates[0]).content.text == "Final answer"
    assert cast(ToolCallProgress, updates[2]).status == "completed"
    usage = cast(UsageUpdate, updates[3])
    assert usage.cost is not None
    assert usage.cost.amount == 0.25
    await bridge.close()
    await agent.close()


async def test_bridge_marks_failed_python_output(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "raise RuntimeError('boom')"},
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            stderr="Execution error: RuntimeError: boom",
        )
    )
    await bridge.flush()

    progress = next(
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    )
    assert progress.status == "failed"
    assert "Execution error: RuntimeError: boom" in str(progress.content)
    await bridge.close()
    await agent.close()


async def test_bridge_omits_usage_when_context_window_is_unknown(tmp_path):
    llm = FakeLLMClient()
    cast(Any, llm)._context_window = None
    agent = CodingInteractiveAgent(llm=llm, cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(LLMComplete(prompt_tokens=40, completion_tokens=10, cost_usd=0.25))
    await bridge.flush()

    assert not any(isinstance(update, UsageUpdate) for _, update in client.updates)
    await bridge.close()
    await agent.close()


class _BlockingClient(_RecordingClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.started.set()
        await self.release.wait()
        await super().session_update(session_id, update, **kwargs)


async def test_cancelled_flush_does_not_stop_update_pump(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _BlockingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    agent.event_manager.add(AgentMessage(content="First"))
    flush_task = asyncio.create_task(bridge.flush())
    await asyncio.wait_for(client.started.wait(), timeout=1)

    flush_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush_task
    client.release.set()
    agent.event_manager.add(AgentMessage(content="Second"))
    await asyncio.wait_for(bridge.flush(), timeout=1)

    messages = [
        update.content.text for _, update in client.updates if isinstance(update, AgentMessageChunk)
    ]
    assert messages == ["First", "Second"]
    await bridge.close()
    await agent.close()
