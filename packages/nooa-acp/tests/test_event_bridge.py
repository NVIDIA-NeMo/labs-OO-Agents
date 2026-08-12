# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for translating NOOA events into ACP updates."""

import asyncio
from typing import Any, cast

import pytest
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
)
from nooa_acp.coding_agent import CodingInteractiveAgent
from nooa_acp.event_bridge import ACPEventBridge
from nooa_cli.coding import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)

from nooa.context_blocks.events import ResultStatus, ToolCallEvent
from nooa.events import LLMComplete, PythonOutput
from nooa.interactive import AgentMessage
from nooa.unifiedllm import FakeLLMClient


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


def _content_text(content: ContentToolCallContent) -> str:
    block = cast(TextContentBlock, content.content)
    return block.text


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
    started = cast(ToolCallStart, updates[1])
    assert started.raw_input == {"code": "print('hello')"}
    assert started.content is not None
    assert len(started.content) == 1
    source = _content_text(cast(ContentToolCallContent, started.content[0]))
    assert source == "```python\nprint('hello')\n```"

    completed = cast(ToolCallProgress, updates[2])
    assert completed.title == "Ran Python"
    assert completed.status == "completed"
    assert completed.content is not None
    assert len(completed.content) == 2
    completed_source = _content_text(cast(ContentToolCallContent, completed.content[0]))
    output = _content_text(cast(ContentToolCallContent, completed.content[1]))
    assert completed_source == "```python\nprint('hello')\n```"
    assert output == "```text\nhello\n```"
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
    assert progress.title == "Python failed"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nraise RuntimeError('boom')\n```"
    assert output == "```text\nExecution error: RuntimeError: boom\n```"
    await bridge.close()
    await agent.close()


async def test_bridge_retains_python_source_when_interrupted(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "await asyncio.sleep(30)"},
        )
    )
    await bridge.fail_open_tools("User canceled")
    await bridge.flush()

    progress = next(
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    )
    assert progress.title == "Python interrupted"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nawait asyncio.sleep(30)\n```"
    assert output == "```text\nUser canceled\n```"
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


async def test_bridge_emits_structured_file_edit(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    path = str(tmp_path / "example.py")

    agent.event_manager.add(
        FileEdit(
            path=path,
            operation="update",
            old_text="old\n",
            new_text="new\n",
            start_line=3,
            end_line=3,
            diff="unused when complete",
        )
    )
    await bridge.flush()

    update = next(
        cast(ToolCallStart, update)
        for _, update in client.updates
        if isinstance(update, ToolCallStart)
    )
    assert update.kind == "edit"
    assert update.status == "completed"
    assert update.locations is not None
    assert update.locations[0].path == path
    assert update.locations[0].line == 2
    content = cast(FileEditToolCallContent, update.content[0])
    assert content.path == path
    assert content.old_text == "old\n"
    assert content.new_text == "new\n"
    await bridge.close()
    await agent.close()


async def test_bridge_emits_terminal_lifecycle(tmp_path):
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="command-1",
            command="pytest -q",
            working_directory=str(tmp_path),
        )
    )
    agent.event_manager.add(TerminalCommandOutput(command_id="command-1", stdout="2 passed\n"))
    agent.event_manager.add(TerminalCommandFinished(command_id="command-1", exit_code=0))
    await bridge.flush()

    updates = [update for _, update in client.updates]
    assert [type(update) for update in updates] == [
        ToolCallStart,
        ToolCallProgress,
        ToolCallProgress,
    ]
    started = cast(ToolCallStart, updates[0])
    assert started.kind == "execute"
    assert started.title == "$ pytest -q"
    progress = cast(ToolCallProgress, updates[1])
    content = cast(ContentToolCallContent, progress.content[0])
    assert content.content.text == "2 passed\n"
    finished = cast(ToolCallProgress, updates[2])
    assert finished.status == "completed"
    assert finished.raw_output == {
        "exit_code": 0,
        "timed_out": False,
        "output_truncated": False,
    }
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


async def test_a_failed_update_does_not_silence_the_session_for_good(tmp_path):
    """A transport failure must end its own turn, not every later one.

    The error was latched and never cleared, so after one failed write the
    bridge dropped every subsequent update and re-raised the stale exception
    on every future flush — the agent kept running turns, at full cost, that
    the client never saw.
    """
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)

    class _FlakyClient:
        def __init__(self) -> None:
            self.updates: list[object] = []
            self.fail_next = True

        async def session_update(self, session_id: str, update: object, **kwargs) -> None:
            if self.fail_next:
                self.fail_next = False
                raise ConnectionResetError("transport went away")
            self.updates.append(update)

    client = _FlakyClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    # Turn 1: the write fails, and the turn is told about it.
    agent.event_manager.add(AgentMessage(content="first turn"))
    with pytest.raises(ConnectionResetError):
        await bridge.flush()

    # Turn 2: the transport is healthy again, so updates must flow.
    agent.event_manager.add(AgentMessage(content="second turn"))
    await bridge.flush()

    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "second turn"
        for update in client.updates
    )
    await bridge.close()


async def test_a_cancelled_command_reads_as_cancellation_not_a_crash(tmp_path):
    """The client must see the user's action, not a Python exception name."""
    agent = CodingInteractiveAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="cmd-1",
            command="sleep 30",
            working_directory=str(tmp_path),
        )
    )
    agent.event_manager.add(TerminalCommandFinished(command_id="cmd-1", cancelled=True))
    await bridge.flush()

    progress = [update for _, update in client.updates if isinstance(update, ToolCallProgress)]
    finished = progress[-1]
    rendered = str(finished)
    assert "Cancelled by user." in rendered
    assert "CancelledError" not in rendered
    await bridge.close()
