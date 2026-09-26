# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for translating NOOA events into ACP updates."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ContentToolCallContent,
    CurrentModeUpdate,
    FileEditToolCallContent,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from nooa_coder.acp.event_bridge import ACPEventBridge
from nooa_coder.coding import (
    CodingAgent,
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)
from nooa_coder.session.items import (
    CancelledUpdate,
    ChildCreatedUpdate,
    ClosedUpdate,
    ItemAdmittedUpdate,
    ModeChangedUpdate,
    SessionInfo,
    TitleChangedUpdate,
    TurnEndedUpdate,
)
from nooa_coder.session.session import Session

from nooa.context_blocks.events import ResultStatus, ToolCallEvent
from nooa.events import LLMResponse, PythonOutput
from nooa.interactive import AgentMessage, Done, NeedInput, RespondResult, Waiting
from nooa.llm_types import AssistantReasoning, LLMUsage
from nooa.unifiedllm import FakeLLMClient


class _FakeSession:
    """The part of a Session the bridge uses: its id, agent, info and updates.

    Like a Session, it counts each model call's usage (with the Session's
    own ``_count_usage``) and emits a ``UsageChangedUpdate``.
    """

    def __init__(self, agent: Any, session_id: str = "session-1") -> None:
        self.id = session_id
        self.agent = agent
        self.info = SessionInfo(id=session_id)
        self.listeners: list[Any] = []
        self.handle = SimpleNamespace(update_usage=lambda usage: None)
        self._emit = self.emit
        agent.event_manager.on("LLMResponse", lambda event: Session._count_usage(self, event))

    def subscribe(self, listener: Any) -> Any:
        self.listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self.listeners:
                self.listeners.remove(listener)

        return unsubscribe

    def emit(self, update: Any) -> None:
        for listener in list(self.listeners):
            listener(update)


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))


def _content_text(content: ContentToolCallContent) -> str:
    block = cast(TextContentBlock, content.content)
    return block.text


@pytest.mark.parametrize("tool_name", ["execute_python", "python_cell"])
async def test_bridge_preserves_message_tool_and_usage_order(tmp_path, tool_name):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

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
            name=tool_name,
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
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()

    updates = [update for _, update in client.updates]
    assert {session_id for session_id, _ in client.updates} == {"session-1"}
    assert [type(update) for update in updates] == [
        AgentMessageChunk,
        ToolCallStart,
        ToolCallProgress,
        UsageUpdate,
    ]
    assert cast(AgentMessageChunk, updates[0]).content.text == "Final answer\n\n"
    started = cast(ToolCallStart, updates[1])
    assert started.kind == "other"
    assert started.status == "in_progress"
    assert started.raw_input == {"code": "print('hello')"}
    assert started.content is not None
    assert len(started.content) == 1
    source = _content_text(cast(ContentToolCallContent, started.content[0]))
    assert source == "```python\nprint('hello')\n```"
    assert started.model_dump(mode="json", by_alias=True, exclude_none=True)["content"][0] == {
        "type": "content",
        "content": {
            "type": "text",
            "text": "```python\nprint('hello')\n```",
        },
    }

    completed = cast(ToolCallProgress, updates[2])
    assert completed.title == "python: print('hello')"
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
    await agent.aclose()


async def test_the_usage_update_carries_the_token_totals(tmp_path):
    from nooa_coder.session.items import Usage

    llm = FakeLLMClient()
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = _RecordingClient()
    session = _FakeSession(agent, "session-1")
    session.info.usage = Usage(attributed_cached_input_tokens=5)
    bridge = ACPEventBridge(session, client)  # type: ignore[arg-type]
    agent.event_manager.add(LLMResponse(usage=LLMUsage(input_tokens=40, cached_input_tokens=30)))
    await bridge.flush()

    [update] = [u for _, u in client.updates if isinstance(u, UsageUpdate)]
    assert update.field_meta is not None
    totals = update.field_meta["dev.nooa/usage"]
    assert (totals["input_tokens"], totals["cached_input_tokens"]) == (40, 35)
    await bridge.close()
    await agent.aclose()


async def test_the_python_card_names_its_code_like_the_shell_card(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]
    long_line = "total = " + " + ".join(str(n) for n in range(40))
    code = f"\n\n  {long_line}\nprint(total)\n"

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="call-1", name="execute_python", arguments={"code": code})
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="call-1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="780\n",
        )
    )
    await bridge.flush()

    started = next(u for _, u in client.updates if isinstance(u, ToolCallStart))
    assert started.raw_input == {"code": code}
    assert started.title == "python: " + long_line[:79] + "…"
    completed = next(u for _, u in client.updates if isinstance(u, ToolCallProgress))
    assert completed.raw_input == {"code": code}
    assert completed.title == started.title
    assert completed.content is not None and len(completed.content) == 2
    assert _content_text(cast(ContentToolCallContent, completed.content[1])) == (
        "```text\n780\n```"
    )
    await bridge.close()
    await agent.aclose()


async def test_bridge_marks_failed_python_output(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

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
    assert progress.title == "python: raise RuntimeError('boom') (failed)"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nraise RuntimeError('boom')\n```"
    assert output == "```text\nExecution error: RuntimeError: boom\n```"
    await bridge.close()
    await agent.aclose()


async def test_bridge_retains_python_source_when_interrupted(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="call-1",
            name="execute_python",
            arguments={"code": "await asyncio.sleep(30)"},
        )
    )
    bridge.fail_open_tools("User canceled")
    await bridge.flush()

    progress = next(
        cast(ToolCallProgress, update)
        for _, update in client.updates
        if isinstance(update, ToolCallProgress)
    )
    assert progress.title == "python: await asyncio.sleep(30) (interrupted)"
    assert progress.status == "failed"
    assert progress.content is not None
    assert len(progress.content) == 2
    source = _content_text(cast(ContentToolCallContent, progress.content[0]))
    output = _content_text(cast(ContentToolCallContent, progress.content[1]))
    assert source == "```python\nawait asyncio.sleep(30)\n```"
    assert output == "```text\nUser canceled\n```"
    await bridge.close()
    await agent.aclose()


async def test_bridge_omits_usage_when_context_window_is_unknown(tmp_path):
    llm = FakeLLMClient()
    cast(Any, llm)._context_window = None
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(AgentMessage(content="alive"))
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()

    # Positive control: prove the bridge is actually forwarding before asserting
    # an absence. Without it this passes even with every handler unsubscribed.
    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "alive\n\n"
        for _, update in client.updates
    )
    assert not any(isinstance(update, UsageUpdate) for _, update in client.updates)
    await bridge.close()
    await agent.aclose()

    # Paired positive: the same event with a known context window must emit a
    # UsageUpdate. Without this, `return` at the top of _on_llm_response passes
    # both halves — an AgentMessageChunk control comes from a different handler
    # and cannot tell "the guard works" from "usage never fires".
    sized = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    sized_client = _RecordingClient()
    sized_bridge = ACPEventBridge(_FakeSession(sized, "session-2"), sized_client)  # type: ignore[arg-type]
    sized.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await sized_bridge.flush()
    assert any(isinstance(update, UsageUpdate) for _, update in sized_client.updates)
    await sized_bridge.close()
    await sized.aclose()


async def test_bridge_emits_structured_file_edit(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]
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
    assert update.content is not None
    content = cast(FileEditToolCallContent, update.content[0])
    assert content.path == path
    assert content.old_text == "old\n"
    assert content.new_text == "new\n"
    await bridge.close()
    await agent.aclose()


async def test_bridge_emits_terminal_lifecycle(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

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
    assert started.raw_input == {
        "command": "pytest -q",
        "working_directory": str(tmp_path),
        "command_truncated": False,
        "stdin_truncated": False,
    }
    assert not started.content
    progress = cast(ToolCallProgress, updates[1])
    assert progress.content is not None
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
    await agent.aclose()


def _texts(update: Any) -> list[str]:
    return [_content_text(content) for content in update.content or []]


async def test_a_truncated_diff_is_flagged_and_says_so(bridged, tmp_path):
    agent, _session, client, bridge = bridged
    agent.event_manager.add(
        FileEdit(
            path=str(tmp_path / "big.py"),
            operation="update",
            diff="@@ -1,400 +1,400 @@\n-old\n+new\n",
            content_complete=False,
            diff_complete=False,
        )
    )
    await bridge.flush()
    [start] = [u for _, u in client.updates if isinstance(u, ToolCallStart)]
    assert start.raw_output == {"content_complete": False, "diff_complete": False}
    texts = _texts(start)
    assert texts[0].startswith("@@ -1,400")
    assert texts[-1] == "The diff was truncated: the edit is larger than shown."


async def test_a_truncated_command_is_flagged_and_says_so_until_it_finishes(bridged):
    agent, _session, client, bridge = bridged
    agent.event_manager.add(
        TerminalCommandStarted(
            command_id="cmd-1",
            command="python - <<'EOF' ...",
            working_directory="/",
            stdin="print(1)...",
            command_truncated=True,
            stdin_truncated=True,
        )
    )
    note = "The command and its standard input were truncated for display."
    agent.event_manager.add(TerminalCommandOutput(command_id="cmd-1", stdout="1\n"))
    agent.event_manager.add(TerminalCommandFinished(command_id="cmd-1", exit_code=0))
    await bridge.flush()
    start, output, finished = [u for _, u in client.updates]
    assert start.raw_input == {
        "command": "python - <<'EOF' ...",
        "working_directory": "/",
        "command_truncated": True,
        "stdin_truncated": True,
    }
    assert _texts(start) == [note]
    assert _texts(output) == [note, "1\n"]
    assert _texts(finished) == [note, "1\n"]


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
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _BlockingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]
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
    assert messages == ["First\n\n", "Second\n\n"]
    await bridge.close()
    await agent.aclose()


async def test_a_failed_update_does_not_silence_the_session_for_good(tmp_path):
    """A transport failure must end its own turn, not every later one.

    The error was latched and never cleared, so after one failed write the
    bridge dropped every subsequent update and re-raised the stale exception
    on every future flush — the agent kept running turns, at full cost, that
    the client never saw.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)

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
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    # Turn 1: the write fails, and the turn is told about it.
    agent.event_manager.add(AgentMessage(content="first turn"))
    with pytest.raises(ConnectionResetError):
        await bridge.flush()

    # Turn 2: the transport is healthy again, so updates must flow.
    agent.event_manager.add(AgentMessage(content="second turn"))
    await bridge.flush()

    assert any(
        isinstance(update, AgentMessageChunk) and update.content.text == "second turn\n\n"
        for update in client.updates
    )
    await bridge.close()


async def test_a_cancelled_command_reads_as_cancellation_not_a_crash(tmp_path):
    """The client must see the user's action, not a Python exception name."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

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
    # Status too: dropping `or event.cancelled` renders a cancelled command as a
    # green completed card while the reason text still reads correctly.
    assert finished.status == "failed"
    await bridge.close()


async def test_bare_expression_result_is_shown_not_reported_as_no_output(tmp_path):
    """A cell whose value is its result must not render as "Completed.".

    codeact puts the value in the model's own context as Out[n], so dropping it
    means the client is told there was no output while the agent reasons from
    one.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "1 + 1"})
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="t1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=3,
            value=42,
        )
    )
    await bridge.flush()

    rendered = "".join(str(update) for _, update in client.updates)
    assert "Out[3]: 42" in rendered, rendered
    assert "Completed." not in rendered
    await bridge.close()


@pytest.mark.parametrize(
    "value",
    [
        NeedInput(question="Which branch?", options=["main", "dev"]),
        Done(explanation="finished"),
        Waiting(explanation="job running", on=["jobs"]),
        RespondResult(kind="DONE", explanation="answered"),
    ],
    ids=["need_input", "done", "waiting", "respond_result"],
)
async def test_a_turn_result_is_not_shown_as_out(tmp_path, value):
    """``return_result(...)`` ends the turn; its value is not output for the card."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="t1", name="execute_python", arguments={"code": "return_result(x)"}
        )
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="t1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=2,
            value=value,
        )
    )
    await bridge.flush()

    rendered = "".join(str(update) for _, update in client.updates)
    assert "Out[" not in rendered, rendered
    await bridge.close()


async def test_synthetic_text_replies_are_not_rendered_as_python_runs(tmp_path):
    """codeact turns a prose-only reply into a synthetic execute_python call.

    Nothing was executed, so surfacing it as a Python tool call shows the user
    a run that never happened, with their model's prose commented out inside.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id="t2",
            name="execute_python",
            arguments={"code": "# I think the answer is 42"},
            metadata={"synthetic": True, "synthetic_type": "text_response"},
        )
    )
    await bridge.flush()

    assert not [u for _, u in client.updates if isinstance(u, ToolCallStart)]
    await bridge.close()


async def test_an_unfinished_tool_call_does_not_leak_for_the_session(tmp_path):
    """Closing the bridge must not leave a card spinning or state retained."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t3", name="execute_python", arguments={"code": "boom"})
    )
    await bridge.flush()
    assert set(bridge._open) == {("session-1", "t3")}

    await bridge.close()
    assert bridge._open == {}
    # And the card was actually closed out for the client: clearing the private
    # state alone leaves it spinning, which is what the docstring forbids.
    closing = [
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and update.tool_call_id == "t3"
    ]
    assert closing and closing[-1].status == "failed", client.updates
    assert closing[-1].title.endswith(" (unfinished)")


async def test_a_cancelled_tool_card_is_titled_cancelled(tmp_path):
    """The collapsed card must say what happened.

    "Python interrupted" was used for every reason, so a user who cancelled saw
    a technical-sounding failure and had to expand the card to learn it was
    their own action.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t9", name="execute_python", arguments={"code": "sleep(60)"})
    )
    await bridge.flush()
    bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
    await bridge.flush()

    progress = [u for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert progress[-1].title == "python: sleep(60) (cancelled)"
    await bridge.close()


async def test_a_force_closed_terminal_card_keeps_what_it_streamed(tmp_path):
    """Cancelling a running command closes its card with its output so far, then the reason."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    client = _RecordingClient()
    bridge = ACPEventBridge(_FakeSession(agent, "session-1"), client)  # type: ignore[arg-type]

    agent.event_manager.add(
        TerminalCommandStarted(command_id="cmd-1", command="make", working_directory="/")
    )
    agent.event_manager.add(TerminalCommandOutput(command_id="cmd-1", stdout="Building...\n50%\n"))
    await bridge.flush()
    bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
    await bridge.flush()

    last = [u for _, u in client.updates if isinstance(u, ToolCallProgress)][-1]
    text = "\n".join(block.content.text for block in last.content)
    assert text == "Building...\n50%\n\nCancelled by user."
    await bridge.close()


async def test_a_dead_pump_fails_flush_instead_of_hanging(tmp_path):
    """A BaseException from the transport must not strand every later flush.

    _pump caught only Exception, so a CancelledError raised by
    client.session_update — a disconnect during transport teardown — killed the
    pump task without resolving the queued flush marker. flush() waits on that
    marker and never observed the task, so it blocked forever, and close()
    flushes before awaiting the pump, hanging session teardown too.
    """
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)

    class _DyingClient:
        async def session_update(self, session_id: str, update: object, **kwargs) -> None:
            raise asyncio.CancelledError()

    bridge = ACPEventBridge(_FakeSession(agent), _DyingClient())  # type: ignore[arg-type]
    agent.event_manager.add(AgentMessage(content="first"))

    # Must be the specific error flush() raises. pytest.raises(BaseException)
    # also accepts the TimeoutError from wait_for, so it passed against the
    # hanging bridge too — the exact regression it claims to cover.
    with pytest.raises(RuntimeError, match="ACP event bridge stopped"):
        await asyncio.wait_for(bridge.flush(), timeout=5)

    # And teardown must not hang either.
    await asyncio.wait_for(bridge.close(), timeout=5)
    await agent.aclose()


def _content_text_or_none(update: AgentMessageChunk) -> str | None:
    block = update.content
    return getattr(block, "text", None)


# ---- the bridge on the Session's updates -------------------------------------


def _types(client: _RecordingClient) -> list[type]:
    return [type(update) for _, update in client.updates]


def _messages(client: _RecordingClient) -> list[str]:
    return [
        cast(TextContentBlock, update.content).text
        for _, update in client.updates
        if isinstance(update, AgentMessageChunk)
    ]


@pytest.fixture
async def bridged(tmp_path):
    """A coding agent behind a fake session, bridged to a recording client."""
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    session = _FakeSession(agent)
    client = _RecordingClient()
    bridge = ACPEventBridge(session, client)  # type: ignore[arg-type]
    yield agent, session, client, bridge
    await bridge.close()
    await agent.aclose()


async def test_title_changes_become_session_info_updates(bridged):
    _agent, session, client, bridge = bridged
    session.emit(TitleChangedUpdate(session_id="session-1", title="Fix the parser", user_set=False))
    await bridge.flush()
    [info] = [update for _, update in client.updates if isinstance(update, SessionInfoUpdate)]
    assert info.title == "Fix the parser"
    assert info.updated_at is not None and info.updated_at.endswith("+00:00")


async def test_mode_changes_become_current_mode_updates(bridged):
    _agent, session, client, bridge = bridged
    session.emit(ModeChangedUpdate(session_id="session-1", mode="auto"))
    await bridge.flush()
    [update] = [u for _, u in client.updates if isinstance(u, CurrentModeUpdate)]
    assert update.current_mode_id == "auto"


async def test_a_cancelled_cell_is_a_failed_card_titled_cancelled_with_its_output(bridged):
    agent, _session, client, bridge = bridged
    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "work()"})
    )
    agent.event_manager.add(
        PythonOutput(
            tool_call_id="t1",
            execution_status=ResultStatus.CANCELLED,
            execution_count=1,
            stdout="step 1 done\n",
        )
    )
    await bridge.flush()
    [progress] = [u for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert progress.status == "failed"
    assert progress.title == "python: work() (cancelled)"
    assert "step 1 done" in str(progress.content)


async def test_the_cancelled_update_closes_open_cards_before_saying_so(bridged, tmp_path):
    agent, session, client, bridge = bridged
    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "sleep()"})
    )
    agent.event_manager.add(
        TerminalCommandStarted(command_id="cmd-1", command="sleep 30", working_directory="/")
    )
    session.emit(CancelledUpdate(session_id="session-1", by="user"))
    await bridge.flush()
    closed = [u for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert {(u.tool_call_id, u.status, u.title) for u in closed} == {
        ("t1", "failed", "python: sleep() (cancelled)"),
        ("cmd-1", "failed", "Cancelled"),
    }
    assert _types(client)[-1] is AgentMessageChunk
    assert _messages(client) == ["Stopped at your request.\n\n"]
    assert bridge._open == {}


async def test_a_question_is_rendered_once_when_the_turn_ends(bridged):
    _agent, session, client, bridge = bridged
    session.emit(
        TurnEndedUpdate(
            session_id="session-1",
            outcome_kind="need_input",
            outcome={"question": "Which branch?", "options": ["main", "dev"]},
        )
    )
    await bridge.flush()
    assert _messages(client) == ["Which branch?\n\n- main\n- dev\n\n"]


async def test_a_failed_turn_closes_its_open_cards_as_unfinished(bridged):
    agent, session, client, bridge = bridged
    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "x"})
    )
    session.emit(
        TurnEndedUpdate(session_id="session-1", outcome_kind="error", outcome={"error": "boom"})
    )
    await bridge.flush()
    [progress] = [u for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert (progress.status, progress.title) == ("failed", "python: x (unfinished)")


async def test_reasoning_becomes_thought_chunks(bridged):
    agent, _session, client, bridge = bridged
    agent.event_manager.add(
        LLMResponse(
            parts=[AssistantReasoning(text="Check the tests first.")],
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )
    )
    await bridge.flush()
    [thought] = [u for _, u in client.updates if isinstance(u, AgentThoughtChunk)]
    assert cast(TextContentBlock, thought.content).text == "Check the tests first."


async def test_todo_changes_become_plan_updates_after_a_cell(bridged):
    agent, _session, client, bridge = bridged

    def cell(tool_call_id: str) -> None:
        agent.event_manager.add(
            ToolCallEvent(tool_call_id=tool_call_id, name="execute_python", arguments={"code": ""})
        )
        agent.event_manager.add(
            PythonOutput(
                tool_call_id=tool_call_id,
                execution_status=ResultStatus.COMPLETE,
                execution_count=1,
            )
        )

    cell("t0")  # no todos yet: no plan
    first = agent.todo.add("Read the parser")
    second = agent.todo.add("Fix the bug")
    agent.todo.activate(second)
    agent.todo.done(first)
    cell("t1")
    cell("t2")  # unchanged: no second plan
    await bridge.flush()
    plans = [u for _, u in client.updates if isinstance(u, AgentPlanUpdate)]
    assert len(plans) == 1
    assert [(entry.content, entry.status) for entry in plans[0].entries] == [
        ("Read the parser", "completed"),
        ("Fix the bug", "in_progress"),
    ]


async def test_messages_from_other_senders_are_echoed_as_user_chunks(bridged):
    _agent, session, client, bridge = bridged
    for channel, source in (
        ("user_messages", "acp"),  # this adapter's own prompt: the client shows it already
        ("user_messages", "user:declined"),
        ("delegates", "child:helper"),
        ("user_messages", "parent:root"),
        ("steer", "tui"),
    ):
        session.emit(
            ItemAdmittedUpdate(
                session_id="session-1",
                channel=channel,
                item_id=f"{channel}-{source}",
                source=source,
                preview=f"from {source}",
            )
        )
    await bridge.flush()
    echoed = [
        cast(TextContentBlock, u.content).text
        for _, u in client.updates
        if isinstance(u, UserMessageChunk)
    ]
    assert echoed == ["from parent:root", "from tui"]


async def test_tool_cards_of_a_child_are_mirrored_under_the_childs_id(tmp_path):
    parent_agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    child_agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    parent = _FakeSession(parent_agent, "parent")
    child = _FakeSession(child_agent, "child-1")
    client = _RecordingClient()
    bridge = ACPEventBridge(
        parent,  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
        resolve_child=lambda child_id: child if child_id == "child-1" else None,
    )
    parent.emit(
        ChildCreatedUpdate(
            session_id="parent", child_id="child-1", name="helper", depth=1, retained=False
        )
    )
    # The same tool call id in parent and child are different cards.
    for agent in (parent_agent, child_agent):
        agent.event_manager.add(
            ToolCallEvent(tool_call_id="c1", name="execute_python", arguments={"code": "x"})
        )
    child_agent.event_manager.add(
        PythonOutput(tool_call_id="c1", execution_status=ResultStatus.COMPLETE, execution_count=1)
    )
    # A child's own messages are not the parent's conversation.
    child_agent.event_manager.add(AgentMessage(content="child chatter"))
    await bridge.flush()

    [info] = [u for _, u in client.updates if isinstance(u, SessionInfoUpdate)]
    assert info.field_meta == {
        "dev.nooa/children": [
            {"sessionId": "child-1", "name": "helper", "depth": 1, "retained": False}
        ]
    }
    starts = [u.tool_call_id for _, u in client.updates if isinstance(u, ToolCallStart)]
    assert starts == ["c1", "child-1:c1"]
    finished = [u.tool_call_id for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert finished == ["child-1:c1"]
    assert set(bridge._open) == {("parent", "c1")}
    assert _messages(client) == []

    # When the child closes, the parent stops mirroring it.
    child.emit(ClosedUpdate(session_id="child-1"))
    assert child.listeners == []
    await bridge.close()
    await parent_agent.aclose()
    await child_agent.aclose()


@pytest.mark.parametrize(
    "update",
    [
        CancelledUpdate(session_id="parent", by="user"),
        TurnEndedUpdate(session_id="parent", outcome_kind="error", outcome={"error": "boom"}),
    ],
    ids=["cancelled", "failed-turn"],
)
async def test_ending_the_parents_turn_leaves_a_mirrored_childs_cards_open(tmp_path, update):
    parent_agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    child_agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    parent = _FakeSession(parent_agent, "parent")
    child = _FakeSession(child_agent, "child-1")
    client = _RecordingClient()
    bridge = ACPEventBridge(
        parent,  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
        resolve_child=lambda child_id: child if child_id == "child-1" else None,
    )
    parent.emit(
        ChildCreatedUpdate(
            session_id="parent", child_id="child-1", name="helper", depth=1, retained=False
        )
    )
    for agent in (parent_agent, child_agent):
        agent.event_manager.add(
            ToolCallEvent(tool_call_id="c1", name="execute_python", arguments={"code": "x"})
        )
    child_agent.event_manager.add(
        TerminalCommandStarted(command_id="cmd-1", command="make", working_directory="/")
    )
    parent.emit(update)
    await bridge.flush()
    closed = [u.tool_call_id for _, u in client.updates if isinstance(u, ToolCallProgress)]
    assert closed == ["c1"]  # the parent's own card only

    # The child goes on running; its real results close its cards, once each.
    child_agent.event_manager.add(
        PythonOutput(tool_call_id="c1", execution_status=ResultStatus.COMPLETE, execution_count=1)
    )
    child_agent.event_manager.add(TerminalCommandFinished(command_id="cmd-1", exit_code=0))
    await bridge.flush()
    child_updates = [
        (u.tool_call_id, u.status)
        for _, u in client.updates
        if isinstance(u, ToolCallProgress) and u.tool_call_id.startswith("child-1:")
    ]
    assert child_updates == [("child-1:c1", "completed"), ("child-1:cmd-1", "completed")]
    await bridge.close()
    await parent_agent.aclose()
    await child_agent.aclose()


async def test_the_bridge_unsubscribes_when_its_session_closes(bridged):
    agent, session, client, bridge = bridged
    session.emit(ClosedUpdate(session_id="session-1"))
    await asyncio.wait_for(bridge.wait_closed(), 5)
    assert session.listeners == []
    agent.event_manager.add(AgentMessage(content="after close"))
    await asyncio.sleep(0)
    assert "after close" not in _messages(client)


async def test_usage_includes_cost_attributed_from_children(bridged):
    agent, session, client, bridge = bridged
    session.info.usage.attributed_cost_usd = 1.0
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()
    [usage] = [u for _, u in client.updates if isinstance(u, UsageUpdate)]
    assert usage.cost is not None and usage.cost.amount == 1.25
    assert usage.field_meta is not None and "dev.nooa/context" in usage.field_meta


async def test_a_resumed_sessions_cost_continues_from_what_it_already_spent(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    session = _FakeSession(agent)
    session.info.usage.cost_usd = 2.0  # spent before this bridge was attached
    client = _RecordingClient()
    bridge = ACPEventBridge(session, client)  # type: ignore[arg-type]
    agent.event_manager.add(
        LLMResponse(usage=LLMUsage(input_tokens=40, output_tokens=10, cost_usd=0.25))
    )
    await bridge.flush()
    [usage] = [u for _, u in client.updates if isinstance(u, UsageUpdate)]
    assert usage.cost is not None and usage.cost.amount == 2.25
    await bridge.close()
    await agent.aclose()


async def test_a_grandchild_is_announced_before_its_mirrored_cards(tmp_path):
    agents = {name: CodingAgent(llm=FakeLLMClient(), cwd=tmp_path) for name in ("p", "c", "g")}
    parent = _FakeSession(agents["p"], "parent")
    sessions = {
        "child-1": _FakeSession(agents["c"], "child-1"),
        "grandchild-1": _FakeSession(agents["g"], "grandchild-1"),
    }
    client = _RecordingClient()
    bridge = ACPEventBridge(
        parent,  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
        resolve_child=sessions.get,
    )
    parent.emit(
        ChildCreatedUpdate(
            session_id="parent", child_id="child-1", name="helper", depth=1, retained=False
        )
    )
    sessions["child-1"].emit(
        ChildCreatedUpdate(
            session_id="child-1", child_id="grandchild-1", name="scout", depth=2, retained=True
        )
    )
    agents["g"].event_manager.add(
        ToolCallEvent(tool_call_id="c1", name="execute_python", arguments={"code": "x"})
    )
    await bridge.flush()

    infos = [u for _, u in client.updates if isinstance(u, SessionInfoUpdate)]
    assert infos[-1].field_meta == {
        "dev.nooa/children": [
            {"sessionId": "child-1", "name": "helper", "depth": 1, "retained": False},
            {"sessionId": "grandchild-1", "name": "scout", "depth": 2, "retained": True},
        ]
    }
    kinds = [type(u) for _, u in client.updates]
    assert kinds == [SessionInfoUpdate, SessionInfoUpdate, ToolCallStart]
    [start] = [u for _, u in client.updates if isinstance(u, ToolCallStart)]
    assert start.tool_call_id == "grandchild-1:c1"
    await bridge.close()
    for agent in agents.values():
        await agent.aclose()


async def test_detaching_leaves_open_cards_alone_even_if_the_session_closed_first(bridged):
    """close(finish_open=False) wins over a close already started by the session."""
    agent, session, client, bridge = bridged
    agent.event_manager.add(
        ToolCallEvent(tool_call_id="t1", name="execute_python", arguments={"code": "x"})
    )
    session.emit(ClosedUpdate(session_id="session-1"))  # starts a close that finishes cards
    await bridge.close(finish_open=False)
    assert [u for _, u in client.updates if isinstance(u, ToolCallProgress)] == []


def test_every_agent_message_ends_its_line():
    """Clients join adjacent agent chunks, and one line break is a soft break in Markdown.

    In Pool 1.0.16 a /usage table after "...billing." rendered as one paragraph
    with its opening fence glued to the text, and "Not posted.\n" followed by
    "Trace viewer:" rendered as "Not posted. Trace viewer:". Every agent
    message therefore ends with a blank line.
    """
    from acp import text_block, update_agent_message, update_agent_thought_text
    from nooa_coder.acp.event_bridge import end_line

    ended = end_line(update_agent_message(text_block("ending in billing.")))
    assert ended.content.text == "ending in billing.\n\n"
    one_break = end_line(update_agent_message(text_block("Not posted.\n")))
    assert one_break.content.text == "Not posted.\n\n"
    already = update_agent_message(text_block("```text\nx\n```\n\n"))
    assert end_line(already) is already
    thought = update_agent_thought_text("thinking")
    assert end_line(thought) is thought
