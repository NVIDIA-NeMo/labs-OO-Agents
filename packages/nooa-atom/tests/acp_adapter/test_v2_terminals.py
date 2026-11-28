# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-owned v2 terminals and client-initiated cancellation."""

import asyncio
import base64
from typing import Any

import pytest
from nooa_atom.acp import v2
from nooa_atom.acp.event_bridge import ACPEventBridge
from nooa_atom.acp.v1_shim import V1Peer
from nooa_atom.agent import work_shell
from nooa_atom.agent.activity import (
    AgentTerminalFinished,
    AgentTerminalOutput,
    AgentTerminalStarted,
)
from nooa_atom.agent.work_shell import Shell
from nooa_atom.session.items import AgentEventUpdate, SessionInfo

from nooa.context_blocks.events import ResultStatus, ToolCallEvent
from nooa.events import PythonOutput


class _Session:
    def __init__(self, session_id: str = "s") -> None:
        self.id = session_id
        self.info = SessionInfo(id=session_id)
        self.listeners: list[Any] = []

    def subscribe(self, listener: Any) -> Any:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def emit(self, event: Any) -> None:
        for listener in list(self.listeners):
            listener(AgentEventUpdate(session_id=self.id, event=event))

    def model_info(self) -> Any:
        return None

    def plan(self) -> list[Any]:
        return []


class _Peer:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict[str, Any]]] = []

    async def update(
        self, session_id: str, update: dict[str, Any], *, mirrored: bool = False
    ) -> None:
        self.updates.append((session_id, update))

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        raise AssertionError(method)


@pytest.fixture
async def bridged():
    session = _Session()
    peer = _Peer()
    bridge = ACPEventBridge(session, peer)  # type: ignore[arg-type]
    yield session, peer, bridge
    await bridge.close()


def _kinds(peer: _Peer) -> list[str]:
    return [update["sessionUpdate"] for _, update in peer.updates]


async def test_terminal_streams_base64_and_exits_without_a_python_card(bridged):
    session, peer, bridge = bridged
    session.emit(AgentTerminalStarted(terminal_id="t1", command="printf x", working_directory="/w"))
    session.emit(AgentTerminalOutput(terminal_id="t1", data=b"x\x00"))
    session.emit(AgentTerminalFinished(terminal_id="t1", exit_code=0))
    await bridge.flush()

    assert _kinds(peer) == [
        "terminal_update",
        "tool_call_update",
        "terminal_output_chunk",
        "terminal_update",
        "tool_call_update",
    ]
    opened = peer.updates[0][1]
    assert opened["capabilities"] == {"cancel": {}}
    assert opened["cwd"] == "/w"
    assert peer.updates[1][1]["content"] == [{"type": "terminal", "terminalId": "t1"}]
    assert base64.b64decode(peer.updates[2][1]["data"]) == b"x\x00"
    closed = peer.updates[3][1]
    assert closed["exitStatus"] == {"exitCode": 0}
    assert closed["capabilities"] is None
    assert peer.updates[4][1]["status"] == "completed"


async def test_a_python_card_keeps_its_terminal_reference_at_completion(bridged):
    session, peer, bridge = bridged
    session.emit(
        ToolCallEvent(tool_call_id="cell", name="python_cell", arguments={"code": "print(1)"})
    )
    session.emit(AgentTerminalStarted(terminal_id="t1", command="echo 1", working_directory="/w"))
    session.emit(AgentTerminalFinished(terminal_id="t1", exit_code=0))
    session.emit(
        PythonOutput(tool_call_id="cell", execution_status=ResultStatus.COMPLETE, execution_count=1)
    )
    await bridge.flush()

    final = [update for _, update in peer.updates if update.get("status") == "completed"][-1]
    assert final["toolCallId"] == "cell"
    assert final["content"][-1] == {"type": "terminal", "terminalId": "t1"}
    assert "tool_call_content_chunk" in _kinds(peer)


async def test_terminal_cancel_is_scoped_to_a_live_terminal(bridged, monkeypatch):
    session, _peer, bridge = bridged
    calls: list[str] = []

    async def cancel(terminal_id: str) -> bool:
        calls.append(terminal_id)
        return True

    monkeypatch.setattr(work_shell, "cancel_terminal", cancel)
    session.emit(AgentTerminalStarted(terminal_id="t1", command="sleep 9", working_directory="/w"))
    assert await bridge.cancel_terminal("t1")
    assert calls == ["t1"]
    session.emit(AgentTerminalFinished(terminal_id="t1", exit_code=130, cancelled=True))
    assert not await bridge.cancel_terminal("t1")
    assert not await bridge.cancel_terminal("unknown")


async def test_background_tool_call_cancel_finishes_only_that_card(bridged):
    _session, peer, bridge = bridged
    bridge._set_background(["build"])

    class Job:
        status = "running"

        async def cancel(self) -> None:
            self.status = "cancelled"

    job = Job()
    assert await bridge.cancel_background("job:build", job)
    assert job.status == "cancelled"
    await bridge.flush()
    cancelled = [u for _, u in peer.updates if u.get("status") == "cancelled"]
    assert cancelled == [
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "job:build",
            "status": "cancelled",
            "capabilities": None,
        }
    ]
    state = [u for _, u in peer.updates if u["sessionUpdate"] == "state_update"][-1]
    assert state["state"] == "idle" and state["background"] == []
    assert not await bridge.cancel_background("job:build", job)


def test_v1_drops_terminal_only_updates_and_content():
    peer = V1Peer(None)  # type: ignore[arg-type]
    assert peer._convert("s", v2.terminal_update("t1", command="x")) is None
    assert peer._convert("s", v2.terminal_output_chunk("t1", b"x")) is None
    converted = peer._convert(
        "s",
        v2.tool_call(
            "cell",
            status="in_progress",
            capabilities={"cancel": {}},
            content=[v2.tool_text("source"), v2.terminal_content("t1")],
        ),
    )
    assert converted is not None
    _target, update = converted
    assert "capabilities" not in update
    assert update["content"] == [v2.tool_text("source")]


async def test_shell_observes_raw_output_and_terminal_cancel(tmp_path):
    observations: list[work_shell.TerminalObservation] = []
    token = work_shell.terminal_observer.set(observations.append)
    shell = Shell(tmp_path)
    try:
        running = asyncio.create_task(shell.run("printf 'abc\n'; sleep 30", timeout=60))
        for _ in range(100):
            if any(item.phase == "output" for item in observations):
                break
            await asyncio.sleep(0.05)
        assert [item.phase for item in observations[:2]] == ["started", "output"]
        assert observations[1].data == b"abc\n"
        terminal_id = observations[0].terminal_id
        assert await work_shell.cancel_terminal(terminal_id)
        result = await asyncio.wait_for(running, 10)
        assert result.returncode == 130 and not result.timed_out
        finished = observations[-1]
        assert finished.phase == "finished"
        assert finished.exit_code == 130 and finished.cancelled
        assert not await work_shell.cancel_terminal(terminal_id)
    finally:
        await shell.aclose()
        work_shell.terminal_observer.reset(token)


async def test_v2_dispatch_cancels_job_and_terminal_through_the_owning_bridge():
    from types import SimpleNamespace

    from acp import RequestError
    from nooa_atom.acp.server import AtomACPAgent
    from nooa_atom.acp.v2 import V2Dispatcher

    calls: list[tuple[str, str]] = []

    class Bridge:
        async def cancel_background(self, tool_call_id: str, job: Any) -> bool:
            calls.append(("job", tool_call_id))
            return True

        async def cancel_terminal(self, terminal_id: str) -> bool:
            calls.append(("terminal", terminal_id))
            return terminal_id == "live"

    job = SimpleNamespace(status="running")
    jobs = SimpleNamespace(get=lambda name: job if name == "build" else None)
    session = SimpleNamespace(parent_id=None, _agent=SimpleNamespace(jobs=jobs))
    adapter = AtomACPAgent(lambda store: None)  # type: ignore[arg-type]
    adapter.session = lambda session_id: session if session_id == "s" else None  # type: ignore[method-assign]
    adapter._bridges["s"] = Bridge()  # type: ignore[assignment]
    dispatch = V2Dispatcher(adapter)

    assert (
        await dispatch("tool_call/cancel", {"sessionId": "s", "toolCallId": "job:build"}, False)
        == {}
    )
    assert await dispatch("terminal/cancel", {"sessionId": "s", "terminalId": "live"}, False) == {}
    assert calls == [("job", "job:build"), ("terminal", "live")]
    with pytest.raises(RequestError) as error:
        await dispatch("tool_call/cancel", {"sessionId": "s", "toolCallId": "cell"}, False)
    assert error.value.code == -32602
    with pytest.raises(RequestError) as error:
        await dispatch("terminal/cancel", {"sessionId": "s", "terminalId": "gone"}, False)
    assert error.value.code == -32602


async def test_background_cancel_does_not_idle_a_new_running_turn(bridged):
    _session, peer, bridge = bridged
    bridge._set_background(["build"])
    bridge._set_state("running")

    class Job:
        status = "running"

        async def cancel(self) -> None:
            self.status = "cancelled"

    assert await bridge.cancel_background("job:build", Job())
    await bridge.flush()
    states = [u for _, u in peer.updates if u["sessionUpdate"] == "state_update"]
    assert [state["state"] for state in states] == ["running"]
    assert any(u.get("status") == "cancelled" for _, u in peer.updates)
