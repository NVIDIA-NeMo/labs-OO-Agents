# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""v2 turn lifecycle for turns Atom starts by itself (no session/prompt)."""

from typing import Any

import pytest
from nooa_atom.acp.event_bridge import ACPEventBridge
from nooa_atom.session.items import SessionInfo, TurnEndedUpdate, TurnStartedUpdate


class _Session:
    def __init__(self, session_id: str = "s") -> None:
        self.id = session_id
        self.info = SessionInfo(id=session_id)
        self.listeners: list[Any] = []

    def subscribe(self, listener: Any) -> Any:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def emit(self, update: Any) -> None:
        for listener in list(self.listeners):
            listener(update)

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


def _states(peer: _Peer) -> list[tuple[str, str | None]]:
    return [
        (u["state"], u.get("stopReason"))
        for _, u in peer.updates
        if u["sessionUpdate"] == "state_update"
    ]


async def test_a_turn_atom_starts_itself_reports_running_then_idle(bridged):
    """A queued item starts a turn with no prompt: the client still sees running, then idle."""
    session, peer, bridge = bridged
    session.emit(TurnStartedUpdate(session_id="s", item_ids=["queued-1"]))
    await bridge.flush()
    assert _states(peer) == [("running", None)]
    session.emit(TurnEndedUpdate(session_id="s", outcome_kind="done"))
    await bridge.flush()
    assert _states(peer) == [("running", None), ("idle", "end_turn")]


async def test_back_to_back_self_started_turns_each_report_their_end(bridged):
    session, peer, bridge = bridged
    for item in ("a", "b"):
        session.emit(TurnStartedUpdate(session_id="s", item_ids=[item]))
        session.emit(TurnEndedUpdate(session_id="s", outcome_kind="done"))
    await bridge.flush()
    assert _states(peer) == [
        ("running", None),
        ("idle", "end_turn"),
        ("running", None),
        ("idle", "end_turn"),
    ]
