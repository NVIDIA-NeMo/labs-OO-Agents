# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell variables and helpers survive a restart: saved with the snapshot, revived on resume."""

import asyncio

from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.storage.json_snapshot import snapshot_from_json, snapshot_to_json

TIMEOUT = 30

FIRST_TURN = cell(
    "counter = 41\n"
    "notes = {'seen': ['a', 'b'], 'ratio': 0.5}\n"
    "def bump(n=1):\n global counter\n counter += n\n return counter\n"
    "import json\n"
    "class Shape:\n pass\n"
    "handle = object()\n"
    "self.v.fact = 'durable'\n"
    "return_result(Done(explanation='saved'))"
)
SECOND_TURN = cell("return_result(Done(explanation=f'{bump()} {notes[\"seen\"]}'))")


def _notes(agent):
    return [str(note) for note in agent._system_messages_in.snapshot()]


async def test_snapshot_carries_cell_state_and_a_restored_agent_can_use_it(tmp_path):
    agent = AtomAgent(cwd=tmp_path, llm=CellLLM([FIRST_TURN]))
    restored = None
    try:
        assert await agent.handle({"user_messages": ["go"]}) == Done(explanation="saved")
        agent.before_checkpoint()  # what the session does right before it serialises
        snapshot = snapshot_to_json(agent)
        saved = snapshot["attributes"]["_saved_cell_state"]
        assert set(saved["values"]) == {"counter", "notes"}
        assert set(saved["functions"]) == {"bump"}
        skipped = {entry["name"]: entry["reason"] for entry in saved["skipped"]}
        assert skipped["json"].startswith("module")
        assert "class" in skipped["Shape"] and skipped["handle"].startswith("not plain data")
        assert "_handle_namespace" not in snapshot["attributes"]

        restored = AtomAgent(cwd=tmp_path, llm=CellLLM([SECOND_TURN]))
        snapshot_from_json(snapshot, restored)  # runs restored.after_restore()
        assert restored.v.fact == "durable" and restored._handle_namespace is None
        [notice] = [n for n in _notes(restored) if n.startswith("[python-state-restored]")]
        assert "available again: bump, counter, notes." in notice
        assert "json (module: import it again)" in notice and "Shape (" in notice
        assert not any("variables and helpers were lost" in n for n in _notes(restored))

        # The first turn of the restored agent sees the revived names and helper.
        assert await restored.handle({"user_messages": ["again"]}) == Done(
            explanation="42 ['a', 'b']"
        )
        ns = restored._handle_namespace
        assert ns.globals["bump"].__globals__ is ns.globals
        assert "json" not in ns.globals and "Shape" not in ns.globals
        # A second after_restore (another load of the same snapshot) does not repeat the notice.
        restored.after_restore()
        assert sum(n.startswith("[python-state-restored]") for n in _notes(restored)) == 1
    finally:
        await agent.aclose()
        if restored is not None:
            await restored.aclose()


async def test_session_checkpoint_and_load_round_trip_cell_state(tmp_path):
    sessions = SessionStore(tmp_path / "sessions")
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="nooa_atom.agent:AtomAgent",
        llm=CellLLM([FIRST_TURN]),
    )
    registry = SessionRegistry(sessions, agent_factory=create_session_agent)
    try:
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="saved")
        await root.wait_for_checkpoint()  # the settled turn's checkpoint ran before_checkpoint
        session_id = root.id
    finally:
        await registry.close_all()

    registry = SessionRegistry(sessions, agent_factory=create_session_agent)
    try:
        loaded = await registry.load(session_id, llm=CellLLM([SECOND_TURN]))
        agent = loaded._agent
        assert agent._saved_cell_state is not None and agent._pending_cell_state is not None
        assert any(n.startswith("[python-state-restored]") for n in _notes(agent))
        assert await asyncio.wait_for(loaded.prompt("again"), TIMEOUT) == Done(
            explanation="42 ['a', 'b']"
        )
        assert agent._pending_cell_state is None
        assert agent._handle_namespace.globals["counter"] == 42
    finally:
        await registry.close_all()
