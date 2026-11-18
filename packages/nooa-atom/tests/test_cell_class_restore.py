# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A class defined in a cell: its instance in ``self.v`` survives save and restore (review #2).

Cells run with ``__name__ == "nooa.runtime.cells"``, so such a class does not
claim to live in ``builtins``; restore finds it while its namespace lives,
and a class only a previous process defined costs only its own entries.
"""

import asyncio
import logging

from atom_test_agents import AtomModels, cell
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.runtime import cells

DEFINE_AND_STORE = (
    "from dataclasses import dataclass\n"
    "@dataclass\n"
    "class Point:\n"
    "    x: int\n"
    "self.v.note = 'kept'\n"
    "self.v.point = Point(3)\n"
    "return_result(Done(explanation='stored'))"
)


async def _save(sessions_dir, tmp_path) -> tuple[str, dict]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_AGENT, sessions_dir=sessions_dir)
    models = AtomModels()
    models.scripts[None] = [cell(DEFINE_AND_STORE)]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("store"), 30) == Done(explanation="stored")
        agent = root._agent
        point_class = type(agent.v.point)
        assert point_class.__module__ == "nooa.runtime.cells"
        root._checkpoint()
        await root.wait_for_checkpoint()
        return root.id, agent._handle_namespace.globals
    finally:
        await registry.close_all()


async def _load_vars(sessions_dir, session_id):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        return dict((await registry.load(session_id))._agent.v)
    finally:
        await registry.close_all()


async def test_a_cell_defined_instance_in_vars_restores_in_a_fresh_agent(sessions_dir, tmp_path):
    session_id, cell_globals = await _save(sessions_dir, tmp_path)
    restored = await _load_vars(sessions_dir, session_id)
    assert restored["note"] == "kept"
    point = restored["point"]
    assert type(point) is cell_globals["Point"] and point.x == 3


async def test_a_class_no_live_cell_defines_drops_only_its_entries(
    sessions_dir, tmp_path, monkeypatch
):
    session_id, _ = await _save(sessions_dir, tmp_path)
    # As after a restart: no live namespace defines the class any more.
    monkeypatch.setattr(cells, "_namespaces", [])
    warnings: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    logger = logging.getLogger("nooa.storage.serialization")
    logger.addHandler(handler)
    try:
        restored = await _load_vars(sessions_dir, session_id)
    finally:
        logger.removeHandler(handler)
    assert restored == {"note": "kept"}
    assert any("'point' could not be restored" in message for message in warnings)
