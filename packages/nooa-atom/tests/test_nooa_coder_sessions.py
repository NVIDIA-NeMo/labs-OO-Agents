# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions saved by nooa-coder, the package before nooa-atom, still load.

``fixtures/nooa_coder_sessions`` holds a session tree that nooa-coder wrote
(SQL dumps of its three session databases): a root ``CodingAgent`` that
added a todo, delegated "Fix parser", spawned "Review", waited for it and
kept both results in ``self.v``. It records the old names three ways: the
agent spec (``nooa_coder.coding.agent:CodingAgent``), the class paths in
the agent's snapshot (``nooa_coder.session.items.TaskResult``) and the
``module:qualname`` of recorded items (``nooa_coder.session.items:ChildResult``).
"""

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.items import ChildRef, ChildResult, TaskResult
from nooa_atom.session.loader import canonical_module, load_agent_class, load_typed
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore, sessions_root

from nooa.unifiedllm import FakeLLMClient

FIXTURES = Path(__file__).parent / "fixtures" / "nooa_coder_sessions"
ROOT = "6756e6fb-0399-4273-bbd7-aa1e052e3b2f"
CHILDREN = {
    "5744946c-b3b2-47bf-9788-c3d38aa29ece": "Fix parser",
    "7c5fdce2-87cd-4219-8e60-538f01a8c4e0": "Review",
}


@pytest.fixture
def workspace(tmp_path):
    """A workspace holding the nooa-coder session tree, as nooa-coder left it."""
    workspace = tmp_path / "ws"
    sessions = sessions_root(workspace)
    sessions.mkdir(parents=True)
    for dump in FIXTURES.glob("*.sql"):
        script = dump.read_text(encoding="utf-8").replace("__WORKSPACE__", str(workspace))
        with closing(sqlite3.connect(sessions / f"{dump.stem}.db")) as db:
            db.executescript(script)
    return workspace


async def test_the_session_tree_is_listed_with_its_children(workspace):
    store = SessionStore(sessions_root(workspace))
    infos = {info.id: info for info in store.list(workspace=workspace, roots_only=False)}
    assert set(infos) == {ROOT, *CHILDREN}
    assert infos[ROOT].parent_id is None
    for child_id, name in CHILDREN.items():
        assert (infos[child_id].parent_id, infos[child_id].name) == (ROOT, name)


async def test_the_root_loads_as_an_atom_agent_with_its_saved_state(workspace, caplog):
    registry = SessionRegistry(
        SessionStore(sessions_root(workspace)), agent_factory=create_session_agent
    )
    try:
        with caplog.at_level(logging.WARNING):
            session = await registry.load(ROOT, llm=FakeLLMClient([]))
        assert "could not restore" not in caplog.text
        assert "Cannot import" not in caplog.text
        agent = session._agent
        assert type(agent) is AtomAgent
        assert "carry the plan over" in agent.todo.status()
        assert agent.v.result == TaskResult(
            solution_description="patched parse()",
            evidence="tests pass",
            how_to_verify="pytest",
            report="REPORT: off-by-one fixed",
        )
        assert isinstance(agent.v.child, ChildRef)
        assert isinstance(agent.v.review, ChildResult)
        assert agent.v.review.done.result.report == "REPORT: off-by-one fixed"
        assert sorted(ref.name for ref in agent.children()) == sorted(CHILDREN.values())
        # The context block was renamed; the restored agent has only the new one.
        assert "workspace_state" in agent.context
        assert "coding_state" not in agent.context
    finally:
        await registry.close_all()


async def test_a_child_loads_as_an_atom_agent(workspace):
    registry = SessionRegistry(
        SessionStore(sessions_root(workspace)), agent_factory=create_session_agent
    )
    try:
        [child_id] = [key for key, name in CHILDREN.items() if name == "Review"]
        child = await registry.load(child_id, llm=FakeLLMClient([]))
        assert type(child._agent) is AtomAgent
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ("nooa_coder.coding.agent:CodingAgent", "AtomAgent"),
        (
            "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent",
            "ExperimentalAtomAgent",
        ),
    ],
)
def test_nooa_coder_agent_specs_load_the_renamed_classes(spec, name):
    assert load_agent_class(spec).__name__ == name


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("nooa_coder", "nooa_atom"),
        ("nooa_coder.session.items", "nooa_atom.session.items"),
        ("nooa_coder.coding.activity", "nooa_atom.agent.activity"),
        ("nooa_coder.coding", "nooa_atom.agent"),
        ("nooa_coder_extra.items", "nooa_coder_extra.items"),
        ("nooa_atom.session.items", "nooa_atom.session.items"),
    ],
)
def test_nooa_coder_modules_map_to_their_nooa_atom_modules(old, new):
    assert canonical_module(old) == new


def test_an_item_recorded_under_a_nooa_coder_type_loads_typed():
    data = {
        "solution_description": "s",
        "evidence": "e",
        "how_to_verify": "v",
        "report": "r",
    }
    assert load_typed("nooa_coder.session.items:TaskResult", data) == TaskResult(**data)
