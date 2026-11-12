"""Built-in tools are installed once, through the skill registry."""

import logging

import pytest
from atom_test_agents import CellLLM
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.tools.todo_list import TodoList

from nooa.storage.json_snapshot import snapshot_from_json, snapshot_to_json

_BUILTINS = {
    "nemo.shell": "shell",
    "nemo.repo": "repo",
    "nemo.todo": "todo",
    "nemo.skill_writing": "skill_writing",
}


def _overwrites(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "overwrites existing agent attr" in r.getMessage()
    ]


def _assert_registry_holds_builtins(agent) -> None:
    registry = agent.skills.registry
    for name, attr in _BUILTINS.items():
        assert name in registry.loaded()
        assert registry._attr_map[name] == attr
    assert agent.repo.session is agent.shell.session


@pytest.mark.parametrize("cls", [AtomAgentBase, AtomAgent])
async def test_fresh_agent_installs_builtins_without_overwrite_warning(cls, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    agent = cls(llm=CellLLM(), cwd=tmp_path)
    try:
        assert _overwrites(caplog) == []
        _assert_registry_holds_builtins(agent)
        if cls is AtomAgent:
            assert isinstance(agent.todo, TodoList)
    finally:
        await agent.aclose()


async def test_real_overwrite_still_warns(tmp_path, caplog):
    agent = AtomAgent(llm=CellLLM(), cwd=tmp_path)
    try:
        agent.unrelated_tool = object()
        caplog.set_level(logging.WARNING)
        caplog.clear()
        agent.skills.register("custom.unrelated_tool", object())
        assert _overwrites(caplog) == [
            "Skill custom.unrelated_tool overwrites existing agent attr 'unrelated_tool'"
        ]
    finally:
        await agent.aclose()


async def test_restored_todo_keeps_items_without_overwrite_warning(tmp_path, caplog):
    agent = AtomAgent(llm=CellLLM(), cwd=tmp_path)
    restored = None
    try:
        item = agent.todo.add("kept", "survives a restore")
        snapshot = snapshot_to_json(agent)
        restored = AtomAgent(llm=CellLLM(), cwd=tmp_path)
        caplog.set_level(logging.WARNING)
        snapshot_from_json(snapshot, restored)  # runs restored.after_restore()
        assert _overwrites(caplog) == []
        assert isinstance(restored.todo, TodoList)
        assert restored.todo.get(item.id).title == "kept"
        assert restored.skills.registry._attr_map["nemo.todo"] == "todo"
    finally:
        await agent.aclose()
        if restored is not None:
            await restored.aclose()
