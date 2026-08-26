# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statusbar built-ins and loaded-skill capabilities."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nooa_cli.tui.statusbar import StatusbarContext, StatusbarRegistry

from nooa.skill import Skill
from nooa.skill_registry import SkillRegistry


def context(agent=None):
    return StatusbarContext(
        model="provider/claude-example",
        working_directory=Path("/work/repo"),
        context_usage="ctx 20%",
        session_id="12345678-abcd",
        session_title="migration",
        agent=agent,
    )


def test_statusbar_renders_configured_items_in_order():
    registry = StatusbarRegistry()
    assert registry.render(["model", "cwd", "context", "session"], context()) == (
        "example · repo · ctx 20% · migration [12345678]"
    )


def test_statusbar_accepts_registered_extension_and_stringifies_values():
    registry = StatusbarRegistry()
    registry.register("count", lambda _: 3)
    assert registry.render(["count"], context()) == "3"


def test_statusbar_rejects_non_callable_provider():
    registry = StatusbarRegistry()
    with pytest.raises(TypeError, match="callable"):
        registry.register("broken", object())  # type: ignore[arg-type]


def test_loaded_skill_contributes_provider_without_activation():
    skill = SimpleNamespace(statusbar_items={"mesh": lambda ctx: ctx.agent.mesh.name})

    class Skills:
        def loaded(self):
            return ["demo.mesh"]

        def __getitem__(self, name):
            return skill

    agent = SimpleNamespace(mesh=SimpleNamespace(name="alpha"), skills=Skills())
    registry = StatusbarRegistry(agent)
    assert registry.render(["mesh"], context(agent)) == "alpha"


def test_skill_capability_can_resolve_safe_self_path():
    skill = SimpleNamespace(statusbar_items={"mesh": "self.mesh.identity.name"})

    class Skills:
        def loaded(self):
            return ["demo.mesh"]

        def __getitem__(self, name):
            return skill

    agent = SimpleNamespace(
        mesh=SimpleNamespace(identity=SimpleNamespace(name="alpha")), skills=Skills()
    )
    registry = StatusbarRegistry(agent)
    assert registry.render(["mesh", "self.mesh.identity.name"], context(agent)) == "alpha · alpha"


@pytest.mark.parametrize(
    "path", ["self._secret", "self.mesh._secret", "self.mesh.name()", "self.mesh[0]", "mesh.name"]
)
def test_unsafe_self_paths_are_unavailable(path):
    registry = StatusbarRegistry()
    assert not registry.accepts(path)
    assert registry.render([path], context(SimpleNamespace())) == ""


def test_provider_failure_is_isolated():
    def fail(_):
        raise RuntimeError("boom")

    registry = StatusbarRegistry()
    registry.register("bad", fail)
    registry.register("good", lambda _: "ok")
    assert registry.render(["bad", "good"], context()) == "ok"


def test_refresh_replaces_reloaded_skill_provider():
    class Skills:
        skill = SimpleNamespace(statusbar_items={"value": lambda _: "old"})

        def loaded(self):
            return ["demo.value"]

        def __getitem__(self, name):
            return self.skill

    skills = Skills()
    agent = SimpleNamespace(skills=skills)
    registry = StatusbarRegistry(agent)
    assert registry.render(["value"], context(agent)) == "old"
    skills.skill = SimpleNamespace(statusbar_items={"value": lambda _: "new"})
    registry.refresh_skills()
    assert registry.render(["value"], context(agent)) == "new"


def test_skill_name_collision_is_deterministic():
    first = SimpleNamespace(statusbar_items={"shared": lambda _: "first"})
    second = SimpleNamespace(statusbar_items={"shared": lambda _: "second"})

    class Skills:
        def loaded(self):
            return ["a.first", "z.second"]

        def __getitem__(self, name):
            return {"a.first": first, "z.second": second}[name]

    agent = SimpleNamespace(skills=Skills())
    registry = StatusbarRegistry(agent)
    assert registry.render(["shared"], context(agent)) == "second"


def test_statusbar_command_preserves_case_in_self_paths():
    from nooa_cli.tui.commands import StatusbarCommand

    agent = SimpleNamespace(mesh=SimpleNamespace(displayName="Alpha"))
    statusbar = StatusbarRegistry(agent)
    config = SimpleNamespace(statusbar_items=[])
    command = StatusbarCommand(
        SimpleNamespace(), config, agent, registry=SimpleNamespace(statusbar_registry=statusbar)
    )

    result = __import__("asyncio").run(command.execute(["set", "MODEL", "self.mesh.displayName"]))

    assert result.success
    assert config.statusbar_items == ["model", "self.mesh.displayName"]
    assert statusbar.render(config.statusbar_items, context(agent)) == "example · Alpha"


def test_deprecated_toolbar_imports_alias_statusbar_types():
    from nooa_cli.tui.toolbar import ToolbarContext, ToolbarProvider, ToolbarRegistry

    assert ToolbarContext is StatusbarContext
    assert ToolbarRegistry is StatusbarRegistry
    assert ToolbarProvider is not None


class _StatusSkill(Skill):
    statusbar_items = {"lifecycle": lambda _: "registered"}


class _StatusbarHost:
    def __init__(self, agent):
        self.statusbar_registry = StatusbarRegistry(agent)
        self.notifications = 0

    def refresh_skill_commands(self):
        self.notifications += 1
        self.statusbar_registry.refresh_skills()


def test_skill_registry_register_and_load_notify_statusbar_without_activation():
    agent = SimpleNamespace()
    entry_point = MagicMock()
    entry_point.name = "demo.loaded"
    entry_point.load.return_value = _StatusSkill
    with patch("nooa.skill_registry.entry_points", return_value=[entry_point]):
        skills = SkillRegistry(agent)
    agent.skills = skills
    host = _StatusbarHost(agent)
    agent._command_registry = host

    skills.register("demo.registered", _StatusSkill())
    assert host.statusbar_registry.render(["lifecycle"], context(agent)) == "registered"
    assert skills.activated() == []

    before_load = host.notifications
    skills.load(["demo.loaded"])
    assert host.notifications == before_load + 1
    assert "demo.loaded" in skills.loaded()
    assert "demo.loaded" not in skills.activated()
    assert host.statusbar_registry.render(["lifecycle"], context(agent)) == "registered"


@pytest.mark.asyncio
async def test_skill_registry_reload_and_aclose_notify_statusbar(tmp_path):
    skill_file = tmp_path / "live.py"
    skill_file.write_text(
        "from nooa.skill import Skill\n"
        "class Live(Skill):\n"
        "    statusbar_items = {'live': lambda context: 'old'}\n"
    )
    agent = SimpleNamespace()
    with patch("nooa.skill_registry.entry_points", return_value=[]):
        skills = SkillRegistry(agent)
    agent.skills = skills
    host = _StatusbarHost(agent)
    agent._command_registry = host

    skills.discover_skills_dirs([tmp_path])
    assert host.statusbar_registry.render(["live"], context(agent)) == "old"
    assert skills.activated() == []

    skill_file.write_text(
        "from nooa.skill import Skill\n"
        "class Live(Skill):\n"
        "    statusbar_items = {'live': lambda context: 'new-value'}\n"
    )
    before_reload = host.notifications
    result = await skills.reload("ext.live")
    assert result.startswith("Reloaded ext.live")
    assert host.notifications == before_reload + 1
    assert host.statusbar_registry.render(["live"], context(agent)) == "new-value"

    before_close = host.notifications
    await skills.aclose()
    assert host.notifications == before_close + 1
    assert host.statusbar_registry.render(["live"], context(agent)) == ""
