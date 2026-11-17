# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillRegistry — entry-point discovery, load, activate lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nooa.context_blocks import Context
from nooa.skill import Skill
from nooa.skill_registry import SkillRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeSkill(Skill):
    """A test skill with no constructor args."""

    def do_thing(self) -> str:
        return "done"


class _FakeAgent:
    pass


@pytest.fixture
def agent():
    return _FakeAgent()


@pytest.fixture
def registry(agent):
    """SkillRegistry with no real entry points discovered."""
    with patch("nooa.skill_registry.entry_points", return_value=[]):
        return SkillRegistry(agent)


# ---------------------------------------------------------------------------
# Tests: Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovered_empty_when_no_entry_points(self, registry):
        assert registry.discovered() == []

    def test_discovered_finds_entry_points(self, agent):
        ep = MagicMock()
        ep.name = "nemo.shell"
        with patch("nooa.skill_registry.entry_points", return_value=[ep]):
            reg = SkillRegistry(agent)
        assert "nemo.shell" in reg.discovered()

    def test_discovered_handles_exception(self, agent):
        with patch(
            "nooa.skill_registry.entry_points",
            side_effect=Exception("broken"),
        ):
            reg = SkillRegistry(agent)
        assert reg.discovered() == []

    def test_entry_returns_record_for_discovered_skill(self, agent):
        ep = MagicMock()
        ep.name = "nemo.shell"
        with patch("nooa.skill_registry.entry_points", return_value=[ep]):
            reg = SkillRegistry(agent)
        entry = reg.entry("nemo.shell")
        assert entry is not None
        assert entry.name == "nemo.shell"
        assert entry.category == "nemo"

    def test_entry_returns_none_for_unknown_skill(self, registry):
        assert registry.entry("does.not.exist") is None


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.parametrize("second", ["support.search", "research.search-tool"])
    def test_distinct_names_cannot_replace_active_skill(self, registry, agent, second):
        first_name = "research.search" if second == "support.search" else "research.search_tool"
        first, rejected = FakeSkill(), FakeSkill()
        registry.register(first_name, first)
        registry.activate([first_name])
        with pytest.raises(ValueError, match="already provided by 'research"):
            registry.register(second, rejected)
        assert registry.loaded() == [first_name]
        assert registry._managed_skills() == (first,)
        assert registry.active_skills() == (first,)
        assert getattr(agent, first_name.split(".")[-1]) is first
        assert rejected._agent is None

    @pytest.mark.parametrize("second", ["support.search", "research.search-tool"])
    def test_distinct_name_replaces_inactive_skill(self, registry, agent, second, caplog):
        first_name = "research.search" if second == "support.search" else "research.search_tool"
        first, replacement = FakeSkill(), FakeSkill()
        registry.register(first_name, first)
        registry.register(second, replacement)
        assert registry.loaded() == [second]
        assert registry._managed_skills() == (replacement,)
        assert getattr(agent, first_name.split(".")[-1]) is replacement
        assert first._agent is None
        assert f"replaces inactive skill {first_name}" in caplog.text
        registry.activate([second])
        assert registry.active_skills() == (replacement,)

    def test_same_full_name_replacement_preserves_activation(self, registry):
        registry.register("research.search", FakeSkill())
        registry.activate(["research.search"])
        replacement = FakeSkill()
        registry.register("research.search", replacement)
        assert registry.active_skills() == (replacement,)
        assert registry._managed_skills() == (replacement,)

    def test_register_adds_to_loaded(self, registry):
        skill = FakeSkill()
        registry.register("nemo.test", skill)
        assert "nemo.test" in registry.loaded()

    def test_register_adds_to_discovered(self, registry):
        skill = FakeSkill()
        registry.register("custom.tool", skill)
        assert "custom.tool" in registry.discovered()


# ---------------------------------------------------------------------------
# Tests: Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_load_reports_a_conflicting_leaf(self, agent, caplog):
        entries = []
        for name in ("research.search", "support.search"):
            entry = MagicMock(name=name)
            entry.name = name
            entry.load.return_value = FakeSkill
            entries.append(entry)
        with patch("nooa.skill_registry.entry_points", return_value=entries):
            registry = SkillRegistry(agent)
        registry.load(["*"])
        # A discovered duplicate leaf is a name conflict, not a replacement.
        assert registry.loaded() == ["research.search"]
        assert registry.name_conflicts() == [
            "skill 'search' not loaded: the agent already has an attribute named search; "
            "rename the skill"
        ]
        registry.activate(["*"])
        assert registry.active_skills() == (agent.search,)

    def test_load_skips_leaf_of_active_skill(self, agent, caplog):
        entry = MagicMock(name="support.search")
        entry.name = "support.search"
        entry.load.return_value = FakeSkill
        with patch("nooa.skill_registry.entry_points", return_value=[entry]):
            registry = SkillRegistry(agent)
        first = FakeSkill()
        registry.register("research.search", first)
        registry.activate(["research.search"])
        registry.load(["support.search"])
        assert registry.loaded() == ["research.search"]
        assert agent.search is first
        assert "already provided by 'research.search'" in caplog.text

    def test_load_from_entry_point(self, agent):
        ep = MagicMock()
        ep.name = "nemo.fake"
        ep.load.return_value = FakeSkill

        with patch("nooa.skill_registry.entry_points", return_value=[ep]):
            reg = SkillRegistry(agent)

        reg.load(["nemo.fake"])
        assert "nemo.fake" in reg.loaded()
        assert hasattr(agent, "fake")
        assert isinstance(agent.fake, FakeSkill)

    def test_load_glob_pattern(self, agent):
        ep1 = MagicMock()
        ep1.name = "nemo.a"
        ep1.load.return_value = FakeSkill
        ep2 = MagicMock()
        ep2.name = "nemo.b"
        ep2.load.return_value = FakeSkill

        with patch("nooa.skill_registry.entry_points", return_value=[ep1, ep2]):
            reg = SkillRegistry(agent)

        reg.load(["nemo.*"])
        assert "nemo.a" in reg.loaded()
        assert "nemo.b" in reg.loaded()

    def test_load_skip_non_skill(self, agent):
        ep = MagicMock()
        ep.name = "nemo.bad"
        ep.load.return_value = "not a skill"

        with patch("nooa.skill_registry.entry_points", return_value=[ep]):
            reg = SkillRegistry(agent)

        reg.load(["nemo.bad"])
        assert "nemo.bad" not in reg.loaded()


# ---------------------------------------------------------------------------
# Tests: Activation
# ---------------------------------------------------------------------------


class TestActivation:
    def test_activate_marks_as_activated(self, registry):
        skill = FakeSkill()
        registry.register("nemo.test", skill)
        registry.activate(["nemo.test"])
        assert "nemo.test" in registry.activated()

    def test_deactivate_removes_from_activated(self, registry):
        skill = FakeSkill()
        registry.register("nemo.test", skill)
        registry.activate(["nemo.test"])
        registry.deactivate(["nemo.test"])
        assert "nemo.test" not in registry.activated()

    def test_activate_glob(self, registry):
        registry.register("nemo.a", FakeSkill())
        registry.register("nemo.b", FakeSkill())
        registry.register("super.c", FakeSkill())
        registry.activate(["nemo.*"])
        assert "nemo.a" in registry.activated()
        assert "nemo.b" in registry.activated()
        assert "super.c" not in registry.activated()

    def test_activate_auto_loads(self, agent):
        ep = MagicMock()
        ep.name = "nemo.auto"
        ep.load.return_value = FakeSkill

        with patch("nooa.skill_registry.entry_points", return_value=[ep]):
            reg = SkillRegistry(agent)

        reg.activate(["nemo.auto"])
        assert "nemo.auto" in reg.loaded()
        assert "nemo.auto" in reg.activated()


# ---------------------------------------------------------------------------
# Tests: Context Block Selection
# ---------------------------------------------------------------------------


class SkillWithContextBlock(Skill):
    """A skill that declares a context block."""

    context_block = ("my_status", Context(expr="self.status()"))

    def status(self) -> str:
        return "ok"


class SkillWithoutContextBlock(Skill):
    """A skill that does not declare a context block."""

    def do_something(self) -> str:
        return "done"


class _FakeAgentWithContext:
    """Agent mock with a context_manager."""

    def __init__(self):
        from nooa.runtime.context_manager import ContextManager

        self.context_manager = ContextManager()


@pytest.fixture
def agent_ctx():
    return _FakeAgentWithContext()


@pytest.fixture
def registry_ctx(agent_ctx):
    """SkillRegistry with context_manager available on the agent."""
    with patch("nooa.skill_registry.entry_points", return_value=[]):
        return SkillRegistry(agent_ctx)


class TestContextBlockSelection:
    def test_activate_selects_skill_without_mutating_context_manager(self, registry_ctx, agent_ctx):
        skill = SkillWithContextBlock()
        registry_ctx.register("nemo.my_skill", skill)
        registry_ctx.activate(["nemo.my_skill"])

        cm = agent_ctx.context_manager
        assert "my_status" not in cm
        assert registry_ctx.active_skills() == (skill,)

    def test_activate_does_not_register_when_no_context_block(self, registry_ctx, agent_ctx):
        skill = SkillWithoutContextBlock()
        registry_ctx.register("nemo.plain", skill)
        registry_ctx.activate(["nemo.plain"])

        cm = agent_ctx.context_manager
        assert "my_status" not in cm

    def test_deactivate_deselects_skill_without_manager_mutation(self, registry_ctx, agent_ctx):
        skill = SkillWithContextBlock()
        registry_ctx.register("nemo.my_skill", skill)
        registry_ctx.activate(["nemo.my_skill"])

        cm = agent_ctx.context_manager
        assert "my_status" not in cm
        assert registry_ctx.active_skills() == (skill,)

        registry_ctx.deactivate(["nemo.my_skill"])
        assert "my_status" not in cm
        assert registry_ctx.active_skills() == ()

    def test_context_block_with_mocked_skill_does_not_mutate_manager(self, registry_ctx, agent_ctx):
        mock_skill = MagicMock()
        registry_ctx.register("nemo.mocked", mock_skill)
        registry_ctx.activate(["nemo.mocked"])

        cm = agent_ctx.context_manager
        assert list(cm) == []

    @pytest.mark.asyncio
    async def test_aclose_does_not_mutate_context_manager(self, registry_ctx, agent_ctx):
        assert "skills" not in agent_ctx.context_manager

        await registry_ctx.aclose()

        assert "skills" not in agent_ctx.context_manager

    def test_skill_registry_never_registers_own_context_block(self, agent_ctx):
        # SkillRegistry is registered on the agent as "skills" attr
        with patch("nooa.skill_registry.entry_points", return_value=[]):
            reg = SkillRegistry(agent_ctx)
        agent_ctx.skills = reg
        reg._attr_map["nemo.skills"] = "skills"
        reg._loaded.add("nemo.skills")
        reg.activate(["nemo.skills"])

        cm = agent_ctx.context_manager
        assert "skills" not in cm


def test_a_skill_cannot_take_over_another_skills_agent_attribute():
    """Two registry names must not resolve to one agent attribute.

    The collision check was skipped once an attr appeared in _attr_map, so the
    second registrant silently replaced the first. That is how a repo-supplied
    `cmd.shell` or a client-supplied `mcp.shell` removed the agent's real shell.
    """

    class _Agent:
        # Declares the attrs carrying its own tools, as CodingAgent does.
        __protected_skill_attrs__ = frozenset({"shell"})

    class _Tool:
        pass

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.register("nemo.shell", _Tool())
    original = agent.shell

    with pytest.raises(ValueError, match="shell"):
        registry.register("mcp.shell", _Tool())

    assert agent.shell is original
    # Re-registering the same name stays allowed.
    registry.register("nemo.shell", _Tool())


def test_load_also_refuses_to_take_over_a_protected_attribute():
    """register() was guarded; load() assigned directly and was not.

    activate() calls load() for anything not yet loaded, so a *discovered*
    mcp.shell could still replace the agent's real shell after nemo.shell
    owned it — the guard only covered the explicit registration path.
    """

    class _Agent:
        __protected_skill_attrs__ = frozenset({"shell"})

    class _Tool(Skill):
        pass

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.register("nemo.shell", _Tool())
    original = agent.shell

    # load() resolves via entry_point.load(), so the stub must provide it.
    entry = SimpleNamespace(
        name="mcp.shell",
        entry_point=SimpleNamespace(load=lambda: _Tool),
        category="entry_point",
    )
    registry._discovered["mcp.shell"] = entry
    registry.load(["mcp.shell"])

    assert agent.shell is original, "a discovered skill took over a protected attr"
    assert "mcp.shell" not in registry.loaded()

    # Control: the same machinery loads a non-colliding name, so the assertions
    # above are about the guard rather than a broken stub.
    registry._discovered["mcp.notes"] = SimpleNamespace(
        name="mcp.notes",
        entry_point=SimpleNamespace(load=lambda: _Tool),
        category="entry_point",
    )
    registry.load(["mcp.notes"])
    assert "mcp.notes" in registry.loaded()


def test_mcp_server_replaces_preloaded_inactive_skill_with_same_leaf():
    """A host's preloaded, never-enabled skill must not block a configured MCP server."""

    class _Agent:
        __protected_skill_attrs__ = frozenset({"shell"})

    class _Tool(Skill):
        pass

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.register("nemo.notes", _Tool())  # loaded inactive, as CodingAgent does
    mcp_tool = _Tool()

    registry.register("mcp.notes", mcp_tool)
    registry.activate(["mcp.notes"])

    assert agent.notes is mcp_tool
    assert registry.loaded() == ["mcp.notes"]
    assert registry.active_skills() == (mcp_tool,)


def test_protected_attribute_is_not_replaced_even_when_owner_is_inactive():
    class _Agent:
        __protected_skill_attrs__ = frozenset({"shell"})

    class _Tool(Skill):
        pass

    agent = _Agent()
    registry = SkillRegistry(agent)
    first = _Tool()
    registry.register("nemo.shell", first)
    with pytest.raises(ValueError, match="already provided by 'nemo.shell'"):
        registry.register("mcp.shell", _Tool())
    assert agent.shell is first


def test_a_directly_assigned_undeclared_attribute_is_not_protected():
    """Direct assignment alone must not confer protection."""

    class _Agent:
        __protected_skill_attrs__ = frozenset({"shell"})

        def __init__(self):
            self.notes = object()

    class _Tool(Skill):
        pass

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.register("mcp.notes", _Tool())
    assert isinstance(agent.notes, _Tool)


# ---------------------------------------------------------------------------
# Tests: name conflicts (every load path)
# ---------------------------------------------------------------------------


def _conflict_line(leaf: str) -> str:
    return (
        f"skill {leaf!r} not loaded: the agent already has an attribute named {leaf}; "
        "rename the skill"
    )


def test_an_entry_point_skill_whose_attribute_is_taken_is_not_loaded():
    class _Agent:
        def __init__(self):
            self.fake = "mine"

    agent = _Agent()
    ep = MagicMock()
    ep.name = "nemo.fake"
    ep.load.return_value = FakeSkill
    with patch("nooa.skill_registry.entry_points", return_value=[ep]):
        reg = SkillRegistry(agent)
    reg.load(["nemo.fake"])
    assert "nemo.fake" not in reg.loaded()
    assert agent.fake == "mine"
    assert reg.name_conflicts() == [_conflict_line("fake")]


def _write_package(root, name: str) -> None:
    pkg = root / name
    (pkg / name).mkdir(parents=True)
    (pkg / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\n[project.entry-points."nooa.skills"]\n'
        f'"local.{name}" = "{name}"\n'
    )
    (pkg / name / "__init__.py").write_text(
        "from nooa.skill import Skill\n\n\nclass Tool(Skill):\n"
        "    def hi(self) -> str:\n        return 'hi'\n"
    )


def test_a_libs_package_skill_whose_attribute_is_taken_is_not_loaded(tmp_path, agent):
    _write_package(tmp_path, "pkgconflict")
    agent.pkgconflict = "mine"
    with patch("nooa.skill_registry.entry_points", return_value=[]):
        reg = SkillRegistry(agent)
    reg.discover_libs(tmp_path)
    assert "local.pkgconflict" not in reg.loaded()
    assert agent.pkgconflict == "mine"
    assert reg.name_conflicts() == [_conflict_line("pkgconflict")]


def test_a_text_skill_registers_from_any_libs_dir(tmp_path, registry, agent):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "SKILL.md").write_text("---\nname: notes\ndescription: d\n---\nbody\n")
    registry.discover_libs(tmp_path)
    assert "local.notes" in registry.loaded()
    assert registry.name_conflicts() == []


async def test_reloading_a_skill_is_not_a_name_conflict(tmp_path, registry, agent):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "SKILL.md").write_text("---\nname: notes\ndescription: d\n---\nbody\n")
    registry.discover_libs(tmp_path)
    result = await registry.reload("local.notes")
    assert result.startswith("Reloaded "), result
    registry.discover_libs(tmp_path)  # rediscovering its own directory is no conflict either
    assert "local.notes" in registry.loaded()
    assert registry.name_conflicts() == []
