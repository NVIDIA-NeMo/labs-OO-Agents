# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillRegistry — entry-point discovery, load, activate lifecycle."""

from unittest.mock import MagicMock, patch

import pytest

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
# Tests: Context Block Registration
# ---------------------------------------------------------------------------


class SkillWithContextBlock(Skill):
    """A skill that declares a context block."""

    context_block = ("my_status", "self.my_skill.status()")

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


class TestContextBlockRegistration:
    def test_activate_registers_context_block(self, registry_ctx, agent_ctx):
        skill = SkillWithContextBlock()
        registry_ctx.register("nemo.my_skill", skill)
        registry_ctx.activate(["nemo.my_skill"])

        # The context_manager should now have the dynamic block
        cm = agent_ctx.context_manager
        assert "my_status" in cm

    def test_activate_does_not_register_when_no_context_block(self, registry_ctx, agent_ctx):
        skill = SkillWithoutContextBlock()
        registry_ctx.register("nemo.plain", skill)
        registry_ctx.activate(["nemo.plain"])

        cm = agent_ctx.context_manager
        assert "my_status" not in cm

    def test_deactivate_removes_context_block(self, registry_ctx, agent_ctx):
        skill = SkillWithContextBlock()
        registry_ctx.register("nemo.my_skill", skill)
        registry_ctx.activate(["nemo.my_skill"])

        # Verify it's registered
        cm = agent_ctx.context_manager
        assert "my_status" in cm

        # Deactivate should remove it
        registry_ctx.deactivate(["nemo.my_skill"])
        assert "my_status" not in cm

    def test_deactivate_does_not_remove_protected_block(self, registry_ctx, agent_ctx):
        """If a context block key happens to be protected, deactivate won't remove it."""
        cm = agent_ctx.context_manager
        cm.set_dynamic_protected("my_status", "self.something()")

        skill = SkillWithContextBlock()
        registry_ctx.register("nemo.my_skill", skill)
        registry_ctx.activate(["nemo.my_skill"])
        registry_ctx.deactivate(["nemo.my_skill"])

        # Protected block should still be present
        assert "my_status" in cm

    def test_context_block_with_mocked_skill_is_ignored(self, registry_ctx, agent_ctx):
        """MagicMock skills (from test patches) should not crash context block registration."""
        mock_skill = MagicMock()
        registry_ctx.register("nemo.mocked", mock_skill)
        # Should not raise — MagicMock.context_block is a Mock, not a tuple
        registry_ctx.activate(["nemo.mocked"])

        cm = agent_ctx.context_manager
        # Only the SkillRegistry's own "skills" block should exist (from __init__),
        # NOT a block from the MagicMock skill
        non_protected = [k for k in cm if k not in cm.protected_keys]
        assert "skills" in non_protected  # from SkillRegistry itself
        assert len(non_protected) == 1  # no block from the mock

    def test_skill_registry_registers_own_context_block(self, agent_ctx):
        """SkillRegistry itself has context_block and registers it on activate."""
        # SkillRegistry is registered on the agent as "skills" attr
        with patch("nooa.skill_registry.entry_points", return_value=[]):
            reg = SkillRegistry(agent_ctx)
        agent_ctx.skills = reg
        # Manually trigger context block registration for the registry itself
        reg._attr_map["nemo.skills"] = "skills"
        reg._loaded.add("nemo.skills")
        reg.activate(["nemo.skills"])

        cm = agent_ctx.context_manager
        assert "skills" in cm
